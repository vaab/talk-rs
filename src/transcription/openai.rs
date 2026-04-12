//! OpenAI API batch transcription backend.
//!
//! This module provides a [`BatchTranscriber`] implementation that uses the
//! OpenAI API to transcribe audio files (Whisper, GPT-4o-transcribe, etc.).

use crate::config::OpenAIConfig;
use crate::error::TalkError;
use crate::transcription::{
    OpenAIProviderMetadata, ProviderSpecificMetadata, TokenUsage, TranscriptionMetadata,
    TranscriptionResult,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::fs::File;
use tokio_stream::wrappers::ReceiverStream;

use super::http::{build_client, parse_u64_field, proportional_timeout};
use super::BatchTranscriber;
use crate::telemetry::{NoOpSink, TelemetrySink, TranscriptionEvent};

/// Default API base URL for the OpenAI API.
pub(crate) const API_BASE: &str = "https://api.openai.com";

/// Response from OpenAI API transcription endpoint.
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    /// The transcribed text.
    text: String,
    /// Model identifier if returned by API.
    #[serde(default)]
    model: Option<String>,
    /// Detected language code.
    #[serde(default)]
    language: Option<String>,
    /// Input audio duration in seconds.
    #[serde(default)]
    duration: Option<f64>,
    /// Optional list of transcript segments.
    #[serde(default)]
    segments: Option<Vec<serde_json::Value>>,
    /// Optional list of per-word timestamps.
    #[serde(default)]
    words: Option<Vec<serde_json::Value>>,
    /// Usage payload (shape varies by model endpoint).
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

fn parse_openai_token_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    let input_tokens = parse_u64_field(usage, "input_tokens");
    let output_tokens = parse_u64_field(usage, "output_tokens");
    let total_tokens = parse_u64_field(usage, "total_tokens");

    if input_tokens.is_none() && output_tokens.is_none() && total_tokens.is_none() {
        None
    } else {
        Some(TokenUsage {
            input_tokens,
            output_tokens,
            total_tokens,
        })
    }
}

fn parse_openai_audio_seconds(usage: &serde_json::Value) -> Option<f64> {
    usage.get("seconds").and_then(|v| v.as_f64())
}

fn extract_rate_limit_headers(headers: &reqwest::header::HeaderMap) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (name, value) in headers {
        let key = name.as_str();
        if key.starts_with("x-ratelimit-") {
            if let Ok(v) = value.to_str() {
                out.insert(key.to_string(), v.to_string());
            }
        }
    }
    out
}

// ── Error detection ─────────────────────────────────────────────────

/// Check if an error is a model-not-found error from the OpenAI API.
///
/// OpenAI returns `"code":"model_not_found"` or `"does not exist"`
/// messages.
pub(crate) fn is_model_error(error: &TalkError) -> bool {
    let msg = error.to_string().to_lowercase();
    msg.contains("model_not_found")
        || (msg.contains("model") && (msg.contains("does not exist") || msg.contains("not found")))
}

/// Filter predicate for OpenAI transcription-relevant models.
pub(crate) fn is_transcription_model(model_id: &str) -> bool {
    model_id.contains("whisper") || model_id.contains("transcri")
}

/// Enrich a model error with available OpenAI transcription models.
pub(crate) async fn enrich_model_error(
    error: TalkError,
    api_key: &str,
    model: &str,
    api_base: &str,
) -> TalkError {
    super::http::enrich_model_error(
        error,
        api_key,
        model,
        api_base,
        is_model_error,
        is_transcription_model,
    )
    .await
}

// ── Model validation (delegates to shared http helper) ──────────────

/// Validate that `model` is available in the OpenAI account reachable
/// at `api_base`.
///
/// Also used by [`super::openai_realtime::OpenAIRealtimeTranscriber`].
pub(crate) async fn validate_openai_model(
    api_key: &str,
    model: &str,
    api_base: &str,
) -> Result<(), TalkError> {
    super::http::validate_model("OpenAI", api_key, model, api_base, is_transcription_model).await
}

/// Batch transcriber implementation using the OpenAI API.
///
/// This implementation sends audio files to the OpenAI API for transcription
/// and parses the JSON response to extract the transcribed text.
pub struct OpenAIBatchTranscriber {
    /// HTTP client for making requests to the OpenAI API.
    client: Client,
    /// OpenAI API configuration (contains API key and model).
    config: OpenAIConfig,
    /// API endpoint URL (can be overridden for testing).
    endpoint: String,
    /// Telemetry event sink for HTTP lifecycle reporting.
    sink: Arc<dyn TelemetrySink>,
}

impl OpenAIBatchTranscriber {
    /// Create a new OpenAI transcriber with the given configuration.
    ///
    /// The transcription endpoint is derived from `config.url` (if set)
    /// by appending `/v1/audio/transcriptions`.  When `config.url` is
    /// `None`, the default OpenAI API base URL is used.
    ///
    /// # Arguments
    ///
    /// * `config` - OpenAI API configuration containing the API key
    pub fn new(config: OpenAIConfig) -> Result<Self, TalkError> {
        let base = config.url.as_deref().unwrap_or(API_BASE);
        let endpoint = format!("{}/v1/audio/transcriptions", base.trim_end_matches('/'));
        Ok(Self {
            client: build_client()?,
            config,
            endpoint,
            sink: Arc::new(NoOpSink),
        })
    }

    /// Create a new OpenAI transcriber with a custom endpoint (for testing).
    ///
    /// # Arguments
    ///
    /// * `config` - OpenAI API configuration containing the API key
    /// * `endpoint` - Custom API endpoint URL
    #[cfg(test)]
    pub fn with_endpoint(config: OpenAIConfig, endpoint: String) -> Result<Self, TalkError> {
        Ok(Self {
            client: build_client()?,
            config,
            endpoint,
            sink: Arc::new(NoOpSink),
        })
    }
}

#[async_trait]
impl BatchTranscriber for OpenAIBatchTranscriber {
    fn set_sink(&mut self, sink: Arc<dyn TelemetrySink>) {
        self.sink = sink;
    }

    async fn validate(&self) -> Result<(), TalkError> {
        // Derive the API base URL from the transcription endpoint.
        // Production: "https://api.openai.com/v1/audio/transcriptions" → "https://api.openai.com"
        // Test:       "http://127.0.0.1:PORT/v1/audio/transcriptions" → "http://127.0.0.1:PORT"
        let api_base = self
            .endpoint
            .find("/v1/")
            .map(|pos| &self.endpoint[..pos])
            .unwrap_or(&self.endpoint);
        validate_openai_model(&self.config.api_key, &self.config.model, api_base).await
    }

    async fn transcribe_file(&self, audio_path: &Path) -> Result<TranscriptionResult, TalkError> {
        // Verify file exists
        if !audio_path.exists() {
            return Err(TalkError::Transcription(format!(
                "Audio file not found: {}",
                audio_path.display()
            )));
        }

        // Read the audio file into memory and wrap in ProgressBody so
        // the telemetry sink receives upload-progress events during
        // retry attempts (same treatment as transcribe_stream).
        use tokio::io::AsyncReadExt;
        let mut file = File::open(audio_path).await.map_err(|err| {
            TalkError::Transcription(format!("Failed to open audio file: {}", err))
        })?;
        let mut file_bytes = Vec::new();
        file.read_to_end(&mut file_bytes).await.map_err(|err| {
            TalkError::Transcription(format!("Failed to read audio file: {}", err))
        })?;
        let file_len = file_bytes.len() as u64;

        let progress_body = super::http::ProgressBody::new(file_bytes, self.sink.clone());
        let body_len = progress_body.len();

        // Get file name for multipart form
        let file_name = audio_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("audio.ogg")
            .to_string();

        // Create multipart form with audio file, model, and response format
        let form = reqwest::multipart::Form::new()
            .text("model", self.config.model.clone())
            .text("response_format", "json")
            .part(
                "file",
                reqwest::multipart::Part::stream_with_length(
                    reqwest::Body::wrap_stream(progress_body),
                    body_len,
                )
                .file_name(file_name),
            );

        // Compute a payload-proportional wall-clock timeout for this
        // request.  See `proportional_timeout` for the formula and
        // rationale.  This bounds the tail when the server accepts
        // the connection and then hangs at the application layer.
        let request_timeout = proportional_timeout(file_len);
        log::warn!(
            "openai batch file: request timeout = {}s (audio = {} KB)",
            request_timeout.as_secs(),
            file_len / 1024
        );

        self.sink.emit(TranscriptionEvent::RequestStarted {
            endpoint: self.endpoint.clone(),
            t: Instant::now(),
        });

        let started = Instant::now();
        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .multipart(form)
            .timeout(request_timeout)
            .send()
            .await
            .map_err(|err| {
                self.sink.emit(TranscriptionEvent::RequestCompleted {
                    success: false,
                    t: Instant::now(),
                });
                TalkError::Transcription(format!("Failed to send request to OpenAI API: {:#}", err))
            })?;

        let request_latency_ms = started.elapsed().as_millis() as u64;
        self.sink.emit(TranscriptionEvent::ResponseHeaders {
            status: response.status().as_u16(),
            t: Instant::now(),
        });
        let headers = response.headers().clone();

        // Check response status
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            self.sink.emit(TranscriptionEvent::RequestCompleted {
                success: false,
                t: Instant::now(),
            });
            return Err(TalkError::Transcription(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        // Stream the response body chunk-by-chunk for download telemetry.
        use futures::StreamExt;
        let mut body_bytes: Vec<u8> = Vec::new();
        let mut body_stream = response.bytes_stream();
        while let Some(chunk_result) = body_stream.next().await {
            let chunk = chunk_result.map_err(|err| {
                self.sink.emit(TranscriptionEvent::RequestCompleted {
                    success: false,
                    t: Instant::now(),
                });
                TalkError::Transcription(format!(
                    "Failed to read OpenAI API response body: {}",
                    err
                ))
            })?;
            body_bytes.extend_from_slice(&chunk);
            self.sink.emit(TranscriptionEvent::DownloadProgress {
                bytes_received: body_bytes.len() as u64,
                total: None,
                t: Instant::now(),
            });
        }
        self.sink.emit(TranscriptionEvent::ResponseComplete {
            total: body_bytes.len() as u64,
            t: Instant::now(),
        });

        let openai_response: OpenAIResponse =
            serde_json::from_slice(&body_bytes).map_err(|err| {
                self.sink.emit(TranscriptionEvent::RequestCompleted {
                    success: false,
                    t: Instant::now(),
                });
                TalkError::Transcription(format!("Failed to parse OpenAI API response: {}", err))
            })?;

        self.sink.emit(TranscriptionEvent::RequestCompleted {
            success: true,
            t: Instant::now(),
        });

        let token_usage = openai_response
            .usage
            .as_ref()
            .and_then(parse_openai_token_usage);
        let audio_seconds = openai_response.duration.or_else(|| {
            openai_response
                .usage
                .as_ref()
                .and_then(parse_openai_audio_seconds)
        });
        let segment_count = openai_response.segments.as_ref().map(std::vec::Vec::len);
        let word_count = openai_response.words.as_ref().map(std::vec::Vec::len);
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);
        let provider_processing_ms = headers
            .get("openai-processing-ms")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        let rate_limit_headers = extract_rate_limit_headers(&headers);

        let text = openai_response.text;
        Ok(TranscriptionResult {
            text,
            metadata: TranscriptionMetadata {
                request_latency_ms: Some(request_latency_ms),
                session_elapsed_ms: None,
                request_id,
                provider_processing_ms,
                detected_language: openai_response.language,
                audio_seconds,
                segment_count,
                word_count,
                token_usage,
                provider_specific: Some(ProviderSpecificMetadata::OpenAI(OpenAIProviderMetadata {
                    model: openai_response.model,
                    usage_raw: openai_response.usage,
                    rate_limit_headers,
                    unknown_event_types: Vec::new(),
                    realtime: None,
                })),
            },
            diarization: None,
        })
    }

    async fn transcribe_stream(
        &self,
        audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        file_name: &str,
    ) -> Result<TranscriptionResult, TalkError> {
        // Convert the mpsc::Receiver into a Stream of Result<Vec<u8>, io::Error>
        let byte_stream = ReceiverStream::new(audio_stream).map(Ok::<Vec<u8>, std::io::Error>);

        // Wrap the stream into a reqwest body for streaming upload
        let body = reqwest::Body::wrap_stream(byte_stream);

        // Create multipart form with streaming audio, model, and response format
        let form = reqwest::multipart::Form::new()
            .text("model", self.config.model.clone())
            .text("response_format", "json")
            .part(
                "file",
                reqwest::multipart::Part::stream(body).file_name(file_name.to_string()),
            );

        let started = Instant::now();
        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|err| {
                TalkError::Transcription(format!(
                    "Failed to send streaming request to OpenAI API: {:#}",
                    err
                ))
            })?;

        let request_latency_ms = started.elapsed().as_millis() as u64;
        let headers = response.headers().clone();

        // Check response status
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(TalkError::Transcription(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        // Parse JSON response
        let openai_response: OpenAIResponse = response.json().await.map_err(|err| {
            TalkError::Transcription(format!("Failed to parse OpenAI API response: {}", err))
        })?;

        let token_usage = openai_response
            .usage
            .as_ref()
            .and_then(parse_openai_token_usage);
        let audio_seconds = openai_response.duration.or_else(|| {
            openai_response
                .usage
                .as_ref()
                .and_then(parse_openai_audio_seconds)
        });
        let segment_count = openai_response.segments.as_ref().map(std::vec::Vec::len);
        let word_count = openai_response.words.as_ref().map(std::vec::Vec::len);
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);
        let provider_processing_ms = headers
            .get("openai-processing-ms")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        let rate_limit_headers = extract_rate_limit_headers(&headers);

        let text = openai_response.text;
        Ok(TranscriptionResult {
            text,
            metadata: TranscriptionMetadata {
                request_latency_ms: Some(request_latency_ms),
                session_elapsed_ms: None,
                request_id,
                provider_processing_ms,
                detected_language: openai_response.language,
                audio_seconds,
                segment_count,
                word_count,
                token_usage,
                provider_specific: Some(ProviderSpecificMetadata::OpenAI(OpenAIProviderMetadata {
                    model: openai_response.model,
                    usage_raw: openai_response.usage,
                    rate_limit_headers,
                    unknown_event_types: Vec::new(),
                    realtime: None,
                })),
            },
            diarization: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn test_new_uses_default_endpoint_when_url_is_none() {
        let config = OpenAIConfig {
            api_key: "key".to_string(),
            url: None,
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::new(config).expect("build client");
        assert_eq!(
            transcriber.endpoint,
            "https://api.openai.com/v1/audio/transcriptions"
        );
    }

    #[test]
    fn test_new_uses_custom_url_for_endpoint() {
        let config = OpenAIConfig {
            api_key: "key".to_string(),
            url: Some("https://custom.example.com".to_string()),
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::new(config).expect("build client");
        assert_eq!(
            transcriber.endpoint,
            "https://custom.example.com/v1/audio/transcriptions"
        );
    }

    #[test]
    fn test_new_trims_trailing_slash_from_url() {
        let config = OpenAIConfig {
            api_key: "key".to_string(),
            url: Some("https://custom.example.com/".to_string()),
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::new(config).expect("build client");
        assert_eq!(
            transcriber.endpoint,
            "https://custom.example.com/v1/audio/transcriptions"
        );
    }

    #[tokio::test]
    async fn test_openai_transcriber_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer sk-test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "This is an OpenAI transcription"
            })))
            .mount(&mock_server)
            .await;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        let config = OpenAIConfig {
            api_key: "sk-test-key".to_string(),
            url: None,
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
        )
        .expect("build client");

        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "This is an OpenAI transcription");
    }

    #[tokio::test]
    async fn test_openai_transcriber_stream_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer sk-test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Streamed OpenAI transcription"
            })))
            .mount(&mock_server)
            .await;

        let config = OpenAIConfig {
            api_key: "sk-test-key".to_string(),
            url: None,
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
        )
        .expect("build client");

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            tx.send(vec![0u8; 100]).await.unwrap();
            tx.send(vec![1u8; 200]).await.unwrap();
        });

        let result = transcriber.transcribe_stream(rx, "test.ogg").await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "Streamed OpenAI transcription");
    }

    #[tokio::test]
    async fn test_openai_transcriber_api_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&mock_server)
            .await;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        let config = OpenAIConfig {
            api_key: "invalid-key".to_string(),
            url: None,
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
        )
        .expect("build client");

        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("401"));
    }

    #[tokio::test]
    async fn test_openai_transcriber_file_not_found() {
        let config = OpenAIConfig {
            api_key: "sk-test-key".to_string(),
            url: None,
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::new(config).expect("build client");

        let result = transcriber
            .transcribe_file(Path::new("/nonexistent/file.ogg"))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_is_model_error_openai_code() {
        let err = TalkError::Transcription(
            r#"OpenAI API error (404): {"error":{"message":"The model 'bad' does not exist","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#.to_string(),
        );
        assert!(is_model_error(&err));
    }

    #[test]
    fn test_is_model_error_does_not_exist() {
        let err = TalkError::Transcription(
            "The model 'xyz' does not exist or you do not have access".to_string(),
        );
        assert!(is_model_error(&err));
    }

    #[test]
    fn test_is_model_error_negative_network() {
        let err = TalkError::Transcription("connection timed out".to_string());
        assert!(!is_model_error(&err));
    }

    #[test]
    fn test_is_model_error_negative_auth() {
        let err = TalkError::Transcription("OpenAI API error (401): Unauthorized".to_string());
        assert!(!is_model_error(&err));
    }

    #[tokio::test]
    async fn test_enrich_non_model_error_unchanged() {
        let err = TalkError::Transcription("connection timed out".to_string());
        let enriched = enrich_model_error(err, "key", "whisper-1", API_BASE).await;
        assert_eq!(
            enriched.to_string(),
            "Transcription error: connection timed out"
        );
    }

    #[test]
    fn test_is_transcription_model_matches() {
        assert!(is_transcription_model("whisper-1"));
        assert!(is_transcription_model("gpt-4o-transcribe"));
        assert!(!is_transcription_model("gpt-4o"));
    }

    #[tokio::test]
    async fn test_openai_transcriber_invalid_json_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("invalid json"))
            .mount(&mock_server)
            .await;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        let config = OpenAIConfig {
            api_key: "sk-test-key".to_string(),
            url: None,
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
        )
        .expect("build client");

        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }
}
