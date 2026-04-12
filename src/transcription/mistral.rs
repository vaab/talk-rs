//! Mistral API batch transcription backend.
//!
//! This module provides a [`BatchTranscriber`] implementation that uses the
//! Mistral API to transcribe audio files.

use crate::config::MistralConfig;
use crate::error::TalkError;
use crate::transcription::{
    DiarizationSegment, MistralProviderMetadata, ProviderSpecificMetadata, TokenUsage,
    TranscriptionMetadata, TranscriptionResult,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::path::Path;
use std::time::Instant;
use tokio::fs::File;

use super::http::{build_client, parse_u64_field, proportional_timeout};
use super::BatchTranscriber;

/// Default API base URL for the Mistral API.
pub(crate) const API_BASE: &str = "https://api.mistral.ai";

/// Response from Mistral API transcription endpoint.
#[derive(Debug, Deserialize)]
struct MistralResponse {
    /// The transcribed text.
    text: String,
    /// Model identifier returned by API.
    #[serde(default)]
    model: Option<String>,
    /// Detected language code.
    #[serde(default)]
    language: Option<String>,
    /// Optional transcript segments.
    #[serde(default)]
    segments: Option<Vec<serde_json::Value>>,
    /// Usage payload returned by Mistral.
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

fn parse_mistral_token_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    let input_tokens = parse_u64_field(usage, "prompt_tokens");
    let output_tokens = parse_u64_field(usage, "completion_tokens");
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

fn parse_mistral_audio_seconds(usage: &serde_json::Value) -> Option<f64> {
    usage.get("prompt_audio_seconds").and_then(|v| v.as_f64())
}

/// Extract diarization segments from the Mistral response segments.
///
/// Each segment is expected to have `speaker_id`, `start`, `end`, and
/// `text` fields when diarization was requested.  Segments without a
/// `speaker_id` are skipped (they come from non-diarized responses).
fn parse_diarization_segments(segments: &[serde_json::Value]) -> Option<Vec<DiarizationSegment>> {
    let mut result = Vec::new();
    for seg in segments {
        let speaker = seg.get("speaker_id").and_then(|v| v.as_str());
        let Some(speaker) = speaker else {
            continue;
        };
        let start = seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let end = seg.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let text = seg
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        result.push(DiarizationSegment {
            speaker: speaker.to_string(),
            start,
            end,
            text,
        });
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

// ── Error detection ─────────────────────────────────────────────────

/// Check if an error is a model-not-found error from the Mistral API.
///
/// Mistral returns `"Unknown model: xyz"` or similar messages.
pub(crate) fn is_model_error(error: &TalkError) -> bool {
    let msg = error.to_string().to_lowercase();
    msg.contains("unknown model")
        || (msg.contains("model") && (msg.contains("not found") || msg.contains("invalid model")))
}

/// Filter predicate for Mistral transcription-relevant models.
pub(crate) fn is_transcription_model(model_id: &str) -> bool {
    model_id.contains("voxtral") || model_id.contains("transcri")
}

/// Enrich a model error with available Mistral transcription models.
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

/// Validate that `model` is available in the Mistral account reachable
/// at `api_base`.
pub(crate) async fn validate_mistral_model(
    api_key: &str,
    model: &str,
    api_base: &str,
) -> Result<(), TalkError> {
    super::http::validate_model("Mistral", api_key, model, api_base, is_transcription_model).await
}

/// Transcriber implementation using the Mistral API.
///
/// This implementation sends audio files to the Mistral API for transcription
/// and parses the JSON response to extract the transcribed text.
pub struct MistralBatchTranscriber {
    /// HTTP client for making requests to the Mistral API.
    client: Client,
    /// Mistral API configuration (contains API key).
    config: MistralConfig,
    /// API endpoint URL (can be overridden for testing).
    endpoint: String,
    /// Whether to request speaker diarization (V2 models only).
    diarize: bool,
}

impl MistralBatchTranscriber {
    /// Create a new Mistral transcriber with the given configuration.
    ///
    /// The transcription endpoint is derived from `config.url` (if set)
    /// by appending `/v1/audio/transcriptions`.  When `config.url` is
    /// `None`, the default Mistral API base URL is used.
    ///
    /// # Arguments
    ///
    /// * `config` - Mistral API configuration containing the API key
    /// * `diarize` - Request speaker diarization (requires V2 model)
    pub fn new(config: MistralConfig, diarize: bool) -> Result<Self, TalkError> {
        let base = config.url.as_deref().unwrap_or(API_BASE);
        let endpoint = format!("{}/v1/audio/transcriptions", base.trim_end_matches('/'));
        Ok(Self {
            client: build_client()?,
            config,
            endpoint,
            diarize,
        })
    }

    /// Create a new Mistral transcriber with a custom endpoint (for testing).
    ///
    /// # Arguments
    ///
    /// * `config` - Mistral API configuration containing the API key
    /// * `endpoint` - Custom API endpoint URL
    /// * `diarize` - Request speaker diarization
    #[cfg(test)]
    pub fn with_endpoint(
        config: MistralConfig,
        endpoint: String,
        diarize: bool,
    ) -> Result<Self, TalkError> {
        Ok(Self {
            client: build_client()?,
            config,
            endpoint,
            diarize,
        })
    }
}

#[async_trait]
impl BatchTranscriber for MistralBatchTranscriber {
    async fn validate(&self) -> Result<(), TalkError> {
        let api_base = self
            .endpoint
            .find("/v1/")
            .map(|pos| &self.endpoint[..pos])
            .unwrap_or(&self.endpoint);
        validate_mistral_model(&self.config.api_key, &self.config.model, api_base).await
    }

    async fn transcribe_file(&self, audio_path: &Path) -> Result<TranscriptionResult, TalkError> {
        // Verify file exists
        if !audio_path.exists() {
            return Err(TalkError::Transcription(format!(
                "Audio file not found: {}",
                audio_path.display()
            )));
        }

        // Open the audio file and determine its size so reqwest can
        // set an explicit Content-Length header.  Without a known
        // length the request is sent with Transfer-Encoding: chunked,
        // which the Mistral API rejects with 411 Length Required.
        let file = File::open(audio_path).await.map_err(|err| {
            TalkError::Transcription(format!("Failed to open audio file: {}", err))
        })?;
        let file_len = file
            .metadata()
            .await
            .map_err(|err| {
                TalkError::Transcription(format!("Failed to read audio file metadata: {}", err))
            })?
            .len();

        // Get file name for multipart form
        let file_name = audio_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("audio.wav")
            .to_string();

        // Create multipart form with audio file and model
        let mut form = reqwest::multipart::Form::new()
            .text("model", self.config.model.clone())
            .part(
                "file",
                reqwest::multipart::Part::stream_with_length(file, file_len).file_name(file_name),
            );

        // Add context bias if configured
        if let Some(ref bias) = self.config.context_bias {
            form = form.text("context_bias", bias.clone());
        }

        // Add diarization parameters (V2 models only)
        if self.diarize {
            form = form
                .text("diarize", "true")
                .text("timestamp_granularities", "segment");
        }

        // Compute a payload-proportional wall-clock timeout for this
        // request.  See `proportional_timeout` for the formula and
        // rationale.  This bounds the tail when the server accepts
        // the connection and then hangs at the application layer
        // (which TCP-level defences cannot detect).
        let request_timeout = proportional_timeout(file_len);
        log::warn!(
            "mistral batch file: request timeout = {}s (audio = {} KB)",
            request_timeout.as_secs(),
            file_len / 1024
        );

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
                TalkError::Transcription(format!(
                    "Failed to send request to Mistral API: {:#}",
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
                "Mistral API error ({}): {}",
                status, body
            )));
        }

        // Parse JSON response
        let mistral_response: MistralResponse = response.json().await.map_err(|err| {
            TalkError::Transcription(format!("Failed to parse Mistral API response: {}", err))
        })?;

        let token_usage = mistral_response
            .usage
            .as_ref()
            .and_then(parse_mistral_token_usage);
        let audio_seconds = mistral_response
            .usage
            .as_ref()
            .and_then(parse_mistral_audio_seconds);
        let segment_count = mistral_response.segments.as_ref().map(std::vec::Vec::len);
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let diarization = mistral_response
            .segments
            .as_deref()
            .and_then(parse_diarization_segments);

        let text = mistral_response.text;
        Ok(TranscriptionResult {
            text,
            metadata: TranscriptionMetadata {
                request_latency_ms: Some(request_latency_ms),
                session_elapsed_ms: None,
                request_id,
                provider_processing_ms: None,
                detected_language: mistral_response.language,
                audio_seconds,
                segment_count,
                word_count: None,
                token_usage,
                provider_specific: Some(ProviderSpecificMetadata::Mistral(
                    MistralProviderMetadata {
                        model: mistral_response.model,
                        usage_raw: mistral_response.usage,
                        unknown_event_types: Vec::new(),
                    },
                )),
            },
            diarization,
        })
    }

    async fn transcribe_stream(
        &self,
        audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        file_name: &str,
    ) -> Result<TranscriptionResult, TalkError> {
        // Collect all audio chunks into a single buffer so we can
        // provide an explicit Content-Length.  The Mistral API rejects
        // chunked Transfer-Encoding with 411 Length Required.
        log::warn!("[DBG] mistral stream: awaiting audio chunks from encoder");
        let collect_start = Instant::now();
        let mut audio_buf = Vec::new();
        let mut rx = audio_stream;
        while let Some(chunk) = rx.recv().await {
            audio_buf.extend_from_slice(&chunk);
        }
        let audio_len = audio_buf.len() as u64;
        log::info!(
            "streaming upload: collected {} bytes for Mistral batch request",
            audio_len
        );
        log::warn!(
            "[DBG] mistral stream: audio collected ({} bytes in {}ms), building request",
            audio_len,
            collect_start.elapsed().as_millis()
        );

        // Create multipart form with known-length audio
        let mut form = reqwest::multipart::Form::new()
            .text("model", self.config.model.clone())
            .part(
                "file",
                reqwest::multipart::Part::stream_with_length(audio_buf, audio_len)
                    .file_name(file_name.to_string()),
            );

        // Add context bias if configured
        if let Some(ref bias) = self.config.context_bias {
            form = form.text("context_bias", bias.clone());
        }

        // Add diarization parameters (V2 models only)
        if self.diarize {
            form = form
                .text("diarize", "true")
                .text("timestamp_granularities", "segment");
        }

        // Compute a payload-proportional wall-clock timeout for this
        // request.  See `proportional_timeout` for the formula and
        // rationale.  This bounds the tail when the server accepts
        // the connection and then hangs at the application layer
        // (which TCP-level defences cannot detect).
        let request_timeout = proportional_timeout(audio_len);
        log::warn!(
            "mistral stream: request timeout = {}s (audio = {} KB)",
            request_timeout.as_secs(),
            audio_len / 1024
        );

        log::warn!("[DBG] mistral stream: POST {} beginning", self.endpoint);
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
                log::warn!(
                    "[DBG] mistral stream: send failed after {}ms: \
                     is_connect={}, is_timeout={}, is_request={}, \
                     is_body={}, is_decode={}, status={:?}",
                    started.elapsed().as_millis(),
                    err.is_connect(),
                    err.is_timeout(),
                    err.is_request(),
                    err.is_body(),
                    err.is_decode(),
                    err.status()
                );
                TalkError::Transcription(format!(
                    "Failed to send streaming request to Mistral API: {:#}",
                    err
                ))
            })?;

        let request_latency_ms = started.elapsed().as_millis() as u64;
        log::warn!(
            "[DBG] mistral stream: response headers received after {}ms, status={}",
            request_latency_ms,
            response.status()
        );
        let headers = response.headers().clone();

        // Check response status
        if !response.status().is_success() {
            let status = response.status();
            log::warn!(
                "[DBG] mistral stream: non-success status={}, reading error body",
                status
            );
            let body_start = Instant::now();
            let body = response.text().await.unwrap_or_default();
            log::warn!(
                "[DBG] mistral stream: error body read ({} bytes in {}ms)",
                body.len(),
                body_start.elapsed().as_millis()
            );
            return Err(TalkError::Transcription(format!(
                "Mistral API error ({}): {}",
                status, body
            )));
        }

        // Parse JSON response
        log::warn!("[DBG] mistral stream: reading JSON body");
        let body_start = Instant::now();
        let mistral_response: MistralResponse = response.json().await.map_err(|err| {
            log::warn!(
                "[DBG] mistral stream: JSON parse failed after {}ms: \
                 is_connect={}, is_timeout={}, is_body={}, is_decode={}",
                body_start.elapsed().as_millis(),
                err.is_connect(),
                err.is_timeout(),
                err.is_body(),
                err.is_decode()
            );
            TalkError::Transcription(format!("Failed to parse Mistral API response: {}", err))
        })?;
        log::warn!(
            "[DBG] mistral stream: JSON parsed ({} chars text) after {}ms body read",
            mistral_response.text.len(),
            body_start.elapsed().as_millis()
        );

        let token_usage = mistral_response
            .usage
            .as_ref()
            .and_then(parse_mistral_token_usage);
        let audio_seconds = mistral_response
            .usage
            .as_ref()
            .and_then(parse_mistral_audio_seconds);
        let segment_count = mistral_response.segments.as_ref().map(std::vec::Vec::len);
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);

        let diarization = mistral_response
            .segments
            .as_deref()
            .and_then(parse_diarization_segments);

        let text = mistral_response.text;
        Ok(TranscriptionResult {
            text,
            metadata: TranscriptionMetadata {
                request_latency_ms: Some(request_latency_ms),
                session_elapsed_ms: None,
                request_id,
                provider_processing_ms: None,
                detected_language: mistral_response.language,
                audio_seconds,
                segment_count,
                word_count: None,
                token_usage,
                provider_specific: Some(ProviderSpecificMetadata::Mistral(
                    MistralProviderMetadata {
                        model: mistral_response.model,
                        usage_raw: mistral_response.usage,
                        unknown_event_types: Vec::new(),
                    },
                )),
            },
            diarization,
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
        let config = MistralConfig {
            api_key: "key".to_string(),
            url: None,
            model: "voxtral-mini-2507".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::new(config, false).expect("build client");
        assert_eq!(
            transcriber.endpoint,
            "https://api.mistral.ai/v1/audio/transcriptions"
        );
    }

    #[test]
    fn test_new_uses_custom_url_for_endpoint() {
        let config = MistralConfig {
            api_key: "key".to_string(),
            url: Some("https://custom.example.com".to_string()),
            model: "voxtral-mini-2507".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::new(config, false).expect("build client");
        assert_eq!(
            transcriber.endpoint,
            "https://custom.example.com/v1/audio/transcriptions"
        );
    }

    #[test]
    fn test_new_trims_trailing_slash_from_url() {
        let config = MistralConfig {
            api_key: "key".to_string(),
            url: Some("https://custom.example.com/".to_string()),
            model: "voxtral-mini-2507".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::new(config, false).expect("build client");
        assert_eq!(
            transcriber.endpoint,
            "https://custom.example.com/v1/audio/transcriptions"
        );
    }

    #[tokio::test]
    async fn test_mistral_transcriber_success() {
        // Start mock server
        let mock_server = MockServer::start().await;

        // Mock the Mistral API endpoint
        // Note: wiremock multipart matching is limited, so we just verify method, path, and auth header
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "This is a test transcription"
            })))
            .mount(&mock_server)
            .await;

        // Create a temporary audio file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        // Create transcriber with mock server URL
        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        // Transcribe the file
        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "This is a test transcription");
    }

    #[tokio::test]
    async fn test_mistral_transcriber_stream_success() {
        // Start mock server
        let mock_server = MockServer::start().await;

        // Mock the Mistral API endpoint
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Streamed transcription result"
            })))
            .mount(&mock_server)
            .await;

        // Create transcriber with mock server URL
        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        // Create a channel and send fake audio data
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            tx.send(vec![0u8; 100]).await.unwrap();
            tx.send(vec![1u8; 200]).await.unwrap();
            // Dropping tx closes the channel
        });

        // Transcribe from the stream
        let result = transcriber.transcribe_stream(rx, "test.wav").await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "Streamed transcription result");
    }

    #[tokio::test]
    async fn test_mistral_transcriber_api_error() {
        // Start mock server
        let mock_server = MockServer::start().await;

        // Mock the Mistral API endpoint to return an error
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&mock_server)
            .await;

        // Create a temporary audio file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        // Create transcriber with mock server URL
        let config = MistralConfig {
            api_key: "invalid-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        // Transcribe the file
        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("401"));
    }

    #[tokio::test]
    async fn test_mistral_transcriber_file_not_found() {
        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::new(config, false).expect("build client");

        let result = transcriber
            .transcribe_file(Path::new("/nonexistent/file.wav"))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_mistral_transcriber_invalid_json_response() {
        // Start mock server
        let mock_server = MockServer::start().await;

        // Mock the Mistral API endpoint to return invalid JSON
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("invalid json"))
            .mount(&mock_server)
            .await;

        // Create a temporary audio file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        // Create transcriber with mock server URL
        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        // Transcribe the file
        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    #[tokio::test]
    async fn test_mistral_diarization_segments_parsed() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Hello. I am fine.",
                "segments": [
                    {
                        "speaker_id": "SPEAKER_00",
                        "start": 0.0,
                        "end": 1.5,
                        "text": "Hello."
                    },
                    {
                        "speaker_id": "SPEAKER_01",
                        "start": 1.5,
                        "end": 3.0,
                        "text": "I am fine."
                    }
                ]
            })))
            .mount(&mock_server)
            .await;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-2602".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            true,
        )
        .expect("build client");

        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.text, "Hello. I am fine.");

        let segments = result.diarization.expect("diarization segments present");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert_eq!(segments[0].text, "Hello.");
        assert_eq!(segments[0].start, 0.0);
        assert_eq!(segments[0].end, 1.5);
        assert_eq!(segments[1].speaker, "SPEAKER_01");
        assert_eq!(segments[1].text, "I am fine.");
    }

    #[test]
    fn test_is_model_error_unknown_model() {
        let err = TalkError::Transcription(
            r#"Mistral API error (422): {"object":"error","message":"Unknown model: bad-model","type":"invalid_request_error"}"#.to_string(),
        );
        assert!(is_model_error(&err));
    }

    #[test]
    fn test_is_model_error_not_found() {
        let err = TalkError::Transcription(
            "Model 'voxtral-mini-9999' not found. Available transcription models: voxtral-mini-2507"
                .to_string(),
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
        let err = TalkError::Transcription("Mistral API error (401): Unauthorized".to_string());
        assert!(!is_model_error(&err));
    }

    #[tokio::test]
    async fn test_enrich_non_model_error_unchanged() {
        let err = TalkError::Transcription("connection timed out".to_string());
        let enriched = enrich_model_error(err, "key", "voxtral-mini-2507", API_BASE).await;
        assert_eq!(
            enriched.to_string(),
            "Transcription error: connection timed out"
        );
    }

    #[test]
    fn test_is_transcription_model_matches() {
        assert!(is_transcription_model("voxtral-mini-2507"));
        assert!(is_transcription_model("voxtral-mini-2602"));
        assert!(is_transcription_model("some-transcription-model"));
        assert!(!is_transcription_model("mistral-large-latest"));
    }

    #[tokio::test]
    async fn test_mistral_no_diarization_without_speaker_id() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Hello world.",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.5,
                        "text": "Hello world."
                    }
                ]
            })))
            .mount(&mock_server)
            .await;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-2602".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_ok());
        let result = result.unwrap();
        // No speaker_id in segments → diarization is None
        assert!(result.diarization.is_none());
    }
}
