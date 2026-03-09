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
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;
use tokio::fs::File;
use tokio_stream::wrappers::ReceiverStream;

use super::BatchTranscriber;

/// Default API base URL for the Mistral API.
const API_BASE: &str = "https://api.mistral.ai";

/// Mistral API transcription endpoint.
const MISTRAL_API_ENDPOINT: &str = "https://api.mistral.ai/v1/audio/transcriptions";

/// Timeout for batch file upload transcription requests.
const BATCH_FILE_TIMEOUT: Duration = Duration::from_secs(300);

/// Timeout for the lightweight model-listing preflight check.
const VALIDATE_TIMEOUT: Duration = Duration::from_secs(10);

/// TCP connect timeout — fail fast when the server is unreachable.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// Build an HTTP client with a TCP connect timeout.
fn build_client() -> Result<Client, TalkError> {
    Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(|e| TalkError::Config(format!("failed to build HTTP client: {}", e)))
}

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

fn parse_u64_field(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(|v| {
        v.as_u64().or_else(|| {
            v.as_i64()
                .and_then(|n| if n >= 0 { Some(n as u64) } else { None })
        })
    })
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
///
/// Returns the error unchanged if it is not a model error or if
/// suggestions cannot be fetched.
pub(crate) async fn enrich_model_error(error: TalkError, api_key: &str, model: &str) -> TalkError {
    if !is_model_error(&error) {
        return error;
    }
    match super::model_suggestions::fetch_transcription_models(
        api_key,
        API_BASE,
        is_transcription_model,
    )
    .await
    {
        Ok(models) if !models.is_empty() => TalkError::Transcription(format!(
            "Model '{}' not found. Available transcription models: {}",
            model,
            models.join(", ")
        )),
        Ok(_) => TalkError::Transcription(format!(
            "Model '{}' not found (no transcription models available in account)",
            model
        )),
        Err(e) => {
            log::warn!("could not fetch model suggestions: {}", e);
            error
        }
    }
}

// ── Shared model validation ─────────────────────────────────────────

/// Response from the Mistral `/v1/models` endpoint.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

/// A single entry in the Mistral models list.
#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

/// Validate that `model` is available in the Mistral account reachable
/// at `api_base` (e.g. `"https://api.mistral.ai"`).
///
/// On success, returns `Ok(())`.  On failure, returns a helpful error
/// that lists the available transcription models.
pub(crate) async fn validate_mistral_model(
    api_key: &str,
    model: &str,
    api_base: &str,
) -> Result<(), TalkError> {
    let models_url = format!("{}/v1/models", api_base);

    let client = build_client()?;
    let response = client
        .get(&models_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .timeout(VALIDATE_TIMEOUT)
        .send()
        .await
        .map_err(|e| TalkError::Config(format!("Failed to connect to Mistral API: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(TalkError::Config(format!(
            "Mistral API error ({}): {}",
            status, body
        )));
    }

    let models: ModelsResponse = response.json().await.map_err(|e| {
        TalkError::Config(format!("Failed to parse Mistral models response: {}", e))
    })?;

    // Check whether the requested model exists.
    if models.data.iter().any(|m| m.id == model) {
        return Ok(());
    }

    // Model not found — collect transcription-relevant alternatives.
    let mut transcription_models: Vec<&str> = models
        .data
        .iter()
        .map(|m| m.id.as_str())
        .filter(|id| id.contains("voxtral") || id.contains("transcri"))
        .collect();
    transcription_models.sort();

    if transcription_models.is_empty() {
        Err(TalkError::Config(format!(
            "Model '{}' not found in Mistral account",
            model
        )))
    } else {
        Err(TalkError::Config(format!(
            "Model '{}' not found. Available transcription models: {}",
            model,
            transcription_models.join(", ")
        )))
    }
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
    /// # Arguments
    ///
    /// * `config` - Mistral API configuration containing the API key
    /// * `diarize` - Request speaker diarization (requires V2 model)
    pub fn new(config: MistralConfig, diarize: bool) -> Result<Self, TalkError> {
        Ok(Self {
            client: build_client()?,
            config,
            endpoint: MISTRAL_API_ENDPOINT.to_string(),
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

        // Open the audio file
        let file = File::open(audio_path).await.map_err(|err| {
            TalkError::Transcription(format!("Failed to open audio file: {}", err))
        })?;

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
                reqwest::multipart::Part::stream(file).file_name(file_name),
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

        let started = Instant::now();
        // [Fix #3] Send request with timeout to prevent hanging on
        // unresponsive servers. Without this, Client::new() has no default
        // timeout and the request could block forever.
        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .timeout(BATCH_FILE_TIMEOUT)
            .multipart(form)
            .send()
            .await
            .map_err(|err| {
                TalkError::Transcription(format!("Failed to send request to Mistral API: {}", err))
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
        // Convert the mpsc::Receiver into a Stream of Result<Vec<u8>, io::Error>
        let byte_stream = ReceiverStream::new(audio_stream).map(Ok::<Vec<u8>, std::io::Error>);

        // Wrap the stream into a reqwest body for streaming upload
        let body = reqwest::Body::wrap_stream(byte_stream);

        // Create multipart form with streaming audio and model
        let mut form = reqwest::multipart::Form::new()
            .text("model", self.config.model.clone())
            .part(
                "file",
                reqwest::multipart::Part::stream(body).file_name(file_name.to_string()),
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

        let started = Instant::now();
        // Send request to Mistral API with extended timeout for long recordings
        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .timeout(Duration::from_secs(600))
            .multipart(form)
            .send()
            .await
            .map_err(|err| {
                TalkError::Transcription(format!(
                    "Failed to send streaming request to Mistral API: {}",
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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
        let enriched = enrich_model_error(err, "key", "voxtral-mini-2507").await;
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
