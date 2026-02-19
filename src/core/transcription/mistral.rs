//! Mistral API batch transcription backend.
//!
//! This module provides a [`BatchTranscriber`] implementation that uses the
//! Mistral API to transcribe audio files.

use crate::core::config::MistralConfig;
use crate::core::error::TalkError;
use crate::core::transcription::{
    MistralProviderMetadata, ProviderSpecificMetadata, TokenUsage, TranscriptionMetadata,
    TranscriptionResult,
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

/// Mistral API transcription endpoint.
const MISTRAL_API_ENDPOINT: &str = "https://api.mistral.ai/v1/audio/transcriptions";

/// Timeout for batch file upload transcription requests.
const BATCH_FILE_TIMEOUT: Duration = Duration::from_secs(300);

/// Timeout for the lightweight model-listing preflight check.
const VALIDATE_TIMEOUT: Duration = Duration::from_secs(10);

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

    let client = Client::new();
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
}

impl MistralBatchTranscriber {
    /// Create a new Mistral transcriber with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Mistral API configuration containing the API key
    pub fn new(config: MistralConfig) -> Self {
        Self {
            client: Client::new(),
            config,
            endpoint: MISTRAL_API_ENDPOINT.to_string(),
        }
    }

    /// Create a new Mistral transcriber with a custom endpoint (for testing).
    ///
    /// # Arguments
    ///
    /// * `config` - Mistral API configuration containing the API key
    /// * `endpoint` - Custom API endpoint URL
    #[cfg(test)]
    pub fn with_endpoint(config: MistralConfig, endpoint: String) -> Self {
        Self {
            client: Client::new(),
            config,
            endpoint,
        }
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
        );

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
        );

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
        );

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
        let transcriber = MistralBatchTranscriber::new(config);

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
        );

        // Transcribe the file
        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }
}
