//! OpenAI API batch transcription backend.
//!
//! This module provides a [`BatchTranscriber`] implementation that uses the
//! OpenAI API to transcribe audio files (Whisper, GPT-4o-transcribe, etc.).

use crate::core::config::OpenAIConfig;
use crate::core::error::TalkError;
use crate::core::transcription::{
    OpenAIProviderMetadata, ProviderSpecificMetadata, TokenUsage, TranscriptionMetadata,
    TranscriptionResult,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;
use tokio::fs::File;
use tokio_stream::wrappers::ReceiverStream;

use super::BatchTranscriber;

/// OpenAI API transcription endpoint.
const OPENAI_API_ENDPOINT: &str = "https://api.openai.com/v1/audio/transcriptions";

/// Timeout for batch file upload transcription requests.
const BATCH_FILE_TIMEOUT: Duration = Duration::from_secs(300);

/// Timeout for the lightweight model-listing preflight check.
const VALIDATE_TIMEOUT: Duration = Duration::from_secs(10);

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

fn parse_u64_field(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(|v| {
        v.as_u64().or_else(|| {
            v.as_i64()
                .and_then(|n| if n >= 0 { Some(n as u64) } else { None })
        })
    })
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

// ── Shared model validation ─────────────────────────────────────────

/// Response from the OpenAI `/v1/models` endpoint.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

/// A single entry in the models list.
#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

/// Validate that `model` is available in the OpenAI account reachable
/// at `api_base` (e.g. `"https://api.openai.com"`).
///
/// On success, returns `Ok(())`.  On failure, returns a helpful error
/// that lists the available transcription models so the user can pick
/// a valid one.
///
/// This function is shared between [`OpenAIBatchTranscriber`] and
/// [`super::openai_realtime::OpenAIRealtimeTranscriber`].
pub(crate) async fn validate_openai_model(
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
        .map_err(|e| TalkError::Config(format!("Failed to connect to OpenAI API: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(TalkError::Config(format!(
            "OpenAI API error ({}): {}",
            status, body
        )));
    }

    let models: ModelsResponse = response
        .json()
        .await
        .map_err(|e| TalkError::Config(format!("Failed to parse OpenAI models response: {}", e)))?;

    // Check whether the requested model exists.
    if models.data.iter().any(|m| m.id == model) {
        return Ok(());
    }

    // Model not found — collect transcription-relevant alternatives.
    let mut transcription_models: Vec<&str> = models
        .data
        .iter()
        .map(|m| m.id.as_str())
        .filter(|id| id.contains("whisper") || id.contains("transcri"))
        .collect();
    transcription_models.sort();

    if transcription_models.is_empty() {
        Err(TalkError::Config(format!(
            "Model '{}' not found in OpenAI account",
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
}

impl OpenAIBatchTranscriber {
    /// Create a new OpenAI transcriber with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - OpenAI API configuration containing the API key
    pub fn new(config: OpenAIConfig) -> Self {
        Self {
            client: Client::new(),
            config,
            endpoint: OPENAI_API_ENDPOINT.to_string(),
        }
    }

    /// Create a new OpenAI transcriber with a custom endpoint (for testing).
    ///
    /// # Arguments
    ///
    /// * `config` - OpenAI API configuration containing the API key
    /// * `endpoint` - Custom API endpoint URL
    #[cfg(test)]
    pub fn with_endpoint(config: OpenAIConfig, endpoint: String) -> Self {
        Self {
            client: Client::new(),
            config,
            endpoint,
        }
    }
}

#[async_trait]
impl BatchTranscriber for OpenAIBatchTranscriber {
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

        // Open the audio file
        let file = File::open(audio_path).await.map_err(|err| {
            TalkError::Transcription(format!("Failed to open audio file: {}", err))
        })?;

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
                reqwest::multipart::Part::stream(file).file_name(file_name),
            );

        let started = Instant::now();
        // Send request with timeout
        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .timeout(BATCH_FILE_TIMEOUT)
            .multipart(form)
            .send()
            .await
            .map_err(|err| {
                TalkError::Transcription(format!("Failed to send request to OpenAI API: {}", err))
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
        // Send request to OpenAI API with extended timeout for long recordings
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
                    "Failed to send streaming request to OpenAI API: {}",
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
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
        );

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
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
        );

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
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
        );

        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("401"));
    }

    #[tokio::test]
    async fn test_openai_transcriber_file_not_found() {
        let config = OpenAIConfig {
            api_key: "sk-test-key".to_string(),
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::new(config);

        let result = transcriber
            .transcribe_file(Path::new("/nonexistent/file.ogg"))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
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
            model: "whisper-1".to_string(),
            realtime_model: "gpt-4o-mini-transcribe".to_string(),
        };
        let transcriber = OpenAIBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
        );

        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }
}
