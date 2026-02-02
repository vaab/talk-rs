//! Mistral API transcription backend.
//!
//! This module provides a `Transcriber` implementation that uses the Mistral API
//! to transcribe audio files.

use crate::core::config::MistralConfig;
use crate::core::error::TalkError;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::path::Path;
use tokio::fs::File;

use super::Transcriber;

/// Mistral API transcription endpoint.
const MISTRAL_API_ENDPOINT: &str = "https://api.mistral.ai/v1/audio/transcriptions";

/// Response from Mistral API transcription endpoint.
#[derive(Debug, Deserialize)]
struct MistralResponse {
    /// The transcribed text.
    text: String,
}

/// Transcriber implementation using the Mistral API.
///
/// This implementation sends audio files to the Mistral API for transcription
/// and parses the JSON response to extract the transcribed text.
pub struct MistralTranscriber {
    /// HTTP client for making requests to the Mistral API.
    client: Client,
    /// Mistral API configuration (contains API key).
    config: MistralConfig,
    /// API endpoint URL (can be overridden for testing).
    endpoint: String,
}

impl MistralTranscriber {
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
impl Transcriber for MistralTranscriber {
    async fn transcribe_file(&self, audio_path: &Path) -> Result<String, TalkError> {
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
        let form = reqwest::multipart::Form::new()
            .text("model", "voxtral-mini-latest")
            .part(
                "file",
                reqwest::multipart::Part::stream(file).file_name(file_name),
            );

        // Send request to Mistral API
        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|err| {
                TalkError::Transcription(format!("Failed to send request to Mistral API: {}", err))
            })?;

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

        Ok(mistral_response.text)
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
        };
        let transcriber = MistralTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
        );

        // Transcribe the file
        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "This is a test transcription");
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
        };
        let transcriber = MistralTranscriber::with_endpoint(
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
        };
        let transcriber = MistralTranscriber::new(config);

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
        };
        let transcriber = MistralTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
        );

        // Transcribe the file
        let result = transcriber.transcribe_file(temp_file.path()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }
}
