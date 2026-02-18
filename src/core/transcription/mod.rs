//! Transcription interfaces and implementations.
//!
//! This module provides traits and implementations for transcribing audio
//! to text using various backends (Mistral, OpenAI).
//!
//! Two traits model the two modes of operation:
//!
//! - [`BatchTranscriber`]: file or byte-stream in, full text out.
//! - [`RealtimeTranscriber`]: raw PCM stream in, incremental event stream out.

use crate::core::config::{Config, Provider};
use crate::core::error::TalkError;
use async_trait::async_trait;
use std::path::Path;

pub mod mistral;
pub mod openai;
pub mod openai_realtime;
pub mod realtime;

pub use mistral::MistralBatchTranscriber;
pub use openai::OpenAIBatchTranscriber;
pub use openai_realtime::OpenAIRealtimeTranscriber;
pub use realtime::{MistralRealtimeTranscriber, TranscriptionEvent};

/// Result type for transcription operations.
pub struct TranscriptionResult {
    /// The transcribed text from the audio file.
    pub text: String,
}

// ── Batch trait ──────────────────────────────────────────────────────

/// Batch transcription: file or byte-stream in, full text out.
///
/// Implementations should handle file I/O, API communication, and error
/// handling.  All implementations must be `Send + Sync` for use in async
/// contexts.
#[async_trait]
pub trait BatchTranscriber: Send + Sync {
    /// Pre-flight check: verify API connectivity and model validity.
    ///
    /// Called before starting audio capture so the user gets immediate
    /// feedback when a provider is misconfigured or a model name is
    /// invalid.  Implementations should make a lightweight API call
    /// (e.g. list available models) and return a helpful error with
    /// available alternatives on failure.
    async fn validate(&self) -> Result<(), TalkError>;

    /// Transcribe an audio file and return the text.
    async fn transcribe_file(&self, audio_path: &Path) -> Result<String, TalkError>;

    /// Transcribe audio from a streaming channel and return the text.
    ///
    /// The channel carries chunks of encoded audio bytes (e.g. OGG Opus).
    /// The HTTP connection is opened immediately and chunks are streamed
    /// as they arrive — this is *not* a record-then-upload flow.
    async fn transcribe_stream(
        &self,
        audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        file_name: &str,
    ) -> Result<String, TalkError>;
}

// ── Realtime trait ───────────────────────────────────────────────────

/// Realtime transcription: raw PCM stream in, incremental events out.
#[async_trait]
pub trait RealtimeTranscriber: Send + Sync {
    /// Pre-flight check: verify API connectivity and model validity.
    ///
    /// Same purpose as [`BatchTranscriber::validate`] — called before
    /// starting audio capture to give the user immediate feedback.
    async fn validate(&self) -> Result<(), TalkError>;

    /// Connect and start streaming.  Returns a channel of events.
    async fn transcribe_realtime(
        &self,
        audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    ) -> Result<tokio::sync::mpsc::Receiver<TranscriptionEvent>, TalkError>;
}

// ── Factory ──────────────────────────────────────────────────────────

/// Create a batch transcriber for the given provider.
///
/// When `model` is `Some`, it overrides the config default for that
/// provider (the `--model` CLI flag).
pub fn create_batch_transcriber(
    config: &Config,
    provider: Provider,
    model: Option<&str>,
) -> Result<Box<dyn BatchTranscriber>, TalkError> {
    match provider {
        Provider::Mistral => {
            let mut cfg = config.providers.mistral.clone().ok_or_else(|| {
                TalkError::Config(
                    "Mistral provider selected but providers.mistral is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.mistral.api_key is required".to_string(),
                ));
            }
            if let Some(m) = model {
                cfg.model = m.to_string();
            }
            Ok(Box::new(MistralBatchTranscriber::new(cfg)))
        }
        Provider::OpenAI => {
            let mut cfg = config.providers.openai.clone().ok_or_else(|| {
                TalkError::Config(
                    "OpenAI provider selected but providers.openai is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.openai.api_key is required".to_string(),
                ));
            }
            if let Some(m) = model {
                cfg.model = m.to_string();
            }
            Ok(Box::new(OpenAIBatchTranscriber::new(cfg)))
        }
    }
}

/// Create a realtime transcriber for the given provider.
///
/// When `model` is `Some`, it overrides the config default for that
/// provider's realtime model (the `--model` CLI flag).
pub fn create_realtime_transcriber(
    config: &Config,
    provider: Provider,
    model: Option<&str>,
) -> Result<Box<dyn RealtimeTranscriber>, TalkError> {
    match provider {
        Provider::Mistral => {
            let cfg = config.providers.mistral.clone().ok_or_else(|| {
                TalkError::Config(
                    "Mistral provider selected but providers.mistral is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.mistral.api_key is required".to_string(),
                ));
            }
            Ok(Box::new(MistralRealtimeTranscriber::new(cfg)))
        }
        Provider::OpenAI => {
            let mut cfg = config.providers.openai.clone().ok_or_else(|| {
                TalkError::Config(
                    "OpenAI provider selected but providers.openai is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.openai.api_key is required".to_string(),
                ));
            }
            if let Some(m) = model {
                cfg.realtime_model = m.to_string();
            }
            Ok(Box::new(OpenAIRealtimeTranscriber::new(cfg)))
        }
    }
}

// ── Mock ─────────────────────────────────────────────────────────────

/// Mock batch transcriber for testing.
///
/// Returns a hardcoded transcription result without making any API calls.
pub struct MockBatchTranscriber {
    /// The text to return when transcribe is called.
    pub response_text: String,
}

impl MockBatchTranscriber {
    /// Create a new mock transcriber with the given response text.
    pub fn new(response_text: impl Into<String>) -> Self {
        Self {
            response_text: response_text.into(),
        }
    }
}

#[async_trait]
impl BatchTranscriber for MockBatchTranscriber {
    async fn validate(&self) -> Result<(), TalkError> {
        Ok(())
    }

    async fn transcribe_file(&self, _audio_path: &Path) -> Result<String, TalkError> {
        Ok(self.response_text.clone())
    }

    async fn transcribe_stream(
        &self,
        mut audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        _file_name: &str,
    ) -> Result<String, TalkError> {
        // Drain the stream
        while audio_stream.recv().await.is_some() {}
        Ok(self.response_text.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_transcriber_returns_text() {
        let mock = MockBatchTranscriber::new("Hello, world!");
        let result = mock.transcribe_file(Path::new("/tmp/test.wav")).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, world!");
    }

    #[tokio::test]
    async fn test_mock_transcriber_stream() {
        let mock = MockBatchTranscriber::new("Streamed transcription");
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        // Send some fake audio data
        tx.send(vec![0u8; 100]).await.unwrap();
        tx.send(vec![1u8; 200]).await.unwrap();
        drop(tx); // Close the channel

        let result = mock.transcribe_stream(rx, "test.wav").await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Streamed transcription");
    }

    #[tokio::test]
    async fn test_mock_transcriber_ignores_path() {
        let mock = MockBatchTranscriber::new("Fixed response");
        let result1 = mock.transcribe_file(Path::new("/path/one.wav")).await;
        let result2 = mock.transcribe_file(Path::new("/path/two.wav")).await;

        assert_eq!(result1.unwrap(), "Fixed response");
        assert_eq!(result2.unwrap(), "Fixed response");
    }

    #[test]
    fn test_provider_from_str() {
        assert_eq!("mistral".parse::<Provider>().unwrap(), Provider::Mistral);
        assert_eq!("openai".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert_eq!("OpenAI".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert!("unknown".parse::<Provider>().is_err());
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(Provider::Mistral.to_string(), "mistral");
        assert_eq!(Provider::OpenAI.to_string(), "openai");
    }
}
