//! Transcription interfaces and implementations.
//!
//! This module provides traits and implementations for transcribing audio files
//! to text using various backends (Mistral, future: Whisper, Deepgram).

use crate::core::error::TalkError;
use async_trait::async_trait;
use std::path::Path;

pub mod mistral;

pub use mistral::MistralTranscriber;

/// Result type for transcription operations.
pub struct TranscriptionResult {
    /// The transcribed text from the audio file.
    pub text: String,
}

/// Trait for transcribing audio files to text.
///
/// Implementations should handle file I/O, API communication, and error handling.
/// All implementations must be `Send + Sync` for use in async contexts.
#[async_trait]
pub trait Transcriber: Send + Sync {
    /// Transcribe an audio file and return the text.
    ///
    /// # Arguments
    ///
    /// * `audio_path` - Path to the audio file to transcribe
    ///
    /// # Returns
    ///
    /// The transcribed text, or a `TalkError` if transcription fails.
    async fn transcribe_file(&self, audio_path: &Path) -> Result<String, TalkError>;

    /// Transcribe audio from a streaming channel and return the text.
    ///
    /// # Arguments
    ///
    /// * `audio_stream` - A receiver of audio data chunks
    /// * `file_name` - The file name to use for the multipart upload
    ///
    /// # Returns
    ///
    /// The transcribed text, or a `TalkError` if transcription fails.
    async fn transcribe_stream(
        &self,
        audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        file_name: &str,
    ) -> Result<String, TalkError>;
}

/// Mock transcriber for testing.
///
/// Returns a hardcoded transcription result without making any API calls.
pub struct MockTranscriber {
    /// The text to return when transcribe_file is called.
    pub response_text: String,
}

impl MockTranscriber {
    /// Create a new mock transcriber with the given response text.
    pub fn new(response_text: impl Into<String>) -> Self {
        Self {
            response_text: response_text.into(),
        }
    }
}

#[async_trait]
impl Transcriber for MockTranscriber {
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
        let mock = MockTranscriber::new("Hello, world!");
        let result = mock.transcribe_file(Path::new("/tmp/test.wav")).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello, world!");
    }

    #[tokio::test]
    async fn test_mock_transcriber_stream() {
        let mock = MockTranscriber::new("Streamed transcription");
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
        let mock = MockTranscriber::new("Fixed response");
        let result1 = mock.transcribe_file(Path::new("/path/one.wav")).await;
        let result2 = mock.transcribe_file(Path::new("/path/two.wav")).await;

        assert_eq!(result1.unwrap(), "Fixed response");
        assert_eq!(result2.unwrap(), "Fixed response");
    }
}
