//! Error types for talk-rs
//!
//! This module defines the error types used throughout the application.
//! Each major component has its own error variant for clear error handling.

use thiserror::Error;

/// Main error type for talk-rs operations.
#[derive(Error, Debug)]
pub enum TalkError {
    /// Configuration-related errors.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Audio capture/encoding errors.
    #[error("Audio error: {0}")]
    Audio(String),

    /// Transcription API errors.
    #[error("Transcription error: {0}")]
    Transcription(String),

    /// IO operation errors.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Clipboard operation errors.
    #[error("Clipboard error: {0}")]
    Clipboard(String),

    /// Session management errors.
    #[error("Session error: {0}")]
    Session(String),

    /// Pick-level lock present: another process is producing the
    /// authoritative transcript for this recording.
    #[error("Transcription already in progress")]
    TranscriptInProgress,

    /// Caller requested cache-only lookup but the per-model sidecar
    /// cache was empty.  Signals that an API call would be required.
    #[error("Transcription not cached (API call forbidden)")]
    CacheOnly,

    /// Per-model lock present: another process is calling the API
    /// for this specific (recording, provider, model) triple.
    #[error("Transcription for this model already in progress")]
    ModelInProgress,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_error_display() {
        let err = TalkError::Config("missing field".to_string());
        assert_eq!(err.to_string(), "Configuration error: missing field");
    }

    #[test]
    fn test_io_error_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: TalkError = io_err.into();
        assert!(matches!(err, TalkError::Io(_)));
    }
}
