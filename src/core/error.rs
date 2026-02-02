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
