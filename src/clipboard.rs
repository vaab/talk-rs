//! Clipboard interfaces and implementations.
//!
//! This module provides traits and implementations for clipboard operations
//! using various backends (X11 via xclip, future: Wayland via wl-copy).

use crate::error::TalkError;
use async_trait::async_trait;

/// Trait for clipboard operations.
///
/// Implementations should handle reading and writing text to the system clipboard.
/// All implementations must be `Send + Sync` for use in async contexts.
#[async_trait]
pub trait Clipboard: Send + Sync {
    /// Get the current clipboard text content.
    async fn get_text(&self) -> Result<String, TalkError>;

    /// Set the clipboard text content.
    async fn set_text(&self, text: &str) -> Result<(), TalkError>;
}

/// X11 clipboard implementation using the `xclip` command-line tool.
///
/// Requires `xclip` to be installed on the system.
pub struct X11Clipboard;

impl Default for X11Clipboard {
    fn default() -> Self {
        Self
    }
}

impl X11Clipboard {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Clipboard for X11Clipboard {
    async fn get_text(&self) -> Result<String, TalkError> {
        let output = tokio::process::Command::new("xclip")
            .args(["-selection", "clipboard", "-o"])
            .output()
            .await
            .map_err(|e| TalkError::Clipboard(format!("failed to run xclip: {e}")))?;

        if !output.status.success() {
            // Empty clipboard returns non-zero exit code
            return Ok(String::new());
        }

        String::from_utf8(output.stdout)
            .map_err(|e| TalkError::Clipboard(format!("clipboard content is not valid UTF-8: {e}")))
    }

    async fn set_text(&self, text: &str) -> Result<(), TalkError> {
        use tokio::io::AsyncWriteExt;

        let mut child = tokio::process::Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| TalkError::Clipboard(format!("failed to run xclip: {e}")))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes()).await.map_err(|e| {
                TalkError::Clipboard(format!("failed to write to xclip stdin: {e}"))
            })?;
        }

        let status = child
            .wait()
            .await
            .map_err(|e| TalkError::Clipboard(format!("failed to wait for xclip: {e}")))?;

        if !status.success() {
            return Err(TalkError::Clipboard(
                "xclip exited with non-zero status".to_string(),
            ));
        }

        Ok(())
    }
}

/// Mock clipboard for testing.
///
/// Stores clipboard content in memory using thread-safe interior mutability.
pub struct MockClipboard {
    content: std::sync::Arc<tokio::sync::Mutex<String>>,
}

impl Default for MockClipboard {
    fn default() -> Self {
        Self {
            content: std::sync::Arc::new(tokio::sync::Mutex::new(String::new())),
        }
    }
}

impl MockClipboard {
    /// Create a new mock clipboard with empty content.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new mock clipboard with initial content.
    pub fn with_content(text: impl Into<String>) -> Self {
        Self {
            content: std::sync::Arc::new(tokio::sync::Mutex::new(text.into())),
        }
    }
}

#[async_trait]
impl Clipboard for MockClipboard {
    async fn get_text(&self) -> Result<String, TalkError> {
        Ok(self.content.lock().await.clone())
    }

    async fn set_text(&self, text: &str) -> Result<(), TalkError> {
        *self.content.lock().await = text.to_string();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_clipboard_starts_empty() {
        let clipboard = MockClipboard::new();
        let text = clipboard.get_text().await.unwrap();
        assert_eq!(text, "");
    }

    #[tokio::test]
    async fn test_mock_clipboard_set_and_get() {
        let clipboard = MockClipboard::new();
        clipboard.set_text("hello world").await.unwrap();
        let text = clipboard.get_text().await.unwrap();
        assert_eq!(text, "hello world");
    }

    #[tokio::test]
    async fn test_mock_clipboard_overwrites_previous() {
        let clipboard = MockClipboard::new();
        clipboard.set_text("first").await.unwrap();
        clipboard.set_text("second").await.unwrap();
        let text = clipboard.get_text().await.unwrap();
        assert_eq!(text, "second");
    }

    #[tokio::test]
    async fn test_mock_clipboard_with_initial_content() {
        let clipboard = MockClipboard::with_content("initial");
        let text = clipboard.get_text().await.unwrap();
        assert_eq!(text, "initial");
    }
}
