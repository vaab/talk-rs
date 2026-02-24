//! Clipboard interfaces and implementations.
//!
//! This module provides traits and implementations for clipboard operations
//! using native X11 via `x11rb` (no external tools required).

use crate::error::TalkError;
use crate::x11::clipboard::{x11_clipboard_get, x11_clipboard_set, ClipboardServeHandle};
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

/// X11 clipboard implementation using native `x11rb` calls.
///
/// Each [`set_text`](Clipboard::set_text) call spawns a short-lived
/// background thread that serves `SelectionRequest` events so the
/// paste target can retrieve the data.  The thread is automatically
/// replaced on the next `set_text` and cleaned up on drop.
pub struct X11Clipboard {
    serve_handle: std::sync::Mutex<Option<ClipboardServeHandle>>,
}

impl Default for X11Clipboard {
    fn default() -> Self {
        Self {
            serve_handle: std::sync::Mutex::new(None),
        }
    }
}

impl X11Clipboard {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Clipboard for X11Clipboard {
    async fn get_text(&self) -> Result<String, TalkError> {
        let result = tokio::task::spawn_blocking(x11_clipboard_get)
            .await
            .map_err(|e| TalkError::Clipboard(format!("clipboard task panicked: {e}")))?;

        // Empty/missing clipboard is not an error — return empty string.
        Ok(result.unwrap_or_default())
    }

    async fn set_text(&self, text: &str) -> Result<(), TalkError> {
        // Claim ownership FIRST.  The X server sends SelectionClear to
        // the previous owner, letting its serve thread finish any
        // pending request before exiting — no aggressive kill needed.
        let owned = text.to_string();
        let handle = tokio::task::spawn_blocking(move || x11_clipboard_set(&owned))
            .await
            .map_err(|e| TalkError::Clipboard(format!("clipboard task panicked: {e}")))?
            .ok_or_else(|| {
                TalkError::Clipboard("failed to claim clipboard ownership".to_string())
            })?;

        let mut guard = self
            .serve_handle
            .lock()
            .map_err(|e| TalkError::Clipboard(format!("clipboard lock poisoned: {e}")))?;
        // Old handle dropped here — its thread already received
        // SelectionClear from the new owner and should exit fast.
        *guard = Some(handle);

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
