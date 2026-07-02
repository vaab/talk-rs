//! Clipboard interfaces and implementations.
//!
//! This module provides traits and implementations for clipboard operations
//! using native X11 via `x11rb` (no external tools required).

use crate::error::TalkError;
#[cfg(feature = "ui")]
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
#[cfg(feature = "ui")]
pub struct X11Clipboard {
    serve_handle: std::sync::Mutex<Option<ClipboardServeHandle>>,
}

#[cfg(feature = "ui")]
impl Default for X11Clipboard {
    fn default() -> Self {
        Self {
            serve_handle: std::sync::Mutex::new(None),
        }
    }
}

#[cfg(feature = "ui")]
impl X11Clipboard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of times a paste target has fetched the content set by
    /// the most recent [`set_text`](Clipboard::set_text) call.
    ///
    /// Returns `0` when no content is currently being served, or when
    /// the serve handle has been dropped.  Used by the paste path for
    /// `-vvv` diagnostics: a count of `0` after a paste keystroke
    /// means the target never actually pulled the offered text.
    pub fn last_served_count(&self) -> u32 {
        self.serve_handle
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|h| h.served_count()))
            .unwrap_or(0)
    }

    /// Block until the served-count of the currently-offered content
    /// exceeds `baseline`, or until `timeout` elapses — whichever
    /// comes first.  Returns the final served count.
    ///
    /// Used by the paste pipeline to wait for the target window to
    /// actually fetch the clipboard content before overwriting it
    /// (next chunk or final restore).  Without this wait a fixed
    /// sleep can race a slow paste target into pulling the WRONG
    /// clipboard generation — leaking restored content into the
    /// document and dropping the last chunk.
    ///
    /// Polls [`last_served_count`](Self::last_served_count) every
    /// [`SERVED_POLL_INTERVAL_MS`] milliseconds.  Returns as soon as
    /// the count crosses `baseline`; on timeout returns the
    /// last-observed (possibly unchanged) count so the caller can
    /// decide whether to emit a warning.
    pub async fn wait_until_served(&self, baseline: u32, timeout: std::time::Duration) -> u32 {
        let deadline = std::time::Instant::now() + timeout;
        let poll = std::time::Duration::from_millis(SERVED_POLL_INTERVAL_MS);
        loop {
            let count = self.last_served_count();
            if count > baseline {
                return count;
            }
            if std::time::Instant::now() >= deadline {
                return count;
            }
            tokio::time::sleep(poll).await;
        }
    }

    /// Number of `UTF8_STRING` fetches answered for the given target
    /// client-base on the CURRENT serve handle.  Returns `0` when no
    /// handle is held (i.e. no clipboard content is currently
    /// served), or when the target client-base has not yet fetched.
    ///
    /// This is the per-target counterpart of [`last_served_count`]
    /// (the legacy total).  The deterministic paste gate uses it to
    /// confirm "the actual target consumed this chunk" rather than
    /// the unreliable "anyone fetched at least once" signal that
    /// counted clipboard managers and dropped real chunks.
    pub fn target_fetch_count(&self, target_client_base: u32) -> u32 {
        self.serve_handle
            .lock()
            .ok()
            .and_then(|guard| {
                guard
                    .as_ref()
                    .map(|h| h.fetches_by_client(target_client_base))
            })
            .unwrap_or(0)
    }

    /// X11 `resource_id_mask` snapshotted from the current serve
    /// handle's connection, or `None` when no handle is held.  The
    /// mask is server-wide (any connection returns the same value),
    /// so callers who only need to derive a client-base typically
    /// reach for [`crate::x11::x11_client_base`] instead — this
    /// accessor is for the rare case of needing the mask the serve
    /// thread is actively using.
    pub fn resource_id_mask(&self) -> Option<u32> {
        self.serve_handle
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|h| h.resource_id_mask()))
    }

    /// Block until the target client-base has fetched at least
    /// `expected` times for the currently-served content, or until
    /// `timeout` elapses.  Returns the final per-target count.
    ///
    /// On timeout returns the last-observed count (possibly below
    /// `expected`); the caller decides whether to abort the paste
    /// loudly or fall back.  The deterministic paste gate calls this
    /// once per chunk: chunk 1 with `expected=1` (to wait for the
    /// initial fetch before measuring quiescence), subsequent chunks
    /// with the learned target-fetch count.
    pub async fn wait_until_target_fetched(
        &self,
        target_client_base: u32,
        expected: u32,
        timeout: std::time::Duration,
    ) -> u32 {
        let deadline = std::time::Instant::now() + timeout;
        let poll = std::time::Duration::from_millis(SERVED_POLL_INTERVAL_MS);
        loop {
            let count = self.target_fetch_count(target_client_base);
            if count >= expected {
                return count;
            }
            if std::time::Instant::now() >= deadline {
                return count;
            }
            tokio::time::sleep(poll).await;
        }
    }
}

/// Poll interval (ms) used by [`X11Clipboard::wait_until_served`].
///
/// Five milliseconds keeps the wait responsive without saturating the
/// async runtime: a single `SelectionRequest` round-trip is typically
/// served within a few milliseconds, so most waits resolve on the
/// first or second poll.
#[cfg(feature = "ui")]
const SERVED_POLL_INTERVAL_MS: u64 = 5;

#[cfg(feature = "ui")]
#[async_trait]
impl Clipboard for X11Clipboard {
    async fn get_text(&self) -> Result<String, TalkError> {
        let result = tokio::task::spawn_blocking(x11_clipboard_get)
            .await
            .map_err(|e| TalkError::Clipboard(format!("clipboard task panicked: {e}")))?;

        // Empty/missing clipboard is not an error — return empty string.
        let text = result.unwrap_or_default();
        log::trace!("clipboard get_text -> {}", crate::paste::log_preview(&text),);
        Ok(text)
    }

    async fn set_text(&self, text: &str) -> Result<(), TalkError> {
        log::trace!("clipboard set_text <- {}", crate::paste::log_preview(text),);
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
