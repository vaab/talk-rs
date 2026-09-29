//! Clipboard interfaces and implementations.
//!
//! This module provides traits and implementations for clipboard operations
//! using native X11 via `x11rb` (no external tools required).

use crate::error::TalkError;
#[cfg(feature = "ui")]
use crate::x11::clipboard::{
    x11_clipboard_get, x11_clipboard_set, x11_clipboard_set_snapshot, x11_clipboard_snapshot,
    ClipboardServeHandle,
};
use async_trait::async_trait;
#[cfg(feature = "ui")]
use std::io::Read;
#[cfg(feature = "ui")]
use std::os::unix::{net::UnixStream, process::CommandExt};
#[cfg(feature = "ui")]
use std::process::{Command, Stdio};
#[cfg(feature = "ui")]
use std::time::Duration;

/// Trait for clipboard operations.
///
/// Implementations handle text operations and selection snapshots.
/// All implementations must be `Send + Sync` for use in async contexts.
#[async_trait]
pub trait Clipboard: Send + Sync {
    /// Capture every supported content target, or None if there is no owner.
    async fn snapshot(&self) -> Result<Option<ClipboardSnapshot>, TalkError>;

    /// Restore a captured selection; None leaves the current selection alone.
    async fn restore_snapshot(&self, saved: Option<ClipboardSnapshot>) -> Result<(), TalkError>;

    /// Get the current clipboard text content.
    async fn get_text(&self) -> Result<String, TalkError>;

    /// Set the clipboard text content.
    async fn set_text(&self, text: &str) -> Result<(), TalkError>;
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ClipboardTarget {
    pub name: String,
    pub property_type: String,
    pub format: u8,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ClipboardSnapshot {
    pub targets: Vec<ClipboardTarget>,
}

impl ClipboardSnapshot {
    fn text(text: &str) -> Self {
        Self {
            targets: vec![ClipboardTarget {
                name: "UTF8_STRING".into(),
                property_type: "UTF8_STRING".into(),
                format: 8,
                bytes: text.as_bytes().to_vec(),
            }],
        }
    }
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
    /// `SERVED_POLL_INTERVAL_MS` milliseconds.  Returns as soon as
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
    /// This is the per-target counterpart of [`Self::last_served_count`]
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

/// The hidden child reads all bytes before claiming the selection and
/// acknowledges ownership on stdout; it then serves until SelectionClear.
#[cfg(feature = "ui")]
pub fn run_holder() -> Result<(), TalkError> {
    let snapshot: ClipboardSnapshot = serde_json::from_reader(std::io::stdin())
        .map_err(|e| TalkError::Clipboard(format!("holder snapshot: {e}")))?;
    crate::x11::clipboard::x11_clipboard_hold_snapshot(&snapshot).map_err(TalkError::Clipboard)
}

/// Restore readable content targets, without claiming an empty selection
/// when the original owner was absent or no target could be fetched.
#[cfg(feature = "ui")]
pub async fn restore_saved(clipboard: &X11Clipboard, saved: Option<ClipboardSnapshot>) {
    let Some(snapshot) = saved.filter(|s| !s.targets.is_empty()) else {
        log::debug!("clipboard restore skipped: no original owner or readable targets");
        return;
    };
    log::trace!(
        "clipboard: restoring {} original targets via detached holder",
        snapshot.targets.len()
    );
    if let Err(error) = tokio::task::spawn_blocking({
        let snapshot = snapshot.clone();
        move || spawn_holder(&snapshot)
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r)
    {
        log::warn!("detached clipboard restore failed ({error}); using in-process fallback");
        if let Err(error) = clipboard.restore_snapshot(Some(snapshot)).await {
            log::warn!("in-process clipboard restore failed: {error}");
        }
    }
}

#[cfg(feature = "ui")]
fn spawn_holder(snapshot: &ClipboardSnapshot) -> Result<(), String> {
    let (mut parent, child_io) = UnixStream::pair().map_err(|e| e.to_string())?;
    parent
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|e| e.to_string())?;
    parent
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(|e| e.to_string())?;
    let child_out = child_io.try_clone().map_err(|e| e.to_string())?;
    let binary = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut command = Command::new(binary);
    command
        .arg("clipboard-hold")
        // Long-lived: do not pin the caller's working directory.
        .current_dir("/")
        .stdin(Stdio::from(std::os::fd::OwnedFd::from(child_io)))
        .stdout(Stdio::from(std::os::fd::OwnedFd::from(child_out)))
        .stderr(Stdio::null());
    // The child has its own session and no terminal; a dedicated waiter
    // reaps it even when the parent stays alive after the paste.
    unsafe {
        command.pre_exec(|| {
            nix::unistd::setsid().map_err(std::io::Error::other)?;
            Ok(())
        });
    }
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let result = (|| {
        serde_json::to_writer(&mut parent, snapshot).map_err(|e| e.to_string())?;
        parent
            .shutdown(std::net::Shutdown::Write)
            .map_err(|e| e.to_string())?;
        let mut ready = [0];
        parent.read_exact(&mut ready).map_err(|e| e.to_string())?;
        if ready != [1] {
            return Err("clipboard holder sent invalid acknowledgement".to_string());
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = child.kill();
    }
    std::thread::spawn(move || {
        if let Err(error) = child.wait() {
            log::warn!("could not reap clipboard holder: {error}");
        }
    });
    result
}

#[cfg(feature = "ui")]
#[async_trait]
impl Clipboard for X11Clipboard {
    async fn snapshot(&self) -> Result<Option<ClipboardSnapshot>, TalkError> {
        tokio::task::spawn_blocking(x11_clipboard_snapshot)
            .await
            .map_err(|e| TalkError::Clipboard(format!("clipboard task panicked: {e}")))?
            .map_err(TalkError::Clipboard)
    }

    async fn restore_snapshot(&self, saved: Option<ClipboardSnapshot>) -> Result<(), TalkError> {
        let Some(snapshot) = saved.filter(|s| !s.targets.is_empty()) else {
            return Ok(());
        };
        let handle = tokio::task::spawn_blocking(move || x11_clipboard_set_snapshot(snapshot))
            .await
            .map_err(|e| TalkError::Clipboard(format!("clipboard task panicked: {e}")))?
            .ok_or_else(|| TalkError::Clipboard("failed to claim clipboard ownership".into()))?;
        *self
            .serve_handle
            .lock()
            .map_err(|e| TalkError::Clipboard(format!("clipboard lock poisoned: {e}")))? =
            Some(handle);
        Ok(())
    }

    async fn get_text(&self) -> Result<String, TalkError> {
        let result = tokio::task::spawn_blocking(x11_clipboard_get)
            .await
            .map_err(|e| TalkError::Clipboard(format!("clipboard task panicked: {e}")))?;

        let text = result.ok_or_else(|| {
            TalkError::Clipboard("could not read UTF8_STRING from clipboard".to_string())
        })?;
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
    content: std::sync::Arc<tokio::sync::Mutex<Option<ClipboardSnapshot>>>,
}

impl Default for MockClipboard {
    fn default() -> Self {
        Self {
            content: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
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
            content: std::sync::Arc::new(tokio::sync::Mutex::new(Some(ClipboardSnapshot::text(
                &text.into(),
            )))),
        }
    }

    pub fn with_snapshot(snapshot: ClipboardSnapshot) -> Self {
        Self {
            content: std::sync::Arc::new(tokio::sync::Mutex::new(Some(snapshot))),
        }
    }
}

#[async_trait]
impl Clipboard for MockClipboard {
    async fn snapshot(&self) -> Result<Option<ClipboardSnapshot>, TalkError> {
        Ok(self.content.lock().await.clone())
    }

    async fn restore_snapshot(&self, saved: Option<ClipboardSnapshot>) -> Result<(), TalkError> {
        if let Some(saved) = saved {
            *self.content.lock().await = Some(saved);
        }
        Ok(())
    }

    async fn get_text(&self) -> Result<String, TalkError> {
        Ok(self
            .content
            .lock()
            .await
            .as_ref()
            .and_then(|s| {
                s.targets
                    .iter()
                    .find(|t| t.name == "UTF8_STRING")
                    .and_then(|t| String::from_utf8(t.bytes.clone()).ok())
            })
            .unwrap_or_default())
    }

    async fn set_text(&self, text: &str) -> Result<(), TalkError> {
        *self.content.lock().await = Some(ClipboardSnapshot::text(text));
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

    #[tokio::test]
    async fn mock_restores_image_and_multiple_targets() {
        let original = ClipboardSnapshot {
            targets: vec![
                ClipboardTarget {
                    name: "image/png".into(),
                    property_type: "image/png".into(),
                    format: 8,
                    bytes: vec![0, 137, 80, 78, 71],
                },
                ClipboardTarget {
                    name: "text/html".into(),
                    property_type: "text/html".into(),
                    format: 8,
                    bytes: b"<b>hi</b>".to_vec(),
                },
            ],
        };
        let clipboard = MockClipboard::with_snapshot(original.clone());
        let saved = clipboard.snapshot().await.unwrap();
        clipboard.set_text("DICTATED").await.unwrap();
        clipboard.restore_snapshot(saved).await.unwrap();
        assert_eq!(clipboard.snapshot().await.unwrap(), Some(original));
    }

    #[tokio::test]
    async fn mock_restores_text_unchanged() {
        let clipboard = MockClipboard::with_content("original");
        let saved = clipboard.snapshot().await.unwrap();
        clipboard.set_text("DICTATED").await.unwrap();
        clipboard.restore_snapshot(saved).await.unwrap();
        assert_eq!(clipboard.get_text().await.unwrap(), "original");
    }

    #[tokio::test]
    async fn mock_no_original_owner_leaves_pasted_text() {
        let clipboard = MockClipboard::new();
        let saved = clipboard.snapshot().await.unwrap();
        clipboard.set_text("DICTATED").await.unwrap();
        clipboard.restore_snapshot(saved).await.unwrap();
        assert_eq!(clipboard.get_text().await.unwrap(), "DICTATED");
    }
}
