//! Clipboard paste utilities, window-focus helpers, and the
//! composable paste-node tree.
//!
//! Two layers live here:
//!
//! * **Primitives** ([`simulate_paste`], [`simulate_backspace`],
//!   [`ensure_focus`], [`split_into_char_chunks`], [`paste_keysyms`],
//!   [`log_preview`], [`PasteTiming`], [`PASTE_CHUNK_CHARS`]) — the
//!   low-level building blocks shared by every variant of the paste
//!   pipeline.  Unchanged from the pre-tree refactor.
//! * **Node tree** ([`PasteNode`], [`PasteCtx`], [`PasteNodeConfig`],
//!   [`build_root_from_config`], [`default_root`], [`paste_with_root`])
//!   — composable nodes that today reproduce the legacy single-path
//!   pipeline `chunk(150) → clipboard(ctrl-shift-v, 200, 400)` and
//!   tomorrow can be swapped or extended without changing call sites.

pub mod node;
pub mod nodes;

pub use node::{PasteCtx, PasteNode, PasteNodeConfig, WmClassPattern};

use crate::clipboard::{Clipboard, X11Clipboard};
use crate::config::PasteShortcut;
use crate::error::TalkError;

/// Number of leading characters shown in a paste-diagnostic preview.
const PASTE_PREVIEW_CHARS: usize = 60;

/// Render a short, single-line preview of `text` for paste-diagnostic
/// trace logs: the character count plus the first
/// [`PASTE_PREVIEW_CHARS`] characters with newlines/tabs escaped so a
/// multi-line paste stays on one log line.
///
/// This DOES include clipboard content (potentially sensitive), which
/// is why every call site is gated behind `-vvv` trace logging.
///
/// Unicode-safe: truncation happens on `char` boundaries, never byte
/// offsets, so multibyte text cannot panic.
pub fn log_preview(text: &str) -> String {
    let char_count = text.chars().count();
    let escaped: String = text
        .chars()
        .take(PASTE_PREVIEW_CHARS)
        .map(|c| match c {
            '\n' => '␊',
            '\r' => '␍',
            '\t' => '␉',
            other => other,
        })
        .collect();
    let ellipsis = if char_count > PASTE_PREVIEW_CHARS {
        "…"
    } else {
        ""
    };
    format!("{char_count} chars: \"{escaped}{ellipsis}\"")
}

/// Maximum number of attempts to focus the target window.
const FOCUS_MAX_RETRIES: u32 = 5;

/// Initial delay between focus retry attempts (doubles each retry).
const FOCUS_INITIAL_DELAY_MS: u64 = 50;

/// Poll interval (ms) used by the pre-restore settle loop in
/// [`paste_text_to_target`].  Five milliseconds matches
/// [`crate::clipboard::X11Clipboard::wait_until_served`] and keeps
/// the wait responsive without saturating the async runtime.
const SETTLE_POLL_INTERVAL_MS: u64 = 5;

/// Timing knobs for the paste pipeline.
///
/// Threaded into [`paste_text_to_target`] so the per-chunk fetch
/// timeout and the pre-restore settle window can be tuned via
/// `paste.chunk_fetch_timeout_ms` / `paste.restore_settle_ms` in
/// the config without growing the call-site signature unboundedly.
///
/// `Default` matches the config defaults (200 / 400) so call sites
/// without a loaded config — and tests — can fall back transparently.
#[derive(Debug, Clone, Copy)]
pub struct PasteTiming {
    /// See `paste.restore_settle_ms`.
    pub restore_settle_ms: u64,
    /// See `paste.chunk_fetch_timeout_ms`.
    pub chunk_fetch_timeout_ms: u64,
}

impl Default for PasteTiming {
    fn default() -> Self {
        Self {
            restore_settle_ms: 200,
            chunk_fetch_timeout_ms: 400,
        }
    }
}

/// Maximum number of characters per clipboard paste operation.
///
/// When the text to paste exceeds this limit it is split into
/// consecutive chunks, each pasted via a separate Ctrl+Shift+V
/// keystroke.  Splits happen on word boundaries so words are never
/// cut in half.  Keeping chunks under 150 characters avoids
/// triggering paste-summary behaviour in terminal applications
/// that collapse large pastes into an opaque block.
pub const PASTE_CHUNK_CHARS: usize = 150;

/// Attempt to focus the target window and verify the active window
/// matches.  Retries with exponential backoff to give the window
/// manager time to settle after destroying a transient window (e.g.
/// the GTK picker).
///
/// Returns `Ok(())` when the target window is confirmed active, or
/// `Err` if focus could not be established after all retries.
pub async fn ensure_focus(window_id: &str) -> Result<(), TalkError> {
    let mut delay_ms = FOCUS_INITIAL_DELAY_MS;

    for attempt in 1..=FOCUS_MAX_RETRIES {
        focus_window(window_id).await;
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;

        if let Some(active) = get_active_window().await {
            if active == window_id {
                log::debug!("target window {} focused (attempt {})", window_id, attempt);
                return Ok(());
            }
            log::debug!(
                "focus attempt {}/{}: expected {}, got {}",
                attempt,
                FOCUS_MAX_RETRIES,
                window_id,
                active,
            );
        } else {
            log::debug!(
                "focus attempt {}/{}: could not determine active window",
                attempt,
                FOCUS_MAX_RETRIES,
            );
        }

        delay_ms *= 2;
    }

    Err(TalkError::Clipboard(format!(
        "could not focus target window {} after {} attempts \
         — aborting to avoid sending keys to the wrong window",
        window_id, FOCUS_MAX_RETRIES,
    )))
}

/// Split `text` into chunks of at most `max_chars` characters each,
/// breaking on word boundaries so words are never cut in half.
///
/// Every chunk after the first is prefixed with a single space so that
/// concatenating all chunks reproduces the original word sequence.
/// If the text is empty (or whitespace-only) a single element containing
/// the original string is returned so that the caller always has at
/// least one chunk to paste.  A single word longer than `max_chars` is
/// emitted as-is (never split mid-word).
pub fn split_into_char_chunks(text: &str, max_chars: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for word in &words {
        let candidate_len = if current.is_empty() {
            word.len()
        } else {
            current.len() + 1 + word.len() // +1 for the space
        };

        if !current.is_empty() && candidate_len > max_chars {
            chunks.push(current);
            current = format!(" {word}");
        } else if current.is_empty() {
            current = (*word).to_string();
        } else {
            current.push(' ');
            current.push_str(word);
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

/// Materialise the default paste tree: `chunk(150) → clipboard(ctrl-shift-v,
/// 200, 400)` — the exact tree that reproduces legacy behaviour when no
/// `paste:` section is present in the YAML config.
///
/// When `no_chunk_paste` is `true`, the `chunk` wrapper is dropped and
/// the tree collapses to a single `clipboard` leaf — matching the
/// legacy `--no-chunk-paste` flag semantics.
pub fn default_root(no_chunk_paste: bool) -> Box<dyn PasteNode> {
    let mut tree = PasteNodeConfig::Chunk {
        chunk_chars: PASTE_CHUNK_CHARS,
        child: Box::new(PasteNodeConfig::Clipboard {
            shortcut: PasteShortcut::CtrlShiftV,
            restore_settle_ms: PasteTiming::default().restore_settle_ms,
            chunk_fetch_timeout_ms: PasteTiming::default().chunk_fetch_timeout_ms,
        }),
    };
    if no_chunk_paste {
        tree = tree.strip_chunks();
    }
    tree.build()
}

/// Build the runtime root node from a [`PasteNodeConfig`].  Applies
/// the `no_chunk_paste` flag by stripping any `chunk` wrappers from
/// the configured tree.
pub fn build_root_from_config(cfg: &PasteNodeConfig, no_chunk_paste: bool) -> Box<dyn PasteNode> {
    if no_chunk_paste {
        cfg.clone().strip_chunks().build()
    } else {
        cfg.build()
    }
}

/// Extract the settle-timing knobs from a configured tree.  The
/// settle loop lives in the wrapper ([`paste_with_root`] /
/// [`RealtimeClipboardGuard`]) — see the deviation note on
/// [`PasteCtx`] for the rationale.
pub fn timing_from_root(cfg: &PasteNodeConfig) -> PasteTiming {
    node::timing_from_tree(cfg)
}

/// Paste `text` through the supplied root node, wrapping with
/// save-clipboard / settle / restore-clipboard.  Direct successor of
/// the legacy `paste_text_to_target`.
///
/// Behaviour is identical to today's pipeline for the default tree:
/// focus → optional backspace → save clipboard → root.paste(text) →
/// settle → restore.  The `timing` argument is used ONLY for the
/// settle loop here; per-clipboard-call timing knobs live on each
/// `Clipboard` node inside the tree.
#[allow(clippy::too_many_arguments)]
pub async fn paste_with_root(
    root: &dyn PasteNode,
    target_window: Option<&String>,
    text: &str,
    delete_chars_before_paste: usize,
    t_stop: Option<std::time::Instant>,
    sink: &dyn crate::telemetry::TelemetrySink,
    timing: PasteTiming,
) -> Result<(), TalkError> {
    let clipboard = X11Clipboard::new();
    let total_chars = text.len() as u64;

    log::trace!(
        "paste: BEGIN delete_before={} target_window={:?} text={}",
        delete_chars_before_paste,
        target_window,
        log_preview(text),
    );

    if let Some(wid) = target_window {
        log::debug!("refocusing target window: {}", wid);
        ensure_focus(wid).await?;
        if let Some(active) = get_active_window().await {
            log::trace!("paste: active window after focus = {}", active);
        }
    }

    if delete_chars_before_paste > 0 {
        log::info!("deleting {} chars before paste", delete_chars_before_paste);
        simulate_backspace(delete_chars_before_paste).await?;
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }

    let saved_clipboard = clipboard.get_text().await.ok();
    log::trace!(
        "paste: saved original clipboard = {}",
        saved_clipboard
            .as_deref()
            .map(log_preview)
            .unwrap_or_else(|| "<none>".to_string()),
    );

    // Legacy "timing: stop +Nms first_paste" log — emitted once,
    // immediately before the first keystroke leaves this process.
    if let Some(t) = t_stop {
        log::info!("timing: stop +{}ms first_paste", t.elapsed().as_millis());
    }

    let target_window_str: Option<&str> = target_window.map(|s| s.as_str());
    let ctx = PasteCtx {
        target_window: target_window_str,
        delete_chars_before_paste,
        t_stop,
        sink,
        clipboard: &clipboard,
    };

    root.paste(text, &ctx).await?;

    // Settle before restoring: same race-protection as the legacy
    // pipeline — see `settle_before_restore` docs.
    settle_before_restore(&clipboard, timing).await;

    if let Some(saved) = saved_clipboard {
        log::trace!(
            "paste: restoring original clipboard = {}",
            log_preview(&saved)
        );
        let _ = clipboard.set_text(&saved).await;
    }

    log::trace!("paste: END (total_chars={})", total_chars);
    Ok(())
}

/// Per-segment paste guard for the realtime path.
///
/// Today the realtime per-segment loop in `dictate/mod.rs` saved the
/// clipboard before the first segment, pasted each segment via an
/// open-coded sequence (set_text → simulate_paste → wait_until_served),
/// then settled + restored at the end.  This guard re-implements
/// EXACTLY that sequence by routing each segment through the SAME
/// root node tree used by the one-shot path — the `chunk` wrapper is
/// stripped (segments are pasted whole; legacy parity) and the
/// shared `X11Clipboard` outlives the loop so `settle_before_restore`
/// observes the right serve counter on `finish`.
pub struct RealtimeClipboardGuard {
    clipboard: X11Clipboard,
    saved: Option<String>,
    timing: PasteTiming,
}

impl RealtimeClipboardGuard {
    /// Save the current clipboard.  Does not pre-focus the target
    /// window — call sites do that separately to preserve today's
    /// ordering.
    pub async fn begin(timing: PasteTiming) -> Self {
        let clipboard = X11Clipboard::new();
        let saved = clipboard.get_text().await.ok();
        log::trace!(
            "paste(realtime): saved original clipboard = {}",
            saved
                .as_deref()
                .map(log_preview)
                .unwrap_or_else(|| "<none>".to_string()),
        );
        Self {
            clipboard,
            saved,
            timing,
        }
    }

    /// Paste one segment through `root`.  Each call is independent —
    /// no chunking is applied (the realtime path always pasted
    /// whole segments).
    pub async fn paste_segment(
        &self,
        root: &dyn PasteNode,
        segment: &str,
        sink: &dyn crate::telemetry::TelemetrySink,
    ) -> Result<(), TalkError> {
        let ctx = PasteCtx {
            target_window: None,
            delete_chars_before_paste: 0,
            t_stop: None,
            sink,
            clipboard: &self.clipboard,
        };
        root.paste(segment, &ctx).await
    }

    /// Settle + restore.  Idempotent; safe to call multiple times.
    pub async fn finish(self) {
        settle_before_restore(&self.clipboard, self.timing).await;
        if let Some(saved) = self.saved {
            log::debug!("restoring original clipboard");
            log::trace!(
                "paste(realtime): restoring original clipboard = {}",
                log_preview(&saved),
            );
            let _ = self.clipboard.set_text(&saved).await;
        }
    }
}

/// Block until the last-chunk served count has been STABLE for
/// `timing.restore_settle_ms`, capped at `timing.chunk_fetch_timeout_ms`
/// total wall-clock.
///
/// "Stable" means: the count observed at time `t` equals the count
/// observed at `t + restore_settle_ms`.  Polls every
/// [`SETTLE_POLL_INTERVAL_MS`].  This is the critical guard against
/// the user's restored clipboard leaking into a slow paste target
/// AFTER the last chunk — the symptom we set out to fix.
async fn settle_before_restore(clipboard: &X11Clipboard, timing: PasteTiming) {
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_millis(timing.chunk_fetch_timeout_ms);
    let settle = std::time::Duration::from_millis(timing.restore_settle_ms);
    let poll = std::time::Duration::from_millis(SETTLE_POLL_INTERVAL_MS);

    let mut last = clipboard.last_served_count();
    let mut stable_since = std::time::Instant::now();
    loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            log::warn!(
                "paste: pre-restore settle timed out after {} ms (served_count={}, \
                 last change still within {} ms window) — proceeding anyway, \
                 slow paste target may still re-fetch and corrupt the paste",
                timing.chunk_fetch_timeout_ms,
                last,
                timing.restore_settle_ms,
            );
            return;
        }
        if now.duration_since(stable_since) >= settle {
            log::trace!(
                "paste: pre-restore settled (served_count={}, stable for {} ms)",
                last,
                timing.restore_settle_ms,
            );
            return;
        }
        tokio::time::sleep(poll).await;
        let current = clipboard.last_served_count();
        if current != last {
            last = current;
            stable_since = std::time::Instant::now();
        }
    }
}

/// Get the currently focused window ID via `_NET_ACTIVE_WINDOW`.
pub async fn get_active_window() -> Option<String> {
    // The X11 call is blocking but fast; run on a blocking thread
    // so we don't stall the async runtime.
    tokio::task::spawn_blocking(|| crate::x11::x11_get_active_window().map(|wid| wid.to_string()))
        .await
        .ok()?
}

/// Focus a window by ID via `_NET_ACTIVE_WINDOW` ClientMessage.
pub async fn focus_window(window_id: &str) -> bool {
    let wid: u32 = match window_id.parse() {
        Ok(v) => v,
        Err(_) => return false,
    };

    tokio::task::spawn_blocking(move || crate::x11::x11_activate_window(wid))
        .await
        .unwrap_or(false)
}

/// Resolve a [`PasteShortcut`] into the X11 keysyms to send.
///
/// Pure function — enables unit testing without an X11 connection.
pub fn paste_keysyms(shortcut: &PasteShortcut) -> Vec<u32> {
    const CONTROL_L: u32 = 0xffe3;
    const SHIFT_L: u32 = 0xffe1;
    const KEY_V: u32 = 0x0076;

    match shortcut {
        PasteShortcut::CtrlShiftV => vec![CONTROL_L, SHIFT_L, KEY_V],
        PasteShortcut::CtrlV => vec![CONTROL_L, KEY_V],
    }
}

/// Simulate a paste keystroke via the XTest extension.
///
/// The exact key combination depends on `shortcut`:
/// - `PasteShortcut::CtrlShiftV` → Ctrl+Shift+V
/// - `PasteShortcut::CtrlV` → Ctrl+V
pub async fn simulate_paste(shortcut: PasteShortcut) -> Result<(), TalkError> {
    let keysyms = paste_keysyms(&shortcut);

    let ok = tokio::task::spawn_blocking(move || crate::x11::x11_send_key_combo(&keysyms))
        .await
        .unwrap_or(false);

    if !ok {
        return Err(TalkError::Clipboard(
            "XTest key simulation failed".to_string(),
        ));
    }
    Ok(())
}

/// Simulate deleting the previous text by sending repeated BackSpace
/// via the XTest extension.
pub async fn simulate_backspace(count: usize) -> Result<(), TalkError> {
    if count == 0 {
        return Ok(());
    }

    // X11 keysym for BackSpace.
    const BACKSPACE: u32 = 0xff08;

    let ok = tokio::task::spawn_blocking(move || crate::x11::x11_send_key_repeat(BACKSPACE, count))
        .await
        .unwrap_or(false);

    if !ok {
        return Err(TalkError::Clipboard(
            "XTest backspace simulation failed".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_short_text_fits_in_one() {
        let chunks = split_into_char_chunks("hello world", 150);
        assert_eq!(chunks, vec!["hello world"]);
    }

    #[test]
    fn test_chunk_exactly_at_limit() {
        // 20 chars exactly, limit 20
        let text = "one two three four f";
        assert_eq!(text.len(), 20);
        let chunks = split_into_char_chunks(text, 20);
        assert_eq!(chunks, vec![text]);
    }

    #[test]
    fn test_chunk_splits_on_word_boundary() {
        // "hello world" = 11 chars, limit 8 → split before "world"
        let chunks = split_into_char_chunks("hello world", 8);
        assert_eq!(chunks, vec!["hello", " world"]);
    }

    #[test]
    fn test_chunk_long_word_exceeds_limit() {
        // A single word longer than the limit is emitted as-is
        let chunks = split_into_char_chunks("supercalifragilistic", 5);
        assert_eq!(chunks, vec!["supercalifragilistic"]);
    }

    #[test]
    fn test_chunk_multiple_chunks() {
        // limit 10: "aaa bbb" (7) fits, "aaa bbb ccc" (11) doesn't
        let text = "aaa bbb ccc ddd eee fff";
        let chunks = split_into_char_chunks(text, 10);
        assert_eq!(chunks, vec!["aaa bbb", " ccc ddd", " eee fff"]);
    }

    #[test]
    fn test_chunk_concatenation_reproduces_original() {
        let text = "The quick brown fox jumps over the lazy dog and then some more words follow after that";
        let chunks = split_into_char_chunks(text, 30);
        let reassembled: String = chunks.concat();
        assert_eq!(reassembled, text);
    }

    #[test]
    fn test_chunk_empty_string() {
        let chunks = split_into_char_chunks("", 150);
        assert_eq!(chunks, vec![""]);
    }

    #[test]
    fn test_chunk_whitespace_only() {
        let chunks = split_into_char_chunks("   ", 150);
        assert_eq!(chunks, vec!["   "]);
    }

    #[test]
    fn test_chunk_single_word() {
        let chunks = split_into_char_chunks("hello", 150);
        assert_eq!(chunks, vec!["hello"]);
    }

    #[test]
    fn test_paste_keysyms_ctrl_shift_v() {
        let keysyms = paste_keysyms(&PasteShortcut::CtrlShiftV);
        assert_eq!(keysyms, vec![0xffe3, 0xffe1, 0x0076]);
    }

    #[test]
    fn test_paste_keysyms_ctrl_v() {
        let keysyms = paste_keysyms(&PasteShortcut::CtrlV);
        assert_eq!(keysyms, vec![0xffe3, 0x0076]);
    }

    #[test]
    fn test_log_preview_short_text_not_truncated() {
        assert_eq!(log_preview("hello"), "5 chars: \"hello\"");
    }

    #[test]
    fn test_log_preview_empty() {
        assert_eq!(log_preview(""), "0 chars: \"\"");
    }

    #[test]
    fn test_log_preview_escapes_newlines_and_tabs() {
        // Newline, carriage return, and tab are replaced with visible
        // control pictures so a multi-line paste stays on one log line.
        assert_eq!(log_preview("a\nb\tc\rd"), "7 chars: \"a␊b␉c␍d\"");
    }

    #[test]
    fn test_log_preview_truncates_with_ellipsis() {
        let text = "x".repeat(PASTE_PREVIEW_CHARS + 10);
        let preview = log_preview(&text);
        let expected_body = "x".repeat(PASTE_PREVIEW_CHARS);
        assert_eq!(
            preview,
            format!("{} chars: \"{}…\"", PASTE_PREVIEW_CHARS + 10, expected_body),
        );
    }

    #[test]
    fn test_log_preview_boundary_exactly_preview_chars_no_ellipsis() {
        let text = "y".repeat(PASTE_PREVIEW_CHARS);
        let preview = log_preview(&text);
        assert!(!preview.contains('…'));
        assert_eq!(
            preview,
            format!("{} chars: \"{}\"", PASTE_PREVIEW_CHARS, text),
        );
    }

    #[test]
    fn test_log_preview_multibyte_char_boundary_safe() {
        // Each emoji is one `char` but 4 bytes; truncation must happen
        // on char boundaries so this never panics and counts chars,
        // not bytes.
        let text = "😀".repeat(PASTE_PREVIEW_CHARS + 5);
        let preview = log_preview(&text);
        assert!(preview.starts_with(&format!("{} chars: ", PASTE_PREVIEW_CHARS + 5)));
        assert!(preview.ends_with("…\""));
        // Exactly PASTE_PREVIEW_CHARS emojis are shown before the ellipsis.
        let shown = "😀".repeat(PASTE_PREVIEW_CHARS);
        assert!(preview.contains(&shown));
    }
}

#[cfg(test)]
mod tree_tests {
    //! Tests for the paste-node tree config + builders.  Covers:
    //! - new tree YAML deserialises
    //! - old flat YAML deserialises into equivalent tree
    //! - missing `paste:` → default tree
    //! - first-match routing in `match-wm-class`
    //! - glob matching for WM_CLASS
    //! - chunk node reproduces `split_into_char_chunks` behaviour

    use super::node::{PasteCtx, PasteNode, PasteNodeConfig};
    use super::nodes::chunk::ChunkNode;
    use super::nodes::glob_match;
    use crate::clipboard::X11Clipboard;
    use crate::config::{Config, PasteConfig, PasteShortcut};
    use crate::telemetry::{NoOpSink, TelemetrySink, TranscriptionEvent};
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::Mutex;

    /// Helper: parse a tiny full Config from inline YAML and return
    /// its `paste` field.
    fn parse_paste(yaml: &str) -> Option<PasteConfig> {
        let cfg: Config = serde_yaml::from_str(yaml).expect("yaml fixture must parse as Config");
        cfg.paste
    }

    #[test]
    fn flat_yaml_deserialises_as_flat_variant() {
        let yaml = r#"
output_dir: /tmp/x
providers: {}
paste:
  chunk_chars: 80
  shortcut: ctrl_v
  restore_settle_ms: 250
  chunk_fetch_timeout_ms: 600
"#;
        let p = parse_paste(yaml).expect("paste section");
        match p {
            PasteConfig::Flat(ref f) => {
                assert_eq!(f.chunk_chars, 80);
                assert_eq!(f.shortcut, PasteShortcut::CtrlV);
                assert_eq!(f.restore_settle_ms, 250);
                assert_eq!(f.chunk_fetch_timeout_ms, 600);
            }
            PasteConfig::Tree(_) => panic!("expected flat variant for legacy YAML"),
        }

        // Building the root tree from flat must collapse to
        // chunk(80) → clipboard(ctrl_v, 250, 600).
        let tree = p.to_tree();
        match tree {
            PasteNodeConfig::Chunk { chunk_chars, child } => {
                assert_eq!(chunk_chars, 80);
                match *child {
                    PasteNodeConfig::Clipboard {
                        shortcut,
                        restore_settle_ms,
                        chunk_fetch_timeout_ms,
                    } => {
                        assert_eq!(shortcut, PasteShortcut::CtrlV);
                        assert_eq!(restore_settle_ms, 250);
                        assert_eq!(chunk_fetch_timeout_ms, 600);
                    }
                    other => panic!("expected Clipboard child, got {:?}", other),
                }
            }
            other => panic!("expected Chunk root, got {:?}", other),
        }
    }

    #[test]
    fn flat_yaml_with_chunk_chars_zero_skips_chunk_wrapper() {
        let yaml = r#"
output_dir: /tmp/x
providers: {}
paste:
  chunk_chars: 0
"#;
        let p = parse_paste(yaml).expect("paste section");
        match p.to_tree() {
            PasteNodeConfig::Clipboard { .. } => {}
            other => panic!("expected Clipboard root for chunk_chars=0, got {:?}", other),
        }
    }

    #[test]
    fn tree_yaml_deserialises_as_tree_variant() {
        let yaml = r#"
output_dir: /tmp/x
providers: {}
paste:
  node: chunk
  chunk_chars: 120
  child:
    node: clipboard
    shortcut: ctrl_shift_v
    restore_settle_ms: 150
    chunk_fetch_timeout_ms: 350
"#;
        let p = parse_paste(yaml).expect("paste section");
        match p {
            PasteConfig::Tree(t) => match t {
                PasteNodeConfig::Chunk { chunk_chars, child } => {
                    assert_eq!(chunk_chars, 120);
                    match *child {
                        PasteNodeConfig::Clipboard {
                            shortcut,
                            restore_settle_ms,
                            chunk_fetch_timeout_ms,
                        } => {
                            assert_eq!(shortcut, PasteShortcut::CtrlShiftV);
                            assert_eq!(restore_settle_ms, 150);
                            assert_eq!(chunk_fetch_timeout_ms, 350);
                        }
                        other => panic!("expected Clipboard child, got {:?}", other),
                    }
                }
                other => panic!("expected Chunk root, got {:?}", other),
            },
            PasteConfig::Flat(_) => panic!("expected tree variant for `node:`-tagged YAML"),
        }
    }

    #[test]
    fn tree_yaml_with_match_wm_class_routing() {
        let yaml = r#"
output_dir: /tmp/x
providers: {}
paste:
  node: match-wm-class
  patterns:
    - match: "firefox.*"
      child:
        node: clipboard
        shortcut: ctrl_v
        restore_settle_ms: 200
        chunk_fetch_timeout_ms: 400
    - match: "*.Emacs"
      child:
        node: xtest-type
  default:
    node: clipboard
    shortcut: ctrl_shift_v
    restore_settle_ms: 200
    chunk_fetch_timeout_ms: 400
"#;
        let p = parse_paste(yaml).expect("paste section");
        match p.to_tree() {
            PasteNodeConfig::MatchWmClass { patterns, default } => {
                assert_eq!(patterns.len(), 2);
                assert_eq!(patterns[0].pattern, "firefox.*");
                assert_eq!(patterns[1].pattern, "*.Emacs");
                assert!(matches!(*default, PasteNodeConfig::Clipboard { .. }));
            }
            other => panic!("expected MatchWmClass root, got {:?}", other),
        }
    }

    #[test]
    fn missing_paste_section_yields_none_and_default_root_replicates_legacy() {
        let yaml = r#"
output_dir: /tmp/x
providers: {}
"#;
        let cfg: Config = serde_yaml::from_str(yaml).expect("parses");
        assert!(cfg.paste.is_none());

        // Default tree: chunk(150) → clipboard(ctrl_shift_v, 200, 400).
        let default = PasteNodeConfig::Chunk {
            chunk_chars: super::PASTE_CHUNK_CHARS,
            child: Box::new(PasteNodeConfig::Clipboard {
                shortcut: PasteShortcut::CtrlShiftV,
                restore_settle_ms: super::PasteTiming::default().restore_settle_ms,
                chunk_fetch_timeout_ms: super::PasteTiming::default().chunk_fetch_timeout_ms,
            }),
        };
        let timing = super::node::timing_from_tree(&default);
        assert_eq!(timing.restore_settle_ms, 200);
        assert_eq!(timing.chunk_fetch_timeout_ms, 400);
    }

    #[test]
    fn glob_first_match_wins_in_wm_class_patterns() {
        // Two patterns that BOTH match "firefox.Firefox" — the first
        // declared wins.
        assert!(glob_match("firefox.*", "firefox.Firefox"));
        assert!(glob_match("*.Firefox", "firefox.Firefox"));
        assert!(glob_match("*", "firefox.Firefox"));
    }

    #[test]
    fn glob_matches_wm_class_strings() {
        assert!(glob_match("*.Emacs", "emacs.Emacs"));
        assert!(!glob_match("*.Emacs", "vim.Vim"));
        assert!(glob_match("Navigator.*", "Navigator.Firefox"));
        assert!(glob_match("*", "anything.AtAll"));
    }

    #[test]
    fn no_chunk_paste_strips_chunk_wrappers_anywhere_in_tree() {
        let tree = PasteNodeConfig::Chunk {
            chunk_chars: 100,
            child: Box::new(PasteNodeConfig::MatchWmClass {
                patterns: vec![super::node::WmClassPattern {
                    pattern: "*".to_string(),
                    child: Box::new(PasteNodeConfig::Chunk {
                        chunk_chars: 50,
                        child: Box::new(PasteNodeConfig::Clipboard {
                            shortcut: PasteShortcut::CtrlV,
                            restore_settle_ms: 200,
                            chunk_fetch_timeout_ms: 400,
                        }),
                    }),
                }],
                default: Box::new(PasteNodeConfig::Clipboard {
                    shortcut: PasteShortcut::CtrlShiftV,
                    restore_settle_ms: 200,
                    chunk_fetch_timeout_ms: 400,
                }),
            }),
        };
        let stripped = tree.strip_chunks();
        // Top-level Chunk is gone; inner Chunk under MatchWmClass is also gone.
        match stripped {
            PasteNodeConfig::MatchWmClass { patterns, default } => {
                assert!(matches!(
                    *patterns[0].child,
                    PasteNodeConfig::Clipboard { .. }
                ));
                assert!(matches!(*default, PasteNodeConfig::Clipboard { .. }));
            }
            other => panic!("expected MatchWmClass after strip, got {:?}", other),
        }
    }

    /// A leaf node that records every payload it sees — lets the
    /// chunk-node test verify what was forwarded.
    struct RecordingSink(Arc<Mutex<Vec<String>>>);

    #[async_trait]
    impl PasteNode for RecordingSink {
        async fn paste(
            &self,
            text: &str,
            _ctx: &PasteCtx<'_>,
        ) -> Result<(), crate::error::TalkError> {
            self.0
                .lock()
                .expect("lock RecordingSink")
                .push(text.to_string());
            Ok(())
        }
    }

    /// A telemetry sink that records every `PasteProgress` event.
    struct ProgressRecorder(Arc<Mutex<Vec<(u64, u64)>>>);

    impl TelemetrySink for ProgressRecorder {
        fn emit(&self, ev: TranscriptionEvent) {
            if let TranscriptionEvent::PasteProgress {
                chars_pasted,
                total_chars,
                ..
            } = ev
            {
                self.0
                    .lock()
                    .expect("lock ProgressRecorder")
                    .push((chars_pasted, total_chars));
            }
        }
    }

    #[tokio::test]
    async fn chunk_node_forwards_same_chunks_as_split_into_char_chunks() {
        let text = "aaa bbb ccc ddd eee fff";
        let chunk_chars = 10;
        let expected = super::split_into_char_chunks(text, chunk_chars);

        let received = Arc::new(Mutex::new(Vec::<String>::new()));
        let progress = Arc::new(Mutex::new(Vec::<(u64, u64)>::new()));
        let progress_sink = ProgressRecorder(progress.clone());

        let chunk = ChunkNode {
            chunk_chars,
            child: Box::new(RecordingSink(received.clone())),
        };

        let clipboard = X11Clipboard::new();
        let ctx = PasteCtx {
            target_window: None,
            delete_chars_before_paste: 0,
            t_stop: None,
            sink: &progress_sink,
            clipboard: &clipboard,
        };

        chunk.paste(text, &ctx).await.expect("paste");

        let got = received.lock().expect("lock received").clone();
        assert_eq!(got, expected);

        // Cumulative chars_pasted progresses to total_chars=text.len().
        let total = text.len() as u64;
        let progress = progress.lock().expect("lock progress").clone();
        assert_eq!(progress.len(), expected.len());
        let mut cum: u64 = 0;
        for ((cp, tc), chunk) in progress.iter().zip(expected.iter()) {
            cum += chunk.len() as u64;
            assert_eq!(*tc, total);
            assert_eq!(*cp, cum);
        }
    }

    #[tokio::test]
    async fn chunk_node_with_zero_chunk_chars_pastes_whole_text_once() {
        let text = "hello world";
        let received = Arc::new(Mutex::new(Vec::<String>::new()));

        let chunk = ChunkNode {
            chunk_chars: 0,
            child: Box::new(RecordingSink(received.clone())),
        };

        let clipboard = X11Clipboard::new();
        let ctx = PasteCtx {
            target_window: None,
            delete_chars_before_paste: 0,
            t_stop: None,
            sink: &NoOpSink,
            clipboard: &clipboard,
        };

        chunk.paste(text, &ctx).await.expect("paste");

        let got = received.lock().expect("lock").clone();
        assert_eq!(got, vec!["hello world".to_string()]);
    }
}
