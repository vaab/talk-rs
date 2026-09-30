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
pub mod target;

pub use node::{ForegroundAppPattern, PasteCtx, PasteNode, PasteNodeConfig, WmClassPattern};

use crate::clipboard::{Clipboard, X11Clipboard};
use crate::config::PasteShortcut;
use crate::error::TalkError;

/// Number of leading characters shown in a paste-diagnostic preview.
const PASTE_PREVIEW_CHARS: usize = 60;

/// Render a short, single-line preview of `text` for paste-diagnostic
/// trace logs: the character count plus the first
/// `PASTE_PREVIEW_CHARS` characters with newlines/tabs escaped so a
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

/// Timing knobs for the paste pipeline (defined with the config
/// schema in [`crate::paste_config`]).
pub use crate::paste_config::PasteTiming;

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
    ensure_focus_with(
        window_id,
        |wid: String| async move { focus_window(&wid).await },
        get_active_window,
    )
    .await
}

/// [`ensure_focus`] with the X11 focus request and active-window query
/// injected, so the focus/backoff policy can be exercised without an
/// X server (see the `ensure_focus_*` tests).
async fn ensure_focus_with<F, FFut, A, AFut>(
    window_id: &str,
    focus: F,
    active_window: A,
) -> Result<(), TalkError>
where
    F: Fn(String) -> FFut,
    FFut: std::future::Future<Output = bool>,
    A: Fn() -> AFut,
    AFut: std::future::Future<Output = Option<String>>,
{
    let mut delay_ms = FOCUS_INITIAL_DELAY_MS;

    for attempt in 1..=FOCUS_MAX_RETRIES {
        focus(window_id.to_string()).await;
        crate::perf_counters::incr(crate::perf_counters::Counter::FocusSleeps);
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;

        if let Some(active) = active_window().await {
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
/// Concatenating the chunks reproduces `text` exactly, including all
/// whitespace. Boundary whitespace stays with the following word when it
/// fits; otherwise it is emitted in its own chunk(s). A single word longer
/// than `max_chars` is emitted as-is (never split mid-word).
pub fn split_into_char_chunks(text: &str, max_chars: usize) -> Vec<String> {
    if text.is_empty() || max_chars == 0 {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut chars = text.char_indices().peekable();
    while let Some(&(start, _)) = chars.peek() {
        let whitespace = chars.peek().is_some_and(|(_, c)| c.is_whitespace());
        while chars
            .peek()
            .is_some_and(|(_, c)| c.is_whitespace() == whitespace)
        {
            chars.next();
        }
        let end = chars.peek().map_or(text.len(), |(index, _)| *index);
        let token = &text[start..end];
        let token_chars = token.chars().count();
        if whitespace {
            // Attach boundary whitespace to the next word when the pair
            // fits, rather than leaving a dangling space in the old chunk.
            let next_word_chars = text[end..]
                .chars()
                .take_while(|c| !c.is_whitespace())
                .count();
            if !current.is_empty()
                && next_word_chars > 0
                && current.chars().count() + token_chars + next_word_chars > max_chars
            {
                chunks.push(std::mem::take(&mut current));
            }
            let mut rest = token;
            while !rest.is_empty() {
                let available = max_chars.saturating_sub(current.chars().count());
                if available == 0 {
                    chunks.push(std::mem::take(&mut current));
                    continue;
                }
                let take = rest.chars().count().min(available);
                let byte_end = rest.char_indices().nth(take).map_or(rest.len(), |(i, _)| i);
                current.push_str(&rest[..byte_end]);
                rest = &rest[byte_end..];
            }
        } else if current.chars().count() + token_chars <= max_chars {
            current.push_str(token);
        } else {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            current.push_str(token);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Materialise the default paste tree:
/// `chunk(150) → clipboard(ctrl-shift-v, 200, 300, 50)` — the exact
/// tree that reproduces post-Phase-2 paste behaviour when no
/// `paste:` section is present in the YAML config.  (Pre-Phase-2 the
/// last knob — `chunk_fetch_timeout_ms` — was 400; lowered to 300 by
/// the per-chunk target-confirmation gate; `200` is the legacy
/// `restore_settle_ms` retained for backward-compat but unused at
/// runtime; `50` is the new `target_quiescence_ms` knob.)
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
            target_quiescence_ms: PasteTiming::default().target_quiescence_ms,
            target_fetch_retries: crate::paste_config::DEFAULT_TARGET_FETCH_RETRIES,
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
    crate::paste_config::timing_from_tree(cfg)
}

/// Paste `text` through the supplied root node, wrapping with
/// save-clipboard / restore-clipboard.  Direct successor of the
/// legacy `paste_text_to_target`.
///
/// Behaviour for the default tree:
/// focus → optional backspace → resolve target client-base → save
/// clipboard → root.paste(text) → restore.  No "settle" stability
/// window is needed at the wrapper level: the
/// `crate::paste::nodes::clipboard::ClipboardNode` gate confirms
/// the target consumed the LAST chunk before returning, so we can
/// restore the original clipboard immediately afterwards.
///
/// `timing` is retained for API compatibility — the per-clipboard
/// gate uses the knobs declared on its own [`PasteNodeConfig::Clipboard`]
/// instance inside the tree, not these struct fields.
#[allow(clippy::too_many_arguments)]
pub async fn paste_with_root(
    root: &dyn PasteNode,
    target_window: Option<&String>,
    text: &str,
    delete_chars_before_paste: usize,
    t_stop: Option<std::time::Instant>,
    sink: &dyn crate::telemetry::TelemetrySink,
    timing: PasteTiming,
    alert: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
) -> Result<(), TalkError> {
    let clipboard = X11Clipboard::new();
    let total_chars = text.chars().count() as u64;

    // Wrapper-level timing knobs are no longer consumed here (the
    // per-chunk gate carries its own).  Kept on the API surface so
    // callers can keep passing their resolved PasteTiming without
    // touching every call site.
    let _ = timing;

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

    let saved_clipboard = clipboard.snapshot().await.unwrap_or_else(|error| {
        log::warn!("could not snapshot original clipboard: {error}");
        None
    });
    log::trace!(
        "paste: saved original clipboard targets = {}",
        saved_clipboard
            .as_ref()
            .map_or(0, |snapshot| snapshot.targets.len()),
    );

    // Legacy "timing: stop +Nms first_paste" log — emitted once,
    // immediately before the first keystroke leaves this process.
    if let Some(t) = t_stop {
        log::info!("timing: stop +{}ms first_paste", t.elapsed().as_millis());
    }

    // Resolve the target X11 client-base ONCE for this paste
    // operation.  See `node::PasteCtx::target_client_base` docs for
    // the contract.  Parse / mask failures fall back to None
    // (blind-paste fallback gate in ClipboardNode) — never hard-fail
    // here, the contract is "best-effort resolution, deterministic
    // path when possible".
    let target_client_base = resolve_target_client_base(target_window).await;

    let target_window_str: Option<&str> = target_window.map(|s| s.as_str());
    let ctx = PasteCtx {
        target_window: target_window_str,
        delete_chars_before_paste,
        t_stop,
        sink,
        clipboard: &clipboard,
        target_client_base,
        expected_target_fetches: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        alert,
    };

    let paste_result = root.paste(text, &ctx).await;

    // Restore the original clipboard regardless of whether paste
    // succeeded: a half-finished paste should not leave clipboard
    // contents from the failed operation in the user's clipboard.
    crate::clipboard::restore_saved(&clipboard, saved_clipboard).await;

    paste_result?;

    log::trace!("paste: END (total_chars={})", total_chars);
    Ok(())
}

/// Resolve the X11 client-base of the optional target window XID
/// string.
///
/// Returns `None` when `target_window` is absent, when the string
/// fails to parse as a base-10 `u32`, or when the X11 connection
/// used to read `resource_id_mask` cannot be established.  All
/// three failure modes are silently mapped to "fall back to the
/// legacy gate" — none of them should abort the paste, since blind
/// pastes have always worked and this is meant to be a
/// best-effort upgrade.
async fn resolve_target_client_base(target_window: Option<&String>) -> Option<u32> {
    let wid = match target_window.and_then(|s| s.parse::<u32>().ok()) {
        Some(w) => w,
        None => {
            if target_window.is_some() {
                log::debug!(
                    "paste: target_window {:?} could not be parsed as u32 \
                     — falling back to legacy served_count gate",
                    target_window,
                );
            }
            return None;
        }
    };
    let base = tokio::task::spawn_blocking(move || crate::x11::x11_client_base(wid))
        .await
        .ok()
        .flatten();
    match base {
        Some(b) => {
            log::debug!(
                "paste: resolved target X11 client-base {:#x} for window {} \
                 (deterministic gate enabled)",
                b,
                wid,
            );
            Some(b)
        }
        None => {
            log::debug!(
                "paste: failed to resolve X11 client-base for window {} \
                 — falling back to legacy served_count gate",
                wid,
            );
            None
        }
    }
}

/// Per-segment paste guard for the realtime path.
///
/// Today the realtime per-segment loop in `dictate/mod.rs` saved the
/// clipboard before the first segment, pasted each segment via the
/// configured paste tree (chunk wrappers stripped — each segment
/// pastes whole), then restored at the end.  The legacy
/// "settle-before-restore" stability window has been REMOVED here:
/// the per-chunk target-confirmation gate inside the clipboard node
/// already waits for the actual target to consume each segment, so
/// no further stability window is needed at finish time.
///
/// The realtime path normally has no specific target window (the
/// segment is pasted into whatever currently has focus), so the
/// clipboard gate falls back to the legacy `served_count > 0`
/// behaviour with a warning — backward compatible.
pub struct RealtimeClipboardGuard {
    clipboard: X11Clipboard,
    saved: Option<crate::clipboard::ClipboardSnapshot>,
}

impl RealtimeClipboardGuard {
    /// Save the current clipboard.  Does not pre-focus the target
    /// window — call sites do that separately to preserve today's
    /// ordering.
    ///
    /// `timing` is accepted for backward API compat but no longer
    /// drives any behaviour at the guard level: the per-chunk gate
    /// inside each clipboard-node call owns the timing knobs that
    /// matter.
    pub async fn begin(timing: PasteTiming) -> Self {
        let _ = timing;
        let clipboard = X11Clipboard::new();
        let saved = clipboard.snapshot().await.unwrap_or_else(|error| {
            log::warn!("could not snapshot original clipboard: {error}");
            None
        });
        log::trace!(
            "paste(realtime): saved original clipboard targets = {}",
            saved.as_ref().map_or(0, |snapshot| snapshot.targets.len()),
        );
        Self { clipboard, saved }
    }

    /// Paste one segment through `root`.  Each call is independent —
    /// no chunking is applied (the realtime path always pasted
    /// whole segments).
    pub async fn paste_segment(
        &self,
        root: &dyn PasteNode,
        segment: &str,
        delete_chars_before_paste: usize,
        target_window: Option<&str>,
        sink: &dyn crate::telemetry::TelemetrySink,
    ) -> Result<(), TalkError> {
        if delete_chars_before_paste > 0 {
            if let Some(window) = target_window {
                ensure_focus(window).await?;
                simulate_backspace(delete_chars_before_paste).await?;
            }
        }
        let ctx = PasteCtx {
            target_window,
            delete_chars_before_paste,
            t_stop: None,
            sink,
            clipboard: &self.clipboard,
            // Realtime path: no specific target window → fall back
            // to the legacy `served_count > 0` gate inside the
            // clipboard node.  Each segment carries its OWN learn
            // state (the AtomicU32 starts at 0 every call), so even
            // when a target_window IS plumbed in later we'd cleanly
            // learn per-segment.
            target_client_base: None,
            expected_target_fetches: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            // Realtime path has no target window and no sound player
            // plumbed in — nothing to signal audibly here.
            alert: None,
        };
        root.paste(segment, &ctx).await
    }

    /// Restore the original clipboard.  Idempotent.  No
    /// settle-stability window: the per-chunk gate inside each
    /// `paste_segment` call already confirmed the target consumed
    /// the last segment before returning.
    pub async fn finish(self) {
        crate::clipboard::restore_saved(&self.clipboard, self.saved).await;
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

    tokio::task::spawn_blocking(move || crate::x11::x11_send_key_combo_checked(&keysyms))
        .await
        .map_err(|error| TalkError::Clipboard(format!("XTest paste worker failed: {error}")))?
        .map_err(|error| TalkError::Clipboard(format!("XTest paste shortcut failed: {error}")))
}

/// Simulate deleting the previous text by sending repeated BackSpace
/// via the XTest extension.
pub async fn simulate_backspace(count: usize) -> Result<(), TalkError> {
    if count == 0 {
        return Ok(());
    }

    // X11 keysym for BackSpace.
    const BACKSPACE: u32 = 0xff08;

    tokio::task::spawn_blocking(move || crate::x11::x11_send_key_repeat_checked(BACKSPACE, count))
        .await
        .map_err(|error| TalkError::Clipboard(format!("XTest backspace worker failed: {error}")))?
        .map_err(|error| TalkError::Clipboard(format!("XTest backspace failed: {error}")))
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
    fn chunking_preserves_every_whitespace_character() {
        let text = "  hello  world\n\tgoodbye \r\n";
        let chunks = split_into_char_chunks(text, 9);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn chunk_limit_counts_unicode_characters() {
        assert_eq!(split_into_char_chunks("é é", 3), vec!["é é"]);
    }

    #[test]
    fn boundary_whitespace_does_not_make_a_chunk_oversize() {
        assert_eq!(split_into_char_chunks("a bb", 2), vec!["a", " ", "bb"]);
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
                assert_eq!(f.target_quiescence_ms, 50);
            }
            PasteConfig::Tree(_) => panic!("expected flat variant for legacy YAML"),
        }

        // Building the root tree from flat must collapse to
        // chunk(80) → clipboard(ctrl_v, 250, 600, <default-quiescence>).
        let tree = p.to_tree();
        match tree {
            PasteNodeConfig::Chunk { chunk_chars, child } => {
                assert_eq!(chunk_chars, 80);
                match *child {
                    PasteNodeConfig::Clipboard {
                        shortcut,
                        restore_settle_ms,
                        chunk_fetch_timeout_ms,
                        target_quiescence_ms,
                        target_fetch_retries,
                    } => {
                        assert_eq!(shortcut, PasteShortcut::CtrlV);
                        assert_eq!(restore_settle_ms, 250);
                        assert_eq!(chunk_fetch_timeout_ms, 600);
                        assert_eq!(target_quiescence_ms, 50);
                        // Flat YAML also omits target_fetch_retries here;
                        // flat → tree adapter falls back to the default.
                        assert_eq!(target_fetch_retries, 2);
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
    target_quiescence_ms: 60
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
                            target_quiescence_ms,
                            target_fetch_retries,
                        } => {
                            assert_eq!(shortcut, PasteShortcut::CtrlShiftV);
                            assert_eq!(restore_settle_ms, 150);
                            assert_eq!(chunk_fetch_timeout_ms, 350);
                            assert_eq!(target_quiescence_ms, 60);
                            // Omitted in this fixture → default.
                            assert_eq!(target_fetch_retries, 2);
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
    fn foreground_app_router_yaml_deserialises_as_tree_variant() {
        let yaml = indoc::indoc! {r#"
            output_dir: /tmp/x
            providers: {}
            paste:
              node: match-foreground-app
              patterns:
                - match: opencode-tui
                  child:
                    node: chunk
                    chunk_chars: 150
                    child:
                      node: clipboard
                      shortcut: ctrl_shift_v
              default:
                node: clipboard
                shortcut: ctrl_shift_v
        "#};

        assert!(matches!(parse_paste(yaml), Some(PasteConfig::Tree(_))));
    }

    #[test]
    fn outer_shortcut_is_independent_from_foreground_app_classification() {
        let yaml = indoc::indoc! {r#"
            output_dir: /tmp/x
            providers: {}
            paste:
              node: match-wm-class
              patterns:
                - match: "@terminal"
                  child:
                    node: match-foreground-app
                    patterns:
                      - match: opencode-tui
                        child:
                          node: chunk
                          chunk_chars: 150
                          child:
                            node: clipboard
                            shortcut: ctrl_shift_v
                    default:
                      node: clipboard
                      shortcut: ctrl_shift_v
              default:
                node: clipboard
                shortcut: ctrl_v
        "#};
        let paste = parse_paste(yaml).expect("paste tree").to_tree();
        let (terminal, gui) = match paste {
            PasteNodeConfig::MatchWmClass { patterns, default } => {
                (patterns[0].child.clone(), default)
            }
            other => panic!("expected WM_CLASS router, got {other:?}"),
        };
        match *terminal {
            PasteNodeConfig::MatchForegroundApp { patterns, default } => {
                assert!(matches!(
                    &*patterns[0].child,
                    PasteNodeConfig::Chunk { child, .. }
                        if matches!(&**child, PasteNodeConfig::Clipboard {
                            shortcut: PasteShortcut::CtrlShiftV,
                            ..
                        })
                ));
                assert!(matches!(
                    *default,
                    PasteNodeConfig::Clipboard {
                        shortcut: PasteShortcut::CtrlShiftV,
                        ..
                    }
                ));
            }
            other => panic!("expected foreground-app router, got {other:?}"),
        }
        assert!(matches!(
            *gui,
            PasteNodeConfig::Clipboard {
                shortcut: PasteShortcut::CtrlV,
                ..
            }
        ));
    }

    #[test]
    fn malformed_foreground_app_router_does_not_fall_back_to_flat_config() {
        let yaml = indoc::indoc! {r#"
            output_dir: /tmp/x
            providers: {}
            paste:
              node: match-foreground-app
              patterns:
                - match: opencode-tui
                  child:
                    node: clipboard
                    shortcut: ctrl_shift_v
        "#};

        serde_yaml::from_str::<Config>(yaml)
            .expect_err("router without a default child must be rejected");
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

        // Default tree: chunk(150) → clipboard(ctrl_shift_v, 200,
        // 500, 50).  Note 500 vs the pre-retry 300 — see the
        // `DEFAULT_CHUNK_FETCH_TIMEOUT_MS` constant doc.
        let default = PasteNodeConfig::Chunk {
            chunk_chars: super::PASTE_CHUNK_CHARS,
            child: Box::new(PasteNodeConfig::Clipboard {
                shortcut: PasteShortcut::CtrlShiftV,
                restore_settle_ms: super::PasteTiming::default().restore_settle_ms,
                chunk_fetch_timeout_ms: super::PasteTiming::default().chunk_fetch_timeout_ms,
                target_quiescence_ms: super::PasteTiming::default().target_quiescence_ms,
                target_fetch_retries: crate::paste_config::DEFAULT_TARGET_FETCH_RETRIES,
            }),
        };
        let timing = crate::paste_config::timing_from_tree(&default);
        assert_eq!(timing.restore_settle_ms, 200);
        assert_eq!(timing.chunk_fetch_timeout_ms, 500);
        assert_eq!(timing.target_quiescence_ms, 50);
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
                            target_quiescence_ms: 50,
                            target_fetch_retries: 2,
                        }),
                    }),
                }],
                default: Box::new(PasteNodeConfig::Clipboard {
                    shortcut: PasteShortcut::CtrlShiftV,
                    restore_settle_ms: 200,
                    chunk_fetch_timeout_ms: 400,
                    target_quiescence_ms: 50,
                    target_fetch_retries: 2,
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

    #[test]
    fn no_chunk_paste_strips_chunks_inside_foreground_app_router() {
        let tree = PasteNodeConfig::MatchForegroundApp {
            patterns: vec![super::node::ForegroundAppPattern {
                pattern: "opencode-tui".to_string(),
                child: Box::new(PasteNodeConfig::Chunk {
                    chunk_chars: 150,
                    child: Box::new(PasteNodeConfig::Clipboard {
                        shortcut: PasteShortcut::CtrlShiftV,
                        restore_settle_ms: 200,
                        chunk_fetch_timeout_ms: 500,
                        target_quiescence_ms: 50,
                        target_fetch_retries: 2,
                    }),
                }),
            }],
            default: Box::new(PasteNodeConfig::Clipboard {
                shortcut: PasteShortcut::CtrlShiftV,
                restore_settle_ms: 200,
                chunk_fetch_timeout_ms: 500,
                target_quiescence_ms: 50,
                target_fetch_retries: 2,
            }),
        };

        match tree.strip_chunks() {
            PasteNodeConfig::MatchForegroundApp { patterns, default } => {
                assert!(matches!(
                    *patterns[0].child,
                    PasteNodeConfig::Clipboard {
                        shortcut: PasteShortcut::CtrlShiftV,
                        ..
                    }
                ));
                assert!(matches!(
                    *default,
                    PasteNodeConfig::Clipboard {
                        shortcut: PasteShortcut::CtrlShiftV,
                        ..
                    }
                ));
            }
            other => panic!("expected foreground-app router, got {other:?}"),
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
    async fn chunk_node_forwards_literal_chunks_and_character_progress() {
        let text = "éé bb cc";
        let chunk_chars = 5;
        let expected = vec!["éé bb", " cc"];

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
            target_client_base: None,
            expected_target_fetches: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            alert: None,
        };

        chunk.paste(text, &ctx).await.expect("paste");

        let got = received.lock().expect("lock received").clone();
        assert_eq!(got, expected);

        let progress = progress.lock().expect("lock progress").clone();
        assert_eq!(progress, vec![(5, 8), (8, 8)]);
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
            target_client_base: None,
            expected_target_fetches: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            alert: None,
        };

        chunk.paste(text, &ctx).await.expect("paste");

        let got = received.lock().expect("lock").clone();
        assert_eq!(got, vec!["hello world".to_string()]);
    }
}
