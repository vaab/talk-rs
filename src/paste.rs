//! Clipboard paste utilities and window-focus helpers.
//!
//! Extracted from `dictate.rs` — these are the low-level building
//! blocks for pasting transcription text into a target application
//! via X11 and the clipboard.

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

/// Paste `text` into the target window, optionally deleting characters
/// first (for replace-last-paste) and chunking long text.
///
/// When `chunk_chars` is `0`, the entire text is pasted in one shot.
/// Otherwise it is split via [`split_into_char_chunks`].
pub async fn paste_text_to_target(
    target_window: Option<&String>,
    text: &str,
    delete_chars_before_paste: usize,
    chunk_chars: usize,
    t_stop: Option<std::time::Instant>,
    sink: &dyn crate::telemetry::TelemetrySink,
    shortcut: PasteShortcut,
) -> Result<(), TalkError> {
    let clipboard = X11Clipboard::new();
    let total_chars = text.len() as u64;

    log::trace!(
        "paste: BEGIN shortcut={:?} chunk_chars={} delete_before={} target_window={:?} text={}",
        shortcut,
        chunk_chars,
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

    let mut chars_pasted: u64 = 0;
    let mut first_paste_logged = false;
    if chunk_chars == 0 {
        // No chunking: paste the entire text in one shot.
        clipboard.set_text(text).await?;
        paste_one(
            &clipboard,
            text,
            0,
            1,
            &shortcut,
            &mut first_paste_logged,
            t_stop,
        )
        .await?;
        chars_pasted = total_chars;
        sink.emit(crate::telemetry::TranscriptionEvent::PasteProgress {
            chars_pasted,
            total_chars,
            t: std::time::Instant::now(),
        });
    } else {
        let chunks = split_into_char_chunks(text, chunk_chars);
        let n_chunks = chunks.len();
        log::trace!("paste: split into {} chunk(s)", n_chunks);
        for (idx, chunk) in chunks.iter().enumerate() {
            clipboard.set_text(chunk).await?;
            paste_one(
                &clipboard,
                chunk,
                idx,
                n_chunks,
                &shortcut,
                &mut first_paste_logged,
                t_stop,
            )
            .await?;
            chars_pasted += chunk.len() as u64;
            sink.emit(crate::telemetry::TranscriptionEvent::PasteProgress {
                chars_pasted,
                total_chars,
                t: std::time::Instant::now(),
            });
        }
    }

    // Extra settle time before restoring the clipboard so the last
    // paste has time to be consumed by the target application.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    if let Some(saved) = saved_clipboard {
        log::trace!(
            "paste: restoring original clipboard = {}",
            log_preview(&saved)
        );
        let _ = clipboard.set_text(&saved).await;
    }

    log::trace!("paste: END ({} chars pasted total)", chars_pasted);
    Ok(())
}

/// Set-then-paste a single chunk and emit `-vvv` diagnostics: a
/// read-back check that the clipboard holds what we intended, and the
/// "served" count proving whether the target actually fetched it.
///
/// `clipboard` must already hold `chunk` (the caller sets it before
/// calling) so the read-back verifies the claim succeeded.
#[allow(clippy::too_many_arguments)]
async fn paste_one(
    clipboard: &X11Clipboard,
    chunk: &str,
    idx: usize,
    n_chunks: usize,
    shortcut: &PasteShortcut,
    first_paste_logged: &mut bool,
    t_stop: Option<std::time::Instant>,
) -> Result<(), TalkError> {
    log::trace!(
        "paste: chunk {}/{} set, content={}",
        idx + 1,
        n_chunks,
        log_preview(chunk),
    );

    // Read-back: confirm the clipboard actually holds our chunk.  A
    // mismatch reveals an ownership-claim race (the prime suspect for
    // "old clipboard content" leaking into the paste).  Only performed
    // under trace verbosity to avoid the extra X round-trip otherwise.
    if log::log_enabled!(log::Level::Trace) {
        match clipboard.get_text().await {
            Ok(readback) if readback == chunk => {
                log::trace!("paste: chunk {}/{} read-back OK", idx + 1, n_chunks);
            }
            Ok(readback) => {
                log::trace!(
                    "paste: chunk {}/{} read-back MISMATCH — clipboard holds {}",
                    idx + 1,
                    n_chunks,
                    log_preview(&readback),
                );
            }
            Err(e) => {
                log::trace!(
                    "paste: chunk {}/{} read-back failed: {}",
                    idx + 1,
                    n_chunks,
                    e,
                );
            }
        }
    }

    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    if !*first_paste_logged {
        if let Some(t) = t_stop {
            log::info!("timing: stop +{}ms first_paste", t.elapsed().as_millis());
        }
        *first_paste_logged = true;
    }

    simulate_paste(*shortcut).await?;
    tokio::time::sleep(std::time::Duration::from_millis(15)).await;

    // Served count: how many times the target fetched the offered
    // text.  `0` here means the paste keystroke did NOT cause the
    // target to pull our clipboard — i.e. wrong selection, lost focus,
    // or a shortcut the app maps to a different paste source.
    let served = clipboard.last_served_count();
    if served == 0 {
        log::trace!(
            "paste: chunk {}/{} pasted but served_count=0 \
             (target did NOT fetch our clipboard — check shortcut/focus)",
            idx + 1,
            n_chunks,
        );
    } else {
        log::trace!(
            "paste: chunk {}/{} consumed (served_count={})",
            idx + 1,
            n_chunks,
            served,
        );
    }

    Ok(())
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
