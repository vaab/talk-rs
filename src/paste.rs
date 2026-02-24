//! Clipboard paste utilities and window-focus helpers.
//!
//! Extracted from `dictate.rs` — these are the low-level building
//! blocks for pasting transcription text into a target application
//! via X11 and the clipboard.

use crate::clipboard::{Clipboard, X11Clipboard};
use crate::error::TalkError;

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
) -> Result<(), TalkError> {
    let clipboard = X11Clipboard::new();

    if let Some(wid) = target_window {
        log::debug!("refocusing target window: {}", wid);
        ensure_focus(wid).await?;
    }

    if delete_chars_before_paste > 0 {
        log::info!("deleting {} chars before paste", delete_chars_before_paste);
        simulate_backspace(delete_chars_before_paste).await?;
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }

    let saved_clipboard = clipboard.get_text().await.ok();

    if chunk_chars == 0 {
        // No chunking: paste the entire text in one shot.
        clipboard.set_text(text).await?;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        simulate_paste().await?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    } else {
        let chunks = split_into_char_chunks(text, chunk_chars);
        for chunk in &chunks {
            clipboard.set_text(chunk).await?;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            simulate_paste().await?;
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    // Extra settle time before restoring the clipboard so the last
    // paste has time to be consumed by the target application.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    if let Some(saved) = saved_clipboard {
        let _ = clipboard.set_text(&saved).await;
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

/// Simulate a Ctrl+Shift+V paste keystroke via the XTest extension.
pub async fn simulate_paste() -> Result<(), TalkError> {
    // X11 keysyms for Control_L, Shift_L, v
    const CONTROL_L: u32 = 0xffe3;
    const SHIFT_L: u32 = 0xffe1;
    const KEY_V: u32 = 0x0076;

    let ok = tokio::task::spawn_blocking(|| {
        crate::x11::x11_send_key_combo(&[CONTROL_L, SHIFT_L, KEY_V])
    })
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
}
