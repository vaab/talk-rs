//! XTest-type node: deliver text via direct XTest keystroke synthesis,
//! bypassing the clipboard.
//!
//! Phase 1 ships ASCII/Latin-1 only.  For each character we resolve
//! the keysym (which for ASCII equals the codepoint, by the X11
//! keysym convention) and send it via `x11_send_key_combo` — for
//! uppercase / shifted characters we prepend the Shift_L keysym.
//!
//! Non-Latin-1 characters (codepoint > 0xFF) are NOT typed: a `warn!`
//! is logged and the character is skipped.  Full Unicode XTest
//! typing requires either XKB-extension keysyms (`U+xxxx`) AND a
//! remapped keyboard mapping, OR injecting via XIM — both
//! out-of-scope for the Phase 1 refactor (Phase 2 may revisit).

use crate::error::TalkError;
use crate::paste::node::{PasteCtx, PasteNode};
use async_trait::async_trait;

pub(crate) struct XtestTypeNode {}

const KEY_SHIFT_L: u32 = 0xffe1;

/// Compute the XTest keysym sequence for `c`, or `None` if the
/// character is out of the Phase 1 supported range.
fn keysyms_for_char(c: char) -> Option<Vec<u32>> {
    let cp = c as u32;
    if cp > 0xFF {
        return None;
    }

    // Per X11 keysym convention: Latin-1 codepoints map 1:1 onto
    // keysyms.  Uppercase ASCII letters require Shift; for other
    // codepoints we trust the caller's mapping (a `set xkb_layout`
    // change could break this — same limitation as `x11_send_key_combo`).
    if c.is_ascii_uppercase() {
        Some(vec![KEY_SHIFT_L, cp])
    } else {
        Some(vec![cp])
    }
}

#[async_trait]
impl PasteNode for XtestTypeNode {
    async fn paste(&self, text: &str, _ctx: &PasteCtx<'_>) -> Result<(), TalkError> {
        for c in text.chars() {
            let syms = match keysyms_for_char(c) {
                Some(s) => s,
                None => {
                    log::warn!(
                        "paste(xtest-type): character U+{:04X} out of Latin-1 range — skipped \
                         (Phase 1 XTest typing supports ASCII/Latin-1 only)",
                        c as u32,
                    );
                    continue;
                }
            };
            let ok = tokio::task::spawn_blocking(move || crate::x11::x11_send_key_combo(&syms))
                .await
                .unwrap_or(false);
            if !ok {
                return Err(TalkError::Clipboard(format!(
                    "XTest typing failed for character {:?}",
                    c,
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::keysyms_for_char;

    #[test]
    fn ascii_lowercase_no_shift() {
        assert_eq!(keysyms_for_char('a'), Some(vec![0x61]));
    }

    #[test]
    fn ascii_uppercase_uses_shift() {
        assert_eq!(keysyms_for_char('A'), Some(vec![0xffe1, 0x41]));
    }

    #[test]
    fn space_is_supported() {
        assert_eq!(keysyms_for_char(' '), Some(vec![0x20]));
    }

    #[test]
    fn non_latin1_returns_none() {
        assert!(keysyms_for_char('é').is_some());
        assert_eq!(keysyms_for_char('😀'), None);
        assert_eq!(keysyms_for_char('漢'), None);
    }
}
