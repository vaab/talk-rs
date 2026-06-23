//! Clipboard paste-node: set_text → simulate keystroke →
//! wait_until_served.
//!
//! This is the leaf where the legacy `paste_one` algorithm lives.
//! The save / settle / restore steps live in the
//! [`crate::paste::paste_with_root`] wrapper so they apply ONCE per
//! whole-paste operation, not per chunk.

use crate::clipboard::Clipboard as _;
use crate::config::PasteShortcut;
use crate::error::TalkError;
use crate::paste::node::{PasteCtx, PasteNode};
use crate::paste::{log_preview, simulate_paste};
use async_trait::async_trait;

/// Tunables for a [`ClipboardNode`].  See
/// [`crate::config::PasteConfig`] for the YAML-facing knobs of the
/// same name.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClipboardNode {
    pub(crate) shortcut: PasteShortcut,
    /// Carried for `timing_from_tree` extraction — the actual settle
    /// loop runs in the wrapper, NOT here.  See `node.rs`.
    #[allow(dead_code)] // Surfaced indirectly via `node::timing_from_tree`.
    pub(crate) restore_settle_ms: u64,
    pub(crate) chunk_fetch_timeout_ms: u64,
}

#[async_trait]
impl PasteNode for ClipboardNode {
    async fn paste(&self, text: &str, ctx: &PasteCtx<'_>) -> Result<(), TalkError> {
        let clipboard = ctx.clipboard;

        log::trace!(
            "paste(clipboard-node): set chunk content={}",
            log_preview(text),
        );

        clipboard.set_text(text).await?;

        // Read-back diagnostic (verbatim from legacy `paste_one`).
        if log::log_enabled!(log::Level::Trace) {
            match clipboard.get_text().await {
                Ok(rb) if rb == text => {
                    log::trace!("paste(clipboard-node): chunk read-back OK");
                }
                Ok(rb) => {
                    log::trace!(
                        "paste(clipboard-node): read-back MISMATCH — clipboard holds {}",
                        log_preview(&rb),
                    );
                }
                Err(e) => {
                    log::trace!("paste(clipboard-node): read-back failed: {}", e);
                }
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(5)).await;

        simulate_paste(self.shortcut).await?;

        // Wait for the target to actually fetch the offered content.
        // Same baseline=0 (fresh serve handle per set_text) and same
        // timeout knob as legacy.
        let served = clipboard
            .wait_until_served(
                0,
                std::time::Duration::from_millis(self.chunk_fetch_timeout_ms),
            )
            .await;
        if served == 0 {
            log::trace!(
                "paste(clipboard-node): pasted but served_count=0 \
                 (target did NOT fetch our clipboard — check shortcut/focus)",
            );
            log::warn!(
                "paste(clipboard-node): timed out after {} ms with served_count=0 \
                 — target never fetched our clipboard, likely paste corruption",
                self.chunk_fetch_timeout_ms,
            );
        } else {
            log::trace!("paste(clipboard-node): consumed (served_count={})", served,);
        }

        // `t_stop` first-paste timing is logged once by the
        // wrapper (see `paste_with_root`), not per chunk here.
        let _ = ctx.t_stop;

        Ok(())
    }
}
