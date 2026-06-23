//! Detect-display-server node: dispatch to the x11 or wayland subtree
//! based on the running session.
//!
//! Detection is environment-based: `$WAYLAND_DISPLAY` non-empty ⇒
//! wayland session, else x11.  When the wayland branch is selected
//! but no `wayland:` child is configured, paste fails with a clear
//! error — this is intentional for Phase 1 (Wayland support is
//! out-of-scope here).

use crate::error::TalkError;
use crate::paste::node::{PasteCtx, PasteNode};
use async_trait::async_trait;

pub(crate) struct DetectDisplayServerNode {
    pub(crate) x11: Box<dyn PasteNode>,
    pub(crate) wayland: Option<Box<dyn PasteNode>>,
}

fn is_wayland_session() -> bool {
    std::env::var("WAYLAND_DISPLAY")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

#[async_trait]
impl PasteNode for DetectDisplayServerNode {
    async fn paste(&self, text: &str, ctx: &PasteCtx<'_>) -> Result<(), TalkError> {
        if is_wayland_session() {
            match self.wayland.as_ref() {
                Some(w) => {
                    log::trace!("paste(detect): routing to wayland branch");
                    w.paste(text, ctx).await
                }
                None => Err(TalkError::Clipboard(
                    "Wayland paste not supported (Phase 1 ships X11 only); \
                     configure a `wayland:` branch under `detect-display-server` \
                     or remove the node to force X11 routing"
                        .to_string(),
                )),
            }
        } else {
            log::trace!("paste(detect): routing to x11 branch");
            self.x11.paste(text, ctx).await
        }
    }
}
