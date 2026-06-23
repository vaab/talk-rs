//! Match-WM-class node: first-match routing by the focused window's
//! `WM_CLASS` property.
//!
//! Patterns are tried in declared order; the first matching pattern's
//! child runs.  If no pattern matches (including the WM_CLASS lookup
//! itself failing), the `default` child runs.
//!
//! The glob surface is intentionally tiny: `*` is the only wildcard.
//! See [`super::glob_match`] for the matcher.

use crate::error::TalkError;
use crate::paste::node::{PasteCtx, PasteNode};
use async_trait::async_trait;

use super::glob_match;

pub(crate) struct MatchWmClassNode {
    pub(crate) patterns: Vec<(String, Box<dyn PasteNode>)>,
    pub(crate) default: Box<dyn PasteNode>,
}

#[async_trait]
impl PasteNode for MatchWmClassNode {
    async fn paste(&self, text: &str, ctx: &PasteCtx<'_>) -> Result<(), TalkError> {
        let wm_class = match ctx.target_window {
            Some(wid_str) => match wid_str.parse::<u32>() {
                Ok(wid) => tokio::task::spawn_blocking(move || crate::x11::x11_get_wm_class(wid))
                    .await
                    .ok()
                    .flatten(),
                Err(_) => None,
            },
            None => None,
        };

        if let Some((instance, class)) = wm_class.as_ref() {
            let key = format!("{}.{}", instance, class);
            log::trace!("paste(wm-class): WM_CLASS={:?}", key);
            for (pattern, child) in &self.patterns {
                if glob_match(pattern, &key) {
                    log::trace!(
                        "paste(wm-class): matched pattern {:?} → routing to child",
                        pattern,
                    );
                    return child.paste(text, ctx).await;
                }
            }
            log::trace!("paste(wm-class): no pattern matched → default");
        } else {
            log::trace!("paste(wm-class): WM_CLASS unavailable → default");
        }

        self.default.paste(text, ctx).await
    }
}
