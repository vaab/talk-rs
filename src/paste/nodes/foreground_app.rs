//! First-match routing by the foreground application inside a terminal.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::TalkError;
use crate::paste::node::{PasteCtx, PasteNode};
use crate::paste::target::{ForegroundAppResolver, SystemForegroundAppResolver, TargetIdentity};

use super::glob_match;

pub(crate) struct MatchForegroundAppNode {
    pub(crate) patterns: Vec<(String, Box<dyn PasteNode>)>,
    pub(crate) default: Box<dyn PasteNode>,
    pub(crate) resolver: Arc<dyn ForegroundAppResolver>,
}

impl MatchForegroundAppNode {
    pub(crate) fn system(
        patterns: Vec<(String, Box<dyn PasteNode>)>,
        default: Box<dyn PasteNode>,
    ) -> Self {
        Self {
            patterns,
            default,
            resolver: Arc::new(SystemForegroundAppResolver),
        }
    }

    async fn resolve(&self, target_xid: u32) -> TargetIdentity {
        let resolver = self.resolver.clone();
        tokio::task::spawn_blocking(move || resolver.resolve(target_xid))
            .await
            .unwrap_or_else(|_| TargetIdentity {
                app: crate::paste::target::ForegroundApp::Unknown,
                terminal: None,
                pane: None,
                reason: "resolver-task-failed".to_string(),
            })
    }
}

#[async_trait]
impl PasteNode for MatchForegroundAppNode {
    async fn paste(&self, text: &str, ctx: &PasteCtx<'_>) -> Result<(), TalkError> {
        let identity = match ctx.target_window.and_then(|wid| wid.parse::<u32>().ok()) {
            Some(wid) => self.resolve(wid).await,
            None => TargetIdentity {
                app: crate::paste::target::ForegroundApp::Unknown,
                terminal: None,
                pane: None,
                reason: "target-window-unavailable".to_string(),
            },
        };
        let label = identity.app.label();
        for (pattern, child) in &self.patterns {
            if glob_match(pattern, label) {
                log_identity(&identity, child.chunks_text());
                return child.paste(text, ctx).await;
            }
        }
        log_identity(&identity, self.default.chunks_text());
        self.default.paste(text, ctx).await
    }
}

fn log_identity(identity: &TargetIdentity, chunk: bool) {
    log::info!(
        "paste(foreground-app): terminal={} pane={} classification={} reason={} chunk={}",
        identity.terminal.as_deref().unwrap_or("unknown"),
        identity.pane.as_deref().unwrap_or("none"),
        identity.app.label(),
        identity.reason,
        chunk,
    );
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::clipboard::X11Clipboard;
    use crate::paste::target::ForegroundApp;
    use crate::telemetry::NoOpSink;

    struct FixedResolver {
        xid: Arc<Mutex<Vec<u32>>>,
        app: ForegroundApp,
    }

    impl ForegroundAppResolver for FixedResolver {
        fn resolve(&self, target_xid: u32) -> TargetIdentity {
            self.xid.lock().expect("resolver xid lock").push(target_xid);
            TargetIdentity {
                app: self.app,
                terminal: Some("fixture".to_string()),
                pane: None,
                reason: "fixture".to_string(),
            }
        }
    }

    struct RecordingNode(&'static str, Arc<Mutex<Vec<String>>>);

    #[async_trait]
    impl PasteNode for RecordingNode {
        async fn paste(&self, text: &str, _ctx: &PasteCtx<'_>) -> Result<(), TalkError> {
            self.1
                .lock()
                .expect("recording node lock")
                .push(format!("{}:{text}", self.0));
            Ok(())
        }
    }

    async fn route(app: ForegroundApp) -> (Vec<String>, Vec<u32>) {
        let routed = Arc::new(Mutex::new(Vec::new()));
        let xids = Arc::new(Mutex::new(Vec::new()));
        let node = MatchForegroundAppNode {
            patterns: vec![(
                "opencode-tui".to_string(),
                Box::new(RecordingNode("chunked", routed.clone())),
            )],
            default: Box::new(RecordingNode("full", routed.clone())),
            resolver: Arc::new(FixedResolver {
                xid: xids.clone(),
                app,
            }),
        };
        let clipboard = X11Clipboard::new();
        let ctx = PasteCtx {
            target_window: Some("4242"),
            delete_chars_before_paste: 0,
            t_stop: None,
            sink: &NoOpSink,
            clipboard: &clipboard,
            target_client_base: None,
            expected_target_fetches: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            alert: None,
        };
        node.paste("Unicode 中文 payload", &ctx)
            .await
            .expect("route payload");
        let routed = routed.lock().expect("routed lock").clone();
        let xids = xids.lock().expect("xids lock").clone();
        (routed, xids)
    }

    #[tokio::test]
    async fn only_positive_opencode_identity_uses_configured_child() {
        assert_eq!(
            route(ForegroundApp::OpenCodeTui).await,
            (vec!["chunked:Unicode 中文 payload".to_string()], vec![4242])
        );
        for app in [
            ForegroundApp::PiTui,
            ForegroundApp::Emacs,
            ForegroundApp::Shell,
            ForegroundApp::Unknown,
        ] {
            assert_eq!(
                route(app).await,
                (vec!["full:Unicode 中文 payload".to_string()], vec![4242])
            );
        }
    }
}
