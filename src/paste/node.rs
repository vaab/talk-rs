//! Composable paste-node abstraction.
//!
//! The paste pipeline is modelled as a tree of [`PasteNode`]s: each
//! node consumes a text payload and either transforms it (chunk,
//! match-wm-class, detect-display-server) or delivers it (clipboard,
//! xtest-type).  The runtime tree is built from a serde-deserializable
//! [`PasteNodeConfig`] (the YAML-facing schema) via
//! [`PasteNodeConfig::build`].
//!
//! Phase 1 invariant: the DEFAULT tree (built from a missing/flat
//! `paste:` section) reproduces the legacy single-path behaviour
//! exactly — `chunk(150) → clipboard(ctrl-shift-v, 200, 400)`.  See
//! [`crate::paste::default_root`].

use crate::clipboard::X11Clipboard;
use crate::error::TalkError;
use crate::telemetry::TelemetrySink;
use async_trait::async_trait;

use super::nodes::{
    chunk::ChunkNode, clipboard::ClipboardNode, detect::DetectDisplayServerNode,
    foreground_app::MatchForegroundAppNode, wm_class::MatchWmClassNode, xtest::XtestTypeNode,
};

// The YAML-facing schema lives in the always-compiled
// `crate::paste_config` so headless builds can parse tree-form
// `paste:` sections; re-exported here to keep the historical paths.
pub use crate::paste_config::{ForegroundAppPattern, PasteNodeConfig, WmClassPattern};

/// Runtime context threaded into every [`PasteNode::paste`] call.
///
/// `target_window`, `delete_chars_before_paste`, `t_stop`, and `sink`
/// are the canonical spec fields.  `clipboard` is an internal routing
/// field: the [`crate::paste::paste_with_root`] wrapper owns the
/// [`X11Clipboard`] handle (so save / restore can reach its serve
/// counter), and the `ClipboardNode` reuses the SAME instance
/// — without this shared handle the per-chunk gate would observe a
/// fresh-zero counter on a different X11Clipboard and the existing
/// race-protection would be lost.
///
/// `target_client_base` and `expected_target_fetches` carry the
/// per-paste-operation state for the deterministic per-chunk gate
/// implemented in `crate::paste::nodes::clipboard::ClipboardNode`.
/// See the field docs for the contract.
pub struct PasteCtx<'a> {
    /// XID of the target window as a base-10 string, or `None` when
    /// pasting blind (no specific window to refocus).
    pub target_window: Option<&'a str>,
    /// Number of backspaces to send before the actual paste
    /// (`--replace-last-paste`).  Currently consumed by the wrapper
    /// before the node tree runs; carried in ctx for nodes that may
    /// want to display it.
    pub delete_chars_before_paste: usize,
    /// Wall-clock instant at which the user pressed "stop", used for
    /// `timing: stop +Nms first_paste` log lines.  `None` outside
    /// the one-shot dictate path.
    pub t_stop: Option<std::time::Instant>,
    /// Telemetry sink — receives [`crate::telemetry::TranscriptionEvent::PasteProgress`]
    /// events emitted by the chunk node.
    pub sink: &'a dyn TelemetrySink,
    /// Shared clipboard handle (see struct docs).
    pub(crate) clipboard: &'a X11Clipboard,
    /// X11 client-base of the target window (= `xid & !resource_id_mask`),
    /// resolved once at paste-operation start in
    /// [`crate::paste::paste_with_root`].
    ///
    /// `Some(base)` enables the DETERMINISTIC per-chunk gate: the
    /// clipboard node waits until the target client-base has
    /// fetched the expected number of times before advancing.
    /// `None` (blind paste, missing target_window, or X11 connection
    /// failure during resolution) falls back to the legacy
    /// `served_count > 0` gate with a warning.
    pub(crate) target_client_base: Option<u32>,
    /// Per-paste-operation state shared across chunk invocations of
    /// [`crate::paste::nodes::clipboard::ClipboardNode`].
    ///
    /// Chunk 1 LEARNS the target's fetch count (typically 1 or 2 —
    /// modern toolkits fetch UTF8_STRING twice for a single paste,
    /// once for size probing and once for the real read).  Once
    /// learned, subsequent chunks CONFIRM the same count is reached
    /// before advancing.
    ///
    /// Encoding: `0` = not yet learned (initial state); `N>0` =
    /// chunk 1 observed exactly `N` target fetches.  Interior
    /// mutability across `&` ctx via `Arc<AtomicU32>` so chunk-node
    /// invocations share state without taking a `&mut` on the ctx.
    pub(crate) expected_target_fetches: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Optional alert hook, invoked ONCE on a FINAL paste abort
    /// (target-confirmation retries exhausted) to give the user an
    /// audible signal that nothing was pasted.  Reuses the same
    /// [`crate::audio::indicator::AlertPlayer`] triple-pulse tone as
    /// the dead-audio "NO SOUND" feature.
    ///
    /// `None` on paste paths that have no sound player (picker /
    /// cached-transcript flows using [`crate::telemetry::NoOpSink`]);
    /// populated from the one-shot dictate path where the
    /// [`crate::audio::indicator::SoundPlayer`] lives.  Wrapped in an
    /// `Arc<dyn Fn()>` so the ctx stays `Send`-safe and cheap to
    /// construct without leaking `AlertPlayer` into the paste API.
    pub(crate) alert: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
}

/// A node in the paste tree.
///
/// `paste(text, ctx)` consumes one text payload; composite nodes
/// (chunk, match-wm-class, detect) delegate to their child(ren); leaf
/// nodes (clipboard, xtest-type) actually deliver the text to the
/// target window.
#[async_trait]
pub trait PasteNode: Send + Sync {
    async fn paste(&self, text: &str, ctx: &PasteCtx<'_>) -> Result<(), TalkError>;

    /// Whether this node starts delivery by splitting the payload.
    fn chunks_text(&self) -> bool {
        false
    }
}

impl PasteNodeConfig {
    /// Materialise the runtime node tree.
    pub fn build(&self) -> Box<dyn PasteNode> {
        match self {
            Self::DetectDisplayServer { x11, wayland } => Box::new(DetectDisplayServerNode {
                x11: x11.build(),
                wayland: wayland.as_ref().map(|w| w.build()),
            }),
            Self::MatchWmClass { patterns, default } => {
                let compiled = patterns
                    .iter()
                    .map(|p| (p.pattern.clone(), p.child.build()))
                    .collect();
                Box::new(MatchWmClassNode {
                    patterns: compiled,
                    default: default.build(),
                })
            }
            Self::MatchForegroundApp { patterns, default } => {
                let compiled = patterns
                    .iter()
                    .map(|pattern| (pattern.pattern.clone(), pattern.child.build()))
                    .collect();
                Box::new(MatchForegroundAppNode::system(compiled, default.build()))
            }
            Self::Chunk { chunk_chars, child } => Box::new(ChunkNode {
                chunk_chars: *chunk_chars,
                child: child.build(),
            }),
            Self::Clipboard {
                shortcut,
                restore_settle_ms,
                chunk_fetch_timeout_ms,
                target_quiescence_ms,
                target_fetch_retries,
            } => Box::new(ClipboardNode {
                shortcut: *shortcut,
                restore_settle_ms: *restore_settle_ms,
                chunk_fetch_timeout_ms: *chunk_fetch_timeout_ms,
                target_quiescence_ms: *target_quiescence_ms,
                target_fetch_retries: *target_fetch_retries,
            }),
            Self::XtestType {} => Box::new(XtestTypeNode {}),
        }
    }
}
