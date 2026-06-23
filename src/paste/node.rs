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
use crate::config::PasteShortcut;
use crate::error::TalkError;
use crate::telemetry::TelemetrySink;
use async_trait::async_trait;
use serde::Deserialize;

use super::nodes::{
    chunk::ChunkNode, clipboard::ClipboardNode, detect::DetectDisplayServerNode,
    wm_class::MatchWmClassNode, xtest::XtestTypeNode,
};

/// Default chunk size when not specified in config — matches the
/// historical [`crate::paste::PASTE_CHUNK_CHARS`] constant.
pub(crate) const DEFAULT_CHUNK_CHARS: usize = 150;

/// Default pre-restore settle window in milliseconds.
pub(crate) const DEFAULT_RESTORE_SETTLE_MS: u64 = 200;

/// Default per-chunk fetch timeout in milliseconds.
pub(crate) const DEFAULT_CHUNK_FETCH_TIMEOUT_MS: u64 = 400;

/// Runtime context threaded into every [`PasteNode::paste`] call.
///
/// `target_window`, `delete_chars_before_paste`, `t_stop`, and `sink`
/// are the canonical spec fields.  `clipboard` is an internal routing
/// field: the [`crate::paste::paste_with_root`] wrapper owns the
/// [`X11Clipboard`] handle (so save / settle / restore can read its
/// serve counter), and the [`ClipboardNode`] reuses the SAME instance
/// — without this shared handle the settle loop would observe a
/// fresh-zero counter on a different X11Clipboard and the existing
/// race-protection would be lost.
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
}

/// WM_CLASS routing pattern: glob plus the child to invoke on match.
#[derive(Debug, Clone, Deserialize)]
pub struct WmClassPattern {
    /// Glob to match against `<instance>.<class>` (e.g.
    /// `"firefox.Firefox"` or `"*.Emacs"`).  Supports `*` as the only
    /// wildcard.
    #[serde(rename = "match")]
    pub pattern: String,
    /// Sub-tree to run when this pattern matches.
    pub child: Box<PasteNodeConfig>,
}

/// Recursive paste-tree configuration — the serde-deserializable
/// surface of the node tree.
///
/// Tagged on the `node:` key so each variant is unambiguous.
/// `build()` walks this tree into a runtime `Box<dyn PasteNode>`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "node", rename_all = "kebab-case")]
pub enum PasteNodeConfig {
    /// Switch on the running display server.  Currently only the
    /// `x11` branch is wired; `wayland` returns a clear error.
    DetectDisplayServer {
        x11: Box<PasteNodeConfig>,
        #[serde(default)]
        wayland: Option<Box<PasteNodeConfig>>,
    },
    /// Route by the focused window's WM_CLASS.  Patterns are tried
    /// in order; the FIRST match wins.  `default` runs when no
    /// pattern matches (or when WM_CLASS is unavailable).
    MatchWmClass {
        patterns: Vec<WmClassPattern>,
        default: Box<PasteNodeConfig>,
    },
    /// Split the text into chunks of at most `chunk_chars` characters
    /// (word-boundary respecting) and invoke `child` once per chunk.
    /// Emits cumulative `PasteProgress` telemetry.
    Chunk {
        #[serde(default = "default_chunk_chars")]
        chunk_chars: usize,
        child: Box<PasteNodeConfig>,
    },
    /// Deliver via the clipboard: set_text + simulate keystroke +
    /// wait_until_served.  Matches the legacy `paste_one` step.
    Clipboard {
        #[serde(default)]
        shortcut: PasteShortcut,
        #[serde(default = "default_restore_settle_ms")]
        restore_settle_ms: u64,
        #[serde(default = "default_chunk_fetch_timeout_ms")]
        chunk_fetch_timeout_ms: u64,
    },
    /// Deliver via XTest keystroke synthesis (no clipboard
    /// involvement).  Phase 1 ships ASCII/Latin-1 only; non-ASCII
    /// falls back to logging a warn and skipping the character.
    XtestType {},
}

fn default_chunk_chars() -> usize {
    DEFAULT_CHUNK_CHARS
}

fn default_restore_settle_ms() -> u64 {
    DEFAULT_RESTORE_SETTLE_MS
}

fn default_chunk_fetch_timeout_ms() -> u64 {
    DEFAULT_CHUNK_FETCH_TIMEOUT_MS
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
            Self::Chunk { chunk_chars, child } => Box::new(ChunkNode {
                chunk_chars: *chunk_chars,
                child: child.build(),
            }),
            Self::Clipboard {
                shortcut,
                restore_settle_ms,
                chunk_fetch_timeout_ms,
            } => Box::new(ClipboardNode {
                shortcut: *shortcut,
                restore_settle_ms: *restore_settle_ms,
                chunk_fetch_timeout_ms: *chunk_fetch_timeout_ms,
            }),
            Self::XtestType {} => Box::new(XtestTypeNode {}),
        }
    }

    /// Recursively strip every `Chunk` node from the tree, replacing
    /// it with its child.  Implements `--no-chunk-paste` for arbitrary
    /// trees: today the flag forces a single clipboard call; for tree
    /// configs we honour the same intent at every level.
    pub fn strip_chunks(self) -> Self {
        match self {
            Self::Chunk { child, .. } => child.strip_chunks(),
            Self::DetectDisplayServer { x11, wayland } => Self::DetectDisplayServer {
                x11: Box::new(x11.strip_chunks()),
                wayland: wayland.map(|w| Box::new(w.strip_chunks())),
            },
            Self::MatchWmClass { patterns, default } => Self::MatchWmClass {
                patterns: patterns
                    .into_iter()
                    .map(|p| WmClassPattern {
                        pattern: p.pattern,
                        child: Box::new(p.child.strip_chunks()),
                    })
                    .collect(),
                default: Box::new(default.strip_chunks()),
            },
            leaf @ (Self::Clipboard { .. } | Self::XtestType {}) => leaf,
        }
    }
}

/// Resolve the path-prefix relevant settle timing from the root
/// configuration.  Used by the legacy wrapper to drive
/// `settle_before_restore` after the tree completes its work.
///
/// We walk the tree to find the FIRST `Clipboard` node and use its
/// timing.  This is a faithful reflection of today's behaviour where
/// there is exactly one clipboard sink and its timing knobs govern
/// the whole pipeline.  When no clipboard node exists (pure XTest
/// tree) the defaults are used — settle becomes a no-op in practice
/// because no clipboard is being served.
pub(crate) fn timing_from_tree(cfg: &PasteNodeConfig) -> super::PasteTiming {
    match cfg {
        PasteNodeConfig::Clipboard {
            restore_settle_ms,
            chunk_fetch_timeout_ms,
            ..
        } => super::PasteTiming {
            restore_settle_ms: *restore_settle_ms,
            chunk_fetch_timeout_ms: *chunk_fetch_timeout_ms,
        },
        PasteNodeConfig::Chunk { child, .. } => timing_from_tree(child),
        PasteNodeConfig::DetectDisplayServer { x11, .. } => timing_from_tree(x11),
        PasteNodeConfig::MatchWmClass { default, .. } => timing_from_tree(default),
        PasteNodeConfig::XtestType {} => super::PasteTiming::default(),
    }
}
