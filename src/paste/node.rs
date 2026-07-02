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
///
/// This is the ABORT deadline for the deterministic target-confirmation
/// gate, not a silent-advance one: when a target client-base is known
/// and it has not fetched the expected count within this window, the
/// clipboard node RETRIES (re-focus + re-send keystroke + re-wait; see
/// [`DEFAULT_TARGET_FETCH_RETRIES`]) and only fails loudly once retries
/// are exhausted.  Widened from 300 → 500 ms after a real session
/// showed a transient focus failure (Ctrl+Shift+V sent before the
/// keyboard focus was effective) causing one abort out of thirteen
/// pastes: the extra 200 ms plus the retry loop cover the worst-case
/// re-focus latency observed in the wild.
pub(crate) const DEFAULT_CHUNK_FETCH_TIMEOUT_MS: u64 = 500;

/// Default number of automatic per-chunk retries when the target
/// client-base does not fetch a chunk within
/// [`DEFAULT_CHUNK_FETCH_TIMEOUT_MS`].
///
/// `2` retries means up to THREE total attempts per chunk: the initial
/// attempt plus two re-tries.  Each retry re-serves the chunk
/// (`set_text`), re-focuses the target window, re-sends the paste
/// keystroke and re-waits on the gate — the retry directly addresses
/// the observed root cause (focus not yet effective when the keystroke
/// was sent).  Only the deterministic target-confirmation path retries;
/// the blind-paste fallback gate is unchanged.
pub(crate) const DEFAULT_TARGET_FETCH_RETRIES: u32 = 2;

/// Default per-chunk target-quiescence window in milliseconds.
///
/// Once the target client-base has fetched the expected number of
/// times for the current chunk, the gate waits this long for any
/// extra fetches (extra clipboard pull from the same client, or a
/// trailing fetch by a clipboard manager) before advancing to the
/// next chunk.  50 ms is large enough to absorb the empirically
/// observed second fetch from real apps and small enough to keep
/// total paste latency unchanged at typical 4-7 chunk pastes.
pub(crate) const DEFAULT_TARGET_QUIESCENCE_MS: u64 = 50;

/// Runtime context threaded into every [`PasteNode::paste`] call.
///
/// `target_window`, `delete_chars_before_paste`, `t_stop`, and `sink`
/// are the canonical spec fields.  `clipboard` is an internal routing
/// field: the [`crate::paste::paste_with_root`] wrapper owns the
/// [`X11Clipboard`] handle (so save / restore can reach its serve
/// counter), and the [`ClipboardNode`] reuses the SAME instance
/// — without this shared handle the per-chunk gate would observe a
/// fresh-zero counter on a different X11Clipboard and the existing
/// race-protection would be lost.
///
/// `target_client_base` and `expected_target_fetches` carry the
/// per-paste-operation state for the deterministic per-chunk gate
/// implemented in [`crate::paste::nodes::clipboard::ClipboardNode`].
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
    /// per-chunk target-confirmation gate.  Replaces the legacy
    /// `paste_one` step with a deterministic chunk advancement (no
    /// more dropped or duplicated chunks; see the field docs).
    Clipboard {
        #[serde(default)]
        shortcut: PasteShortcut,
        /// **Backward-compat only.**  Was the pre-restore "settle"
        /// window in the legacy heuristic.  Replaced by the
        /// per-chunk target-confirmation gate (which removes the
        /// need for a stability window at the end of the operation);
        /// kept as an accepted-but-ignored config field so existing
        /// YAML configs do not need editing.
        #[serde(default = "default_restore_settle_ms")]
        restore_settle_ms: u64,
        /// Per-chunk ABORT deadline in milliseconds.  When a target
        /// client-base is resolved, the paste fails loudly after
        /// this many milliseconds without the target reaching the
        /// expected fetch count.  When no target client-base could
        /// be resolved (blind paste), governs the fallback
        /// `served_count > 0` timeout (with a warning, not an
        /// abort, to preserve backward compatibility).
        #[serde(default = "default_chunk_fetch_timeout_ms")]
        chunk_fetch_timeout_ms: u64,
        /// Per-chunk target-quiescence window in milliseconds.  Once
        /// the target client-base has reached its fetch count for
        /// the current chunk, the gate waits this long for any
        /// trailing fetches (extra read from the same client, or a
        /// late clipboard manager) before advancing.
        #[serde(default = "default_target_quiescence_ms")]
        target_quiescence_ms: u64,
        /// Number of automatic per-chunk retries on the deterministic
        /// target-confirmation path.  When the target client-base does
        /// not fetch a chunk within `chunk_fetch_timeout_ms`, the node
        /// re-serves the chunk, re-focuses the target window, re-sends
        /// the paste keystroke and re-waits, up to this many extra
        /// times before aborting.  Default `2` (= up to 3 attempts).
        /// Has no effect on the blind-paste fallback gate.
        #[serde(default = "default_target_fetch_retries")]
        target_fetch_retries: u32,
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

fn default_target_quiescence_ms() -> u64 {
    DEFAULT_TARGET_QUIESCENCE_MS
}

fn default_target_fetch_retries() -> u32 {
    DEFAULT_TARGET_FETCH_RETRIES
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

/// Resolve the relevant paste timing from the root configuration.
/// Used by the wrapper to drive the per-chunk target-confirmation
/// abort deadline and the final clipboard restore once the tree has
/// completed its work.
///
/// We walk the tree to find the FIRST `Clipboard` node and use its
/// timing.  This is a faithful reflection of today's behaviour where
/// there is exactly one clipboard sink and its timing knobs govern
/// the whole pipeline.  When no clipboard node exists (pure XTest
/// tree) the defaults are used — timing is irrelevant in practice
/// because no clipboard is being served.
pub(crate) fn timing_from_tree(cfg: &PasteNodeConfig) -> super::PasteTiming {
    match cfg {
        PasteNodeConfig::Clipboard {
            restore_settle_ms,
            chunk_fetch_timeout_ms,
            target_quiescence_ms,
            ..
        } => super::PasteTiming {
            restore_settle_ms: *restore_settle_ms,
            chunk_fetch_timeout_ms: *chunk_fetch_timeout_ms,
            target_quiescence_ms: *target_quiescence_ms,
        },
        PasteNodeConfig::Chunk { child, .. } => timing_from_tree(child),
        PasteNodeConfig::DetectDisplayServer { x11, .. } => timing_from_tree(x11),
        PasteNodeConfig::MatchWmClass { default, .. } => timing_from_tree(default),
        PasteNodeConfig::XtestType {} => super::PasteTiming::default(),
    }
}
