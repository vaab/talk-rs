//! Paste-tree configuration schema (pure data, no desktop stack).
//!
//! These types are the YAML-facing half of the composable paste-node
//! tree.  They live outside the `ui`-gated [`crate::paste`] module so
//! that every feature set — including headless
//! (`--no-default-features`) builds that embed talk-rs only for
//! transcription — can parse any valid `paste:` section of the shared
//! config file.  Materialising a tree into runtime X11 nodes
//! (`PasteNodeConfig::build`) stays in `crate::paste::node`, behind the
//! `ui` feature.

use crate::config::PasteShortcut;
use serde::Deserialize;

/// Default chunk size when not specified in config — matches the
/// historical `crate::paste::PASTE_CHUNK_CHARS` constant.
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
/// one observed failure mode (focus not yet effective when the keystroke
/// was sent). Only the deterministic target-confirmation path retries;
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

/// Timing knobs for the paste pipeline.
///
/// Threaded through `crate::paste::paste_with_root` for callers that
/// want to override the per-chunk gate deadline / quiescence window
/// without growing the call-site signature unboundedly.
///
/// `restore_settle_ms` is RETAINED on this struct (and in the YAML
/// schema) for backward compatibility but is no longer used at
/// runtime: the pre-restore "settle" heuristic has been replaced by
/// the deterministic per-chunk target-confirmation gate inside the
/// clipboard node, which removes the race between the last chunk's
/// fetch and the clipboard restore by construction.
///
/// `Default` matches the config defaults (200 / 500 / 50).
#[derive(Debug, Clone, Copy)]
pub struct PasteTiming {
    /// **Backward-compat only.**  See struct doc.
    pub restore_settle_ms: u64,
    /// See `paste.chunk_fetch_timeout_ms`.
    pub chunk_fetch_timeout_ms: u64,
    /// See `paste.target_quiescence_ms`.
    pub target_quiescence_ms: u64,
}

impl Default for PasteTiming {
    fn default() -> Self {
        Self {
            restore_settle_ms: DEFAULT_RESTORE_SETTLE_MS,
            chunk_fetch_timeout_ms: DEFAULT_CHUNK_FETCH_TIMEOUT_MS,
            target_quiescence_ms: DEFAULT_TARGET_QUIESCENCE_MS,
        }
    }
}

/// WM_CLASS routing pattern: glob plus the child to invoke on match.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WmClassPattern {
    /// Glob to match against `<instance>.<class>` (e.g.
    /// `"firefox.Firefox"` or `"*.Emacs"`).  Supports `*` as the only
    /// wildcard.
    #[serde(rename = "match")]
    pub pattern: String,
    /// Sub-tree to run when this pattern matches.
    pub child: Box<PasteNodeConfig>,
}

/// Foreground-application routing pattern: normalized label plus child.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForegroundAppPattern {
    /// Glob matched against labels such as `opencode-tui`.
    #[serde(rename = "match")]
    pub pattern: String,
    pub child: Box<PasteNodeConfig>,
}

/// Recursive paste-tree configuration — the serde-deserializable
/// surface of the node tree.
///
/// Tagged on the `node:` key so each variant is unambiguous.
/// With the `ui` feature, `build()` walks this tree into a runtime
/// `Box<dyn crate::paste::PasteNode>`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "node", rename_all = "kebab-case", deny_unknown_fields)]
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
    /// Route by the foreground process bound to the remembered terminal
    /// surface. Unknown or ambiguous identities use `default`.
    MatchForegroundApp {
        patterns: Vec<ForegroundAppPattern>,
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
            Self::MatchForegroundApp { patterns, default } => Self::MatchForegroundApp {
                patterns: patterns
                    .into_iter()
                    .map(|pattern| ForegroundAppPattern {
                        pattern: pattern.pattern,
                        child: Box::new(pattern.child.strip_chunks()),
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
pub(crate) fn timing_from_tree(cfg: &PasteNodeConfig) -> PasteTiming {
    match cfg {
        PasteNodeConfig::Clipboard {
            restore_settle_ms,
            chunk_fetch_timeout_ms,
            target_quiescence_ms,
            ..
        } => PasteTiming {
            restore_settle_ms: *restore_settle_ms,
            chunk_fetch_timeout_ms: *chunk_fetch_timeout_ms,
            target_quiescence_ms: *target_quiescence_ms,
        },
        PasteNodeConfig::Chunk { child, .. } => timing_from_tree(child),
        PasteNodeConfig::DetectDisplayServer { x11, .. } => timing_from_tree(x11),
        PasteNodeConfig::MatchWmClass { default, .. } => timing_from_tree(default),
        PasteNodeConfig::MatchForegroundApp { default, .. } => timing_from_tree(default),
        PasteNodeConfig::XtestType {} => PasteTiming::default(),
    }
}
