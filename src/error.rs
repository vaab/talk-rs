//! Error types for talk-rs
//!
//! This module defines the error types used throughout the application.
//! Each major component has its own error variant for clear error handling.

use std::time::Duration;
use thiserror::Error;

/// Main error type for talk-rs operations.
#[derive(Error, Debug)]
pub enum TalkError {
    /// Configuration-related errors.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Audio capture/encoding errors.
    #[error("Audio error: {0}")]
    Audio(String),

    /// Transcription API errors.
    #[error("Transcription error: {0}")]
    Transcription(String),

    /// IO operation errors.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Clipboard operation errors.
    #[error("Clipboard error: {0}")]
    Clipboard(String),

    /// Session management errors.
    #[error("Session error: {0}")]
    Session(String),

    /// Pick-level lock present: another process is producing the
    /// authoritative transcript for this recording.
    #[error("Transcription already in progress")]
    TranscriptInProgress,

    /// Caller requested cache-only lookup but the per-model sidecar
    /// cache was empty.  Signals that an API call would be required.
    #[error("Transcription not cached (API call forbidden)")]
    CacheOnly,

    /// Per-model lock present: another process is calling the API
    /// for this specific (recording, provider, model) triple.
    #[error("Transcription for this model already in progress")]
    ModelInProgress,

    /// Structured failure inside the HTTP transcription pipeline
    /// (preflight `/v1/models` validation OR batch
    /// `/v1/audio/transcriptions` request).  Replaces the older
    /// `Config(String)` / `Transcription(String)` wrappings that
    /// flattened structured cause data into a single string.
    ///
    /// Boxed to keep `TalkError`'s stack size small — without
    /// the box, `Result<T, TalkError>` would carry
    /// `PipelineFailure`'s ~140-byte payload through every
    /// function returning a `TalkError`-flavoured Result.
    ///
    /// Consumers wanting detail pattern-match on the inner
    /// [`PipelineFailure`]; consumers wanting a sentence rely on
    /// the `Display` impl.
    #[error(transparent)]
    Pipeline(Box<PipelineFailure>),
}

/// Auto-box conversions: producers write `pf.into()` (or use the
/// `?` operator on a `Result<_, PipelineFailure>`) and the value
/// arrives boxed inside `TalkError::Pipeline`.  Eliminates
/// manual `Box::new(...)` at every producer call site.
impl From<PipelineFailure> for TalkError {
    fn from(pf: PipelineFailure) -> Self {
        TalkError::Pipeline(Box::new(pf))
    }
}

/// Phase of the HTTP transcription pipeline at which a failure
/// occurred.
///
/// Distinguishes the two HTTP round-trips a transcription performs:
///
/// - [`Self::Validate`] — the model preflight (GET `/v1/models`).
/// - [`Self::Request`] — the actual transcription POST
///   (`/v1/audio/transcriptions`).
///
/// Carried as a typed field of [`PipelineFailure`] so consumers
/// can render phase-specific copy without re-parsing the URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelinePhase {
    /// `/v1/models` preflight validation.
    Validate,
    /// `/v1/audio/transcriptions` actual transcription request.
    Request,
}

impl PipelinePhase {
    /// Human verb fragment used in [`PipelineFailure::Display`]:
    /// `"validation"` or `"transcription request"`.  Stable copy;
    /// log greppers may match on this exact string.
    fn verb(self) -> &'static str {
        match self {
            Self::Validate => "validation",
            Self::Request => "transcription request",
        }
    }
}

/// Stringified description of a single timer that was active at a
/// pipeline-failure call site.
///
/// Sibling to [`crate::transcription::transport::http::TimerSpec`]
/// but carries already-rendered duration text so [`error`] does not
/// import the transport-layer types.  Built by
/// [`crate::transcription::transport::http::build_pipeline_failure_kind`]
/// from the `TimerSpec` slice that was active at the call site.
#[derive(Debug, Clone)]
pub struct TimerLabel {
    /// Stable identifier (e.g. `"connect_timeout"`,
    /// `"request_wall_clock"`, `"validate_request"`,
    /// `"kernel_tcp_unspecified"`).  Treated as wire format by
    /// log readers.
    pub name: String,
    /// Human-readable rendered budget (e.g. `"2s"`, `"0.050s"`,
    /// `"3s+8s"` for kernel-tcp ranges).  Stringified at
    /// construction so the `Display` impl on [`PipelineFailure`]
    /// stays free of `Duration` formatting concerns.
    pub budget: String,
}

impl TimerLabel {
    /// Convenience constructor that formats a `Duration` using the
    /// project-wide compact "Ns" / "N.NNNs" convention.
    pub fn from_duration(name: impl Into<String>, budget: Duration) -> Self {
        Self {
            name: name.into(),
            budget: fmt_duration(budget),
        }
    }
}

/// Format a `Duration` for log/error output: whole seconds when
/// possible, fractional seconds otherwise.  Mirrors the convention
/// previously private to `transport::http`; lifted here because
/// [`TimerLabel::from_duration`] is the natural producer of these
/// strings.
fn fmt_duration(d: Duration) -> String {
    if d.subsec_nanos() == 0 {
        format!("{}s", d.as_secs())
    } else {
        format!("{:.3}s", d.as_secs_f64())
    }
}

/// Coarse classification of a network-layer failure during a
/// pipeline call.  Mirrors the structural attribution rules used
/// by [`crate::transcription::transport::http::attribute_timer`]
/// but in typed form so renderers don't re-parse strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkKind {
    /// TCP connect / TLS handshake phase failed (connect_timeout,
    /// DNS, ECONNREFUSED, …).  Source chain typically carries the
    /// underlying OS error.
    Connect,
    /// Per-request wall-clock fired AFTER connection was
    /// established.  This is the case the user cares about most:
    /// the server accepted us and then took too long.
    WallClock,
    /// Kernel-level TCP timeout (TCP_USER_TIMEOUT or unanswered
    /// keepalive probes) killed an established socket.  Reqwest
    /// cannot distinguish which kernel timer expired, so the
    /// `TimerLabel` carries the merged budget range.
    KernelTcp,
    /// Anything else: HTTP/2 protocol error, body decode error
    /// surfaced as a network failure, etc.  Source chain is the
    /// only signal; rendered without a timer name.
    Other,
}

/// Terminal cause of a [`PipelineFailure`].  Cleanly typed so
/// consumers can pattern-match instead of string-grep.
#[derive(Error, Debug)]
pub enum PipelineFailureKind {
    /// Network / TCP / TLS / wall-clock failure.  Carries the
    /// timer that fired (when the cause maps to a known timer)
    /// plus the underlying error so the Display impl can dedup
    /// the source chain against the structured fields.
    #[error("{kind:?} timeout")]
    Network {
        kind: NetworkKind,
        /// `None` only for [`NetworkKind::Other`] where no timer
        /// classification applies.
        timer: Option<TimerLabel>,
        /// The underlying reqwest / hyper / OS error.  Walked by
        /// [`PipelineFailure::Display`] for novel layers (DNS,
        /// ECONNREFUSED, TLS messages) and dedup'd against the
        /// already-emitted structured fields.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Server returned a non-2xx HTTP response.
    #[error("HTTP {status}: {body}")]
    HttpStatus { status: u16, body: String },
    /// Server replied 200 OK but the model isn't in the listed
    /// set.  Permanent — no retry can change the answer.
    /// Pre-migration this case used to surface as
    /// `TalkError::Config(...)`; structural detection lifts it
    /// out of the string-matching legacy.
    #[error("model '{model}' not in available list")]
    ModelRejected {
        model: String,
        /// Sorted list of transcription models the provider DID
        /// list.  Empty when the provider returned no
        /// transcription-class models at all.
        suggestions: Vec<String>,
    },
    /// Response body parse error.
    #[error("could not parse response: {0}")]
    Decode(String),
}

/// A single HTTP transcription-pipeline failure with structured
/// diagnostic context.
///
/// Constructed by the transport layer (`transport/http.rs`,
/// `mistral.rs::send_once`, `openai.rs::send_once`) and consumed
/// by anyone with a `TalkError`: log line writers, the picker UI,
/// or future structured exporters.  The `Display` impl produces
/// a one-line rendering with intelligent source-chain dedup.
#[derive(Error, Debug)]
pub struct PipelineFailure {
    /// Provider display name (`"Mistral"`, `"OpenAI"`) — stored
    /// as the human-facing form because producers already have
    /// the title-cased name in scope (e.g. the `provider_name`
    /// argument to
    /// [`crate::transcription::transport::http::validate_model`]).
    /// Stored as `String` (not `crate::config::Provider`) to
    /// avoid the import cycle: `config` already imports `error`.
    pub provider: String,
    /// At which pipeline phase the failure occurred.
    pub phase: PipelinePhase,
    /// 1-indexed count of attempts actually made before this
    /// failure surfaced.  `attempts == max_attempts` means the
    /// retry budget was fully exhausted.
    pub attempts: u32,
    /// Total retry budget configured for this phase.
    pub max_attempts: u32,
    /// The fully-qualified URL that was being called.  Rendered
    /// once in the structured tag block; never printed twice in
    /// the same Display output.
    pub url: String,
    /// Structured cause.
    #[source]
    pub kind: PipelineFailureKind,
}

impl PipelineFailure {
    /// Build a new structured failure.
    ///
    /// Producers in `transport::http` and provider modules use
    /// this constructor exclusively — there is no `From` impl
    /// from `reqwest::Error` to keep the
    /// "reqwest mechanics live in transport, not error" boundary.
    pub fn new(
        provider: impl Into<String>,
        phase: PipelinePhase,
        attempts: u32,
        max_attempts: u32,
        url: impl Into<String>,
        kind: PipelineFailureKind,
    ) -> Self {
        Self {
            provider: provider.into(),
            phase,
            attempts,
            max_attempts,
            url: url.into(),
            kind,
        }
    }
}

/// Render a [`PipelineFailure`] as a clean one-line message:
///
/// ```text
/// {Provider} model {phase} failed [name=<timer>, budget=<budget>, url=<url>] (after {attempts}/{max} attempts) -> <novel source-chain layers>
/// ```
///
/// Specific examples:
///
/// ```text
/// OpenAI model validation failed [name=connect_timeout, budget=2s, url=https://...] (after 5/5 attempts)
/// Mistral transcription request failed [name=request_wall_clock, budget=15s, url=https://...] (after 1/6 attempts) -> Connection refused (os error 111)
/// Mistral model 'voxtral-mini-9999' not found (preflight to /v1/models). Available: voxtral-mini-2507, voxtral-mini-2602
/// ```
///
/// # Source-chain dedup
///
/// For `PipelineFailureKind::Network`, the `Display` walks
/// `std::error::Error::source` from the inner error and skips
/// layers that just restate fields already present in the
/// structured output (e.g., "client error (Connect)",
/// "operation timed out").  DNS errors, ECONNREFUSED, TLS alerts,
/// and OS errno layers are kept because they carry information
/// that is NOT in the structured fields.
impl std::fmt::Display for PipelineFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `provider` is already in display form ("Mistral",
        // "OpenAI") per the field doc; render verbatim.
        let provider_display = &self.provider;

        match &self.kind {
            PipelineFailureKind::Network {
                kind: _,
                timer,
                source,
            } => {
                write!(
                    f,
                    "{} model {} failed [",
                    provider_display,
                    self.phase.verb(),
                )?;
                if let Some(t) = timer {
                    write!(f, "name={}, budget={}, ", t.name, t.budget)?;
                }
                write!(
                    f,
                    "url={}] (after {}/{} attempts)",
                    self.url, self.attempts, self.max_attempts
                )?;
                render_dedup_source_chain(f, source.as_ref(), timer.as_ref(), &self.url)?;
                Ok(())
            }
            PipelineFailureKind::HttpStatus { status, body } => {
                let body_trimmed = body.trim();
                let body_short = if body_trimmed.len() > 200 {
                    format!("{}…", &body_trimmed[..200])
                } else {
                    body_trimmed.to_string()
                };
                // 4xx is permanent (no retry could change the
                // answer); 5xx is data-phase-retryable (the
                // transport exhausted its data-retry budget if
                // attempts > 1).  Annotate so users don't
                // think the "1/5" attempts in a 404 means we
                // gave up early.
                let retry_note: &'static str = if (400..500).contains(status) {
                    " — 4xx permanent, no retry"
                } else if (500..600).contains(status) && self.attempts >= self.max_attempts {
                    " — server-retry budget exhausted"
                } else {
                    ""
                };
                write!(
                    f,
                    "{} model {} failed [status={}, url={}] (after {}/{} attempts{})",
                    provider_display,
                    self.phase.verb(),
                    status,
                    self.url,
                    self.attempts,
                    self.max_attempts,
                    retry_note,
                )?;
                if !body_short.is_empty() {
                    write!(f, ": {}", body_short)?;
                }
                Ok(())
            }
            PipelineFailureKind::ModelRejected { model, suggestions } => {
                let phase_note = match self.phase {
                    PipelinePhase::Validate => " (preflight to /v1/models)",
                    PipelinePhase::Request => "",
                };
                write!(
                    f,
                    "{} model '{}' not found{}",
                    provider_display, model, phase_note,
                )?;
                if !suggestions.is_empty() {
                    write!(f, ". Available: {}", suggestions.join(", "))?;
                }
                Ok(())
            }
            PipelineFailureKind::Decode(msg) => {
                write!(
                    f,
                    "{} model {} failed [decode, url={}] (after {}/{} attempts): {}",
                    provider_display,
                    self.phase.verb(),
                    self.url,
                    self.attempts,
                    self.max_attempts,
                    msg,
                )
            }
        }
    }
}

/// Walk `source()` from `err` and append " -> <layer>" for each
/// layer whose lowercased text adds information beyond what the
/// structured tag block already showed.
///
/// Skipped (redundant) patterns:
///
/// - "client error (connect)" / "client error (timeout)" — same
///   information as `kind: NetworkKind::Connect/WallClock` plus
///   the timer name.
/// - "operation timed out" / "deadline has elapsed" / "request
///   timeout" — same as the timer name.
/// - The literal substring of an already-emitted layer (the
///   pre-migration dedup we kept).
///
/// Kept (novel) patterns:
///
/// - DNS lookup messages ("failed to lookup address information",
///   "name or service not known").
/// - OS errno messages ("Connection refused (os error 111)").
/// - TLS / certificate errors ("invalid peer certificate",
///   "tls handshake eof").
/// - Anything not matching the skip patterns.
fn render_dedup_source_chain(
    f: &mut std::fmt::Formatter<'_>,
    err: &(dyn std::error::Error + 'static),
    timer: Option<&TimerLabel>,
    url: &str,
) -> std::fmt::Result {
    let mut shown: Vec<String> = Vec::new();
    let mut current: Option<&dyn std::error::Error> = Some(err);
    while let Some(e) = current {
        let msg = e.to_string();
        let lower = msg.to_lowercase();

        let redundant = is_layer_redundant(&lower, timer, url)
            || shown
                .iter()
                .any(|prev| prev.contains(&msg) || msg.contains(prev.as_str()));

        if !redundant {
            write!(f, " -> {}", msg)?;
            shown.push(msg);
        }
        current = e.source();
    }
    Ok(())
}

/// Returns `true` when the source-chain layer's lowercased text is
/// a restatement of fields already shown in the structured tag
/// block.  Kept private — its shape mirrors the
/// [`PipelineFailureKind::Network`] Display rules and changes in
/// lockstep.
fn is_layer_redundant(lower: &str, timer: Option<&TimerLabel>, url: &str) -> bool {
    // Restatements of `kind: NetworkKind::Connect` /
    // `name=connect_timeout`.
    if lower.contains("client error (connect)") {
        return true;
    }
    // Restatements of any timeout-class timer.
    if lower == "operation timed out"
        || lower == "deadline has elapsed"
        || lower == "request timeout"
        || lower == "client error (timeout)"
    {
        return true;
    }
    // If a timer was set and the layer just repeats the timer
    // name verbatim, drop it.
    if let Some(t) = timer {
        if lower == t.name {
            return true;
        }
    }
    // Reqwest restates the URL in its top-level error
    // ``error sending request for url (...)``.  We already
    // surface the URL in the ``url=...`` tag block; the layer
    // is pure noise.
    let url_lower = url.to_lowercase();
    if !url_lower.is_empty()
        && lower.contains(&format!("error sending request for url ({})", url_lower))
    {
        return true;
    }
    // Same dedup for the trailing ``for url (URL)`` fragment
    // when reqwest nests it inside a wider message.
    if !url_lower.is_empty() && lower == format!("error sending request for url ({})", url_lower) {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_error_display() {
        let err = TalkError::Config("missing field".to_string());
        assert_eq!(err.to_string(), "Configuration error: missing field");
    }

    #[test]
    fn test_io_error_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: TalkError = io_err.into();
        assert!(matches!(err, TalkError::Io(_)));
    }

    // ── PipelineFailure Display ─────────────────────────────────

    /// Spec: a Network failure with a timer renders as a clean
    /// one-liner with name, budget, url, and attempts counter.
    #[test]
    fn pipeline_failure_network_renders_cleanly() {
        let inner = std::io::Error::new(std::io::ErrorKind::TimedOut, "operation timed out");
        let pf = PipelineFailure::new(
            "OpenAI",
            PipelinePhase::Validate,
            5,
            5,
            "https://api.openai.com/v1/models",
            PipelineFailureKind::Network {
                kind: NetworkKind::Connect,
                timer: Some(TimerLabel::from_duration(
                    "connect_timeout",
                    Duration::from_secs(2),
                )),
                source: Box::new(inner),
            },
        );

        let s = pf.to_string();
        // Lead phrase, no "Configuration error:" prefix.
        assert!(
            s.starts_with("OpenAI model validation failed ["),
            "got: {}",
            s
        );
        assert!(s.contains("name=connect_timeout"), "got: {}", s);
        assert!(s.contains("budget=2s"), "got: {}", s);
        assert!(
            s.contains("url=https://api.openai.com/v1/models"),
            "got: {}",
            s
        );
        assert!(s.contains("(after 5/5 attempts)"), "got: {}", s);
    }

    /// Spec: redundant source-chain layers ("operation timed out")
    /// are NOT appended when their text would just restate the
    /// already-shown timer info.
    #[test]
    fn pipeline_failure_dedups_operation_timed_out() {
        let inner = std::io::Error::new(std::io::ErrorKind::TimedOut, "operation timed out");
        let pf = PipelineFailure::new(
            "OpenAI",
            PipelinePhase::Validate,
            5,
            5,
            "https://x",
            PipelineFailureKind::Network {
                kind: NetworkKind::Connect,
                timer: Some(TimerLabel::from_duration(
                    "connect_timeout",
                    Duration::from_secs(2),
                )),
                source: Box::new(inner),
            },
        );

        let s = pf.to_string();
        // The bare "operation timed out" layer must not appear —
        // it's redundant with the timer name.
        assert!(
            !s.contains(" -> operation timed out"),
            "redundant layer must be skipped, got: {}",
            s
        );
    }

    /// Spec: novel source-chain layers (DNS, ECONNREFUSED, TLS)
    /// ARE appended because they carry information not in the
    /// structured fields.
    #[test]
    fn pipeline_failure_keeps_econnrefused_layer() {
        let inner = std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "Connection refused (os error 111)",
        );
        let pf = PipelineFailure::new(
            "Mistral",
            PipelinePhase::Request,
            1,
            6,
            "https://x",
            PipelineFailureKind::Network {
                kind: NetworkKind::Connect,
                timer: Some(TimerLabel::from_duration(
                    "connect_timeout",
                    Duration::from_secs(2),
                )),
                source: Box::new(inner),
            },
        );

        let s = pf.to_string();
        assert!(
            s.contains("Connection refused (os error 111)"),
            "ECONNREFUSED layer must be kept, got: {}",
            s
        );
    }

    /// Spec (Step 13 of transport-consolidation): reqwest's
    /// top-level ``error sending request for url (URL)`` layer is
    /// dropped from the source chain because the URL is already
    /// surfaced in the structured ``url=`` tag.  The same URL must
    /// not appear twice in the rendered failure.
    #[test]
    fn pipeline_failure_dedups_reqwest_url_restatement() {
        let url = "https://mistral.vps-03.0k.io/v1/audio/transcriptions";
        let reqwest_layer =
            std::io::Error::other(format!("error sending request for url ({})", url));
        let pf = PipelineFailure::new(
            "Mistral",
            PipelinePhase::Request,
            5,
            5,
            url,
            PipelineFailureKind::Network {
                kind: NetworkKind::Connect,
                timer: Some(TimerLabel::from_duration(
                    "connect_timeout",
                    Duration::from_secs(2),
                )),
                source: Box::new(reqwest_layer),
            },
        );

        let s = pf.to_string();
        // URL appears exactly once.
        let url_occurrences = s.matches(url).count();
        assert_eq!(
            url_occurrences, 1,
            "URL must appear exactly once (Step 13 dedup); got {} occurrences in {}",
            url_occurrences, s
        );
        // Specifically: the reqwest restatement is not in the chain.
        assert!(
            !s.contains(" -> error sending request for url"),
            "reqwest URL-restatement layer must be dropped; got: {}",
            s
        );
    }

    /// Spec: ModelRejected renders with "not found" + suggestions
    /// when present, no tag block, no `Configuration error:`
    /// prefix.
    #[test]
    fn pipeline_failure_model_rejected_renders_with_suggestions() {
        let pf = PipelineFailure::new(
            "Mistral",
            PipelinePhase::Validate,
            1,
            5,
            "https://x",
            PipelineFailureKind::ModelRejected {
                model: "voxtral-mini-9999".into(),
                suggestions: vec!["voxtral-mini-2507".into(), "voxtral-mini-2602".into()],
            },
        );

        let s = pf.to_string();
        assert!(
            s.starts_with("Mistral model 'voxtral-mini-9999' not found (preflight to /v1/models)"),
            "got: {}",
            s
        );
        assert!(
            s.contains("Available: voxtral-mini-2507, voxtral-mini-2602"),
            "got: {}",
            s
        );
        assert!(!s.contains("Configuration error:"), "got: {}", s);
    }

    /// Spec: ModelRejected with empty suggestions omits the
    /// "Available: …" tail.
    #[test]
    fn pipeline_failure_model_rejected_no_suggestions() {
        let pf = PipelineFailure::new(
            "OpenAI",
            PipelinePhase::Validate,
            1,
            5,
            "https://x",
            PipelineFailureKind::ModelRejected {
                model: "ghost-model".into(),
                suggestions: vec![],
            },
        );
        let s = pf.to_string();
        assert!(
            s.contains("model 'ghost-model' not found (preflight to /v1/models)"),
            "got: {}",
            s
        );
        assert!(!s.contains("Available:"), "got: {}", s);
    }

    /// Spec: HttpStatus renders the status, URL, attempts, and a
    /// trimmed body.  4xx responses additionally render the
    /// `4xx permanent, no retry` annotation so users don't
    /// mistakenly think we gave up early (1/5 attempts for 4xx
    /// is correct — retrying a 4xx is wasted effort).
    #[test]
    fn pipeline_failure_http_status_renders() {
        let pf = PipelineFailure::new(
            "OpenAI",
            PipelinePhase::Validate,
            1,
            5,
            "https://api.openai.com/v1/models",
            PipelineFailureKind::HttpStatus {
                status: 401,
                body: "Unauthorized".into(),
            },
        );
        let s = pf.to_string();
        assert!(s.contains("status=401"), "got: {}", s);
        assert!(s.contains("Unauthorized"), "got: {}", s);
        assert!(s.contains("(after 1/5 attempts"), "got: {}", s);
        assert!(s.contains("4xx permanent, no retry"), "got: {}", s);
    }

    /// Spec: a 5xx with exhausted retries shows the
    /// "server-retry budget exhausted" annotation; a 5xx with
    /// remaining budget (attempts < max_attempts) shows no
    /// annotation (the caller is in the middle of the retry
    /// loop and isn't ready to surface the failure yet).
    #[test]
    fn pipeline_failure_http_status_500_annotates_when_exhausted() {
        let pf = PipelineFailure::new(
            "OpenAI",
            PipelinePhase::Request,
            3,
            3,
            "https://x",
            PipelineFailureKind::HttpStatus {
                status: 503,
                body: "Service Unavailable".into(),
            },
        );
        let s = pf.to_string();
        assert!(s.contains("status=503"), "got: {}", s);
        assert!(
            s.contains("server-retry budget exhausted"),
            "exhausted 5xx must annotate; got: {}",
            s
        );

        let mid = PipelineFailure::new(
            "OpenAI",
            PipelinePhase::Request,
            1,
            3,
            "https://x",
            PipelineFailureKind::HttpStatus {
                status: 503,
                body: "Service Unavailable".into(),
            },
        );
        let s = mid.to_string();
        assert!(
            !s.contains("server-retry budget exhausted"),
            "mid-retry 5xx must not annotate yet; got: {}",
            s
        );
    }

    /// Spec: PipelineFailure converts via `?` from `TalkError`
    /// thanks to `#[from]`, and the resulting `TalkError`
    /// displays IDENTICALLY to the inner `PipelineFailure` (the
    /// `#[error(transparent)]` contract).  Confirms no extra
    /// "Configuration error:" / "Transcription error:" prefix
    /// sneaks in.
    #[test]
    fn talk_error_pipeline_display_is_transparent() {
        let pf = PipelineFailure::new(
            "OpenAI",
            PipelinePhase::Validate,
            1,
            5,
            "https://x",
            PipelineFailureKind::ModelRejected {
                model: "ghost".into(),
                suggestions: vec![],
            },
        );
        let pf_string = pf.to_string();
        let te: TalkError = pf.into();
        assert_eq!(
            te.to_string(),
            pf_string,
            "TalkError::Pipeline must be transparent"
        );
    }

    /// Spec: `TimerLabel::from_duration` formats whole seconds
    /// without a decimal point and sub-second durations with three
    /// decimals.  Same convention the legacy `fmt_budget` used.
    #[test]
    fn timer_label_formats_duration_compactly() {
        let t = TimerLabel::from_duration("connect_timeout", Duration::from_secs(2));
        assert_eq!(t.budget, "2s");
        let t = TimerLabel::from_duration("request_wall_clock", Duration::from_millis(50));
        assert_eq!(t.budget, "0.050s");
    }
}
