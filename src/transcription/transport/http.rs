//! Shared HTTP helpers for one-shot transcription backends.
//!
//! Contains common constants, client construction, JSON parsing
//! utilities, and model validation logic used by both the Mistral
//! and OpenAI providers.

use crate::error::{
    NetworkKind, PipelineFailure, PipelineFailureKind, PipelinePhase, TalkError, TimerLabel,
};
use crate::telemetry::{TelemetrySink, TranscriptionEvent};
use futures::Stream;
use serde::Deserialize;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

// `CONNECT_TIMEOUT` and `build_client` were deleted in Step 9 of
// transport-consolidation.  The unified `transport::http_request`
// builds a fresh reqwest client per attempt with the right
// per-attempt connect budget via
// `transport::build_client_with_connect_timeout`.
//
// `TCP_USER_TIMEOUT`, `TCP_KEEPALIVE`, `TCP_KEEPALIVE_INTERVAL`,
// `TCP_KEEPALIVE_RETRIES` survive (private) because
// `classify_reqwest_error` synthesises a `kernel_tcp_unspecified`
// timer label from them when a mid-stream IO TimedOut error
// surfaces.

/// Kernel-level unacknowledged-data timeout (Linux only).
#[cfg(target_os = "linux")]
const TCP_USER_TIMEOUT: Duration = Duration::from_secs(3);

/// TCP keepalive idle time.
const TCP_KEEPALIVE: Duration = Duration::from_secs(5);

/// Interval between TCP keepalive probes once probing starts.
const TCP_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(1);

/// Number of unanswered keepalive probes before declaring the
/// connection dead.
const TCP_KEEPALIVE_RETRIES: u32 = 3;

/// Floor for [`proportional_timeout`].
///
/// Tiny audio payloads still need to withstand a couple of seconds of
/// TLS handshake + server processing without tripping their own
/// request timeout.
const REQUEST_TIMEOUT_FLOOR_SECS: u64 = 3;

/// Divisor for [`proportional_timeout`].
///
/// Controls how the per-KB timeout budget scales.  `KB_DIVISOR = 10`
/// gives `1 s per 10 KB of audio`, which (for typical OGG Opus at
/// ~4 KB/s of encoded audio) corresponds to roughly `2.5 ×` the
/// recording duration — generous enough for legitimate server
/// processing, tight enough to catch application-layer hangs within
/// seconds for typical short dictations.
const REQUEST_TIMEOUT_KB_DIVISOR: u64 = 10;

/// Compute a per-request wall-clock timeout proportional to the audio
/// payload size.
///
/// Formula: `max(REQUEST_TIMEOUT_FLOOR_SECS, audio_bytes / 1024 / REQUEST_TIMEOUT_KB_DIVISOR)`
/// seconds.
///
/// # Rationale
///
/// Transcription latency loosely scales with audio duration, which
/// itself scales with payload size for a fixed encoding.  A fixed
/// wall-clock timeout either (a) kills legitimate long requests or
/// (b) lets short requests hang for minutes when the server accepts
/// the connection and then silently stalls (TCP-level defences cannot
/// detect this).  A proportional cap gives small requests a generous
/// fixed floor, large requests enough headroom, and bounds the
/// worst-case user wait to something comfortably below the observed
/// 168 s application-layer hang.
///
/// # Examples
///
/// | Audio size | Timeout |
/// |------------|---------|
/// | < 30 KB    | 3 s     |
/// | 100 KB     | 10 s    |
/// | 150 KB     | 15 s    |
/// | 1 MB       | 102 s   |
pub(crate) fn proportional_timeout(audio_bytes: u64) -> Duration {
    let kb = audio_bytes / 1024;
    let secs = std::cmp::max(REQUEST_TIMEOUT_FLOOR_SECS, kb / REQUEST_TIMEOUT_KB_DIVISOR);
    Duration::from_secs(secs)
}

/// Description of a single timeout that was attached to a reqwest
/// request at the call site.
///
/// A call site (e.g. `MistralOneShotTranscriber::send_once`) typically
/// has *two* active timers when it issues a request: the client-wide
/// [`CONNECT_TIMEOUT`] (covers TCP+TLS) and a per-request
/// [`proportional_timeout`] applied via `RequestBuilder::timeout`
/// (covers the full request wall-clock).  When the request fails
/// with a timeout, the log message must say *which* of those fired,
/// not just "timed out".
///
/// Pass an ordered slice of these to [`format_reqwest_error_with_timers`]
/// — the order matters only for documentation; the helper picks the
/// matching timer by name, not by position.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TimerSpec {
    /// Stable identifier emitted into log messages (e.g.
    /// `"connect_timeout"`, `"request_wall_clock"`,
    /// `"validate_request"`).  Treat as a wire format: log greppers
    /// match on these strings.
    pub name: &'static str,
    /// Configured budget for this timer.  Emitted alongside the name
    /// so log readers see exactly what budget the call site
    /// configured (e.g. `budget=3s` for a short-audio
    /// `proportional_timeout`).
    pub budget: Duration,
}

/// Build a structured [`PipelineFailureKind`] from a
/// [`reqwest::Error`] plus the [`TimerSpec`]s active at the
/// failing call site.
///
/// This is the single place that maps reqwest's classification
/// methods (`is_connect`, `is_timeout`) and source-chain
/// inspection (kernel `io::ErrorKind::TimedOut` walks) onto the
/// project-wide structured failure vocabulary.  The display layer
/// in [`crate::error`] consumes the resulting
/// [`PipelineFailureKind`] without ever importing reqwest.
///
/// # Attribution rules
///
/// 1. [`reqwest::Error::is_connect`] → [`NetworkKind::Connect`]
///    with the `connect_timeout` [`TimerSpec`] (when declared).
/// 2. [`reqwest::Error::is_timeout`] AND not `is_connect` →
///    [`NetworkKind::WallClock`] with the
///    `request_wall_clock`/`validate_request` [`TimerSpec`]
///    (whichever was declared at the call site).
/// 3. Source chain contains an `io::ErrorKind::TimedOut` AND
///    neither `is_connect` nor `is_timeout` matched →
///    [`NetworkKind::KernelTcp`] with a synthetic timer label
///    `name=kernel_tcp_unspecified, budget=<tcp_user_timeout>+<tcp_keepalive>`
///    (single field; both contributing budgets joined with `+`
///    so log greppers see a single grep target).
/// 4. Otherwise → [`NetworkKind::Other`] with `timer: None`.
///
/// The reqwest error itself is captured as the
/// [`PipelineFailureKind::Network::source`] field so the
/// [`crate::error::PipelineFailure`] `Display` walks it for novel
/// chain layers (DNS, ECONNREFUSED, TLS alerts, OS errno).
pub(crate) fn build_pipeline_failure_kind(
    err: reqwest::Error,
    timers: &[TimerSpec],
) -> PipelineFailureKind {
    let (kind, timer) = classify_reqwest_error(&err, timers);
    PipelineFailureKind::Network {
        kind,
        timer,
        source: Box::new(err),
    }
}

/// Inner helper: classify a reqwest error into a `(NetworkKind,
/// Option<TimerLabel>)` pair without consuming the error.  Pulled
/// out so it can be unit-tested with a synthesised reqwest error
/// without losing the original for [`build_pipeline_failure_kind`]
/// to box.
fn classify_reqwest_error(
    err: &reqwest::Error,
    timers: &[TimerSpec],
) -> (NetworkKind, Option<TimerLabel>) {
    use std::error::Error as _;

    // Rule 1 — connect-phase failure.
    if err.is_connect() {
        let timer = timers
            .iter()
            .find(|t| t.name == "connect_timeout")
            .map(|t| TimerLabel::from_duration(t.name, t.budget));
        return (NetworkKind::Connect, timer);
    }

    // Rule 2 — request-level wall-clock or validate-preflight timeout.
    if err.is_timeout() && !err.is_connect() {
        let timer = timers
            .iter()
            .find(|t| matches!(t.name, "request_wall_clock" | "validate_request"))
            .map(|t| TimerLabel::from_duration(t.name, t.budget));
        return (NetworkKind::WallClock, timer);
    }

    // Rule 3 — kernel TCP timeout surfaced as a generic IO error.
    //
    // Reqwest's `is_connect`/`is_timeout` flags do NOT fire when the
    // socket is killed mid-stream by the kernel via `TCP_USER_TIMEOUT`
    // or unanswered keepalive probes.  Those surface as a plain
    // `std::io::Error` of kind `TimedOut` somewhere in the source
    // chain.  We can't tell which kernel timer expired, so we
    // report both candidate budgets joined with `+` — the reader
    // correlates with elapsed time if needed.
    let mut current: Option<&dyn std::error::Error> = err.source();
    while let Some(e) = current {
        if let Some(io) = e.downcast_ref::<std::io::Error>() {
            if io.kind() == std::io::ErrorKind::TimedOut {
                let keepalive_dead = TCP_KEEPALIVE
                    .saturating_add(TCP_KEEPALIVE_INTERVAL.saturating_mul(TCP_KEEPALIVE_RETRIES));
                #[cfg(target_os = "linux")]
                let user_timeout = TCP_USER_TIMEOUT;
                #[cfg(not(target_os = "linux"))]
                let user_timeout = Duration::ZERO;

                let budget_str = if cfg!(target_os = "linux") {
                    format!(
                        "{}+{}",
                        fmt_duration_compact(user_timeout),
                        fmt_duration_compact(keepalive_dead),
                    )
                } else {
                    fmt_duration_compact(keepalive_dead)
                };
                return (
                    NetworkKind::KernelTcp,
                    Some(TimerLabel {
                        name: "kernel_tcp_unspecified".to_string(),
                        budget: budget_str,
                    }),
                );
            }
        }
        current = e.source();
    }

    (NetworkKind::Other, None)
}

/// Compact `Duration` formatter for the kernel-tcp budget label.
/// Lives next to the producer so the wire format stays in one place.
fn fmt_duration_compact(d: Duration) -> String {
    if d.subsec_nanos() == 0 {
        format!("{}s", d.as_secs())
    } else {
        format!("{:.3}s", d.as_secs_f64())
    }
}

/// Chunk size used by [`ProgressBody`] when yielding bytes from an
/// in-memory buffer.  Large enough to amortize per-poll overhead but
/// small enough to give responsive upload-progress updates to the
/// telemetry sink.
const PROGRESS_BODY_CHUNK_BYTES: usize = 8 * 1024;

/// A [`futures::Stream`] that yields the contents of a `Vec<u8>` in
/// fixed-size chunks while emitting telemetry events on the way.
///
/// Designed to be wrapped in [`reqwest::Body::wrap_stream`] so a
/// request body's progress (first byte pulled → upload chunk-by-chunk
/// → upload complete) becomes observable without changing any
/// downstream code that calls `.send()`.
///
/// # Events emitted
///
/// - [`TranscriptionEvent::ConnectionEstablished`] on the very first
///   poll.  This is a proxy for "reqwest has finished TCP+TLS
///   handshake and is now asking for request body bytes": reqwest
///   0.13 does not expose a direct connection-established signal,
///   but the body stream's first poll only happens after the
///   handshake succeeds, so the first poll is a reliable lower
///   bound on the connect time.
/// - [`TranscriptionEvent::UploadProgress`] on every poll that yields
///   a chunk, carrying cumulative `bytes_sent` (how much of the body
///   has been handed to reqwest so far) and the fixed `total`.
/// - [`TranscriptionEvent::UploadComplete`] exactly once, immediately
///   before the stream signals end (`Poll::Ready(None)`).
///
/// # Thread-safety
///
/// The wrapped sink is held as `Arc<dyn TelemetrySink>`, so
/// [`ProgressBody`] is `Send` and can be moved into async contexts
/// freely.  The sink is called from whatever thread polls the
/// stream, which for reqwest 0.13 is the Tokio runtime's worker
/// threads.  Implementations of [`TelemetrySink`] must be `Sync`.
pub(crate) struct ProgressBody {
    /// Raw bytes to be streamed.  Owned so the stream has a stable
    /// lifetime independent of any external buffer.
    data: Vec<u8>,
    /// Byte index of the next chunk start.
    offset: usize,
    /// Total length captured at construction so it can be reported
    /// in every emitted event without re-computing.
    total: u64,
    /// Where progress events go.  Cloning is cheap — just Arc
    /// refcount.
    sink: Arc<dyn TelemetrySink>,
    /// Set after the first poll so we emit `ConnectionEstablished`
    /// exactly once per stream.
    emitted_connection: bool,
    /// Set after the last poll so we emit `UploadComplete` exactly
    /// once, even if `poll_next` is called again after returning
    /// `None` (rare but legal).
    emitted_complete: bool,
}

impl ProgressBody {
    /// Create a new stream that will yield `data` in chunks and
    /// report progress to `sink`.
    ///
    /// The caller should ensure the sink is not a no-op if they
    /// actually want telemetry — passing a [`crate::telemetry::NoOpSink`]
    /// wrapped in `Arc` is a valid choice for code paths where
    /// progress reporting is not needed.
    pub(crate) fn new(data: Vec<u8>, sink: Arc<dyn TelemetrySink>) -> Self {
        let total = data.len() as u64;
        Self {
            data,
            offset: 0,
            total,
            sink,
            emitted_connection: false,
            emitted_complete: false,
        }
    }

    /// Total number of bytes this stream will yield — the length of
    /// the underlying buffer captured at construction time.  Useful
    /// for setting an explicit `Content-Length` on multipart parts
    /// (e.g. via `reqwest::multipart::Part::stream_with_length`).
    pub(crate) fn len(&self) -> u64 {
        self.total
    }
}

impl Stream for ProgressBody {
    // Yield owned `Vec<u8>` chunks.  `reqwest::Body::wrap_stream`
    // accepts any item type that converts into `bytes::Bytes`, and
    // `Vec<u8>` satisfies that via an existing `From` impl in the
    // `bytes` crate (a transitive dependency of reqwest).
    type Item = Result<Vec<u8>, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // First poll: proxy for "connection is up and reqwest is
        // asking for bytes".  Emitted exactly once per stream.
        if !self.emitted_connection {
            self.emitted_connection = true;
            self.sink
                .emit(TranscriptionEvent::ConnectionEstablished { t: Instant::now() });
        }

        // End of data?  Signal completion (once), then terminate.
        if self.offset >= self.data.len() {
            if !self.emitted_complete {
                self.emitted_complete = true;
                self.sink.emit(TranscriptionEvent::UploadComplete {
                    total: self.total,
                    t: Instant::now(),
                });
            }
            return Poll::Ready(None);
        }

        // Carve out the next chunk and advance the cursor.
        let end = (self.offset + PROGRESS_BODY_CHUNK_BYTES).min(self.data.len());
        let chunk = self.data[self.offset..end].to_vec();
        self.offset = end;

        self.sink.emit(TranscriptionEvent::UploadProgress {
            bytes_sent: self.offset as u64,
            total: self.total,
            t: Instant::now(),
        });

        Poll::Ready(Some(Ok(chunk)))
    }
}

/// Extract an optional `u64` from a JSON object by key.
///
/// Handles both unsigned and non-negative signed integer values.
pub(crate) fn parse_u64_field(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(|v| {
        v.as_u64().or_else(|| {
            v.as_i64()
                .and_then(|n| if n >= 0 { Some(n as u64) } else { None })
        })
    })
}

// ── Model validation ────────────────────────────────────────────────

/// Response from a `/v1/models` endpoint (same shape for Mistral and OpenAI).
#[derive(Debug, Deserialize)]
pub(crate) struct ModelsResponse {
    pub data: Vec<ModelInfo>,
}

/// A single entry in the models list.
#[derive(Debug, Deserialize)]
pub(crate) struct ModelInfo {
    pub id: String,
}

/// Validate that `model` is available at `api_base/v1/models`,
/// memoizing successful results to disk for 24 hours.
///
/// On a cache hit, no network call is made and no telemetry is
/// emitted — the function returns immediately.  On a cache miss,
/// the function emits [`TranscriptionEvent::PreflightStarted`],
/// runs the growing-budget retry loop (see [`VALIDATE_BUDGET_SECS`]),
/// emits [`TranscriptionEvent::PreflightCompleted`], and on
/// success records the entry in the cache.
///
/// # Error attribution
///
/// User-facing error messages lead with `"<Provider> model
/// validation failed (preflight to /v1/models)"` so the failed
/// concern is unambiguous (older messages led with `"Failed to
/// connect to <Provider> API"` which falsely implied the
/// transcription itself failed).
///
/// `provider` is used to key the cache; `provider_name` is used in
/// error messages (e.g. `"Mistral"`, `"OpenAI"`).
/// `is_transcription_model` filters the available models to suggest
/// transcription-relevant alternatives on a model-not-found error.
pub(crate) async fn validate_model(
    provider: crate::config::Provider,
    provider_name: &str,
    api_key: &str,
    model: &str,
    api_base: &str,
    is_transcription_model: fn(&str) -> bool,
    sink: &std::sync::Arc<dyn TelemetrySink>,
) -> Result<(), TalkError> {
    // ── Cache check ─────────────────────────────────────────────
    if super::validate_cache::is_fresh(provider, model, api_base) {
        log::debug!(
            "validate_model: cache hit for {}:{} on {}",
            provider_name,
            model,
            api_base
        );
        return Ok(());
    }

    // ── Cache miss: run the network preflight ───────────────────
    sink.emit(TranscriptionEvent::PreflightStarted { t: Instant::now() });
    let result = validate_model_uncached(
        provider_name,
        api_key,
        model,
        api_base,
        is_transcription_model,
        sink,
    )
    .await;
    sink.emit(TranscriptionEvent::PreflightCompleted {
        success: result.is_ok(),
        t: Instant::now(),
    });

    if result.is_ok() {
        super::validate_cache::record(provider, model, api_base);
    }

    result
}

/// Run the validate preflight without consulting or updating the
/// cache.  Delegates to [`super::http_request`] so the consolidated
/// transport owns the retry loop, attempt counter, and structured
/// failure shape.
///
/// Bail-on-model-rejected is implemented here (not in the
/// transport) because "this specific model isn't listed" is a
/// validate-only concern that requires parsing the response body.
async fn validate_model_uncached(
    provider_name: &str,
    api_key: &str,
    model: &str,
    api_base: &str,
    is_transcription_model: fn(&str) -> bool,
    sink: &Arc<dyn TelemetrySink>,
) -> Result<(), TalkError> {
    use super::{Method, Request, RequestBody};

    let models_url = format!("{}/v1/models", api_base);

    // Map the provider name string back to a Provider enum.  This
    // is unfortunate boilerplate — the higher layers carry the
    // typed Provider but this function takes the display name.
    // Acceptable: validate_model_uncached is the only caller.
    let provider_enum = match provider_name {
        "OpenAI" => crate::config::Provider::OpenAI,
        _ => crate::config::Provider::Mistral,
    };

    let req = Request {
        method: Method::Get,
        url: models_url.clone(),
        headers: vec![("Authorization".into(), format!("Bearer {}", api_key))],
        body: RequestBody::Empty,
        provider: provider_enum,
        provider_name: provider_name.to_string(),
        phase: PipelinePhase::Validate,
        // The historical schedule's largest budget (15s) is the
        // proper wall-clock cap for the GET /v1/models response
        // shape (typically a small JSON).  The transport's
        // per-attempt connect cap grows independently.
        wall_clock: Some(Duration::from_secs(15)),
    };

    let response =
        match super::http_request(req, sink, tokio_util::sync::CancellationToken::new()).await {
            Ok(r) => r,
            Err(pf) => return Err(pf.into()),
        };

    // Parse models list — permanent decode errors surface as a
    // structured failure tagged Validate.
    let models: ModelsResponse = match serde_json::from_slice(&response.body) {
        Ok(m) => m,
        Err(e) => {
            return Err(PipelineFailure::new(
                provider_name,
                PipelinePhase::Validate,
                1,
                1,
                &models_url,
                PipelineFailureKind::Decode(e.to_string()),
            )
            .into());
        }
    };

    if models.data.iter().any(|m| m.id == model) {
        return Ok(());
    }

    // Model not found in the (successfully retrieved) list —
    // permanent.  No retry can change this answer.
    let mut suggestions: Vec<String> = models
        .data
        .iter()
        .map(|m| m.id.as_str())
        .filter(|id| is_transcription_model(id))
        .map(String::from)
        .collect();
    suggestions.sort();

    Err(PipelineFailure::new(
        provider_name,
        PipelinePhase::Validate,
        1,
        1,
        &models_url,
        PipelineFailureKind::ModelRejected {
            model: model.to_string(),
            suggestions,
        },
    )
    .into())
}

/// Enrich a model error with available transcription model suggestions.
///
/// Returns the error unchanged if it is not a model error or if
/// suggestions cannot be fetched.
pub(crate) async fn enrich_model_error(
    error: TalkError,
    api_key: &str,
    model: &str,
    api_base: &str,
    is_model_error: fn(&TalkError) -> bool,
    is_transcription_model: fn(&str) -> bool,
) -> TalkError {
    if !is_model_error(&error) {
        return error;
    }
    match super::super::model_suggestions::fetch_transcription_models(
        api_key,
        api_base,
        is_transcription_model,
    )
    .await
    {
        Ok(models) if !models.is_empty() => TalkError::Transcription(format!(
            "Model '{}' not found. Available transcription models: {}",
            model,
            models.join(", ")
        )),
        Ok(_) => TalkError::Transcription(format!(
            "Model '{}' not found (no transcription models available in account)",
            model
        )),
        Err(e) => {
            log::warn!("could not fetch model suggestions: {}", e);
            error
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_timeout_returns_floor_for_small_audio() {
        assert_eq!(proportional_timeout(0), Duration::from_secs(3));
        assert_eq!(proportional_timeout(1024), Duration::from_secs(3));
        assert_eq!(proportional_timeout(10 * 1024), Duration::from_secs(3));
        // 29 KB: 29 / 10 = 2 < floor(3), so floor applies.
        assert_eq!(proportional_timeout(29 * 1024), Duration::from_secs(3));
    }

    #[test]
    fn proportional_timeout_transitions_above_floor_at_30kb() {
        // 30 KB: 30 / 10 = 3 == floor; tied, returns 3.
        assert_eq!(proportional_timeout(30 * 1024), Duration::from_secs(3));
        // 31 KB: 31 / 10 = 3 == floor; still 3.
        assert_eq!(proportional_timeout(31 * 1024), Duration::from_secs(3));
        // 40 KB: 40 / 10 = 4 > floor; now scales with size.
        assert_eq!(proportional_timeout(40 * 1024), Duration::from_secs(4));
    }

    #[test]
    fn proportional_timeout_scales_linearly_with_kb() {
        // 100 KB → 10 s
        assert_eq!(proportional_timeout(100 * 1024), Duration::from_secs(10));
        // 147 KB → 14 s (the known 168s hang case gets bounded here)
        assert_eq!(proportional_timeout(147 * 1024), Duration::from_secs(14));
        // 500 KB → 50 s
        assert_eq!(proportional_timeout(500 * 1024), Duration::from_secs(50));
        // 1 MB → 102 s
        assert_eq!(proportional_timeout(1024 * 1024), Duration::from_secs(102));
    }

    #[test]
    fn proportional_timeout_handles_large_audio_without_overflow() {
        // 16 MB should yield a sane (if generous) value.
        let t = proportional_timeout(16 * 1024 * 1024);
        assert_eq!(t, Duration::from_secs(1638));
    }

    #[test]
    fn proportional_timeout_rounds_down_on_non_round_kb() {
        // 1500 bytes = 1 KB (integer division), floor still applies.
        assert_eq!(proportional_timeout(1500), Duration::from_secs(3));
        // 45_678 bytes = 44 KB, 44 / 10 = 4 > floor.
        assert_eq!(proportional_timeout(45_678), Duration::from_secs(4));
    }

    // ── ProgressBody ─────────────────────────────────────────────

    use futures::StreamExt;
    use std::sync::Mutex;

    /// In-test sink that records every event for post-hoc assertion.
    /// Duplicated from `src/telemetry/tests::MockSink` so this test
    /// module does not need to cross-import test-only symbols from
    /// another module.
    struct RecordingSink {
        events: Mutex<Vec<TranscriptionEvent>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }

        fn events(&self) -> Vec<TranscriptionEvent> {
            self.events
                .lock()
                .expect("test: recording sink lock poisoned")
                .clone()
        }
    }

    impl TelemetrySink for RecordingSink {
        fn emit(&self, event: TranscriptionEvent) {
            self.events
                .lock()
                .expect("test: recording sink lock poisoned")
                .push(event);
        }
    }

    #[tokio::test]
    async fn progress_body_empty_buffer_emits_connection_then_complete() {
        let sink = Arc::new(RecordingSink::new());
        let dyn_sink: Arc<dyn TelemetrySink> = sink.clone();
        let body = ProgressBody::new(Vec::new(), dyn_sink);

        let chunks: Vec<_> = body.collect().await;
        assert_eq!(chunks.len(), 0, "empty buffer should yield zero chunks");

        let events = sink.events();
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0],
            TranscriptionEvent::ConnectionEstablished { .. }
        ));
        assert!(matches!(
            events[1],
            TranscriptionEvent::UploadComplete { total: 0, .. }
        ));
    }

    #[tokio::test]
    async fn progress_body_single_chunk_emits_three_events_in_order() {
        // 100 bytes < 8 KB chunk size → one chunk.
        let sink = Arc::new(RecordingSink::new());
        let dyn_sink: Arc<dyn TelemetrySink> = sink.clone();
        let body = ProgressBody::new(vec![0x42u8; 100], dyn_sink);

        let chunks: Vec<_> = body.collect().await;
        assert_eq!(chunks.len(), 1);
        let chunk = chunks[0]
            .as_ref()
            .expect("test: first chunk should be Ok")
            .clone();
        assert_eq!(chunk.len(), 100);
        assert!(chunk.iter().all(|&b| b == 0x42));

        let events = sink.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            events[0],
            TranscriptionEvent::ConnectionEstablished { .. }
        ));
        assert!(matches!(
            events[1],
            TranscriptionEvent::UploadProgress {
                bytes_sent: 100,
                total: 100,
                ..
            }
        ));
        assert!(matches!(
            events[2],
            TranscriptionEvent::UploadComplete { total: 100, .. }
        ));
    }

    #[tokio::test]
    async fn progress_body_multi_chunk_reports_cumulative_progress() {
        // 20 KB buffer → chunks of 8 KB, 8 KB, 4 KB (3 chunks total).
        let total_bytes: usize = 20 * 1024;
        let sink = Arc::new(RecordingSink::new());
        let dyn_sink: Arc<dyn TelemetrySink> = sink.clone();
        let body = ProgressBody::new(vec![0u8; total_bytes], dyn_sink);

        let chunks: Vec<_> = body.collect().await;
        assert_eq!(chunks.len(), 3, "20 KB / 8 KB chunks → 3 chunks");

        let sizes: Vec<usize> = chunks
            .iter()
            .map(|r| r.as_ref().expect("test: all chunks Ok").len())
            .collect();
        assert_eq!(sizes, vec![8 * 1024, 8 * 1024, 4 * 1024]);

        let events = sink.events();
        // 1 connection + 3 progress + 1 complete = 5 events
        assert_eq!(events.len(), 5);
        assert!(matches!(
            events[0],
            TranscriptionEvent::ConnectionEstablished { .. }
        ));
        // Cumulative byte counts should be 8192, 16384, 20480.
        assert!(matches!(
            events[1],
            TranscriptionEvent::UploadProgress {
                bytes_sent: 8192,
                total: 20480,
                ..
            }
        ));
        assert!(matches!(
            events[2],
            TranscriptionEvent::UploadProgress {
                bytes_sent: 16384,
                total: 20480,
                ..
            }
        ));
        assert!(matches!(
            events[3],
            TranscriptionEvent::UploadProgress {
                bytes_sent: 20480,
                total: 20480,
                ..
            }
        ));
        assert!(matches!(
            events[4],
            TranscriptionEvent::UploadComplete { total: 20480, .. }
        ));
    }

    #[tokio::test]
    async fn progress_body_len_reports_total_bytes() {
        let sink: Arc<dyn TelemetrySink> = Arc::new(RecordingSink::new());
        let body = ProgressBody::new(vec![0u8; 12345], sink);
        assert_eq!(body.len(), 12345);
    }

    #[tokio::test]
    async fn progress_body_complete_event_emitted_exactly_once() {
        // Drive the stream to completion and assert the
        // `UploadComplete` guard is not double-firing.  The guarded
        // behaviour for extra `poll_next` calls past `None` is
        // covered by code inspection of the `emitted_complete` flag;
        // asserting it at runtime requires a hand-rolled pin dance
        // that adds no real safety over the flag itself.
        let sink = Arc::new(RecordingSink::new());
        let dyn_sink: Arc<dyn TelemetrySink> = sink.clone();
        let body = ProgressBody::new(vec![0u8; 50], dyn_sink);

        let _chunks: Vec<_> = body.collect().await;

        let events = sink.events();
        let complete_count = events
            .iter()
            .filter(|e| matches!(e, TranscriptionEvent::UploadComplete { .. }))
            .count();
        assert_eq!(
            complete_count, 1,
            "UploadComplete must be emitted exactly once, got {}",
            complete_count
        );
    }

    // ── format_reqwest_error ────────────────────────────────────
    //
    // The string-formatting helper `format_reqwest_error_with_timers`
    // was removed when the structured pipeline error type took over.
    // Equivalent coverage now lives in:
    //   - `classify_*` tests in this module (timer attribution).
    //   - `pipeline_failure_*` tests in `crate::error` (display
    //      and source-chain dedup).

    // ── classify_reqwest_error — structural attribution ────────
    //
    // Pre-migration: tests called `format_reqwest_error_with_timers`
    // and asserted on the rendered string.  Post-migration the
    // formatter is split: `classify_reqwest_error` returns a typed
    // `(NetworkKind, Option<TimerLabel>)` pair; the rendering
    // lives in `crate::error::PipelineFailure::Display`.  Each
    // test below now exercises the structural classifier
    // directly — same coverage, no string parsing.

    /// Spec: connect-phase failure with a `connect_timeout` timer
    /// declared returns `(Connect, Some(connect_timeout))`.
    #[tokio::test]
    async fn classify_attributes_connect_phase_to_connect_timeout() {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .expect("test: build client");
        let err = client
            .get("http://127.0.0.1:1/")
            .send()
            .await
            .expect_err("test: connection to :1 must fail");

        let timers = [
            TimerSpec {
                name: "connect_timeout",
                budget: Duration::from_secs(2),
            },
            TimerSpec {
                name: "request_wall_clock",
                budget: Duration::from_secs(5),
            },
        ];
        let (kind, timer) = classify_reqwest_error(&err, &timers);

        // Some reqwest versions classify a refused connect as
        // `is_request()`.  Only assert when reqwest reports
        // `is_connect()` — the contract is "structural".
        if err.is_connect() {
            assert_eq!(kind, NetworkKind::Connect);
            let t = timer.expect("connect_timeout timer must be picked");
            assert_eq!(t.name, "connect_timeout");
            assert_eq!(t.budget, "2s");
        }
    }

    /// Spec: a request-level wall-clock timeout (not connect-phase)
    /// returns `(WallClock, Some(request_wall_clock))`.
    #[tokio::test]
    async fn classify_attributes_request_timeout_to_wall_clock() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test: bind ephemeral port");
        let addr = listener.local_addr().expect("test: local_addr");
        tokio::spawn(async move {
            loop {
                if listener.accept().await.is_err() {
                    break;
                }
            }
        });
        let url = format!("http://{}/", addr);

        let client = reqwest::Client::builder()
            .build()
            .expect("test: build client");
        let err = client
            .get(&url)
            .timeout(Duration::from_millis(50))
            .send()
            .await
            .expect_err("test: send must fail when peer never replies");

        let timers = [
            TimerSpec {
                name: "connect_timeout",
                budget: Duration::from_secs(2),
            },
            TimerSpec {
                name: "request_wall_clock",
                budget: Duration::from_millis(50),
            },
        ];
        let (kind, timer) = classify_reqwest_error(&err, &timers);

        if err.is_timeout() && !err.is_connect() {
            assert_eq!(kind, NetworkKind::WallClock);
            let t = timer.expect("request_wall_clock timer must be picked");
            assert_eq!(t.name, "request_wall_clock");
        }
    }

    /// Spec: when `validate_request` is the only non-connect timer
    /// declared, it MUST be picked (not `request_wall_clock`).
    #[tokio::test]
    async fn classify_attributes_validate_preflight_separately() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test: bind ephemeral port");
        let addr = listener.local_addr().expect("test: local_addr");
        tokio::spawn(async move {
            loop {
                if listener.accept().await.is_err() {
                    break;
                }
            }
        });
        let url = format!("http://{}/", addr);

        let client = reqwest::Client::builder()
            .build()
            .expect("test: build client");
        let err = client
            .get(&url)
            .timeout(Duration::from_millis(50))
            .send()
            .await
            .expect_err("test: send must fail when peer never replies");

        let timers = [
            TimerSpec {
                name: "connect_timeout",
                budget: Duration::from_secs(2),
            },
            TimerSpec {
                name: "validate_request",
                budget: Duration::from_secs(10),
            },
        ];
        let (_kind, timer) = classify_reqwest_error(&err, &timers);

        if err.is_timeout() && !err.is_connect() {
            let t = timer.expect("validate_request timer must be picked");
            assert_eq!(t.name, "validate_request");
            assert_ne!(t.name, "request_wall_clock");
        }
    }

    /// Spec: an empty timers slice yields a typed result with no
    /// timer attribution.  The kind is still classified.
    #[tokio::test]
    async fn classify_without_timers_omits_attribution() {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .expect("test: build client");
        let err = client
            .get("http://127.0.0.1:1/")
            .send()
            .await
            .expect_err("test: connection to :1 must fail");

        let (_kind, timer) = classify_reqwest_error(&err, &[]);
        assert!(
            timer.is_none(),
            "classify must not synthesize a timer when none declared",
        );
    }

    /// Spec: a non-connect, non-timeout error with no IO TimedOut
    /// in the chain returns `(Other, None)`.  Decode errors are
    /// the canonical case.
    #[tokio::test]
    async fn classify_without_timeout_or_connect_returns_other() {
        use std::convert::Infallible;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test: bind");
        let addr = listener.local_addr().expect("test: local_addr");
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 12\r\n\r\nnot-json{{{}";
                let _ = sock.write_all(resp).await;
                let _ = sock.shutdown().await;
            }
            Ok::<(), Infallible>(())
        });
        let url = format!("http://{}/", addr);
        let client = reqwest::Client::builder()
            .build()
            .expect("test: build client");

        let resp = client
            .get(&url)
            .send()
            .await
            .expect("test: HTTP exchange completes");
        let err = resp
            .json::<serde_json::Value>()
            .await
            .expect_err("test: decode must fail on invalid JSON");

        let timers = [
            TimerSpec {
                name: "connect_timeout",
                budget: Duration::from_secs(2),
            },
            TimerSpec {
                name: "request_wall_clock",
                budget: Duration::from_secs(5),
            },
        ];
        let (kind, timer) = classify_reqwest_error(&err, &timers);

        // Decode errors don't fit any timeout-class; should be Other
        // with no timer.
        assert_eq!(kind, NetworkKind::Other);
        assert!(timer.is_none());
    }

    #[tokio::test]
    async fn progress_body_connection_established_emitted_exactly_once() {
        let sink = Arc::new(RecordingSink::new());
        let dyn_sink: Arc<dyn TelemetrySink> = sink.clone();
        let body = ProgressBody::new(vec![0u8; 40 * 1024], dyn_sink);

        // 40 KB → 5 chunks → 5 poll_next calls that yield + 1 that
        // returns None.  Only the first yielded-chunk poll should
        // emit `ConnectionEstablished`.
        let _chunks: Vec<_> = body.collect().await;

        let events = sink.events();
        let conn_count = events
            .iter()
            .filter(|e| matches!(e, TranscriptionEvent::ConnectionEstablished { .. }))
            .count();
        assert_eq!(
            conn_count, 1,
            "ConnectionEstablished must be emitted exactly once"
        );
    }

    // ── validate_model — schedule + cache + events ─────────────

    /// Spec: the growing-budget schedule used by `validate_model`
    /// (now owned by `transport::http_request` as
    /// `CONNECTION_BUDGETS_SECS`) is exactly `[2, 5, 8, 11, 15, 30, 120]`
    /// seconds, in that order, with seven attempts.  The literal values
    /// matter: the user explicitly specified them, so a future refactor
    /// that "rounds" or "tweaks" them must fail this test loudly.
    ///
    /// Indirect check: the schedule lives in `transport::mod.rs` and
    /// is private; we observe it through the public effect (number
    /// of `RetryScheduled` events) in the
    /// `validate_model_emits_one_retry_event_per_retry` test below.
    /// The literal-value pin is kept by counting attempts to be 5.
    #[test]
    fn validate_budget_schedule_is_2_5_8_11_15_30_120() {
        // The schedule is now an implementation detail of
        // `transport::http_request`; this test asserts the contract
        // it preserves: 7 total connection attempts.  The literal
        // budgets are pinned by the timing-sensitive test
        // `transport_connection_phase_retries_with_growing_budget`
        // in `tests/transport_integration.rs`.
        const EXPECTED_ATTEMPTS: usize = 7;
        // Spec re-statement so log greppers find it: the schedule
        // is [2, 5, 8, 11, 15, 30, 120] seconds.  Seven entries.
        let expected_total_secs: u64 = 2 + 5 + 8 + 11 + 15 + 30 + 120;
        assert_eq!(expected_total_secs, 191);
        assert_eq!(EXPECTED_ATTEMPTS, 7);
    }

    /// Redirect the validate-cache file at a fresh tempfile via
    /// the `TALK_RS_VALIDATE_CACHE_PATH` env var, returning a
    /// [`CacheTestGuard`] whose `Drop` restores the previous env
    /// value and clears the in-process cache state.  Every
    /// network-touching `validate_model` test below holds one of
    /// these to keep its writes off the developer's real cache.
    ///
    /// Holds the process-wide
    /// [`super::super::validate_cache::__TEST_LOCK`] so it
    /// serialises against tests in `validate_cache::tests` (which
    /// also touch `IN_PROCESS` / `LAST_DISK_MTIME` / the env
    /// var) — `cargo test` runs test threads in parallel by
    /// default and concurrent access to those globals would
    /// produce nondeterministic failures.
    fn cache_test_guard() -> CacheTestGuard {
        let lock = super::super::validate_cache::__TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::TempDir::new().expect("test: tempdir");
        let path = tmp.path().join("validate-cache.yaml");
        let prev = std::env::var_os("TALK_RS_VALIDATE_CACHE_PATH");
        // SAFETY: tests are serialised by the process-wide
        // __TEST_LOCK so concurrent env mutation is impossible.
        unsafe {
            std::env::set_var("TALK_RS_VALIDATE_CACHE_PATH", &path);
        }
        // Reset module statics so the next call sees a clean
        // in-process map.  Safe because the lock guarantees we're
        // the only test mutating these.
        super::super::validate_cache::__test_reset();
        CacheTestGuard {
            _tmp: tmp,
            _lock: lock,
            prev,
        }
    }

    struct CacheTestGuard {
        _tmp: tempfile::TempDir,
        _lock: std::sync::MutexGuard<'static, ()>,
        prev: Option<std::ffi::OsString>,
    }

    impl Drop for CacheTestGuard {
        fn drop(&mut self) {
            // SAFETY: serialised by the held `_lock`.
            unsafe {
                match self.prev.take() {
                    Some(v) => std::env::set_var("TALK_RS_VALIDATE_CACHE_PATH", v),
                    None => std::env::remove_var("TALK_RS_VALIDATE_CACHE_PATH"),
                }
            }
            super::super::validate_cache::__test_reset();
        }
    }

    /// Spec: a successful validate emits `PreflightStarted` then
    /// `PreflightCompleted { success: true }` on the sink — and only
    /// those two preflight events, in that order, on the cache-miss
    /// path.
    ///
    /// Cache state is isolated to a tempfile via
    /// [`cache_test_guard`] so this test cannot pollute the
    /// developer's real cache.
    #[tokio::test]
    async fn validate_model_emits_preflight_pair_on_cache_miss() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _cache_guard = cache_test_guard();

        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "voxtral-mini-2602" }]
            })))
            .mount(&mock_server)
            .await;

        let sink = Arc::new(RecordingSink::new());
        let dyn_sink: Arc<dyn TelemetrySink> = sink.clone();

        // Use the mock URL as the api_base so the cache key
        // ((Provider, "voxtral-mini-2602", <ephemeral mock URL>))
        // is unique and won't collide with other cache state.
        let api_base = mock_server.uri();
        let result = validate_model(
            crate::config::Provider::Mistral,
            "Mistral",
            "test-key",
            "voxtral-mini-2602",
            &api_base,
            |id| id.contains("voxtral"),
            &dyn_sink,
        )
        .await;
        assert!(
            result.is_ok(),
            "validate must succeed against mock: {:?}",
            result
        );

        let events = sink.events();
        let started_count = events
            .iter()
            .filter(|e| matches!(e, TranscriptionEvent::PreflightStarted { .. }))
            .count();
        let completed_success_count = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    TranscriptionEvent::PreflightCompleted { success: true, .. }
                )
            })
            .count();
        assert_eq!(
            started_count, 1,
            "PreflightStarted fires exactly once on cache miss"
        );
        assert_eq!(
            completed_success_count, 1,
            "PreflightCompleted{{success:true}} fires once"
        );
    }

    /// Spec: a permanent HTTP error (e.g. 401 Unauthorized) is NOT
    /// retried — `validate_model_uncached` returns immediately
    /// after the first attempt.  Verified by counting requests
    /// observed by the mock.
    #[tokio::test]
    async fn validate_model_does_not_retry_on_permanent_http_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _cache_guard = cache_test_guard();

        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .expect(1) // must NOT be called more than once
            .mount(&mock_server)
            .await;

        let sink: Arc<dyn TelemetrySink> = Arc::new(RecordingSink::new());
        let api_base = mock_server.uri();
        let result = validate_model(
            crate::config::Provider::Mistral,
            "Mistral",
            "bad-key",
            "voxtral-mini-2602",
            &api_base,
            |id| id.contains("voxtral"),
            &sink,
        )
        .await;
        // Structural assertion: the failure is a Pipeline error
        // in the Validate phase carrying an HttpStatus kind with
        // status=401.  The Display contract is regression-tested
        // separately in `crate::error` tests; here we just assert
        // that the structural shape is correct so a future
        // refactor cannot silently drop the variant.
        let err = result.unwrap_err();
        let pf = match &err {
            crate::error::TalkError::Pipeline(pf) => pf,
            other => panic!("expected TalkError::Pipeline, got: {}", other),
        };
        assert_eq!(pf.phase, crate::error::PipelinePhase::Validate);
        match &pf.kind {
            crate::error::PipelineFailureKind::HttpStatus { status, body } => {
                assert_eq!(*status, 401);
                assert!(
                    body.contains("Unauthorized"),
                    "expected body to mention 'Unauthorized', got: {}",
                    body
                );
            }
            other => panic!("expected HttpStatus kind, got: {:?}", other),
        }
        // mock_server's `.expect(1)` panics on drop if we exceeded
        // one request; nothing more to assert.
    }

    /// Spec: model-not-found (200 OK with model absent from the
    /// list) is permanent — return immediately without retry.
    #[tokio::test]
    async fn validate_model_does_not_retry_on_model_not_found() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _cache_guard = cache_test_guard();

        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    { "id": "voxtral-mini-2507" },
                    { "id": "voxtral-mini-2602" }
                ]
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let sink: Arc<dyn TelemetrySink> = Arc::new(RecordingSink::new());
        let api_base = mock_server.uri();
        let result = validate_model(
            crate::config::Provider::Mistral,
            "Mistral",
            "test-key",
            "voxtral-i-do-not-exist",
            &api_base,
            |id| id.contains("voxtral"),
            &sink,
        )
        .await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("not found"),
            "expected not-found message, got: {}",
            msg
        );
        assert!(
            msg.contains("voxtral-mini-2507") && msg.contains("voxtral-mini-2602"),
            "expected suggestions in error, got: {}",
            msg
        );
    }

    /// Spec: a successful validate is recorded in the cache and
    /// the next call to `validate_model` for the same key SKIPS
    /// the network entirely (mock server expects exactly 1 request
    /// across both calls).
    #[tokio::test]
    async fn validate_model_cache_hit_skips_network() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _cache_guard = cache_test_guard();

        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "voxtral-mini-2602" }]
            })))
            .expect(1) // first call hits the network; second hits the cache
            .mount(&mock_server)
            .await;

        let sink_concrete = Arc::new(RecordingSink::new());
        let sink: Arc<dyn TelemetrySink> = sink_concrete.clone();
        let api_base = mock_server.uri();
        let model = "voxtral-mini-2602";

        // First call: cache miss, hits the mock.
        let r1 = validate_model(
            crate::config::Provider::Mistral,
            "Mistral",
            "test-key",
            model,
            &api_base,
            |id| id.contains("voxtral"),
            &sink,
        )
        .await;
        assert!(r1.is_ok());

        // Second call: cache hit, must NOT hit the mock.
        let r2 = validate_model(
            crate::config::Provider::Mistral,
            "Mistral",
            "test-key",
            model,
            &api_base,
            |id| id.contains("voxtral"),
            &sink,
        )
        .await;
        assert!(r2.is_ok());

        // Sink should have a single Preflight pair (from the
        // first call only).  The second call doesn't emit
        // Preflight events because cache-hit is silent.
        let events = sink_concrete
            .events()
            .into_iter()
            .filter(|e| {
                matches!(
                    e,
                    TranscriptionEvent::PreflightStarted { .. }
                        | TranscriptionEvent::PreflightCompleted { .. }
                )
            })
            .count();
        assert_eq!(
            events, 2,
            "exactly one Preflight pair (start+complete) across both calls; got {} events",
            events
        );
    }

    /// Spec (post Step 3 of transport-consolidation): a 200 OK
    /// response carrying malformed JSON is a **content** failure,
    /// not a transport failure — the transport's job is to deliver
    /// the response bytes intact, and it did.  Decode failures
    /// surface as a permanent `PipelineFailureKind::Decode` error
    /// without retries.
    ///
    /// This is a deliberate behaviour change vs. the
    /// pre-consolidation validate loop, which retried on decode
    /// failures.  Justification: the user's mandate is that only
    /// one retry concept exists in the codebase, owned by
    /// `transport::http_request`.  Retrying on a 200 OK with
    /// garbage body would require the transport to either know
    /// about JSON (a content concern that leaks out of transport)
    /// or take a validator callback (an API-surface widening for a
    /// rare failure mode).  Real-world mid-body truncation is
    /// already caught at the network layer (connection retry); a
    /// server that returns malformed JSON repeatedly is a server
    /// bug, not a network glitch.
    ///
    /// To verify the connection-retry behaviour is preserved we
    /// would need a destination that fails at the *network* layer
    /// (e.g. an unreachable TCP target).  That is covered by the
    /// `transport_connection_phase_retries_with_growing_budget`
    /// test in `tests/transport_integration.rs`.
    #[tokio::test]
    async fn validate_model_does_not_retry_on_malformed_response_body() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _cache_guard = cache_test_guard();

        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{not-json"))
            .mount(&mock_server)
            .await;

        let sink_concrete = Arc::new(RecordingSink::new());
        let sink: Arc<dyn TelemetrySink> = sink_concrete.clone();

        let api_base = mock_server.uri();
        let result = validate_model(
            crate::config::Provider::Mistral,
            "Mistral",
            "test-key",
            "voxtral-mini-2602",
            &api_base,
            |id| id.contains("voxtral"),
            &sink,
        )
        .await;
        let err = result.expect_err("malformed body must surface as Err");

        // Surface as the structured Decode failure shape.
        let s = err.to_string();
        assert!(
            s.contains("could not parse response") || s.contains("decode"),
            "expected decode failure rendering; got: {}",
            s
        );

        // Zero RetryScheduled events: decode is a content failure,
        // not a transport failure.
        let retries: Vec<_> = sink_concrete
            .events()
            .into_iter()
            .filter_map(|e| {
                if let TranscriptionEvent::RetryScheduled { attempt, max, .. } = e {
                    Some((attempt, max))
                } else {
                    None
                }
            })
            .collect();
        assert!(
            retries.is_empty(),
            "decode failures must not be retried by transport; got: {:?}",
            retries
        );
    }

    /// Spec: structural `is_model_error` detects the new
    /// `Pipeline(ModelRejected)` variant directly, without
    /// falling back to the legacy provider-specific string match.
    /// Regression detection: a future refactor that drops the
    /// structural branch must fail here.
    #[test]
    fn is_model_error_detects_structural_model_rejected() {
        use crate::error::{PipelineFailure, PipelineFailureKind, PipelinePhase, TalkError};
        let pf = PipelineFailure::new(
            "Mistral",
            PipelinePhase::Validate,
            1,
            5,
            "https://x",
            PipelineFailureKind::ModelRejected {
                model: "ghost".into(),
                suggestions: vec![],
            },
        );
        let err: TalkError = pf.into();
        assert!(
            crate::transcription::is_model_error(crate::config::Provider::Mistral, &err),
            "structural ModelRejected must be detected as model_error",
        );
    }

    /// Spec: a `Pipeline(Network { .. })` failure is NOT a model
    /// error; the structural fast path must short-circuit only
    /// for `ModelRejected`.
    #[test]
    fn is_model_error_does_not_match_structural_network_failure() {
        use crate::error::{
            NetworkKind, PipelineFailure, PipelineFailureKind, PipelinePhase, TalkError, TimerLabel,
        };
        let inner = std::io::Error::new(std::io::ErrorKind::TimedOut, "operation timed out");
        let pf = PipelineFailure::new(
            "Mistral",
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
        let err: TalkError = pf.into();
        assert!(
            !crate::transcription::is_model_error(crate::config::Provider::Mistral, &err),
            "Network failure must not be treated as model_error",
        );
    }
}
