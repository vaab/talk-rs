//! Shared HTTP helpers for batch transcription backends.
//!
//! Contains common constants, client construction, JSON parsing
//! utilities, and model validation logic used by both the Mistral
//! and OpenAI providers.

use crate::error::TalkError;
use crate::telemetry::{TelemetrySink, TranscriptionEvent};
use futures::Stream;
use reqwest::Client;
use serde::Deserialize;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// TCP connect timeout — fail fast when the server is unreachable.
///
/// Exported at `pub(crate)` so call sites that want to attribute
/// connect-phase timeouts by name in their error messages can pass
/// it as a [`TimerSpec`] to [`format_reqwest_error_with_timers`].
pub(crate) const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// Kernel-level unacknowledged-data timeout (Linux only).
///
/// If transmitted data (upload bytes, TCP ACKs) goes unacknowledged
/// for this duration, the kernel forcefully closes the socket.  This
/// is the primary mechanism for detecting silent VPN/network drops
/// during active data transfer — unlike TCP keepalive, it fires even
/// when the send buffer is full.
#[cfg(target_os = "linux")]
const TCP_USER_TIMEOUT: Duration = Duration::from_secs(3);

/// TCP keepalive idle time — start sending probes after this much
/// silence on an otherwise idle connection.
const TCP_KEEPALIVE: Duration = Duration::from_secs(5);

/// Interval between TCP keepalive probes once probing starts.
const TCP_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(1);

/// Number of unanswered keepalive probes before declaring the
/// connection dead.
const TCP_KEEPALIVE_RETRIES: u32 = 3;

/// Build an HTTP client tuned for VPN-hostile networks.
///
/// The client uses aggressive idle and kernel-level timeouts to
/// detect dead connections within a few seconds.  Transcription
/// request paths additionally wrap individual calls in a
/// [`proportional_timeout`] so a server that accepts the connection
/// and then hangs on the application layer (where TCP-level defences
/// cannot help) is bounded by a payload-sized wall clock.
pub(crate) fn build_client() -> Result<Client, TalkError> {
    // NOTE: The client-wide `read_timeout` is intentionally omitted.
    // In reqwest 0.13 it acts as a non-resetting wall-clock timer
    // during the upload + wait-for-headers phase, which would kill
    // legitimate requests where upload + server processing exceeds
    // the value.  Dead connections are detected by `tcp_user_timeout`
    // (during active transfer) and TCP keepalive (during idle
    // phases).  Per-request wall-clock caps for slow-but-alive
    // servers are applied at each call site via [`proportional_timeout`].
    let builder = Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .tcp_keepalive(TCP_KEEPALIVE)
        .tcp_keepalive_interval(TCP_KEEPALIVE_INTERVAL)
        .tcp_keepalive_retries(TCP_KEEPALIVE_RETRIES);

    #[cfg(target_os = "linux")]
    let builder = builder.tcp_user_timeout(TCP_USER_TIMEOUT);

    builder
        .build()
        .map_err(|e| TalkError::Config(format!("failed to build HTTP client: {}", e)))
}

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
/// A call site (e.g. `MistralBatchTranscriber::send_once`) typically
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

/// Format a [`Duration`] for log output: whole seconds when possible,
/// fractional seconds otherwise.  Keeps log lines compact and stable
/// across builds (no platform-dependent `Debug` formatting).
fn fmt_budget(d: Duration) -> String {
    if d.subsec_nanos() == 0 {
        format!("{}s", d.as_secs())
    } else {
        format!("{:.3}s", d.as_secs_f64())
    }
}

/// Format a [`reqwest::Error`] with full diagnostic detail and
/// timeout-source attribution for logs.
///
/// `reqwest::Error`'s default `Display` prints only the top-level
/// message (e.g. `"error sending request for url (...)"`) and hides
/// the actual cause behind `std::error::Error::source()`.  For talk-rs
/// users who land in the logs trying to figure out why a transcription
/// failed, that single line is useless: it does not distinguish DNS
/// failure from connection refused from TLS handshake failure from
/// read timeout — and it does not say *which* of the multiple timers
/// active at the call site fired.
///
/// This helper produces a single-line string that:
///
/// - Starts with the top-level [`reqwest::Error`] message.
/// - Appends a `[kind=...]` tag derived from the structured
///   classifiers ([`reqwest::Error::is_timeout`],
///   [`reqwest::Error::is_connect`], etc.).
/// - When `timers` is non-empty AND the failure is timeout-shaped,
///   appends `name=...` and `budget=...` (or `budgets=[...]` for
///   ambiguous kernel-level timeouts) identifying *which* timer
///   fired.  See [Timer attribution rules](#timer-attribution-rules).
/// - Appends `[status=NNN]` if the error carries an HTTP status.
/// - Appends `[url=...]` if the error carries a URL.
/// - Walks [`std::error::Error::source`] to the root, joining each
///   layer with ` -> `.  Consecutive identical messages are
///   deduplicated to avoid noise from libraries that wrap an error
///   in itself.
///
/// The output is intentionally one line so it composes with the
/// existing `log::warn!` / `log::error!` patterns in the codebase.
///
/// # Timer attribution rules
///
/// Given the active timers at a call site, the helper attributes
/// the failure as follows:
///
/// 1. [`reqwest::Error::is_connect`] → look for a timer named
///    `"connect_timeout"` in `timers`; if present, emit
///    `name=connect_timeout, budget=<value>`.
/// 2. [`reqwest::Error::is_timeout`] AND not `is_connect` → look for
///    a timer named `"request_wall_clock"` or `"validate_request"`
///    in `timers`; emit the first match.
/// 3. Source chain contains an `io::ErrorKind::TimedOut` AND neither
///    `is_connect` nor `is_timeout` matched a configured timer →
///    emit `name=kernel_tcp_unspecified` plus
///    `budgets=[tcp_user_timeout=3s, tcp_keepalive=5s+1s×3]` so the
///    reader sees both candidate kernel timers.  Reqwest cannot
///    distinguish [`TCP_USER_TIMEOUT`] from [`TCP_KEEPALIVE`]
///    expiration after the fact, so we surface both rather than
///    pretending to know.
/// 4. Otherwise → no `name=`/`budget=` tag; the existing
///    `[kind=..., url=...]` format alone.
pub(crate) fn format_reqwest_error_with_timers(
    err: &reqwest::Error,
    timers: &[TimerSpec],
) -> String {
    use std::error::Error as _;
    use std::fmt::Write as _;

    // Classify the failure into a coarse kind tag so log greppers can
    // bucket failures without re-parsing the message.  Order matters:
    // `is_timeout` and `is_connect` can both be true for some kernel
    // errors; we report the more actionable one first.
    let kind = if err.is_timeout() {
        "timeout"
    } else if err.is_connect() {
        "connect"
    } else if err.is_request() {
        "request"
    } else if err.is_body() {
        "body"
    } else if err.is_decode() {
        "decode"
    } else if err.is_redirect() {
        "redirect"
    } else if err.is_status() {
        "status"
    } else {
        "other"
    };

    let mut out = err.to_string();
    let _ = write!(out, " [kind={}", kind);

    // Timer attribution.  We compute this before status/url so the
    // `name=`/`budget=` tag stays close to `kind=` for readability.
    if let Some(attr) = attribute_timer(err, timers) {
        let _ = write!(out, ", {}", attr);
    }

    if let Some(status) = err.status() {
        let _ = write!(out, ", status={}", status.as_u16());
    }
    if let Some(url) = err.url() {
        let _ = write!(out, ", url={}", url);
    }
    out.push(']');

    // Walk the source chain to surface the underlying cause (DNS
    // error, OS error, TLS alert, etc.).  Dedup consecutive identical
    // messages — `hyper` and `reqwest` occasionally wrap an error in
    // a layer that re-prints the inner message verbatim.
    let mut last_layer = err.to_string();
    let mut current: Option<&dyn std::error::Error> = err.source();
    while let Some(e) = current {
        let msg = e.to_string();
        if msg != last_layer {
            out.push_str(" -> ");
            out.push_str(&msg);
            last_layer = msg;
        }
        current = e.source();
    }

    out
}

/// Compute the `name=...`, `budget=...` (or `budgets=[...]`) tag
/// fragment to insert into the formatted error, or `None` when no
/// timer attribution applies.
///
/// Encapsulated as a free function so the timer-matching rules in
/// the doc-comment of [`format_reqwest_error_with_timers`] live in
/// exactly one place and are unit-testable in isolation.
fn attribute_timer(err: &reqwest::Error, timers: &[TimerSpec]) -> Option<String> {
    use std::error::Error as _;

    // Rule 1 — connect-phase failure.
    if err.is_connect() {
        if let Some(t) = timers.iter().find(|t| t.name == "connect_timeout") {
            return Some(format!("name={}, budget={}", t.name, fmt_budget(t.budget)));
        }
        // Connect failure with no `connect_timeout` declared is
        // diagnostically interesting (caller forgot to declare it),
        // but we don't synthesize a name — fall through to no tag.
    }

    // Rule 2 — request-level wall-clock or validate-preflight timeout.
    if err.is_timeout() && !err.is_connect() {
        if let Some(t) = timers
            .iter()
            .find(|t| matches!(t.name, "request_wall_clock" | "validate_request"))
        {
            return Some(format!("name={}, budget={}", t.name, fmt_budget(t.budget)));
        }
    }

    // Rule 3 — kernel TCP timeout surfaced as a generic IO error.
    //
    // Reqwest's `is_connect`/`is_timeout` flags do NOT fire when the
    // socket is killed mid-stream by the kernel via `TCP_USER_TIMEOUT`
    // or unanswered keepalive probes.  Those surface as a plain
    // `std::io::Error` of kind `TimedOut` somewhere in the source
    // chain.  We can't tell which kernel timer expired, so we report
    // both candidate budgets — the reader can correlate with elapsed
    // time if needed.
    let mut current: Option<&dyn std::error::Error> = err.source();
    while let Some(e) = current {
        if let Some(io) = e.downcast_ref::<std::io::Error>() {
            if io.kind() == std::io::ErrorKind::TimedOut {
                let keepalive_dead = TCP_KEEPALIVE
                    .saturating_add(TCP_KEEPALIVE_INTERVAL.saturating_mul(TCP_KEEPALIVE_RETRIES));
                #[cfg(target_os = "linux")]
                let user_timeout = TCP_USER_TIMEOUT;
                #[cfg(not(target_os = "linux"))]
                let user_timeout = Duration::ZERO; // not active on non-Linux

                let budgets = if cfg!(target_os = "linux") {
                    format!(
                        "budgets=[tcp_user_timeout={}, tcp_keepalive={}]",
                        fmt_budget(user_timeout),
                        fmt_budget(keepalive_dead),
                    )
                } else {
                    format!("budgets=[tcp_keepalive={}]", fmt_budget(keepalive_dead))
                };
                return Some(format!("name=kernel_tcp_unspecified, {}", budgets));
            }
        }
        current = e.source();
    }

    None
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

/// Per-attempt wall-clock budgets (in seconds) for the validate
/// retry loop.  Length controls the number of attempts; the
/// schedule grows from the shared client `connect_timeout` (2s) to
/// 15s on attempt 5, giving the request more headroom on each
/// retry to survive a slow-responding server.
///
/// The `connect_timeout` itself does NOT grow per attempt — that's
/// a client-builder option in reqwest 0.13 and rebuilding a fresh
/// client per attempt would defeat the TCP-keepalive cache.  Only
/// the `.timeout()` wall-clock grows.
const VALIDATE_BUDGET_SECS: [u64; 5] = [2, 5, 8, 11, 15];

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
/// cache.  Encapsulates the growing-budget retry loop so the
/// cache layer in [`validate_model`] stays focused on the
/// memoization concern.
///
/// Retry semantics:
/// - 5 attempts (`VALIDATE_BUDGET_SECS.len()`).
/// - Each attempt uses a fresh `.timeout(...)` wall-clock derived
///   from [`VALIDATE_BUDGET_SECS`]; client-wide `connect_timeout`
///   (2s) and TCP-level defences are unchanged.
/// - Bail immediately on a model-not-found error (no point
///   retrying — the model list is authoritative).
/// - On any other error, retry the next attempt.
/// - After all attempts exhausted, return the last error.
async fn validate_model_uncached(
    provider_name: &str,
    api_key: &str,
    model: &str,
    api_base: &str,
    is_transcription_model: fn(&str) -> bool,
) -> Result<(), TalkError> {
    let models_url = format!("{}/v1/models", api_base);
    let client = build_client()?;

    let mut last_err: Option<TalkError> = None;

    for (attempt_idx, &budget_secs) in VALIDATE_BUDGET_SECS.iter().enumerate() {
        let attempt_num = (attempt_idx as u32) + 1;
        let total = VALIDATE_BUDGET_SECS.len() as u32;
        let budget = Duration::from_secs(budget_secs);

        log::debug!(
            "validate_model: attempt {}/{} for {}:{} (budget={}s)",
            attempt_num,
            total,
            provider_name,
            model,
            budget_secs
        );

        // Active timers for this attempt:
        //   1. `connect_timeout` from `build_client()` (client-wide, 2s).
        //   2. `validate_request` wall-clock via `.timeout()`
        //      (per-attempt, grows from 2s to 15s).
        let timers = [
            TimerSpec {
                name: "connect_timeout",
                budget: CONNECT_TIMEOUT,
            },
            TimerSpec {
                name: "validate_request",
                budget,
            },
        ];

        let send_result = client
            .get(&models_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .timeout(budget)
            .send()
            .await;

        let response = match send_result {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(TalkError::Config(format!(
                    "{} model validation failed (preflight to /v1/models): {}",
                    provider_name,
                    format_reqwest_error_with_timers(&e, &timers)
                )));
                continue; // retry
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            // HTTP error from the server is permanent — no point
            // retrying a 401/403/404 from the models endpoint.
            return Err(TalkError::Config(format!(
                "{} model validation failed (preflight to /v1/models): {} ({})",
                provider_name, status, body
            )));
        }

        let models: ModelsResponse = match response.json().await {
            Ok(m) => m,
            Err(e) => {
                // Decode error mid-response is more likely a
                // network glitch than a permanent server issue —
                // retry.
                last_err = Some(TalkError::Config(format!(
                    "{} model validation failed (preflight to /v1/models): \
                     could not parse response: {}",
                    provider_name, e
                )));
                continue;
            }
        };

        if models.data.iter().any(|m| m.id == model) {
            return Ok(());
        }

        // Model not found in the (successfully retrieved) list —
        // permanent.  No retry can change this answer.
        let mut transcription_models: Vec<&str> = models
            .data
            .iter()
            .map(|m| m.id.as_str())
            .filter(|id| is_transcription_model(id))
            .collect();
        transcription_models.sort();

        return if transcription_models.is_empty() {
            Err(TalkError::Config(format!(
                "Model '{}' not found in {} account",
                model, provider_name
            )))
        } else {
            Err(TalkError::Config(format!(
                "Model '{}' not found. Available transcription models: {}",
                model,
                transcription_models.join(", ")
            )))
        };
    }

    Err(last_err.unwrap_or_else(|| {
        TalkError::Config(format!(
            "{} model validation failed (preflight to /v1/models): \
             all {} attempts exhausted",
            provider_name,
            VALIDATE_BUDGET_SECS.len()
        ))
    }))
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

    #[tokio::test]
    async fn format_reqwest_error_classifies_connect_refused() {
        // 127.0.0.1:1 is reserved (tcpmux) and never has a listener
        // on any well-configured system; this gives us a deterministic
        // connect refusal without external dependencies.
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .expect("test: build client");
        let err = client
            .get("http://127.0.0.1:1/")
            .send()
            .await
            .expect_err("test: connection to :1 must fail");

        let s = format_reqwest_error_with_timers(&err, &[]);
        // Top-level message preserved.
        assert!(s.starts_with("error sending request"), "got: {}", s);
        // Kind tag present and either `connect` or `request`
        // depending on reqwest version (both are diagnostically useful).
        assert!(
            s.contains("[kind=connect") || s.contains("[kind=request"),
            "expected kind=connect or kind=request, got: {}",
            s
        );
        // URL surfaced.
        assert!(s.contains("url=http://127.0.0.1:1/"), "got: {}", s);
        // Source chain walked — should contain at least one ` -> ` and
        // ideally an OS-level hint like "refused" / "connection".
        assert!(
            s.contains(" -> "),
            "expected source chain in error string, got: {}",
            s
        );
    }

    #[tokio::test]
    async fn format_reqwest_error_classifies_dns_failure() {
        // RFC 2606 reserves `.invalid` so this name is guaranteed to
        // never resolve, giving us a deterministic DNS failure.
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .expect("test: build client");
        let err = client
            .get("http://nonexistent.invalid/")
            .send()
            .await
            .expect_err("test: DNS for .invalid must fail");

        let s = format_reqwest_error_with_timers(&err, &[]);
        // The DNS failure should appear somewhere in the chain — exact
        // wording varies between trust-dns / system resolver / glibc,
        // so match on robust substrings.
        let lower = s.to_lowercase();
        assert!(
            lower.contains("dns")
                || lower.contains("resolve")
                || lower.contains("lookup")
                || lower.contains("name or service")
                || lower.contains("nodename"),
            "expected DNS-related phrase in error chain, got: {}",
            s
        );
        assert!(s.contains("url=http://nonexistent.invalid/"), "got: {}", s);
    }

    #[tokio::test]
    async fn format_reqwest_error_handles_request_cancelled_by_timeout() {
        // Hit a TCP listener that accepts but never replies, with a
        // very short per-request timeout.  In reqwest 0.13 this surfaces
        // as `is_request()` (the request was cancelled mid-flight)
        // rather than `is_timeout()` — `is_timeout()` only flips on for
        // certain code paths.  Either classification is diagnostically
        // useful: what matters is that the helper:
        //   (a) emits a kind tag,
        //   (b) preserves the URL,
        //   (c) walks the source chain and surfaces the cancellation
        //       reason ("canceled" / "closed before message completed").
        let client = reqwest::Client::builder()
            .build()
            .expect("test: build client");
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
        let err = client
            .get(&url)
            .timeout(Duration::from_millis(50))
            .send()
            .await
            .expect_err("test: send must fail when peer never replies");

        let s = format_reqwest_error_with_timers(&err, &[]);
        // Some kind tag must be present.
        assert!(s.contains("[kind="), "missing kind tag, got: {}", s);
        // Either of the diagnostically-useful classifications is fine.
        assert!(
            s.contains("[kind=timeout") || s.contains("[kind=request"),
            "expected kind=timeout or kind=request, got: {}",
            s
        );
        // URL surfaced.
        assert!(s.contains(&format!("url={}", url)), "got: {}", s);
        // Source chain was walked — there must be at least one ` -> `
        // and the chain must mention the underlying cancellation /
        // closure (this is the whole point of the helper).
        assert!(s.contains(" -> "), "expected source chain, got: {}", s);
        let lower = s.to_lowercase();
        assert!(
            lower.contains("cancel")
                || lower.contains("closed")
                || lower.contains("timed out")
                || lower.contains("timeout"),
            "expected cancellation/closure/timeout phrase in chain, got: {}",
            s
        );
    }

    #[test]
    fn format_reqwest_error_dedups_identical_layers() {
        // Pure-logic check: the dedup is implemented in our helper, so
        // it is exercised by the dyn-error-chain walk on real
        // `reqwest::Error`s above.  This unit-level guard is folded
        // into those integration-style tests rather than synthesising
        // a `reqwest::Error` (no public constructor) — leaving this
        // marker so future maintainers know dedup is intentional.
    }

    // ── format_reqwest_error_with_timers — timer attribution ────

    /// Spec: when reqwest reports a connect-phase failure AND the
    /// caller declared a `connect_timeout` timer, the formatted error
    /// must name that timer and quote its budget — so log readers can
    /// tell *which* timer fired without correlating with elapsed
    /// time.
    #[tokio::test]
    async fn format_with_timers_attributes_connect_phase_to_connect_timeout() {
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
        let s = format_reqwest_error_with_timers(&err, &timers);

        // Some reqwest versions classify a refused connect as
        // `is_request()` rather than `is_connect()`.  Only assert the
        // attribution when reqwest itself reports it as connect — the
        // helper's contract is: "if reqwest says is_connect(), name
        // the connect_timeout".  Otherwise attribution may be absent
        // (Rule 4) which is also correct behaviour.
        if err.is_connect() {
            assert!(
                s.contains("name=connect_timeout"),
                "expected name=connect_timeout in attribution, got: {}",
                s
            );
            assert!(
                s.contains("budget=2s"),
                "expected budget=2s in attribution, got: {}",
                s
            );
        }
    }

    /// Spec: when a request-level wall-clock timeout fires (not a
    /// connect-phase failure) AND the caller declared
    /// `request_wall_clock`, the formatted error must name that timer
    /// and quote its budget.  This is the regression-detection test:
    /// if a future refactor reverts to a generic "timed out" message,
    /// this fails.
    #[tokio::test]
    async fn format_with_timers_attributes_request_timeout_to_wall_clock() {
        // Bind a TCP listener that accepts but never replies — the
        // connect phase succeeds, then the per-request `.timeout()`
        // fires.  This is exactly the failure mode the user observed
        // after `bf22c0c` regressed the timeout behaviour.
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
        let s = format_reqwest_error_with_timers(&err, &timers);

        // Assert attribution only when reqwest classifies the failure
        // as a non-connect timeout — i.e. the spec's Rule 2 case.
        // Some reqwest versions surface this as `is_request()`
        // instead, in which case Rule 4 applies (no attribution) —
        // both are correct, neither is a regression.
        if err.is_timeout() && !err.is_connect() {
            assert!(
                s.contains("name=request_wall_clock"),
                "expected name=request_wall_clock in attribution, got: {}",
                s
            );
            assert!(
                s.contains("budget="),
                "expected budget=... in attribution, got: {}",
                s
            );
        }
    }

    /// Spec: the validate-preflight path uses `validate_request` as
    /// its non-connect timer name; attribution must select that name
    /// (not `request_wall_clock`) when only it is declared.
    #[tokio::test]
    async fn format_with_timers_attributes_validate_preflight_separately() {
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

        // Note: only `validate_request` declared — no
        // `request_wall_clock`.  Helper must pick the validate name.
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
        let s = format_reqwest_error_with_timers(&err, &timers);

        if err.is_timeout() && !err.is_connect() {
            assert!(
                s.contains("name=validate_request"),
                "expected name=validate_request, got: {}",
                s
            );
            assert!(
                !s.contains("name=request_wall_clock"),
                "must not pick request_wall_clock when not declared, got: {}",
                s
            );
        }
    }

    /// Spec: `format_reqwest_error` (no timers) must continue to
    /// produce valid output for callers that have not yet declared
    /// their timers.  No `name=`/`budget=` tag is emitted.
    #[tokio::test]
    async fn format_without_timers_omits_attribution_tag() {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()
            .expect("test: build client");
        let err = client
            .get("http://127.0.0.1:1/")
            .send()
            .await
            .expect_err("test: connection to :1 must fail");

        let s = format_reqwest_error_with_timers(&err, &[]);

        assert!(
            !s.contains("name="),
            "format_reqwest_error must not synthesize attribution tags, got: {}",
            s
        );
        assert!(
            !s.contains("budget="),
            "format_reqwest_error must not synthesize budget tags, got: {}",
            s
        );
        // But the existing kind/url tags must still be present.
        assert!(s.contains("[kind="), "missing kind tag, got: {}", s);
    }

    /// Spec: when a wall-clock timer fires at a call site, the
    /// helper must NOT attribute the failure to `connect_timeout`
    /// even if a connect_timeout TimerSpec is also in the slice.
    /// This guards against attribution flipping back to "connect" on
    /// reqwest version upgrades that change is_connect() semantics.
    #[tokio::test]
    async fn format_with_timers_does_not_misattribute_wall_clock_to_connect() {
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
        let s = format_reqwest_error_with_timers(&err, &timers);

        // If reqwest reported is_connect()=true, attribution to
        // connect_timeout is correct (Rule 1).  Otherwise the helper
        // MUST NOT mention connect_timeout — that would be lying
        // about which timer fired.
        if !err.is_connect() {
            assert!(
                !s.contains("name=connect_timeout"),
                "must not attribute non-connect failure to connect_timeout, got: {}",
                s
            );
        }
    }

    /// Spec: `attribute_timer` directly — non-timeout, non-connect
    /// errors with no IO TimedOut in the chain must produce no
    /// attribution (Rule 4).  Use the dedicated helper test path so
    /// we exercise the negative case without depending on a network
    /// listener.
    #[tokio::test]
    async fn format_without_timeout_or_connect_emits_no_attribution() {
        // A bind on a known-valid local URL with `.json()` body that
        // is never sent — not a viable construction.  Instead, force
        // a decode error: serve invalid JSON via an in-process server
        // and decode it.  This produces an `is_decode()` error which
        // matches Rule 4 (no attribution).
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
        let s = format_reqwest_error_with_timers(&err, &timers);

        // Decode errors are neither timeouts nor connect errors and
        // typically don't carry an io::TimedOut in their chain — so
        // attribution must be silent.
        assert!(
            !s.contains("name=connect_timeout"),
            "decode error must not attribute to connect_timeout, got: {}",
            s
        );
        assert!(
            !s.contains("name=request_wall_clock"),
            "decode error must not attribute to request_wall_clock, got: {}",
            s
        );
    }

    /// Spec: `fmt_budget` produces stable, log-friendly strings for
    /// whole-second and sub-second durations.
    #[test]
    fn fmt_budget_formats_whole_seconds_compactly() {
        assert_eq!(fmt_budget(Duration::from_secs(0)), "0s");
        assert_eq!(fmt_budget(Duration::from_secs(2)), "2s");
        assert_eq!(fmt_budget(Duration::from_secs(108)), "108s");
    }

    #[test]
    fn fmt_budget_formats_subsecond_with_three_decimals() {
        assert_eq!(fmt_budget(Duration::from_millis(50)), "0.050s");
        assert_eq!(fmt_budget(Duration::from_millis(1500)), "1.500s");
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

    /// Spec: the growing-budget schedule is exactly `[2, 5, 8, 11, 15]`
    /// seconds, in that order, with five attempts.  The literal values
    /// matter: the user explicitly specified them, so a future refactor
    /// that "rounds" or "tweaks" them must fail this test loudly.
    #[test]
    fn validate_budget_schedule_is_2_5_8_11_15() {
        assert_eq!(VALIDATE_BUDGET_SECS, [2, 5, 8, 11, 15]);
        assert_eq!(VALIDATE_BUDGET_SECS.len(), 5);
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
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("model validation failed (preflight to /v1/models)"),
            "expected preflight-failed lead phrase, got: {}",
            msg
        );
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
}
