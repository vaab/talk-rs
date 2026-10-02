//! Transport primitives for transcription API calls.
//!
//! This module contains the resilience and connectivity concerns
//! shared by every provider, across both one-shot (HTTP POST) and
//! realtime (WebSocket) protocols.
//!
//! # Architecture
//!
//! All outgoing connections to transcription providers funnel
//! through this module.  The public surface is intentionally small:
//!
//! - [`http_request`] — the single entry point for HTTP requests
//!   (one-shot transcription POST, model validation GET, model list
//!   GET).  Handles retries (connection + data) internally; emits
//!   [`ConnectionEvent`]s; supports cancellation via
//!   [`tokio_util::sync::CancellationToken`].
//! - [`ws_upgrade`] — the single entry point for WebSocket upgrade
//!   handshakes (realtime transcription).  Same retry/event/cancel
//!   semantics as [`http_request`].
//! - [`Request`], [`RequestBody`], [`Response`],
//!   [`ConnectionEvent`], [`RetryKind`] — the public vocabulary.
//!
//! Outside this module, **no code** opens a network socket to a
//! provider, no code knows the word "retry" or "attempt" or
//! "connect_timeout", and no code constructs a
//! [`crate::error::PipelineFailure`] for a network call — the
//! transport builds them itself with truthful attempt counters.
//!
//! # Legacy sub-modules (being migrated)
//!
//! - `http`: reqwest client configuration, per-request
//!   proportional timeout, progress-reporting request body, model
//!   validation, and model-error enrichment.  Functions here will
//!   either move behind [`http_request`] or become private helpers
//!   over the course of the transport consolidation.
//! - `retry`: the legacy retry primitive used by one-shot HTTP calls
//!   and realtime WebSocket upgrade handshakes.  Being absorbed
//!   into [`http_request`] / [`ws_upgrade`].
//! - `validate_cache`: disk-backed memoization of `/v1/models`
//!   preflight results — unchanged.
//! - `ws`: shared WebSocket helpers — being populated as part of
//!   the consolidation.

pub(crate) mod http;
pub(crate) mod validate_cache;
pub(crate) mod ws;

#[cfg(test)]
mod client_tests;

use crate::config::Provider;
use crate::error::{NetworkKind, PipelineFailure, PipelineFailureKind, TimerLabel};
use crate::telemetry::{TelemetrySink, TranscriptionEvent};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

// ── Internal retry schedules ────────────────────────────────────────

/// Per-attempt budgets (seconds) for the connection-phase retry
/// loop.  Length = total connection attempts; the value at index
/// `i` is the connect-timeout used on attempt `i+1`.  Matches the
/// historical `VALIDATE_BUDGET_SECS` schedule so the consolidated
/// transport doesn't regress validate-path behaviour.
///
/// The budget caps the whole attempt (TCP + TLS + upload + response)
/// **unless** the request carries a larger [`Request::wall_clock`],
/// in which case the wall-clock wins — see `attempt_cap`.  Without
/// that rule a 26 MB upload is structurally doomed on the first six
/// slots and only the 120 s slot can ever succeed.
pub const CONNECTION_BUDGETS_SECS: [u64; 7] = [2, 5, 8, 11, 15, 30, 120];

/// Wait (seconds) before data-phase retry `n` (1-based) when the
/// server answered with a retryable status (5xx or 429) and sent no
/// usable `Retry-After`.  Length = number of data-phase retries; the
/// data-phase attempt budget is `len() + 1` (initial call + retries).
///
/// Sized for a *saturated* provider, not a flaky one: field reports
/// show Mistral answering `503 high load, please retry` and
/// `429 backend_out_of_capacity` for several minutes at a time.
/// Retrying seven times in under two minutes (the previous
/// behaviour) just burns upload bandwidth; the schedule below gives
/// the provider ~5.5 minutes to recover before giving up.
pub const DATA_BACKOFF_SECS: [u64; 6] = [5, 15, 30, 60, 120, 120];

/// Retry budgets for one HTTP request. The default is the production
/// schedule; callers may supply shorter budgets for local endpoints.
#[derive(Debug, Clone)]
pub struct RetrySchedule {
    pub connection_budgets: Vec<Duration>,
    pub data_backoffs: Vec<Duration>,
    pub max_data_wait: Option<Duration>,
    /// Set by the first request; clones for preflight and upload share the start.
    pub wait_started: Option<Arc<OnceLock<Instant>>>,
    /// Chain entries may wait on a busy WebSocket handshake; legacy callers do not.
    pub retry_ws_busy: bool,
}

impl Default for RetrySchedule {
    fn default() -> Self {
        Self {
            connection_budgets: CONNECTION_BUDGETS_SECS.map(Duration::from_secs).to_vec(),
            data_backoffs: DATA_BACKOFF_SECS.map(Duration::from_secs).to_vec(),
            max_data_wait: None,
            wait_started: None,
            retry_ws_busy: false,
        }
    }
}

/// Upper bound applied to a server-supplied `Retry-After` so a
/// misbehaving header cannot park an unattended pipeline for hours.
/// Equal to the largest schedule slot.
const RETRY_AFTER_CAP_SECS: u64 = 120;

// ── Public vocabulary ───────────────────────────────────────────────

/// HTTP method for a [`Request`].
///
/// Intentionally narrow — transcription providers only ever need
/// GET (model listing, validation) and POST (transcription
/// submission).  Adding more is a deliberate act, not a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// GET — used for `/v1/models` validation and model suggestion
    /// fetches.
    Get,
    /// POST — used for `/v1/audio/transcriptions` submission.
    Post,
}

/// Body source for a transport [`Request`].
///
/// Each variant maps to a reqwest body strategy; the transport
/// chooses the right one without leaking reqwest types to callers.
///
/// Note: the [`Multipart`](Self::Multipart) variant takes a
/// **factory** closure rather than a constructed form because
/// `reqwest::multipart::Form` is not `Clone` and retries need to
/// rebuild it fresh per attempt.  Each call to the factory
/// produces a fresh, sendable form (typically wrapping the same
/// underlying audio buffer in a fresh `ProgressBody` stream).
pub enum RequestBody {
    /// No body (e.g. for GETs).
    Empty,
    /// In-memory bytes.  Sent with `Content-Length`.  Cheaply
    /// reusable across retries — the transport `.clone()`s the
    /// underlying `Arc<Vec<u8>>` per attempt.
    Bytes(std::sync::Arc<Vec<u8>>),
    /// `multipart/form-data` body produced by a factory closure.
    /// The transport invokes the closure once per attempt.  The
    /// closure must be cheap to call (the audio buffer is
    /// typically wrapped in `Arc` and reused; only the multipart
    /// envelope and progress-stream wrapper are rebuilt).
    Multipart(Box<dyn Fn() -> reqwest::multipart::Form + Send + Sync>),
}

/// A request to be sent through the transport.
///
/// Constructed by callers (providers, validate_model) and passed
/// to [`http_request`].  Carries everything the transport needs to
/// know to send the request, retry it on failure, and produce a
/// truthful [`PipelineFailure`] if the request never succeeds.
pub struct Request {
    /// HTTP method.
    pub method: Method,
    /// Fully-qualified URL.
    pub url: String,
    /// Request headers as `(name, value)` pairs.  The transport
    /// applies them verbatim.
    pub headers: Vec<(String, String)>,
    /// Request body.
    pub body: RequestBody,
    /// Provider this request belongs to.  Used for [`PipelineFailure`]
    /// tagging and provider-specific error classification.
    pub provider: Provider,
    /// Provider display name (e.g. `"Mistral"`, `"OpenAI"`) — used
    /// verbatim in [`PipelineFailure::provider`] so error messages
    /// read naturally.
    pub provider_name: String,
    /// Phase tag used to build [`PipelineFailure`] on error.
    pub phase: crate::error::PipelinePhase,
    /// Optional per-attempt wall-clock budget.  When `Some(budget)`,
    /// the transport sets the reqwest `.timeout(budget)` on the
    /// underlying request.  When `None`, no wall-clock timeout is
    /// attached (user-attended mode: the request is allowed to
    /// take as long as it takes, bounded only by the connect
    /// timeout and TCP-level defences).
    pub wall_clock: Option<Duration>,
    /// Connection and data retry budgets for this request.
    pub retry_schedule: RetrySchedule,
}

/// A response from the transport.
///
/// Mirrors a small subset of `reqwest::Response` — just enough for
/// providers to read status, headers, and body without importing
/// reqwest directly.
#[derive(Debug)]
pub struct Response {
    /// HTTP status code.
    pub status: u16,
    /// Response headers as `(name, value)` pairs.
    pub headers: Vec<(String, String)>,
    /// Response body bytes (fully buffered).
    pub body: Vec<u8>,
}

/// Distinguishes the two retry concerns inside the transport.
///
/// Surfaces in [`ConnectionEvent::RetryScheduled`] so log and UI
/// consumers can say `"connect retry 2/5"` vs `"server retry 1/3"`
/// instead of a generic "retry N/M".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryKind {
    /// Retry of the TCP / TLS / DNS connection phase.  Grows the
    /// per-attempt connect budget on each retry.
    Connection,
    /// Retry after the connection was established and the server
    /// returned a transient failure (5xx, decode error mid-body).
    Data,
}

/// A single event in the lifecycle of a transport call.
///
/// Streamed to the caller's [`TelemetrySink`] as the request
/// progresses.  Consumers (picker UI, streaming overlay,
/// cross-process socket observer) derive higher-level state from
/// the event sequence.
#[derive(Debug, Clone)]
pub enum ConnectionEvent {
    /// DNS resolution in progress.
    ResolvingDns { t: Instant },
    /// TCP connection attempt in progress.
    Connecting { t: Instant },
    /// TLS handshake in progress.
    TlsHandshake { t: Instant },
    /// Connection is ready — bytes can flow.
    ConnectionReady { t: Instant },
    /// Upload progress: `bytes_sent` of `total`.
    Uploading {
        bytes_sent: u64,
        total: u64,
        t: Instant,
    },
    /// All request bytes have been sent.
    UploadComplete { total: u64, t: Instant },
    /// Waiting for response headers from the server.
    AwaitingResponse { t: Instant },
    /// Response headers received.
    ResponseHeaders { status: u16, t: Instant },
    /// Download progress: `bytes_received` of optional `total`.
    Downloading {
        bytes_received: u64,
        total: Option<u64>,
        t: Instant,
    },
    /// Response body fully received.
    ResponseComplete { total: u64, t: Instant },
    /// A retry has been scheduled.  Emitted BEFORE the retry's
    /// first network call.
    RetryScheduled {
        /// Whether this is a connection-phase or data-phase retry.
        kind: RetryKind,
        /// 1-indexed retry number (`1` = the first retry; the
        /// initial attempt does not emit this event).
        attempt: u32,
        /// Maximum number of retries configured for this `kind`.
        max: u32,
        /// Budget that the upcoming attempt will use (for
        /// connection retries, this is the connect timeout for the
        /// next attempt; for data retries, the wall-clock).
        budget: Duration,
        /// Human-readable reason for the retry (typically the
        /// previous attempt's error message).
        reason: String,
        t: Instant,
    },
    /// Terminal success.
    Completed { t: Instant },
    /// Terminal failure.  Carries the structured [`PipelineFailure`]
    /// that will be returned to the caller.  Wrapped in `Arc` so
    /// the event is cheaply [`Clone`]able for broadcast fan-out.
    Failed {
        error: Arc<PipelineFailure>,
        t: Instant,
    },
}

// ── Public entry points ─────────────────────────────────────────────

/// Send an HTTP request through the transport.
///
/// Handles connection-phase retries (growing budget) and
/// data-phase retries (bounded) internally.  Emits
/// [`ConnectionEvent`]s on `sink` throughout.  Cancellation via
/// `cancel` aborts the in-flight request and any pending retries.
///
/// # Cancellation
///
/// When `cancel` is triggered the function returns promptly with a
/// [`PipelineFailure`] of [`crate::error::NetworkKind::Other`]
/// carrying a "cancelled by caller" message in its source chain.
/// The retry loops also stop immediately.
///
/// # Errors
///
/// Returns [`PipelineFailure`] with truthful `attempts` /
/// `max_attempts` counters when all retries are exhausted, or when
/// the server returns a permanent failure (model-not-found, 401, …).
///
/// # Implementation notes
///
/// The body runs two nested retry loops, each with its own budget:
///
/// 1. **Data retry** — outer loop, waits indexed against
///    [`DATA_BACKOFF_SECS`].  A retryable status (5xx or 429) after
///    a completed HTTP transaction triggers a wait (the server's
///    `Retry-After` when present, else the schedule slot) and a
///    fresh connection-phase loop.  Emits [`RetryKind::Data`].
///
/// 2. **Connection retry** — inner loop, attempts indexed against
///    [`CONNECTION_BUDGETS_SECS`].  Each attempt obtains a shared
///    reqwest client with the current per-attempt connect timeout
///    and tries once.  On a connect-class failure (DNS, ECONNREFUSED,
///    TCP/TLS timeout) the loop emits [`RetryKind::Connection`] and
///    advances to the next attempt.  The connection schedule restarts
///    from its first slot on every data retry, so a server retry is
///    never charged against the connection budget (and vice versa).
///
/// Cancellation is wired via `tokio::select!` around both the
/// in-flight request and the backoff wait, so the call returns
/// promptly the moment the token fires.
pub async fn http_request(
    req: Request,
    sink: &Arc<dyn TelemetrySink>,
    cancel: CancellationToken,
) -> Result<Response, PipelineFailure> {
    let max_data_attempts = req.retry_schedule.data_backoffs.len() as u32 + 1;

    sink.emit(TranscriptionEvent::RequestStarted {
        endpoint: req.url.clone(),
        t: Instant::now(),
    });

    let mut data_attempt: u32 = 1;
    let started = Instant::now();
    let wait_started = req
        .retry_schedule
        .wait_started
        .as_ref()
        .map_or(started, |shared| *shared.get_or_init(|| started));
    let deadline = req
        .retry_schedule
        .max_data_wait
        .map(|limit| wait_started + limit);
    loop {
        let outcome = run_connection_phase(
            &req,
            sink,
            &cancel,
            data_attempt,
            max_data_attempts,
            deadline,
        )
        .await;

        let (pf, retry_after) = match outcome {
            ConnectionPhase::Done(result) => {
                sink.emit(TranscriptionEvent::RequestCompleted {
                    success: result.is_ok(),
                    t: Instant::now(),
                });
                return result;
            }
            ConnectionPhase::DataRetryable {
                failure,
                retry_after,
            } => (failure, retry_after),
        };

        // Data-phase retry decision.  `data_attempt` is the attempt
        // that just failed; retry `n` (1-based) waits
        // `DATA_BACKOFF_SECS[n-1]` unless the server said otherwise.
        let retry_index = data_attempt as usize - 1;
        let Some(&slot) = req.retry_schedule.data_backoffs.get(retry_index) else {
            sink.emit(TranscriptionEvent::RequestCompleted {
                success: false,
                t: Instant::now(),
            });
            return Err(pf);
        };
        let wait = retry_after
            .map(|d| d.min(Duration::from_secs(RETRY_AFTER_CAP_SECS)))
            .unwrap_or(slot);
        let wait = if let Some(deadline) = deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if wait >= remaining {
                sink.emit(TranscriptionEvent::RequestCompleted {
                    success: false,
                    t: Instant::now(),
                });
                return Err(pf);
            }
            wait
        } else {
            wait
        };
        let retry_num = data_attempt;
        let reason = pf.to_string();
        log::info!(
            "{} busy — retrying in {}s (server retry {}/{}): {}",
            req.provider_name,
            wait.as_secs(),
            retry_num,
            req.retry_schedule.data_backoffs.len(),
            reason,
        );
        crate::perf_counters::incr(crate::perf_counters::Counter::DataRetries);
        sink.emit(TranscriptionEvent::RetryScheduled {
            kind: crate::telemetry::RetryKind::Data,
            attempt: retry_num,
            max: req.retry_schedule.data_backoffs.len() as u32,
            reason,
            delay: wait,
            t: Instant::now(),
        });

        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                sink.emit(TranscriptionEvent::RequestCompleted {
                    success: false,
                    t: Instant::now(),
                });
                return Err(build_cancellation_failure(&req, data_attempt, max_data_attempts));
            }
            _ = tokio::time::sleep(wait) => {}
        }
        data_attempt += 1;
    }
}

/// Outcome of one full pass through the connection-phase schedule.
enum ConnectionPhase {
    /// Terminal for this request: success, cancellation, permanent
    /// failure, or connection budget exhausted.
    Done(Result<Response, PipelineFailure>),
    /// The server answered with a retryable status; the caller
    /// decides whether to wait and run another pass.  `retry_after`
    /// is the parsed `Retry-After` header when the server sent one.
    DataRetryable {
        failure: PipelineFailure,
        retry_after: Option<Duration>,
    },
}

/// Run the connection-phase schedule once: up to
/// `CONNECTION_BUDGETS_SECS.len()` attempts with growing budgets.
///
/// `data_attempt` / `max_data_attempts` are stamped into any
/// [`PipelineFailure`] produced by a completed HTTP transaction so
/// the counters describe the *server* retry budget (the one that was
/// actually exhausted); connection-class failures keep reporting the
/// connection budget.
async fn run_connection_phase(
    req: &Request,
    sink: &Arc<dyn TelemetrySink>,
    cancel: &CancellationToken,
    data_attempt: u32,
    max_data_attempts: u32,
    deadline: Option<Instant>,
) -> ConnectionPhase {
    let max_connection_attempts = req.retry_schedule.connection_budgets.len() as u32;
    let mut last_failure: Option<PipelineFailure> = None;

    for (idx, &connect_budget) in req.retry_schedule.connection_budgets.iter().enumerate() {
        if idx > 0 && deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return ConnectionPhase::Done(Err(last_failure.unwrap_or_else(|| {
                build_generic_exhausted_failure(req, max_connection_attempts)
            })));
        }
        let attempt_num = (idx as u32) + 1;

        // Emit a `RetryScheduled` BEFORE every attempt past the
        // first.  This is the connection-phase retry signal; the
        // data-phase retry is signalled by the caller.
        if idx > 0 {
            let reason = last_failure
                .as_ref()
                .map(|f| f.to_string())
                .unwrap_or_else(|| "connection failed".into());
            crate::perf_counters::incr(crate::perf_counters::Counter::ConnectionRetries);
            sink.emit(TranscriptionEvent::RetryScheduled {
                kind: crate::telemetry::RetryKind::Connection,
                attempt: idx as u32,
                max: max_connection_attempts.saturating_sub(1),
                reason,
                delay: Duration::ZERO,
                t: Instant::now(),
            });
        }

        // ── Bail early if the caller cancelled. ─────────────────
        if cancel.is_cancelled() {
            return ConnectionPhase::Done(Err(build_cancellation_failure(
                req,
                attempt_num,
                max_connection_attempts,
            )));
        }

        // ── Single attempt ──────────────────────────────────────
        let attempt_outcome = run_single_http_attempt(
            req,
            connect_budget,
            sink,
            cancel,
            attempt_num,
            max_connection_attempts,
            data_attempt,
            max_data_attempts,
        )
        .await;

        match attempt_outcome {
            SingleAttempt::Success(resp) => return ConnectionPhase::Done(Ok(resp)),
            SingleAttempt::Cancelled(pf) | SingleAttempt::Permanent(pf) => {
                return ConnectionPhase::Done(Err(pf));
            }
            SingleAttempt::ConnectionRetryable(pf) => {
                last_failure = Some(pf);
                continue;
            }
            SingleAttempt::DataRetryable {
                failure,
                retry_after,
            } => {
                return ConnectionPhase::DataRetryable {
                    failure,
                    retry_after,
                };
            }
        }
    }

    ConnectionPhase::Done(Err(last_failure.unwrap_or_else(|| {
        build_generic_exhausted_failure(req, max_connection_attempts)
    })))
}

/// Per-attempt wall-clock cap: the connection-phase budget, lifted to
/// the request's own `wall_clock` when that is larger.
///
/// The connection schedule starts at 2 s — fine for a `/v1/models`
/// GET, hopeless for a multi-megabyte upload.  A caller that sized
/// its `wall_clock` to the payload (see
/// `transport::http::proportional_timeout`) has already stated how
/// long one attempt may legitimately take; the connection slot must
/// not undercut it.  Requests without a `wall_clock` (user-attended
/// mode, small GETs) keep the growing connection schedule as their
/// only cap.
fn attempt_cap(connect_budget: Duration, wall_clock: Option<Duration>) -> Duration {
    match wall_clock {
        Some(wc) if wc > connect_budget => wc,
        _ => connect_budget,
    }
}

/// True for HTTP statuses the transport treats as "provider busy,
/// try again later": every 5xx, plus 429 (rate limit / capacity).
fn is_data_retryable_status(status: u16) -> bool {
    status == 429 || (500..600).contains(&status)
}

/// Parse either Retry-After form at a supplied instant; past dates mean no wait.
fn parse_retry_after_at(
    headers: &[(String, String)],
    now: std::time::SystemTime,
) -> Option<Duration> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("retry-after"))
        .and_then(|(_, value)| {
            let value = value.trim();
            value
                .parse::<u64>()
                .ok()
                .map(Duration::from_secs)
                .or_else(|| {
                    httpdate::parse_http_date(value)
                        .ok()
                        .map(|date| date.duration_since(now).unwrap_or_default())
                })
        })
}

/// Outcome of a single HTTP attempt inside [`http_request`].
enum SingleAttempt {
    /// Request succeeded with a 2xx response.
    Success(Response),
    /// The caller's `CancellationToken` fired while the attempt
    /// was in flight.  Carries the cancellation-tagged failure to
    /// surface to the caller.
    Cancelled(PipelineFailure),
    /// Connection-class failure — eligible for a connection retry
    /// (next attempt in the growing-budget schedule).
    ConnectionRetryable(PipelineFailure),
    /// Server returned a retryable status (5xx or 429) — eligible
    /// for a data retry after a backoff wait (schedule:
    /// [`DATA_BACKOFF_SECS`]).  `retry_after` carries the server's
    /// own hint when it sent one.
    DataRetryable {
        failure: PipelineFailure,
        retry_after: Option<Duration>,
    },
    /// Permanent failure (4xx, decode error mid-body, …).  Do not
    /// retry; return the failure to the caller.
    Permanent(PipelineFailure),
}

/// Run a single HTTP attempt and classify the outcome.
///
/// `attempt_num` / `max_attempts` go into any [`PipelineFailure`]
/// the attempt produces so the caller sees the truthful retry
/// counter — killing the historical `1/1` lie in
/// `mistral::send_once` / `openai::send_once`.
#[allow(clippy::too_many_arguments)] // two (attempt, max) pairs: connection budget + data budget
async fn run_single_http_attempt(
    req: &Request,
    connect_budget: Duration,
    sink: &Arc<dyn TelemetrySink>,
    cancel: &CancellationToken,
    attempt_num: u32,
    max_attempts: u32,
    data_attempt: u32,
    max_data_attempts: u32,
) -> SingleAttempt {
    // Client policy is shared; headers, body and wall-clock cap remain request-local.
    let client = match http_client(connect_budget, cancel).await {
        Ok(c) => c,
        Err(_e) if cancel.is_cancelled() => {
            return SingleAttempt::Cancelled(build_cancellation_failure(
                req,
                attempt_num,
                max_attempts,
            ));
        }
        Err(e) => {
            return SingleAttempt::Permanent(PipelineFailure::new(
                req.provider_name.clone(),
                req.phase,
                attempt_num,
                max_attempts,
                req.url.clone(),
                PipelineFailureKind::Decode(format!("transport client build failed: {}", e)),
            ));
        }
    };

    // ── Build the request ──────────────────────────────────────
    let mut request_builder = match req.method {
        Method::Get => client.get(&req.url),
        Method::Post => client.post(&req.url),
    };
    for (name, value) in &req.headers {
        request_builder = request_builder.header(name, value);
    }
    // Bodies: the transport owns the body construction so the
    // caller never touches reqwest::Body directly.  Each retry
    // rebuilds the body fresh by either cloning the Arc<Vec<u8>>
    // (cheap) or invoking the multipart factory (rebuilds the
    // form envelope; the underlying audio buffer is typically
    // Arc-shared so the rebuild is also cheap).
    match &req.body {
        RequestBody::Empty => {}
        RequestBody::Bytes(bytes) => {
            request_builder = request_builder.body(bytes.as_ref().clone());
        }
        RequestBody::Multipart(factory) => {
            let form = factory();
            request_builder = request_builder.multipart(form);
        }
    }
    if let Some(budget) = req.wall_clock {
        request_builder = request_builder.timeout(budget);
    }

    // ── Send + read body, racing against cancellation ──────────
    sink.emit(TranscriptionEvent::ConnectionEstablished { t: Instant::now() });

    let send_fut = async {
        let response = request_builder.send().await?;
        let status = response.status();
        sink.emit(TranscriptionEvent::ResponseHeaders {
            status: status.as_u16(),
            t: Instant::now(),
        });
        let headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .map(|(n, v)| {
                (
                    n.as_str().to_string(),
                    v.to_str().unwrap_or_default().to_string(),
                )
            })
            .collect();
        let body = response.bytes().await?;
        Ok::<_, reqwest::Error>(Response {
            status: status.as_u16(),
            headers,
            body: body.to_vec(),
        })
    };

    // When `wall_clock` is set, the request builder applies that timeout
    // and the outer timeout caps the whole send at the larger of it and
    // this attempt's connection budget.  With `wall_clock: None`, there
    // is no outer cap: reqwest's connect timeout and TCP defences still
    // apply, while a connected user-attended request may keep waiting.
    let cap = attempt_cap(connect_budget, req.wall_clock);
    let send_with_cap = async {
        match req.wall_clock {
            Some(_) => tokio::time::timeout(cap, send_fut).await,
            None => Ok(send_fut.await),
        }
    };

    let outcome = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            return SingleAttempt::Cancelled(build_cancellation_failure(
                req, attempt_num, max_attempts,
            ));
        }
        result = send_with_cap => result,
    };

    let response = match outcome {
        // Outer `tokio::time::timeout` fired → connect-class retry.
        Err(_elapsed) => {
            let pf = PipelineFailure::new(
                req.provider_name.clone(),
                req.phase,
                attempt_num,
                max_attempts,
                req.url.clone(),
                PipelineFailureKind::Network {
                    kind: NetworkKind::Connect,
                    timer: Some(TimerLabel::from_duration("connect_timeout", cap)),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!(
                            "connection attempt did not complete \
                             within {}s",
                            cap.as_secs()
                        ),
                    )),
                },
            );
            return SingleAttempt::ConnectionRetryable(pf);
        }
        Ok(Ok(r)) => r,
        Ok(Err(err)) => {
            // Reqwest itself produced an error (could be
            // connect-class or post-headers).
            let is_connect_class = err.is_connect() || err.is_timeout() || matches_kernel_tcp(&err);
            let timers = [http::TimerSpec {
                name: "connect_timeout",
                budget: connect_budget,
            }];
            let kind = http::build_pipeline_failure_kind(err, &timers);
            let pf = PipelineFailure::new(
                req.provider_name.clone(),
                req.phase,
                attempt_num,
                max_attempts,
                req.url.clone(),
                kind,
            );
            if is_connect_class {
                return SingleAttempt::ConnectionRetryable(pf);
            }
            return SingleAttempt::Permanent(pf);
        }
    };

    sink.emit(TranscriptionEvent::ResponseComplete {
        total: response.body.len() as u64,
        t: Instant::now(),
    });

    // ── Classify HTTP status ──────────────────────────────────
    if (200..300).contains(&response.status) {
        SingleAttempt::Success(response)
    } else if is_data_retryable_status(response.status) {
        // 5xx / 429: data-phase retryable.  The counters describe
        // the *data* budget, which is the one this failure will
        // exhaust if the server never recovers.
        let retry_after = parse_retry_after_at(&response.headers, std::time::SystemTime::now());
        SingleAttempt::DataRetryable {
            failure: PipelineFailure {
                retry_after,
                ..PipelineFailure::new(
                    req.provider_name.clone(),
                    req.phase,
                    data_attempt,
                    max_data_attempts,
                    req.url.clone(),
                    PipelineFailureKind::HttpStatus {
                        status: response.status,
                        body: String::from_utf8_lossy(&response.body).into_owned(),
                    },
                )
            },
            retry_after,
        }
    } else {
        // 4xx and other non-2xx: permanent.
        SingleAttempt::Permanent(PipelineFailure::new(
            req.provider_name.clone(),
            req.phase,
            attempt_num,
            max_attempts,
            req.url.clone(),
            PipelineFailureKind::HttpStatus {
                status: response.status,
                body: String::from_utf8_lossy(&response.body).into_owned(),
            },
        ))
    }
}

const MAX_HTTP_CLIENT_POLICIES: usize = 16;
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const HTTP2_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(20);
const HTTP2_KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Default)]
struct HttpClients {
    by_budget: BTreeMap<Duration, reqwest::Client>,
    recent: VecDeque<Duration>,
}

impl HttpClients {
    fn get(&mut self, budget: Duration) -> Option<reqwest::Client> {
        let client = self.by_budget.get(&budget)?.clone();
        self.recent.retain(|key| *key != budget);
        self.recent.push_back(budget);
        Some(client)
    }

    #[cfg(test)]
    fn client(&mut self, budget: Duration) -> Result<reqwest::Client, String> {
        self.client_with_builder(budget, build_client_with_connect_timeout)
    }

    #[cfg(test)]
    fn client_with_builder(
        &mut self,
        budget: Duration,
        builder: fn(Duration) -> Result<reqwest::Client, String>,
    ) -> Result<reqwest::Client, String> {
        if let Some(client) = self.get(budget) {
            return Ok(client);
        }
        let client = builder(budget)?;
        Ok(self.insert_built(budget, client))
    }

    fn insert_built(&mut self, budget: Duration, client: reqwest::Client) -> reqwest::Client {
        if let Some(existing) = self.get(budget) {
            return existing;
        }
        if self.by_budget.len() >= MAX_HTTP_CLIENT_POLICIES {
            // The default first attempt is the socket warmed during dictation.
            // Do not let unrelated custom budgets evict it mid-recording.
            let protected = Duration::from_secs(CONNECTION_BUDGETS_SECS[0]);
            if let Some(position) = self.recent.iter().position(|key| *key != protected) {
                if let Some(oldest) = self.recent.remove(position) {
                    self.by_budget.remove(&oldest);
                }
            }
        }
        self.by_budget.insert(budget, client.clone());
        self.recent.push_back(budget);
        client
    }
}

fn clients() -> &'static Mutex<HttpClients> {
    static CLIENTS: OnceLock<Mutex<HttpClients>> = OnceLock::new();
    CLIENTS.get_or_init(|| Mutex::new(HttpClients::default()))
}

fn lock_clients() -> std::sync::MutexGuard<'static, HttpClients> {
    // A panic while updating retention metadata cannot invalidate a reqwest client;
    // rebuild the recency list from actual keys if a previous holder poisoned it.
    match clients().lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            guard.recent = guard.by_budget.keys().copied().collect();
            guard
        }
    }
}

/// Cold certificate-store setup runs off the async worker. Cancellation
/// drops the waiting future, while an already-running build may finish in
/// the blocking pool and retain the reusable client for later calls.
async fn http_client(
    budget: Duration,
    cancel: &CancellationToken,
) -> Result<reqwest::Client, String> {
    http_client_with_builder(budget, cancel, build_client_with_connect_timeout).await
}

async fn http_client_with_builder(
    budget: Duration,
    cancel: &CancellationToken,
    builder: fn(Duration) -> Result<reqwest::Client, String>,
) -> Result<reqwest::Client, String> {
    if let Some(client) = lock_clients().get(budget) {
        return Ok(client);
    }
    let build = tokio::task::spawn_blocking(move || {
        let constructed = builder(budget)?;
        Ok::<_, String>(lock_clients().insert_built(budget, constructed))
    });
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err("HTTP client construction cancelled".into()),
        result = build => result.map_err(|e| format!("HTTP client build task failed: {e}"))?,
    }
}

/// Best-effort unauthenticated connection setup while an encoded pipe is
/// collected. A completed HEAD can return a socket to the same pool as POST.
pub(crate) async fn prewarm_http(
    endpoint: &str,
    schedule: &RetrySchedule,
    cancel: &CancellationToken,
) {
    let Some(&budget) = schedule.connection_budgets.first() else {
        return;
    };
    if cancel.is_cancelled() {
        return;
    }
    let warmup = async {
        let client = http_client(budget, cancel).await?;
        let response = client
            .head(endpoint)
            .timeout(budget)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        response.bytes().await.map_err(|e| e.to_string())?;
        Ok::<(), String>(())
    };
    tokio::select! {
        biased;
        _ = cancel.cancelled() => {}
        result = tokio::time::timeout(budget, warmup) => match result {
            Ok(Ok(())) => log::debug!("HTTP connection warmup completed for {endpoint}"),
            Ok(Err(e)) => log::debug!("HTTP connection warmup unavailable for {endpoint}: {e}"),
            Err(_) => log::debug!("HTTP connection warmup timed out for {endpoint}"),
        }
    }
}

/// Build a [`reqwest::Client`] for one exact per-attempt connect budget.
fn build_client_with_connect_timeout(connect_timeout: Duration) -> Result<reqwest::Client, String> {
    crate::perf_counters::incr(crate::perf_counters::Counter::HttpClientBuilds);
    let builder = reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .pool_idle_timeout(POOL_IDLE_TIMEOUT)
        .http2_keep_alive_interval(HTTP2_KEEPALIVE_INTERVAL)
        .http2_keep_alive_timeout(HTTP2_KEEPALIVE_TIMEOUT)
        .http2_keep_alive_while_idle(true)
        .tcp_keepalive(Duration::from_secs(5))
        .tcp_keepalive_interval(Duration::from_secs(1))
        .tcp_keepalive_retries(3);

    #[cfg(target_os = "linux")]
    let builder = builder.tcp_user_timeout(Duration::from_secs(3));

    builder.build().map_err(|e| e.to_string())
}

/// Walk a reqwest error's source chain looking for a kernel-level
/// `io::ErrorKind::TimedOut` — the signal that
/// `TCP_USER_TIMEOUT` or unanswered keepalives killed the socket.
/// Used by [`run_single_http_attempt`] to keep
/// kernel-induced failures in the connection-retryable bucket
/// even when reqwest's own `is_connect`/`is_timeout` flags miss.
fn matches_kernel_tcp(err: &reqwest::Error) -> bool {
    use std::error::Error as _;
    let mut current: Option<&dyn std::error::Error> = err.source();
    while let Some(e) = current {
        if let Some(io) = e.downcast_ref::<std::io::Error>() {
            if io.kind() == std::io::ErrorKind::TimedOut {
                return true;
            }
        }
        current = e.source();
    }
    false
}

/// Build a [`PipelineFailure`] tagged as a cancellation.
fn build_cancellation_failure(
    req: &Request,
    attempt_num: u32,
    max_attempts: u32,
) -> PipelineFailure {
    PipelineFailure::new(
        req.provider_name.clone(),
        req.phase,
        attempt_num,
        max_attempts,
        req.url.clone(),
        PipelineFailureKind::Network {
            kind: NetworkKind::Other,
            timer: Some(TimerLabel {
                name: "cancelled".to_string(),
                budget: "0s".to_string(),
            }),
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "cancelled by caller",
            )),
        },
    )
}

/// Fallback [`PipelineFailure`] used only on an unreachable path
/// (every attempt should set `last_failure`, but if the loop
/// somehow exits without one we surface a structural failure
/// rather than panicking).
fn build_generic_exhausted_failure(req: &Request, max_attempts: u32) -> PipelineFailure {
    PipelineFailure::new(
        req.provider_name.clone(),
        req.phase,
        max_attempts,
        max_attempts,
        req.url.clone(),
        PipelineFailureKind::Decode(format!(
            "transport: all {} connection attempts exhausted \
             with no recorded cause",
            max_attempts
        )),
    )
}

/// Open a WebSocket upgrade connection through the transport.
///
/// Same retry / event / cancellation semantics as [`http_request`].
/// Returns a connected [`tokio_tungstenite::WebSocketStream`] on
/// success.
///
/// # Implementation notes (Step 6)
///
/// Uses the same connection-retry schedule as [`http_request`]
/// ([`CONNECTION_BUDGETS_SECS`]), wrapping each
/// [`tokio_tungstenite::connect_async`] attempt in
/// `tokio::time::timeout(connect_budget)` so a hung TCP SYN does
/// not block beyond the budget for that attempt.
///
/// WebSocket upgrades have no "data retry" concept — once the
/// upgrade succeeds, the caller drives the WS frame loop and
/// any transient frame error is its concern (typically end of
/// session).  Only connection-phase retries apply.
///
/// `req.body` must be [`RequestBody::Empty`] — the WS upgrade
/// handshake is a `GET` with `Connection: Upgrade` headers and
/// no body.  Any other body shape returns a `Decode` failure.
pub async fn ws_upgrade(
    req: Request,
    sink: &Arc<dyn TelemetrySink>,
    cancel: CancellationToken,
) -> Result<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    PipelineFailure,
> {
    let max_connection_attempts = req.retry_schedule.connection_budgets.len() as u32;
    let mut last_failure: Option<PipelineFailure> = None;
    let started = Instant::now();
    let wait_started = req
        .retry_schedule
        .wait_started
        .as_ref()
        .map_or(started, |shared| *shared.get_or_init(|| started));
    let deadline = req
        .retry_schedule
        .max_data_wait
        .map(|limit| wait_started + limit);

    if !matches!(req.body, RequestBody::Empty) {
        return Err(PipelineFailure::new(
            req.provider_name.clone(),
            req.phase,
            1,
            1,
            req.url.clone(),
            PipelineFailureKind::Decode(
                "transport::ws_upgrade: only RequestBody::Empty is supported \
                 (a WebSocket upgrade handshake carries no body)"
                    .into(),
            ),
        ));
    }

    sink.emit(TranscriptionEvent::RequestStarted {
        endpoint: req.url.clone(),
        t: Instant::now(),
    });

    for (idx, &connect_budget) in req.retry_schedule.connection_budgets.iter().enumerate() {
        if idx > 0 && deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            break;
        }
        let attempt_num = (idx as u32) + 1;

        // Retry telemetry on attempts past the first.
        if idx > 0 {
            let reason = last_failure
                .as_ref()
                .map(|f| f.to_string())
                .unwrap_or_else(|| "ws connection failed".into());
            crate::perf_counters::incr(crate::perf_counters::Counter::ConnectionRetries);
            sink.emit(TranscriptionEvent::RetryScheduled {
                kind: crate::telemetry::RetryKind::Connection,
                attempt: idx as u32,
                max: max_connection_attempts.saturating_sub(1),
                reason,
                delay: Duration::ZERO,
                t: Instant::now(),
            });
        }

        if cancel.is_cancelled() {
            let pf = build_cancellation_failure(&req, attempt_num, max_connection_attempts);
            sink.emit(TranscriptionEvent::RequestCompleted {
                success: false,
                t: Instant::now(),
            });
            return Err(pf);
        }

        match run_single_ws_attempt(
            &req,
            connect_budget,
            &cancel,
            attempt_num,
            max_connection_attempts,
        )
        .await
        {
            WsAttempt::Success(stream) => {
                sink.emit(TranscriptionEvent::ConnectionEstablished { t: Instant::now() });
                sink.emit(TranscriptionEvent::RequestCompleted {
                    success: true,
                    t: Instant::now(),
                });
                return Ok(stream);
            }
            WsAttempt::Cancelled(pf) => {
                sink.emit(TranscriptionEvent::RequestCompleted {
                    success: false,
                    t: Instant::now(),
                });
                return Err(pf);
            }
            WsAttempt::ConnectionRetryable(pf) => {
                last_failure = Some(pf);
                continue;
            }
            WsAttempt::Permanent(pf)
                if req.retry_schedule.retry_ws_busy && pf.is_fallback_worthy() =>
            {
                let Some(&slot) = req.retry_schedule.data_backoffs.get(idx) else {
                    sink.emit(TranscriptionEvent::RequestCompleted {
                        success: false,
                        t: Instant::now(),
                    });
                    return Err(pf);
                };
                let delay = pf
                    .retry_after
                    .map(|d| d.min(Duration::from_secs(RETRY_AFTER_CAP_SECS)))
                    .unwrap_or(slot);
                if deadline.is_some_and(|deadline| {
                    delay >= deadline.saturating_duration_since(Instant::now())
                }) {
                    sink.emit(TranscriptionEvent::RequestCompleted {
                        success: false,
                        t: Instant::now(),
                    });
                    return Err(pf);
                }
                log::info!(
                    "{} busy — retrying WebSocket handshake in {}s",
                    req.provider_name,
                    delay.as_secs()
                );
                crate::perf_counters::incr(crate::perf_counters::Counter::DataRetries);
                sink.emit(TranscriptionEvent::RetryScheduled {
                    kind: crate::telemetry::RetryKind::Data,
                    attempt: attempt_num,
                    max: req.retry_schedule.data_backoffs.len() as u32,
                    reason: pf.to_string(),
                    delay,
                    t: Instant::now(),
                });
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return Err(build_cancellation_failure(&req, attempt_num, max_connection_attempts)),
                    _ = tokio::time::sleep(delay) => {}
                }
                last_failure = Some(pf);
            }
            WsAttempt::Permanent(pf) => {
                sink.emit(TranscriptionEvent::RequestCompleted {
                    success: false,
                    t: Instant::now(),
                });
                return Err(pf);
            }
        }
    }

    sink.emit(TranscriptionEvent::RequestCompleted {
        success: false,
        t: Instant::now(),
    });
    Err(last_failure
        .unwrap_or_else(|| build_generic_exhausted_failure(&req, max_connection_attempts)))
}

/// Outcome of a single WebSocket upgrade attempt.
enum WsAttempt {
    Success(
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ),
    Cancelled(PipelineFailure),
    ConnectionRetryable(PipelineFailure),
    Permanent(PipelineFailure),
}

/// Issue one WebSocket upgrade attempt against `req.url`, applying
/// `req.headers` to the handshake and `connect_budget` as an outer
/// wall-clock cap.
async fn run_single_ws_attempt(
    req: &Request,
    connect_budget: Duration,
    cancel: &CancellationToken,
    attempt_num: u32,
    max_attempts: u32,
) -> WsAttempt {
    // Parse URL for the Host header (tungstenite needs it
    // explicitly when we build a custom Request).
    let parsed = match url::Url::parse(&req.url) {
        Ok(u) => u,
        Err(e) => {
            return WsAttempt::Permanent(PipelineFailure::new(
                req.provider_name.clone(),
                req.phase,
                attempt_num,
                max_attempts,
                req.url.clone(),
                PipelineFailureKind::Decode(format!("invalid WebSocket URL: {}", e)),
            ));
        }
    };
    let host = match parsed.host_str() {
        Some(h) => h.to_string(),
        None => {
            return WsAttempt::Permanent(PipelineFailure::new(
                req.provider_name.clone(),
                req.phase,
                attempt_num,
                max_attempts,
                req.url.clone(),
                PipelineFailureKind::Decode("WebSocket URL is missing a host component".into()),
            ));
        }
    };

    // Build the upgrade request.  Caller-supplied headers are
    // applied verbatim AFTER the mandatory upgrade machinery so
    // the caller can override anything (rare; typically just
    // adds Authorization).
    let mut builder = tokio_tungstenite::tungstenite::http::Request::builder()
        .uri(&req.url)
        .header("Host", &host)
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header(
            "Sec-WebSocket-Key",
            tokio_tungstenite::tungstenite::handshake::client::generate_key(),
        );
    for (name, value) in &req.headers {
        builder = builder.header(name, value);
    }
    let request = match builder.body(()) {
        Ok(r) => r,
        Err(e) => {
            return WsAttempt::Permanent(PipelineFailure::new(
                req.provider_name.clone(),
                req.phase,
                attempt_num,
                max_attempts,
                req.url.clone(),
                PipelineFailureKind::Decode(format!("WebSocket request build failed: {}", e)),
            ));
        }
    };

    let connect_fut = tokio_tungstenite::connect_async(request);
    let capped = tokio::time::timeout(connect_budget, connect_fut);

    let outcome = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            return WsAttempt::Cancelled(build_cancellation_failure(
                req, attempt_num, max_attempts,
            ));
        }
        result = capped => result,
    };

    match outcome {
        Err(_elapsed) => WsAttempt::ConnectionRetryable(PipelineFailure::new(
            req.provider_name.clone(),
            req.phase,
            attempt_num,
            max_attempts,
            req.url.clone(),
            PipelineFailureKind::Network {
                kind: NetworkKind::Connect,
                timer: Some(TimerLabel::from_duration("connect_timeout", connect_budget)),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!(
                        "WebSocket upgrade did not complete within {}s",
                        connect_budget.as_secs()
                    ),
                )),
            },
        )),
        Ok(Ok((stream, _response))) => WsAttempt::Success(stream),
        Ok(Err(err)) => {
            if let tokio_tungstenite::tungstenite::Error::Http(response) = &err {
                let status = response.status().as_u16();
                let headers: Vec<(String, String)> = response
                    .headers()
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.as_str().to_string(),
                            v.to_str().unwrap_or_default().to_string(),
                        )
                    })
                    .collect();
                let retry_after = parse_retry_after_at(&headers, std::time::SystemTime::now());
                return WsAttempt::Permanent(PipelineFailure {
                    retry_after,
                    ..PipelineFailure::new(
                        req.provider_name.clone(),
                        req.phase,
                        attempt_num,
                        max_attempts,
                        req.url.clone(),
                        PipelineFailureKind::HttpStatus {
                            status,
                            body: err.to_string(),
                        },
                    )
                });
            }
            // tungstenite errors: classify into connect-retryable
            // (network / IO / TLS) vs permanent (HTTP 4xx, protocol
            // violation, decode).
            let is_connect_class = matches!(
                err,
                tokio_tungstenite::tungstenite::Error::Io(_)
                    | tokio_tungstenite::tungstenite::Error::Tls(_)
                    | tokio_tungstenite::tungstenite::Error::ConnectionClosed
                    | tokio_tungstenite::tungstenite::Error::AlreadyClosed
            );
            let pf = PipelineFailure::new(
                req.provider_name.clone(),
                req.phase,
                attempt_num,
                max_attempts,
                req.url.clone(),
                PipelineFailureKind::Network {
                    kind: if is_connect_class {
                        NetworkKind::Connect
                    } else {
                        NetworkKind::Other
                    },
                    timer: Some(TimerLabel::from_duration("connect_timeout", connect_budget)),
                    source: Box::new(std::io::Error::other(err.to_string())),
                },
            );
            if is_connect_class {
                WsAttempt::ConnectionRetryable(pf)
            } else {
                WsAttempt::Permanent(pf)
            }
        }
    }
}

#[cfg(test)]
mod retry_after_tests {
    use super::*;

    #[tokio::test]
    async fn websocket_upgrade_rejects_audio_body_before_connecting() {
        let sink: Arc<dyn TelemetrySink> = Arc::new(crate::telemetry::NoOpSink);
        let error = ws_upgrade(
            Request {
                method: Method::Get,
                url: "ws://127.0.0.1:1/v1/realtime".into(),
                headers: Vec::new(),
                body: RequestBody::Bytes(Arc::new(vec![1, 2, 3])),
                provider: Provider::OpenAI,
                provider_name: "OpenAI".into(),
                phase: crate::error::PipelinePhase::Request,
                wall_clock: None,
                retry_schedule: RetrySchedule::default(),
            },
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect_err("a WebSocket upgrade cannot send audio in the handshake");
        assert_eq!((error.attempts, error.max_attempts), (1, 1));
        assert!(
            matches!(error.kind, PipelineFailureKind::Decode(ref message) if message == "transport::ws_upgrade: only RequestBody::Empty is supported (a WebSocket upgrade handshake carries no body)")
        );
    }

    #[test]
    fn http_date_retry_after_is_a_delay() {
        let now = std::time::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let date = httpdate::fmt_http_date(now + Duration::from_secs(30));
        let headers = vec![("Retry-After".into(), date)];
        assert_eq!(
            parse_retry_after_at(&headers, now),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            parse_retry_after_at(&[("retry-after".into(), "7".into())], now),
            Some(Duration::from_secs(7))
        );
        assert_eq!(
            parse_retry_after_at(
                &[(
                    "Retry-After".into(),
                    httpdate::fmt_http_date(now - Duration::from_secs(1))
                )],
                now
            ),
            Some(Duration::ZERO)
        );
    }
}
