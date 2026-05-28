//! Transport primitives for transcription API calls.
//!
//! This module contains the resilience and connectivity concerns
//! shared by every provider, across both batch (HTTP POST) and
//! realtime (WebSocket) protocols.
//!
//! # Architecture
//!
//! All outgoing connections to transcription providers funnel
//! through this module.  The public surface is intentionally small:
//!
//! - [`http_request`] — the single entry point for HTTP requests
//!   (batch transcription POST, model validation GET, model list
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
//! - [`http`]: reqwest client configuration, per-request
//!   proportional timeout, progress-reporting request body, model
//!   validation, and model-error enrichment.  Functions here will
//!   either move behind [`http_request`] or become private helpers
//!   over the course of the transport consolidation.
//! - [`retry`]: the legacy retry primitive used by batch HTTP calls
//!   and realtime WebSocket upgrade handshakes.  Being absorbed
//!   into [`http_request`] / [`ws_upgrade`].
//! - [`validate_cache`]: disk-backed memoization of `/v1/models`
//!   preflight results — unchanged.
//! - [`ws`]: shared WebSocket helpers — being populated as part of
//!   the consolidation.

pub(crate) mod http;
pub(crate) mod retry;
pub(crate) mod validate_cache;
pub(crate) mod ws;

use crate::config::Provider;
use crate::error::{NetworkKind, PipelineFailure, PipelineFailureKind, TimerLabel};
use crate::telemetry::{TelemetrySink, TranscriptionEvent};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

// ── Internal retry schedules ────────────────────────────────────────

/// Per-attempt budgets (seconds) for the connection-phase retry
/// loop.  Length = total connection attempts; the value at index
/// `i` is the connect-timeout used on attempt `i+1`.  Matches the
/// historical `VALIDATE_BUDGET_SECS` schedule so the consolidated
/// transport doesn't regress validate-path behaviour.
const CONNECTION_BUDGETS_SECS: [u64; 5] = [2, 5, 8, 11, 15];

/// Maximum number of data-phase attempts (the initial call plus
/// `DATA_MAX_ATTEMPTS - 1` retries) when the server returns 5xx or
/// a transient body-decode error after headers arrived.
const DATA_MAX_ATTEMPTS: u32 = 3;

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
/// # Implementation notes (Step 2)
///
/// The body runs two nested retry loops:
///
/// 1. **Connection retry** — outer loop, attempts indexed against
///    [`CONNECTION_BUDGETS_SECS`].  Each attempt builds a fresh
///    reqwest client with the current per-attempt connect timeout
///    and tries once.  On a connect-class failure (DNS, ECONNREFUSED,
///    TCP/TLS timeout) the loop emits
///    [`RetryKind::Connection`] and advances to the next attempt.
///    On any other outcome it forwards to the data-retry decision.
///
/// 2. **Data retry** — inner check on a successful HTTP transaction.
///    A 5xx response triggers a fresh outer attempt counted as a
///    [`RetryKind::Data`] retry, up to [`DATA_MAX_ATTEMPTS`].
///
/// Cancellation is wired via `tokio::select!` so the call returns
/// promptly the moment the token fires.
pub async fn http_request(
    req: Request,
    sink: &Arc<dyn TelemetrySink>,
    cancel: CancellationToken,
) -> Result<Response, PipelineFailure> {
    let max_connection_attempts = CONNECTION_BUDGETS_SECS.len() as u32;
    let mut data_retries_used: u32 = 0;
    let mut last_failure: Option<PipelineFailure> = None;

    sink.emit(TranscriptionEvent::RequestStarted {
        endpoint: req.url.clone(),
        t: Instant::now(),
    });

    for (idx, &budget_secs) in CONNECTION_BUDGETS_SECS.iter().enumerate() {
        let connect_budget = Duration::from_secs(budget_secs);
        let attempt_num = (idx as u32) + 1;

        // Emit a `RetryScheduled` BEFORE every attempt past the
        // first.  This is the connection-phase retry signal; the
        // data-phase retry is signalled separately below when a
        // 5xx triggers an outer retry.
        if idx > 0 {
            let reason = last_failure
                .as_ref()
                .map(|f| f.to_string())
                .unwrap_or_else(|| "connection failed".into());
            sink.emit(TranscriptionEvent::RetryScheduled {
                attempt: idx as u32,
                max: max_connection_attempts.saturating_sub(1),
                reason,
                t: Instant::now(),
            });
        }

        // ── Bail early if the caller cancelled. ─────────────────
        if cancel.is_cancelled() {
            let pf = build_cancellation_failure(&req, attempt_num, max_connection_attempts);
            sink.emit(TranscriptionEvent::RequestCompleted {
                success: false,
                t: Instant::now(),
            });
            return Err(pf);
        }

        // ── Single attempt ──────────────────────────────────────
        let attempt_outcome = run_single_http_attempt(
            &req,
            connect_budget,
            sink,
            &cancel,
            attempt_num,
            max_connection_attempts,
        )
        .await;

        match attempt_outcome {
            SingleAttempt::Success(resp) => {
                sink.emit(TranscriptionEvent::RequestCompleted {
                    success: true,
                    t: Instant::now(),
                });
                return Ok(resp);
            }
            SingleAttempt::Cancelled(pf) => {
                sink.emit(TranscriptionEvent::RequestCompleted {
                    success: false,
                    t: Instant::now(),
                });
                return Err(pf);
            }
            SingleAttempt::ConnectionRetryable(pf) => {
                last_failure = Some(pf);
                continue;
            }
            SingleAttempt::DataRetryable(pf) => {
                data_retries_used = data_retries_used.saturating_add(1);
                if data_retries_used >= DATA_MAX_ATTEMPTS.saturating_sub(1) {
                    sink.emit(TranscriptionEvent::RequestCompleted {
                        success: false,
                        t: Instant::now(),
                    });
                    return Err(pf);
                }
                let reason = pf.to_string();
                sink.emit(TranscriptionEvent::RetryScheduled {
                    attempt: data_retries_used,
                    max: DATA_MAX_ATTEMPTS.saturating_sub(1),
                    reason,
                    t: Instant::now(),
                });
                last_failure = Some(pf);
                continue;
            }
            SingleAttempt::Permanent(pf) => {
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
    /// Server returned a 5xx — eligible for a data retry (up to
    /// [`DATA_MAX_ATTEMPTS`]).
    DataRetryable(PipelineFailure),
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
async fn run_single_http_attempt(
    req: &Request,
    connect_budget: Duration,
    sink: &Arc<dyn TelemetrySink>,
    cancel: &CancellationToken,
    attempt_num: u32,
    max_attempts: u32,
) -> SingleAttempt {
    // ── Build a per-attempt client with the right connect timeout ──
    let client = match build_client_with_connect_timeout(connect_budget) {
        Ok(c) => c,
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

    // Wall-clock cap on the whole attempt.  Two concerns layered:
    //
    //   * `connect_budget` (the per-attempt growing connect window)
    //     wraps the whole `send_fut` because reqwest's
    //     `connect_timeout` only fires AFTER TCP is established
    //     (it's the TLS-handshake window), not for stuck SYNs.
    //     Without this outer cap a blackhole destination would
    //     hang for the OS-default ~125s TCP SYN timeout.
    //
    //   * The caller's `wall_clock` (if any) is already applied
    //     via `.timeout(budget)` on the RequestBuilder — that
    //     governs successful-connect, slow-server cases.
    //
    // The outer cap below uses `connect_budget` so the
    // growing-budget schedule actually grows the wait per attempt.
    let send_with_cap = tokio::time::timeout(connect_budget, send_fut);

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
                    timer: Some(TimerLabel::from_duration("connect_timeout", connect_budget)),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!(
                            "connection attempt did not complete \
                             within {}s",
                            connect_budget.as_secs()
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
    } else if (500..600).contains(&response.status) {
        // 5xx: data-phase retryable.
        SingleAttempt::DataRetryable(PipelineFailure::new(
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

/// Build a [`reqwest::Client`] with the given per-attempt connect
/// timeout.  Other client settings (TCP keepalive, user_timeout)
/// match the shared client builder in [`http::build_client`].
fn build_client_with_connect_timeout(connect_timeout: Duration) -> Result<reqwest::Client, String> {
    let builder = reqwest::Client::builder()
        .connect_timeout(connect_timeout)
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
/// # Status — Step 1 stub
///
/// Body is `unimplemented!()`; Step 6 of the transport-consolidation
/// plan implements it.  Step 7 migrates the realtime modules to
/// call this function.
pub async fn ws_upgrade(
    _req: Request,
    _sink: &Arc<dyn TelemetrySink>,
    _cancel: CancellationToken,
) -> Result<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    PipelineFailure,
> {
    unimplemented!(
        "transport::ws_upgrade is a Step-1 stub; Step 6 of the \
         transport-consolidation plan will implement it"
    )
}
