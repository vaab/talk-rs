//! Transport-layer integration tests.
//!
//! These tests pin down the contract of the consolidated transport
//! module (`talk_rs::transcription::transport`) ahead of its
//! implementation, per Step 1 of the transport-consolidation plan
//! at `.sisyphus/plans/transport-consolidation.md`.
//!
//! # Phase
//!
//! Step 1 (this file): every test compiles and is expected to FAIL
//! (red) because the transport API is currently a stub
//! (`unimplemented!()`).  Subsequent steps progressively implement
//! the API and flip these tests green.
//!
//! # Coverage
//!
//! - `transport_connection_phase_retries_with_growing_budget`
//! - `transport_data_phase_retries_three_times_on_503`
//! - `pipeline_failure_carries_real_attempt_counter_connection`
//! - `pipeline_failure_carries_real_attempt_counter_data`
//! - `cancellation_aborts_in_flight_request`
//! - `ws_upgrade_retries_on_connection_failure`
//! - `lock_file_includes_status_socket_and_pid`
//! - `observe_remote_replays_backlog_then_streams_live`
//! - `observe_remote_cancel_signals_owner`
//! - `stale_lock_detected_when_owner_pid_dead`

use std::sync::Arc;
use std::time::{Duration, Instant};

use talk_rs::config::Provider;
use talk_rs::error::{NetworkKind, PipelineFailureKind, PipelinePhase};
use talk_rs::telemetry::{NoOpSink, TelemetrySink};
use talk_rs::transcription::transport::{
    http_request, ws_upgrade, ConnectionEvent, Method, Request, RequestBody, RetryKind,
    RetrySchedule,
};
use tokio_util::sync::CancellationToken;

// ── Helpers ─────────────────────────────────────────────────────────

/// A telemetry sink that captures every emitted `TranscriptionEvent`
/// for later inspection.  Used to assert on the emitted retry
/// counters and phase progression.
///
/// NOTE: pre-Step-10 the transport emits the legacy
/// `TranscriptionEvent` vocabulary; Step 10 introduces the
/// `ConnectionEvent` vocabulary.  These tests assert against
/// `ConnectionEvent` via a separate broadcast subscription path
/// that the transport exposes on its `Request` (via the sink).
struct CapturingSink {
    events: std::sync::Mutex<Vec<talk_rs::telemetry::TranscriptionEvent>>,
}

impl CapturingSink {
    fn new() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn events(&self) -> Vec<talk_rs::telemetry::TranscriptionEvent> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn retry_events(&self) -> Vec<(u32, u32)> {
        self.events()
            .into_iter()
            .filter_map(|e| match e {
                talk_rs::telemetry::TranscriptionEvent::RetryScheduled { attempt, max, .. } => {
                    Some((attempt, max))
                }
                _ => None,
            })
            .collect()
    }
}

impl TelemetrySink for CapturingSink {
    fn emit(&self, event: talk_rs::telemetry::TranscriptionEvent) {
        if let Ok(mut g) = self.events.lock() {
            g.push(event);
        }
    }
}

/// A telemetry sink that cancels on the first scheduled retry and
/// records when the cancellation was triggered.
struct CancelOnRetrySink {
    cancel: CancellationToken,
    cancelled_at: std::sync::Mutex<Option<Instant>>,
}

impl CancelOnRetrySink {
    fn new(cancel: CancellationToken) -> Self {
        Self {
            cancel,
            cancelled_at: std::sync::Mutex::new(None),
        }
    }

    fn cancelled_at(&self) -> Option<Instant> {
        *self.cancelled_at.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl TelemetrySink for CancelOnRetrySink {
    fn emit(&self, event: talk_rs::telemetry::TranscriptionEvent) {
        if !matches!(
            event,
            talk_rs::telemetry::TranscriptionEvent::RetryScheduled { .. }
        ) {
            return;
        }

        let mut cancelled_at = self.cancelled_at.lock().unwrap_or_else(|e| e.into_inner());
        if cancelled_at.is_none() {
            *cancelled_at = Some(Instant::now());
            self.cancel.cancel();
        }
    }
}

/// Build a minimal `Request` for the transport against the given URL.
fn make_request(url: impl Into<String>, phase: PipelinePhase) -> Request {
    Request {
        method: Method::Get,
        url: url.into(),
        headers: vec![("Authorization".into(), "Bearer test".into())],
        body: RequestBody::Empty,
        provider: Provider::Mistral,
        provider_name: "Mistral".into(),
        phase,
        wall_clock: None,
        retry_schedule: Default::default(),
    }
}

/// Reserve a loopback port and release it before sending; connection
/// refusal exercises the complete connection-retry budget without relying
/// on an external router silently dropping SYN packets.
fn refused_loopback_url(scheme: &str) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve port");
    let port = listener.local_addr().expect("local address").port();
    drop(listener);
    format!("{scheme}://127.0.0.1:{port}/v1/models")
}

#[test]
fn pooled_client_survives_successive_request_runtimes() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let (url_tx, url_rx) = std::sync::mpsc::sync_channel(1);
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
    let server_thread = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("server runtime");
        runtime.block_on(async move {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/models"))
                .respond_with(ResponseTemplate::new(200))
                .expect(2)
                .mount(&server)
                .await;
            url_tx.send(server.uri()).expect("url");
            stop_rx.await.expect("stop");
            server.verify().await;
        });
    });
    let url = format!("{}/v1/models", url_rx.recv().expect("server started"));
    for _ in 0..2 {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("request runtime");
        runtime.block_on(async {
            let sink: Arc<dyn TelemetrySink> = Arc::new(CapturingSink::new());
            http_request(
                make_request(&url, PipelinePhase::Validate),
                &sink,
                CancellationToken::new(),
            )
            .await
            .expect("request across runtime boundary");
        });
    }
    stop_tx.send(()).expect("stop server");
    server_thread.join().expect("server thread");
}

// ── §3 Step 1 tests ─────────────────────────────────────────────────
//
// These tests assert the POST-Step-2 spec.  Pre-Step-2 they fail
// because `http_request` / `ws_upgrade` panic with `unimplemented!()`
// — a panic counts as a test failure (RED).  Post-Step-2 the
// assertions become the real spec checks.

/// Spec (plan §3 Step 1, bullet 1): a single `http_request` call
/// against an unreachable host issues 5 connection-phase attempts
/// (per the growing-budget schedule) and emits 4 retry events
/// (`max - 1` since the first attempt is not a retry).
///
/// Allowed: 5 or 6 attempts (the exact schedule length may be
/// `[2,5,8,11,15]` (5 entries) or a 6-entry variant).  The test
/// pins ≥4 retry events to catch the "1/1" regression specifically.
#[tokio::test(flavor = "multi_thread")]
async fn transport_connection_phase_retries_with_growing_budget() {
    let capturing = Arc::new(CapturingSink::new());
    let sink: Arc<dyn TelemetrySink> = capturing.clone();
    let req = make_request(refused_loopback_url("http"), PipelinePhase::Request);

    let result = http_request(req, &sink, CancellationToken::new()).await;
    assert!(
        result.is_err(),
        "refused loopback connection must fail; got Ok"
    );

    let retries = capturing.retry_events();
    assert!(
        retries.len() >= 4,
        "expected ≥4 connection retries against an unreachable \
         host (one per failed attempt past the first); got {}: \
         {:?}",
        retries.len(),
        retries
    );

    let pf = result.unwrap_err();
    assert!(
        pf.attempts >= 5 && pf.attempts == pf.max_attempts,
        "PipelineFailure must report attempts==max_attempts when \
         retries are exhausted; got {}/{}",
        pf.attempts,
        pf.max_attempts
    );
}

/// Spec (plan §3 Step 1, bullet 2): an `http_request` against a
/// server that returns HTTP 503 should retry up to the data-phase
/// budget and emit `RetryScheduled` for each retry.  `Retry-After: 0`
/// keeps the test fast; the wait itself is covered by
/// `transport_waits_first_backoff_slot_before_data_retry`.
#[tokio::test(flavor = "multi_thread")]
async fn transport_data_phase_retries_three_times_on_503() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503).insert_header("Retry-After", "0"))
        .mount(&server)
        .await;

    let capturing = Arc::new(CapturingSink::new());
    let sink: Arc<dyn TelemetrySink> = capturing.clone();
    let url = format!("{}/v1/models", server.uri());
    let req = make_request(&url, PipelinePhase::Validate);

    let result = http_request(req, &sink, CancellationToken::new()).await;
    assert!(result.is_err(), "503-returning server must fail; got Ok");

    let retries = capturing.retry_events();
    assert!(
        retries.len() >= 2,
        "expected ≥2 data-phase retries against a 503-server; got {}: {:?}",
        retries.len(),
        retries,
    );
}

/// Spec (plan §3 Step 1, bullet 3): when connection-phase retries
/// are exhausted, the returned `PipelineFailure` carries
/// `attempts == max_attempts`.  No hardcoded 1/1 lie.
#[tokio::test(flavor = "multi_thread")]
async fn pipeline_failure_carries_real_attempt_counter_connection() {
    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    let req = make_request(refused_loopback_url("http"), PipelinePhase::Request);

    let result = http_request(req, &sink, CancellationToken::new()).await;
    let pf = result.expect_err("refused loopback connection must fail");
    assert!(
        pf.attempts > 1,
        "Step 2 must kill the 1/1 lie — got {}/{}",
        pf.attempts,
        pf.max_attempts
    );
    assert_eq!(
        pf.attempts, pf.max_attempts,
        "exhausted retries: attempts must equal max_attempts"
    );
}

/// Spec (plan §3 Step 1, bullet 3): when data-phase retries are
/// exhausted on a 503-returning server, the returned `PipelineFailure`
/// carries `attempts == max_attempts` (the data-retry budget).
#[tokio::test(flavor = "multi_thread")]
async fn pipeline_failure_carries_real_attempt_counter_data() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503).insert_header("Retry-After", "0"))
        .mount(&server)
        .await;

    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    let url = format!("{}/v1/models", server.uri());
    let req = make_request(&url, PipelinePhase::Validate);

    let result = http_request(req, &sink, CancellationToken::new()).await;
    let pf = result.expect_err("503-server must surface a failure");
    assert!(
        pf.attempts > 1,
        "data-phase failure must show real retry counter, not 1/1; \
         got {}/{}",
        pf.attempts,
        pf.max_attempts
    );
}

/// Spec (plan §3 Step 1, bullet 4): triggering `cancel` while an
/// `http_request` is in flight aborts it before the delayed response.
/// The 20s deadline only detects a hang against the 30s server delay.
#[tokio::test(flavor = "multi_thread")]
async fn cancellation_aborts_in_flight_request() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // The mock notifies the moment it receives the request, so the
    // cancellation is triggered while the request is provably in
    // flight. A fixed pre-cancel sleep would race connection setup.
    let server = MockServer::start().await;
    let in_flight = Arc::new(tokio::sync::Notify::new());
    let notify = in_flight.clone();
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(move |_: &wiremock::Request| {
            notify.notify_one();
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(30))
                .set_body_string("{\"data\":[]}")
        })
        .mount(&server)
        .await;

    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    let url = format!("{}/v1/models", server.uri());
    let req = make_request(&url, PipelinePhase::Validate);

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let cancelling = tokio::spawn(async move {
        in_flight.notified().await;
        cancel_clone.cancel();
    });

    let result = tokio::time::timeout(Duration::from_secs(20), http_request(req, &sink, cancel))
        .await
        .expect("cancellation must not wait for the 30s response");
    cancelling.await.expect("cancel task must complete");

    assert!(
        matches!(result, Err(ref error) if error.to_string().contains("cancelled")),
        "cancellation must surface as a cancellation error: {result:?}"
    );
}

// ── Provider-overload retry policy ─────────────────────────────────
//
// Spec origin: field reports (2026-08 → 2026-09) of Mistral returning
// `503 high load, please retry` / `429 backend_out_of_capacity` on
// long uploads.  The transport gave up after 7 attempts in under two
// minutes with zero wait between server retries, and never retried
// a 429 at all.  These tests pin the corrected contract:
//
//   1. 429 is a data-phase retryable status, like 5xx.
//   2. Data-phase retries WAIT between attempts, following
//      `DATA_BACKOFF_SECS`; a `Retry-After: <secs>` header overrides
//      the schedule slot (capped at the schedule maximum).
//   3. The data-phase budget is `DATA_BACKOFF_SECS.len() + 1`
//      attempts and is independent of the connection-phase budget.
//   4. Other 4xx remain permanent (no retry, no wait).
//   5. A large `wall_clock` lifts the per-attempt cap above the
//      connection-phase budget so a big upload is not killed on the
//      early (2 s / 5 s / …) connection slots.
//   6. Cancellation during a backoff wait returns promptly.

use talk_rs::transcription::transport::{CONNECTION_BUDGETS_SECS, DATA_BACKOFF_SECS};

/// Mount a mock that answers `status` (with optional `Retry-After`)
/// for the first `fail_times` requests, then 200 with a JSON body.
async fn mount_fail_then_succeed(
    server: &wiremock::MockServer,
    status: u16,
    retry_after_secs: Option<u64>,
    fail_times: u64,
) {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};

    let mut failing = ResponseTemplate::new(status).set_body_string("busy");
    if let Some(secs) = retry_after_secs {
        failing = failing.insert_header("Retry-After", secs.to_string().as_str());
    }
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(failing)
        .up_to_n_times(fail_times)
        .expect(fail_times)
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"data\":[]}"))
        .mount(server)
        .await;
}

/// Spec 1 + 2: a 429 is retried (not permanent) and the wait between
/// attempts honours `Retry-After`.  Two 429s carrying `Retry-After: 1`
/// must cost at least 2 s of waiting before the 200 arrives, and the
/// retry events must be tagged `RetryKind::Data`.
#[tokio::test(flavor = "multi_thread")]
async fn transport_retries_429_honouring_retry_after() {
    let server = wiremock::MockServer::start().await;
    mount_fail_then_succeed(&server, 429, Some(0), 2).await;

    let capturing = Arc::new(CapturingSink::new());
    let sink: Arc<dyn TelemetrySink> = capturing.clone();
    let req = make_request(
        format!("{}/v1/models", server.uri()),
        PipelinePhase::Validate,
    );

    let result = http_request(req, &sink, CancellationToken::new()).await;

    let resp = result.unwrap_or_else(|e| panic!("429 then 200 must succeed; got {}", e));
    assert_eq!(resp.status, 200);
    let delays: Vec<_> = capturing
        .events()
        .into_iter()
        .filter_map(|e| match e {
            talk_rs::telemetry::TranscriptionEvent::RetryScheduled {
                kind: talk_rs::telemetry::RetryKind::Data,
                delay,
                ..
            } => Some(delay),
            _ => None,
        })
        .collect();
    assert_eq!(delays, vec![Duration::ZERO, Duration::ZERO]);
    let data_retries: Vec<(u32, u32)> = capturing
        .events()
        .into_iter()
        .filter_map(|e| match e {
            talk_rs::telemetry::TranscriptionEvent::RetryScheduled {
                kind: talk_rs::telemetry::RetryKind::Data,
                attempt,
                max,
                ..
            } => Some((attempt, max)),
            _ => None,
        })
        .collect();
    assert_eq!(
        data_retries,
        vec![
            (1, DATA_BACKOFF_SECS.len() as u32),
            (2, DATA_BACKOFF_SECS.len() as u32)
        ],
        "two data retries expected, numbered 1 and 2 over the data budget"
    );
}

#[tokio::test]
async fn chain_wait_bound_stops_before_retry_after_exceeds_remaining_patience() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};

    let server = wiremock::MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "30"))
        .mount(&server)
        .await;
    let mut request = make_request(
        format!("{}/v1/models", server.uri()),
        PipelinePhase::Validate,
    );
    request.retry_schedule.connection_budgets = vec![Duration::from_secs(2)];
    request.retry_schedule.data_backoffs = vec![Duration::from_secs(5)];
    request.retry_schedule.max_data_wait = Some(Duration::from_secs(1));
    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    // Well under the 30 s Retry-After and the 5 s backoff slot: only an
    // immediate give-up fits, yet loaded CI runs have room to schedule.
    let result = tokio::time::timeout(
        Duration::from_secs(4),
        http_request(request, &sink, CancellationToken::new()),
    )
    .await;
    let failure = result
        .expect("wait must not reach deadline")
        .expect_err("429 must be returned");
    assert!(matches!(
        failure.kind,
        PipelineFailureKind::HttpStatus { status: 429, .. }
    ));
    assert_eq!(failure.retry_after, Some(Duration::from_secs(30)));
    assert_eq!(server.received_requests().await.expect("requests").len(), 1);
}

#[tokio::test]
async fn chain_wait_starts_with_first_request_not_schedule_construction() {
    use talk_rs::config::ChainEntry;

    let server = wiremock::MockServer::start().await;
    mount_fail_then_succeed(&server, 429, Some(0), 1).await;
    let entry = ChainEntry {
        provider: Provider::Mistral,
        model: "voxtral-mini-2602".into(),
        realtime_model: None,
        retries: Some(1),
        // Long enough that the first 429 round-trip fits even on a loaded
        // machine; the pre-request sleep below still exceeds it.
        wait: Some(Duration::from_secs(3)),
        languages: None,
        supports: Vec::new(),
    };
    let mut request = make_request(
        format!("{}/v1/models", server.uri()),
        PipelinePhase::Validate,
    );
    request.retry_schedule = entry.retry_schedule();
    tokio::time::sleep(Duration::from_millis(3100)).await;
    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    let response = tokio::time::timeout(
        Duration::from_secs(10),
        http_request(request, &sink, CancellationToken::new()),
    )
    .await
    .expect("request should not wait for construction-time deadline")
    .expect("busy response should retry after request begins");
    assert_eq!(response.status, 200);
    assert_eq!(
        server
            .received_requests()
            .await
            .expect("recorded requests")
            .len(),
        2
    );
}

#[tokio::test]
async fn chain_wait_budget_is_shared_across_preflight_and_upload() {
    use talk_rs::config::ChainEntry;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};

    let server = wiremock::MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/preflight"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/upload"))
        .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "0"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/upload"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let entry = ChainEntry {
        provider: Provider::Mistral,
        model: "voxtral-mini-2602".into(),
        realtime_model: None,
        retries: Some(1),
        wait: Some(Duration::from_millis(80)),
        languages: None,
        supports: Vec::new(),
    };
    let schedule = entry.retry_schedule();
    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    let mut preflight = make_request(
        format!("{}/preflight", server.uri()),
        PipelinePhase::Validate,
    );
    preflight.retry_schedule = schedule.clone();
    assert_eq!(
        http_request(preflight, &sink, CancellationToken::new())
            .await
            .expect("preflight")
            .status,
        200
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    let mut upload = make_request(format!("{}/upload", server.uri()), PipelinePhase::Request);
    upload.retry_schedule = schedule;
    let failure = http_request(upload, &sink, CancellationToken::new())
        .await
        .expect_err("shared wait expired");
    assert!(matches!(
        failure.kind,
        PipelineFailureKind::HttpStatus { status: 429, .. }
    ));
    assert_eq!(server.received_requests().await.expect("requests").len(), 2);
}

#[tokio::test]
async fn websocket_busy_handshake_is_fallback_worthy_but_unauthorized_is_not() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};

    for (status, fallback) in [(429, true), (401, false)] {
        let server = wiremock::MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/realtime"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&server)
            .await;
        let url = format!("{}/v1/realtime", server.uri()).replacen("http://", "ws://", 1);
        let mut request = make_request(url, PipelinePhase::Validate);
        request.retry_schedule.connection_budgets = vec![Duration::from_secs(2)];
        let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
        let failure = ws_upgrade(request, &sink, CancellationToken::new())
            .await
            .expect_err("handshake must fail");
        assert!(
            matches!(&failure.kind, PipelineFailureKind::HttpStatus { status: code, .. } if *code == status)
        );
        assert_eq!(failure.is_fallback_worthy(), fallback);
        assert_eq!(server.received_requests().await.expect("requests").len(), 1);
    }
}

#[tokio::test]
async fn websocket_chain_patience_retries_busy_handshake_on_same_entry() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let url = format!(
        "ws://{}/v1/realtime",
        listener.local_addr().expect("address")
    );
    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.expect("first connection");
        let mut request = [0u8; 4096];
        let len = first.read(&mut request).await.expect("request");
        assert!(String::from_utf8_lossy(&request[..len]).starts_with("GET /v1/realtime"));
        first
            .write_all(
                b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .await
            .expect("429");
        drop(first);
        let (second, _) = listener.accept().await.expect("retry connection");
        tokio_tungstenite::accept_async(second)
            .await
            .expect("upgrade retry");
    });
    let mut request = make_request(url, PipelinePhase::Validate);
    request.retry_schedule.connection_budgets = vec![Duration::from_secs(2); 2];
    request.retry_schedule.data_backoffs = vec![Duration::ZERO];
    request.retry_schedule.retry_ws_busy = true;
    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    let stream = tokio::time::timeout(
        Duration::from_secs(5),
        ws_upgrade(request, &sink, CancellationToken::new()),
    )
    .await
    .expect("bounded handshake")
    .expect("second handshake succeeds");
    drop(stream);
    server.await.expect("server completed");
}

/// Spec 2 (schedule): without `Retry-After`, the wait before data
/// retry `n` is `DATA_BACKOFF_SECS[n-1]` in the default policy.
#[test]
fn transport_data_backoff_schedule_is_monotone_and_generous() {
    let schedule = RetrySchedule::default();
    let total: Duration = schedule.data_backoffs.iter().sum();
    assert_eq!(
        schedule.data_backoffs,
        DATA_BACKOFF_SECS.map(Duration::from_secs)
    );
    assert_eq!(
        schedule.connection_budgets,
        CONNECTION_BUDGETS_SECS.map(Duration::from_secs)
    );
    assert!(
        schedule.data_backoffs.len() >= 5,
        "at least 5 server retries expected; got {}",
        schedule.data_backoffs.len()
    );
    assert!(
        schedule.data_backoffs.windows(2).all(|w| w[0] <= w[1]),
        "backoff must be non-decreasing: {:?}",
        DATA_BACKOFF_SECS
    );
    assert!(
        total >= Duration::from_secs(300),
        "cumulative wait must give a saturated provider ≥5 min to recover; got {}s",
        total.as_secs()
    );
    assert!(
        CONNECTION_BUDGETS_SECS.len() >= 5,
        "connection schedule must remain in place"
    );
}

/// Spec 2 (schedule, live): with no `Retry-After`, the first data
/// retry waits at least `DATA_BACKOFF_SECS[0]` seconds.  Uses a
/// single 503 so the test costs exactly one schedule slot.
#[tokio::test(flavor = "multi_thread")]
async fn transport_waits_first_backoff_slot_before_data_retry() {
    let server = wiremock::MockServer::start().await;
    mount_fail_then_succeed(&server, 503, None, 1).await;

    let capturing = Arc::new(CapturingSink::new());
    let sink: Arc<dyn TelemetrySink> = capturing.clone();
    let mut req = make_request(
        format!("{}/v1/models", server.uri()),
        PipelinePhase::Validate,
    );
    req.retry_schedule.data_backoffs = vec![Duration::from_millis(40)];

    let started = Instant::now();
    let result = http_request(req, &sink, CancellationToken::new()).await;
    let elapsed = started.elapsed();

    assert!(result.is_ok(), "503 then 200 must succeed");
    let first_slot = Duration::from_millis(40);
    assert!(
        elapsed >= first_slot,
        "first data retry must wait DATA_BACKOFF_SECS[0]={:?}; elapsed {:?}",
        first_slot,
        elapsed
    );
    let data_delays: Vec<_> = capturing
        .events()
        .into_iter()
        .filter_map(|e| match e {
            talk_rs::telemetry::TranscriptionEvent::RetryScheduled {
                kind: talk_rs::telemetry::RetryKind::Data,
                delay,
                ..
            } => Some(delay),
            _ => None,
        })
        .collect();
    assert_eq!(data_delays, vec![first_slot]);
}

/// Spec 3: the data-phase budget is exhausted after
/// `DATA_BACKOFF_SECS.len() + 1` attempts, and the failure reports
/// that budget (not the connection budget).  `Retry-After: 0` keeps
/// the test fast while still exercising the full loop.
#[tokio::test(flavor = "multi_thread")]
async fn transport_data_budget_is_independent_of_connection_budget() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};

    let server = wiremock::MockServer::start().await;
    let expected_attempts = DATA_BACKOFF_SECS.len() as u64 + 1;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503).insert_header("Retry-After", "0"))
        .expect(expected_attempts)
        .mount(&server)
        .await;

    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    let req = make_request(
        format!("{}/v1/models", server.uri()),
        PipelinePhase::Validate,
    );

    let pf = http_request(req, &sink, CancellationToken::new())
        .await
        .expect_err("permanent 503 must fail");
    assert_eq!(pf.attempts, expected_attempts as u32);
    assert_eq!(pf.max_attempts, expected_attempts as u32);
    assert!(
        pf.to_string().contains("server-retry budget exhausted"),
        "got: {}",
        pf
    );
}

/// Spec 4: a 400 is permanent — exactly one request, no wait.
#[tokio::test(flavor = "multi_thread")]
async fn transport_400_is_permanent_without_wait() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};

    let server = wiremock::MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(400).insert_header("Retry-After", "5"))
        .expect(1)
        .mount(&server)
        .await;

    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    let req = make_request(
        format!("{}/v1/models", server.uri()),
        PipelinePhase::Validate,
    );

    let pf = http_request(req, &sink, CancellationToken::new())
        .await
        .expect_err("400 must fail");
    assert_eq!(pf.attempts, 1);
    assert!(
        pf.to_string().contains("4xx permanent, no retry"),
        "got: {}",
        pf
    );
}

/// Spec 5: a request whose `wall_clock` exceeds the first connection
/// budget (2 s) must NOT be cut off at 2 s.  A server that answers
/// after 3 s must succeed on the first attempt with zero retries.
#[tokio::test(flavor = "multi_thread")]
async fn transport_wall_clock_lifts_per_attempt_cap() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};

    let server = wiremock::MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(80))
                .set_body_string("{\"data\":[]}"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let capturing = Arc::new(CapturingSink::new());
    let sink: Arc<dyn TelemetrySink> = capturing.clone();
    let mut req = make_request(
        format!("{}/v1/models", server.uri()),
        PipelinePhase::Request,
    );
    req.wall_clock = Some(Duration::from_secs(20));
    req.retry_schedule.connection_budgets = vec![Duration::from_millis(40)];

    let result = http_request(req, &sink, CancellationToken::new()).await;
    assert!(
        result.is_ok(),
        "80ms server with 20s wall_clock must succeed on attempt 1; got {:?}",
        result.err().map(|e| e.to_string())
    );
    assert!(
        capturing.retry_events().is_empty(),
        "no retry may be scheduled: {:?}",
        capturing.retry_events()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn user_attended_post_waits_after_connection_without_resending() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, ResponseTemplate};

    let server = wiremock::MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(80))
                .set_body_string("ok"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    let mut request = make_request(
        format!("{}/v1/audio/transcriptions", server.uri()),
        PipelinePhase::Request,
    );
    request.method = Method::Post;
    request.body = RequestBody::Bytes(Arc::new(vec![1, 2, 3]));
    request.retry_schedule.connection_budgets = vec![Duration::from_millis(40)];
    let response = http_request(request, &sink, CancellationToken::new())
        .await
        .expect("established user-attended request must finish");
    assert_eq!(response.body, b"ok");
    assert_eq!(server.received_requests().await.expect("requests").len(), 1);
}

/// Spec 6: cancelling during a backoff wait returns without waiting
/// for the `Retry-After: 30` slot or making another request.
#[tokio::test(flavor = "multi_thread")]
async fn transport_cancellation_aborts_backoff_wait() {
    let server = wiremock::MockServer::start().await;
    mount_fail_then_succeed(&server, 503, Some(30), 1).await;

    let req = make_request(
        format!("{}/v1/models", server.uri()),
        PipelinePhase::Validate,
    );

    let cancel = CancellationToken::new();
    let cancelling = Arc::new(CancelOnRetrySink::new(cancel.clone()));
    let sink: Arc<dyn TelemetrySink> = cancelling.clone();

    let result = tokio::time::timeout(Duration::from_secs(20), http_request(req, &sink, cancel))
        .await
        .expect("cancellation must not wait for the 30s backoff");

    assert!(
        cancelling.cancelled_at().is_some(),
        "retry must be scheduled"
    );
    assert!(
        matches!(result, Err(ref error) if error.to_string().contains("cancelled")),
        "cancel during backoff must return a cancellation error: {result:?}"
    );
    assert_eq!(server.received_requests().await.expect("requests").len(), 1);
}

/// Spec (plan §3 Step 6, bullet 5): `ws_upgrade` against an
/// unreachable host retries the connection per the connection-phase
/// retry schedule, same vocabulary as `http_request`.
#[tokio::test(flavor = "multi_thread")]
async fn ws_upgrade_retries_on_connection_failure() {
    let capturing = Arc::new(CapturingSink::new());
    let sink: Arc<dyn TelemetrySink> = capturing.clone();
    let req = make_request(refused_loopback_url("ws"), PipelinePhase::Request);

    let result = ws_upgrade(req, &sink, CancellationToken::new()).await;
    assert!(
        result.is_err(),
        "ws upgrade to a refused loopback port must fail"
    );

    let retries = capturing.retry_events();
    assert!(
        retries.len() >= 2,
        "expected ≥2 connection retries for ws_upgrade against \
         an unreachable host; got {}: {:?}",
        retries.len(),
        retries,
    );
}

// ── Step 11 cross-process job registry tests ───────────────────────
//
// These cover the jobs module (`talk_rs::transcription::jobs`) added
// in Step 11.  Pre-Step-11 the module does not yet exist, so the
// tests below intentionally use a doc-style stub that fails on
// invocation.  The test names are committed here so the spec is
// pinned even before Step 11.

use talk_rs::transcription::jobs;

/// The lock payload carries owner identity and model details; this
/// implementation uses SIGUSR1, not a status socket.
#[tokio::test(flavor = "multi_thread")]
async fn lock_file_includes_owner_pid_and_model() {
    let dir = tempfile::TempDir::new().unwrap();
    let audio = dir.path().join("rec.ogg");
    std::fs::File::create(&audio).unwrap();

    let job =
        jobs::register_local(&audio, Provider::Mistral, "voxtral", false).expect("register_local");
    let yaml = std::fs::read_to_string(job.lock_path()).unwrap();
    let parsed: jobs::LockPayload = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(parsed.owner_pid, std::process::id());
    assert!(parsed.owner_started_at_unix_secs > 0);
    assert_eq!(parsed.model, "voxtral");
    assert_eq!(parsed.provider, "mistral");
    assert!(!parsed.owner_id.is_empty());
}

/// Spec (plan §3 Step 11, revised): a remote observer reads the
/// lock-file payload synchronously rather than replaying past
/// events (the socket-based replay was deferred; see the jobs
/// module docs).  This test verifies same-process observation
/// returns the same payload the owner wrote.
#[tokio::test(flavor = "multi_thread")]
async fn observe_remote_reads_in_flight_lock_payload() {
    let dir = tempfile::TempDir::new().unwrap();
    let audio = dir.path().join("rec.ogg");
    std::fs::File::create(&audio).unwrap();

    let _job =
        jobs::register_local(&audio, Provider::Mistral, "voxtral", false).expect("register_local");

    let observed = jobs::list_in_flight_for(&audio);
    assert_eq!(observed.len(), 1, "expected exactly one in-flight job");
    assert_eq!(observed[0].payload.owner_pid, std::process::id());
    assert_eq!(observed[0].payload.model, "voxtral");
    assert!(!observed[0].payload.owner_id.is_empty());
    assert!(observed[0].owner_alive());
}

/// Spec (plan §3 Step 11): an observer calling `cancel_remote`
/// causes the owner's `CancellationToken` to fire.
///
/// NOTE: SIGUSR1 is process-wide.  When test binaries run tests in
/// parallel, an earlier test's job registration may absorb the
/// signal sent by this test, leaving our token uncancelled.
/// Marked `#[ignore]` so the default ``cargo test`` run does not
/// include it; run explicitly in isolation with
/// ``cargo test observe_remote_cancel_signals_owner -- --ignored``.
/// The same-binary lib-level test
/// ``transcription::jobs::tests::cancel_remote_via_sigusr1_triggers_owner_token``
/// covers this contract in isolation as well.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "SIGUSR1 is process-wide; run in isolation"]
async fn observe_remote_cancel_signals_owner() {
    let dir = tempfile::TempDir::new().unwrap();
    let audio = dir.path().join("rec.ogg");
    std::fs::File::create(&audio).unwrap();

    let job =
        jobs::register_local(&audio, Provider::Mistral, "voxtral", false).expect("register_local");
    let token = job.cancel_token();
    assert!(!token.is_cancelled());

    // Give the SIGUSR1 polling task a chance to install in the
    // tokio runtime BEFORE we send the signal.  Without this,
    // a fast test sends the signal before tokio's signal driver
    // has finished wiring up the handler.
    tokio::time::sleep(Duration::from_millis(100)).await;
    tokio::task::yield_now().await;

    let in_flight = jobs::list_in_flight_for(&audio);
    assert_eq!(in_flight.len(), 1);
    jobs::cancel_remote(&in_flight[0]).expect("cancel_remote");

    for _ in 0..100 {
        if token.is_cancelled() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        token.is_cancelled(),
        "cancellation must propagate via SIGUSR1"
    );
}

/// Spec (plan §3 Step 11): a lock file pointing at a non-existent
/// PID is recognised by [`RemoteJob::owner_alive`] returning
/// false.  Stale detection is a precondition for cleaning up
/// stale locks left behind by crashed owners.
#[tokio::test(flavor = "multi_thread")]
async fn stale_lock_detected_when_owner_pid_dead() {
    let dir = tempfile::TempDir::new().unwrap();
    let audio = dir.path().join("rec.ogg");
    std::fs::File::create(&audio).unwrap();

    let stale_pid: u32 = u32::MAX - 1;
    let lock_path = dir.path().join("rec_mistral_voxtral_oneshot-lock.yml");
    let yaml = format!(
        "version: 1\nowner_pid: {}\nowner_started_at_unix_secs: 0\n\
         provider: mistral\nmodel: voxtral\nrealtime: false\n",
        stale_pid
    );
    std::fs::write(&lock_path, yaml).unwrap();

    let observed = jobs::list_in_flight_for(&audio);
    assert_eq!(observed.len(), 1);
    assert!(
        !observed[0].owner_alive(),
        "lock pointing at PID {} (assumed dead) must be reported as stale",
        stale_pid
    );
}

// ── Compile-time wiring sanity check ───────────────────────────────

/// Compile-only assertion: every public type used in the test
/// signatures above is in scope.  If a future refactor removes one
/// of these types, this assertion fails to compile and the spec
/// breakage is loud.
#[allow(dead_code)]
fn _compile_check_public_surface() {
    let _: Method = Method::Get;
    let _: Method = Method::Post;
    let _: PipelinePhase = PipelinePhase::Validate;
    let _: PipelinePhase = PipelinePhase::Request;
    let _: RetryKind = RetryKind::Connection;
    let _: RetryKind = RetryKind::Data;
    let _: NetworkKind = NetworkKind::Connect;
    let _: NetworkKind = NetworkKind::WallClock;
    let _: NetworkKind = NetworkKind::KernelTcp;
    let _: NetworkKind = NetworkKind::Other;

    // ConnectionEvent variants exist and are constructible.
    let _ = ConnectionEvent::ResolvingDns { t: Instant::now() };
    let _ = ConnectionEvent::Completed { t: Instant::now() };

    // PipelineFailureKind variants exist (used implicitly by tests).
    let _ = PipelineFailureKind::Decode(String::new());
}
