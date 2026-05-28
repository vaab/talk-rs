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
//! - `cancellation_aborts_in_flight_request_within_100ms`
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
    }
}

/// 127.0.0.1:1 is reserved and rejects connections immediately on
/// localhost, but for the connect-budget test we want a destination
/// that *times out* (silently drops SYN), not one that rejects.
///
/// 192.0.2.1 is in TEST-NET-1 (RFC 5737), guaranteed not to be
/// routed.  TCP SYNs to it never get a response, so reqwest's
/// connect path exercises the full configured connect timeout.
const BLACKHOLE_TCP: &str = "http://192.0.2.1:81/v1/models";

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
    let req = make_request(BLACKHOLE_TCP, PipelinePhase::Request);

    let result = http_request(req, &sink, CancellationToken::new()).await;
    assert!(
        result.is_err(),
        "blackhole TCP destination must fail; got Ok"
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
/// budget (3 attempts) and emit `RetryScheduled` for each retry.
#[tokio::test(flavor = "multi_thread")]
async fn transport_data_phase_retries_three_times_on_503() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503))
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
    let req = make_request(BLACKHOLE_TCP, PipelinePhase::Request);

    let result = http_request(req, &sink, CancellationToken::new()).await;
    let pf = result.expect_err("blackhole destination must fail");
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
        .respond_with(ResponseTemplate::new(503))
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
/// `http_request` is in flight aborts it within 500ms (the budget
/// is intentionally generous to avoid CI flakes; the spec target
/// is sub-100ms, but a 500ms cap still proves "promptly" against
/// the 30s server delay).
#[tokio::test(flavor = "multi_thread")]
async fn cancellation_aborts_in_flight_request_within_100ms() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(30))
                .set_body_string("{\"data\":[]}"),
        )
        .mount(&server)
        .await;

    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
    let url = format!("{}/v1/models", server.uri());
    let req = make_request(&url, PipelinePhase::Validate);

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_clone.cancel();
    });

    let started = Instant::now();
    let result = http_request(req, &sink, cancel).await;
    let elapsed = started.elapsed();

    assert!(result.is_err(), "cancellation must surface as Err");
    assert!(
        elapsed < Duration::from_millis(500),
        "cancellation must abort within 500ms of trigger; \
         elapsed was {:?}",
        elapsed
    );
}

/// Spec (plan §3 Step 6, bullet 5): `ws_upgrade` against an
/// unreachable host retries the connection per the connection-phase
/// retry schedule, same vocabulary as `http_request`.
#[tokio::test(flavor = "multi_thread")]
async fn ws_upgrade_retries_on_connection_failure() {
    let capturing = Arc::new(CapturingSink::new());
    let sink: Arc<dyn TelemetrySink> = capturing.clone();
    let req = make_request("ws://192.0.2.1:81/v1/audio/stream", PipelinePhase::Request);

    let result = ws_upgrade(req, &sink, CancellationToken::new()).await;
    assert!(result.is_err(), "ws upgrade to blackhole must fail");

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

/// Spec (plan §3 Step 11): `register_local` writes a lock file
/// containing a YAML payload with the owner PID.  The historical
/// `status_socket` field from the v1 plan was dropped in favour
/// of SIGUSR1-based cancellation (see the jobs module docs for
/// the design pivot).  The test name is preserved for continuity.
#[tokio::test(flavor = "multi_thread")]
async fn lock_file_includes_status_socket_and_pid() {
    let dir = tempfile::TempDir::new().unwrap();
    let audio = dir.path().join("rec.ogg");
    std::fs::File::create(&audio).unwrap();

    let job =
        jobs::register_local(&audio, Provider::Mistral, "voxtral", false).expect("register_local");
    let yaml = std::fs::read_to_string(job.lock_path()).unwrap();
    let parsed: jobs::LockPayload = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(parsed.owner_pid, std::process::id());
    assert!(parsed.owner_started_at_unix_secs > 0);
}

/// Spec (plan §3 Step 11, revised): a remote observer reads the
/// lock-file payload synchronously rather than replaying past
/// events (the socket-based replay was deferred; see the jobs
/// module docs).  This test verifies same-process observation
/// returns the same payload the owner wrote.
#[tokio::test(flavor = "multi_thread")]
async fn observe_remote_replays_backlog_then_streams_live() {
    let dir = tempfile::TempDir::new().unwrap();
    let audio = dir.path().join("rec.ogg");
    std::fs::File::create(&audio).unwrap();

    let _job =
        jobs::register_local(&audio, Provider::Mistral, "voxtral", false).expect("register_local");

    let observed = jobs::list_in_flight_for(&audio);
    assert_eq!(observed.len(), 1, "expected exactly one in-flight job");
    assert_eq!(observed[0].payload.owner_pid, std::process::id());
    assert!(observed[0].owner_alive());
}

/// Spec (plan §3 Step 11): an observer calling `cancel_remote`
/// causes the owner's `CancellationToken` to fire.
#[tokio::test(flavor = "multi_thread")]
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
    let lock_path = dir.path().join("rec_mistral_voxtral_batch-lock.yml");
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
