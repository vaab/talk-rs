//! Item `http-client-pool-prewarm`: transport contracts a pooled /
//! prewarmed client must keep, driven through the public
//! `transport::http_request` API against a local HTTP/1.1 keep-alive
//! server that records which TCP connection served each request and
//! with which credentials.
//!
//! - Sequential same-policy requests: connections and client builds
//!   used (the pooling measurement).
//! - Distinct credentials and distinct origins: each request carries
//!   its own `Authorization` and reaches its own server.
//! - A server that drops the idle keep-alive connection: the next
//!   request still succeeds, with no connection-retry charged.
//! - A slow response under `wall_clock: None` (user attended) outlives
//!   the connect budget without a retry.
//! - Cancellation aborts promptly.
//! - Growing connection budget: a refused origin still exhausts the
//!   full schedule (attempts == max).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use talk_rs::config::Provider;
use talk_rs::error::PipelinePhase;
use talk_rs::telemetry::{RetryKind, TelemetrySink, TranscriptionEvent};
use talk_rs::transcription::transport::{http_request, Method, Request, RequestBody};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::require_counters;
use crate::support::Metrics;

/// One served request: (connection id, authorization header, path).
type Served = (usize, String, String);

/// Minimal HTTP/1.1 keep-alive server.
struct KeepAliveServer {
    url: String,
    served: Arc<Mutex<Vec<Served>>>,
    accepted: Arc<Mutex<usize>>,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Clone, Copy)]
struct ServerOptions {
    /// Close each connection after it served this many requests.
    close_after: Option<usize>,
    /// Delay before each response.
    delay: Duration,
}

impl KeepAliveServer {
    async fn start(options: ServerOptions) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let url = format!("http://{}", listener.local_addr().expect("addr"));
        let served = Arc::new(Mutex::new(Vec::new()));
        let accepted = Arc::new(Mutex::new(0usize));
        let (log, count) = (Arc::clone(&served), Arc::clone(&accepted));
        let task = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let conn = {
                    let mut n = count.lock().expect("count");
                    *n += 1;
                    *n
                };
                let log = Arc::clone(&log);
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stream);
                    let mut handled = 0;
                    loop {
                        let mut request_line = String::new();
                        if reader.read_line(&mut request_line).await.unwrap_or(0) == 0 {
                            return;
                        }
                        let path = request_line
                            .split_whitespace()
                            .nth(1)
                            .unwrap_or("")
                            .to_string();
                        let (mut auth, mut length) = (String::new(), 0usize);
                        loop {
                            let mut header = String::new();
                            if reader.read_line(&mut header).await.unwrap_or(0) == 0 {
                                return;
                            }
                            let header = header.trim_end();
                            if header.is_empty() {
                                break;
                            }
                            let lower = header.to_ascii_lowercase();
                            if let Some(v) = lower.strip_prefix("content-length:") {
                                length = v.trim().parse().unwrap_or(0);
                            }
                            if lower.starts_with("authorization:") {
                                auth = header["authorization:".len()..].trim().to_string();
                            }
                        }
                        let mut body = vec![0u8; length];
                        if reader.read_exact(&mut body).await.is_err() {
                            return;
                        }
                        log.lock().expect("log").push((conn, auth, path));
                        tokio::time::sleep(options.delay).await;
                        handled += 1;
                        let close = options.close_after.is_some_and(|n| handled >= n);
                        let payload = br#"{"data":[]}"#;
                        let response = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n{}\r\n",
                            payload.len(),
                            if close { "connection: close\r\n" } else { "" }
                        );
                        let stream = reader.get_mut();
                        if stream.write_all(response.as_bytes()).await.is_err()
                            || stream.write_all(payload).await.is_err()
                        {
                            return;
                        }
                        if close {
                            let _ = stream.shutdown().await;
                            return;
                        }
                    }
                });
            }
        });
        Self {
            url,
            served,
            accepted,
            task,
        }
    }

    fn served(&self) -> Vec<Served> {
        self.served.lock().expect("served").clone()
    }

    fn accepted(&self) -> usize {
        *self.accepted.lock().expect("accepted")
    }
}

impl Drop for KeepAliveServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Default)]
struct RetrySink(Mutex<Vec<RetryKind>>);

impl TelemetrySink for RetrySink {
    fn emit(&self, event: TranscriptionEvent) {
        if let TranscriptionEvent::RetryScheduled { kind, .. } = event {
            self.0.lock().expect("retries").push(kind);
        }
    }
}

fn request(url: &str, key: &str, wall_clock: Option<Duration>) -> Request {
    Request {
        method: Method::Get,
        url: format!("{url}/v1/models"),
        headers: vec![("Authorization".into(), format!("Bearer {key}"))],
        body: RequestBody::Empty,
        provider: Provider::Mistral,
        provider_name: "Mistral".into(),
        phase: PipelinePhase::Validate,
        wall_clock,
        retry_schedule: Default::default(),
    }
}

fn counter(name: &str) -> u64 {
    talk_rs::perf_counters::snapshot()
        .into_iter()
        .find(|(n, _)| n == name)
        .map_or(0, |(_, v)| v)
}

/// Serialises the tests of this module: they read process-wide counters.
static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// 20 sequential same-policy requests: TCP connections and client
/// builds used.  Every request must succeed with its own credentials.
#[tokio::test(flavor = "multi_thread")]
async fn perf_transport_sequential_same_policy() {
    require_counters!();
    let _l = LOCK.lock().await;
    let server = KeepAliveServer::start(ServerOptions {
        close_after: None,
        delay: Duration::ZERO,
    })
    .await;
    let sink = Arc::new(RetrySink::default());
    let dyn_sink: Arc<dyn TelemetrySink> = sink.clone();
    let builds = counter("http_client_builds");
    for i in 0..20 {
        let r = http_request(
            request(&server.url, &format!("k{i}"), None),
            &dyn_sink,
            CancellationToken::new(),
        )
        .await
        .expect("request");
        assert_eq!(r.status, 200);
    }
    let served = server.served();
    assert_eq!(served.len(), 20);
    for (i, (_, auth, path)) in served.iter().enumerate() {
        assert_eq!(auth, &format!("Bearer k{i}"), "credentials of request {i}");
        assert_eq!(path, "/v1/models");
    }
    assert!(
        sink.0.lock().expect("retries").is_empty(),
        "no retry on a healthy server"
    );
    Metrics::new("http-client-pool-prewarm", "transport-sequential-20")
        .set("requests", 20.0)
        .set("tcp_connections", server.accepted() as f64)
        .set(
            "http_client_builds",
            (counter("http_client_builds") - builds) as f64,
        )
        .write();
}

/// Two origins with two keys, interleaved: each request reaches its
/// own origin with its own key (a pool must never cross them).
#[tokio::test(flavor = "multi_thread")]
async fn perf_transport_distinct_origins_and_credentials() {
    let _l = LOCK.lock().await;
    let opts = ServerOptions {
        close_after: None,
        delay: Duration::ZERO,
    };
    let (a, b) = (
        KeepAliveServer::start(opts).await,
        KeepAliveServer::start(opts).await,
    );
    let sink: Arc<dyn TelemetrySink> = Arc::new(RetrySink::default());
    for i in 0..6 {
        let (server, key) = if i % 2 == 0 {
            (&a, "alpha")
        } else {
            (&b, "beta")
        };
        http_request(
            request(&server.url, key, None),
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect("request");
    }
    assert!(a.served().iter().all(|(_, auth, _)| auth == "Bearer alpha"));
    assert!(b.served().iter().all(|(_, auth, _)| auth == "Bearer beta"));
    assert_eq!((a.served().len(), b.served().len()), (3, 3));
}

/// The server closes every connection after one response (a dropped
/// keep-alive): each next request must still succeed, with no
/// connection retry charged to the user-visible budget.
#[tokio::test(flavor = "multi_thread")]
async fn perf_transport_stale_connection_is_not_a_retry() {
    let _l = LOCK.lock().await;
    let server = KeepAliveServer::start(ServerOptions {
        close_after: Some(1),
        delay: Duration::ZERO,
    })
    .await;
    let sink = Arc::new(RetrySink::default());
    let dyn_sink: Arc<dyn TelemetrySink> = sink.clone();
    for _ in 0..5 {
        http_request(
            request(&server.url, "k", None),
            &dyn_sink,
            CancellationToken::new(),
        )
        .await
        .expect("request after a dropped keep-alive");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(server.served().len(), 5);
    let retries = sink.0.lock().expect("retries").clone();
    assert!(
        !retries.iter().any(|k| matches!(k, RetryKind::Connection)),
        "stale keep-alive charged as a connection retry: {retries:?}"
    );
}

/// A response slower than the first connect budget (2 s) under a
/// user-attended policy completes without any retry.
#[tokio::test(flavor = "multi_thread")]
async fn perf_transport_slow_attended_response_is_not_retried() {
    let _l = LOCK.lock().await;
    let server = KeepAliveServer::start(ServerOptions {
        close_after: None,
        delay: Duration::from_millis(2_500),
    })
    .await;
    let sink = Arc::new(RetrySink::default());
    let dyn_sink: Arc<dyn TelemetrySink> = sink.clone();
    let r = http_request(
        request(&server.url, "k", None),
        &dyn_sink,
        CancellationToken::new(),
    )
    .await
    .expect("slow response");
    assert_eq!(r.status, 200);
    assert_eq!(server.served().len(), 1);
    assert!(sink.0.lock().expect("retries").is_empty());
}

/// Cancellation of an in-flight request returns within 500 ms.
#[tokio::test(flavor = "multi_thread")]
async fn perf_transport_cancellation_is_prompt() {
    let _l = LOCK.lock().await;
    let server = KeepAliveServer::start(ServerOptions {
        close_after: None,
        delay: Duration::from_secs(30),
    })
    .await;
    let sink: Arc<dyn TelemetrySink> = Arc::new(RetrySink::default());
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        trigger.cancel();
    });
    let t0 = Instant::now();
    let r = http_request(request(&server.url, "k", None), &sink, cancel).await;
    assert!(r.is_err(), "cancelled request must fail");
    assert!(
        t0.elapsed() < Duration::from_millis(800),
        "took {:?}",
        t0.elapsed()
    );
}

/// Growing connect budget: a refused origin exhausts the complete
/// connection schedule (a shared client must keep per-attempt budgets).
#[tokio::test(flavor = "multi_thread")]
async fn perf_transport_refused_origin_exhausts_schedule() {
    let _l = LOCK.lock().await;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("port");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    drop(listener);
    let sink = Arc::new(RetrySink::default());
    let dyn_sink: Arc<dyn TelemetrySink> = sink.clone();
    let r = http_request(
        request(&url, "k", None),
        &dyn_sink,
        CancellationToken::new(),
    )
    .await;
    let failure = r.expect_err("refused origin");
    assert_eq!(failure.attempts, failure.max_attempts);
    let connection_retries = sink
        .0
        .lock()
        .expect("retries")
        .iter()
        .filter(|k| matches!(k, RetryKind::Connection))
        .count();
    assert_eq!(connection_retries as u32, failure.max_attempts - 1);
}
