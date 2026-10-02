use super::*;
use crate::error::TalkError;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn warmed_socket_carries_the_following_upload() {
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let endpoint = format!(
        "http://{}/v1/audio/transcriptions",
        listener.local_addr().expect("addr")
    );
    let (observed_tx, mut observed_rx) = tokio::sync::mpsc::channel(2);
    let server = tokio::spawn(async move {
        let mut handlers = tokio::task::JoinSet::new();
        let mut accepted = 0usize;
        loop {
            let (stream, _) = listener.accept().await.expect("accept");
            accepted += 1;
            let observed_tx = observed_tx.clone();
            handlers.spawn(async move {
                let mut reader = BufReader::new(stream);
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).await.expect("request line") == 0 {
                        break;
                    }
                    let method = line.split_whitespace().next().expect("method").to_string();
                    let mut length = 0usize;
                    loop {
                        let mut header = String::new();
                        reader.read_line(&mut header).await.expect("header");
                        if header == "\r\n" {
                            break;
                        }
                        if let Some(value) =
                            header.to_ascii_lowercase().strip_prefix("content-length:")
                        {
                            length = value.trim().parse().expect("length");
                        }
                    }
                    let mut body = vec![0; length];
                    reader.read_exact(&mut body).await.expect("body");
                    observed_tx
                        .send((accepted, method.clone(), body))
                        .await
                        .expect("observe");
                    let response = if method == "HEAD" {
                        "HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n"
                    } else {
                        "HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nOK"
                    };
                    reader
                        .get_mut()
                        .write_all(response.as_bytes())
                        .await
                        .expect("respond");
                }
            });
        }
    });
    prewarm_http(
        &endpoint,
        &RetrySchedule::default(),
        &CancellationToken::new(),
    )
    .await;
    let sink: Arc<dyn crate::telemetry::TelemetrySink> = Arc::new(crate::telemetry::NoOpSink);
    let response = http_request(
        Request {
            method: Method::Post,
            url: endpoint,
            headers: vec![("Authorization".into(), "Bearer test".into())],
            body: RequestBody::Bytes(Arc::new(vec![1, 2, 3])),
            provider: crate::config::Provider::Mistral,
            provider_name: "Mistral".into(),
            phase: crate::error::PipelinePhase::Request,
            wall_clock: Some(Duration::from_secs(3)),
            retry_schedule: RetrySchedule::default(),
        },
        &sink,
        CancellationToken::new(),
    )
    .await
    .expect("upload");
    assert_eq!(response.body, b"OK");
    let first = tokio::time::timeout(Duration::from_secs(2), observed_rx.recv())
        .await
        .expect("HEAD")
        .expect("observed");
    let second = tokio::time::timeout(Duration::from_secs(2), observed_rx.recv())
        .await
        .expect("POST")
        .expect("observed");
    assert_eq!((first.0, first.1.as_str(), first.2), (1, "HEAD", vec![]));
    assert_eq!(
        (second.0, second.1.as_str(), second.2),
        (1, "POST", vec![1, 2, 3])
    );
    server.abort();
    server.await.expect_err("server stopped");
}

#[test]
fn client_pool_keeps_hot_budget_and_distinguishes_exact_durations() {
    let mut clients = HttpClients::default();
    let hot = Duration::from_secs(2);
    let before =
        crate::perf_counters::thread_value(crate::perf_counters::Counter::HttpClientBuilds);
    clients.client(hot).expect("hot client");
    for micros in 1100..1140 {
        clients
            .client(Duration::from_micros(micros))
            .expect("custom client");
    }
    clients.client(hot).expect("hot client after churn");
    assert_eq!(clients.by_budget.len(), MAX_HTTP_CLIENT_POLICIES);
    let builds =
        crate::perf_counters::thread_value(crate::perf_counters::Counter::HttpClientBuilds)
            - before;
    assert_eq!(
        builds, 41,
        "distinct sub-millisecond budgets and retained hot client"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stalled_cold_builder_does_not_block_hot_lookup_or_cancellation() {
    use std::sync::{mpsc, Mutex, OnceLock};
    type Gate = (mpsc::Sender<()>, mpsc::Receiver<()>);
    static GATE: OnceLock<Mutex<Option<Gate>>> = OnceLock::new();
    fn stalled_builder(budget: Duration) -> Result<reqwest::Client, String> {
        let (entered, release) = GATE
            .get()
            .expect("gate")
            .lock()
            .expect("lock")
            .take()
            .expect("one builder");
        entered.send(()).expect("entered");
        release.recv().expect("release");
        build_client_with_connect_timeout(budget)
    }

    let hot = Duration::from_micros(314_159);
    let cold = Duration::from_micros(314_160);
    let cached = http_client(hot, &CancellationToken::new())
        .await
        .expect("hot client");
    let (entered, ready) = mpsc::channel();
    let (release, waiting) = mpsc::channel();
    *GATE.get_or_init(|| Mutex::new(None)).lock().expect("gate") = Some((entered, waiting));
    let cancel = CancellationToken::new();
    let cold_token = cancel.clone();
    let cold_task =
        tokio::spawn(
            async move { http_client_with_builder(cold, &cold_token, stalled_builder).await },
        );
    tokio::task::block_in_place(|| ready.recv().expect("build entered"));
    let hot_task = tokio::spawn(async move { http_client(hot, &CancellationToken::new()).await });
    cancel.cancel();
    let cancelled = tokio::time::timeout(Duration::from_secs(1), cold_task).await;
    let hot_result = tokio::time::timeout(Duration::from_millis(300), hot_task).await;
    release.send(()).expect("release stalled builder");
    assert!(cancelled
        .expect("cancellation promptly returned")
        .expect("join")
        .is_err());
    let hot_client = hot_result
        .expect("hot lookup blocked behind unrelated build")
        .expect("hot join")
        .expect("hot client");
    assert_eq!(format!("{hot_client:?}"), format!("{cached:?}"));
}

#[tokio::test]
async fn collection_warms_without_credentials_and_preserves_audio() {
    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;
    let endpoint = format!("{}/v1/audio/transcriptions", server.uri());
    let (tx, rx) = tokio::sync::mpsc::channel(2);
    let task = tokio::spawn(async move {
        http::collect_upload(
            rx,
            &endpoint,
            &RetrySchedule::default(),
            &CancellationToken::new(),
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(2), async {
        while server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("warmup while pipe open");
    tx.send(vec![1, 2, 3]).await.expect("audio");
    tx.send(vec![4, 5]).await.expect("tail");
    drop(tx);
    assert_eq!(
        task.await.expect("task").expect("bytes"),
        vec![1, 2, 3, 4, 5]
    );
    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 1);
    assert!(requests[0].body.is_empty());
    assert!(!requests[0].headers.contains_key("authorization"));
}

#[tokio::test]
async fn collection_eof_and_cancel_do_not_wait_for_stalled_head() {
    let server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(30)))
        .mount(&server)
        .await;
    let endpoint = format!("{}/v1/audio/transcriptions", server.uri());
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let cancel = CancellationToken::new();
    let task = tokio::spawn(async move {
        http::collect_upload(rx, &endpoint, &RetrySchedule::default(), &cancel).await
    });
    tokio::time::timeout(Duration::from_secs(2), async {
        while server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("HEAD started");
    tx.send(vec![7, 8]).await.expect("audio");
    drop(tx);
    assert_eq!(
        tokio::time::timeout(Duration::from_millis(300), task)
            .await
            .expect("no HEAD wait")
            .expect("task")
            .expect("bytes"),
        vec![7, 8]
    );

    let endpoint = format!("{}/v1/audio/transcriptions", server.uri());
    let (_tx, rx) = tokio::sync::mpsc::channel(1);
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    let task = tokio::spawn(async move {
        http::collect_upload(rx, &endpoint, &RetrySchedule::default(), &cancel).await
    });
    tokio::task::yield_now().await;
    trigger.cancel();
    let result = tokio::time::timeout(Duration::from_millis(300), task)
        .await
        .expect("prompt cancel")
        .expect("task");
    assert!(
        matches!(result, Err(TalkError::Transcription(message)) if message == "cancelled by caller")
    );
}
