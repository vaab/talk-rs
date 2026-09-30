//! Mock transcription provider (wiremock) behind a counting TCP proxy.
//!
//! The proxy is what makes connection reuse observable: wiremock's
//! request log cannot tell whether two requests shared a socket, the
//! proxy's `accepted()` count can.  It can also delay each accept (to
//! emulate TCP+TLS round trips) and close idle client connections (to
//! emulate a server dropping keep-alives).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

pub const TRANSCRIPT: &str = "the quick brown fox jumps over the lazy dog";

/// Replies with the next status in `statuses` (the last one repeats);
/// 200 carries a Mistral/OpenAI-compatible transcript body.
struct Sequence {
    statuses: Vec<u16>,
    next: AtomicU64,
    delay: Duration,
    received: Arc<Mutex<Vec<Instant>>>,
}

impl Respond for Sequence {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        if let Ok(mut guard) = self.received.lock() {
            guard.push(Instant::now());
        }
        let i = self.next.fetch_add(1, Ordering::SeqCst) as usize;
        let status = self.statuses[i.min(self.statuses.len() - 1)];
        let template = if status == 200 {
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": TRANSCRIPT,
                "language": "en",
                "model": "voxtral-mini-2602",
                "segments": [],
                "usage": {"prompt_audio_seconds": 3.0}
            }))
        } else {
            ResponseTemplate::new(status).set_body_string("unavailable")
        };
        template.set_delay(self.delay)
    }
}

/// A mock provider reachable through a counting proxy.
pub struct MockProvider {
    pub server: MockServer,
    pub proxy: CountingProxy,
    /// When each POST was fully received by the mock (in order).
    post_times: Arc<Mutex<Vec<Instant>>>,
}

impl MockProvider {
    /// `/v1/models` lists every catalog model; POST
    /// `/v1/audio/transcriptions` answers with `statuses` in order.
    pub async fn start(statuses: &[u16], delay: Duration, proxy: ProxyOptions) -> Self {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"id": "voxtral-mini-2602"}, {"id": "voxtral-mini-2507"},
                    {"id": "gpt-transcribe"}, {"id": "gpt-4o-transcribe"},
                    {"id": "gpt-4o-mini-transcribe"}
                ]
            })))
            .mount(&server)
            .await;
        let post_times = Arc::new(Mutex::new(Vec::new()));
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(Sequence {
                statuses: statuses.to_vec(),
                next: AtomicU64::new(0),
                delay,
                received: Arc::clone(&post_times),
            })
            .mount(&server)
            .await;
        let upstream = *server.address();
        let proxy = CountingProxy::start(upstream, proxy).await;
        Self {
            server,
            proxy,
            post_times,
        }
    }

    pub fn post_times(&self) -> Vec<Instant> {
        self.post_times
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Base URL (through the proxy) for `providers.<p>.url`.
    pub fn url(&self) -> String {
        format!("http://{}", self.proxy.addr)
    }

    pub async fn posts(&self) -> Vec<Request> {
        self.server
            .received_requests()
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|r| r.method.as_str() == "POST")
            .collect()
    }
}

/// Extract the multipart `file` part's bytes from an upload request.
pub fn multipart_file(request: &Request) -> Vec<u8> {
    let content_type = request
        .headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    let boundary = content_type
        .split("boundary=")
        .nth(1)
        .unwrap_or_default()
        .trim_matches('"');
    let delimiter = format!("--{boundary}");
    let body = &request.body;
    let mut parts = Vec::new();
    let mut start = 0;
    while let Some(pos) = find(&body[start..], delimiter.as_bytes()) {
        parts.push(start + pos);
        start += pos + delimiter.len();
    }
    for window in parts.windows(2) {
        let part = &body[window[0] + delimiter.len()..window[1]];
        if let Some(split) = find(part, b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&part[..split]);
            if headers.contains("name=\"file\"") {
                let data = &part[split + 4..];
                return data.strip_suffix(b"\r\n").unwrap_or(data).to_vec();
            }
        }
    }
    panic!("no multipart file part in upload");
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[derive(Clone, Copy, Default)]
pub struct ProxyOptions {
    /// Sleep before forwarding each accepted connection.
    pub accept_delay: Duration,
    /// Close a client connection after this long without traffic.
    pub idle_close: Option<Duration>,
}

/// TCP proxy that records every accepted connection.
pub struct CountingProxy {
    pub addr: std::net::SocketAddr,
    accepts: Arc<Mutex<Vec<Instant>>>,
    task: tokio::task::JoinHandle<()>,
}

impl CountingProxy {
    pub async fn start(upstream: std::net::SocketAddr, options: ProxyOptions) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("proxy bind");
        let addr = listener.local_addr().expect("proxy addr");
        let accepts = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&accepts);
        let task = tokio::spawn(async move {
            while let Ok((client, _)) = listener.accept().await {
                if let Ok(mut guard) = log.lock() {
                    guard.push(Instant::now());
                }
                tokio::spawn(forward(client, upstream, options));
            }
        });
        Self {
            addr,
            accepts,
            task,
        }
    }

    /// Number of TCP connections accepted so far.
    pub fn accepted(&self) -> usize {
        self.accepts.lock().map(|g| g.len()).unwrap_or_default()
    }

    pub fn accept_times(&self) -> Vec<Instant> {
        self.accepts.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

impl Drop for CountingProxy {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn forward(mut client: TcpStream, upstream: std::net::SocketAddr, options: ProxyOptions) {
    tokio::time::sleep(options.accept_delay).await;
    let Ok(mut server) = TcpStream::connect(upstream).await else {
        return;
    };
    let (mut cr, mut cw) = client.split();
    let (mut sr, mut sw) = server.split();
    let mut cbuf = vec![0u8; 64 * 1024];
    let mut sbuf = vec![0u8; 64 * 1024];
    let idle = options.idle_close.unwrap_or(Duration::from_secs(3600));
    loop {
        tokio::select! {
            n = cr.read(&mut cbuf) => match n {
                Ok(0) | Err(_) => break,
                Ok(n) => if sw.write_all(&cbuf[..n]).await.is_err() { break },
            },
            n = sr.read(&mut sbuf) => match n {
                Ok(0) | Err(_) => break,
                Ok(n) => if cw.write_all(&sbuf[..n]).await.is_err() { break },
            },
            _ = tokio::time::sleep(idle) => break,
        }
    }
}
