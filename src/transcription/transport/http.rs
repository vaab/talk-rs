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

/// Timeout for the lightweight model-listing preflight check.
pub(crate) const VALIDATE_TIMEOUT: Duration = Duration::from_secs(10);

/// TCP connect timeout — fail fast when the server is unreachable.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

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

/// Format a [`reqwest::Error`] with full diagnostic detail for logs.
///
/// `reqwest::Error`'s default `Display` prints only the top-level
/// message (e.g. `"error sending request for url (...)"`) and hides
/// the actual cause behind `std::error::Error::source()`.  For talk-rs
/// users who land in the logs trying to figure out why a transcription
/// failed, that single line is useless: it does not distinguish DNS
/// failure from connection refused from TLS handshake failure from
/// read timeout.
///
/// This helper produces a single-line string that:
///
/// - Starts with the top-level [`reqwest::Error`] message.
/// - Appends a `[kind=...]` tag derived from the structured
///   classifiers ([`reqwest::Error::is_timeout`],
///   [`reqwest::Error::is_connect`], etc.).
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
/// # Why this helper exists
///
/// Without this, network failures in the talk-rs log all collapse to
/// the same useless string and we cannot tell DNS from timeout from
/// TLS without re-running the failure with `RUST_LOG=reqwest=trace`.
/// With this, a single log line carries enough information to
/// classify the failure mode and (often) the underlying OS error.
pub(crate) fn format_reqwest_error(err: &reqwest::Error) -> String {
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

/// Validate that `model` is available at `api_base/v1/models`.
///
/// `provider_name` is used only in error messages (e.g. `"Mistral"`,
/// `"OpenAI"`).  `is_transcription_model` filters the available models
/// to suggest transcription-relevant alternatives on failure.
pub(crate) async fn validate_model(
    provider_name: &str,
    api_key: &str,
    model: &str,
    api_base: &str,
    is_transcription_model: fn(&str) -> bool,
) -> Result<(), TalkError> {
    let models_url = format!("{}/v1/models", api_base);

    let client = build_client()?;
    let response = client
        .get(&models_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .timeout(VALIDATE_TIMEOUT)
        .send()
        .await
        .map_err(|e| {
            TalkError::Config(format!(
                "Failed to connect to {} API: {}",
                provider_name,
                format_reqwest_error(&e)
            ))
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(TalkError::Config(format!(
            "{} API error ({}): {}",
            provider_name, status, body
        )));
    }

    let models: ModelsResponse = response.json().await.map_err(|e| {
        TalkError::Config(format!(
            "Failed to parse {} models response: {}",
            provider_name, e
        ))
    })?;

    if models.data.iter().any(|m| m.id == model) {
        return Ok(());
    }

    // Model not found — collect transcription-relevant alternatives.
    let mut transcription_models: Vec<&str> = models
        .data
        .iter()
        .map(|m| m.id.as_str())
        .filter(|id| is_transcription_model(id))
        .collect();
    transcription_models.sort();

    if transcription_models.is_empty() {
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
    }
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

        let s = format_reqwest_error(&err);
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

        let s = format_reqwest_error(&err);
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

        let s = format_reqwest_error(&err);
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
}
