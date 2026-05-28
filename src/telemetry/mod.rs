//! Display-agnostic transcription telemetry layer.
//!
//! This module captures raw events from the transcription pipeline
//! (HTTP requests, byte progress, retries, paste operations) and
//! delivers them to consumers through a trait-based sink.  Consumers
//! are completely decoupled from producers — the HTTP code never
//! imports from any display module, and display adapters never
//! import from the HTTP code.  Swapping the display (X11 overlay,
//! CLI status line, JSON log, no display at all) requires zero
//! changes to the producer side.
//!
//! # Architecture
//!
//! ```text
//! [Producers]                  [Sink]                  [Consumers]
//! mistral.rs    ──emit──> BroadcastSink ──fanout──>  overlay thread
//! openai.rs     ──emit──>        │                    (future) CLI
//! dictate/mod   ──emit──>        │                    (future) log
//! ```
//!
//! The sink trait takes `&self` (not `&mut self`) so producers can
//! share a single sink behind `Arc` without locking at the API level.
//! Implementations that need mutation use interior mutability
//! ([`BroadcastSink`] uses `tokio::sync::broadcast` which handles
//! this internally).
//!
//! # Adding a new event type
//!
//! 1. Add a variant to [`TranscriptionEvent`]
//! 2. Producers emit it via `sink.emit(...)`
//! 3. Consumers match on it in their receiver loop
//!
//! Events carry [`Instant`] timestamps (monotonic) so time-axis
//! renderings can compare them reliably across NTP jumps.
//!
//! # Dependency rules
//!
//! `src/telemetry/` has **zero** imports from `crate::x11::*`,
//! `crate::audio::*`, `crate::dictate::*`, or any other display or
//! pipeline module.  If you find yourself wanting to import from
//! those, stop — put the logic in the consumer or producer instead.

use std::time::Instant;
use tokio::sync::broadcast;

/// Distinguishes the two independently-budgeted retry concerns
/// the transport tracks: connection-phase (DNS / TCP / TLS) and
/// data-phase (server-side transient errors after the connection
/// was established).  Carried on
/// [`TranscriptionEvent::RetryScheduled`] so consumers can label
/// `connect retry N/M` separately from `server retry N/M` instead
/// of an ambiguous `retry N/M`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryKind {
    /// Retry of the TCP / TLS / DNS connection phase.  Grows the
    /// per-attempt connect budget on each retry.
    Connection,
    /// Retry after the connection was established and the server
    /// returned a transient failure (5xx, decode error mid-body).
    Data,
}

/// A single raw event from the transcription pipeline.
///
/// Each variant represents one fact about what happened at a given
/// instant — no derived state, no display concerns.  Downstream
/// consumers can derive higher-level state (current phase,
/// throughput windows, retry counts, etc.) from a stream of these.
#[derive(Debug, Clone)]
pub enum TranscriptionEvent {
    /// Model preflight (`/v1/models` validation) started — emitted
    /// by [`crate::transcription::transport::http::validate_model`]
    /// only on cache miss.  When the validate-cache hits, the
    /// preflight is skipped entirely and no `Preflight*` events are
    /// emitted, so consumers must be prepared for [`Self::RequestStarted`]
    /// to arrive without a preceding [`Self::PreflightStarted`].
    PreflightStarted { t: Instant },

    /// Model preflight finished (success or failure).  Emitted only
    /// on cache miss, paired with [`Self::PreflightStarted`].  On
    /// `success: true`, the next event is normally
    /// [`Self::RequestStarted`] from the actual transcription
    /// request.  On `success: false`, the call site returns the
    /// error and no transcription request is made.
    PreflightCompleted { success: bool, t: Instant },

    /// Wire goes hot: reqwest's `.send()` has been called.  No
    /// bytes have been sent yet; the connection may or may not be
    /// established.
    RequestStarted {
        /// Fully-qualified target URL.
        endpoint: String,
        t: Instant,
    },

    /// Connection is ready — reqwest is pulling the first byte from
    /// the request body.
    ///
    /// This is a proxy for "TCP+TLS handshake complete" because
    /// reqwest 0.13 does not expose a direct connection-established
    /// signal, but the body stream's first poll only happens after
    /// the handshake succeeds.
    ConnectionEstablished { t: Instant },

    /// Another chunk of request body was pulled from the stream.
    /// `bytes_sent` is the cumulative total so far; `total` is the
    /// overall request body size (from known `Content-Length` or
    /// file metadata).
    UploadProgress {
        bytes_sent: u64,
        total: u64,
        t: Instant,
    },

    /// Last byte of the request body was pulled; we are now waiting
    /// for the server response.
    UploadComplete { total: u64, t: Instant },

    /// Response headers have been received.
    ResponseHeaders { status: u16, t: Instant },

    /// Another chunk of response body was received.  `bytes_received`
    /// is the cumulative total so far.  `total` is optional because
    /// the response `Content-Length` header is often absent for JSON
    /// responses.
    DownloadProgress {
        bytes_received: u64,
        total: Option<u64>,
        t: Instant,
    },

    /// Response body fully received.
    ResponseComplete { total: u64, t: Instant },

    /// The HTTP request completed (success or failure).  This is
    /// the terminal event for a single attempt — a retry would
    /// start a fresh `RequestStarted`.
    RequestCompleted { success: bool, t: Instant },

    /// A retry attempt is being scheduled after a previous failure.
    /// `attempt` is 1-indexed (`1` for the first retry); `max` is
    /// the configured maximum.  `kind` distinguishes the two
    /// independently-budgeted retry concerns the transport tracks:
    ///
    /// - [`RetryKind::Connection`] — TCP / TLS / DNS retry with
    ///   growing budget `[2, 5, 8, 11, 15]` seconds.  Picker UI
    ///   should render as `connect retry N/M…`.
    /// - [`RetryKind::Data`] — server returned a transient 5xx
    ///   after the connection was established.  Picker UI should
    ///   render as `server retry N/M…`.
    RetryScheduled {
        kind: RetryKind,
        attempt: u32,
        max: u32,
        reason: String,
        t: Instant,
    },

    /// Free-form status update for phases that do not map onto a
    /// structured event (typically post-upgrade WS session
    /// lifecycle: ``session handshake…``, ``session ready,
    /// awaiting audio…``, ``streaming audio…``).
    ///
    /// Producers should emit this BEFORE any visible action that
    /// would otherwise leave the UI silent.  Consumers (picker,
    /// overlay) treat the string as the row's current status
    /// label.  The message is replaced by the next structured
    /// event (a fresh ``Status`` or, terminally, a transcript
    /// text update).
    Status { message: String, t: Instant },

    /// Paste phase started (keystroke / clipboard injection to the
    /// target application).
    PasteStarted { t: Instant },

    /// Another chunk of text was pasted into the target window.
    /// `chars_pasted` is cumulative; `total_chars` is the full
    /// transcription length.  In streaming mode, paste chunks can
    /// overlap with upload / download activity.
    PasteProgress {
        chars_pasted: u64,
        total_chars: u64,
        t: Instant,
    },

    /// Paste phase completed.
    PasteCompleted { t: Instant },

    /// Terminal success — the entire dictation cycle finished.
    Done { t: Instant },

    /// Terminal failure — the entire dictation cycle gave up.
    Failed { reason: String, t: Instant },
}

/// Where events go.
///
/// Implementations are shared via `Arc<dyn TelemetrySink>` and must
/// be `Send + Sync`.  `emit` takes `&self` so producers can share a
/// single sink without locking at the API level; implementations
/// with state use interior mutability.
pub trait TelemetrySink: Send + Sync {
    /// Deliver an event to whoever is listening.
    ///
    /// Must not block.  If delivery cannot happen immediately (no
    /// consumers, lagging consumer, etc.) the implementation should
    /// silently drop the event rather than block the producer.
    fn emit(&self, event: TranscriptionEvent);
}

/// Default sink: drops every event on the floor.
///
/// Used when no consumer is attached — for example in unit tests
/// that don't care about telemetry, or in non-interactive code
/// paths that have no display.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpSink;

impl TelemetrySink for NoOpSink {
    fn emit(&self, _event: TranscriptionEvent) {}
}

/// Broadcast sink: fans events out to any number of subscribers.
///
/// Uses `tokio::sync::broadcast` under the hood, so consumers can
/// subscribe at any time and receive events going forward.  If all
/// subscribers are lagging or disconnected, [`TelemetrySink::emit`]
/// silently drops the event — the producer never blocks.
///
/// The channel capacity determines how many unread events a slow
/// consumer can tolerate before it starts missing events (older
/// events are dropped first).
pub struct BroadcastSink {
    tx: broadcast::Sender<TranscriptionEvent>,
}

impl BroadcastSink {
    /// Create a new broadcast sink with the given channel capacity.
    ///
    /// A capacity of 256 is a sensible default — enough to buffer a
    /// few seconds of events at typical rates (a few per second)
    /// even when consumers are briefly slow.
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Subscribe to this sink's event stream.
    ///
    /// Returns a fresh receiver that will see every event emitted
    /// **after** this call.  Multiple receivers may subscribe
    /// independently; each sees the full event stream going forward.
    /// Late subscribers do not see past events.
    pub fn subscribe(&self) -> broadcast::Receiver<TranscriptionEvent> {
        self.tx.subscribe()
    }

    /// Current number of active subscribers.
    ///
    /// Useful for producers that want to skip expensive event
    /// construction when nobody is listening.
    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl TelemetrySink for BroadcastSink {
    fn emit(&self, event: TranscriptionEvent) {
        // If there are no receivers, `send` returns an error which
        // we deliberately drop — producers don't care whether
        // anyone is listening.
        let _ = self.tx.send(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// In-test sink that records every event it receives, so tests
    /// can assert on the emitted sequence.
    struct MockSink {
        events: Mutex<Vec<TranscriptionEvent>>,
    }

    impl MockSink {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }

        fn recorded(&self) -> Vec<TranscriptionEvent> {
            self.events
                .lock()
                .expect("test: mock sink lock poisoned")
                .clone()
        }
    }

    impl TelemetrySink for MockSink {
        fn emit(&self, event: TranscriptionEvent) {
            self.events
                .lock()
                .expect("test: mock sink lock poisoned")
                .push(event);
        }
    }

    #[test]
    fn noop_sink_drops_events_without_panicking() {
        let sink = NoOpSink;
        sink.emit(TranscriptionEvent::Done { t: Instant::now() });
        sink.emit(TranscriptionEvent::Failed {
            reason: "test".to_string(),
            t: Instant::now(),
        });
        // Nothing to assert beyond "doesn't crash".
    }

    #[test]
    fn mock_sink_records_events_in_order() {
        let sink = MockSink::new();
        sink.emit(TranscriptionEvent::RequestStarted {
            endpoint: "https://example.com".to_string(),
            t: Instant::now(),
        });
        sink.emit(TranscriptionEvent::ConnectionEstablished { t: Instant::now() });

        let events = sink.recorded();
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0],
            TranscriptionEvent::RequestStarted { .. }
        ));
        assert!(matches!(
            events[1],
            TranscriptionEvent::ConnectionEstablished { .. }
        ));
    }

    #[test]
    fn broadcast_sink_delivers_to_every_active_subscriber() {
        let sink = BroadcastSink::new(16);
        let mut rx1 = sink.subscribe();
        let mut rx2 = sink.subscribe();

        assert_eq!(sink.receiver_count(), 2);

        sink.emit(TranscriptionEvent::Done { t: Instant::now() });

        let e1 = rx1
            .try_recv()
            .expect("test: rx1 should have received the event");
        let e2 = rx2
            .try_recv()
            .expect("test: rx2 should have received the event");
        assert!(matches!(e1, TranscriptionEvent::Done { .. }));
        assert!(matches!(e2, TranscriptionEvent::Done { .. }));
    }

    #[test]
    fn broadcast_sink_drops_when_no_subscribers_are_attached() {
        let sink = BroadcastSink::new(16);
        assert_eq!(sink.receiver_count(), 0);
        // No subscribers — emit should silently drop.
        sink.emit(TranscriptionEvent::Done { t: Instant::now() });

        // A late subscriber MUST NOT see the past event.
        let mut rx = sink.subscribe();
        assert!(
            rx.try_recv().is_err(),
            "late subscribers must not see events emitted before they subscribed"
        );
    }

    #[test]
    fn broadcast_sink_is_usable_behind_arc_dyn() {
        // Validate that `BroadcastSink` can be shared as
        // `Arc<dyn TelemetrySink>`, which is how producers will
        // normally hold onto it.
        let sink: Arc<dyn TelemetrySink> = Arc::new(BroadcastSink::new(16));

        let sink_a = Arc::clone(&sink);
        let sink_b = Arc::clone(&sink);

        sink_a.emit(TranscriptionEvent::Done { t: Instant::now() });
        sink_b.emit(TranscriptionEvent::Done { t: Instant::now() });

        // Neither call should block or panic.  We cannot observe
        // receipt because there are no subscribers, but the Arc
        // sharing pattern compiles and runs.
    }

    #[test]
    fn all_transcription_event_variants_are_clone() {
        // Smoke test that every variant can be cloned — required
        // for broadcast fan-out (every receiver gets its own clone)
        // and for any consumer that wants to keep event history.
        let events = vec![
            TranscriptionEvent::RequestStarted {
                endpoint: "x".to_string(),
                t: Instant::now(),
            },
            TranscriptionEvent::ConnectionEstablished { t: Instant::now() },
            TranscriptionEvent::UploadProgress {
                bytes_sent: 100,
                total: 1000,
                t: Instant::now(),
            },
            TranscriptionEvent::UploadComplete {
                total: 1000,
                t: Instant::now(),
            },
            TranscriptionEvent::ResponseHeaders {
                status: 200,
                t: Instant::now(),
            },
            TranscriptionEvent::DownloadProgress {
                bytes_received: 50,
                total: Some(500),
                t: Instant::now(),
            },
            TranscriptionEvent::ResponseComplete {
                total: 500,
                t: Instant::now(),
            },
            TranscriptionEvent::RequestCompleted {
                success: true,
                t: Instant::now(),
            },
            TranscriptionEvent::RetryScheduled {
                kind: RetryKind::Connection,
                attempt: 1,
                max: 5,
                reason: "timeout".to_string(),
                t: Instant::now(),
            },
            TranscriptionEvent::PasteStarted { t: Instant::now() },
            TranscriptionEvent::PasteProgress {
                chars_pasted: 50,
                total_chars: 100,
                t: Instant::now(),
            },
            TranscriptionEvent::PasteCompleted { t: Instant::now() },
            TranscriptionEvent::Done { t: Instant::now() },
            TranscriptionEvent::Failed {
                reason: "x".to_string(),
                t: Instant::now(),
            },
        ];

        for e in &events {
            let _cloned = e.clone();
        }
    }

    #[test]
    fn broadcast_sink_keeps_working_after_a_receiver_is_dropped() {
        let sink = BroadcastSink::new(16);
        let mut rx_keep = sink.subscribe();
        {
            let _rx_drop = sink.subscribe();
            // _rx_drop falls out of scope here
        }
        assert_eq!(sink.receiver_count(), 1);

        sink.emit(TranscriptionEvent::Done { t: Instant::now() });
        let evt = rx_keep
            .try_recv()
            .expect("test: remaining receiver should still deliver");
        assert!(matches!(evt, TranscriptionEvent::Done { .. }));
    }
}
