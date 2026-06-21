//! Async transcription backend for the picker.
//!
//! Contains the transcription helpers that run concurrently with the
//! GTK picker window: one-shot task spawning, realtime WebSocket
//! streaming, and WAV PCM reading.

use super::ui::{PickerCandidate, PickerMessage};
use crate::config::{Config, Provider};
use crate::telemetry::{TelemetrySink, TranscriptionEvent as PipelineEvent};
use crate::transcription::{
    RealtimeTranscriber, TranscriptSegment, TranscriptionEvent, TranscriptionMetadata,
};
use std::path::PathBuf;
use std::sync::Mutex;

/// Telemetry sink that translates HTTP-pipeline events into picker
/// row status strings.
///
/// One instance per in-flight candidate.  The sink is plugged in via
/// [`crate::transcription::OneShotTranscriber::set_sink`] before the
/// transcription request starts and forwards
/// [`PickerMessage::CandidateStatus`] messages to the GTK channel as
/// the request transitions between phases.
///
/// # Status mapping
///
/// | Telemetry event                        | Status string             |
/// |----------------------------------------|---------------------------|
/// | `PreflightStarted`                     | `pre-validating model…`   |
/// | `PreflightCompleted`                   | (silent — next event      |
/// |                                        |  replaces the status)     |
/// | `RequestStarted`                       | `connecting…`             |
/// | `ConnectionEstablished`                | `uploading…`              |
/// | `UploadComplete`                       | `waiting for server…`     |
/// | `ResponseHeaders { status: 2xx }`      | `transcribing…`           |
/// | `RetryScheduled { attempt, max }`      | `retry attempt/max…`      |
/// | `RequestCompleted { success: false }`  | (silent — final candidate |
/// |                                        |  message takes over)      |
///
/// Other events (`UploadProgress`, `DownloadProgress`,
/// `ResponseComplete`, `PasteStarted`, …) are dropped per the
/// "phase transitions + retry attempts only" UX choice.
///
/// `PreflightStarted` only fires on a validate-cache miss (see
/// [`super::super::transcription::transport::validate_cache`]); a
/// cache hit skips the preflight entirely and the row's status
/// jumps straight to `connecting…` from `RequestStarted`.
///
/// # Idempotency
///
/// The sink only emits when the status string actually changes —
/// repeated identical phase transitions don't flood the GTK channel.
/// Achieved via the `last` mutex storing the most-recently-emitted
/// status text.
pub(super) struct PickerStatusSink {
    tx: std::sync::mpsc::Sender<PickerMessage>,
    provider: Provider,
    model: String,
    /// Whether this sink belongs to the realtime row or
    /// the one-shot row.  Carried on every emitted
    /// [`PickerMessage::CandidateStatus`] so the picker UI's
    /// row-matcher can disambiguate when both rows exist for the
    /// same (provider, model) pair.
    streaming: bool,
    last: Mutex<Option<String>>,
}

impl PickerStatusSink {
    /// Build a new sink for the given candidate row.  `tx` is the
    /// shared GTK channel; `provider`/`model`/`streaming` identify
    /// which row the status updates belong to.
    pub(super) fn new(
        tx: std::sync::mpsc::Sender<PickerMessage>,
        provider: Provider,
        model: String,
        streaming: bool,
    ) -> Self {
        Self {
            tx,
            provider,
            model,
            streaming,
            last: Mutex::new(None),
        }
    }

    /// Send a status update to the GTK channel, deduplicating
    /// against the last emitted string.  Errors on a closed channel
    /// are silently dropped — the picker may have already closed.
    fn push(&self, status_text: String) {
        if let Ok(mut last) = self.last.lock() {
            if last.as_deref() == Some(status_text.as_str()) {
                return;
            }
            *last = Some(status_text.clone());
        }
        let _ = self.tx.send(PickerMessage::CandidateStatus {
            provider: self.provider,
            model: self.model.clone(),
            streaming: self.streaming,
            status_text,
        });
    }
}

impl TelemetrySink for PickerStatusSink {
    fn emit(&self, event: PipelineEvent) {
        match event {
            PipelineEvent::PreflightStarted { .. } => {
                self.push("pre-validating model…".to_string())
            }
            // PreflightCompleted is intentionally silent — the
            // next event (RequestStarted on success, or a final
            // error candidate on failure) replaces the status.
            PipelineEvent::PreflightCompleted { .. } => {}
            PipelineEvent::RequestStarted { .. } => self.push("connecting…".to_string()),
            PipelineEvent::ConnectionEstablished { .. } => self.push("uploading…".to_string()),
            PipelineEvent::UploadComplete { .. } => self.push("waiting for server…".to_string()),
            PipelineEvent::ResponseHeaders { status, .. } if (200..300).contains(&status) => {
                self.push("transcribing…".to_string())
            }
            PipelineEvent::RetryScheduled {
                kind, attempt, max, ..
            } => {
                let label = match kind {
                    crate::telemetry::RetryKind::Connection => "connect retry",
                    crate::telemetry::RetryKind::Data => "server retry",
                };
                self.push(format!("{} {}/{}…", label, attempt, max))
            }
            // Free-form status: pass the producer's message
            // through verbatim.  Used by the realtime path for
            // post-upgrade phases that have no structured event
            // (session handshake, awaiting audio, etc.) — see
            // `realtime.rs` / `openai_realtime.rs` for emission
            // sites.
            PipelineEvent::Status { message, .. } => self.push(message),
            // All other events (progress, paste, terminal events,
            // non-2xx response headers) are dropped — phase
            // transitions + retry attempts only.
            _ => {}
        }
    }
}

/// PCM chunk size for realtime WAV feeding (480 samples = 30 ms at
/// 16 kHz).
const REALTIME_FEED_CHUNK: usize = 480;

/// Run a realtime transcription session: connect the transcriber,
/// feed PCM samples from `samples`, and forward transcription events
/// to the GTK channel via `tx`.
///
/// On success the final text is sent as a [`PickerMessage::Candidate`];
/// on error an error candidate is sent instead.  Intermediate text
/// updates are forwarded as [`PickerMessage::StreamUpdate`].
///
/// Before connecting, attaches a [`PickerStatusSink`] to the
/// transcriber so the WS-upgrade phase (and its growing-budget
/// retries) reach the row's status column.  Without this hop, the
/// row would sit in an empty spinner state for up to 41s before
/// the transport surfaces a connection error — see plan section
/// Step 8 for the diagnosis.
pub(super) async fn run_realtime_transcription(
    mut transcriber: Box<dyn RealtimeTranscriber>,
    samples: std::sync::Arc<Vec<i16>>,
    tx: std::sync::mpsc::Sender<PickerMessage>,
    provider: Provider,
    model: String,
    cancel_token: tokio_util::sync::CancellationToken,
) {
    // Attach the row's status sink BEFORE the WS upgrade so
    // connecting / retry events reach the UI.  `streaming=true`
    // tags every emitted `CandidateStatus` so the picker matches
    // the realtime row, not the one-shot row of the same model.
    let status_sink: std::sync::Arc<dyn TelemetrySink> = std::sync::Arc::new(
        PickerStatusSink::new(tx.clone(), provider, model.clone(), true),
    );
    transcriber.set_sink(status_sink);
    transcriber.set_cancel_token(cancel_token);

    // Connect to the realtime WebSocket.
    let (audio_tx, audio_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
    let event_rx = match transcriber.transcribe_realtime(audio_rx).await {
        Ok(rx) => rx,
        Err(e) => {
            log::warn!("realtime {}:{} connect failed: {}", provider, model, e);
            let _ = tx.send(PickerMessage::Candidate(Box::new(PickerCandidate::error(
                provider,
                model,
                format!("{e}"),
                true,
            ))));
            return;
        }
    };

    // Spawn feeder: send PCM chunks paced at 10 ms intervals.
    let feeder_samples = samples.clone();
    tokio::spawn(async move {
        let data = &*feeder_samples;
        let mut offset = 0;
        while offset < data.len() {
            let end = (offset + REALTIME_FEED_CHUNK).min(data.len());
            let chunk = data[offset..end].to_vec();
            if audio_tx.send(chunk).await.is_err() {
                break;
            }
            offset = end;
            // Pace the feed to avoid overwhelming the WebSocket.
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        // Drop audio_tx to signal end-of-audio.
    });

    // Listen for events and forward to GTK.
    let mut event_rx = event_rx;
    let mut accumulated = String::new();
    let mut timed_segments: Vec<TranscriptSegment> = Vec::new();
    loop {
        match event_rx.recv().await {
            Some(TranscriptionEvent::TextDelta { text }) => {
                accumulated.push_str(&text);
                let _ = tx.send(PickerMessage::StreamUpdate {
                    provider,
                    model: model.clone(),
                    accumulated_text: accumulated.clone(),
                });
            }
            Some(TranscriptionEvent::SegmentDelta { text, start, end }) => {
                if !text.is_empty() {
                    let trimmed = text.trim().to_string();
                    if let (Some(start), Some(end)) = (start, end) {
                        timed_segments.push(TranscriptSegment {
                            start,
                            end,
                            text: trimmed.clone(),
                        });
                    }
                    if !accumulated.is_empty() {
                        accumulated.push(' ');
                    }
                    accumulated.push_str(&trimmed);
                    let _ = tx.send(PickerMessage::StreamUpdate {
                        provider,
                        model: model.clone(),
                        accumulated_text: accumulated.clone(),
                    });
                }
            }
            Some(TranscriptionEvent::Done) => {
                let final_text = accumulated.trim().to_string();
                let _ = tx.send(PickerMessage::Candidate(Box::new(
                    PickerCandidate::success(
                        provider,
                        model,
                        final_text,
                        true,
                        if timed_segments.is_empty() {
                            None
                        } else {
                            Some(timed_segments)
                        },
                        TranscriptionMetadata::default(),
                    ),
                )));
                return;
            }
            Some(TranscriptionEvent::Error { message }) => {
                let _ = tx.send(PickerMessage::Candidate(Box::new(PickerCandidate::error(
                    provider, model, message, true,
                ))));
                return;
            }
            None => {
                // Channel closed without Done — use what we have.
                let final_text = accumulated.trim().to_string();
                let _ = tx.send(PickerMessage::Candidate(Box::new(
                    PickerCandidate::success(
                        provider,
                        model,
                        final_text,
                        true,
                        if timed_segments.is_empty() {
                            None
                        } else {
                            Some(timed_segments)
                        },
                        TranscriptionMetadata::default(),
                    ),
                )));
                return;
            }
            _ => {
                // Ignore SessionCreated, Language, etc.
            }
        }
    }
}

/// Spawn a single one-shot transcription task for one picker row.
///
/// The picker is interactive — the user is watching the row and can
/// dismiss the picker at any time — so the request runs under
/// [`RequestTimeoutPolicy::UserAttended`]: only the
/// `connect_timeout` (TCP+TLS phase) and the kernel-level TCP
/// defences (`tcp_user_timeout`, TCP keepalive) bound the wait.
/// There is intentionally no per-request wall-clock cap — a slow
/// server is allowed to take its time.
///
/// While the request is in progress, a [`PickerStatusSink`] forwards
/// HTTP-pipeline phase transitions (and retry events) to the GTK
/// channel as [`PickerMessage::CandidateStatus`] messages so the
/// row can show `connecting…` / `uploading…` / `waiting for
/// server…` / `transcribing…` / `retry N/M…` in italic dimmed text.
/// The final [`PickerMessage::Candidate`] (success or error)
/// replaces the status line with the actual transcript or error.
pub(super) fn spawn_transcription(
    tasks: &mut tokio::task::JoinSet<()>,
    tx: std::sync::mpsc::Sender<PickerMessage>,
    audio: PathBuf,
    provider: Provider,
    model: String,
    config: std::sync::Arc<Config>,
    _ignored_cancel: tokio_util::sync::CancellationToken,
) {
    // `streaming=false` tags this sink for the one-shot row.
    let sink: std::sync::Arc<dyn TelemetrySink> = std::sync::Arc::new(PickerStatusSink::new(
        tx.clone(),
        provider,
        model.clone(),
        false,
    ));
    tasks.spawn(async move {
        // Register this in-flight transcription with the jobs
        // module.  This (a) writes a YAML lock file describing
        // us so another talk-rs process can cancel us via
        // `cancel_remote`, and (b) installs a SIGUSR1 handler
        // that fires `local_job.cancel_token()` when a Stop
        // button click in the picker GTK sends SIGUSR1 to our
        // own PID.
        let local_job =
            match crate::transcription::jobs::register_local(&audio, provider, &model, false) {
                Ok(j) => Some(j),
                Err(e) => {
                    log::warn!(
                        "picker: failed to register job for {}:{}: {}",
                        provider,
                        model,
                        e
                    );
                    None
                }
            };
        let cancel_token = local_job
            .as_ref()
            .map(|j| j.cancel_token())
            .unwrap_or_default();
        let msg = match crate::transcription::transcribe_audio(
            &audio,
            config.as_ref(),
            provider,
            Some(&model),
            false,
            crate::transcription::TranscribeOptions {
                allow_api: true,
                policy: crate::transcription::RequestTimeoutPolicy::UserAttended,
                cancel_token: Some(cancel_token),
                // We registered via `register_local` above, which
                // wrote the lock file with the YAML payload.
                // `transcribe_audio` must not race us on the same
                // file path.
                skip_legacy_lock: local_job.is_some(),
            },
            &sink,
        )
        .await
        {
            Ok(res) => {
                let text = res.text.trim().to_string();
                PickerMessage::Candidate(Box::new(PickerCandidate::success(
                    provider,
                    model,
                    text,
                    false,
                    res.segments,
                    res.metadata,
                )))
            }
            Err(e) => {
                log::warn!("candidate {}:{} failed: {}", provider, model, e);
                PickerMessage::Candidate(Box::new(PickerCandidate::error(
                    provider,
                    model,
                    format!("{e}"),
                    false,
                )))
            }
        };
        // Drop the LocalJob explicitly so the lock file is
        // removed before we ack the result.  Without this,
        // an immediate retry would race against the still-held
        // lock.
        drop(local_job);
        let _ = tx.send(msg);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Drain every queued message and return the
    /// `(provider, model, status_text)` tuples in arrival order.
    /// Non-`CandidateStatus` messages are returned as `None`.
    fn drain_status(
        rx: &std::sync::mpsc::Receiver<PickerMessage>,
    ) -> Vec<Option<(Provider, String, String)>> {
        let mut out = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            out.push(match msg {
                PickerMessage::CandidateStatus {
                    provider,
                    model,
                    status_text,
                    ..
                } => Some((provider, model, status_text)),
                _ => None,
            });
        }
        out
    }

    /// Spec: the sink translates a representative event sequence
    /// (including the validate-preflight phase that fires only on
    /// cache miss) into the documented status strings, in order,
    /// with no duplicates from idempotency dedup.
    #[test]
    fn picker_status_sink_translates_phase_transitions() {
        let (tx, rx) = std::sync::mpsc::channel::<PickerMessage>();
        let sink = PickerStatusSink::new(
            tx,
            Provider::Mistral,
            "voxtral-mini-2602".to_string(),
            false,
        );

        let now = Instant::now();
        // Cache-miss path: preflight events come first.
        sink.emit(PipelineEvent::PreflightStarted { t: now });
        sink.emit(PipelineEvent::PreflightCompleted {
            success: true,
            t: now,
        });
        sink.emit(PipelineEvent::RequestStarted {
            endpoint: "https://example.invalid".into(),
            t: now,
        });
        sink.emit(PipelineEvent::ConnectionEstablished { t: now });
        // Progress events are intentionally dropped (phase
        // transitions only).
        sink.emit(PipelineEvent::UploadProgress {
            bytes_sent: 1024,
            total: 8192,
            t: now,
        });
        sink.emit(PipelineEvent::UploadProgress {
            bytes_sent: 4096,
            total: 8192,
            t: now,
        });
        sink.emit(PipelineEvent::UploadComplete {
            total: 8192,
            t: now,
        });
        sink.emit(PipelineEvent::ResponseHeaders {
            status: 200,
            t: now,
        });
        // Download progress also dropped.
        sink.emit(PipelineEvent::DownloadProgress {
            bytes_received: 64,
            total: None,
            t: now,
        });
        sink.emit(PipelineEvent::ResponseComplete { total: 64, t: now });
        // RequestCompleted with success=true is silent — final
        // candidate carries the result.
        sink.emit(PipelineEvent::RequestCompleted {
            success: true,
            t: now,
        });

        let drained = drain_status(&rx);
        let strings: Vec<&str> = drained
            .iter()
            .filter_map(|o| o.as_ref().map(|(_, _, s)| s.as_str()))
            .collect();
        assert_eq!(
            strings,
            vec![
                "pre-validating model…",
                "connecting…",
                "uploading…",
                "waiting for server…",
                "transcribing…",
            ],
            "phase-transition strings must arrive in the documented order"
        );
    }

    /// Spec: on cache hit, no preflight events fire — the row's
    /// status starts directly at `connecting…`.  This test
    /// simulates the cache-hit path by simply omitting the
    /// `Preflight*` events; the sink must still produce the
    /// downstream phases correctly.
    #[test]
    fn picker_status_sink_cache_hit_starts_at_connecting() {
        let (tx, rx) = std::sync::mpsc::channel::<PickerMessage>();
        let sink = PickerStatusSink::new(tx, Provider::Mistral, "voxtral".to_string(), false);

        let now = Instant::now();
        // No PreflightStarted / PreflightCompleted — cache hit.
        sink.emit(PipelineEvent::RequestStarted {
            endpoint: "https://x".into(),
            t: now,
        });
        sink.emit(PipelineEvent::ConnectionEstablished { t: now });
        sink.emit(PipelineEvent::UploadComplete {
            total: 1024,
            t: now,
        });
        sink.emit(PipelineEvent::ResponseHeaders {
            status: 200,
            t: now,
        });

        let drained = drain_status(&rx);
        let strings: Vec<&str> = drained
            .iter()
            .filter_map(|o| o.as_ref().map(|(_, _, s)| s.as_str()))
            .collect();
        assert_eq!(
            strings,
            vec![
                "connecting…",
                "uploading…",
                "waiting for server…",
                "transcribing…",
            ],
            "cache-hit path must skip the pre-validating status"
        );
    }

    /// Spec: `PreflightCompleted` is intentionally silent — it
    /// doesn't push its own status; the next event (either
    /// `RequestStarted` on success or a final error candidate on
    /// failure, neither produced here) is what the user sees.
    #[test]
    fn picker_status_sink_preflight_completed_is_silent() {
        let (tx, rx) = std::sync::mpsc::channel::<PickerMessage>();
        let sink = PickerStatusSink::new(tx, Provider::Mistral, "voxtral".to_string(), false);

        let now = Instant::now();
        sink.emit(PipelineEvent::PreflightStarted { t: now });
        // Drain the pre-validating message so the next assertion
        // sees only what `PreflightCompleted` itself produces.
        let _ = drain_status(&rx);

        sink.emit(PipelineEvent::PreflightCompleted {
            success: true,
            t: now,
        });
        sink.emit(PipelineEvent::PreflightCompleted {
            success: false,
            t: now,
        });

        let drained = drain_status(&rx);
        assert!(
            drained.is_empty(),
            "PreflightCompleted must not push any status; got {:?}",
            drained
        );
    }

    /// Spec: identical consecutive emissions are deduplicated so
    /// the GTK channel does not receive redundant updates.
    #[test]
    fn picker_status_sink_dedups_repeated_phase() {
        let (tx, rx) = std::sync::mpsc::channel::<PickerMessage>();
        let sink = PickerStatusSink::new(tx, Provider::Mistral, "voxtral".to_string(), false);

        let now = Instant::now();
        sink.emit(PipelineEvent::RequestStarted {
            endpoint: "https://x".into(),
            t: now,
        });
        sink.emit(PipelineEvent::RequestStarted {
            endpoint: "https://x".into(),
            t: now,
        });
        sink.emit(PipelineEvent::RequestStarted {
            endpoint: "https://x".into(),
            t: now,
        });

        let drained = drain_status(&rx);
        assert_eq!(drained.len(), 1, "dedup must collapse repeated phases");
    }

    /// Spec: retry events surface attempt/max in the status string,
    /// in `connect retry N/M…` / `server retry N/M…` form.
    #[test]
    fn picker_status_sink_formats_retry_attempts() {
        let (tx, rx) = std::sync::mpsc::channel::<PickerMessage>();
        let sink = PickerStatusSink::new(tx, Provider::OpenAI, "whisper-1".to_string(), false);

        let now = Instant::now();
        sink.emit(PipelineEvent::RetryScheduled {
            kind: crate::telemetry::RetryKind::Connection,
            attempt: 2,
            max: 5,
            reason: "transient".into(),
            t: now,
        });
        sink.emit(PipelineEvent::RetryScheduled {
            kind: crate::telemetry::RetryKind::Data,
            attempt: 1,
            max: 3,
            reason: "5xx".into(),
            t: now,
        });

        let drained = drain_status(&rx);
        let strings: Vec<&str> = drained
            .iter()
            .filter_map(|o| o.as_ref().map(|(_, _, s)| s.as_str()))
            .collect();
        assert_eq!(strings, vec!["connect retry 2/5…", "server retry 1/3…"]);
    }

    /// Spec: non-2xx response headers do NOT produce a
    /// `transcribing…` status — the row will get a final error
    /// candidate shortly and we don't want to flicker through a
    /// misleading "transcribing" state on a 4xx/5xx.
    #[test]
    fn picker_status_sink_drops_non_2xx_response_headers() {
        let (tx, rx) = std::sync::mpsc::channel::<PickerMessage>();
        let sink = PickerStatusSink::new(tx, Provider::Mistral, "voxtral".to_string(), false);

        let now = Instant::now();
        sink.emit(PipelineEvent::ResponseHeaders {
            status: 401,
            t: now,
        });
        sink.emit(PipelineEvent::ResponseHeaders {
            status: 500,
            t: now,
        });

        let drained = drain_status(&rx);
        assert!(
            drained.is_empty(),
            "non-2xx response headers must not emit a status; got: {:?}",
            drained
        );
    }

    /// Spec: the sink carries the (provider, model) identity it
    /// was constructed with verbatim — the picker UI relies on
    /// these to route the status to the correct row.
    #[test]
    fn picker_status_sink_preserves_row_identity() {
        let (tx, rx) = std::sync::mpsc::channel::<PickerMessage>();
        let sink = PickerStatusSink::new(
            tx,
            Provider::Mistral,
            "voxtral-mini-2602".to_string(),
            false,
        );

        let now = Instant::now();
        sink.emit(PipelineEvent::RequestStarted {
            endpoint: "https://x".into(),
            t: now,
        });

        let drained = drain_status(&rx);
        assert_eq!(drained.len(), 1);
        let (p, m, _) = drained[0].as_ref().expect("status message");
        assert_eq!(*p, Provider::Mistral);
        assert_eq!(m, "voxtral-mini-2602");
    }
}
