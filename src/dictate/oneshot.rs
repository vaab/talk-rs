//! One-shot dictation mode.
//!
//! Encodes audio and streams it directly to a single transcription
//! request.  Audio is independently recorded to a cache OGG via the
//! shared [`AudioBuffer`](super::realtime::AudioBuffer) so that a
//! transcription failure never truncates the recording.

use super::realtime::{buffer_feeder, ogg_recording_task, AudioBuffer};
use crate::audio::bt_profile;
use crate::audio::recording_feedback::{RecordingBadgeTeardown, RecordingFeedback};
use crate::audio::{AudioCapture, AudioWriter, OggOpusWriter};
use crate::config::{AudioConfig, Config, Provider};
use crate::error::TalkError;
use crate::transcription::{self, OneShotTranscriber, TranscriptionBody, TranscriptionResult};
use crate::x11::overlay::IndicatorKind;
use crate::x11::visualizer::VisualizerHandle;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Maximum number of retry attempts for the transcription pipeline
/// during recording (before the user stops).
const MAX_LIVE_RETRIES: u32 = 3;

// ── Encode pipeline ─────────────────────────────────────────────────

/// Spawn an encode task (PCM → OGG Opus) and a transcription task.
///
/// Returns `(feeder_handle, encode_handle, transcribe_handle,
/// encode_done_rx)`.  The `encode_done_rx` oneshot fires when the
/// encode task finishes (needed for the file-input race).
#[allow(clippy::type_complexity)]
fn spawn_encode_pipeline(
    buffer: &Arc<AudioBuffer>,
    audio_config: AudioConfig,
    transcriber: Box<dyn OneShotTranscriber>,
) -> (
    tokio::task::JoinHandle<()>,
    tokio::task::JoinHandle<Result<(), TalkError>>,
    tokio::task::JoinHandle<Result<TranscriptionResult, TalkError>>,
    tokio::sync::oneshot::Receiver<()>,
) {
    let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
    let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(25);
    let (encode_done_tx, encode_done_rx) = tokio::sync::oneshot::channel::<()>();

    // Feeder: buffer → fwd_tx
    let feeder_handle = tokio::spawn(buffer_feeder(Arc::clone(buffer), fwd_tx, 0));

    // Encode: fwd_rx → OGG Opus → stream_tx
    let encode_handle = tokio::spawn(async move {
        let mut rx = fwd_rx;
        let mut writer = OggOpusWriter::new(audio_config)?;

        let header = writer.header()?;
        if stream_tx.send(header).await.is_err() {
            log::warn!("transcription stream closed during header send");
            let _ = encode_done_tx.send(());
            return Ok::<(), TalkError>(());
        }

        while let Some(pcm_chunk) = rx.recv().await {
            let encoded_data = writer.write_pcm(&pcm_chunk)?;
            if !encoded_data.is_empty() && stream_tx.send(encoded_data).await.is_err() {
                log::warn!("transcription stream closed during audio send");
                break;
            }
        }

        // Finalize writer
        let remaining = writer.finalize()?;
        if !remaining.is_empty() {
            let _ = stream_tx.send(remaining).await;
        }

        // stream_tx is dropped here, closing the channel.
        let _ = encode_done_tx.send(());
        Ok::<(), TalkError>(())
    });

    // Transcribe: stream_rx → API
    let transcribe_handle = tokio::spawn(async move {
        transcriber
            .fetch_transcription(TranscriptionBody::Pipe {
                chunks: stream_rx,
                file_name: "audio.ogg".to_string(),
            })
            .await
    });

    (
        feeder_handle,
        encode_handle,
        transcribe_handle,
        encode_done_rx,
    )
}

/// Abort a pipeline's feeder, encode, and transcribe tasks.
fn abort_pipeline(
    feeder: &tokio::task::JoinHandle<()>,
    encode: &tokio::task::JoinHandle<Result<(), TalkError>>,
    transcribe: &tokio::task::JoinHandle<Result<TranscriptionResult, TalkError>>,
) {
    feeder.abort();
    encode.abort();
    transcribe.abort();
}

// ── Main entry point ────────────────────────────────────────────────

/// One-shot dictation mode.
///
/// Records audio to a cache OGG via [`ogg_recording_task`] and
/// independently streams encoded OGG to a transcription API.  If the
/// transcription pipeline fails during recording, it is restarted
/// immediately from the beginning of the shared [`AudioBuffer`] so
/// no audio is lost.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn dictate_oneshot(
    capture: &mut dyn AudioCapture,
    from_file: bool,
    audio_config: AudioConfig,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    cache_ogg_path: &std::path::Path,
    transcriber: Box<dyn OneShotTranscriber>,
    shutdown: &CancellationToken,
    feedback: &mut RecordingFeedback,
    visualizer: Option<&VisualizerHandle>,
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    diarize: bool,
    chain_active: bool,
    mut bt_guard: bt_profile::HeadsetGuard,
) -> (
    Result<TranscriptionResult, TalkError>,
    Option<std::time::Instant>,
) {
    // ── OGG recording (independent of transcription) ────────────────
    log::info!("caching audio to: {}", cache_ogg_path.display());
    let buffer = Arc::new(AudioBuffer::new());
    let cache_ogg_task = tokio::spawn(ogg_recording_task(
        audio_rx,
        cache_ogg_path.to_path_buf(),
        audio_config.clone(),
        Arc::clone(&buffer),
    ));

    // ── Initial transcription pipeline ──────────────────────────────
    let (mut feeder_handle, mut encode_handle, mut transcribe_handle, mut encode_done_rx) =
        spawn_encode_pipeline(&buffer, audio_config.clone(), transcriber);

    // ── Wait for recording to end ───────────────────────────────────
    let rec_start = std::time::Instant::now();
    let t_stop: Option<std::time::Instant>;
    if from_file {
        log::info!("transcribing audio file (one-shot)...");
        tokio::select! {
            _ = shutdown.cancelled() => {
                t_stop = Some(std::time::Instant::now());
                log::info!("aborting after {:.2}s", rec_start.elapsed().as_secs_f64());
                if let Err(e) = capture.stop() {
                    return (Err(e), t_stop);
                }
            }
            _ = &mut encode_done_rx => {
                t_stop = Some(std::time::Instant::now());
                log::info!(
                    "audio file playback complete after {:.2}s",
                    rec_start.elapsed().as_secs_f64()
                );
            }
        }
    } else {
        // Live recording — wait for SIGINT while monitoring the
        // transcription pipeline health.  If the feeder task finishes
        // prematurely (pipeline broke), retry immediately.
        let mut live_retries: u32 = 0;

        log::info!("recording — waiting for shutdown signal");
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    t_stop = Some(std::time::Instant::now());
                    log::info!(
                        "shutdown signal received after {:.2}s recording — stopping capture",
                        rec_start.elapsed().as_secs_f64()
                    );
                    if let Err(e) = capture.stop() {
                        return (Err(e), t_stop);
                    }
                    if let Some(t) = t_stop {
                        log::info!("timing: stop +{}ms capture_stopped", t.elapsed().as_millis());
                    }
                    break;
                }
                feeder_result = &mut feeder_handle => {
                    // Feeder finished during recording → the pipeline
                    // broke (transcriber or encode task closed its end).
                    let reason = match feeder_result {
                        Ok(()) => "pipeline closed".to_string(),
                        Err(e) => format!("feeder panic: {}", e),
                    };
                    log::warn!(
                         "transcription pipeline failed during recording: {} — \
                         OGG recording continues",
                         reason
                     );

                    abort_pipeline(
                        &feeder_handle,
                        &encode_handle,
                        &transcribe_handle,
                    );

                    live_retries += 1;
                    if chain_active || live_retries > MAX_LIVE_RETRIES {
                        let msg = if chain_active {
                            "Transcription busy — trying next chain entry after recording".to_string()
                        } else { format!("Transcription failed ({} retries exhausted) — will retry after recording", MAX_LIVE_RETRIES) };
                        log::warn!("{}", msg);
                        if let Some(viz) = visualizer {
                            viz.push_message(&msg);
                        }
                        // Continue recording — will retry from OGG
                        // after recording stops.
                        shutdown.cancelled().await;
                        t_stop = Some(std::time::Instant::now());
                        log::info!(
                            "shutdown after {:.2}s — stopping capture",
                            rec_start.elapsed().as_secs_f64()
                        );
                        if let Err(e) = capture.stop() {
                            return (Err(e), t_stop);
                        }
                        if let Some(t) = t_stop {
                            log::info!("timing: stop +{}ms capture_stopped", t.elapsed().as_millis());
                        }
                        break;
                    }

                    let msg = format!(
                        "Transcription failed: {} — retrying ({}/{})",
                        reason, live_retries, MAX_LIVE_RETRIES
                    );
                    log::info!("{}", msg);
                    if let Some(viz) = visualizer {
                        viz.push_message(&msg);
                    }
                    match transcription::create_oneshot_transcriber(
                        config,
                        provider,
                        model,
                        diarize,
                        // Reconnect after a one-shot failure stays
                        // on the autonomous-pipeline `Proportional`
                        // policy — dictate has no human watching.
                        transcription::RequestTimeoutPolicy::Proportional,
                    ) {
                        Ok(new_transcriber) => {
                            let pipeline = spawn_encode_pipeline(
                                &buffer,
                                audio_config.clone(),
                                new_transcriber,
                            );
                            feeder_handle = pipeline.0;
                            encode_handle = pipeline.1;
                            transcribe_handle = pipeline.2;
                            // encode_done_rx is only awaited in the
                            // from-file branch; keep the receiver
                            // alive so the sender side does not error.
                            _ = std::mem::replace(&mut encode_done_rx, pipeline.3);
                        }
                        Err(e) => {
                            let msg = format!(
                                "Transcription reconnect failed: {} — \
                                 will retry after recording",
                                e
                            );
                            log::warn!("{}", msg);
                            if let Some(viz) = visualizer {
                                viz.push_message(&msg);
                            }
                            // Fall through — will retry from OGG after
                            // recording stops.
                            shutdown.cancelled().await;
                            t_stop = Some(std::time::Instant::now());
                            log::info!(
                                "shutdown after {:.2}s — stopping capture",
                                rec_start.elapsed().as_secs_f64()
                            );
                            if let Err(e) = capture.stop() {
                                return (Err(e), t_stop);
                            }
                            if let Some(t) = t_stop {
                                log::info!("timing: stop +{}ms capture_stopped", t.elapsed().as_millis());
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Restore the Bluetooth headset to its high-quality profile
        // (typically A2DP) the instant the microphone capture stops,
        // in parallel with transcription + paste.  The user hears
        // their music / system audio in stereo again within ~1 s of
        // pressing the toggle, instead of waiting for the entire
        // dictation pipeline to finish.  Drop on the empty guard at
        // function exit is then a no-op.
        bt_guard.restore_now_async();

        // Immediate audible + visual feedback: the user hears the
        // stop sound and sees the "transcribing" badge the instant
        // they toggle, not after the API call finishes.
        feedback.teardown_recording(RecordingBadgeTeardown::KeepVisible);
        feedback.play_stop().await;
        if let Some(viz) = visualizer {
            viz.hide();
        }
        if let Some(o) = feedback.overlay() {
            o.show(IndicatorKind::Transcribing);
        }
        if let Some(t) = t_stop {
            log::debug!(
                "one-shot: overlay→Transcribing, +{}ms since SIGINT",
                t.elapsed().as_millis()
            );
        }
    }

    // ── Wait for OGG recording task (no timeout) ────────────────────
    //
    // The OGG task finishes as soon as the source channel closes
    // (trailing bytes + fsync — very fast).  Never abandon it.
    match cache_ogg_task.await {
        Ok(Ok(())) => log::debug!("cache OGG task completed"),
        Ok(Err(err)) => {
            feeder_handle.abort();
            encode_handle.abort();
            transcribe_handle.abort();
            return (Err(err), t_stop);
        }
        Err(err) => {
            feeder_handle.abort();
            encode_handle.abort();
            transcribe_handle.abort();
            return (
                Err(TalkError::Audio(format!("cache OGG task failed: {err}"))),
                t_stop,
            );
        }
    };
    if let Some(t) = t_stop {
        log::info!("timing: stop +{}ms ogg_flushed", t.elapsed().as_millis());
        log::debug!(
            "one-shot: cache_ogg finalized, +{}ms since SIGINT",
            t.elapsed().as_millis()
        );
    }

    // ── Empty / dead-signal recording guard ────────────────────────────
    //
    // Skip transcription when no usable audio was captured:
    //  - buffer is empty (instant stop, or dead signal paused the pipeline)
    //  - overlay reports no live audio was ever detected (entire session
    //    was dead signal)
    let skip = buffer.is_empty().await
        || feedback
            .overlay()
            .is_some_and(|overlay| !overlay.had_live_audio());
    if skip {
        log::warn!("no usable audio recorded — skipping transcription");
        feeder_handle.abort();
        encode_handle.abort();
        transcribe_handle.abort();
        return (Ok(TranscriptionResult::default()), t_stop);
    }

    // ── Wait for transcription result ───────────────────────────────
    //
    // No artificial timeout here — the HTTP client's `read_timeout`,
    // `tcp_user_timeout`, and TCP keepalive settings detect dead
    // connections within a few seconds.  The server may legitimately
    // take a while to process long recordings.
    let t0 = std::time::Instant::now();
    log::info!("waiting for transcription result");
    log::debug!("one-shot: awaiting transcribe_handle (no timeout)");

    // Heartbeat: log every 2s while transcribe_handle is pending so
    // we can distinguish a slow HTTP response from a deadlock or lost
    // task.  Cancelled via `heartbeat_cancel` once the await returns.
    let heartbeat_cancel = CancellationToken::new();
    let hb_token = heartbeat_cancel.clone();
    let hb_start = std::time::Instant::now();
    let hb_handle = tokio::spawn(async move {
        let mut elapsed = 0u64;
        loop {
            tokio::select! {
                _ = hb_token.cancelled() => break,
                _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {
                    elapsed += 2;
                    log::debug!(
                        "one-shot: still waiting for transcribe result ({}s elapsed, wall {:.1}s)",
                        elapsed,
                        hb_start.elapsed().as_secs_f64()
                    );
                }
            }
        }
    });

    let result = match transcribe_handle.await {
        Ok(Ok(result)) => {
            log::info!(
                "transcription completed after {:.2}s",
                t0.elapsed().as_secs_f64()
            );
            log::debug!(
                "one-shot: transcribe_handle returned OK after {}ms",
                t0.elapsed().as_millis()
            );
            Ok(result)
        }
        Ok(Err(err)) => {
            log::warn!(
                "transcription failed after {:.2}s: {}",
                t0.elapsed().as_secs_f64(),
                err
            );
            log::debug!(
                "one-shot: transcribe_handle returned ERR after {}ms",
                t0.elapsed().as_millis()
            );
            Err(err)
        }
        Err(err) => {
            log::warn!(
                "one-shot: transcription task panicked after {}ms: {}",
                t0.elapsed().as_millis(),
                err
            );
            Err(TalkError::Transcription(format!(
                "transcription task panicked: {}",
                err
            )))
        }
    };

    // Stop the heartbeat task before continuing.
    heartbeat_cancel.cancel();
    let _ = hb_handle.await;

    // Clean up: abort remaining pipeline tasks (may already be done).
    feeder_handle.abort();
    encode_handle.abort();

    log::info!(
        "dictate_oneshot total elapsed: {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    (result, t_stop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::file_source::OggFileSource;
    use crate::audio::mock::MockAudioCapture;
    use crate::audio::recording_feedback::{RecordingFeedbackOptions, RecordingOverlayOptions};
    use crate::transcription::TranscriptionMetadata;
    use async_trait::async_trait;
    use std::path::Path;
    use tokio::sync::{mpsc, oneshot};

    struct InspectingTranscriber {
        payload: std::sync::Mutex<Option<oneshot::Sender<Vec<u8>>>>,
        fail_early: bool,
    }

    #[async_trait]
    impl OneShotTranscriber for InspectingTranscriber {
        async fn validate(&self) -> Result<(), TalkError> {
            Ok(())
        }

        async fn fetch_transcription(
            &self,
            body: TranscriptionBody,
        ) -> Result<TranscriptionResult, TalkError> {
            if self.fail_early {
                return Err(TalkError::Transcription("mock downstream failure".into()));
            }
            let TranscriptionBody::Pipe {
                mut chunks,
                file_name,
            } = body
            else {
                panic!("production pipeline must pass encoded chunks");
            };
            assert_eq!(file_name, "audio.ogg");
            let mut bytes = Vec::new();
            while let Some(chunk) = chunks.recv().await {
                bytes.extend(chunk);
            }
            self.payload
                .lock()
                .expect("lock payload")
                .take()
                .expect("payload sender")
                .send(bytes)
                .expect("receive payload");
            Ok(TranscriptionResult {
                text: "captured speech".into(),
                metadata: TranscriptionMetadata::default(),
                diarization: None,
                segments: None,
            })
        }
    }

    fn inspecting_transcriber() -> (Box<dyn OneShotTranscriber>, oneshot::Receiver<Vec<u8>>) {
        let (tx, rx) = oneshot::channel();
        (
            Box::new(InspectingTranscriber {
                payload: std::sync::Mutex::new(Some(tx)),
                fail_early: false,
            }),
            rx,
        )
    }

    struct BusyTranscriber {
        started: std::sync::Mutex<Option<oneshot::Sender<()>>>,
    }

    #[async_trait]
    impl OneShotTranscriber for BusyTranscriber {
        async fn validate(&self) -> Result<(), TalkError> {
            Ok(())
        }

        async fn fetch_transcription(
            &self,
            _body: TranscriptionBody,
        ) -> Result<TranscriptionResult, TalkError> {
            if let Some(started) = self.started.lock().expect("started lock").take() {
                let _ = started.send(());
            }
            Err(crate::error::PipelineFailure::new(
                "Mistral",
                crate::error::PipelinePhase::Request,
                1,
                1,
                "http://localhost/v1/audio/transcriptions",
                crate::error::PipelineFailureKind::HttpStatus {
                    status: 429,
                    body: "busy".into(),
                },
            )
            .into())
        }
    }

    #[tokio::test]
    async fn live_chain_preserves_busy_failure_for_file_fallback() {
        let dir = tempfile::tempdir().expect("temp directory");
        let cache = dir.path().join("busy.ogg");
        struct StopNotifyingCapture(Option<oneshot::Sender<()>>);
        impl AudioCapture for StopNotifyingCapture {
            fn start(&mut self) -> Result<mpsc::Receiver<Vec<i16>>, TalkError> {
                Err(TalkError::Audio("test capture is already started".into()))
            }

            fn stop(&mut self) -> Result<(), TalkError> {
                if let Some(stopped) = self.0.take() {
                    let _ = stopped.send(());
                }
                Ok(())
            }
        }
        let (stopped_tx, stopped_rx) = oneshot::channel();
        let mut capture = StopNotifyingCapture(Some(stopped_tx));
        let (tx, rx) = mpsc::channel(8);
        let (started_tx, started_rx) = oneshot::channel();
        let shutdown = CancellationToken::new();
        let stop = shutdown.clone();
        let feeder = tokio::spawn(async move {
            for _ in 0..5 {
                tx.send(vec![9000; 320]).await.expect("PCM sent");
            }
            started_rx.await.expect("busy request started");
            stop.cancel();
            stopped_rx
                .await
                .expect("capture stopped before input closes");
            drop(tx);
        });
        let mut feedback = no_device_feedback();
        let (result, _) = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            dictate_oneshot(
                &mut capture,
                false,
                AudioConfig::new(),
                rx,
                &cache,
                Box::new(BusyTranscriber {
                    started: std::sync::Mutex::new(Some(started_tx)),
                }),
                &shutdown,
                &mut feedback,
                None,
                &offline_config(dir.path()),
                Provider::Mistral,
                None,
                false,
                true,
                bt_profile::HeadsetGuard::new(None),
            ),
        )
        .await
        .expect("busy pipeline finishes");
        feeder.await.expect("PCM feeder finished");
        let error = result.expect_err("busy error");
        assert!(matches!(error, TalkError::Pipeline(ref failure)
            if matches!(failure.kind, crate::error::PipelineFailureKind::HttpStatus { status: 429, .. })));
        assert!(cache.is_file());
    }

    fn no_device_feedback() -> RecordingFeedback {
        RecordingFeedback::new(RecordingFeedbackOptions {
            no_sounds: true,
            no_boop: true,
            no_overlay: true,
            viz: None,
            mono: false,
            boop_interval_ms: 0,
            capture_rate: 16_000,
            pause_audio: false,
            suppress_boop: None,
            overlay: RecordingOverlayOptions {
                silence_tx: None,
                auto_pause: false,
                telemetry_rx: None,
            },
        })
    }

    fn offline_config(dir: &Path) -> Config {
        serde_yaml::from_str(&format!("output_dir: {}\nproviders: {{}}\n", dir.display()))
            .expect("offline configuration")
    }

    async fn decoded_samples(bytes: &[u8]) -> Vec<i16> {
        let dir = tempfile::tempdir().expect("temp directory");
        let path = dir.path().join("payload.ogg");
        tokio::fs::write(&path, bytes).await.expect("write payload");
        let mut source = OggFileSource::new(&path).expect("valid OGG header");
        let mut rx = source.start().expect("decode OGG");
        let mut decoded = Vec::new();
        while let Some(chunk) = rx.recv().await {
            decoded.extend(chunk);
        }
        decoded
    }

    #[tokio::test]
    async fn production_pipeline_flushes_complete_decodable_ogg_to_transcriber() {
        let buffer = Arc::new(AudioBuffer::new());
        let samples: Vec<i16> = (0..(3 * 320 + 67))
            .map(|i| ((i as f32 * 0.12).sin() * 12000.0) as i16)
            .collect();
        buffer.push(samples.clone()).await;
        buffer.close();
        let (transcriber, payload) = inspecting_transcriber();

        let (feeder, encoder, transcription, done) =
            spawn_encode_pipeline(&buffer, AudioConfig::new(), transcriber);
        done.await.expect("encoder completion signal");
        feeder.await.expect("feeder completed");
        encoder.await.expect("encoder joined").expect("encoded");
        assert_eq!(
            transcription
                .await
                .expect("transcriber joined")
                .expect("text")
                .text,
            "captured speech"
        );
        let decoded = decoded_samples(&payload.await.expect("transcriber payload")).await;
        assert!(decoded.len() >= samples.len());
        assert!(decoded.len() - samples.len() < 320);
        assert!(decoded.iter().any(|&sample| sample.abs() > 1000));
    }

    #[tokio::test]
    async fn mock_capture_composes_with_production_oneshot_and_cache() {
        let dir = tempfile::tempdir().expect("temp directory");
        let cache = dir.path().join("capture.ogg");
        let mut capture = MockAudioCapture::new(16_000, 1, 440.0);
        let mut raw_rx = capture.start().expect("start mock capture");
        let (tx, rx) = mpsc::channel(8);
        let shutdown = CancellationToken::new();
        let stop = shutdown.clone();
        let relay = tokio::spawn(async move {
            for _ in 0..5 {
                tx.send(raw_rx.recv().await.expect("mock PCM chunk"))
                    .await
                    .expect("relay PCM");
            }
            drop(tx);
            stop.cancel();
        });
        let (transcriber, payload) = inspecting_transcriber();
        let mut feedback = no_device_feedback();
        let (result, _) = dictate_oneshot(
            &mut capture,
            false,
            AudioConfig::new(),
            rx,
            &cache,
            transcriber,
            &shutdown,
            &mut feedback,
            None,
            &offline_config(dir.path()),
            Provider::Mistral,
            None,
            false,
            false,
            bt_profile::HeadsetGuard::new(None),
        )
        .await;
        relay.await.expect("relay finished");
        assert_eq!(result.expect("dictation result").text, "captured speech");
        let uploaded = decoded_samples(&payload.await.expect("uploaded OGG")).await;
        let cached = decoded_samples(&tokio::fs::read(&cache).await.expect("cache OGG")).await;
        for pcm in [uploaded, cached] {
            assert!(pcm.len().abs_diff(5 * 320) <= 320);
            assert!(pcm.iter().any(|&sample| sample.abs() > 1000));
        }
    }

    #[tokio::test]
    async fn downstream_failure_does_not_truncate_cached_ogg() {
        let dir = tempfile::tempdir().expect("temp directory");
        let cache = dir.path().join("failed.ogg");
        let mut capture = MockAudioCapture::new(16_000, 1, 440.0);
        let (tx, rx) = mpsc::channel(8);
        for _ in 0..4 {
            tx.send(vec![9000; 320]).await.expect("send PCM");
        }
        drop(tx);
        let mut feedback = no_device_feedback();
        let (result, _) = dictate_oneshot(
            &mut capture,
            true,
            AudioConfig::new(),
            rx,
            &cache,
            Box::new(InspectingTranscriber {
                payload: std::sync::Mutex::new(None),
                fail_early: true,
            }),
            &CancellationToken::new(),
            &mut feedback,
            None,
            &offline_config(dir.path()),
            Provider::Mistral,
            None,
            false,
            false,
            bt_profile::HeadsetGuard::new(None),
        )
        .await;
        assert_eq!(
            result.expect_err("transcriber fails").to_string(),
            "Transcription error: mock downstream failure"
        );
        let cached = decoded_samples(&tokio::fs::read(&cache).await.expect("cached OGG")).await;
        assert!(cached.len().abs_diff(4 * 320) <= 320);
        assert!(cached.iter().any(|&sample| sample.abs() > 1000));
    }

    #[tokio::test]
    async fn cache_creation_failure_closes_audio_and_returns_error() {
        let dir = tempfile::tempdir().expect("temp directory");
        let cache = dir.path().join("missing-parent/capture.ogg");
        let mut capture = MockAudioCapture::new(16_000, 1, 440.0);
        let (tx, rx) = mpsc::channel(8);
        tx.send(vec![9000; 320]).await.expect("send finite PCM");
        drop(tx);
        let mut feedback = no_device_feedback();
        let (transcriber, payload) = inspecting_transcriber();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            dictate_oneshot(
                &mut capture,
                true,
                AudioConfig::new(),
                rx,
                &cache,
                transcriber,
                &CancellationToken::new(),
                &mut feedback,
                None,
                &offline_config(dir.path()),
                Provider::Mistral,
                None,
                false,
                false,
                bt_profile::HeadsetGuard::new(None),
            ),
        )
        .await
        .expect("dictation must not hang after cache failure")
        .0
        .expect_err("cache failure must be returned");
        assert!(result.to_string().contains("missing-parent"), "{result}");
        drop(payload);
    }
}

/// Performance harness, item `single-opus-encode`: the correctness a
/// one-encoder design must keep, proven on the real one-shot pipeline
/// (cache writer + live upload pipeline + cursor-zero replay):
///
/// - the cache OGG and every upload are well-formed Ogg Opus (one
///   logical stream, `OpusHead`/`OpusTags`, monotonic granules, EOS on
///   the last page), decode with libopus to exactly the accepted
///   samples (finite capture with a non-frame-aligned tail), and match
///   the reference PCM within the codec's error (SNR > 10 dB);
/// - a transcriber that fails early, one that fails mid-stream (forcing
///   a cursor-zero replay into a fresh pipeline), and one that blocks
///   never truncate the recording, and the replayed upload is the
///   complete recording from its first sample.
///
/// Encode work per recorded frame is measured by the binary cuts in
/// `tests/perf` (`encode-passes`).
#[cfg(test)]
mod perf_single_encode {
    use super::*;
    use crate::audio::perf_audio::{decode_ogg_strict, envelope_correlation};
    use async_trait::async_trait;
    use tokio::sync::{mpsc, Mutex as AsyncMutex};

    /// 3.07 s of speech-like 16 kHz PCM: 153 chunks of 320 plus a
    /// 237-sample tail (not frame aligned).
    fn reference_pcm() -> Vec<i16> {
        (0..(153 * 320 + 237))
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let env = (t * 2.7 * std::f32::consts::TAU).sin().abs();
                ((env
                    * (0.35 * (t * 190.0 * std::f32::consts::TAU).sin()
                        + 0.2 * (t * 760.0 * std::f32::consts::TAU).sin()))
                    * 20_000.0) as i16
            })
            .collect()
    }

    fn assert_complete_recording(what: &str, bytes: &[u8], reference: &[i16]) {
        let pcm = decode_ogg_strict(bytes);
        assert_eq!(
            pcm.len(),
            reference.len(),
            "{what}: sample count (tail included)"
        );
        let r = envelope_correlation(reference, &pcm);
        assert!(
            r > 0.95,
            "{what}: decoded audio does not match the input (r={r:.3})"
        );
    }

    #[derive(Clone, Copy)]
    enum Behaviour {
        Succeed,
        /// Fail before reading any audio.
        FailEarly,
        /// Read `n` chunks then drop the stream (pipeline breaks).
        FailAfter(usize),
        /// Read slowly (5 ms per chunk).
        Slow,
    }

    /// Records every upload body it receives (per pipeline instance).
    struct Recorder {
        behaviour: Behaviour,
        uploads: Arc<AsyncMutex<Vec<Vec<u8>>>>,
    }

    #[async_trait]
    impl OneShotTranscriber for Recorder {
        async fn validate(&self) -> Result<(), TalkError> {
            Ok(())
        }

        async fn fetch_transcription(
            &self,
            body: TranscriptionBody,
        ) -> Result<TranscriptionResult, TalkError> {
            let TranscriptionBody::Pipe { mut chunks, .. } = body else {
                panic!("live pipeline must stream chunks");
            };
            let mut bytes = Vec::new();
            let mut n = 0;
            match self.behaviour {
                Behaviour::FailEarly => return Err(TalkError::Transcription("fail early".into())),
                Behaviour::FailAfter(limit) => {
                    while n < limit {
                        match chunks.recv().await {
                            Some(c) => bytes.extend(c),
                            None => break,
                        }
                        n += 1;
                    }
                    self.uploads.lock().await.push(bytes);
                    return Err(TalkError::Transcription("mid-stream failure".into()));
                }
                Behaviour::Succeed | Behaviour::Slow => {
                    while let Some(c) = chunks.recv().await {
                        if matches!(self.behaviour, Behaviour::Slow) {
                            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                        }
                        bytes.extend(c);
                    }
                }
            }
            self.uploads.lock().await.push(bytes);
            Ok(TranscriptionResult {
                text: "ok".into(),
                ..TranscriptionResult::default()
            })
        }
    }

    struct Finished {
        cache: Vec<u8>,
        uploads: Vec<Vec<u8>>,
        result: Result<TranscriptionResult, TalkError>,
    }

    /// Run the production live one-shot path (`dictate_oneshot`, not
    /// from file) on the finite reference capture, stopped like SIGINT
    /// once every chunk was delivered.
    async fn record(behaviour: Behaviour) -> Finished {
        let dir = tempfile::tempdir().expect("temp dir");
        let cache = dir.path().join("rec.ogg");
        let reference = reference_pcm();
        let (tx, rx) = mpsc::channel(8);
        let shutdown = CancellationToken::new();
        let stop = shutdown.clone();
        // Real stop order: SIGINT → capture.stop() → the capture's
        // channel closes.  Closing it first would look like a broken
        // pipeline to the live loop.
        struct StopNotifying(Option<tokio::sync::oneshot::Sender<()>>);
        impl AudioCapture for StopNotifying {
            fn start(&mut self) -> Result<mpsc::Receiver<Vec<i16>>, TalkError> {
                Err(TalkError::Audio("already started".into()))
            }
            fn stop(&mut self) -> Result<(), TalkError> {
                if let Some(tx) = self.0.take() {
                    let _ = tx.send(());
                }
                Ok(())
            }
        }
        let (stopped_tx, stopped_rx) = tokio::sync::oneshot::channel();
        let feeder = tokio::spawn(async move {
            for chunk in reference.chunks(320) {
                tx.send(chunk.to_vec()).await.expect("PCM accepted");
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
            stop.cancel();
            stopped_rx.await.expect("capture stopped");
            drop(tx);
        });
        let uploads = Arc::new(AsyncMutex::new(Vec::new()));
        let mut capture = StopNotifying(Some(stopped_tx));
        let mut feedback = no_device_feedback();
        let (result, _) = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            dictate_oneshot(
                &mut capture,
                false,
                AudioConfig::new(),
                rx,
                &cache,
                Box::new(Recorder {
                    behaviour,
                    uploads: Arc::clone(&uploads),
                }),
                &shutdown,
                &mut feedback,
                None,
                &offline_config(dir.path()),
                Provider::Mistral,
                None,
                false,
                false,
                bt_profile::HeadsetGuard::new(None),
            ),
        )
        .await
        .expect("dictation finishes");
        feeder.await.expect("feeder");
        let cache_bytes = tokio::fs::read(&cache).await.expect("cache OGG");
        let uploads = uploads.lock().await.clone();
        Finished {
            cache: cache_bytes,
            uploads,
            result,
        }
    }

    /// Success: the cache and the upload are both the complete,
    /// decodable recording of exactly the accepted samples.
    #[tokio::test(flavor = "multi_thread")]
    async fn perf_single_encode_cache_and_upload_are_the_recording() {
        let reference = reference_pcm();
        let done = record(Behaviour::Succeed).await;
        assert_eq!(done.result.expect("transcription").text, "ok");
        assert_complete_recording("cache", &done.cache, &reference);
        assert_eq!(done.uploads.len(), 1);
        assert_complete_recording("upload", &done.uploads[0], &reference);
    }

    /// Upload failures and a slow consumer never truncate or corrupt
    /// the recording (it is owned by the recorder, not the network).
    #[tokio::test(flavor = "multi_thread")]
    async fn perf_single_encode_failures_never_truncate_cache() {
        let reference = reference_pcm();
        for behaviour in [
            Behaviour::FailEarly,
            Behaviour::FailAfter(20),
            Behaviour::Slow,
        ] {
            let done = record(behaviour).await;
            assert_complete_recording("cache", &done.cache, &reference);
            if matches!(behaviour, Behaviour::Slow) {
                assert!(done.result.is_ok());
                assert_complete_recording("slow upload", &done.uploads[0], &reference);
            }
        }
    }

    /// Cursor-zero replay, the mechanism `dictate_oneshot` uses after a
    /// mid-recording pipeline failure: the first pipeline breaks after
    /// 20 chunks while audio is still arriving; a fresh pipeline on the
    /// same shared buffer from cursor 0 uploads the complete recording
    /// from its first sample, byte-compatible with a single pass.
    #[tokio::test(flavor = "multi_thread")]
    async fn perf_single_encode_replay_from_cursor_zero_is_complete() {
        let reference = reference_pcm();
        let buffer = Arc::new(AudioBuffer::new());
        let uploads = Arc::new(AsyncMutex::new(Vec::new()));
        let producer = {
            let buffer = Arc::clone(&buffer);
            let reference = reference.clone();
            tokio::spawn(async move {
                for chunk in reference.chunks(320) {
                    buffer.push(chunk.to_vec()).await;
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                buffer.close();
            })
        };
        let first = spawn_encode_pipeline(
            &buffer,
            AudioConfig::new(),
            Box::new(Recorder {
                behaviour: Behaviour::FailAfter(20),
                uploads: Arc::clone(&uploads),
            }),
        );
        assert!(first.2.await.expect("first pipeline joined").is_err());
        abort_pipeline(
            &first.0,
            &first.1,
            &tokio::spawn(async {
                Err::<TranscriptionResult, TalkError>(TalkError::Transcription("done".into()))
            }),
        );
        let second = spawn_encode_pipeline(
            &buffer,
            AudioConfig::new(),
            Box::new(Recorder {
                behaviour: Behaviour::Succeed,
                uploads: Arc::clone(&uploads),
            }),
        );
        assert_eq!(
            second.2.await.expect("replay joined").expect("replay").text,
            "ok"
        );
        producer.await.expect("producer");
        let uploads = uploads.lock().await.clone();
        assert_eq!(uploads.len(), 2);
        assert!(
            uploads[0].len() < uploads[1].len(),
            "first upload was cut short"
        );
        assert_complete_recording("replayed upload", &uploads[1], &reference);
    }

    fn no_device_feedback() -> RecordingFeedback {
        RecordingFeedback::new(crate::audio::recording_feedback::RecordingFeedbackOptions {
            no_sounds: true,
            no_boop: true,
            no_overlay: true,
            viz: None,
            mono: false,
            boop_interval_ms: 0,
            capture_rate: 16_000,
            pause_audio: false,
            suppress_boop: None,
            overlay: crate::audio::recording_feedback::RecordingOverlayOptions {
                silence_tx: None,
                auto_pause: false,
                telemetry_rx: None,
            },
        })
    }

    fn offline_config(dir: &std::path::Path) -> Config {
        serde_yaml::from_str(&format!("output_dir: {}\nproviders: {{}}\n", dir.display()))
            .expect("offline configuration")
    }
}
