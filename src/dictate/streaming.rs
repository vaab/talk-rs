//! Batch dictation mode.
//!
//! Encodes audio and streams it directly to a single transcription
//! request.  Audio is independently recorded to a cache WAV via the
//! shared [`AudioBuffer`](super::realtime::AudioBuffer) so that a
//! transcription failure never truncates the recording.

use super::realtime::{buffer_feeder, wav_recording_task, AudioBuffer};
use crate::audio::indicator::SoundPlayer;
use crate::audio::{AudioCapture, AudioWriter, OggOpusWriter};
use crate::config::{AudioConfig, Config, Provider};
use crate::error::TalkError;
use crate::transcription::{self, BatchTranscriber, TranscriptionResult};
use crate::x11::overlay::{IndicatorKind, OverlayHandle};
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
    transcriber: Box<dyn BatchTranscriber>,
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

        // stream_tx is dropped here, closing the channel
        let _ = encode_done_tx.send(());
        Ok::<(), TalkError>(())
    });

    // Transcribe: stream_rx → API
    let transcribe_handle =
        tokio::spawn(async move { transcriber.transcribe_stream(stream_rx, "audio.ogg").await });

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

/// Batch dictation mode.
///
/// Records audio to a cache WAV via [`wav_recording_task`] and
/// independently streams encoded OGG to a transcription API.  If the
/// transcription pipeline fails during recording, it is restarted
/// immediately from the beginning of the shared [`AudioBuffer`] so
/// no audio is lost.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn dictate_streaming(
    capture: &mut dyn AudioCapture,
    from_file: bool,
    audio_config: AudioConfig,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    cache_wav_path: &std::path::Path,
    transcriber: Box<dyn BatchTranscriber>,
    shutdown: &CancellationToken,
    player: Option<&SoundPlayer>,
    boop_token: Option<&CancellationToken>,
    overlay: Option<&OverlayHandle>,
    visualizer: Option<&VisualizerHandle>,
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    diarize: bool,
) -> Result<TranscriptionResult, TalkError> {
    // ── WAV recording (independent of transcription) ────────────────
    log::info!("caching audio to: {}", cache_wav_path.display());
    let buffer = Arc::new(AudioBuffer::new());
    let cache_wav_task = tokio::spawn(wav_recording_task(
        audio_rx,
        cache_wav_path.to_path_buf(),
        audio_config.clone(),
        Arc::clone(&buffer),
    ));

    // ── Initial transcription pipeline ──────────────────────────────
    let (mut feeder_handle, mut encode_handle, mut transcribe_handle, mut encode_done_rx) =
        spawn_encode_pipeline(&buffer, audio_config.clone(), transcriber);

    // ── Wait for recording to end ───────────────────────────────────
    let rec_start = std::time::Instant::now();
    if from_file {
        log::info!("transcribing audio file (batch)...");
        tokio::select! {
            _ = shutdown.cancelled() => {
                log::info!("aborting after {:.2}s", rec_start.elapsed().as_secs_f64());
                capture.stop()?;
            }
            _ = &mut encode_done_rx => {
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
                    log::info!(
                        "shutdown signal received after {:.2}s recording — stopping capture",
                        rec_start.elapsed().as_secs_f64()
                    );
                    capture.stop()?;
                    log::info!("capture stopped");
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
                         WAV recording continues",
                        reason
                    );

                    abort_pipeline(
                        &feeder_handle,
                        &encode_handle,
                        &transcribe_handle,
                    );

                    live_retries += 1;
                    if live_retries > MAX_LIVE_RETRIES {
                        let msg = format!(
                            "Transcription failed ({} retries exhausted) — \
                             will retry after recording",
                            MAX_LIVE_RETRIES
                        );
                        log::warn!("{}", msg);
                        if let Some(viz) = visualizer {
                            viz.push_message(&msg);
                        }
                        // Continue recording — will retry from WAV
                        // after recording stops.
                        shutdown.cancelled().await;
                        log::info!(
                            "shutdown after {:.2}s — stopping capture",
                            rec_start.elapsed().as_secs_f64()
                        );
                        capture.stop()?;
                        log::info!("capture stopped");
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
                    match transcription::create_batch_transcriber(
                        config, provider, model, diarize,
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
                            // Fall through — will retry from WAV after
                            // recording stops.
                            shutdown.cancelled().await;
                            log::info!(
                                "shutdown after {:.2}s — stopping capture",
                                rec_start.elapsed().as_secs_f64()
                            );
                            capture.stop()?;
                            break;
                        }
                    }
                }
            }
        }

        // Immediate audible + visual feedback: the user hears the
        // stop sound and sees the "transcribing" badge the instant
        // they toggle, not after the API call finishes.
        if let Some(token) = boop_token {
            token.cancel();
        }
        if let Some(p) = player {
            p.play_stop().await;
        }
        if let Some(viz) = visualizer {
            viz.hide();
        }
        if let Some(o) = overlay {
            o.show(IndicatorKind::Transcribing);
        }
    }

    // ── Wait for WAV recording task (no timeout) ────────────────────
    //
    // The WAV task finishes as soon as the source channel closes (a
    // header-patch + fsync — very fast).  Never abandon it.
    match cache_wav_task.await {
        Ok(Ok(())) => log::debug!("cache WAV task completed"),
        Ok(Err(err)) => log::warn!("cache WAV write error: {}", err),
        Err(err) => log::warn!("cache WAV task panicked: {}", err),
    }

    // ── Wait for transcription result ───────────────────────────────
    const TRANSCRIPTION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    let t0 = std::time::Instant::now();
    log::info!(
        "waiting for transcription (timeout: {:?})",
        TRANSCRIPTION_TIMEOUT
    );

    let result = tokio::select! {
        res = &mut transcribe_handle => {
            log::info!("transcription completed after {:.2}s", t0.elapsed().as_secs_f64());
            match res {
                Ok(Ok(result)) => Ok(result),
                Ok(Err(err)) => Err(err),
                Err(err) => Err(TalkError::Transcription(format!(
                    "transcription task panicked: {}", err
                ))),
            }
        }
        _ = tokio::time::sleep(TRANSCRIPTION_TIMEOUT) => {
            log::warn!(
                "transcription timed out after {:.2}s — aborting",
                t0.elapsed().as_secs_f64()
            );
            transcribe_handle.abort();
            Err(TalkError::Transcription(
                "transcription timed out after 5 s".to_string(),
            ))
        }
    };

    // Clean up: abort remaining pipeline tasks (may already be done).
    feeder_handle.abort();
    encode_handle.abort();

    log::info!(
        "dictate_streaming total elapsed: {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    result
}
