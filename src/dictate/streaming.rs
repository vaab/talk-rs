//! Batch dictation mode.
//!
//! Encodes audio and streams it directly to a single transcription
//! request.  Audio is also tee'd to a cache WAV for the recording cache.

use super::realtime::audio_tee_to_wav;
use crate::audio::indicator::SoundPlayer;
use crate::audio::{AudioCapture, AudioWriter, OggOpusWriter};
use crate::config::AudioConfig;
use crate::error::TalkError;
use crate::overlay::{IndicatorKind, OverlayHandle};
use crate::transcription::{BatchTranscriber, TranscriptionResult};
use crate::visualizer::VisualizerHandle;
use tokio_util::sync::CancellationToken;

/// Batch dictation mode.
///
/// Encodes audio and streams it directly to a single transcription request.
/// Audio is also tee'd to `cache_wav_path` for the recording cache.
///
/// When `from_file` is true, the function waits for the audio source to
/// exhaust naturally (in addition to allowing Ctrl+C for early abort).
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
) -> Result<TranscriptionResult, TalkError> {
    // Tee raw PCM to cache WAV before encoding
    log::info!("caching audio to: {}", cache_wav_path.display());
    let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
    let cache_wav_task = tokio::spawn(audio_tee_to_wav(
        audio_rx,
        fwd_tx,
        cache_wav_path.to_path_buf(),
        audio_config.clone(),
    ));

    // Create channel for streaming encoded audio to transcriber
    let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(25);

    // Oneshot to signal when encoding is done (file source exhausted
    // or capture stopped).  Enables the stop logic to race Ctrl+C
    // against natural completion for file input.
    let (encode_done_tx, encode_done_rx) = tokio::sync::oneshot::channel::<()>();

    // Spawn encode task: PCM → OGG Opus → stream_tx
    let encode_task = tokio::spawn(async move {
        let mut rx = fwd_rx;
        let mut writer = match OggOpusWriter::new(audio_config) {
            Ok(writer) => writer,
            Err(err) => {
                log::error!("error creating audio writer: {}", err);
                return Err(err);
            }
        };

        let header = match writer.header() {
            Ok(bytes) => bytes,
            Err(err) => {
                log::error!("error creating audio header: {}", err);
                return Err(err);
            }
        };
        if stream_tx.send(header).await.is_err() {
            log::warn!("transcription stream closed during header send");
            let _ = encode_done_tx.send(());
            return Ok::<(), TalkError>(());
        }

        while let Some(pcm_chunk) = rx.recv().await {
            let encoded_data = match writer.write_pcm(&pcm_chunk) {
                Ok(data) => data,
                Err(err) => {
                    log::error!("error encoding audio: {}", err);
                    let _ = encode_done_tx.send(());
                    return Err(err);
                }
            };
            if !encoded_data.is_empty() && stream_tx.send(encoded_data).await.is_err() {
                log::warn!("transcription stream closed during audio send");
                break;
            }
        }

        // Finalize writer
        match writer.finalize() {
            Ok(remaining) => {
                if !remaining.is_empty() {
                    let _ = stream_tx.send(remaining).await;
                }
            }
            Err(err) => {
                log::error!("error finalizing audio writer: {}", err);
                let _ = encode_done_tx.send(());
                return Err(err);
            }
        }

        // stream_tx is dropped here, closing the channel
        let _ = encode_done_tx.send(());
        Ok::<(), TalkError>(())
    });

    // Spawn transcription task
    let mut transcribe_task =
        tokio::spawn(async move { transcriber.transcribe_stream(stream_rx, "audio.ogg").await });

    // Wait for recording to end: SIGINT (via shared shutdown token)
    // for live mic, natural completion for file input, or SIGINT to
    // abort file early.  The shutdown token is registered early in
    // dictate() so there is no race window.
    let rec_start = std::time::Instant::now();
    if from_file {
        log::info!("transcribing audio file (batch)...");
        tokio::select! {
            _ = shutdown.cancelled() => {
                log::info!("aborting after {:.2}s", rec_start.elapsed().as_secs_f64());
                capture.stop()?;
            }
            _ = encode_done_rx => {
                log::info!("audio file playback complete after {:.2}s", rec_start.elapsed().as_secs_f64());
            }
        }
    } else {
        log::info!("recording — waiting for shutdown signal");
        shutdown.cancelled().await;
        log::info!(
            "shutdown signal received after {:.2}s recording — stopping capture",
            rec_start.elapsed().as_secs_f64()
        );
        capture.stop()?;
        log::info!("capture stopped");

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

    // Race the transcription result against a 5-second timeout FIRST.
    //
    // We must NOT await encode_task before this: when the API is hung
    // the transcriber stops reading from stream_rx, so the bounded
    // channel (capacity 25) fills up and stream_tx.send() inside
    // encode_task blocks forever.  Awaiting encode_task would therefore
    // deadlock.  Instead we race the transcription against the timeout,
    // then clean up the pipeline afterwards.
    const TRANSCRIPTION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    let t0 = std::time::Instant::now();
    log::info!(
        "waiting for transcription (timeout: {:?})",
        TRANSCRIPTION_TIMEOUT
    );

    let result = tokio::select! {
        res = &mut transcribe_task => {
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
            log::warn!("transcription timed out after {:.2}s — aborting", t0.elapsed().as_secs_f64());
            transcribe_task.abort();
            Err(TalkError::Transcription(
                "transcription timed out after 5 s".to_string(),
            ))
        }
    };

    // Clean up the pipeline.  Aborting the transcribe_task (on timeout)
    // drops stream_rx, which unblocks encode_task's stream_tx.send().
    // Give encode_task a short grace period to finish, then abort it.
    const CLEANUP_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

    match tokio::time::timeout(CLEANUP_GRACE, encode_task).await {
        Ok(Ok(Ok(()))) => log::debug!("encode task completed"),
        Ok(Ok(Err(err))) => log::warn!("encode error: {}", err),
        Ok(Err(err)) => log::warn!("encode task panicked: {}", err),
        Err(_) => log::warn!(
            "encode task did not finish within {:?} — abandoned",
            CLEANUP_GRACE
        ),
    }

    match tokio::time::timeout(CLEANUP_GRACE, cache_wav_task).await {
        Ok(Ok(Ok(()))) => log::debug!("cache WAV task completed"),
        Ok(Ok(Err(err))) => log::warn!("cache WAV write error: {}", err),
        Ok(Err(err)) => log::warn!("cache WAV task panicked: {}", err),
        Err(_) => log::warn!(
            "cache WAV task did not finish within {:?} — abandoned",
            CLEANUP_GRACE
        ),
    }

    log::info!(
        "dictate_streaming total elapsed: {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    result
}
