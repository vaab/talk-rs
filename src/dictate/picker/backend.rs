//! Async transcription backend for the picker.
//!
//! Contains the transcription helpers that run concurrently with the
//! GTK picker window: batch task spawning, realtime WebSocket
//! streaming, and WAV PCM reading.

use super::ui::{PickerCandidate, PickerMessage};
use crate::config::{Config, Provider};
use crate::transcription::{
    RealtimeTranscriber, TranscriptSegment, TranscriptionEvent, TranscriptionMetadata,
};
use std::path::PathBuf;

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
pub(super) async fn run_realtime_transcription(
    transcriber: Box<dyn RealtimeTranscriber>,
    samples: std::sync::Arc<Vec<i16>>,
    tx: std::sync::mpsc::Sender<PickerMessage>,
    provider: Provider,
    model: String,
) {
    // Connect to the realtime WebSocket.
    let (audio_tx, audio_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
    let event_rx = match transcriber.transcribe_realtime(audio_rx).await {
        Ok(rx) => rx,
        Err(e) => {
            log::warn!("realtime {}:{} connect failed: {}", provider, model, e);
            let _ = tx.send(PickerMessage::Candidate(PickerCandidate::error(
                provider,
                model,
                format!("{e}"),
                true,
            )));
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
                let _ = tx.send(PickerMessage::Candidate(PickerCandidate::success(
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
                )));
                return;
            }
            Some(TranscriptionEvent::Error { message }) => {
                let _ = tx.send(PickerMessage::Candidate(PickerCandidate::error(
                    provider, model, message, true,
                )));
                return;
            }
            None => {
                // Channel closed without Done — use what we have.
                let final_text = accumulated.trim().to_string();
                let _ = tx.send(PickerMessage::Candidate(PickerCandidate::success(
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
                )));
                return;
            }
            _ => {
                // Ignore SessionCreated, Language, etc.
            }
        }
    }
}

/// Spawn a single batch transcription task.
///
/// Dead connections are detected by the HTTP client's
/// `tcp_user_timeout` / `read_timeout` / keepalive settings — no
/// artificial outer timeout is needed.  Sends a
/// [`PickerMessage::Candidate`] (success or error) to `tx` when
/// complete.
pub(super) fn spawn_transcription(
    tasks: &mut tokio::task::JoinSet<()>,
    tx: std::sync::mpsc::Sender<PickerMessage>,
    audio: PathBuf,
    provider: Provider,
    model: String,
    config: std::sync::Arc<Config>,
) {
    tasks.spawn(async move {
        let msg = match crate::transcription::transcribe_audio(
            &audio,
            config.as_ref(),
            provider,
            Some(&model),
            false,
        )
        .await
        {
            Ok(res) => {
                let text = res.text.trim().to_string();
                if text.is_empty() {
                    return;
                }
                PickerMessage::Candidate(PickerCandidate::success(
                    provider,
                    model,
                    text,
                    false,
                    res.segments,
                    res.metadata,
                ))
            }
            Err(e) => {
                log::warn!("candidate {}:{} failed: {}", provider, model, e);
                PickerMessage::Candidate(PickerCandidate::error(
                    provider,
                    model,
                    format!("{e}"),
                    false,
                ))
            }
        };
        let _ = tx.send(msg);
    });
}
