//! Async transcription backend for the picker.
//!
//! Contains the transcription helpers that run concurrently with the
//! GTK picker window: batch task spawning, realtime WebSocket
//! streaming, and WAV PCM reading.

use super::ui::{PickerCandidate, PickerMessage};
use crate::config::Provider;
use crate::error::TalkError;
use crate::transcription::{BatchTranscriber, RealtimeTranscriber, TranscriptionEvent};
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
            Some(TranscriptionEvent::SegmentDelta { text, .. }) => {
                if !text.is_empty() {
                    if !accumulated.is_empty() {
                        accumulated.push(' ');
                    }
                    accumulated.push_str(text.trim());
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
                    provider, model, final_text, true,
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
                    provider, model, final_text, true,
                )));
                return;
            }
            _ => {
                // Ignore SessionCreated, Language, etc.
            }
        }
    }
}

/// Spawn a single batch transcription task with an 8-second timeout.
///
/// Sends a [`PickerMessage::Candidate`] (success or error) to `tx`
/// when complete.
pub(super) fn spawn_transcription(
    tasks: &mut tokio::task::JoinSet<()>,
    tx: std::sync::mpsc::Sender<PickerMessage>,
    audio: PathBuf,
    provider: Provider,
    model: String,
    transcriber: Box<dyn BatchTranscriber>,
) {
    tasks.spawn(async move {
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(8),
            transcriber.transcribe_file(&audio),
        )
        .await;
        let msg = match result {
            Ok(Ok(res)) => {
                let text = res.text.trim().to_string();
                if text.is_empty() {
                    return;
                }
                PickerMessage::Candidate(PickerCandidate::success(provider, model, text, false))
            }
            Ok(Err(e)) => {
                log::warn!("candidate {}:{} failed: {}", provider, model, e);
                PickerMessage::Candidate(PickerCandidate::error(
                    provider,
                    model,
                    format!("{e}"),
                    false,
                ))
            }
            Err(_) => {
                log::warn!("candidate {}:{} timed out after 8s", provider, model);
                PickerMessage::Candidate(PickerCandidate::error(
                    provider,
                    model,
                    "timed out".into(),
                    false,
                ))
            }
        };
        let _ = tx.send(msg);
    });
}

/// Read all PCM i16 samples from a 16-bit WAV file.
///
/// Skips directly to the `data` chunk and reads every sample as
/// little-endian i16.  Returns an error if the file is not a valid
/// WAV or does not contain 16-bit PCM.
pub(super) fn read_wav_pcm_samples(path: &std::path::Path) -> Result<Vec<i16>, TalkError> {
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::{Read, Seek, SeekFrom};

    let err = |msg: &str| TalkError::Audio(format!("{}: {msg}", path.display()));

    let mut file = std::fs::File::open(path)
        .map_err(|e| TalkError::Audio(format!("failed to open WAV {}: {e}", path.display())))?;

    // Validate RIFF/WAVE header.
    let mut tag = [0u8; 4];
    file.read_exact(&mut tag)
        .map_err(|_| err("truncated RIFF"))?;
    if &tag != b"RIFF" {
        return Err(err("not a WAV file (missing RIFF)"));
    }
    file.read_u32::<LittleEndian>()
        .map_err(|_| err("truncated RIFF"))?; // file size
    file.read_exact(&mut tag)
        .map_err(|_| err("truncated WAVE"))?;
    if &tag != b"WAVE" {
        return Err(err("not a WAV file (missing WAVE)"));
    }

    // Walk chunks until we find "data".
    let data_size: u32 = loop {
        if file.read_exact(&mut tag).is_err() {
            return Err(err("data chunk not found"));
        }
        let chunk_size = file
            .read_u32::<LittleEndian>()
            .map_err(|_| err("truncated chunk header"))?;
        if &tag == b"data" {
            break chunk_size;
        }
        // Skip unknown chunk (pad to even).
        let skip = if chunk_size % 2 == 0 {
            chunk_size as i64
        } else {
            chunk_size as i64 + 1
        };
        file.seek(SeekFrom::Current(skip))
            .map_err(|_| err("seek past chunk failed"))?;
    };

    let num_samples = data_size as usize / 2; // 16-bit = 2 bytes per sample
    let mut samples = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        match file.read_i16::<LittleEndian>() {
            Ok(s) => samples.push(s),
            Err(_) => break,
        }
    }
    Ok(samples)
}
