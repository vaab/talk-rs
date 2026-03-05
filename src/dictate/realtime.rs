//! Realtime dictation mode via WebSocket.
//!
//! Streams raw PCM audio to the transcription API and receives
//! incremental transcription events.  Returns the accumulated text.
//!
//! Also provides [`AudioBuffer`], [`wav_recording_task`], and
//! [`buffer_feeder`] — the shared infrastructure that decouples WAV
//! recording from transcription so that a transcription failure never
//! truncates the cached recording.

use super::text::flush_sentences;
use crate::audio::indicator::SoundPlayer;
use crate::audio::{AudioCapture, AudioWriter, WavWriter};
use crate::config::{AudioConfig, Config, Provider};
use crate::error::TalkError;
use crate::transcription::{
    self, MistralProviderMetadata, OpenAIProviderMetadata, OpenAIRealtimeMetadata,
    ProviderSpecificMetadata, TranscriptionEvent, TranscriptionMetadata, TranscriptionResult,
};
use crate::x11::visualizer::VisualizerHandle;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

// ── Shared audio buffer ─────────────────────────────────────────────

/// Append-only buffer of PCM audio chunks.
///
/// The WAV recording task pushes every chunk here.  Feeder tasks read
/// from any position and wait for new data.  When the recording stops,
/// [`close`](AudioBuffer::close) is called to unblock waiting feeders.
///
/// This decouples the WAV recording from transcription: the WAV task
/// writes chunks to the file and the buffer unconditionally, while
/// feeder tasks can fail and be restarted from cursor 0 without
/// affecting the recording.
pub(super) struct AudioBuffer {
    chunks: tokio::sync::Mutex<Vec<Vec<i16>>>,
    notify: tokio::sync::Notify,
    closed: AtomicBool,
}

impl AudioBuffer {
    pub(super) fn new() -> Self {
        Self {
            chunks: tokio::sync::Mutex::new(Vec::new()),
            notify: tokio::sync::Notify::new(),
            closed: AtomicBool::new(false),
        }
    }

    /// Append a chunk and wake any waiting feeders.
    pub(super) async fn push(&self, chunk: Vec<i16>) {
        self.chunks.lock().await.push(chunk);
        self.notify.notify_waiters();
    }

    /// Mark the buffer as complete — no more chunks will arrive.
    pub(super) fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    /// Read new chunks starting at `cursor`.
    ///
    /// Returns `(chunks, new_cursor)`.  Blocks until data is available
    /// or the buffer is closed.  Returns an empty vec when closed and
    /// fully drained.
    pub(super) async fn read_from(&self, cursor: usize) -> (Vec<Vec<i16>>, usize) {
        loop {
            {
                let buf = self.chunks.lock().await;
                if buf.len() > cursor {
                    let new_chunks = buf[cursor..].to_vec();
                    return (new_chunks, buf.len());
                }
                if self.closed.load(Ordering::Acquire) {
                    return (Vec::new(), cursor);
                }
            }
            // No new data — wait for a push() or close().
            // Tiny race window (notification between lock release and
            // here) is harmless: the next push() wakes us within ≤20 ms.
            self.notify.notified().await;
        }
    }
}

// ── WAV recording task ──────────────────────────────────────────────

/// Record every PCM chunk to a WAV file and into the shared buffer.
///
/// This task is completely independent of the transcription pipeline.
/// It runs until the `source` channel closes (capture stopped), then
/// patches the WAV header with the final data size and syncs to disk.
pub(super) async fn wav_recording_task(
    mut source: tokio::sync::mpsc::Receiver<Vec<i16>>,
    wav_path: PathBuf,
    audio_config: AudioConfig,
    buffer: Arc<AudioBuffer>,
) -> Result<(), TalkError> {
    let mut writer = WavWriter::new(audio_config);
    let header = writer.header()?;

    let mut file = tokio::fs::File::create(&wav_path)
        .await
        .map_err(TalkError::Io)?;
    file.write_all(&header).await.map_err(TalkError::Io)?;

    let mut total_samples: u64 = 0;

    while let Some(pcm_chunk) = source.recv().await {
        // Write PCM bytes to the WAV file.
        total_samples += pcm_chunk.len() as u64;
        let pcm_bytes = writer.write_pcm(&pcm_chunk)?;
        file.write_all(&pcm_bytes).await.map_err(TalkError::Io)?;

        // Append to the shared buffer (feeders read from here).
        buffer.push(pcm_chunk).await;
    }

    // No more audio — tell feeders there is nothing left to wait for.
    buffer.close();

    // Patch WAV header with actual data size.
    let final_header = writer.finalize()?;
    file.seek(std::io::SeekFrom::Start(0))
        .await
        .map_err(TalkError::Io)?;
    file.write_all(&final_header).await.map_err(TalkError::Io)?;
    file.sync_all().await.map_err(TalkError::Io)?;

    log::info!(
        "cache WAV: {} samples ({:.1}s) saved to {}",
        total_samples,
        total_samples as f64 / 16000.0,
        wav_path.display()
    );

    Ok(())
}

// ── Buffer feeder ───────────────────────────────────────────────────

/// Feed chunks from the shared [`AudioBuffer`] into a channel.
///
/// Starts reading at `cursor` (0 for a fresh pipeline, >0 when
/// resuming a partially-replayed buffer).  Returns when:
///
/// - The buffer is closed and fully drained (normal completion), or
/// - The receiving end of `fwd_tx` is dropped (pipeline failure).
///
/// The caller should monitor the returned `JoinHandle` to detect
/// pipeline failures and spawn a replacement feeder at cursor 0.
pub(super) async fn buffer_feeder(
    buffer: Arc<AudioBuffer>,
    fwd_tx: tokio::sync::mpsc::Sender<Vec<i16>>,
    start_cursor: usize,
) {
    let mut cursor = start_cursor;
    loop {
        let (chunks, new_cursor) = buffer.read_from(cursor).await;
        if chunks.is_empty() {
            // Buffer closed and fully drained.
            break;
        }
        for chunk in chunks {
            if fwd_tx.send(chunk).await.is_err() {
                log::warn!(
                    "transcriber channel closed at chunk {} — feeder stopping",
                    cursor
                );
                return;
            }
            cursor += 1;
        }
        cursor = new_cursor;
    }
    // fwd_tx dropped here → signals end-of-audio downstream.
}

/// Realtime dictation mode via WebSocket.
///
/// Streams raw PCM audio to the transcription API and receives
/// incremental transcription events. Returns the accumulated text.
///
/// Audio is always tee'd to `cache_wav_path` so the recording is
/// cached for later review.
///
/// `player` and `boop_token` are passed so that when recording stops
/// (SIGINT), the stop sound fires immediately — the user hears it the
/// instant they toggle, not after the WebSocket finishes.
///
/// When `visualizer` is provided, the live transcription text is pushed
/// to the text overlay as words arrive.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn dictate_realtime(
    config: Config,
    provider: Provider,
    model: Option<&str>,
    cache_wav_path: &std::path::Path,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    capture: &mut dyn AudioCapture,
    from_file: bool,
    player: Option<&SoundPlayer>,
    boop_token: Option<&CancellationToken>,
    segment_tx: Option<tokio::sync::mpsc::Sender<String>>,
    visualizer: Option<&VisualizerHandle>,
    shutdown: &CancellationToken,
) -> Result<TranscriptionResult, TalkError> {
    // Always record audio to the cache WAV independently of transcription.
    log::info!("caching audio to: {}", cache_wav_path.display());
    let buffer = Arc::new(AudioBuffer::new());
    let wav_task = tokio::spawn(wav_recording_task(
        audio_rx,
        cache_wav_path.to_path_buf(),
        AudioConfig::new(),
        Arc::clone(&buffer),
    ));

    // Create initial transcription pipeline: buffer → feeder → transcriber.
    let transcriber = transcription::create_realtime_transcriber(&config, provider, model)?;
    let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
    let mut feeder_handle = tokio::spawn(buffer_feeder(Arc::clone(&buffer), fwd_tx, 0));
    let mut event_rx = transcriber.transcribe_realtime(fwd_rx).await?;
    let started = std::time::Instant::now();

    if from_file {
        log::info!("transcribing audio file (realtime)...");
    } else {
        log::info!("recording (realtime)... press Ctrl+C to stop");
    }

    let capture_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let capture_stop_clone = capture_stop.clone();

    // Wait for the shared shutdown token (registered early in dictate())
    // instead of a local ctrl_c() handler.  This avoids a race window
    // where SIGINT arrives before this task is spawned.
    let shutdown_clone = shutdown.clone();
    let ctrlc_task = tokio::spawn(async move {
        log::warn!("[DBG] dictate_realtime: waiting on shutdown token");
        shutdown_clone.cancelled().await;
        log::warn!("[DBG] dictate_realtime: shutdown token fired, setting capture_stop");
        capture_stop_clone.store(true, std::sync::atomic::Ordering::Release);
    });

    // Completed sentences/phrases emitted so far.
    let mut segments: Vec<String> = Vec::new();
    // Buffer for the current in-progress phrase (live TextDelta).
    let mut current_line = String::new();
    let mut detected_language: Option<String> = None;
    let mut unknown_event_types: Vec<String> = Vec::new();
    let mut event_counts: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    let mut api_segment_count: usize = 0;
    let mut session_id: Option<String> = None;
    let mut conversation_id: Option<String> = None;
    let mut last_rate_limits: Option<serde_json::Value> = None;
    let mut ws_upgrade_headers: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();

    let bump = |key: &str, counts: &mut std::collections::BTreeMap<String, u64>| {
        let entry = counts.entry(key.to_string()).or_insert(0);
        *entry += 1;
    };

    loop {
        // Check if Ctrl+C was pressed — stop capture to trigger end-of-audio
        if capture_stop.load(std::sync::atomic::Ordering::Acquire) {
            log::info!("stopping recording");

            // Immediate audible + visual feedback: the user hears the
            // stop sound the instant they toggle, not after the
            // transcription WebSocket finishes.
            if let Some(token) = boop_token {
                token.cancel();
            }
            if let Some(p) = player {
                let stop = p.sounds.stop.clone();
                p.play(&stop);
            }

            capture.stop()?;
            // Reset so we don't stop again
            capture_stop.store(false, std::sync::atomic::Ordering::Release);
        }

        tokio::select! {
            event = event_rx.recv() => {
                match event {
                    Some(TranscriptionEvent::TextDelta { text }) => {
                        bump("text_delta", &mut event_counts);
                        current_line.push_str(&text);
                        eprint!("\r{}", current_line);

                        // Push live text to the overlay.
                        if let Some(viz) = visualizer {
                            let mut live = segments.join(" ");
                            if !live.is_empty() && !current_line.is_empty() {
                                live.push(' ');
                            }
                            live.push_str(&current_line);
                            viz.set_text(&live);
                        }

                        // Flush completed sentences from the buffer.
                        // Split on sentence-ending punctuation followed by
                        // whitespace or end-of-string.
                        let prev_count = segments.len();
                        flush_sentences(&mut current_line, &mut segments);
                        if let Some(ref tx) = segment_tx {
                            for seg in &segments[prev_count..] {
                                let _ = tx.send(seg.clone()).await;
                            }
                        }
                    }
                    Some(TranscriptionEvent::SegmentDelta { text, .. }) => {
                        bump("segment_delta", &mut event_counts);
                        api_segment_count += 1;
                        // If the API sends segment events, use them as
                        // authoritative sentence boundaries.
                        let segment_text = text.trim().to_string();
                        if !segment_text.is_empty() {
                            println!("{}", segment_text);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(segment_text.clone()).await;
                            }
                            segments.push(segment_text);
                        }
                        let blank = " ".repeat(current_line.len());
                        eprint!("\r{}\r", blank);
                        current_line.clear();

                        // Update overlay with completed segments.
                        if let Some(viz) = visualizer {
                            viz.set_text(&segments.join(" "));
                        }
                    }
                    Some(TranscriptionEvent::Done) => {
                        bump("done", &mut event_counts);
                        // Flush any trailing text that didn't end with punctuation
                        let trailing = current_line.trim().to_string();
                        if !trailing.is_empty() {
                            println!("{}", trailing);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(trailing.clone()).await;
                            }
                            segments.push(trailing);
                        }
                        eprintln!();
                        break;
                    }
                    Some(TranscriptionEvent::Error { message }) => {
                        bump("error", &mut event_counts);
                        log::warn!("realtime transcription error: {} — attempting reconnect", message);

                        // Try to reconnect with a fresh transcriber and
                        // replay all audio from the beginning.
                        feeder_handle.abort();
                        match transcription::create_realtime_transcriber(&config, provider, model)
                        {
                            Ok(new_transcriber) => {
                                let (new_fwd_tx, new_fwd_rx) =
                                    tokio::sync::mpsc::channel::<Vec<i16>>(100);
                                feeder_handle = tokio::spawn(buffer_feeder(
                                    Arc::clone(&buffer),
                                    new_fwd_tx,
                                    0,
                                ));
                                match new_transcriber.transcribe_realtime(new_fwd_rx).await {
                                    Ok(new_rx) => {
                                        log::info!("realtime transcription reconnected");
                                        event_rx = new_rx;
                                        continue;
                                    }
                                    Err(e) => {
                                        log::warn!("reconnect failed: {}", e);
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                log::warn!("could not create replacement transcriber: {}", e);
                                break;
                            }
                        }
                    }
                    Some(TranscriptionEvent::SessionCreated) => {
                        bump("session_created", &mut event_counts);
                        log::debug!("session created event received");
                    }
                    Some(TranscriptionEvent::SessionInfo { session_id: sid, conversation_id: cid }) => {
                        bump("session_info", &mut event_counts);
                        if sid.is_some() {
                            session_id = sid;
                        }
                        if cid.is_some() {
                            conversation_id = cid;
                        }
                    }
                    Some(TranscriptionEvent::RateLimitsUpdated { raw }) => {
                        bump("rate_limits_updated", &mut event_counts);
                        last_rate_limits = Some(raw);
                    }
                    Some(TranscriptionEvent::TransportMetadata { headers }) => {
                        bump("transport_metadata", &mut event_counts);
                        ws_upgrade_headers.extend(headers);
                    }
                    Some(TranscriptionEvent::Language { language }) => {
                        bump("language", &mut event_counts);
                        log::info!("detected language: {}", language);
                        detected_language = Some(language);
                    }
                    Some(TranscriptionEvent::Unknown { event_type, .. }) => {
                        bump("unknown", &mut event_counts);
                        if let Some(kind) = event_type {
                            bump(&format!("event:{kind}"), &mut event_counts);
                            if !unknown_event_types.contains(&kind) {
                                unknown_event_types.push(kind);
                            }
                        }
                    }
                    None => {
                        // Channel closed without Done event — the
                        // WebSocket may have disconnected.  Try to
                        // reconnect and replay from the beginning.
                        bump("channel_closed", &mut event_counts);
                        log::warn!("realtime event channel closed — attempting reconnect");

                        feeder_handle.abort();
                        match transcription::create_realtime_transcriber(&config, provider, model)
                        {
                            Ok(new_transcriber) => {
                                let (new_fwd_tx, new_fwd_rx) =
                                    tokio::sync::mpsc::channel::<Vec<i16>>(100);
                                feeder_handle = tokio::spawn(buffer_feeder(
                                    Arc::clone(&buffer),
                                    new_fwd_tx,
                                    0,
                                ));
                                match new_transcriber.transcribe_realtime(new_fwd_rx).await {
                                    Ok(new_rx) => {
                                        log::info!("realtime transcription reconnected");
                                        event_rx = new_rx;
                                        continue;
                                    }
                                    Err(e) => {
                                        log::warn!("reconnect failed: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                log::warn!("could not create replacement transcriber: {}", e);
                            }
                        }
                        // Reconnect failed — flush trailing text and exit.
                        let trailing = current_line.trim().to_string();
                        if !trailing.is_empty() {
                            println!("{}", trailing);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(trailing.clone()).await;
                            }
                            segments.push(trailing);
                        }
                        eprintln!();
                        break;
                    }
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                // Periodic check for Ctrl+C flag
            }
        }
    }

    ctrlc_task.abort();
    feeder_handle.abort();

    // Wait for WAV recording task to finish (no timeout — it
    // completes as soon as the source channel closes and the header
    // is patched, which is fast).
    match wav_task.await {
        Ok(Ok(())) => log::debug!("cache WAV saved"),
        Ok(Err(e)) => log::warn!("cache WAV write error: {}", e),
        Err(e) => log::warn!("cache WAV task panicked: {}", e),
    }

    let provider_specific = match provider {
        Provider::OpenAI => Some(ProviderSpecificMetadata::OpenAI(OpenAIProviderMetadata {
            model: model.map(str::to_string),
            usage_raw: None,
            rate_limit_headers: std::collections::BTreeMap::new(),
            unknown_event_types,
            realtime: Some(OpenAIRealtimeMetadata {
                session_id,
                conversation_id,
                event_counts,
                last_rate_limits,
                ws_upgrade_headers: ws_upgrade_headers.clone(),
            }),
        })),
        Provider::Mistral => Some(ProviderSpecificMetadata::Mistral(MistralProviderMetadata {
            model: model.map(str::to_string),
            usage_raw: None,
            unknown_event_types,
        })),
    };

    Ok(TranscriptionResult {
        text: segments.join(" "),
        metadata: TranscriptionMetadata {
            request_latency_ms: None,
            session_elapsed_ms: Some(started.elapsed().as_millis() as u64),
            request_id: ws_upgrade_headers.get("x-request-id").cloned(),
            provider_processing_ms: ws_upgrade_headers
                .get("openai-processing-ms")
                .and_then(|s| s.parse::<u64>().ok()),
            detected_language,
            audio_seconds: None,
            segment_count: Some(if api_segment_count > 0 {
                api_segment_count
            } else {
                segments.len()
            }),
            word_count: None,
            token_usage: None,
            provider_specific,
        },
        diarization: None,
    })
}

// Old `audio_tee_to_wav` removed — replaced by `wav_recording_task`
// + `buffer_feeder` above.  The WAV recording is now fully decoupled
// from the transcription pipeline.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AudioConfig;

    // ── AudioBuffer tests ───────────────────────────────────────────

    #[tokio::test]
    async fn audio_buffer_push_then_read_returns_chunks() {
        let buf = AudioBuffer::new();
        buf.push(vec![1, 2, 3]).await;
        buf.push(vec![4, 5, 6]).await;

        let (chunks, cursor) = buf.read_from(0).await;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[1], vec![4, 5, 6]);
        assert_eq!(cursor, 2);
    }

    #[tokio::test]
    async fn audio_buffer_read_from_cursor_skips_earlier() {
        let buf = AudioBuffer::new();
        buf.push(vec![10]).await;
        buf.push(vec![20]).await;
        buf.push(vec![30]).await;

        let (chunks, cursor) = buf.read_from(2).await;
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], vec![30]);
        assert_eq!(cursor, 3);
    }

    #[tokio::test]
    async fn audio_buffer_close_unblocks_empty_read() {
        let buf = Arc::new(AudioBuffer::new());
        buf.push(vec![1]).await;

        // Drain all data.
        let (_chunks, cursor) = buf.read_from(0).await;
        assert_eq!(cursor, 1);

        // Close from another task.
        let buf2 = Arc::clone(&buf);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            buf2.close();
        });

        // read_from should return empty once closed.
        let (chunks, cursor) = buf.read_from(1).await;
        assert!(chunks.is_empty());
        assert_eq!(cursor, 1);
    }

    #[tokio::test]
    async fn audio_buffer_push_after_close_still_accessible() {
        // close() only sets a flag — pre-existing data is readable.
        let buf = AudioBuffer::new();
        buf.push(vec![42]).await;
        buf.close();

        let (chunks, _) = buf.read_from(0).await;
        assert_eq!(chunks, vec![vec![42]]);
    }

    // ── buffer_feeder tests ─────────────────────────────────────────

    #[tokio::test]
    async fn buffer_feeder_replays_from_cursor_zero() {
        let buf = Arc::new(AudioBuffer::new());
        buf.push(vec![1, 2]).await;
        buf.push(vec![3, 4]).await;
        buf.close();

        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        buffer_feeder(buf, tx, 0).await;

        let c1 = rx.recv().await;
        let c2 = rx.recv().await;
        let c3 = rx.recv().await;
        assert_eq!(c1, Some(vec![1, 2]));
        assert_eq!(c2, Some(vec![3, 4]));
        assert!(c3.is_none()); // channel closed
    }

    #[tokio::test]
    async fn buffer_feeder_stops_when_receiver_dropped() {
        let buf = Arc::new(AudioBuffer::new());
        buf.push(vec![10]).await;
        buf.push(vec![20]).await;

        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx); // drop receiver immediately

        // feeder should exit quickly without hanging.
        let handle = tokio::spawn(buffer_feeder(buf, tx, 0));
        tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .expect("feeder should finish promptly")
            .expect("feeder should not panic");
    }

    #[tokio::test]
    async fn buffer_feeder_starts_from_nonzero_cursor() {
        let buf = Arc::new(AudioBuffer::new());
        buf.push(vec![100]).await;
        buf.push(vec![200]).await;
        buf.push(vec![300]).await;
        buf.close();

        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        buffer_feeder(buf, tx, 2).await;

        let c1 = rx.recv().await;
        let c2 = rx.recv().await;
        assert_eq!(c1, Some(vec![300]));
        assert!(c2.is_none());
    }

    // ── wav_recording_task tests ────────────────────────────────────

    #[tokio::test]
    async fn wav_recording_task_writes_complete_wav() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let wav_path = dir.path().join("test.wav");
        let audio_config = AudioConfig::new();
        let buffer = Arc::new(AudioBuffer::new());

        let (tx, rx) = tokio::sync::mpsc::channel(10);

        let buf_clone = Arc::clone(&buffer);
        let path_clone = wav_path.clone();
        let handle = tokio::spawn(wav_recording_task(rx, path_clone, audio_config, buf_clone));

        // Send 5 chunks of 320 samples (20ms at 16kHz mono).
        for i in 0..5u16 {
            let chunk: Vec<i16> = (0..320)
                .map(|s| (s as i16).wrapping_mul(i as i16))
                .collect();
            tx.send(chunk).await.expect("send chunk");
        }
        drop(tx); // close channel → task finishes

        handle.await.expect("task join").expect("wav write");

        // Verify WAV file.
        let data = std::fs::read(&wav_path).expect("read wav");
        assert_eq!(&data[0..4], b"RIFF");
        assert_eq!(&data[8..12], b"WAVE");

        // Total PCM data: 5 chunks × 320 samples × 2 bytes = 3200 bytes.
        // WAV header is 44 bytes.
        assert_eq!(data.len(), 44 + 5 * 320 * 2);
    }

    #[tokio::test]
    async fn wav_recording_task_populates_buffer() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let wav_path = dir.path().join("test.wav");
        let audio_config = AudioConfig::new();
        let buffer = Arc::new(AudioBuffer::new());

        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let buf_clone = Arc::clone(&buffer);
        let handle = tokio::spawn(wav_recording_task(rx, wav_path, audio_config, buf_clone));

        tx.send(vec![1, 2, 3]).await.expect("send");
        tx.send(vec![4, 5, 6]).await.expect("send");
        drop(tx);

        handle.await.expect("join").expect("wav");

        // Buffer should have both chunks and be closed.
        let (chunks, _) = buffer.read_from(0).await;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[1], vec![4, 5, 6]);

        // Confirm closed: read_from at end returns empty.
        let (empty, _) = buffer.read_from(2).await;
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn wav_recording_independent_of_feeder_failure() {
        // Verify that the WAV file is complete even when the feeder
        // (downstream transcription pipeline) fails.
        let dir = tempfile::tempdir().expect("create temp dir");
        let wav_path = dir.path().join("test.wav");
        let audio_config = AudioConfig::new();
        let buffer = Arc::new(AudioBuffer::new());

        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let buf_clone = Arc::clone(&buffer);
        let path_clone = wav_path.clone();
        let wav_handle = tokio::spawn(wav_recording_task(rx, path_clone, audio_config, buf_clone));

        // Start a feeder that will be killed.
        let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel(10);
        let feeder = tokio::spawn(buffer_feeder(Arc::clone(&buffer), fwd_tx, 0));

        // Send some audio.
        tx.send(vec![10; 320]).await.expect("send");
        tx.send(vec![20; 320]).await.expect("send");

        // Kill the feeder by dropping the receiver.
        drop(fwd_rx);
        // Wait for feeder to notice and exit.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), feeder).await;

        // Send more audio AFTER the feeder died — WAV must still record.
        tx.send(vec![30; 320])
            .await
            .expect("send after feeder death");
        drop(tx);

        wav_handle.await.expect("join").expect("wav");

        // All 3 chunks must be in the WAV.
        let data = std::fs::read(&wav_path).expect("read wav");
        assert_eq!(data.len(), 44 + 3 * 320 * 2);

        // All 3 chunks must be in the buffer.
        let (chunks, _) = buffer.read_from(0).await;
        assert_eq!(chunks.len(), 3);
    }
}
