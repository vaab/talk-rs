//! Realtime dictation mode via WebSocket.
//!
//! Streams raw PCM audio to the transcription API and receives
//! incremental transcription events.  Returns the accumulated text.
//!
//! Also provides [`AudioBuffer`], [`ogg_recording_task`], and
//! [`buffer_feeder`] — the shared infrastructure that decouples OGG
//! recording from transcription so that a transcription failure never
//! truncates the cached recording.

use super::text::flush_sentences;
use crate::audio::bt_profile;
use crate::audio::recording_feedback::{RecordingBadgeTeardown, RecordingFeedback};
use crate::audio::{AudioCapture, AudioWriter, OggOpusWriter};
use crate::config::{AudioConfig, Config, Provider};
use crate::error::TalkError;
use crate::transcription::realtime::{join_segment, join_segments};
use crate::transcription::{
    self, MistralProviderMetadata, OpenAIProviderMetadata, OpenAIRealtimeMetadata,
    OrderedItemTranscript, ProviderSpecificMetadata, TranscriptSegment, TranscriptionEvent,
    TranscriptionMetadata, TranscriptionResult,
};
use crate::x11::visualizer::VisualizerHandle;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

// ── Shared audio buffer ─────────────────────────────────────────────

/// Append-only buffer of PCM audio chunks.
///
/// The OGG recording task pushes every chunk here.  Feeder tasks read
/// from any position and wait for new data.  When the recording stops,
/// [`close`](AudioBuffer::close) is called to unblock waiting feeders.
///
/// This decouples the OGG recording from transcription: the OGG task
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

    /// Return `true` if no audio chunks were pushed to this buffer.
    pub(super) async fn is_empty(&self) -> bool {
        self.chunks.lock().await.is_empty()
    }

    /// Read new chunks starting at `cursor`.
    ///
    /// Returns `(chunks, new_cursor)`.  Blocks until data is available
    /// or the buffer is closed.  Returns an empty vec when closed and
    /// fully drained.
    pub(super) async fn read_from(&self, cursor: usize) -> (Vec<Vec<i16>>, usize) {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
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
            // Register before checking state so close cannot race this wait.
            notified.await;
        }
    }
}

// ── OGG recording task ──────────────────────────────────────────────

/// Record every PCM chunk to an OGG file and into the shared buffer.
///
/// This task is completely independent of the transcription pipeline.
/// It runs until the `source` channel closes (capture stopped), then
/// appends any trailing OGG bytes and syncs to disk.
pub(super) async fn ogg_recording_task(
    mut source: tokio::sync::mpsc::Receiver<Vec<i16>>,
    ogg_path: PathBuf,
    audio_config: AudioConfig,
    buffer: Arc<AudioBuffer>,
) -> Result<(), TalkError> {
    struct CloseBuffer(Arc<AudioBuffer>);
    impl Drop for CloseBuffer {
        fn drop(&mut self) {
            self.0.close();
        }
    }
    let _close_buffer = CloseBuffer(buffer.clone());
    let (mut writer, header) = tokio::task::spawn_blocking(move || {
        let mut writer = OggOpusWriter::new(audio_config)?;
        let header = writer.header()?;
        Ok::<_, TalkError>((writer, header))
    })
    .await
    .map_err(|error| TalkError::Audio(format!("OGG encoder task failed: {error}")))??;

    let mut file = tokio::fs::File::create(&ogg_path).await.map_err(|error| {
        TalkError::Audio(format!("failed to create {}: {error}", ogg_path.display()))
    })?;
    file.write_all(&header).await.map_err(TalkError::Io)?;

    let mut total_samples: u64 = 0;

    while let Some(pcm_chunk) = source.recv().await {
        // Write encoded bytes to the OGG file.
        total_samples += pcm_chunk.len() as u64;
        let (next_writer, pcm_chunk, encoded_bytes) = tokio::task::spawn_blocking(move || {
            let encoded_bytes = writer.write_pcm(&pcm_chunk)?;
            Ok::<_, TalkError>((writer, pcm_chunk, encoded_bytes))
        })
        .await
        .map_err(|error| TalkError::Audio(format!("OGG encoder task failed: {error}")))??;
        writer = next_writer;
        if !encoded_bytes.is_empty() {
            file.write_all(&encoded_bytes)
                .await
                .map_err(TalkError::Io)?;
        }

        // Append to the shared buffer (feeders read from here).
        buffer.push(pcm_chunk).await;
    }

    // No more audio — tell feeders there is nothing left to wait for.
    buffer.close();

    let trailing_bytes = tokio::task::spawn_blocking(move || writer.finalize())
        .await
        .map_err(|error| TalkError::Audio(format!("OGG encoder task failed: {error}")))??;
    if !trailing_bytes.is_empty() {
        file.write_all(&trailing_bytes)
            .await
            .map_err(TalkError::Io)?;
    }
    file.sync_all().await.map_err(TalkError::Io)?;

    log::info!(
        "cache OGG: {} samples ({:.1}s) saved to {}",
        total_samples,
        total_samples as f64 / 16000.0,
        ogg_path.display()
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

async fn finish_live_recording(
    capture: &mut dyn AudioCapture,
    feedback: &mut RecordingFeedback,
    bt_guard: &mut bt_profile::HeadsetGuard,
    ogg_task: tokio::task::JoinHandle<Result<(), TalkError>>,
    capture_stopped: bool,
) -> Result<(), TalkError> {
    let stop_result = if capture_stopped {
        Ok(())
    } else {
        feedback.teardown_recording(RecordingBadgeTeardown::KeepVisible);
        feedback.play_stop_now();
        let result = capture.stop();
        bt_guard.restore_now_async();
        result
    };
    match ogg_task.await {
        Ok(Ok(())) => log::debug!("cache OGG saved"),
        Ok(Err(e)) => log::warn!("cache OGG write error: {}", e),
        Err(e) => log::warn!("cache OGG task panicked: {}", e),
    }
    stop_result
}

#[derive(Debug, Default, PartialEq, Eq)]
struct NormalTranscriptUpdate {
    live_text: String,
    segments_to_send: Vec<String>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct FinishedNormalTranscript {
    text: String,
    segments_to_send: Vec<String>,
}

#[derive(Debug, Default)]
struct NormalTranscriptAccumulator {
    generic_segments: Vec<String>,
    current_line: String,
    item_text: OrderedItemTranscript,
    item_segments: Vec<String>,
    replay_prefix: VecDeque<String>,
}

impl NormalTranscriptAccumulator {
    fn apply(&mut self, event: TranscriptionEvent) -> NormalTranscriptUpdate {
        let segments_to_send = match event {
            TranscriptionEvent::TextDelta { text } => {
                self.current_line.push_str(&text);
                let previous_len = self.generic_segments.len();
                flush_sentences(&mut self.current_line, &mut self.generic_segments);
                self.generic_segments[previous_len..].to_vec()
            }
            TranscriptionEvent::SegmentDelta { text, .. } => {
                let segment = text;
                self.current_line.clear();
                if segment.trim().is_empty() {
                    Vec::new()
                } else {
                    self.generic_segments.push(segment.clone());
                    vec![segment]
                }
            }
            TranscriptionEvent::ItemCreated {
                item_id,
                previous_item_id,
            } => {
                self.item_text
                    .item_created(&item_id, previous_item_id.as_deref());
                let drained = self.item_text.drain_completed_prefix();
                self.accept_item_drain(drained)
            }
            TranscriptionEvent::ItemTextDelta {
                item_id,
                content_index,
                text,
            } => {
                self.item_text.append_delta(&item_id, content_index, &text);
                Vec::new()
            }
            TranscriptionEvent::ItemTextCompleted {
                item_id,
                content_index,
                transcript,
            } => {
                self.item_text
                    .complete(&item_id, content_index, &transcript);
                let drained = self.item_text.drain_completed_prefix();
                self.accept_item_drain(drained)
            }
            _ => Vec::new(),
        };

        NormalTranscriptUpdate {
            live_text: self.live_text(),
            segments_to_send,
        }
    }

    fn finish(&mut self) -> FinishedNormalTranscript {
        if !self.item_text.is_empty() || !self.item_segments.is_empty() {
            let drained = self.item_text.drain_terminal();
            let segments_to_send = self.accept_item_drain(drained);
            return FinishedNormalTranscript {
                text: join_segments(self.item_segments.iter().map(String::as_str)),
                segments_to_send,
            };
        }

        let trailing = self.current_line.trim().to_string();
        let segments_to_send = if trailing.is_empty() {
            Vec::new()
        } else {
            self.generic_segments.push(trailing.clone());
            vec![trailing]
        };
        self.current_line.clear();
        FinishedNormalTranscript {
            text: join_segments(self.generic_segments.iter().map(String::as_str)),
            segments_to_send,
        }
    }

    fn live_text(&self) -> String {
        if !self.item_text.is_empty() {
            return self.item_text.snapshot();
        }
        if !self.item_segments.is_empty() {
            return join_segments(self.item_segments.iter().map(String::as_str));
        }
        let mut live = join_segments(self.generic_segments.iter().map(String::as_str));
        live.push_str(&join_segment(live.chars().last(), &self.current_line));
        live
    }

    fn text(&self) -> String {
        if self.item_segments.is_empty() {
            join_segments(self.generic_segments.iter().map(String::as_str))
        } else {
            join_segments(self.item_segments.iter().map(String::as_str))
        }
    }

    fn segment_count(&self) -> usize {
        if self.item_segments.is_empty() {
            self.generic_segments.len()
        } else {
            self.item_segments.len()
        }
    }

    fn reset_item_generation_for_replay(&mut self) {
        self.item_text.reset_generation();
        self.replay_prefix = self.item_segments.clone().into();
    }

    fn accept_item_drain(&mut self, drained: Vec<String>) -> Vec<String> {
        let mut segments_to_send = Vec::new();
        for segment in drained {
            if !self.replay_prefix.is_empty() {
                let index = self.item_segments.len() - self.replay_prefix.len();
                self.replay_prefix.pop_front();
                if self.item_segments[index] != segment {
                    // The cache keeps the authoritative replay correction. The
                    // already-delivered paste is left untouched: re-pasting a
                    // correction here would duplicate text in the target app.
                    self.item_segments[index] = segment;
                }
                continue;
            }
            self.item_segments.push(segment.clone());
            segments_to_send.push(segment);
        }
        segments_to_send
    }
}

/// Realtime dictation mode via WebSocket.
///
/// Streams raw PCM audio to the transcription API and receives
/// incremental transcription events. Returns the accumulated text.
///
/// Audio is always tee'd to `cache_ogg_path` so the recording is
/// cached for later review.
///
/// `feedback` is passed so recording-phase feedback tears down and the stop
/// sound starts immediately on SIGINT rather than after the WebSocket closes.
///
/// When `visualizer` is provided, the live transcription text is pushed
/// to the text overlay as words arrive.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn dictate_realtime(
    config: Config,
    provider: Provider,
    model: Option<&str>,
    cache_ogg_path: &std::path::Path,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    capture: &mut dyn AudioCapture,
    from_file: bool,
    feedback: &mut RecordingFeedback,
    segment_tx: Option<tokio::sync::mpsc::Sender<String>>,
    visualizer: Option<&VisualizerHandle>,
    shutdown: &CancellationToken,
    mut bt_guard: bt_profile::HeadsetGuard,
    chain: Option<&crate::config::ResolvedChain>,
    outage_path: Option<&std::path::Path>,
) -> Result<(TranscriptionResult, Provider, String), TalkError> {
    // Always record audio to the cache OGG independently of transcription.
    log::info!("caching audio to: {}", cache_ogg_path.display());
    let buffer = Arc::new(AudioBuffer::new());
    let ogg_task = tokio::spawn(ogg_recording_task(
        audio_rx,
        cache_ogg_path.to_path_buf(),
        AudioConfig::new(),
        Arc::clone(&buffer),
    ));

    let outage = outage_path.map(std::path::Path::to_path_buf);
    let choices: Vec<_> = match (chain, &outage) {
        (Some(chain), Some(path)) => chain
            .available_entries(&config, path.clone())
            .into_iter()
            .map(|entry| {
                (
                    entry.provider,
                    Some(entry.model.clone()),
                    Some(entry.retry_schedule()),
                )
            })
            .collect(),
        _ => vec![(provider, model.map(str::to_string), None)],
    };
    let mut attempts = Vec::new();
    let mut selected = None;
    let mut last_error = None;
    for (index, (candidate, candidate_model, schedule)) in choices.iter().enumerate() {
        let mut transcriber = match transcription::create_realtime_transcriber(
            &config,
            *candidate,
            candidate_model.as_deref(),
        ) {
            Ok(transcriber) => transcriber,
            Err(error) => {
                last_error = Some(error);
                break;
            }
        };
        if let Some(schedule) = schedule {
            transcriber.set_retry_schedule(schedule.clone());
        }
        let connected = match transcriber.validate().await {
            Ok(()) => {
                let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
                let feeder = tokio::spawn(buffer_feeder(Arc::clone(&buffer), fwd_tx, 0));
                match transcriber.transcribe_realtime(fwd_rx).await {
                    Ok(events) => Ok((events, feeder)),
                    Err(error) => {
                        feeder.abort();
                        Err(error)
                    }
                }
            }
            Err(error) => Err(error),
        };
        match connected {
            Ok((events, feeder)) => {
                if let (Some(chain), Some(path)) = (chain, &outage) {
                    chain.clear_outage(*candidate, path.clone());
                    attempts.push(crate::transcription::chain::Attempt {
                        provider: candidate.to_string(),
                        model: candidate_model.clone().unwrap_or_default(),
                        outcome: "success".into(),
                    });
                    if let (Some(viz), Some(notice)) = (
                        visualizer,
                        candidate_model
                            .as_deref()
                            .and_then(|model| chain.fallback_notice(*candidate, model)),
                    ) {
                        viz.pin_message(&notice);
                    }
                }
                selected = Some((events, feeder, *candidate, candidate_model.clone()));
                break;
            }
            Err(error) if chain.is_some() && error.is_fallback_worthy() => {
                if let (Some(chain), Some(path)) = (chain, &outage) {
                    chain.record_busy(*candidate, error.retry_after(), path.clone());
                }
                attempts.push(crate::transcription::chain::Attempt {
                    provider: candidate.to_string(),
                    model: candidate_model.clone().unwrap_or_default(),
                    outcome: "busy".into(),
                });
                if let Some((_, Some(next), _)) = choices.get(index + 1) {
                    let message = format!(
                        "{} busy → {}",
                        candidate_model.as_deref().unwrap_or("model"),
                        next
                    );
                    log::info!("{message}");
                    if let Some(viz) = visualizer {
                        viz.push_message(&message);
                    }
                }
                last_error = Some(error);
            }
            Err(error) => {
                last_error = Some(error);
                break;
            }
        }
    }
    let Some((mut event_rx, mut feeder_handle, selected_provider, selected_model)) = selected
    else {
        finish_live_recording(capture, feedback, &mut bt_guard, ogg_task, false).await?;
        return Err(last_error.unwrap_or_else(|| {
            TalkError::Config("realtime chain has no connection candidate".into())
        }));
    };
    let started = std::time::Instant::now();

    if from_file {
        log::info!("transcribing audio file (realtime)...");
    } else {
        log::info!("recording (realtime)... press Ctrl+C to stop");
    }

    let capture_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut capture_stopped = false;
    let capture_stop_clone = capture_stop.clone();

    // Wait for the shared shutdown token (registered early in dictate())
    // instead of a local ctrl_c() handler.  This avoids a race window
    // where SIGINT arrives before this task is spawned.
    let shutdown_clone = shutdown.clone();
    let ctrlc_task = tokio::spawn(async move {
        log::debug!("dictate_realtime: waiting on shutdown token");
        shutdown_clone.cancelled().await;
        log::debug!("dictate_realtime: shutdown token fired, setting capture_stop");
        capture_stop_clone.store(true, std::sync::atomic::Ordering::Release);
    });

    let mut transcript = NormalTranscriptAccumulator::default();
    let mut timed_segments: Vec<TranscriptSegment> = Vec::new();
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
            feedback.teardown_recording(RecordingBadgeTeardown::KeepVisible);
            feedback.play_stop_now();

            capture.stop()?;
            capture_stopped = true;
            // Restore the Bluetooth headset to its high-quality
            // profile (typically A2DP) the instant the microphone
            // capture stops, in parallel with the WebSocket finishing
            // and the paste pipeline.  Drop on the empty guard at
            // function exit is then a no-op.
            bt_guard.restore_now_async();
            // Reset so we don't stop again
            capture_stop.store(false, std::sync::atomic::Ordering::Release);
        }

        tokio::select! {
            event = event_rx.recv() => {
                match event {
                    Some(TranscriptionEvent::TextDelta { text }) => {
                        bump("text_delta", &mut event_counts);
                        let update = transcript.apply(TranscriptionEvent::TextDelta { text });
                        eprint!("\r{}", update.live_text);
                        if let Some(viz) = visualizer {
                            viz.set_text(&update.live_text);
                        }
                        if let Some(ref tx) = segment_tx {
                            for segment in update.segments_to_send {
                                let _ = tx.send(segment).await;
                            }
                        }
                    }
                    Some(TranscriptionEvent::SegmentDelta { text, start, end }) => {
                        bump("segment_delta", &mut event_counts);
                        api_segment_count += 1;
                        // If the API sends segment events, use them as
                        // authoritative sentence boundaries.
                        let segment_text = text.trim().to_string();
                        if !segment_text.is_empty() {
                            if let (Some(start), Some(end)) = (start, end) {
                                timed_segments.push(TranscriptSegment {
                                    start,
                                    end,
                                    text: segment_text.clone(),
                                });
                            }
                        }
                        let update = transcript.apply(TranscriptionEvent::SegmentDelta {
                            text,
                            start,
                            end,
                        });
                        for segment in update.segments_to_send {
                            println!("{}", segment);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(segment).await;
                            }
                        }
                        if let Some(viz) = visualizer {
                            viz.set_text(&update.live_text);
                        }
                    }
                    Some(event @ TranscriptionEvent::ItemCreated { .. }) => {
                        bump("item_created", &mut event_counts);
                        let update = transcript.apply(event);
                        for segment in update.segments_to_send {
                            println!("{}", segment);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(segment).await;
                            }
                        }
                    }
                    Some(event @ TranscriptionEvent::ItemTextDelta { .. }) => {
                        bump("item_text_delta", &mut event_counts);
                        let update = transcript.apply(event);
                        eprint!("\r{}", update.live_text);
                        if let Some(viz) = visualizer {
                            viz.set_text(&update.live_text);
                        }
                    }
                    Some(event @ TranscriptionEvent::ItemTextCompleted { .. }) => {
                        bump("item_text_completed", &mut event_counts);
                        api_segment_count += 1;
                        let update = transcript.apply(event);
                        eprint!("\r{}", update.live_text);
                        if let Some(viz) = visualizer {
                            viz.set_text(&update.live_text);
                        }
                        for segment in update.segments_to_send {
                            println!("{}", segment);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(segment).await;
                            }
                        }
                    }
                    Some(TranscriptionEvent::Done) => {
                        bump("done", &mut event_counts);
                        break;
                    }
                    Some(TranscriptionEvent::Error { message }) => {
                        bump("error", &mut event_counts);
                        let msg = format!(
                            "Transcription error: {} — reconnecting",
                            message
                        );
                        log::warn!("{}", msg);
                        if let Some(viz) = visualizer {
                            viz.push_message(&msg);
                        }

                        // Try to reconnect with a fresh transcriber and
                        // replay all audio from the beginning.
                        feeder_handle.abort();
                        match transcription::create_realtime_transcriber(&config, selected_provider, selected_model.as_deref().or(model))
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
                                        transcript.reset_item_generation_for_replay();
                                        event_rx = new_rx;
                                        continue;
                                    }
                                    Err(e) => {
                                        let msg = format!(
                                            "Reconnect failed: {}",
                                            e
                                        );
                                        log::warn!("{}", msg);
                                        if let Some(viz) = visualizer {
                                            viz.push_message(&msg);
                                        }
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                let msg = format!(
                                    "Reconnect failed: {}",
                                    e
                                );
                                log::warn!("{}", msg);
                                if let Some(viz) = visualizer {
                                    viz.push_message(&msg);
                                }
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
                        if let Some(viz) = visualizer {
                            viz.push_message("Connection lost — reconnecting");
                        }

                        feeder_handle.abort();
                        match transcription::create_realtime_transcriber(&config, selected_provider, selected_model.as_deref().or(model))
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
                                        transcript.reset_item_generation_for_replay();
                                        event_rx = new_rx;
                                        continue;
                                    }
                                    Err(e) => {
                                        let msg = format!("Reconnect failed: {}", e);
                                        log::warn!("{}", msg);
                                        if let Some(viz) = visualizer {
                                            viz.push_message(&msg);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                let msg = format!("Reconnect failed: {}", e);
                                log::warn!("{}", msg);
                                if let Some(viz) = visualizer {
                                    viz.push_message(&msg);
                                }
                            }
                        }
                        break;
                    }
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                // Periodic check for Ctrl+C flag
            }
        }
    }

    // Every terminal exit, including reconnect failure, drains exactly the
    // remaining authoritative/provisional text once.
    let finished = transcript.finish();
    for segment in finished.segments_to_send {
        println!("{}", segment);
        if let Some(ref tx) = segment_tx {
            let _ = tx.send(segment).await;
        }
    }
    eprintln!();

    ctrlc_task.abort();
    feeder_handle.abort();

    finish_live_recording(capture, feedback, &mut bt_guard, ogg_task, capture_stopped).await?;

    let provider_specific = match selected_provider {
        Provider::OpenAI => Some(ProviderSpecificMetadata::OpenAI(OpenAIProviderMetadata {
            model: selected_model.clone().or_else(|| model.map(str::to_string)),
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
            model: selected_model.clone().or_else(|| model.map(str::to_string)),
            usage_raw: None,
            unknown_event_types,
        })),
        // Parakeet has no realtime mode; realtime code paths never
        // dispatch here for Parakeet.  Unreachable in practice, but
        // the match must be total.
        Provider::Parakeet => None,
    };

    Ok((
        TranscriptionResult {
            text: transcript.text(),
            metadata: TranscriptionMetadata {
                attempts,
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
                    transcript.segment_count()
                }),
                word_count: None,
                token_usage: None,
                provider_specific,
            },
            diarization: None,
            segments: if timed_segments.is_empty() {
                None
            } else {
                Some(timed_segments)
            },
        },
        selected_provider,
        selected_model.unwrap_or_else(|| model.unwrap_or_default().to_string()),
    ))
}

// Old `audio_tee_to_wav` removed — replaced by `ogg_recording_task`
// + `buffer_feeder` above.  The OGG recording is now fully decoupled
// from the transcription pipeline.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AudioConfig;

    #[derive(Default)]
    struct HoldingCapture {
        sender: Option<tokio::sync::mpsc::Sender<Vec<i16>>>,
        stopped: bool,
    }

    impl AudioCapture for HoldingCapture {
        fn start(&mut self) -> Result<tokio::sync::mpsc::Receiver<Vec<i16>>, TalkError> {
            let (sender, receiver) = tokio::sync::mpsc::channel(1);
            self.sender = Some(sender);
            Ok(receiver)
        }

        fn stop(&mut self) -> Result<(), TalkError> {
            self.stopped = true;
            self.sender.take();
            Ok(())
        }
    }

    #[tokio::test]
    async fn terminal_recording_cleanup_stops_live_capture_before_ogg_join() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut capture = HoldingCapture::default();
        let audio_rx = capture.start().expect("start capture");
        let buffer = Arc::new(AudioBuffer::new());
        let ogg_task = tokio::spawn(ogg_recording_task(
            audio_rx,
            dir.path().join("capture.ogg"),
            AudioConfig::new(),
            buffer,
        ));
        let mut feedback =
            RecordingFeedback::new(crate::audio::recording_feedback::RecordingFeedbackOptions {
                no_sounds: true,
                no_boop: true,
                no_overlay: true,
                capture_rate: 16_000,
                viz: None,
                mono: false,
                boop_interval_ms: 0,
                pause_audio: false,
                suppress_boop: None,
                overlay: crate::audio::recording_feedback::RecordingOverlayOptions {
                    silence_tx: None,
                    auto_pause: false,
                    telemetry_rx: None,
                },
            });
        let mut guard = bt_profile::HeadsetGuard::new(None);

        tokio::time::timeout(
            std::time::Duration::from_secs(30),
            finish_live_recording(&mut capture, &mut feedback, &mut guard, ogg_task, false),
        )
        .await
        .expect("terminal cleanup must not wait on running capture")
        .expect("capture stop succeeds");
        assert!(capture.stopped);
    }

    async fn scripted_terminal_session(fail_reconnect: bool) {
        use futures::SinkExt;
        use tokio_tungstenite::tungstenite::Message;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("local listener");
        let endpoint = format!(
            "http://{}",
            listener.local_addr().expect("listener address")
        );
        let server = tokio::spawn(async move {
            for session in 0..if fail_reconnect { 3 } else { 2 } {
                let (stream, _) = listener.accept().await.expect("local session");
                let mut ws = tokio_tungstenite::accept_async(stream)
                    .await
                    .expect("upgrade");
                if session != 2 {
                    ws.send(Message::Text(r#"{"type":"session.created"}"#.into()))
                        .await
                        .expect("session created");
                }
                if session == 0 {
                    ws.send(Message::Text(r#"{"type":"session.updated"}"#.into()))
                        .await
                        .expect("session updated");
                } else if fail_reconnect {
                    ws.send(Message::Text(
                        r#"{"type":"error","error":{"message":"session unavailable"}}"#.into(),
                    ))
                    .await
                    .expect("scripted error");
                } else {
                    ws.send(Message::Text(r#"{"type":"transcription.done"}"#.into()))
                        .await
                        .expect("early done");
                }
            }
        });

        let dir = tempfile::tempdir().expect("temp dir");
        let mut capture = HoldingCapture::default();
        let audio_rx = capture.start().expect("capture start");
        let config: Config = serde_yaml::from_str(&format!(
            "output_dir: {}\nproviders:\n  mistral:\n    api_key: local\n    url: {}\n",
            dir.path().display(),
            endpoint,
        ))
        .expect("local config");
        let mut feedback =
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
            });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            dictate_realtime(
                config,
                Provider::Mistral,
                None,
                &dir.path().join("recording.ogg"),
                audio_rx,
                &mut capture,
                false,
                &mut feedback,
                None,
                None,
                &CancellationToken::new(),
                bt_profile::HeadsetGuard::new(None),
                None,
                None,
            ),
        )
        .await
        .expect("early Done must not hang")
        .expect("dictation result");
        assert_eq!(result.0.text, "");
        assert!(capture.stopped);
        server.await.expect("local server completed");
    }

    #[tokio::test]
    async fn realtime_chain_switches_on_busy_handshake_before_streaming() {
        use futures::SinkExt;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio_tungstenite::tungstenite::Message;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("local listener");
        let endpoint = format!(
            "http://{}",
            listener.local_addr().expect("listener address")
        );
        let server = tokio::spawn(async move {
            let (mut busy, _) = listener.accept().await.expect("busy handshake");
            let mut request = [0u8; 4096];
            let mut len = 0;
            while !request[..len].windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                assert!(len < request.len(), "upgrade headers exceed test buffer");
                let read = busy
                    .read(&mut request[len..])
                    .await
                    .expect("upgrade request");
                assert!(read > 0, "connection closed mid-upgrade");
                len += read;
            }
            assert!(String::from_utf8_lossy(&request[..len])
                .contains("model=voxtral-mini-transcribe-realtime-2602"));
            busy.write_all(
                b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .await
            .expect("busy response");
            drop(busy);
            for phase in 0..2 {
                let (stream, _) = listener.accept().await.expect("next model handshake");
                let mut ws = tokio_tungstenite::accept_async(stream)
                    .await
                    .expect("upgrade next model");
                ws.send(Message::Text(r#"{"type":"session.created"}"#.into()))
                    .await
                    .expect("session created");
                if phase == 1 {
                    ws.send(Message::Text(r#"{"type":"transcription.done"}"#.into()))
                        .await
                        .expect("transcription complete");
                }
            }
        });
        let dir = tempfile::tempdir().expect("temp dir");
        let config: Config = serde_yaml::from_str(&format!(
            "output_dir: {}\nproviders:\n  mistral:\n    api_key: local\n    url: {}\ntranscription:\n  chains:\n    live:\n      - mistral/voxtral-mini-transcribe-realtime-2602\n      - model: mistral/future-rt\n        supports: [realtime]\n",
            dir.path().display(), endpoint,
        )).expect("local chain config");
        let chain = config
            .resolve_chain(
                crate::config::ChainCommand::Dictate,
                Some("live"),
                None,
                None,
            )
            .expect("resolve")
            .expect("live chain")
            .eligible(false, true, None)
            .expect("realtime eligible");
        let mut capture = HoldingCapture::default();
        let audio_rx = capture.start().expect("start capture");
        let mut feedback =
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
            });
        let outage_path = dir.path().join("outages.yml");
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            dictate_realtime(
                config,
                Provider::Mistral,
                Some("voxtral-mini-transcribe-realtime-2602"),
                &dir.path().join("recording.ogg"),
                audio_rx,
                &mut capture,
                false,
                &mut feedback,
                None,
                None,
                &CancellationToken::new(),
                bt_profile::HeadsetGuard::new(None),
                Some(&chain),
                Some(&outage_path),
            ),
        )
        .await
        .expect("connection fallback must finish")
        .expect("second connection succeeds");
        assert_eq!(result.1, Provider::Mistral);
        assert_eq!(result.2, "future-rt");
        assert_eq!(
            result
                .0
                .metadata
                .attempts
                .iter()
                .map(|attempt| attempt.outcome.as_str())
                .collect::<Vec<_>>(),
            ["busy", "success"]
        );
        assert!(capture.stopped);
        server.await.expect("scripted server");
    }

    #[tokio::test]
    async fn early_provider_done_stops_live_capture_and_finishes_dictation() {
        scripted_terminal_session(false).await;
    }

    #[tokio::test]
    async fn failed_reconnect_stops_live_capture_and_finishes_dictation() {
        scripted_terminal_session(true).await;
    }

    #[test]
    fn normal_openai_completion_emits_incrementally_and_finish_does_not_resend() {
        let mut transcript = NormalTranscriptAccumulator::default();
        transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "item-1".to_string(),
            previous_item_id: None,
        });
        let delta = transcript.apply(TranscriptionEvent::ItemTextDelta {
            item_id: "item-1".to_string(),
            content_index: 0,
            text: "Hello world.".to_string(),
        });
        assert_eq!(delta.live_text, "Hello world.");
        assert!(delta.segments_to_send.is_empty());

        let completed = transcript.apply(TranscriptionEvent::ItemTextCompleted {
            item_id: "item-1".to_string(),
            content_index: 0,
            transcript: "Hello, corrected world.".to_string(),
        });
        assert_eq!(completed.live_text, "Hello, corrected world.");
        assert_eq!(
            completed.segments_to_send,
            vec!["Hello, corrected world.".to_string()]
        );

        let finished = transcript.finish();
        assert_eq!(finished.text, "Hello, corrected world.");
        assert!(finished.segments_to_send.is_empty());
    }

    #[test]
    fn normal_openai_reverse_completion_emits_in_conversation_order() {
        let mut transcript = NormalTranscriptAccumulator::default();
        transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "item-1".to_string(),
            previous_item_id: None,
        });
        transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "item-2".to_string(),
            previous_item_id: Some("item-1".to_string()),
        });

        let second = transcript.apply(TranscriptionEvent::ItemTextCompleted {
            item_id: "item-2".to_string(),
            content_index: 0,
            transcript: "second".to_string(),
        });
        assert!(second.segments_to_send.is_empty());
        let first = transcript.apply(TranscriptionEvent::ItemTextCompleted {
            item_id: "item-1".to_string(),
            content_index: 0,
            transcript: "first".to_string(),
        });

        assert_eq!(
            first.segments_to_send,
            vec!["first".to_string(), "second".to_string()]
        );
        assert_eq!(transcript.finish().text, "first second");
    }

    #[test]
    fn normal_openai_late_item_created_event_unblocks_incremental_emission() {
        let mut transcript = NormalTranscriptAccumulator::default();
        let second = transcript.apply(TranscriptionEvent::ItemTextCompleted {
            item_id: "item-2".to_string(),
            content_index: 0,
            transcript: "second".to_string(),
        });
        assert!(second.segments_to_send.is_empty());
        transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "item-1".to_string(),
            previous_item_id: None,
        });
        let first = transcript.apply(TranscriptionEvent::ItemTextCompleted {
            item_id: "item-1".to_string(),
            content_index: 0,
            transcript: "first".to_string(),
        });
        assert!(first.segments_to_send.is_empty());

        let ordered = transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "item-2".to_string(),
            previous_item_id: Some("item-1".to_string()),
        });

        assert_eq!(
            ordered.segments_to_send,
            vec!["first".to_string(), "second".to_string()]
        );
    }

    #[test]
    fn normal_openai_finish_preserves_provisional_terminal_text() {
        let mut transcript = NormalTranscriptAccumulator::default();
        transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "item-1".to_string(),
            previous_item_id: None,
        });
        transcript.apply(TranscriptionEvent::ItemTextDelta {
            item_id: "item-1".to_string(),
            content_index: 0,
            text: "provisional terminal text".to_string(),
        });

        let finished = transcript.finish();

        assert_eq!(finished.text, "provisional terminal text");
        assert_eq!(
            finished.segments_to_send,
            vec!["provisional terminal text".to_string()]
        );
    }

    #[test]
    fn normal_openai_replay_reset_deduplicates_emitted_prefix() {
        let mut transcript = NormalTranscriptAccumulator::default();
        transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "old-item".to_string(),
            previous_item_id: None,
        });
        let old = transcript.apply(TranscriptionEvent::ItemTextCompleted {
            item_id: "old-item".to_string(),
            content_index: 0,
            transcript: "old text".to_string(),
        });
        assert_eq!(old.segments_to_send, vec!["old text".to_string()]);

        transcript.reset_item_generation_for_replay();
        transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "fresh-1".to_string(),
            previous_item_id: None,
        });
        let replayed = transcript.apply(TranscriptionEvent::ItemTextCompleted {
            item_id: "fresh-1".to_string(),
            content_index: 0,
            transcript: "old text".to_string(),
        });
        assert!(replayed.segments_to_send.is_empty());
        transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "fresh-2".to_string(),
            previous_item_id: Some("fresh-1".to_string()),
        });
        let suffix = transcript.apply(TranscriptionEvent::ItemTextCompleted {
            item_id: "fresh-2".to_string(),
            content_index: 0,
            transcript: "new text".to_string(),
        });

        assert_eq!(suffix.segments_to_send, vec!["new text".to_string()]);
        let finished = transcript.finish();
        assert_eq!(finished.text, "old text new text");
        assert!(finished.segments_to_send.is_empty());
    }

    #[test]
    fn replayed_correction_replaces_transcript_without_duplicate_paste() {
        let mut transcript = NormalTranscriptAccumulator::default();
        transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "old".into(),
            previous_item_id: None,
        });
        let first = transcript.apply(TranscriptionEvent::ItemTextCompleted {
            item_id: "old".into(),
            content_index: 0,
            transcript: "Hello world".into(),
        });
        assert_eq!(first.segments_to_send, ["Hello world"]);
        transcript.reset_item_generation_for_replay();
        transcript.apply(TranscriptionEvent::ItemCreated {
            item_id: "replay".into(),
            previous_item_id: None,
        });
        let corrected = transcript.apply(TranscriptionEvent::ItemTextCompleted {
            item_id: "replay".into(),
            content_index: 0,
            transcript: "Hello, world".into(),
        });

        assert!(corrected.segments_to_send.is_empty());
        assert_eq!(transcript.finish().text, "Hello, world");
    }

    #[test]
    fn normal_generic_segments_remain_additive() {
        let mut transcript = NormalTranscriptAccumulator::default();
        let first = transcript.apply(TranscriptionEvent::SegmentDelta {
            text: "first".to_string(),
            start: None,
            end: None,
        });
        let second = transcript.apply(TranscriptionEvent::SegmentDelta {
            text: "second".to_string(),
            start: None,
            end: None,
        });

        assert_eq!(first.segments_to_send, vec!["first".to_string()]);
        assert_eq!(second.segments_to_send, vec!["second".to_string()]);
        assert_eq!(transcript.finish().text, "first second");
    }

    #[test]
    fn realtime_segments_preserve_existing_boundary_whitespace() {
        let mut transcript = NormalTranscriptAccumulator::default();
        transcript.apply(TranscriptionEvent::SegmentDelta {
            text: "Hello\n".into(),
            start: None,
            end: None,
        });
        transcript.apply(TranscriptionEvent::SegmentDelta {
            text: "world".into(),
            start: None,
            end: None,
        });
        assert_eq!(transcript.finish().text, "Hello\nworld");
    }

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

        // Poll the waiter to Pending before closing, rather than guessing
        // when a spawned task reaches its wait with a fixed delay.
        let mut waiting = Box::pin(buf.read_from(1));
        assert!(futures::poll!(waiting.as_mut()).is_pending());
        buf.close();
        let (chunks, cursor) = waiting.await;
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

        // A closed receiver makes the feeder exit; the deadline only detects a hang.
        let handle = tokio::spawn(buffer_feeder(buf, tx, 0));
        tokio::time::timeout(std::time::Duration::from_secs(30), handle)
            .await
            .expect("feeder should not hang")
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

    #[tokio::test]
    async fn buffer_feeder_replays_requested_cursor_and_drains_on_close() {
        let buffer = Arc::new(AudioBuffer::new());
        buffer.push(vec![10]).await;
        buffer.push(vec![20]).await;
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let feeder = tokio::spawn(buffer_feeder(Arc::clone(&buffer), tx, 1));
        assert_eq!(rx.recv().await, Some(vec![20]));
        buffer.push(vec![30]).await;
        buffer.close();
        assert_eq!(rx.recv().await, Some(vec![30]));
        assert_eq!(rx.recv().await, None);
        feeder.await.expect("feeder completed");
    }

    fn read_ogg_packets(path: &std::path::Path) -> Vec<Vec<u8>> {
        let file = std::fs::File::open(path).expect("open ogg");
        let mut reader = ogg::reading::PacketReader::new(std::io::BufReader::new(file));
        let mut packets = Vec::new();

        while let Some(packet) = reader.read_packet().expect("read packet") {
            packets.push(packet.data);
        }

        packets
    }

    // ── ogg_recording_task tests ────────────────────────────────────

    #[tokio::test]
    async fn ogg_recording_task_writes_complete_ogg() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let ogg_path = dir.path().join("test.ogg");
        let audio_config = AudioConfig::new();
        let buffer = Arc::new(AudioBuffer::new());

        let (tx, rx) = tokio::sync::mpsc::channel(10);

        let buf_clone = Arc::clone(&buffer);
        let path_clone = ogg_path.clone();
        let handle = tokio::spawn(ogg_recording_task(rx, path_clone, audio_config, buf_clone));

        // Send 5 chunks of 320 samples (20ms at 16kHz mono).
        for i in 0..5u16 {
            let chunk: Vec<i16> = (0..320)
                .map(|s| (s as i16).wrapping_mul(i as i16))
                .collect();
            tx.send(chunk).await.expect("send chunk");
        }
        drop(tx); // close channel → task finishes

        handle.await.expect("task join").expect("ogg write");

        let data = std::fs::read(&ogg_path).expect("read ogg");
        assert_eq!(&data[0..4], b"OggS");

        let packets = read_ogg_packets(&ogg_path);
        assert_eq!(&packets[0][..8], b"OpusHead");
        assert_eq!(&packets[1][..8], b"OpusTags");
        assert_eq!(packets.len(), 8); // headers, five live frames, one lookahead tail
    }

    #[tokio::test]
    async fn ogg_recording_task_populates_buffer() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let ogg_path = dir.path().join("test.ogg");
        let audio_config = AudioConfig::new();
        let buffer = Arc::new(AudioBuffer::new());

        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let buf_clone = Arc::clone(&buffer);
        let handle = tokio::spawn(ogg_recording_task(rx, ogg_path, audio_config, buf_clone));

        tx.send(vec![1, 2, 3]).await.expect("send");
        tx.send(vec![4, 5, 6]).await.expect("send");
        drop(tx);

        handle.await.expect("join").expect("ogg");

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
    async fn ogg_recording_independent_of_feeder_failure() {
        // Verify that the OGG file is complete even when the feeder
        // (downstream transcription pipeline) fails.
        let dir = tempfile::tempdir().expect("create temp dir");
        let ogg_path = dir.path().join("test.ogg");
        let audio_config = AudioConfig::new();
        let buffer = Arc::new(AudioBuffer::new());

        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let buf_clone = Arc::clone(&buffer);
        let path_clone = ogg_path.clone();
        let ogg_handle = tokio::spawn(ogg_recording_task(rx, path_clone, audio_config, buf_clone));

        // Start a feeder that will be killed.
        let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel(10);
        let feeder = tokio::spawn(buffer_feeder(Arc::clone(&buffer), fwd_tx, 0));

        // Send some audio.
        tx.send(vec![10; 320]).await.expect("send");
        tx.send(vec![20; 320]).await.expect("send");

        // Kill the feeder by dropping the receiver.
        drop(fwd_rx);
        // Wait for feeder to notice and exit.
        feeder.await.expect("feeder stopped when downstream closed");

        // Send more audio AFTER the feeder died — OGG must still record.
        tx.send(vec![30; 320])
            .await
            .expect("send after feeder death");
        drop(tx);

        ogg_handle.await.expect("join").expect("ogg");

        // All 3 chunks must be encoded into the OGG stream.
        let packets = read_ogg_packets(&ogg_path);
        assert_eq!(packets.len(), 6); // headers, three live frames, one lookahead tail

        let mut source =
            crate::audio::file_source::OggFileSource::new(&ogg_path).expect("valid cached OGG");
        let mut decoded_rx = source.start().expect("decode cached OGG");
        let mut decoded = Vec::new();
        while let Some(chunk) = decoded_rx.recv().await {
            decoded.extend(chunk);
        }
        assert!(decoded.len().abs_diff(3 * 320) <= 320);
        assert!(decoded.iter().any(|sample| *sample != 0));

        // All 3 chunks must be in the buffer.
        let (chunks, _) = buffer.read_from(0).await;
        assert_eq!(chunks.len(), 3);
    }
}
