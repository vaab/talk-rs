//! OpenAI Realtime transcription via WebSocket.
//!
//! Connects to the OpenAI Realtime API over WebSocket in
//! transcription-only mode (`?intent=transcription`), configures
//! the session via `transcription_session.update`, streams PCM
//! audio resampled from 16 kHz to 24 kHz, and receives
//! incremental transcription events.

use super::realtime::TranscriptionEvent;
use super::RealtimeTranscriber;
use crate::core::config::OpenAIConfig;
use crate::core::error::TalkError;
use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use futures::stream::SplitSink;
use futures::{SinkExt, StreamExt};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;

/// Default WebSocket base endpoint.
const DEFAULT_OPENAI_REALTIME_ENDPOINT: &str = "wss://api.openai.com";

/// WebSocket path for the realtime API.
const REALTIME_PATH: &str = "/v1/realtime";

/// Query parameter that tells the Realtime API we want a
/// transcription-only session (no AI responses).
const REALTIME_INTENT: &str = "intent=transcription";

/// Timeout for the initial WebSocket connection (TCP + TLS + upgrade).
const WS_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// Timeout for receiving the `session.created` event after connecting.
const SESSION_CREATED_TIMEOUT: Duration = Duration::from_secs(15);

/// Interval between WebSocket ping frames for keepalive.
const WS_PING_INTERVAL: Duration = Duration::from_secs(30);

/// Time to wait for final transcription events after audio ends.
const POST_COMMIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Sample rate expected by the OpenAI Realtime API for PCM16.
const OPENAI_SAMPLE_RATE: u32 = 24000;

/// Source sample rate from our audio capture pipeline.
const SOURCE_SAMPLE_RATE: u32 = 16000;

// ── Resampling ──────────────────────────────────────────────────────

/// Resample PCM i16 audio from 16 kHz to 24 kHz using linear
/// interpolation.
///
/// The ratio 24000/16000 = 3/2, so for every 2 input samples we
/// produce 3 output samples.  This is good enough for speech audio.
pub fn resample_16k_to_24k(input: &[i16]) -> Vec<i16> {
    if input.is_empty() {
        return Vec::new();
    }
    let in_len = input.len();
    let out_len = (in_len as u64 * OPENAI_SAMPLE_RATE as u64 / SOURCE_SAMPLE_RATE as u64) as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        // Position in the input signal (fixed-point via f64).
        let src_pos = i as f64 * SOURCE_SAMPLE_RATE as f64 / OPENAI_SAMPLE_RATE as f64;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;

        let sample = if idx + 1 < in_len {
            let a = input[idx] as f64;
            let b = input[idx + 1] as f64;
            (a + (b - a) * frac) as i16
        } else {
            input[in_len - 1]
        };
        output.push(sample);
    }
    output
}

// ── Encoding helpers ────────────────────────────────────────────────

/// Convert a slice of `i16` PCM samples to little-endian bytes.
fn pcm_to_bytes(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

/// Encode PCM bytes as base64.
fn pcm_bytes_to_base64(bytes: &[u8]) -> String {
    BASE64_STANDARD.encode(bytes)
}

// ── Event parsing ───────────────────────────────────────────────────

/// Parse a JSON text frame from the OpenAI Realtime API into a
/// [`TranscriptionEvent`].
pub fn parse_openai_event(json_str: &str) -> TranscriptionEvent {
    let value: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => {
            return TranscriptionEvent::Unknown {
                event_type: None,
                raw: json_str.to_string(),
            };
        }
    };

    let event_type = value.get("type").and_then(|v| v.as_str());

    match event_type {
        Some("session.created")
        | Some("session.updated")
        | Some("transcription_session.created")
        | Some("transcription_session.updated") => TranscriptionEvent::SessionCreated,

        Some("conversation.item.input_audio_transcription.delta") => {
            let text = value
                .get("delta")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            TranscriptionEvent::TextDelta { text }
        }

        Some("conversation.item.input_audio_transcription.completed") => {
            let text = value
                .get("transcript")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            TranscriptionEvent::SegmentDelta {
                text,
                start: None,
                end: None,
            }
        }

        Some("error") => {
            let message = value
                .get("error")
                .and_then(|v| v.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error")
                .to_string();
            TranscriptionEvent::Error { message }
        }

        // VAD events — logged by caller, no user-visible event.
        Some("input_audio_buffer.speech_started")
        | Some("input_audio_buffer.speech_stopped")
        | Some("input_audio_buffer.committed")
        | Some("conversation.item.created") => TranscriptionEvent::Unknown {
            event_type: event_type.map(|s| s.to_string()),
            raw: json_str.to_string(),
        },

        _ => TranscriptionEvent::Unknown {
            event_type: event_type.map(|s| s.to_string()),
            raw: json_str.to_string(),
        },
    }
}

// ── WebSocket URL ───────────────────────────────────────────────────

/// Build the WebSocket URL from a base endpoint.
///
/// Uses `?intent=transcription` to create a transcription-only
/// session.  The transcription model is set separately in
/// `transcription_session.update`.
pub fn build_ws_url(endpoint: &str) -> String {
    format!("{}{}?{}", endpoint, REALTIME_PATH, REALTIME_INTENT)
}

// ── Transcriber ─────────────────────────────────────────────────────

/// Realtime transcriber that connects via WebSocket to the OpenAI
/// Realtime API in transcription-only mode.
pub struct OpenAIRealtimeTranscriber {
    config: OpenAIConfig,
    /// Transcription model (e.g. `gpt-4o-mini-transcribe`).
    model: String,
    endpoint: String,
}

impl OpenAIRealtimeTranscriber {
    /// Create a new realtime transcriber with the given configuration.
    ///
    /// Uses the `realtime_model` from config as the default model.
    pub fn new(config: OpenAIConfig) -> Self {
        let model = config.realtime_model.clone();
        Self {
            config,
            model,
            endpoint: DEFAULT_OPENAI_REALTIME_ENDPOINT.to_string(),
        }
    }

    /// Create a new realtime transcriber with an explicit model override.
    pub fn with_model(config: OpenAIConfig, model: String) -> Self {
        Self {
            config,
            model,
            endpoint: DEFAULT_OPENAI_REALTIME_ENDPOINT.to_string(),
        }
    }

    /// Create a new realtime transcriber with a custom endpoint (for testing).
    #[cfg(test)]
    pub fn with_endpoint(config: OpenAIConfig, endpoint: String) -> Self {
        let model = config.realtime_model.clone();
        Self {
            config,
            model,
            endpoint,
        }
    }

    /// Open a throwaway WebSocket connection, send
    /// `transcription_session.update` with our model config, and
    /// wait for the API's answer.
    ///
    /// Returns `Ok(())` if the API accepts the session configuration,
    /// or an error with the API's message (e.g. "model X is not
    /// supported in realtime mode").
    async fn validate_realtime_session(&self) -> Result<(), TalkError> {
        let ws_url = build_ws_url(&self.endpoint);

        let parsed_url = url::Url::parse(&ws_url)
            .map_err(|e| TalkError::Config(format!("Invalid WebSocket URL: {}", e)))?;
        let host = parsed_url
            .host_str()
            .ok_or_else(|| TalkError::Config("No host in WebSocket URL".to_string()))?
            .to_string();

        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&ws_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("OpenAI-Beta", "realtime=v1")
            .header("Host", &host)
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .body(())
            .map_err(|e| TalkError::Config(format!("Failed to build WebSocket request: {}", e)))?;

        log::debug!("validation: connecting to {}", ws_url);

        let (ws_stream, _) = tokio::time::timeout(
            WS_CONNECT_TIMEOUT,
            tokio_tungstenite::connect_async(request),
        )
        .await
        .map_err(|_| {
            TalkError::Config(format!(
                "WebSocket connection timed out after {}s",
                WS_CONNECT_TIMEOUT.as_secs()
            ))
        })?
        .map_err(|e| TalkError::Config(format!("WebSocket connection failed: {}", e)))?;

        let (mut sink, mut source) = ws_stream.split();

        // Wait for session.created.
        tokio::time::timeout(
            SESSION_CREATED_TIMEOUT,
            wait_for_session_created(&mut source),
        )
        .await
        .map_err(|_| {
            TalkError::Config(format!(
                "Timed out waiting for session.created after {}s",
                SESSION_CREATED_TIMEOUT.as_secs()
            ))
        })??;

        // Send transcription_session.update with flat beta format.
        let session_update = serde_json::json!({
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": self.model,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        });
        sink.send(Message::Text(session_update.to_string()))
            .await
            .map_err(|e| TalkError::Config(format!("Failed to send session.update: {}", e)))?;

        // Wait for session.updated (success) or error.
        let result = tokio::time::timeout(SESSION_CREATED_TIMEOUT, async {
            while let Some(msg_result) = source.next().await {
                let msg = msg_result.map_err(|e| {
                    TalkError::Config(format!("WebSocket error during validation: {}", e))
                })?;
                if let Message::Text(text) = msg {
                    let event = parse_openai_event(&text);
                    match event {
                        TranscriptionEvent::SessionCreated => {
                            // session.updated → config accepted
                            return Ok(());
                        }
                        TranscriptionEvent::Error { message } => {
                            return Err(TalkError::Config(format!(
                                "Realtime session rejected: {}",
                                message
                            )));
                        }
                        _ => continue,
                    }
                }
            }
            Err(TalkError::Config(
                "WebSocket closed before session was confirmed".to_string(),
            ))
        })
        .await
        .map_err(|_| TalkError::Config("Timed out waiting for session validation".to_string()))?;

        // Close the validation connection cleanly.
        let _ = sink.send(Message::Close(None)).await;

        result
    }

    /// Connect to the OpenAI Realtime API and stream audio for
    /// transcription.
    ///
    /// Reads `Vec<i16>` PCM chunks (16 kHz) from `audio_rx`, resamples
    /// to 24 kHz, encodes as base64, and sends over WebSocket.  Returns
    /// a receiver of transcription events.
    pub async fn transcribe_realtime(
        &self,
        audio_rx: mpsc::Receiver<Vec<i16>>,
    ) -> Result<mpsc::Receiver<TranscriptionEvent>, TalkError> {
        let ws_url = build_ws_url(&self.endpoint);

        // Parse the URL to extract the host for the HTTP header.
        let parsed_url = url::Url::parse(&ws_url)
            .map_err(|e| TalkError::Transcription(format!("Invalid WebSocket URL: {}", e)))?;
        let host = parsed_url
            .host_str()
            .ok_or_else(|| TalkError::Transcription("No host in WebSocket URL".to_string()))?
            .to_string();

        // Build the WebSocket upgrade request with auth header.
        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&ws_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("OpenAI-Beta", "realtime=v1")
            .header("Host", &host)
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .body(())
            .map_err(|e| {
                TalkError::Transcription(format!("Failed to build WebSocket request: {}", e))
            })?;

        log::debug!("connecting to OpenAI Realtime WebSocket: {}", ws_url);

        // Connect with timeout.
        let (ws_stream, _response) = tokio::time::timeout(
            WS_CONNECT_TIMEOUT,
            tokio_tungstenite::connect_async(request),
        )
        .await
        .map_err(|_| {
            TalkError::Transcription(format!(
                "WebSocket connection timed out after {}s",
                WS_CONNECT_TIMEOUT.as_secs()
            ))
        })?
        .map_err(|e| TalkError::Transcription(format!("WebSocket connection failed: {}", e)))?;

        let (mut ws_sink, mut ws_source) = ws_stream.split();

        log::debug!("OpenAI WebSocket connected, waiting for session.created");

        // Wait for session.created with timeout.
        let session_event = tokio::time::timeout(
            SESSION_CREATED_TIMEOUT,
            wait_for_session_created(&mut ws_source),
        )
        .await
        .map_err(|_| {
            TalkError::Transcription(format!(
                "Timed out waiting for session.created after {}s",
                SESSION_CREATED_TIMEOUT.as_secs()
            ))
        })??;
        log::info!("OpenAI realtime session established");

        // Send transcription_session.update with flat beta format.
        let session_update = serde_json::json!({
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": self.model,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        });
        log::debug!(
            "sending transcription_session.update with model={}",
            self.model
        );
        ws_sink
            .send(Message::Text(session_update.to_string()))
            .await
            .map_err(|e| {
                TalkError::Transcription(format!("Failed to send session.update: {}", e))
            })?;

        // Create event channel.
        let (event_tx, event_rx) = mpsc::channel::<TranscriptionEvent>(100);

        // Forward the initial session event.
        let _ = event_tx.send(session_event).await;

        let cancel = CancellationToken::new();
        let audio_done = Arc::new(AtomicBool::new(false));

        // Spawn sender task.
        let sender_task = tokio::spawn(sender_loop(
            audio_rx,
            ws_sink,
            cancel.clone(),
            audio_done.clone(),
        ));

        // Spawn receiver task.
        let receiver_task = tokio::spawn(receiver_loop(
            ws_source,
            event_tx,
            cancel.clone(),
            audio_done,
        ));

        // Cleanup task that logs panics.
        tokio::spawn(async move {
            if let Err(e) = sender_task.await {
                log::error!("OpenAI sender task panicked: {}", e);
            }
            if let Err(e) = receiver_task.await {
                log::error!("OpenAI receiver task panicked: {}", e);
            }
        });

        Ok(event_rx)
    }
}

#[async_trait]
impl RealtimeTranscriber for OpenAIRealtimeTranscriber {
    async fn validate(&self) -> Result<(), TalkError> {
        // Step 1: REST check — validates API key + model existence,
        // and lists available transcription models on failure.
        let api_base = self
            .endpoint
            .replace("wss://", "https://")
            .replace("ws://", "http://");
        super::openai::validate_openai_model(&self.config.api_key, &self.model, &api_base).await?;

        // Step 2: WebSocket check — connect, send
        // transcription_session.update with our model config, and wait
        // for the API's answer.  This catches errors like "model X is
        // not supported in realtime mode" that the REST models endpoint
        // cannot detect.
        self.validate_realtime_session().await
    }

    async fn transcribe_realtime(
        &self,
        audio_rx: mpsc::Receiver<Vec<i16>>,
    ) -> Result<mpsc::Receiver<TranscriptionEvent>, TalkError> {
        self.transcribe_realtime(audio_rx).await
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Wait for the `session.created` event from the WebSocket stream.
async fn wait_for_session_created<S>(ws_source: &mut S) -> Result<TranscriptionEvent, TalkError>
where
    S: futures::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    while let Some(msg_result) = ws_source.next().await {
        let msg = msg_result.map_err(|e| {
            TalkError::Transcription(format!(
                "WebSocket error waiting for session.created: {}",
                e
            ))
        })?;

        if let Message::Text(text) = msg {
            let event = parse_openai_event(&text);
            match event {
                TranscriptionEvent::SessionCreated => return Ok(event),
                TranscriptionEvent::Error { ref message } => {
                    return Err(TalkError::Transcription(format!(
                        "Server error during session setup: {}",
                        message
                    )));
                }
                _ => {
                    // Ignore other events during setup.
                }
            }
        }
    }

    Err(TalkError::Transcription(
        "WebSocket closed before session.created received".to_string(),
    ))
}

/// Sender loop: reads PCM chunks (16 kHz), resamples to 24 kHz,
/// base64-encodes, and sends as `input_audio_buffer.append` over
/// WebSocket.
///
/// When the audio channel closes, sends `input_audio_buffer.commit`
/// and sets the `audio_done` flag so the receiver knows to expect
/// no more audio.
async fn sender_loop<S>(
    mut audio_rx: mpsc::Receiver<Vec<i16>>,
    mut ws_sink: SplitSink<S, Message>,
    cancel: CancellationToken,
    audio_done: Arc<AtomicBool>,
) where
    S: futures::Sink<Message> + Unpin,
    <S as futures::Sink<Message>>::Error: std::fmt::Display,
{
    let mut ping_interval = tokio::time::interval(WS_PING_INTERVAL);
    // Skip the first immediate tick.
    ping_interval.tick().await;

    loop {
        tokio::select! {
            chunk = audio_rx.recv() => {
                match chunk {
                    Some(pcm_chunk) => {
                        // Resample 16 kHz → 24 kHz.
                        let resampled = resample_16k_to_24k(&pcm_chunk);
                        let bytes = pcm_to_bytes(&resampled);
                        log::trace!(
                            "sending audio chunk: {} in → {} out samples, {} bytes",
                            pcm_chunk.len(),
                            resampled.len(),
                            bytes.len(),
                        );
                        let b64 = pcm_bytes_to_base64(&bytes);

                        let msg = serde_json::json!({
                            "type": "input_audio_buffer.append",
                            "audio": b64
                        });

                        if let Err(e) = ws_sink.send(Message::Text(msg.to_string())).await {
                            log::error!("OpenAI WebSocket send error: {}", e);
                            cancel.cancel();
                            return;
                        }
                    }
                    None => break, // Audio channel closed normally.
                }
            }
            _ = ping_interval.tick() => {
                if let Err(e) = ws_sink.send(Message::Ping(vec![])).await {
                    log::warn!("OpenAI WebSocket ping failed: {}", e);
                    cancel.cancel();
                    return;
                }
            }
            _ = cancel.cancelled() => {
                return;
            }
        }
    }

    // Audio channel closed — commit any remaining audio in the buffer.
    let commit_msg = serde_json::json!({
        "type": "input_audio_buffer.commit"
    });
    log::debug!("sending input_audio_buffer.commit");
    if let Err(e) = ws_sink.send(Message::Text(commit_msg.to_string())).await {
        log::error!("OpenAI WebSocket send error (commit): {}", e);
        cancel.cancel();
    }

    // Signal that no more audio will be sent.  The receiver uses
    // this to start a timeout for final transcription events.
    audio_done.store(true, Ordering::Release);

    // Do NOT close the WebSocket — the server still needs to send
    // remaining transcription events.  Just drop the sink.
}

/// Receiver loop: reads WebSocket messages, parses events, forwards
/// to the event channel.
///
/// Once `audio_done` is set and no transcription events arrive for
/// [`POST_COMMIT_TIMEOUT`], sends [`TranscriptionEvent::Done`] and
/// exits.
async fn receiver_loop<S>(
    mut ws_source: S,
    event_tx: mpsc::Sender<TranscriptionEvent>,
    cancel: CancellationToken,
    audio_done: Arc<AtomicBool>,
) where
    S: futures::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        // Choose timeout based on whether audio is done.
        let timeout_dur = if audio_done.load(Ordering::Acquire) {
            POST_COMMIT_TIMEOUT
        } else {
            // Effectively infinite while audio is still streaming.
            Duration::from_secs(3600)
        };

        tokio::select! {
            msg_opt = ws_source.next() => {
                let msg_result = match msg_opt {
                    Some(r) => r,
                    None => {
                        log::warn!("OpenAI WebSocket stream ended unexpectedly");
                        let _ = event_tx.send(TranscriptionEvent::Done).await;
                        cancel.cancel();
                        return;
                    }
                };
                let msg = match msg_result {
                    Ok(m) => m,
                    Err(e) => {
                        log::error!("OpenAI WebSocket receive error: {}", e);
                        let _ = event_tx
                            .send(TranscriptionEvent::Error {
                                message: format!("WebSocket error: {}", e),
                            })
                            .await;
                        cancel.cancel();
                        return;
                    }
                };

                match msg {
                    Message::Text(text) => {
                        log::trace!("received OpenAI WS text: {}", text);
                        let event = parse_openai_event(&text);

                        // Log VAD events at debug level.
                        if let TranscriptionEvent::Unknown {
                            event_type: Some(ref t),
                            ..
                        } = event
                        {
                            log::debug!("OpenAI event: {}", t);
                            continue;
                        }

                        let is_error = matches!(event, TranscriptionEvent::Error { .. });
                        if event_tx.send(event).await.is_err() {
                            cancel.cancel();
                            return;
                        }
                        if is_error {
                            return;
                        }
                    }
                    Message::Close(frame) => {
                        log::debug!("received OpenAI WS Close frame: {:?}", frame);
                        let _ = event_tx.send(TranscriptionEvent::Done).await;
                        return;
                    }
                    Message::Pong(_) => {
                        log::trace!("received OpenAI WS Pong");
                    }
                    _ => {
                        // Ignore binary frames.
                    }
                }
            }
            _ = tokio::time::sleep(timeout_dur) => {
                // Post-commit timeout expired with no events — we are
                // done collecting transcription results.
                log::debug!(
                    "no events for {}s after commit, finalising",
                    POST_COMMIT_TIMEOUT.as_secs(),
                );
                let _ = event_tx.send(TranscriptionEvent::Done).await;
                return;
            }
            _ = cancel.cancelled() => {
                return;
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_empty() {
        assert!(resample_16k_to_24k(&[]).is_empty());
    }

    #[test]
    fn test_resample_ratio() {
        // 100 samples at 16 kHz → ~150 samples at 24 kHz.
        let input: Vec<i16> = (0..100).collect();
        let output = resample_16k_to_24k(&input);
        assert_eq!(output.len(), 150);
    }

    #[test]
    fn test_resample_preserves_endpoints() {
        let input: Vec<i16> = vec![0, 1000, 2000, 3000];
        let output = resample_16k_to_24k(&input);
        // First sample should be the same.
        assert_eq!(output[0], 0);
        // Last sample should be close to 3000.
        assert!((output[output.len() - 1] - 3000).unsigned_abs() <= 1);
    }

    #[test]
    fn test_resample_single_sample() {
        let output = resample_16k_to_24k(&[42]);
        // A single sample resamples to ceil(1 * 24000/16000) = 1 or 2.
        assert!(!output.is_empty());
        assert_eq!(output[0], 42);
    }

    #[test]
    fn test_parse_openai_event_session_created() {
        let json = r#"{"type": "session.created", "session": {}}"#;
        assert!(matches!(
            parse_openai_event(json),
            TranscriptionEvent::SessionCreated
        ));
    }

    #[test]
    fn test_parse_openai_event_transcription_session_created() {
        let json = r#"{"type": "transcription_session.created", "session": {}}"#;
        assert!(matches!(
            parse_openai_event(json),
            TranscriptionEvent::SessionCreated
        ));
    }

    #[test]
    fn test_parse_openai_event_transcription_session_updated() {
        let json = r#"{"type": "transcription_session.updated", "session": {}}"#;
        assert!(matches!(
            parse_openai_event(json),
            TranscriptionEvent::SessionCreated
        ));
    }

    #[test]
    fn test_parse_openai_event_transcription_delta() {
        let json =
            r#"{"type": "conversation.item.input_audio_transcription.delta", "delta": "hello "}"#;
        match parse_openai_event(json) {
            TranscriptionEvent::TextDelta { text } => assert_eq!(text, "hello "),
            other => panic!("expected TextDelta, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_openai_event_transcription_completed() {
        let json = r#"{"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello world"}"#;
        match parse_openai_event(json) {
            TranscriptionEvent::SegmentDelta { text, .. } => {
                assert_eq!(text, "hello world");
            }
            other => panic!("expected SegmentDelta, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_openai_event_error() {
        let json = r#"{"type": "error", "error": {"message": "bad request"}}"#;
        match parse_openai_event(json) {
            TranscriptionEvent::Error { message } => assert_eq!(message, "bad request"),
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_openai_event_vad_returns_unknown() {
        let json = r#"{"type": "input_audio_buffer.speech_started"}"#;
        assert!(matches!(
            parse_openai_event(json),
            TranscriptionEvent::Unknown { .. }
        ));
    }

    #[test]
    fn test_parse_openai_event_invalid_json() {
        let json = "not json{{";
        assert!(matches!(
            parse_openai_event(json),
            TranscriptionEvent::Unknown {
                event_type: None,
                ..
            }
        ));
    }

    #[test]
    fn test_build_ws_url() {
        let url = build_ws_url("wss://api.openai.com");
        assert_eq!(url, "wss://api.openai.com/v1/realtime?intent=transcription");
    }

    #[test]
    fn test_build_ws_url_custom_endpoint() {
        let url = build_ws_url("wss://custom.example.com");
        assert_eq!(
            url,
            "wss://custom.example.com/v1/realtime?intent=transcription"
        );
    }

    #[test]
    fn test_pcm_to_base64_roundtrip() {
        let samples: Vec<i16> = vec![256, 32767, -1, -32768];
        let bytes = pcm_to_bytes(&samples);
        let b64 = pcm_bytes_to_base64(&bytes);
        let decoded = BASE64_STANDARD.decode(&b64).expect("valid base64");
        assert_eq!(decoded, bytes);
    }
}
