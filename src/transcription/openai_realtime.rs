//! OpenAI Realtime transcription via WebSocket.
//!
//! Connects to the OpenAI Realtime API over WebSocket in
//! transcription-only mode (`?intent=transcription`), configures
//! the session via `session.update`, streams PCM
//! audio resampled from 16 kHz to 24 kHz, and receives
//! incremental transcription events.

use super::realtime::TranscriptionEvent;
use super::RealtimeTranscriber;
use crate::config::OpenAIConfig;
use crate::error::TalkError;
use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use futures::stream::SplitSink;
use futures::{SinkExt, StreamExt};
use std::collections::BTreeMap;
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

// `WS_CONNECT_TIMEOUT` deleted in Step 7 of transport-consolidation.
// Per-attempt connect budgets now live inside
// `transport::ws_upgrade` (`CONNECTION_BUDGETS_SECS = [2, 5, 8, 11, 15]`),
// shared with the HTTP path.

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
        | Some("transcription_session.updated") => {
            let session_id = value
                .get("session")
                .and_then(|v| v.get("id"))
                .and_then(|v| v.as_str())
                .map(ToString::to_string);
            let conversation_id = value
                .get("conversation")
                .and_then(|v| v.get("id"))
                .and_then(|v| v.as_str())
                .map(ToString::to_string);
            TranscriptionEvent::SessionInfo {
                session_id,
                conversation_id,
            }
        }

        Some("conversation.item.input_audio_transcription.delta") => {
            match (openai_item_key(&value), string_field(&value, "delta")) {
                (Some((item_id, content_index)), Some(text)) => TranscriptionEvent::ItemTextDelta {
                    item_id,
                    content_index,
                    text,
                },
                _ => unknown_openai_event(event_type, json_str),
            }
        }

        Some("conversation.item.input_audio_transcription.completed") => {
            match (openai_item_key(&value), string_field(&value, "transcript")) {
                (Some((item_id, content_index)), Some(transcript)) => {
                    TranscriptionEvent::ItemTextCompleted {
                        item_id,
                        content_index,
                        transcript,
                    }
                }
                _ => unknown_openai_event(event_type, json_str),
            }
        }

        Some("conversation.item.created") => {
            let item_id = value
                .get("item")
                .and_then(|item| item.get("id"))
                .and_then(|id| id.as_str())
                .or_else(|| value.get("item_id").and_then(|id| id.as_str()));
            match item_id {
                Some(item_id) => TranscriptionEvent::ItemCreated {
                    item_id: item_id.to_string(),
                    previous_item_id: optional_string_field(&value, "previous_item_id"),
                },
                None => unknown_openai_event(event_type, json_str),
            }
        }

        Some("input_audio_buffer.committed") => match string_field(&value, "item_id") {
            Some(item_id) => TranscriptionEvent::ItemCreated {
                item_id,
                previous_item_id: optional_string_field(&value, "previous_item_id"),
            },
            None => unknown_openai_event(event_type, json_str),
        },

        Some("error") => {
            let message = value
                .get("error")
                .and_then(|v| v.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error")
                .to_string();
            TranscriptionEvent::Error { message }
        }

        Some("rate_limits.updated") => TranscriptionEvent::RateLimitsUpdated { raw: value },

        // VAD events — logged by caller, no user-visible event.
        Some("input_audio_buffer.speech_started") | Some("input_audio_buffer.speech_stopped") => {
            unknown_openai_event(event_type, json_str)
        }

        _ => unknown_openai_event(event_type, json_str),
    }
}

fn openai_item_key(value: &serde_json::Value) -> Option<(String, u64)> {
    let item_id = value.get("item_id")?.as_str()?.to_string();
    let content_index = value.get("content_index")?.as_u64()?;
    Some((item_id, content_index))
}

fn string_field(value: &serde_json::Value, field: &str) -> Option<String> {
    value.get(field)?.as_str().map(ToString::to_string)
}

fn optional_string_field(value: &serde_json::Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(|v| v.as_str())
        .map(ToString::to_string)
}

fn unknown_openai_event(event_type: Option<&str>, raw: &str) -> TranscriptionEvent {
    TranscriptionEvent::Unknown {
        event_type: event_type.map(ToString::to_string),
        raw: raw.to_string(),
    }
}

fn extract_ws_upgrade_headers(
    headers: &tokio_tungstenite::tungstenite::http::HeaderMap,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (name, value) in headers {
        let key = name.as_str();
        let should_keep = key == "x-request-id"
            || key == "openai-processing-ms"
            || key.starts_with("x-ratelimit-");
        if should_keep {
            if let Ok(v) = value.to_str() {
                out.insert(key.to_string(), v.to_string());
            }
        }
    }
    out
}

// ── WebSocket URL ───────────────────────────────────────────────────

/// Build the WebSocket URL from a base endpoint.
///
/// Uses `?intent=transcription` to create a transcription-only
/// session.  The transcription model is set separately in
/// `session.update`.
pub fn build_ws_url(endpoint: &str) -> String {
    format!("{}{}?{}", endpoint, REALTIME_PATH, REALTIME_INTENT)
}

/// Convert an HTTP(S) base URL to a WebSocket URL.
///
/// `https://` becomes `wss://`, `http://` becomes `ws://`.
/// URLs that already use a `ws://` or `wss://` scheme are returned
/// unchanged.
fn http_to_ws(url: &str) -> String {
    if let Some(rest) = url.strip_prefix("https://") {
        format!("wss://{}", rest)
    } else if let Some(rest) = url.strip_prefix("http://") {
        format!("ws://{}", rest)
    } else {
        url.to_string()
    }
}

fn build_session_update(
    config: &OpenAIConfig,
    model: &str,
) -> Result<serde_json::Value, TalkError> {
    let capability = super::openai::validate_openai_hints(
        super::openai::OpenAITranscriptionMode::Realtime,
        model,
        config.prompt.as_deref(),
        config.keywords.as_deref(),
        config.languages.as_deref(),
        config.realtime_delay,
    )?;

    let mut transcription = serde_json::Map::new();
    transcription.insert("model".to_string(), serde_json::json!(model));
    if let Some(prompt) = &config.prompt {
        transcription.insert("prompt".to_string(), serde_json::json!(prompt));
    }
    match capability {
        super::openai::OpenAIModelCapability::GptLiveTranscribe => {
            if let Some(keywords) = &config.keywords {
                transcription.insert("keywords".to_string(), serde_json::json!(keywords));
            }
            if let Some(languages) = &config.languages {
                transcription.insert("languages".to_string(), serde_json::json!(languages));
            }
            if let Some(delay) = config.realtime_delay {
                transcription.insert("delay".to_string(), serde_json::json!(delay.to_string()));
            }
        }
        super::openai::OpenAIModelCapability::LegacyRealtime => {
            if let Some(language) = config.languages.as_ref().and_then(|values| values.first()) {
                transcription.insert("language".to_string(), serde_json::json!(language));
            }
        }
        super::openai::OpenAIModelCapability::GptTranscribe
        | super::openai::OpenAIModelCapability::LegacyBatch => {
            return Err(TalkError::Config(format!(
                "OpenAI model '{model}' cannot be encoded for realtime transcription"
            )));
        }
    }

    Ok(serde_json::json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": transcription
                }
            }
        }
    }))
}

// ── Transcriber ─────────────────────────────────────────────────────

/// Realtime transcriber that connects via WebSocket to the OpenAI
/// Realtime API in transcription-only mode.
pub struct OpenAIRealtimeTranscriber {
    config: OpenAIConfig,
    /// Realtime transcription model (e.g. `gpt-live-transcribe`).
    model: String,
    endpoint: String,
    /// Telemetry sink for WS upgrade lifecycle events.
    sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink>,
    /// Cancellation token wired into the WS upgrade.  See
    /// [`super::realtime::MistralRealtimeTranscriber::cancel_token`]
    /// for the wiring rationale.
    cancel_token: CancellationToken,
}

impl OpenAIRealtimeTranscriber {
    /// Create a new realtime transcriber with the given configuration.
    pub fn new(config: OpenAIConfig) -> Self {
        let model = config.realtime_model.clone();
        let endpoint = config
            .url
            .as_deref()
            .map(|u| http_to_ws(u.trim_end_matches('/')))
            .unwrap_or_else(|| DEFAULT_OPENAI_REALTIME_ENDPOINT.to_string());
        Self {
            config,
            model,
            endpoint,
            sink: std::sync::Arc::new(crate::telemetry::NoOpSink),
            cancel_token: CancellationToken::new(),
        }
    }

    /// Create a new realtime transcriber with an explicit model override.
    pub fn with_model(config: OpenAIConfig, model: String) -> Self {
        let endpoint = config
            .url
            .as_deref()
            .map(|u| http_to_ws(u.trim_end_matches('/')))
            .unwrap_or_else(|| DEFAULT_OPENAI_REALTIME_ENDPOINT.to_string());
        Self {
            config,
            model,
            endpoint,
            sink: std::sync::Arc::new(crate::telemetry::NoOpSink),
            cancel_token: CancellationToken::new(),
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
            sink: std::sync::Arc::new(crate::telemetry::NoOpSink),
            cancel_token: CancellationToken::new(),
        }
    }

    /// Open a throwaway WebSocket connection, send `session.update`
    /// with our model config, and
    /// wait for the API's answer.
    ///
    /// Uses [`super::transport::ws_upgrade`] for the handshake so
    /// retries / growing budget / cancellation share the unified
    /// transport machinery.
    #[allow(dead_code)]
    async fn validate_realtime_session(&self) -> Result<(), TalkError> {
        let session_update = build_session_update(&self.config, &self.model)?;
        let ws_url = build_ws_url(&self.endpoint);

        log::debug!("validation: connecting to {}", ws_url);

        let req = super::transport::Request {
            method: super::transport::Method::Get,
            url: ws_url.clone(),
            // OpenAI deprecated the ``OpenAI-Beta: realtime=v1``
            // header on 2026-02-27; the GA endpoint
            // ``/v1/realtime`` rejects requests carrying it with
            // "The Realtime Beta API is no longer supported.
            // Please use /v1/realtime for the GA API."
            headers: vec![(
                "Authorization".into(),
                format!("Bearer {}", self.config.api_key),
            )],
            body: super::transport::RequestBody::Empty,
            provider: crate::config::Provider::OpenAI,
            provider_name: "OpenAI".into(),
            phase: crate::error::PipelinePhase::Validate,
            wall_clock: None,
        };
        let ws_stream = super::transport::ws_upgrade(req, &self.sink, self.cancel_token.clone())
            .await
            .map_err(|pf| TalkError::Config(pf.to_string()))?;

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

        // Send session.update in the GA shape.  The flat beta
        // format (``type: transcription_session.update``,
        // ``session.input_audio_format``, etc.) was deprecated on
        // 2026-02-27; the GA endpoint replies with an
        // ``unknown_event_type`` error and the session is closed.
        //
        // GA shape (see https://developers.openai.com/api/docs/guides/realtime-transcription):
        // ``{ type: "session.update", session: { type: "transcription",
        //     audio: { input: { format: {type:"audio/pcm",rate:24000},
        //              transcription: {model: <name>} } } } }``.
        //
        // GA also dropped the separate ``transcription_session``
        // namespace — the same ``session.update`` event handles
        // both speech-to-speech and transcription, discriminated
        // by the inner ``session.type`` field.
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
                        TranscriptionEvent::SessionInfo { .. }
                        | TranscriptionEvent::SessionCreated => {
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
    ///
    /// The WS upgrade goes through
    /// [`super::transport::ws_upgrade`], which handles connection
    /// retries (growing budget `[2, 5, 8, 11, 15]` seconds),
    /// cancellation, and `ConnectionEvent` emission.
    pub async fn transcribe_realtime(
        &self,
        audio_rx: mpsc::Receiver<Vec<i16>>,
    ) -> Result<mpsc::Receiver<TranscriptionEvent>, TalkError> {
        let session_update = build_session_update(&self.config, &self.model)?;
        let ws_url = build_ws_url(&self.endpoint);

        log::debug!("connecting to OpenAI Realtime WebSocket: {}", ws_url);

        let req = super::transport::Request {
            method: super::transport::Method::Get,
            url: ws_url.clone(),
            // OpenAI deprecated the ``OpenAI-Beta: realtime=v1``
            // header on 2026-02-27 — the GA endpoint rejects it.
            headers: vec![(
                "Authorization".into(),
                format!("Bearer {}", self.config.api_key),
            )],
            body: super::transport::RequestBody::Empty,
            provider: crate::config::Provider::OpenAI,
            provider_name: "OpenAI".into(),
            phase: crate::error::PipelinePhase::Request,
            wall_clock: None,
        };
        let ws_stream = super::transport::ws_upgrade(req, &self.sink, self.cancel_token.clone())
            .await
            .map_err(|pf| TalkError::Transcription(pf.to_string()))?;
        // No HTTP response wrapper exposed by the transport; the
        // OpenAI realtime path's downstream code paths that
        // previously inspected `response` (mainly for the
        // `x-request-id` header) need to live without it for now.
        // Step 10 of the plan adds a richer transport response if
        // any consumer actually needs the headers.
        let response: Option<()> = None;

        let (mut ws_sink, mut ws_source) = ws_stream.split();

        // Surface the post-upgrade phase to the picker UI so the
        // row doesn't appear silent during the 100ms-3s window
        // between WS open and the first transcription delta.
        self.sink
            .emit(crate::telemetry::TranscriptionEvent::Status {
                message: "session handshake…".into(),
                t: std::time::Instant::now(),
            });
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
        self.sink
            .emit(crate::telemetry::TranscriptionEvent::Status {
                message: "session ready, awaiting audio…".into(),
                t: std::time::Instant::now(),
            });

        // Send session.update in the GA shape — see the mirror
        // call in `validate_realtime_session` for the rationale
        // (flat beta format was deprecated 2026-02-27).
        log::debug!("sending session.update (GA) with model={}", self.model);
        ws_sink
            .send(Message::Text(session_update.to_string()))
            .await
            .map_err(|e| {
                TalkError::Transcription(format!("Failed to send session.update: {}", e))
            })?;
        self.sink
            .emit(crate::telemetry::TranscriptionEvent::Status {
                message: "streaming audio…".into(),
                t: std::time::Instant::now(),
            });

        // Create event channel.
        let (event_tx, event_rx) = mpsc::channel::<TranscriptionEvent>(100);

        // The transport's `ws_upgrade` does not surface the HTTP
        // upgrade response headers today (the value would land at
        // `response` above as `Option<()>`).  Step 10 of the
        // transport-consolidation plan extends the transport's
        // response shape to carry headers when an actual consumer
        // (this site, the OpenAI rate-limit dashboard) needs them.
        // Until then we silence the
        // `TranscriptionEvent::TransportMetadata` emission; it was
        // diagnostic-only.
        let _suppressed_unused = &response;
        let _: fn(_) -> _ = extract_ws_upgrade_headers; // keep helper alive for Step 10

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
        super::openai::validate_openai_hints(
            super::openai::OpenAITranscriptionMode::Realtime,
            &self.model,
            self.config.prompt.as_deref(),
            self.config.keywords.as_deref(),
            self.config.languages.as_deref(),
            self.config.realtime_delay,
        )?;
        // Step 1: REST check — validates API key + model existence,
        // and lists available transcription models on failure.
        let api_base = self
            .endpoint
            .replace("wss://", "https://")
            .replace("ws://", "http://");
        // The realtime path does not yet thread a telemetry sink
        // Preflight events go through `self.sink`, so when a UI is
        // attached (via `set_sink`) they reach it.
        super::openai::validate_openai_model(
            &self.config.api_key,
            &self.model,
            &api_base,
            &self.sink,
        )
        .await?;

        // Step 2: WebSocket check — connect, send
        // session.update with our model config, and wait
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

    fn set_sink(&mut self, sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink>) {
        self.sink = sink;
    }

    fn set_cancel_token(&mut self, token: CancellationToken) {
        self.cancel_token = token;
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
                TranscriptionEvent::SessionInfo { .. } | TranscriptionEvent::SessionCreated => {
                    return Ok(event);
                }
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

                        // Log unknown events at debug level.
                        if let TranscriptionEvent::Unknown {
                            event_type: Some(ref t),
                            ..
                        } = event
                        {
                            log::debug!("OpenAI event: {}", t);
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

    fn openai_config(model: &str) -> OpenAIConfig {
        OpenAIConfig {
            api_key: "key".to_string(),
            url: None,
            model: "gpt-transcribe".to_string(),
            realtime_model: model.to_string(),
            prompt: None,
            keywords: None,
            languages: None,
            realtime_delay: None,
        }
    }

    #[test]
    fn test_http_to_ws_https() {
        assert_eq!(http_to_ws("https://api.openai.com"), "wss://api.openai.com");
    }

    #[test]
    fn test_http_to_ws_http() {
        assert_eq!(http_to_ws("http://localhost:8080"), "ws://localhost:8080");
    }

    #[test]
    fn test_new_uses_custom_url() {
        let config = OpenAIConfig {
            api_key: "key".to_string(),
            url: Some("https://custom.example.com".to_string()),
            model: "whisper-1".to_string(),
            realtime_model: "gpt-live-transcribe".to_string(),
            prompt: None,
            keywords: None,
            languages: None,
            realtime_delay: None,
        };
        let transcriber = OpenAIRealtimeTranscriber::new(config);
        assert_eq!(transcriber.endpoint, "wss://custom.example.com");
    }

    #[test]
    fn test_new_default_endpoint() {
        let config = OpenAIConfig {
            api_key: "key".to_string(),
            url: None,
            model: "whisper-1".to_string(),
            realtime_model: "gpt-live-transcribe".to_string(),
            prompt: None,
            keywords: None,
            languages: None,
            realtime_delay: None,
        };
        let transcriber = OpenAIRealtimeTranscriber::new(config);
        assert_eq!(transcriber.endpoint, "wss://api.openai.com");
    }

    #[test]
    fn test_with_model_uses_custom_url() {
        let config = OpenAIConfig {
            api_key: "key".to_string(),
            url: Some("https://custom.example.com".to_string()),
            model: "whisper-1".to_string(),
            realtime_model: "gpt-live-transcribe".to_string(),
            prompt: None,
            keywords: None,
            languages: None,
            realtime_delay: None,
        };
        let transcriber =
            OpenAIRealtimeTranscriber::with_model(config, "gpt-realtime-whisper".to_string());
        assert_eq!(transcriber.endpoint, "wss://custom.example.com");
    }

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
            TranscriptionEvent::SessionInfo { .. }
        ));
    }

    #[test]
    fn test_parse_openai_event_transcription_session_created() {
        let json = r#"{"type": "transcription_session.created", "session": {}}"#;
        assert!(matches!(
            parse_openai_event(json),
            TranscriptionEvent::SessionInfo { .. }
        ));
    }

    #[test]
    fn test_parse_openai_event_transcription_session_updated() {
        let json = r#"{"type": "transcription_session.updated", "session": {}}"#;
        assert!(matches!(
            parse_openai_event(json),
            TranscriptionEvent::SessionInfo { .. }
        ));
    }

    #[test]
    fn test_parse_openai_event_rate_limits_updated() {
        let json = r#"{"type": "rate_limits.updated", "rate_limits": []}"#;
        assert!(matches!(
            parse_openai_event(json),
            TranscriptionEvent::RateLimitsUpdated { .. }
        ));
    }

    #[test]
    fn openai_delta_preserves_item_key_and_text() {
        let json = r#"{"type":"conversation.item.input_audio_transcription.delta","item_id":"item-7","content_index":2,"delta":"hello "}"#;
        match parse_openai_event(json) {
            TranscriptionEvent::ItemTextDelta {
                item_id,
                content_index,
                text,
            } => {
                assert_eq!(item_id, "item-7");
                assert_eq!(content_index, 2);
                assert_eq!(text, "hello ");
            }
            other => panic!("expected ItemTextDelta, got {:?}", other),
        }
    }

    #[test]
    fn openai_completed_preserves_item_key_and_authoritative_transcript() {
        let json = r#"{"type":"conversation.item.input_audio_transcription.completed","item_id":"item-7","content_index":2,"transcript":"hello, corrected world."}"#;
        match parse_openai_event(json) {
            TranscriptionEvent::ItemTextCompleted {
                item_id,
                content_index,
                transcript,
            } => {
                assert_eq!(item_id, "item-7");
                assert_eq!(content_index, 2);
                assert_eq!(transcript, "hello, corrected world.");
            }
            other => panic!("expected ItemTextCompleted, got {:?}", other),
        }
    }

    #[test]
    fn openai_order_events_preserve_item_links() {
        let created = r#"{"type":"conversation.item.created","previous_item_id":"item-1","item":{"id":"item-2"}}"#;
        let committed = r#"{"type":"input_audio_buffer.committed","item_id":"item-3","previous_item_id":"item-2"}"#;

        for (json, expected_item, expected_previous) in [
            (created, "item-2", Some("item-1")),
            (committed, "item-3", Some("item-2")),
        ] {
            match parse_openai_event(json) {
                TranscriptionEvent::ItemCreated {
                    item_id,
                    previous_item_id,
                } => {
                    assert_eq!(item_id, expected_item);
                    assert_eq!(previous_item_id.as_deref(), expected_previous);
                }
                other => panic!("expected ItemCreated, got {:?}", other),
            }
        }
    }

    #[test]
    fn openai_malformed_item_event_remains_nonfatal() {
        let json = r#"{"type":"conversation.item.input_audio_transcription.delta","content_index":0,"delta":"hello"}"#;
        assert!(matches!(
            parse_openai_event(json),
            TranscriptionEvent::Unknown { .. }
        ));
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

    #[test]
    fn gpt_live_session_update_nests_exact_migration_fields() {
        let mut config = openai_config("gpt-live-transcribe");
        config.prompt = Some("Keep names exact.".to_string());
        config.keywords = Some(vec!["Kalysto".to_string(), "talk-rs".to_string()]);
        config.languages = Some(vec!["fr".to_string(), "en".to_string()]);
        config.realtime_delay = Some(crate::config::OpenAIRealtimeDelay::High);

        let actual = build_session_update(&config, "gpt-live-transcribe").expect("valid update");
        assert_eq!(
            actual,
            serde_json::json!({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {
                                "model": "gpt-live-transcribe",
                                "prompt": "Keep names exact.",
                                "keywords": ["Kalysto", "talk-rs"],
                                "languages": ["fr", "en"],
                                "delay": "high"
                            }
                        }
                    }
                }
            })
        );
    }

    #[test]
    fn gpt_live_session_update_omits_unconfigured_hints() {
        let config = openai_config("gpt-live-transcribe");
        let actual = build_session_update(&config, "gpt-live-transcribe").expect("valid update");
        assert_eq!(
            actual,
            serde_json::json!({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {"model": "gpt-live-transcribe"}
                        }
                    }
                }
            })
        );
    }

    #[test]
    fn legacy_realtime_maps_one_language_and_rejects_incompatible_hints() {
        let mut config = openai_config("gpt-realtime-whisper");
        config.prompt = Some("Keep names exact.".to_string());
        config.languages = Some(vec!["fr".to_string()]);
        let actual =
            build_session_update(&config, "gpt-realtime-whisper").expect("valid legacy update");
        assert_eq!(
            actual["session"]["audio"]["input"]["transcription"],
            serde_json::json!({
                "model": "gpt-realtime-whisper",
                "prompt": "Keep names exact.",
                "language": "fr"
            })
        );

        config.keywords = Some(vec!["Kalysto".to_string()]);
        let error =
            build_session_update(&config, "gpt-realtime-whisper").expect_err("keywords rejected");
        assert_eq!(error.to_string(), "Configuration error: OpenAI model 'gpt-realtime-whisper' does not support field 'keywords'");

        config.keywords = None;
        config.languages = Some(vec!["fr".to_string(), "en".to_string()]);
        let error =
            build_session_update(&config, "gpt-realtime-whisper").expect_err("languages rejected");
        assert_eq!(error.to_string(), "Configuration error: OpenAI model 'gpt-realtime-whisper' does not support multiple values for field 'languages'");
    }

    #[tokio::test]
    async fn legacy_realtime_rejects_hints_before_websocket_upgrade() {
        for (keywords, languages, expected) in [
            (
                Some(vec!["Kalysto".to_string()]),
                None,
                "Configuration error: OpenAI model 'gpt-realtime-whisper' does not support field 'keywords'",
            ),
            (
                None,
                Some(vec!["fr".to_string(), "en".to_string()]),
                "Configuration error: OpenAI model 'gpt-realtime-whisper' does not support multiple values for field 'languages'",
            ),
        ] {
            let mut config = openai_config("gpt-realtime-whisper");
            config.keywords = keywords;
            config.languages = languages;
            let transcriber =
                OpenAIRealtimeTranscriber::with_endpoint(config, "ws://127.0.0.1:1".to_string());
            let (_audio_tx, audio_rx) = mpsc::channel(1);

            let error = transcriber
                .transcribe_realtime(audio_rx)
                .await
                .expect_err("invalid hints must fail before connecting");
            assert_eq!(error.to_string(), expected);
        }
    }

    #[tokio::test]
    async fn realtime_validate_rejects_hints_before_rest_preflight() {
        for (keywords, languages, expected) in [
            (
                Some(vec!["Kalysto".to_string()]),
                None,
                "Configuration error: OpenAI model 'gpt-realtime-whisper' does not support field 'keywords'",
            ),
            (
                None,
                Some(vec!["fr".to_string(), "en".to_string()]),
                "Configuration error: OpenAI model 'gpt-realtime-whisper' does not support multiple values for field 'languages'",
            ),
        ] {
            let mut config = openai_config("gpt-realtime-whisper");
            config.keywords = keywords;
            config.languages = languages;
            let transcriber =
                OpenAIRealtimeTranscriber::with_endpoint(config, "ws://127.0.0.1:1".to_string());

            let error = RealtimeTranscriber::validate(&transcriber)
                .await
                .expect_err("hints rejected before REST preflight");
            assert_eq!(error.to_string(), expected);
        }
    }

    #[tokio::test]
    async fn realtime_validate_rejects_known_batch_model_before_rest_preflight() {
        for model in crate::transcription::openai::OPENAI_BATCH_MODELS {
            let config = openai_config(model);
            let transcriber =
                OpenAIRealtimeTranscriber::with_endpoint(config, "ws://127.0.0.1:1".to_string());

            let error = RealtimeTranscriber::validate(&transcriber)
                .await
                .expect_err("batch model rejected before REST preflight");
            assert_eq!(
                error.to_string(),
                format!(
                    "Configuration error: OpenAI model '{model}' is batch-only and cannot be used for realtime transcription"
                )
            );
        }
    }

    #[test]
    fn realtime_session_builder_rejects_known_batch_model() {
        let config = openai_config("gpt-transcribe");
        let error = build_session_update(&config, "gpt-transcribe")
            .expect_err("batch model rejected by realtime builder");
        assert_eq!(
            error.to_string(),
            "Configuration error: OpenAI model 'gpt-transcribe' is batch-only and cannot be used for realtime transcription"
        );
    }
}
