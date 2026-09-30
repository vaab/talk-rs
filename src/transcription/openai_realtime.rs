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
#[derive(Default)]
struct Resampler16To24 {
    previous: Option<i16>,
    input_count: usize,
}

impl Resampler16To24 {
    fn process(&mut self, input: &[i16]) -> Vec<i16> {
        let mut output = Vec::with_capacity(
            (input.len() * OPENAI_SAMPLE_RATE as usize).div_ceil(SOURCE_SAMPLE_RATE as usize),
        );
        for &sample in input {
            if self.input_count % 2 == 1 {
                if let Some(previous) = self.previous {
                    output.push(
                        (previous as f64 + (sample as f64 - previous as f64) * 2.0 / 3.0) as i16,
                    );
                }
            }
            output.push(sample);
            self.previous = Some(sample);
            self.input_count += 1;
        }
        output
    }
}

pub fn resample_16k_to_24k(input: &[i16]) -> Vec<i16> {
    Resampler16To24::default().process(input)
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
    retry_schedule: super::transport::RetrySchedule,
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
            retry_schedule: Default::default(),
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
            retry_schedule: Default::default(),
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
            retry_schedule: Default::default(),
        }
    }

    /// Open a throwaway WebSocket connection, send `session.update`
    /// with our model config, and
    /// wait for the API's answer.
    ///
    /// Uses [`super::transport::ws_upgrade`] for the handshake so
    /// retries / growing budget / cancellation share the unified
    /// transport machinery.
    async fn validate_realtime_session(&self) -> Result<(), TalkError> {
        let session_update = build_session_update(&self.config, &self.model)?;
        let ws_url = build_ws_url(&self.endpoint);

        log::debug!("validation: connecting to {}", ws_url);

        let req = super::transport::Request {
            retry_schedule: self.retry_schedule.clone(),
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
            .map_err(TalkError::from)?;

        let (mut sink, mut source) = ws_stream.split();

        // Wait for session.created.
        tokio::time::timeout(
            SESSION_CREATED_TIMEOUT,
            wait_for_session_created(&mut source),
        )
        .await
        .map_err(|_| {
            crate::error::PipelineFailure::session_timeout(
                "OpenAI",
                crate::error::PipelinePhase::Validate,
                &ws_url,
                SESSION_CREATED_TIMEOUT,
            )
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
        .map_err(|_| {
            crate::error::PipelineFailure::session_timeout(
                "OpenAI",
                crate::error::PipelinePhase::Validate,
                &ws_url,
                SESSION_CREATED_TIMEOUT,
            )
        })?;

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
            retry_schedule: self.retry_schedule.clone(),
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
            .map_err(TalkError::from)?;
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
            crate::error::PipelineFailure::session_timeout(
                "OpenAI",
                crate::error::PipelinePhase::Request,
                &ws_url,
                SESSION_CREATED_TIMEOUT,
            )
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
        let (audio_end_tx, audio_end_rx) = tokio::sync::watch::channel(None);

        // Spawn sender task.
        let sender_task = tokio::spawn(sender_loop(
            audio_rx,
            ws_sink,
            cancel.clone(),
            audio_done.clone(),
            audio_end_tx,
        ));

        // Spawn receiver task.
        let receiver_task = tokio::spawn(receiver_loop(
            ws_source,
            event_tx,
            cancel.clone(),
            audio_done,
            audio_end_rx,
            super::realtime::FINAL_TRANSCRIPT_DEADLINE,
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
    fn set_retry_schedule(&mut self, schedule: super::transport::RetrySchedule) {
        self.retry_schedule = schedule;
    }
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
            self.cancel_token.clone(),
            self.retry_schedule.clone(),
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
    audio_end_tx: tokio::sync::watch::Sender<Option<tokio::time::Instant>>,
) where
    S: futures::Sink<Message> + Unpin,
    <S as futures::Sink<Message>>::Error: std::fmt::Display,
{
    let mut ping_interval = tokio::time::interval(WS_PING_INTERVAL);
    let mut resampler = Resampler16To24::default();
    // Skip the first immediate tick.
    ping_interval.tick().await;

    loop {
        tokio::select! {
            chunk = audio_rx.recv() => {
                match chunk {
                    Some(pcm_chunk) => {
                        // Resample 16 kHz → 24 kHz.
                        let resampled = resampler.process(&pcm_chunk);
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
    audio_end_tx.send_replace(Some(tokio::time::Instant::now()));

    // Do NOT close the WebSocket — the server still needs to send
    // remaining transcription events.  Just drop the sink.
}

/// Receiver loop: reads WebSocket messages, parses events, forwards
/// to the event channel.
///
/// Once audio ends, accepts late events for a fixed final deadline.
async fn receiver_loop<S>(
    mut ws_source: S,
    event_tx: mpsc::Sender<TranscriptionEvent>,
    cancel: CancellationToken,
    audio_done: Arc<AtomicBool>,
    mut audio_end: tokio::sync::watch::Receiver<Option<tokio::time::Instant>>,
    final_deadline: Duration,
) where
    S: futures::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let mut saw_final = false;
    loop {
        let end = *audio_end.borrow();
        tokio::select! {
            changed = audio_end.changed(), if end.is_none() => {
                if changed.is_err() { return; }
            }
            _ = async {
                if let Some(end) = end {
                    tokio::time::sleep_until(end + final_deadline).await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {
                log::warn!("OpenAI final transcription deadline reached after {}s; keeping accumulated text", final_deadline.as_secs());
                let _ = event_tx.send(TranscriptionEvent::Done).await;
                return;
            }
            msg_opt = ws_source.next() => {
                let msg_result = match msg_opt {
                    Some(r) => r,
                    None => {
                        log::warn!("OpenAI WebSocket stream ended unexpectedly");
                        let _ = event_tx.send(if audio_done.load(Ordering::Acquire) && saw_final {
                            TranscriptionEvent::Done
                        } else {
                            TranscriptionEvent::Error { message: "WebSocket closed before final transcription".to_string() }
                        }).await;
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
                        saw_final |= matches!(event, TranscriptionEvent::ItemTextCompleted { .. });
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
                        let _ = event_tx.send(if audio_done.load(Ordering::Acquire) && saw_final {
                            TranscriptionEvent::Done
                        } else {
                            TranscriptionEvent::Error { message: "WebSocket closed before final transcription".to_string() }
                        }).await;
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

    #[tokio::test]
    async fn setup_ignores_non_text_then_returns_created_session_identifiers() {
        let mut source = futures::stream::iter([
            Ok(Message::Ping(Vec::new())),
            Ok(Message::Text(r#"{"type":"rate_limits.updated"}"#.into())),
            Ok(Message::Text(
                r#"{"type":"session.created","session":{"id":"session-7"},"conversation":{"id":"conversation-8"}}"#.into(),
            )),
        ]);
        let event = wait_for_session_created(&mut source)
            .await
            .expect("session creation follows unrelated frames");
        match event {
            TranscriptionEvent::SessionInfo {
                session_id,
                conversation_id,
            } => {
                assert_eq!(session_id.as_deref(), Some("session-7"));
                assert_eq!(conversation_id.as_deref(), Some("conversation-8"));
            }
            other => panic!("expected session creation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn setup_eof_without_session_creation_is_an_error() {
        let mut source = futures::stream::iter(vec![Ok(Message::Text(
            r#"{"type":"rate_limits.updated"}"#.into(),
        ))]);
        let error = wait_for_session_created(&mut source)
            .await
            .expect_err("unrelated event cannot complete setup");
        assert_eq!(
            error.to_string(),
            "Transcription error: WebSocket closed before session.created received"
        );
    }

    #[tokio::test]
    async fn receiver_forwards_server_error_once_after_irrelevant_binary_frame() {
        let (tx, mut rx) = mpsc::channel(4);
        receiver_loop(
            futures::stream::iter(vec![
                Ok(Message::Binary(vec![0, 1, 2])),
                Ok(Message::Text(
                    r#"{"type":"error","error":{"message":"model unavailable"}}"#.into(),
                )),
            ]),
            tx,
            CancellationToken::new(),
            Arc::new(AtomicBool::new(false)),
            tokio::sync::watch::channel(None).1,
            Duration::from_secs(15),
        )
        .await;
        match rx.recv().await {
            Some(TranscriptionEvent::Error { message }) => {
                assert_eq!(message, "model unavailable");
            }
            other => panic!("expected server error, got {other:?}"),
        }
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn receiver_reports_transport_error_without_successful_completion() {
        let (tx, mut rx) = mpsc::channel(4);
        let cancel = CancellationToken::new();
        receiver_loop(
            futures::stream::iter(vec![Err(tokio_tungstenite::tungstenite::Error::Io(
                std::io::Error::new(std::io::ErrorKind::ConnectionReset, "fixture reset"),
            ))]),
            tx,
            cancel.clone(),
            Arc::new(AtomicBool::new(true)),
            tokio::sync::watch::channel(None).1,
            Duration::from_secs(15),
        )
        .await;
        match rx.recv().await {
            Some(TranscriptionEvent::Error { message }) => {
                assert_eq!(message, "WebSocket error: IO error: fixture reset");
            }
            other => panic!("expected transport error, got {other:?}"),
        }
        assert!(rx.recv().await.is_none());
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn upgrade_diagnostics_omit_unapproved_headers() {
        let mut headers = tokio_tungstenite::tungstenite::http::HeaderMap::new();
        headers.insert("x-request-id", "request-7".parse().expect("header"));
        headers.insert("openai-processing-ms", "31".parse().expect("header"));
        headers.insert(
            "x-ratelimit-remaining-requests",
            "4".parse().expect("header"),
        );
        headers.insert("authorization", "Bearer private".parse().expect("header"));
        headers.insert("set-cookie", "session=private".parse().expect("header"));
        assert_eq!(
            extract_ws_upgrade_headers(&headers),
            BTreeMap::from([
                ("openai-processing-ms".to_string(), "31".to_string()),
                (
                    "x-ratelimit-remaining-requests".to_string(),
                    "4".to_string()
                ),
                ("x-request-id".to_string(), "request-7".to_string()),
            ])
        );
    }

    #[tokio::test]
    async fn sender_resamples_across_chunks_then_commits_and_marks_audio_end() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        let client_ws = tokio_tungstenite::WebSocketStream::from_raw_socket(
            client_io,
            tokio_tungstenite::tungstenite::protocol::Role::Client,
            None,
        )
        .await;
        let mut server_ws = tokio_tungstenite::WebSocketStream::from_raw_socket(
            server_io,
            tokio_tungstenite::tungstenite::protocol::Role::Server,
            None,
        )
        .await;
        let (sink, _source) = client_ws.split();
        let (audio_tx, audio_rx) = mpsc::channel(2);
        audio_tx.send(vec![0]).await.expect("first PCM chunk");
        audio_tx
            .send(vec![1200, 2400])
            .await
            .expect("second PCM chunk");
        drop(audio_tx);
        let audio_done = Arc::new(AtomicBool::new(false));
        let (end_tx, end_rx) = tokio::sync::watch::channel(None);

        sender_loop(
            audio_rx,
            sink,
            CancellationToken::new(),
            Arc::clone(&audio_done),
            end_tx,
        )
        .await;

        let mut frames = Vec::new();
        for _ in 0..3 {
            let message = server_ws
                .next()
                .await
                .expect("frame arrived")
                .expect("valid frame");
            let Message::Text(json) = message else {
                panic!("expected JSON text frame");
            };
            frames.push(serde_json::from_str::<serde_json::Value>(&json).expect("valid JSON"));
        }
        let first = BASE64_STANDARD
            .decode(frames[0]["audio"].as_str().expect("first audio"))
            .expect("base64");
        let second = BASE64_STANDARD
            .decode(frames[1]["audio"].as_str().expect("second audio"))
            .expect("base64");
        assert_eq!(frames[0]["type"], "input_audio_buffer.append");
        assert_eq!(first, pcm_to_bytes(&[0]));
        assert_eq!(frames[1]["type"], "input_audio_buffer.append");
        assert_eq!(second, pcm_to_bytes(&[800, 1200, 2400]));
        assert_eq!(
            frames[2],
            serde_json::json!({"type": "input_audio_buffer.commit"})
        );
        assert!(audio_done.load(Ordering::Acquire));
        assert!(end_rx.borrow().is_some());
    }

    #[tokio::test]
    async fn setup_rejects_server_error_after_unrelated_event() {
        let mut source = futures::stream::iter([
            Ok(Message::Text(r#"{"type":"rate_limits.updated"}"#.into())),
            Ok(Message::Text(
                r#"{"type":"error","error":{"message":"invalid model"}}"#.into(),
            )),
        ]);
        let error = wait_for_session_created(&mut source)
            .await
            .expect_err("setup must not accept a server error");
        assert_eq!(
            error.to_string(),
            "Transcription error: Server error during session setup: invalid model"
        );
    }

    #[tokio::test]
    async fn receiver_requires_final_event_before_eof_after_audio_ends() {
        let (tx, mut rx) = mpsc::channel(4);
        let cancel = CancellationToken::new();
        receiver_loop(
            futures::stream::iter(vec![Ok(Message::Text(
                r#"{"type":"conversation.item.input_audio_transcription.delta","item_id":"one","content_index":0,"delta":"unfinished"}"#.into(),
            ))]),
            tx,
            cancel.clone(),
            Arc::new(AtomicBool::new(true)),
            tokio::sync::watch::channel(None).1,
            Duration::from_secs(15),
        )
        .await;
        match rx.recv().await {
            Some(TranscriptionEvent::ItemTextDelta {
                item_id,
                content_index,
                text,
            }) => {
                assert_eq!(
                    (item_id.as_str(), content_index, text.as_str()),
                    ("one", 0, "unfinished")
                );
            }
            other => panic!("expected provisional text, got {other:?}"),
        }
        match rx.recv().await {
            Some(TranscriptionEvent::Error { message }) => {
                assert_eq!(message, "WebSocket closed before final transcription");
            }
            other => panic!("expected incomplete-stream error, got {other:?}"),
        }
        assert!(rx.recv().await.is_none());
        assert!(cancel.is_cancelled());
    }

    #[tokio::test]
    async fn receiver_completes_after_final_event_and_eof() {
        let (tx, mut rx) = mpsc::channel(4);
        receiver_loop(
            futures::stream::iter(vec![Ok(Message::Text(
                r#"{"type":"conversation.item.input_audio_transcription.completed","item_id":"one","content_index":0,"transcript":"final text"}"#.into(),
            ))]),
            tx,
            CancellationToken::new(),
            Arc::new(AtomicBool::new(true)),
            tokio::sync::watch::channel(None).1,
            Duration::from_secs(15),
        )
        .await;
        match rx.recv().await {
            Some(TranscriptionEvent::ItemTextCompleted {
                item_id,
                content_index,
                transcript,
            }) => {
                assert_eq!(
                    (item_id.as_str(), content_index, transcript.as_str()),
                    ("one", 0, "final text")
                );
            }
            other => panic!("expected final text, got {other:?}"),
        }
        assert!(matches!(rx.recv().await, Some(TranscriptionEvent::Done)));
        assert!(rx.recv().await.is_none());
    }

    #[test]
    fn session_info_keeps_identifiers() {
        let event = parse_openai_event(
            r#"{"type":"transcription_session.created","session":{"id":"sess-12"},"conversation":{"id":"conv-34"}}"#,
        );
        match event {
            TranscriptionEvent::SessionInfo {
                session_id,
                conversation_id,
            } => {
                assert_eq!(session_id.as_deref(), Some("sess-12"));
                assert_eq!(conversation_id.as_deref(), Some("conv-34"));
            }
            other => panic!("expected session identifiers, got {other:?}"),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn openai_unknown_events_do_not_extend_fifteen_second_final_deadline() {
        let (audio_end_tx, audio_end_rx) = tokio::sync::watch::channel(None);
        let (msg_tx, msg_rx) =
            mpsc::channel::<Result<Message, tokio_tungstenite::tungstenite::Error>>(4);
        let (tx, mut rx) = mpsc::channel(4);
        let receiver = tokio::spawn(receiver_loop(
            tokio_stream::wrappers::ReceiverStream::new(msg_rx),
            tx,
            CancellationToken::new(),
            Arc::new(AtomicBool::new(true)),
            audio_end_rx,
            super::super::realtime::FINAL_TRANSCRIPT_DEADLINE,
        ));
        audio_end_tx.send_replace(Some(tokio::time::Instant::now()));
        tokio::time::advance(Duration::from_secs(9)).await;
        msg_tx
            .send(Ok(Message::Text(r#"{"type":"unrecognized"}"#.into())))
            .await
            .expect("unknown event");
        assert!(matches!(
            rx.recv().await,
            Some(TranscriptionEvent::Unknown { .. })
        ));
        tokio::time::advance(Duration::from_secs(5)).await;
        assert!(rx.try_recv().is_err());
        tokio::time::advance(Duration::from_secs(1)).await;
        assert!(matches!(rx.recv().await, Some(TranscriptionEvent::Done)));
        receiver.await.expect("receiver finished");
    }

    #[tokio::test]
    async fn premature_openai_close_and_eof_reconnect_instead_of_finishing() {
        for messages in [vec![Ok(Message::Close(None))], vec![]] {
            let (tx, mut rx) = mpsc::channel(4);
            receiver_loop(
                futures::stream::iter(messages),
                tx,
                CancellationToken::new(),
                Arc::new(AtomicBool::new(false)),
                tokio::sync::watch::channel(None).1,
                super::super::realtime::FINAL_TRANSCRIPT_DEADLINE,
            )
            .await;
            assert!(matches!(
                rx.recv().await,
                Some(TranscriptionEvent::Error { .. })
            ));
            assert!(rx.recv().await.is_none());
        }
    }

    #[tokio::test]
    async fn openai_close_after_audio_end_finishes() {
        let (tx, mut rx) = mpsc::channel(4);
        receiver_loop(
            futures::stream::iter(vec![
                Ok(Message::Text(r#"{"type":"conversation.item.input_audio_transcription.completed","item_id":"item-1","content_index":0,"transcript":"hello"}"#.to_string())),
                Ok(Message::Close(None)),
            ]),
            tx,
            CancellationToken::new(),
            Arc::new(AtomicBool::new(true)),
            tokio::sync::watch::channel(None).1,
            super::super::realtime::FINAL_TRANSCRIPT_DEADLINE,
        )
        .await;
        assert!(matches!(
            rx.recv().await,
            Some(TranscriptionEvent::ItemTextCompleted { .. })
        ));
        assert!(matches!(rx.recv().await, Some(TranscriptionEvent::Done)));
    }

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
    fn constructor_resolves_endpoint_and_model_variants() {
        for (url, expected) in [
            (None, "wss://api.openai.com"),
            (
                Some("https://custom.example.com"),
                "wss://custom.example.com",
            ),
        ] {
            let mut config = openai_config("gpt-live-transcribe");
            config.url = url.map(str::to_string);
            let transcriber = OpenAIRealtimeTranscriber::new(config.clone());
            assert_eq!(transcriber.endpoint, expected);
            assert_eq!(transcriber.model, "gpt-live-transcribe");
            let overridden =
                OpenAIRealtimeTranscriber::with_model(config, "gpt-realtime-whisper".into());
            assert_eq!(overridden.endpoint, expected);
            assert_eq!(overridden.model, "gpt-realtime-whisper");
        }
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
    fn resampling_is_identical_across_chunk_boundaries() {
        let samples: Vec<i16> = vec![42, 1000, -500, 3000, -2000, 700, 800];
        let whole = resample_16k_to_24k(&samples);
        assert_eq!(whole.len(), samples.len() * 3 / 2);
        for split in 1..samples.len() {
            let mut resampler = Resampler16To24::default();
            let mut chunks = resampler.process(&samples[..split]);
            chunks.extend(resampler.process(&samples[split..]));
            assert_eq!(chunks, whole, "split at {split}");
        }
        assert_eq!(resample_16k_to_24k(&[42]), [42]);
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
    fn ws_url_builder_handles_default_and_custom_endpoints() {
        for (endpoint, expected) in [
            (
                "wss://api.openai.com",
                "wss://api.openai.com/v1/realtime?intent=transcription",
            ),
            (
                "wss://custom.example.com",
                "wss://custom.example.com/v1/realtime?intent=transcription",
            ),
        ] {
            assert_eq!(build_ws_url(endpoint), expected);
        }
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

/// Performance harness, item `openai-realtime-final-early-exit`: how
/// long the receiver keeps a realtime session open after the audio
/// ended, when the server has already delivered the committed item's
/// final transcript but keeps the socket open (paused tokio clock, so
/// the measurement is exact and instant).
#[cfg(test)]
mod perf_realtime_end {
    use super::*;

    fn committed(item: &str) -> String {
        format!(
            r#"{{"type":"input_audio_buffer.committed","item_id":"{item}","previous_item_id":null}}"#
        )
    }

    fn completed(item: &str, transcript: &str) -> String {
        format!(
            r#"{{"type":"conversation.item.input_audio_transcription.completed","item_id":"{item}","content_index":0,"transcript":"{transcript}"}}"#
        )
    }

    /// Drive `receiver_loop` with a scripted server.  Each script step
    /// is `(virtual ms after audio end, message)`; after the script the
    /// server stays silent with the socket open.  Returns the virtual
    /// ms from audio end to `Done`, and every final transcript seen.
    async fn session(script: Vec<(u64, String)>) -> (u128, Vec<String>) {
        let (audio_end_tx, audio_end_rx) = tokio::sync::watch::channel(None);
        let (msg_tx, msg_rx) =
            mpsc::channel::<Result<Message, tokio_tungstenite::tungstenite::Error>>(16);
        let (tx, mut rx) = mpsc::channel(16);
        let receiver = tokio::spawn(receiver_loop(
            tokio_stream::wrappers::ReceiverStream::new(msg_rx),
            tx,
            CancellationToken::new(),
            Arc::new(AtomicBool::new(true)),
            audio_end_rx,
            super::super::realtime::FINAL_TRANSCRIPT_DEADLINE,
        ));
        let end = tokio::time::Instant::now();
        audio_end_tx.send_replace(Some(end));
        let server = tokio::spawn(async move {
            for (at_ms, msg) in script {
                tokio::time::sleep_until(end + Duration::from_millis(at_ms)).await;
                if msg_tx.send(Ok(Message::Text(msg))).await.is_err() {
                    return;
                }
            }
            // Keep the socket open: never send Close.
            std::future::pending::<()>().await;
        });
        let mut finals = Vec::new();
        let done_at = loop {
            match rx.recv().await {
                Some(TranscriptionEvent::ItemTextCompleted { transcript, .. }) => {
                    finals.push(transcript)
                }
                Some(TranscriptionEvent::Done) | None => break end.elapsed().as_millis(),
                Some(_) => {}
            }
        };
        server.abort();
        let _ = receiver.await;
        (done_at, finals)
    }

    #[tokio::test(start_paused = true)]
    async fn perf_realtime_session_end_after_final_item() {
        let (done_ms, finals) = session(vec![
            (40, committed("item-2")),
            (300, completed("item-2", "final words")),
        ])
        .await;
        assert_eq!(finals, vec!["final words".to_string()]);
        crate::perf_counters::record_metrics(
            "openai-realtime-final-early-exit",
            "final-item-socket-open",
            &[("final_item_ms", 300.0), ("session_end_ms", done_ms as f64)],
        );
    }

    /// VAD safety invariant: an earlier item's completion (item-1,
    /// before the stop commit) must not end the session; the final
    /// item-2 completion arriving 3 s later must still be delivered.
    #[tokio::test(start_paused = true)]
    async fn perf_realtime_earlier_vad_item_does_not_end_session() {
        let (done_ms, finals) = session(vec![
            (10, completed("item-1", "first sentence")),
            (40, committed("item-2")),
            (3_000, completed("item-2", "second sentence")),
        ])
        .await;
        assert_eq!(
            finals,
            vec!["first sentence".to_string(), "second sentence".to_string()]
        );
        assert!(done_ms >= 3_000, "session ended before the final item");
        crate::perf_counters::record_metrics(
            "openai-realtime-final-early-exit",
            "vad-earlier-item",
            &[("session_end_ms", done_ms as f64)],
        );
    }
}
