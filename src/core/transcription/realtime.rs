//! Voxtral Realtime transcription via WebSocket.
//!
//! Connects to the Mistral Voxtral Realtime API over WebSocket,
//! streams raw PCM audio, and receives incremental transcription events.

use crate::core::config::MistralConfig;
use crate::core::error::TalkError;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use futures::stream::SplitSink;
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

/// Default model for realtime transcription.
const DEFAULT_REALTIME_MODEL: &str = "voxtral-mini-transcribe-realtime-2602";

/// Default WebSocket base endpoint.
const DEFAULT_REALTIME_ENDPOINT: &str = "wss://api.mistral.ai";

/// WebSocket path for realtime transcription.
const REALTIME_PATH: &str = "/v1/audio/transcriptions/realtime";

/// Events received from the Voxtral Realtime API.
#[derive(Debug, Clone)]
pub enum TranscriptionEvent {
    /// Connection established, session created.
    SessionCreated,
    /// Incremental text delta from the transcription.
    TextDelta { text: String },
    /// Segment with optional timestamps.
    SegmentDelta {
        text: String,
        start: Option<f64>,
        end: Option<f64>,
    },
    /// Detected language.
    Language { language: String },
    /// Transcription complete.
    Done,
    /// Error from the API.
    Error { message: String },
    /// Unknown or malformed event.
    Unknown {
        event_type: Option<String>,
        raw: String,
    },
}

/// Parse a JSON text frame into a `TranscriptionEvent`.
pub fn parse_event(json_str: &str) -> TranscriptionEvent {
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
        Some("session.created") | Some("session.updated") => TranscriptionEvent::SessionCreated,
        Some("transcription.text.delta") => {
            let text = value
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            TranscriptionEvent::TextDelta { text }
        }
        Some("transcription.segment") => {
            let text = value
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let start = value.get("start").and_then(|v| v.as_f64());
            let end = value.get("end").and_then(|v| v.as_f64());
            TranscriptionEvent::SegmentDelta { text, start, end }
        }
        Some("transcription.language") => {
            let language = value
                .get("language")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            TranscriptionEvent::Language { language }
        }
        Some("transcription.done") => TranscriptionEvent::Done,
        Some("error") => {
            let message = value
                .get("error")
                .and_then(|v| v.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error")
                .to_string();
            TranscriptionEvent::Error { message }
        }
        _ => TranscriptionEvent::Unknown {
            event_type: event_type.map(|s| s.to_string()),
            raw: json_str.to_string(),
        },
    }
}

/// Build the WebSocket URL from a base endpoint and model name.
pub fn build_ws_url(endpoint: &str, model: &str) -> String {
    format!("{}{}?model={}", endpoint, REALTIME_PATH, model)
}

/// Convert a slice of `i16` PCM samples to little-endian bytes.
pub fn pcm_to_bytes(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

/// Encode PCM bytes as base64.
pub fn pcm_bytes_to_base64(bytes: &[u8]) -> String {
    BASE64_STANDARD.encode(bytes)
}

/// Realtime transcriber that connects via WebSocket to the Voxtral API.
pub struct MistralRealtimeTranscriber {
    config: MistralConfig,
    endpoint: String,
}

impl MistralRealtimeTranscriber {
    /// Create a new realtime transcriber with the given configuration.
    ///
    /// Uses the default realtime model and endpoint.
    pub fn new(config: MistralConfig) -> Self {
        Self {
            config,
            endpoint: DEFAULT_REALTIME_ENDPOINT.to_string(),
        }
    }

    /// Create a new realtime transcriber with a custom endpoint (for testing).
    #[cfg(test)]
    pub fn with_endpoint(config: MistralConfig, endpoint: String) -> Self {
        Self { config, endpoint }
    }

    /// Return the realtime model name.
    ///
    /// Uses `DEFAULT_REALTIME_MODEL` since the batch model name differs
    /// from the realtime model name.
    fn realtime_model(&self) -> &str {
        DEFAULT_REALTIME_MODEL
    }

    /// Connect to the Voxtral Realtime API and stream audio for transcription.
    ///
    /// Reads `Vec<i16>` PCM chunks from `audio_rx`, encodes them as base64,
    /// and sends them over WebSocket. Returns a receiver of transcription events.
    pub async fn transcribe_realtime(
        &self,
        audio_rx: mpsc::Receiver<Vec<i16>>,
    ) -> Result<mpsc::Receiver<TranscriptionEvent>, TalkError> {
        let ws_url = build_ws_url(&self.endpoint, self.realtime_model());

        // Parse the URL to extract the host for the HTTP header
        let parsed_url = url::Url::parse(&ws_url)
            .map_err(|e| TalkError::Transcription(format!("Invalid WebSocket URL: {}", e)))?;
        let host = parsed_url
            .host_str()
            .ok_or_else(|| TalkError::Transcription("No host in WebSocket URL".to_string()))?
            .to_string();

        // Build the WebSocket upgrade request with auth header
        let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(&ws_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
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

        // Connect to the WebSocket server
        let (ws_stream, _response) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| TalkError::Transcription(format!("WebSocket connection failed: {}", e)))?;

        let (mut ws_sink, mut ws_source) = ws_stream.split();

        // Wait for session.created event
        let session_event = wait_for_session_created(&mut ws_source).await?;
        eprintln!("Realtime session established");

        // Send session.update with audio format
        let session_update = serde_json::json!({
            "type": "session.update",
            "session": {
                "audio_format": {
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000
                }
            }
        });
        ws_sink
            .send(Message::Text(session_update.to_string()))
            .await
            .map_err(|e| {
                TalkError::Transcription(format!("Failed to send session.update: {}", e))
            })?;

        // Create event channel
        let (event_tx, event_rx) = mpsc::channel::<TranscriptionEvent>(100);

        // Send the initial session event through the channel
        let _ = event_tx.send(session_event).await;

        // Spawn sender task: reads PCM from audio_rx, encodes, sends over WS
        let sender_task = tokio::spawn(sender_loop(audio_rx, ws_sink));

        // Spawn receiver task: reads WS messages, parses events, sends to event_tx
        let receiver_task = tokio::spawn(receiver_loop(ws_source, event_tx));

        // Spawn a cleanup task that waits for both to finish
        tokio::spawn(async move {
            let _ = sender_task.await;
            let _ = receiver_task.await;
        });

        Ok(event_rx)
    }
}

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
            let event = parse_event(&text);
            match event {
                TranscriptionEvent::SessionCreated => return Ok(event),
                TranscriptionEvent::Error { ref message } => {
                    return Err(TalkError::Transcription(format!(
                        "Server error during session setup: {}",
                        message
                    )));
                }
                _ => {
                    // Ignore other events during setup
                }
            }
        }
    }

    Err(TalkError::Transcription(
        "WebSocket closed before session.created received".to_string(),
    ))
}

/// Sender loop: reads PCM chunks, base64-encodes, sends as JSON over WebSocket.
async fn sender_loop<S>(mut audio_rx: mpsc::Receiver<Vec<i16>>, mut ws_sink: SplitSink<S, Message>)
where
    S: futures::Sink<Message> + Unpin,
    <S as futures::Sink<Message>>::Error: std::fmt::Display,
{
    while let Some(pcm_chunk) = audio_rx.recv().await {
        let bytes = pcm_to_bytes(&pcm_chunk);
        let b64 = pcm_bytes_to_base64(&bytes);

        let msg = serde_json::json!({
            "type": "input_audio.append",
            "audio": b64
        });

        if let Err(e) = ws_sink.send(Message::Text(msg.to_string())).await {
            eprintln!("WebSocket send error: {}", e);
            break;
        }
    }

    // Audio channel closed — send end-of-audio signal
    let end_msg = serde_json::json!({
        "type": "input_audio.end"
    });
    if let Err(e) = ws_sink.send(Message::Text(end_msg.to_string())).await {
        eprintln!("WebSocket send error (input_audio.end): {}", e);
    }

    let _ = ws_sink.close().await;
}

/// Receiver loop: reads WebSocket messages, parses events, forwards to channel.
async fn receiver_loop<S>(mut ws_source: S, event_tx: mpsc::Sender<TranscriptionEvent>)
where
    S: futures::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    while let Some(msg_result) = ws_source.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                eprintln!("WebSocket receive error: {}", e);
                let _ = event_tx
                    .send(TranscriptionEvent::Error {
                        message: format!("WebSocket error: {}", e),
                    })
                    .await;
                break;
            }
        };

        match msg {
            Message::Text(text) => {
                let event = parse_event(&text);
                let is_terminal = matches!(
                    event,
                    TranscriptionEvent::Done | TranscriptionEvent::Error { .. }
                );
                if event_tx.send(event).await.is_err() {
                    // Receiver dropped
                    break;
                }
                if is_terminal {
                    break;
                }
            }
            Message::Close(_) => {
                let _ = event_tx.send(TranscriptionEvent::Done).await;
                break;
            }
            _ => {
                // Ignore binary, ping, pong frames
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_event_text_delta() {
        let json = r#"{"type": "transcription.text.delta", "text": "hello"}"#;
        let event = parse_event(json);
        match event {
            TranscriptionEvent::TextDelta { text } => assert_eq!(text, "hello"),
            other => panic!("Expected TextDelta, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_event_done() {
        let json = r#"{"type": "transcription.done"}"#;
        let event = parse_event(json);
        assert!(matches!(event, TranscriptionEvent::Done));
    }

    #[test]
    fn test_parse_event_error() {
        let json = r#"{"type": "error", "error": {"message": "bad"}}"#;
        let event = parse_event(json);
        match event {
            TranscriptionEvent::Error { message } => assert_eq!(message, "bad"),
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_event_session_created() {
        let json = r#"{"type": "session.created", "session": {}}"#;
        let event = parse_event(json);
        assert!(matches!(event, TranscriptionEvent::SessionCreated));
    }

    #[test]
    fn test_parse_event_language() {
        let json = r#"{"type": "transcription.language", "language": "en"}"#;
        let event = parse_event(json);
        match event {
            TranscriptionEvent::Language { language } => assert_eq!(language, "en"),
            other => panic!("Expected Language, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_event_segment() {
        let json =
            r#"{"type": "transcription.segment", "text": "hello world", "start": 0.0, "end": 1.5}"#;
        let event = parse_event(json);
        match event {
            TranscriptionEvent::SegmentDelta { text, start, end } => {
                assert_eq!(text, "hello world");
                assert_eq!(start, Some(0.0));
                assert_eq!(end, Some(1.5));
            }
            other => panic!("Expected SegmentDelta, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_event_unknown_type() {
        let json = r#"{"type": "foo"}"#;
        let event = parse_event(json);
        match event {
            TranscriptionEvent::Unknown { event_type, raw } => {
                assert_eq!(event_type, Some("foo".to_string()));
                assert_eq!(raw, json);
            }
            other => panic!("Expected Unknown, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_event_invalid_json() {
        let json = "not json{{";
        let event = parse_event(json);
        match event {
            TranscriptionEvent::Unknown { event_type, raw } => {
                assert!(event_type.is_none());
                assert_eq!(raw, json);
            }
            other => panic!("Expected Unknown, got {:?}", other),
        }
    }

    #[test]
    fn test_build_ws_url() {
        let url = build_ws_url("wss://api.mistral.ai", "my-model");
        assert_eq!(
            url,
            "wss://api.mistral.ai/v1/audio/transcriptions/realtime?model=my-model"
        );
    }

    #[test]
    fn test_build_ws_url_custom_endpoint() {
        let url = build_ws_url("wss://custom.example.com", "test-model");
        assert_eq!(
            url,
            "wss://custom.example.com/v1/audio/transcriptions/realtime?model=test-model"
        );
    }

    #[test]
    fn test_pcm_to_base64() {
        // Two samples: 0x0100 (256) and 0xFF7F (32767)
        let samples: Vec<i16> = vec![256, 32767];
        let bytes = pcm_to_bytes(&samples);
        // 256 in LE = [0x00, 0x01], 32767 in LE = [0xFF, 0x7F]
        assert_eq!(bytes, vec![0x00, 0x01, 0xFF, 0x7F]);

        let b64 = pcm_bytes_to_base64(&bytes);
        // Verify round-trip
        let decoded = BASE64_STANDARD.decode(&b64).expect("valid base64");
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_pcm_to_bytes_empty() {
        let samples: Vec<i16> = vec![];
        let bytes = pcm_to_bytes(&samples);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_pcm_to_bytes_negative() {
        let samples: Vec<i16> = vec![-1, -32768];
        let bytes = pcm_to_bytes(&samples);
        // -1 in LE = [0xFF, 0xFF], -32768 in LE = [0x00, 0x80]
        assert_eq!(bytes, vec![0xFF, 0xFF, 0x00, 0x80]);
    }

    #[test]
    fn test_parse_event_error_missing_message() {
        let json = r#"{"type": "error", "error": {}}"#;
        let event = parse_event(json);
        match event {
            TranscriptionEvent::Error { message } => {
                assert_eq!(message, "unknown error");
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_event_text_delta_missing_text() {
        let json = r#"{"type": "transcription.text.delta"}"#;
        let event = parse_event(json);
        match event {
            TranscriptionEvent::TextDelta { text } => assert_eq!(text, ""),
            other => panic!("Expected TextDelta, got {:?}", other),
        }
    }
}
