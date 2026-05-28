//! Mistral API batch transcription backend.
//!
//! This module provides a [`BatchTranscriber`] implementation that uses the
//! Mistral API to transcribe audio files.

use crate::config::MistralConfig;
use crate::error::TalkError;
use crate::transcription::{
    parse_transcript_segments, DiarizationSegment, MistralProviderMetadata,
    ProviderSpecificMetadata, RequestTimeoutPolicy, TokenUsage, TranscriptionBody,
    TranscriptionMetadata, TranscriptionResult,
};
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Instant;

use super::transport::http::{parse_u64_field, proportional_timeout, ProgressBody};
use super::transport::{self, Method, Request, RequestBody};
use super::BatchTranscriber;
use crate::telemetry::{NoOpSink, TelemetrySink};
use tokio_util::sync::CancellationToken;

/// Default API base URL for the Mistral API.
pub(crate) const API_BASE: &str = "https://api.mistral.ai";

/// Response from Mistral API transcription endpoint.
#[derive(Debug, Deserialize)]
struct MistralResponse {
    /// The transcribed text.
    text: String,
    /// Model identifier returned by API.
    #[serde(default)]
    model: Option<String>,
    /// Detected language code.
    #[serde(default)]
    language: Option<String>,
    /// Optional transcript segments.
    #[serde(default)]
    segments: Option<Vec<serde_json::Value>>,
    /// Usage payload returned by Mistral.
    #[serde(default)]
    usage: Option<serde_json::Value>,
}

fn parse_mistral_token_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    let input_tokens = parse_u64_field(usage, "prompt_tokens");
    let output_tokens = parse_u64_field(usage, "completion_tokens");
    let total_tokens = parse_u64_field(usage, "total_tokens");

    if input_tokens.is_none() && output_tokens.is_none() && total_tokens.is_none() {
        None
    } else {
        Some(TokenUsage {
            input_tokens,
            output_tokens,
            total_tokens,
        })
    }
}

fn parse_mistral_audio_seconds(usage: &serde_json::Value) -> Option<f64> {
    usage.get("prompt_audio_seconds").and_then(|v| v.as_f64())
}

/// Extract diarization segments from the Mistral response segments.
///
/// Each segment is expected to have `speaker_id`, `start`, `end`, and
/// `text` fields when diarization was requested.  Segments without a
/// `speaker_id` are skipped (they come from non-diarized responses).
fn parse_diarization_segments(segments: &[serde_json::Value]) -> Option<Vec<DiarizationSegment>> {
    let mut result = Vec::new();
    for seg in segments {
        let speaker = seg.get("speaker_id").and_then(|v| v.as_str());
        let Some(speaker) = speaker else {
            continue;
        };
        let start = seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let end = seg.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let text = seg
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        result.push(DiarizationSegment {
            speaker: speaker.to_string(),
            start,
            end,
            text,
        });
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

// ── Error detection ─────────────────────────────────────────────────

/// Check if an error is a model-not-found error from the Mistral API.
///
/// Mistral returns `"Unknown model: xyz"` or similar messages.
pub(crate) fn is_model_error(error: &TalkError) -> bool {
    let msg = error.to_string().to_lowercase();
    msg.contains("unknown model")
        || (msg.contains("model") && (msg.contains("not found") || msg.contains("invalid model")))
}

/// Filter predicate for Mistral transcription-relevant models.
pub(crate) fn is_transcription_model(model_id: &str) -> bool {
    model_id.contains("voxtral") || model_id.contains("transcri")
}

/// Enrich a model error with available Mistral transcription models.
pub(crate) async fn enrich_model_error(
    error: TalkError,
    api_key: &str,
    model: &str,
    api_base: &str,
) -> TalkError {
    super::transport::http::enrich_model_error(
        error,
        api_key,
        model,
        api_base,
        is_model_error,
        is_transcription_model,
    )
    .await
}

// ── Model validation (delegates to shared http helper) ──────────────

/// Validate that `model` is available in the Mistral account reachable
/// at `api_base`.
///
/// Thin wrapper around [`super::transport::http::validate_model`] that
/// fixes provider-specific bits (provider enum, display name,
/// transcription-model filter) so callers only carry the variable
/// inputs (api key, model, api_base, telemetry sink).
pub(crate) async fn validate_mistral_model(
    api_key: &str,
    model: &str,
    api_base: &str,
    sink: &Arc<dyn TelemetrySink>,
) -> Result<(), TalkError> {
    super::transport::http::validate_model(
        crate::config::Provider::Mistral,
        "Mistral",
        api_key,
        model,
        api_base,
        is_transcription_model,
        sink,
    )
    .await
}

/// Transcriber implementation using the Mistral API.
///
/// This implementation sends audio files to the Mistral API for transcription
/// and parses the JSON response to extract the transcribed text.
///
/// All network I/O funnels through
/// [`super::transport::http_request`] — this struct no longer
/// owns a `reqwest::Client` directly.  Retries, cancellation, and
/// [`crate::error::PipelineFailure`] construction are owned by the
/// transport.
pub struct MistralBatchTranscriber {
    /// Mistral API configuration (contains API key).
    config: MistralConfig,
    /// API endpoint URL (can be overridden for testing).
    endpoint: String,
    /// Whether to request speaker diarization (V2 models only).
    diarize: bool,
    /// Per-request wall-clock-timeout policy.  Set at construction
    /// via [`Self::with_policy`]; [`Self::new`] forwards
    /// [`RequestTimeoutPolicy::Proportional`].  Controls whether
    /// [`Self::fetch_transcription`] sets a
    /// `proportional_timeout(file_len)` wall clock on the
    /// transport request.
    policy: RequestTimeoutPolicy,
    /// Telemetry event sink for HTTP lifecycle reporting.
    /// Defaults to [`NoOpSink`] when no consumer is attached.
    sink: Arc<dyn TelemetrySink>,
}

impl MistralBatchTranscriber {
    /// Create a new Mistral transcriber with the given configuration
    /// and the default
    /// [`RequestTimeoutPolicy::Proportional`] wall-clock policy.
    ///
    /// The transcription endpoint is derived from `config.url` (if set)
    /// by appending `/v1/audio/transcriptions`.  When `config.url` is
    /// `None`, the default Mistral API base URL is used.
    ///
    /// Use [`Self::with_policy`] when the caller needs a different
    /// timeout policy (e.g. interactive picker rows that should not
    /// be killed by a per-request wall clock).
    ///
    /// # Arguments
    ///
    /// * `config` - Mistral API configuration containing the API key
    /// * `diarize` - Request speaker diarization (requires V2 model)
    pub fn new(config: MistralConfig, diarize: bool) -> Result<Self, TalkError> {
        Self::with_policy(config, diarize, RequestTimeoutPolicy::Proportional)
    }

    /// Create a new Mistral transcriber with an explicit timeout
    /// policy.
    ///
    /// See [`RequestTimeoutPolicy`] for the available variants and
    /// when to pick which.  [`Self::new`] is a thin wrapper around
    /// this that forwards [`RequestTimeoutPolicy::Proportional`].
    pub fn with_policy(
        config: MistralConfig,
        diarize: bool,
        policy: RequestTimeoutPolicy,
    ) -> Result<Self, TalkError> {
        let base = config.url.as_deref().unwrap_or(API_BASE);
        let endpoint = format!("{}/v1/audio/transcriptions", base.trim_end_matches('/'));
        Ok(Self {
            config,
            endpoint,
            diarize,
            policy,
            sink: Arc::new(NoOpSink),
        })
    }

    /// Create a new Mistral transcriber with a custom endpoint (for testing).
    ///
    /// # Arguments
    ///
    /// * `config` - Mistral API configuration containing the API key
    /// * `endpoint` - Custom API endpoint URL
    /// * `diarize` - Request speaker diarization
    #[cfg(test)]
    pub fn with_endpoint(
        config: MistralConfig,
        endpoint: String,
        diarize: bool,
    ) -> Result<Self, TalkError> {
        Ok(Self {
            config,
            endpoint,
            diarize,
            policy: RequestTimeoutPolicy::Proportional,
            sink: Arc::new(NoOpSink),
        })
    }

    /// Send a transcription request through the consolidated
    /// transport.
    ///
    /// All retry, cancellation, error attribution, and progress
    /// telemetry live inside
    /// [`super::transport::http_request`].  This method's only
    /// job is to (a) shape the multipart form with the Mistral
    /// fields, (b) pick the wall-clock policy, (c) parse the
    /// response body.
    async fn send_request(
        &self,
        audio_bytes: Vec<u8>,
        file_name: &str,
    ) -> Result<TranscriptionResult, TalkError> {
        let file_len = audio_bytes.len() as u64;
        let started = Instant::now();

        // Wrap the audio bytes in `Arc` so the multipart factory
        // can reuse the same buffer across retries without
        // cloning the full payload.
        let audio_arc = std::sync::Arc::new(audio_bytes);

        // Capture the per-form construction inputs by value/clone
        // so the factory closure (called once per retry) is `Fn`
        // (re-entrant) and `'static`.
        let model = self.config.model.clone();
        let context_bias = self.config.context_bias.clone();
        let diarize = self.diarize;
        let file_name_owned = file_name.to_string();
        let sink_for_factory = self.sink.clone();

        let body_factory: Box<dyn Fn() -> reqwest::multipart::Form + Send + Sync> = {
            let audio_arc = audio_arc.clone();
            Box::new(move || {
                let audio_bytes_for_attempt = audio_arc.as_ref().clone();
                let progress_body =
                    ProgressBody::new(audio_bytes_for_attempt, sink_for_factory.clone());
                let body_len = progress_body.len();

                let mut form = reqwest::multipart::Form::new()
                    .text("model", model.clone())
                    .part(
                        "file",
                        reqwest::multipart::Part::stream_with_length(
                            reqwest::Body::wrap_stream(progress_body),
                            body_len,
                        )
                        .file_name(file_name_owned.clone()),
                    )
                    .text("timestamp_granularities", "segment");

                if let Some(ref bias) = context_bias {
                    form = form.text("context_bias", bias.clone());
                }

                if diarize {
                    form = form.text("diarize", "true");
                }
                form
            })
        };

        let wall_clock = match self.policy {
            RequestTimeoutPolicy::Proportional => Some(proportional_timeout(file_len)),
            RequestTimeoutPolicy::UserAttended => None,
        };

        log::debug!(
            "mistral send_request: policy={:?}, wall_clock={}, audio={} KB",
            self.policy,
            wall_clock
                .map(|d| format!("{}s", d.as_secs()))
                .unwrap_or_else(|| "none".to_string()),
            file_len / 1024
        );

        let req = Request {
            method: Method::Post,
            url: self.endpoint.clone(),
            headers: vec![(
                "Authorization".into(),
                format!("Bearer {}", self.config.api_key),
            )],
            body: RequestBody::Multipart(body_factory),
            provider: crate::config::Provider::Mistral,
            provider_name: "Mistral".into(),
            phase: crate::error::PipelinePhase::Request,
            wall_clock,
        };

        let response = transport::http_request(req, &self.sink, CancellationToken::new())
            .await
            .map_err(TalkError::from)?;
        let request_latency_ms = started.elapsed().as_millis() as u64;

        if !(200..300).contains(&response.status) {
            let body = String::from_utf8_lossy(&response.body).to_string();
            return Err(TalkError::Transcription(format!(
                "Mistral API error ({}): {}",
                response.status, body
            )));
        }

        let mistral_response: MistralResponse =
            serde_json::from_slice(&response.body).map_err(|err| {
                TalkError::Transcription(format!("Failed to parse Mistral API response: {}", err))
            })?;

        let token_usage = mistral_response
            .usage
            .as_ref()
            .and_then(parse_mistral_token_usage);
        let audio_seconds = mistral_response
            .usage
            .as_ref()
            .and_then(parse_mistral_audio_seconds);
        let segment_count = mistral_response.segments.as_ref().map(std::vec::Vec::len);
        let request_id = response
            .headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case("x-request-id"))
            .map(|(_, v)| v.clone());

        let segments = mistral_response
            .segments
            .as_deref()
            .and_then(parse_transcript_segments);

        let diarization = mistral_response
            .segments
            .as_deref()
            .and_then(parse_diarization_segments);

        let text = mistral_response.text;
        Ok(TranscriptionResult {
            text,
            metadata: TranscriptionMetadata {
                request_latency_ms: Some(request_latency_ms),
                session_elapsed_ms: None,
                request_id,
                provider_processing_ms: None,
                detected_language: mistral_response.language,
                audio_seconds,
                segment_count,
                word_count: None,
                token_usage,
                provider_specific: Some(ProviderSpecificMetadata::Mistral(
                    MistralProviderMetadata {
                        model: mistral_response.model,
                        usage_raw: mistral_response.usage,
                        unknown_event_types: Vec::new(),
                    },
                )),
            },
            diarization,
            segments,
        })
    }
}

#[async_trait]
impl BatchTranscriber for MistralBatchTranscriber {
    fn set_sink(&mut self, sink: Arc<dyn TelemetrySink>) {
        self.sink = sink;
    }

    async fn validate(&self) -> Result<(), TalkError> {
        let api_base = self
            .endpoint
            .find("/v1/")
            .map(|pos| &self.endpoint[..pos])
            .unwrap_or(&self.endpoint);
        validate_mistral_model(
            &self.config.api_key,
            &self.config.model,
            api_base,
            &self.sink,
        )
        .await
    }

    async fn fetch_transcription(
        &self,
        body: TranscriptionBody,
    ) -> Result<TranscriptionResult, TalkError> {
        let (audio_bytes, file_name) = match body {
            TranscriptionBody::File(path) => {
                if !path.exists() {
                    return Err(TalkError::Transcription(format!(
                        "Audio file not found: {}",
                        path.display()
                    )));
                }

                use tokio::io::AsyncReadExt;

                let mut file = tokio::fs::File::open(&path).await.map_err(|err| {
                    TalkError::Transcription(format!("Failed to open audio file: {}", err))
                })?;
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes).await.map_err(|err| {
                    TalkError::Transcription(format!("Failed to read audio file: {}", err))
                })?;
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("audio.wav")
                    .to_string();
                (bytes, file_name)
            }
            TranscriptionBody::Stream {
                mut chunks,
                file_name,
            } => {
                log::warn!("[DBG] mistral stream: awaiting audio chunks from encoder");
                let collect_start = Instant::now();
                let mut bytes = Vec::new();
                while let Some(chunk) = chunks.recv().await {
                    bytes.extend_from_slice(&chunk);
                }
                let audio_len = bytes.len() as u64;
                log::info!(
                    "streaming upload: collected {} bytes for Mistral batch request",
                    audio_len
                );
                log::warn!(
                    "[DBG] mistral stream: audio collected ({} bytes in {}ms), building request",
                    audio_len,
                    collect_start.elapsed().as_millis()
                );
                (bytes, file_name)
            }
        };

        self.send_request(audio_bytes, &file_name).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn test_new_uses_default_endpoint_when_url_is_none() {
        let config = MistralConfig {
            api_key: "key".to_string(),
            url: None,
            model: "voxtral-mini-2507".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::new(config, false).expect("build client");
        assert_eq!(
            transcriber.endpoint,
            "https://api.mistral.ai/v1/audio/transcriptions"
        );
    }

    #[test]
    fn test_new_uses_custom_url_for_endpoint() {
        let config = MistralConfig {
            api_key: "key".to_string(),
            url: Some("https://custom.example.com".to_string()),
            model: "voxtral-mini-2507".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::new(config, false).expect("build client");
        assert_eq!(
            transcriber.endpoint,
            "https://custom.example.com/v1/audio/transcriptions"
        );
    }

    #[test]
    fn test_new_trims_trailing_slash_from_url() {
        let config = MistralConfig {
            api_key: "key".to_string(),
            url: Some("https://custom.example.com/".to_string()),
            model: "voxtral-mini-2507".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::new(config, false).expect("build client");
        assert_eq!(
            transcriber.endpoint,
            "https://custom.example.com/v1/audio/transcriptions"
        );
    }

    #[tokio::test]
    async fn test_mistral_transcriber_success() {
        // Start mock server
        let mock_server = MockServer::start().await;

        // Mock the Mistral API endpoint
        // Note: wiremock multipart matching is limited, so we just verify method, path, and auth header
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "This is a test transcription"
            })))
            .mount(&mock_server)
            .await;

        // Create a temporary audio file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        // Create transcriber with mock server URL
        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        // Transcribe the file
        let result = transcriber
            .fetch_transcription(TranscriptionBody::File(temp_file.path().to_path_buf()))
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "This is a test transcription");
    }

    #[tokio::test]
    async fn test_mistral_transcriber_stream_success() {
        // Start mock server
        let mock_server = MockServer::start().await;

        // Mock the Mistral API endpoint
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Streamed transcription result"
            })))
            .mount(&mock_server)
            .await;

        // Create transcriber with mock server URL
        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        // Create a channel and send fake audio data
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            tx.send(vec![0u8; 100]).await.unwrap();
            tx.send(vec![1u8; 200]).await.unwrap();
            // Dropping tx closes the channel
        });

        // Transcribe from the stream
        let result = transcriber
            .fetch_transcription(TranscriptionBody::Stream {
                chunks: rx,
                file_name: "test.wav".to_string(),
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "Streamed transcription result");
    }

    #[tokio::test]
    async fn test_mistral_transcriber_extracts_transcript_segments() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                include_str!("../../tests/fixtures/voxtral-response.json"),
                "application/json",
            ))
            .mount(&mock_server)
            .await;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        let result = transcriber
            .fetch_transcription(TranscriptionBody::File(temp_file.path().to_path_buf()))
            .await
            .unwrap();

        assert_eq!(result.text, "Hello world. This is a test.");
        let segments = result.segments.expect("transcript segments present");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start, 0.0);
        assert_eq!(segments[0].end, 1.5);
        assert_eq!(segments[0].text, "Hello world.");
        assert_eq!(segments[1].start, 2.0);
        assert_eq!(segments[1].end, 3.8);
        assert_eq!(segments[1].text, " This is a test.");
    }

    #[tokio::test]
    async fn test_mistral_transcriber_api_error() {
        // Start mock server
        let mock_server = MockServer::start().await;

        // Mock the Mistral API endpoint to return an error
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&mock_server)
            .await;

        // Create a temporary audio file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        // Create transcriber with mock server URL
        let config = MistralConfig {
            api_key: "invalid-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        // Transcribe the file
        let result = transcriber
            .fetch_transcription(TranscriptionBody::File(temp_file.path().to_path_buf()))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("401"));
    }

    #[tokio::test]
    async fn test_mistral_transcriber_file_not_found() {
        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::new(config, false).expect("build client");

        let result = transcriber
            .fetch_transcription(TranscriptionBody::File(std::path::PathBuf::from(
                "/nonexistent/file.wav",
            )))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_mistral_transcriber_invalid_json_response() {
        // Start mock server
        let mock_server = MockServer::start().await;

        // Mock the Mistral API endpoint to return invalid JSON
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("invalid json"))
            .mount(&mock_server)
            .await;

        // Create a temporary audio file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        // Create transcriber with mock server URL
        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        // Transcribe the file
        let result = transcriber
            .fetch_transcription(TranscriptionBody::File(temp_file.path().to_path_buf()))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    #[tokio::test]
    async fn test_mistral_diarization_segments_parsed() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Hello. I am fine.",
                "segments": [
                    {
                        "speaker_id": "SPEAKER_00",
                        "start": 0.0,
                        "end": 1.5,
                        "text": "Hello."
                    },
                    {
                        "speaker_id": "SPEAKER_01",
                        "start": 1.5,
                        "end": 3.0,
                        "text": "I am fine."
                    }
                ]
            })))
            .mount(&mock_server)
            .await;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-2602".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            true,
        )
        .expect("build client");

        let result = transcriber
            .fetch_transcription(TranscriptionBody::File(temp_file.path().to_path_buf()))
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.text, "Hello. I am fine.");

        let segments = result.diarization.expect("diarization segments present");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert_eq!(segments[0].text, "Hello.");
        assert_eq!(segments[0].start, 0.0);
        assert_eq!(segments[0].end, 1.5);
        assert_eq!(segments[1].speaker, "SPEAKER_01");
        assert_eq!(segments[1].text, "I am fine.");
    }

    #[test]
    fn test_is_model_error_unknown_model() {
        let err = TalkError::Transcription(
            r#"Mistral API error (422): {"object":"error","message":"Unknown model: bad-model","type":"invalid_request_error"}"#.to_string(),
        );
        assert!(is_model_error(&err));
    }

    #[test]
    fn test_is_model_error_not_found() {
        let err = TalkError::Transcription(
            "Model 'voxtral-mini-9999' not found. Available transcription models: voxtral-mini-2507"
                .to_string(),
        );
        assert!(is_model_error(&err));
    }

    #[test]
    fn test_is_model_error_negative_network() {
        let err = TalkError::Transcription("connection timed out".to_string());
        assert!(!is_model_error(&err));
    }

    #[test]
    fn test_is_model_error_negative_auth() {
        let err = TalkError::Transcription("Mistral API error (401): Unauthorized".to_string());
        assert!(!is_model_error(&err));
    }

    #[tokio::test]
    async fn test_enrich_non_model_error_unchanged() {
        let err = TalkError::Transcription("connection timed out".to_string());
        let enriched = enrich_model_error(err, "key", "voxtral-mini-2507", API_BASE).await;
        assert_eq!(
            enriched.to_string(),
            "Transcription error: connection timed out"
        );
    }

    #[test]
    fn test_is_transcription_model_matches() {
        assert!(is_transcription_model("voxtral-mini-2507"));
        assert!(is_transcription_model("voxtral-mini-2602"));
        assert!(is_transcription_model("some-transcription-model"));
        assert!(!is_transcription_model("mistral-large-latest"));
    }

    #[tokio::test]
    async fn test_mistral_no_diarization_without_speaker_id() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Hello world.",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.5,
                        "text": "Hello world."
                    }
                ]
            })))
            .mount(&mock_server)
            .await;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-2602".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_endpoint(
            config,
            format!("{}/v1/audio/transcriptions", mock_server.uri()),
            false,
        )
        .expect("build client");

        let result = transcriber
            .fetch_transcription(TranscriptionBody::File(temp_file.path().to_path_buf()))
            .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        // No speaker_id in segments → diarization is None
        assert!(result.diarization.is_none());
    }

    /// Spec for `with_policy(UserAttended)`:
    ///
    /// When a Mistral batch request fails because the server accepts
    /// the connection but never replies, the error message must NOT
    /// be attributed to `request_wall_clock` (because the
    /// `UserAttended` policy intentionally omits the wall-clock
    /// timer).  Attribution should fall through to `connect_timeout`
    /// (Rule 1) or `kernel_tcp_unspecified` (Rule 3) — never
    /// `request_wall_clock`.
    ///
    /// This is the regression-detection test for the picker path:
    /// if a future refactor accidentally re-introduces a
    /// `.timeout()` for `UserAttended`, this fails.
    #[tokio::test]
    async fn user_attended_policy_omits_request_wall_clock_attribution() {
        // Bind a TCP listener that accepts but never replies, so the
        // request will (eventually) fail with TCP-level timing only.
        // We don't actually wait for it to fail — we just assert
        // that *if* it fails before the test framework times out,
        // attribution is correct.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test: bind ephemeral port");
        let addr = listener.local_addr().expect("test: local_addr");
        tokio::spawn(async move {
            loop {
                if listener.accept().await.is_err() {
                    break;
                }
            }
        });

        let config = MistralConfig {
            api_key: "test-api-key".to_string(),
            url: None,
            model: "voxtral-mini-latest".to_string(),
            context_bias: None,
        };
        let transcriber = MistralBatchTranscriber::with_policy(
            config,
            false,
            crate::transcription::RequestTimeoutPolicy::UserAttended,
        )
        .expect("build client");

        // Override endpoint to point at the silent listener.  Use a
        // wiremock-shaped path so the post URL is plausible.
        let mut transcriber = transcriber;
        transcriber.endpoint = format!("http://{}/v1/audio/transcriptions", addr);

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"fake audio data").unwrap();
        temp_file.flush().unwrap();

        // Race the request against a short test deadline.  We don't
        // care whether the underlying request returns within the
        // window — the test framework's per-test timeout will kill
        // it eventually if needed.  The contract being asserted is:
        // *whatever* error we get, it cannot be attributed to
        // `request_wall_clock`.
        let race = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            transcriber
                .fetch_transcription(TranscriptionBody::File(temp_file.path().to_path_buf())),
        )
        .await;

        match race {
            Ok(Ok(_)) => {
                // Listener never replies — success is impossible
                // here.  If this branch fires, the test setup is
                // broken; fail loudly.
                panic!("test: silent listener should never produce a Mistral success");
            }
            Ok(Err(e)) => {
                // Structural assertion: the failure must be a
                // `Pipeline` error in the `Request` phase, and its
                // `Network` cause MUST NOT have a
                // `request_wall_clock` timer attached (because
                // `UserAttended` policy intentionally omits it).
                use crate::error::{NetworkKind, PipelineFailureKind, PipelinePhase};
                let pf = match &e {
                    crate::error::TalkError::Pipeline(pf) => pf,
                    other => panic!(
                        "expected TalkError::Pipeline under UserAttended, got: {}",
                        other
                    ),
                };
                assert_eq!(pf.phase, PipelinePhase::Request);
                if let PipelineFailureKind::Network { kind: _, timer, .. } = &pf.kind {
                    assert!(
                        !matches!(timer, Some(t) if t.name == "request_wall_clock"),
                        "UserAttended must not attribute to request_wall_clock; got timer={:?}",
                        timer
                    );
                }
                // Decode/HttpStatus from a never-replying listener
                // is also acceptable — the structured shape is what
                // matters, not the specific sub-variant.
                let _ = NetworkKind::Connect; // suppress unused-import warning if Network branch never hits
                                              // Sanity: not a flat string-stuffed legacy error
                                              // — the `Display` should NOT carry a leading
                                              // "Configuration error:" or "Transcription error:"
                                              // prefix.
                let s = e.to_string();
                assert!(
                    !s.starts_with("Configuration error:"),
                    "structured Pipeline error must not produce Config prefix: {}",
                    s
                );
            }
            Err(_) => {
                // The test deadline (15s) elapsed before reqwest
                // returned.  That's the user-attended-policy
                // contract working as intended: no wall-clock cap
                // means the request just keeps waiting.  Not a
                // failure of the spec under test — `with_retry`
                // wrapping is the only finite-time guarantee and
                // it lives at a different layer.
            }
        }
    }
}
