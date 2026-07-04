//! Transcription interfaces and implementations.
//!
//! This module provides traits and implementations for transcribing audio
//! to text using various backends (Mistral, OpenAI).
//!
//! Two traits model the two modes of operation:
//!
//! - [`OneShotTranscriber`]: file or byte-stream in, full text out.
//! - [`RealtimeTranscriber`]: raw PCM stream in, incremental event stream out.

use crate::config::{Config, Provider};
use crate::error::TalkError;
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Audio format for one-shot uploads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum UploadFormat {
    /// Uncompressed WAV (default).
    #[default]
    Wav,
    /// OGG Opus compressed audio (smaller uploads).
    Ogg,
}

/// Body source for a one-shot transcription request.
pub(crate) enum TranscriptionBody {
    /// Full audio file on disk. The transport reads it.
    File(PathBuf),
    /// Chunks arriving through a channel — used during
    /// record-while-upload streaming. The transport collects them.
    //
    // The `Pipe` variant is only *constructed* by the live
    // record-while-upload path (`dictate`), which requires `capture`.
    // In a build without `capture` the variant is only ever pattern-
    // matched (e.g. by the parakeet backend), never constructed — but
    // it remains part of the transcription-transport contract, hence
    // the scoped allow.
    #[cfg_attr(not(feature = "capture"), allow(dead_code))]
    Pipe {
        chunks: tokio::sync::mpsc::Receiver<Vec<u8>>,
        file_name: String,
    },
}

/// Normalize an on-disk audio file into the API-optimal upload form:
/// **16 kHz mono OGG/Opus**.
///
/// Both transcription providers (Mistral Voxtral and OpenAI Whisper /
/// gpt-4o-transcribe) operate internally at 16 kHz mono — they
/// downsample and downmix any input server-side, and accept nothing
/// above 8 kHz audio bandwidth.  Uploading the original (potentially
/// 48 kHz stereo, e.g. a high-fidelity `record` output) wastes
/// bandwidth, increases latency, and can hit OpenAI's 25 MB cap, all
/// for zero accuracy benefit.  So we always re-encode to 16 kHz mono
/// before sending.
///
/// Returns `(bytes, file_name)` where `file_name` carries an `.ogg`
/// extension so the multipart upload advertises the correct format.
///
/// On any decode/encode failure (e.g. an exotic container we cannot
/// decode locally), this falls back to the original file bytes with a
/// warning, so an upload that worked before this normalization existed
/// keeps working.
pub(crate) fn normalize_file_for_upload(
    path: &std::path::Path,
) -> Result<(Vec<u8>, String), TalkError> {
    if !path.exists() {
        return Err(TalkError::Transcription(format!(
            "Audio file not found: {}",
            path.display()
        )));
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("audio");
    let normalized_name = format!("{stem}.ogg");

    match encode_16k_mono_ogg(path) {
        Ok(bytes) => {
            log::info!(
                "upload normalization: {} -> 16kHz mono ogg ({} bytes)",
                path.display(),
                bytes.len()
            );
            Ok((bytes, normalized_name))
        }
        Err(err) => {
            // Never break a previously-working upload: fall back to the
            // raw file bytes (with the original file name).
            log::warn!(
                "upload normalization failed for {} ({}); uploading original bytes",
                path.display(),
                err
            );
            let bytes = std::fs::read(path)
                .map_err(|e| TalkError::Transcription(format!("Failed to read audio file: {e}")))?;
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("audio.wav")
                .to_string();
            Ok((bytes, file_name))
        }
    }
}

/// Decode any supported audio file to 16 kHz mono PCM and re-encode it
/// as an OGG/Opus byte stream (16 kHz mono).
fn encode_16k_mono_ogg(path: &std::path::Path) -> Result<Vec<u8>, TalkError> {
    use crate::audio::{AudioWriter, OggOpusWriter};

    // `read_audio_as_i16` decodes wav/ogg/opus/m4a/mp4/aac, resampling
    // to 16 kHz and downmixing to mono — exactly the target form.
    let pcm = crate::record::audio::read_audio_as_i16(path)?;

    // 16 kHz mono is the hardcoded transcription profile.
    let mut writer = OggOpusWriter::new(crate::config::AudioConfig::new())?;
    let mut out = writer.header()?;
    out.extend_from_slice(&writer.write_pcm(&pcm)?);
    out.extend_from_slice(&writer.finalize()?);
    Ok(out)
}

/// Per-request wall-clock-timeout policy for one-shot transcription.
///
/// Two distinct call contexts demand opposite defaults:
///
/// - Autonomous pipelines (dictate end-of-recording, `transcribe`
///   CLI) MUST not hang forever — there is no human watching to
///   abort.  These callers pick [`Self::Proportional`].
/// - Interactive callers (the GTK picker row) prefer "wait as long
///   as it takes" — the user can dismiss the picker if a candidate
///   is taking forever.  These callers pick [`Self::UserAttended`].
///
/// The policy is stored on each [`OneShotTranscriber`] (chosen at
/// construction via `with_policy`) and consulted by `send_once`
/// when it issues the HTTP request.  In both cases the
/// `connect_timeout` from `build_client()` and the kernel-level TCP
/// defences (TCP keepalive, `tcp_user_timeout` on Linux) still
/// fire, so dead connections cannot hang either path indefinitely
/// — only legitimately slow servers benefit from `UserAttended`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestTimeoutPolicy {
    /// Cap each attempt at `proportional_timeout(file_len)` so an
    /// unattended pipeline cannot hang forever.  This is the legacy
    /// behaviour and remains the default of [`MistralOneShotTranscriber::new`]
    /// / [`OpenAIOneShotTranscriber::new`].
    #[default]
    Proportional,
    /// No per-request wall-clock cap; rely solely on
    /// `connect_timeout` plus TCP-level defences.  Used by the
    /// picker so a slow-responding server can take its time without
    /// being killed mid-transcription.  Cancellation is the user's
    /// responsibility (close the picker, or — once wired — a
    /// per-row cancel button).
    UserAttended,
}

/// Caller-controlled options for [`transcribe_audio`].
///
/// Bundles the orthogonal axes that affect a single transcription
/// call so the function signature stays readable as new axes
/// appear.  Each field is a typed concept that means something
/// specific at the call site:
///
/// - `allow_api`: whether the cache-miss path may hit the network.
///   `false` = cache-only probe (returns [`TalkError::CacheOnly`]
///   on miss); `true` = full pipeline with API call.
/// - `policy`: per-request wall-clock policy.  See
///   [`RequestTimeoutPolicy`].
///
/// Construct via the public fields directly — there is no builder.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// Permit network I/O on cache miss.  `false` makes the call
    /// cache-only (returns [`TalkError::CacheOnly`] on miss);
    /// `true` runs the full pipeline including API.
    pub allow_api: bool,
    /// Per-request wall-clock-timeout policy.  See
    /// [`RequestTimeoutPolicy`] for variant semantics.
    pub policy: RequestTimeoutPolicy,
    /// Cancellation token to thread into the transport's request
    /// loop.  When the token fires, the in-flight HTTP request
    /// aborts within milliseconds.  Defaults to `None` (no
    /// cancellation wired); the picker / streaming orchestrator
    /// supplies a real token when it wants the user to be able to
    /// stop the request.
    pub cancel_token: Option<tokio_util::sync::CancellationToken>,
    /// Skip `acquire_model_lock`/`release_model_lock` (the legacy
    /// per-model lock).  Used by call sites that already register
    /// the job via [`crate::transcription::jobs::register_local`]
    /// (which writes the same lock file with a richer YAML
    /// payload) so we don't conflict-on-self.
    pub skip_legacy_lock: bool,
}

pub mod jobs;
pub mod mistral;
pub mod model_suggestions;
pub mod openai;
pub mod openai_realtime;
#[cfg(feature = "parakeet")]
pub mod parakeet;
pub mod realtime;
pub mod transport;

pub use mistral::MistralOneShotTranscriber;
pub use openai::OpenAIOneShotTranscriber;
pub use openai_realtime::OpenAIRealtimeTranscriber;
#[cfg(feature = "parakeet")]
pub use parakeet::ParakeetOneShotTranscriber;
pub use realtime::{MistralRealtimeTranscriber, TranscriptionEvent};

/// Result type for transcription operations.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionResult {
    /// The transcribed text from the audio file.
    pub text: String,
    /// Optional metadata extracted from provider responses/headers.
    pub metadata: TranscriptionMetadata,
    /// Speaker-diarized segments, when diarization was requested and
    /// supported by the provider.
    pub diarization: Option<Vec<DiarizationSegment>>,
    /// Time-localized transcript segments, when the provider returned
    /// them.  Independent of diarization — a diarized response may
    /// populate both `diarization` and `segments`.  Used to reconstruct
    /// timelines in downstream consumers such as `activity-memo`.
    pub segments: Option<Vec<TranscriptSegment>>,
}

/// A single speaker-attributed segment from diarization.
///
/// Provider-agnostic: each provider maps its native speaker labels
/// into this structure.
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizationSegment {
    /// Speaker identifier (e.g. `"SPEAKER_00"`).
    pub speaker: String,
    /// Segment start time in seconds.
    pub start: f64,
    /// Segment end time in seconds.
    pub end: f64,
    /// Transcribed text for this segment.
    pub text: String,
}

/// A time-localized transcript segment without speaker attribution.
///
/// Provider-agnostic: each one-shot/realtime backend maps its native
/// segment representation into this shape.  Unlike `DiarizationSegment`,
/// a `TranscriptSegment` does not carry a speaker identifier — it is
/// just a start/end window around a piece of text, suitable for
/// timeline reconstruction and sub-minute granularity in consumers.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptSegment {
    /// Segment start time in seconds, from the beginning of the recording.
    pub start: f64,
    /// Segment end time in seconds, from the beginning of the recording.
    pub end: f64,
    /// Transcribed text for this segment.
    pub text: String,
}

/// Extract generic transcript segments from a raw provider response.
///
/// Accepts a slice of `serde_json::Value` (as returned by both Mistral
/// Voxtral and OpenAI Whisper under `verbose_json`) and pulls out
/// `start`, `end`, and `text` from every segment that has them.
/// Segments with missing timing or empty text are skipped.  Returns
/// `None` when no usable segments survive the filter so that call
/// sites can leave `TranscriptionResult.segments` as `None` rather
/// than `Some(vec![])`.
pub(crate) fn parse_transcript_segments(
    segments: &[serde_json::Value],
) -> Option<Vec<TranscriptSegment>> {
    let mut result = Vec::new();
    for seg in segments {
        let start = seg.get("start").and_then(|v| v.as_f64());
        let end = seg.get("end").and_then(|v| v.as_f64());
        let text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("");
        if let (Some(start), Some(end)) = (start, end) {
            if !text.is_empty() {
                result.push(TranscriptSegment {
                    start,
                    end,
                    text: text.to_string(),
                });
            }
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Format transcription output, using diarized segments when available.
///
/// When diarization segments are present, each line is prefixed with
/// `[SPEAKER_ID]`.  Adjacent segments from the same speaker are merged
/// into a single block.  When no diarization is present, returns the
/// plain transcript text.
///
/// When `timestamp` is true and diarization is present, each line is
/// prefixed with `[HH:MM:SS]` before the speaker label.
pub fn format_transcription_output(result: &TranscriptionResult, timestamp: bool) -> String {
    let Some(ref segments) = result.diarization else {
        return result.text.clone();
    };

    if segments.is_empty() {
        return result.text.clone();
    }

    let mut lines = Vec::new();
    let mut current_speaker: Option<&str> = None;
    let mut current_texts: Vec<&str> = Vec::new();
    let mut current_start: f64 = 0.0;

    for seg in segments {
        if current_speaker == Some(seg.speaker.as_str()) {
            current_texts.push(seg.text.trim());
        } else {
            // Flush previous speaker block
            if let Some(speaker) = current_speaker {
                if timestamp {
                    lines.push(format!(
                        "[{}] {} {}",
                        format_timestamp(current_start),
                        speaker,
                        current_texts.join(" ")
                    ));
                } else {
                    lines.push(format!("[{}] {}", speaker, current_texts.join(" ")));
                }
            }
            current_speaker = Some(&seg.speaker);
            current_start = seg.start;
            current_texts.clear();
            current_texts.push(seg.text.trim());
        }
    }
    // Flush last block
    if let Some(speaker) = current_speaker {
        if timestamp {
            lines.push(format!(
                "[{}] {} {}",
                format_timestamp(current_start),
                speaker,
                current_texts.join(" ")
            ));
        } else {
            lines.push(format!("[{}] {}", speaker, current_texts.join(" ")));
        }
    }

    lines.join("\n")
}

/// Format seconds as HH:MM:SS.
fn format_timestamp(seconds: f64) -> String {
    let total_secs = seconds as u64;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, secs)
}

/// Provider-agnostic metadata that can be written to YAML.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionMetadata {
    /// End-to-end API call latency measured client-side.
    pub request_latency_ms: Option<u64>,
    /// End-to-end realtime session duration measured client-side.
    pub session_elapsed_ms: Option<u64>,
    /// Request identifier from provider response headers.
    pub request_id: Option<String>,
    /// Provider-side processing duration in milliseconds when available.
    pub provider_processing_ms: Option<u64>,
    /// Detected language code if returned by provider.
    pub detected_language: Option<String>,
    /// Audio duration reported by provider usage/response.
    pub audio_seconds: Option<f64>,
    /// Number of transcript segments returned by provider.
    pub segment_count: Option<usize>,
    /// Number of word-level timestamps returned by provider.
    pub word_count: Option<usize>,
    /// Token usage summary when available.
    pub token_usage: Option<TokenUsage>,
    /// Provider-specific payload for advanced diagnostics.
    pub provider_specific: Option<ProviderSpecificMetadata>,
}

/// Common token usage summary.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Provider-specific metadata captured from API responses.
#[derive(Debug, Clone)]
pub enum ProviderSpecificMetadata {
    OpenAI(OpenAIProviderMetadata),
    Mistral(MistralProviderMetadata),
}

/// OpenAI-specific metadata.
#[derive(Debug, Clone, Default)]
pub struct OpenAIProviderMetadata {
    pub model: Option<String>,
    pub usage_raw: Option<serde_json::Value>,
    pub rate_limit_headers: BTreeMap<String, String>,
    pub unknown_event_types: Vec<String>,
    pub realtime: Option<OpenAIRealtimeMetadata>,
}

/// OpenAI realtime-specific metadata.
#[derive(Debug, Clone, Default)]
pub struct OpenAIRealtimeMetadata {
    pub session_id: Option<String>,
    pub conversation_id: Option<String>,
    pub event_counts: BTreeMap<String, u64>,
    pub last_rate_limits: Option<serde_json::Value>,
    pub ws_upgrade_headers: BTreeMap<String, String>,
}

/// Mistral-specific metadata.
#[derive(Debug, Clone, Default)]
pub struct MistralProviderMetadata {
    pub model: Option<String>,
    pub usage_raw: Option<serde_json::Value>,
    pub unknown_event_types: Vec<String>,
}

// ── One-shot trait ──────────────────────────────────────────────────

/// One-shot transcription: file or byte-stream in, full text out.
///
/// Implementations should handle file I/O, API communication, and error
/// handling.  All implementations must be `Send + Sync` for use in async
/// contexts.
#[async_trait]
pub(crate) trait OneShotTranscriber: Send + Sync {
    /// Pre-flight check: verify API connectivity and model validity.
    ///
    /// Called before starting audio capture so the user gets immediate
    /// feedback when a provider is misconfigured or a model name is
    /// invalid.  Implementations should make a lightweight API call
    /// (e.g. list available models) and return a helpful error with
    /// available alternatives on failure.
    async fn validate(&self) -> Result<(), TalkError>;

    /// Fetch a transcription from either a file or an encoded stream.
    async fn fetch_transcription(
        &self,
        body: TranscriptionBody,
    ) -> Result<TranscriptionResult, TalkError>;

    /// Inject a telemetry sink for event emission during HTTP calls.
    ///
    /// The default implementation is a no-op — override in concrete
    /// types that support telemetry.  Called by the orchestrator
    /// (`dictate/mod.rs`) right after construction, before any
    /// `fetch_transcription` is invoked.
    fn set_sink(&mut self, _sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink>) {}

    /// Inject a cancellation token for the transcription request.
    ///
    /// When the token fires, the in-flight ``http_request`` aborts
    /// within milliseconds.  Used by the picker Stop button and by
    /// SIGUSR1-routed cross-process cancellation via
    /// [`crate::transcription::jobs::cancel_remote`].
    fn set_cancel_token(&mut self, _token: tokio_util::sync::CancellationToken) {}
}

// ── Realtime trait ───────────────────────────────────────────────────

/// Realtime transcription: raw PCM stream in, incremental events out.
//
// The trait is driven exclusively by the live dictation path, which
// requires both `capture` and `ui`.  In a headless build the trait
// methods are never invoked (the concrete realtime transcribers still
// expose inherent `transcribe_realtime` methods for library
// consumers), so the dead-code allow is scoped to that configuration.
#[cfg_attr(not(all(feature = "capture", feature = "ui")), allow(dead_code))]
#[async_trait]
pub(crate) trait RealtimeTranscriber: Send + Sync {
    /// Pre-flight check: verify API connectivity and model validity.
    ///
    /// Same purpose as [`OneShotTranscriber::validate`] — called before
    /// starting audio capture to give the user immediate feedback.
    #[allow(dead_code)]
    async fn validate(&self) -> Result<(), TalkError>;

    /// Connect and start streaming.  Returns a channel of events.
    async fn transcribe_realtime(
        &self,
        audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    ) -> Result<tokio::sync::mpsc::Receiver<TranscriptionEvent>, TalkError>;

    /// Inject a telemetry sink for event emission during the WS
    /// upgrade phase (and, later, the in-session frame events).
    fn set_sink(&mut self, _sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink>) {}

    /// Inject a cancellation token for the realtime session.
    ///
    /// When the token fires, the in-flight WebSocket upgrade or
    /// active session aborts.  See [`OneShotTranscriber::set_cancel_token`]
    /// for the wiring rationale.
    fn set_cancel_token(&mut self, _token: tokio_util::sync::CancellationToken) {}
}

// ── Error detection / enrichment dispatchers ────────────────────────

/// Check if an error is a model-not-found error for the given provider.
///
/// Detection order:
///
/// 1. **Structural fast path** — if the error is a
///    [`TalkError::Pipeline`] carrying a
///    [`crate::error::PipelineFailureKind::ModelRejected`], return
///    `true` regardless of provider.  Producers (validate-path
///    HTTP code) emit this variant directly, so consumers can
///    detect model-rejection without parsing strings.
/// 2. **Legacy string-matching fallback** — for any pre-migration
///    call sites that still surface model-not-found as
///    `TalkError::Config(String)` / `TalkError::Transcription(String)`,
///    dispatch to the provider's own detection logic.
///
/// The structural path runs first so a future refactor cannot
/// accidentally regress retry/bail behaviour by skipping the
/// legacy string match — structural producers always win.
pub fn is_model_error(provider: Provider, error: &TalkError) -> bool {
    use crate::error::PipelineFailureKind;
    if let TalkError::Pipeline(pf) = error {
        if matches!(pf.kind, PipelineFailureKind::ModelRejected { .. }) {
            return true;
        }
    }
    match provider {
        Provider::Mistral => mistral::is_model_error(error),
        Provider::OpenAI => openai::is_model_error(error),
        // Parakeet is a local backend — no remote model-rejection concept.
        Provider::Parakeet => false,
    }
}

/// Enrich a model error with available transcription model suggestions.
///
/// Dispatches to the provider module's own enrichment logic.  Returns
/// the error unchanged if it is not a model error or if the provider
/// section is not configured.
pub async fn enrich_model_error(
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    error: TalkError,
) -> TalkError {
    match provider {
        Provider::Mistral => {
            let Some(ref cfg) = config.providers.mistral else {
                return error;
            };
            let model_name = model.unwrap_or(&cfg.model);
            let api_base = cfg.url.as_deref().unwrap_or(mistral::API_BASE);
            mistral::enrich_model_error(error, &cfg.api_key, model_name, api_base).await
        }
        Provider::OpenAI => {
            let Some(ref cfg) = config.providers.openai else {
                return error;
            };
            let model_name = model.unwrap_or(&cfg.model);
            let api_base = cfg.url.as_deref().unwrap_or(openai::API_BASE);
            openai::enrich_model_error(error, &cfg.api_key, model_name, api_base).await
        }
        // Parakeet is local; no API model catalog to enrich against —
        // pass the error through unchanged.
        Provider::Parakeet => error,
    }
}

// ── Factory ──────────────────────────────────────────────────────────

/// Create a one-shot transcriber for the given provider.
///
/// When `model` is `Some`, it overrides the config default for that
/// provider (the `--model` CLI flag).  When `diarize` is `true`, the
/// transcriber will request speaker diarization (if supported by the
/// provider).  The `policy` is stored on the transcriber and
/// consulted by its `send_once` implementation to decide whether to
/// attach a per-request wall-clock timeout — see
/// [`RequestTimeoutPolicy`] for variant semantics.
pub(crate) fn create_oneshot_transcriber(
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    diarize: bool,
    policy: RequestTimeoutPolicy,
) -> Result<Box<dyn OneShotTranscriber>, TalkError> {
    match provider {
        Provider::Mistral => {
            let mut cfg = config.providers.mistral.clone().ok_or_else(|| {
                TalkError::Config(
                    "Mistral provider selected but providers.mistral is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.mistral.api_key is required".to_string(),
                ));
            }
            if let Some(m) = model {
                cfg.model = m.to_string();
            }
            Ok(Box::new(MistralOneShotTranscriber::with_policy(
                cfg, diarize, policy,
            )?))
        }
        Provider::OpenAI => {
            let mut cfg = config.providers.openai.clone().ok_or_else(|| {
                TalkError::Config(
                    "OpenAI provider selected but providers.openai is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.openai.api_key is required".to_string(),
                ));
            }
            if let Some(m) = model {
                cfg.model = m.to_string();
            }
            Ok(Box::new(OpenAIOneShotTranscriber::with_policy(
                cfg, policy,
            )?))
        }
        // Phase 3: local Parakeet backend.  Construction is cheap
        // (path resolution only); the heavy model load happens on
        // first `fetch_transcription`.  `validate` only CHECKS model
        // presence (model::ensure_present) — it never downloads.  The
        // download is an explicit, consented step driven by each entry
        // surface (CLI / toggle / picker) before transcription begins.
        #[cfg(feature = "parakeet")]
        Provider::Parakeet => {
            let mut cfg = config.providers.parakeet.clone().unwrap_or_default();
            if let Some(m) = model {
                cfg.model = Some(m.to_string());
            }
            Ok(Box::new(ParakeetOneShotTranscriber::with_policy(
                cfg, policy,
            )?))
        }
        #[cfg(not(feature = "parakeet"))]
        Provider::Parakeet => Err(TalkError::Config(
            "talk-rs was built without the 'parakeet' feature; rebuild without \
             --no-default-features to enable the local Parakeet backend"
                .to_string(),
        )),
    }
}

/// Read the transcript for a recording from the cache, falling
/// through all layers without making any API call.
///
/// Waterfall (in priority order):
///
/// 1. **Pick file** (``<stem>.pick.yml``) -- the authoritative
///    transcript, possibly user-edited.  Returned if present.
/// 2. **Default-provider / default-model sidecar** -- the
///    one-shot-mode cache for the default transcription model.
///    Read synchronously, no API call.
/// 3. **None** -- no transcript cached.
///
/// Used by consumers that want to display a transcript if one is
/// cheaply available but MUST NOT trigger network I/O.  The record
/// UI, for example, uses this to decide between "show transcript"
/// and "show audio player".
///
/// This function is synchronous because every step is a local
/// filesystem read.  It never calls the transcription API.
pub fn read_cached_transcript(audio_path: &std::path::Path, config: &Config) -> Option<String> {
    use crate::recording_cache::{get_transcript, TranscriptStatus, TranscriptionCache};

    // Layer 1: pick file (authoritative).
    match get_transcript(audio_path) {
        TranscriptStatus::Available(text) => return Some(text),
        // In-progress or unavailable: fall through to sidecar probe.
        TranscriptStatus::InProgress | TranscriptStatus::NotAvailable => {}
    }

    // Layer 3: per-model sidecar cache for the default
    // provider/model.  Synchronous file read, no API call.
    let provider = config
        .transcription
        .as_ref()
        .map(|t| t.default_provider)
        .unwrap_or(Provider::Mistral);
    let effective_model = resolve_effective_model(config, provider, None);
    TranscriptionCache::get(audio_path, provider, &effective_model).map(|r| r.text)
}

/// Produce the authoritative transcript for a recording (Layer 2).
///
/// 1. Calls [`recording_cache::get_transcript`]:
///    - [`TranscriptStatus::Available`] -> returns the text.
///    - [`TranscriptStatus::InProgress`] -> returns
///      [`TalkError::TranscriptInProgress`].
///    - [`TranscriptStatus::NotAvailable`] -> continues.
/// 2. Acquires the pick lock.
/// 3. Calls [`transcribe_audio`] with `allow_api = true`.
/// 4. Writes the pick file with the resulting text.
/// 5. Releases the pick lock.
///
/// The pick lock is always released even on error paths.
pub async fn produce_transcript(
    audio_path: &std::path::Path,
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    sink: &std::sync::Arc<dyn crate::telemetry::TelemetrySink>,
) -> Result<String, TalkError> {
    use crate::recording_cache::{self, TranscriptStatus};

    match recording_cache::get_transcript(audio_path) {
        TranscriptStatus::Available(text) => return Ok(text),
        TranscriptStatus::InProgress => return Err(TalkError::TranscriptInProgress),
        TranscriptStatus::NotAvailable => {}
    }

    recording_cache::acquire_pick_lock(audio_path)?;

    let effective_model = resolve_effective_model(config, provider, model);

    // `produce_transcript` is called from autonomous backends
    // (recording cache layer 2 producer) — pick `Proportional` so a
    // hung server cannot wedge the producer indefinitely.
    let result = transcribe_audio(
        audio_path,
        config,
        provider,
        model,
        false,
        TranscribeOptions {
            allow_api: true,
            policy: RequestTimeoutPolicy::Proportional,
            cancel_token: None,
            skip_legacy_lock: false,
        },
        sink,
    )
    .await;

    let final_result = match result {
        Ok(r) => {
            let text = r.text.trim().to_string();
            if let Err(e) = recording_cache::write_pick(
                audio_path,
                &provider.to_string(),
                &effective_model,
                false,
                &text,
            ) {
                log::warn!("failed to write pick file: {}", e);
            }
            Ok(text)
        }
        Err(e) => Err(e),
    };

    if let Err(e) = recording_cache::release_pick_lock(audio_path) {
        log::warn!("failed to release pick lock: {}", e);
    }

    final_result
}

/// Transcribe an audio file.
///
/// THE single transcription entry point for one-shot-from-file
/// transcription (Layer 3).  Checks the sidecar cache first; on
/// miss and when `allow_api` is true, acquires a per-model lock,
/// calls the provider API, stores the sidecar, releases the lock.
///
/// Callers never see cache or lock files.
///
/// # Errors
///
/// - [`TalkError::CacheOnly`]: sidecar miss and `allow_api = false`.
/// - [`TalkError::ModelInProgress`]: per-model lock already held by
///   another producer.
pub async fn transcribe_audio(
    audio_path: &Path,
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    diarize: bool,
    options: TranscribeOptions,
    sink: &std::sync::Arc<dyn crate::telemetry::TelemetrySink>,
) -> Result<TranscriptionResult, TalkError> {
    let TranscribeOptions {
        allow_api,
        policy,
        cancel_token,
        skip_legacy_lock,
    } = options;
    use crate::recording_cache::{self, TranscriptionCache};

    let effective_model = resolve_effective_model(config, provider, model);

    if !diarize {
        if let Some(cached) = TranscriptionCache::get(audio_path, provider, &effective_model) {
            log::info!(
                "transcription cache hit for {}:{} on {}",
                provider,
                effective_model,
                audio_path.display()
            );
            return Ok(cached);
        }
    }

    if !allow_api {
        log::debug!(
            "transcription cache miss for {}:{} on {} — API call forbidden",
            provider,
            effective_model,
            audio_path.display()
        );
        return Err(TalkError::CacheOnly);
    }

    // Acquire per-model lock before calling the API, unless the
    // caller has already registered via `jobs::register_local`
    // (which writes the same lock with a richer payload).
    if !skip_legacy_lock {
        recording_cache::acquire_model_lock(audio_path, provider, &effective_model, false)?;
    }

    log::info!(
        "transcription cache miss for {}:{} on {} — calling API",
        provider,
        effective_model,
        audio_path.display()
    );

    // Wrap API call in a closure so we can always release the lock.
    let api_result = async {
        let mut transcriber = create_oneshot_transcriber(config, provider, model, diarize, policy)?;
        transcriber.set_sink(sink.clone());
        if let Some(token) = cancel_token {
            transcriber.set_cancel_token(token);
        }
        transcriber.validate().await?;
        transcriber
            .fetch_transcription(TranscriptionBody::File(audio_path.to_path_buf()))
            .await
    }
    .await;

    // Lock release is skipped when the caller owns the lock via
    // `jobs::register_local` (the `LocalJob` Drop removes the
    // file).  Otherwise release here.
    let release_lock = |context: &str| {
        if skip_legacy_lock {
            return;
        }
        if let Err(e) =
            recording_cache::release_model_lock(audio_path, provider, &effective_model, false)
        {
            log::warn!("failed to release model lock {}: {}", context, e);
        }
    };

    match api_result {
        Ok(result) => {
            if let Err(e) =
                TranscriptionCache::store(audio_path, provider, &effective_model, false, &result)
            {
                log::warn!("failed to cache transcription result: {}", e);
            }
            release_lock("after success");
            Ok(result)
        }
        Err(e) => {
            release_lock("after error");
            Err(e)
        }
    }
}

/// Resolve the effective model name from CLI override or config default.
fn resolve_effective_model(config: &Config, provider: Provider, model: Option<&str>) -> String {
    if let Some(m) = model {
        return m.to_string();
    }
    match provider {
        Provider::Mistral => config
            .providers
            .mistral
            .as_ref()
            .map(|c| c.model.clone())
            .unwrap_or_else(|| "voxtral-mini-latest".to_string()),
        Provider::OpenAI => config
            .providers
            .openai
            .as_ref()
            .map(|c| c.model.clone())
            .unwrap_or_else(|| "whisper-1".to_string()),
        Provider::Parakeet => config
            .providers
            .parakeet
            .as_ref()
            .map(|c| c.resolved_model_name())
            .unwrap_or_else(|| "parakeet-tdt-0.6b-v3-int8".to_string()),
    }
}

/// Create a realtime transcriber for the given provider.
///
/// When `model` is `Some`, it overrides the config default for that
/// provider's realtime model (the `--model` CLI flag).
//
// Only the live dictation path (`dictate` / picker) constructs a
// realtime transcriber; that path requires both `capture` and `ui`.
// In a headless build it is never called, but the factory + the
// `RealtimeTranscriber` trait remain part of the exposed realtime
// API surface, so the allow is scoped to the headless configuration.
#[cfg_attr(not(all(feature = "capture", feature = "ui")), allow(dead_code))]
pub(crate) fn create_realtime_transcriber(
    config: &Config,
    provider: Provider,
    model: Option<&str>,
) -> Result<Box<dyn RealtimeTranscriber>, TalkError> {
    match provider {
        Provider::Mistral => {
            let cfg = config.providers.mistral.clone().ok_or_else(|| {
                TalkError::Config(
                    "Mistral provider selected but providers.mistral is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.mistral.api_key is required".to_string(),
                ));
            }
            Ok(Box::new(MistralRealtimeTranscriber::new(cfg)))
        }
        Provider::OpenAI => {
            let mut cfg = config.providers.openai.clone().ok_or_else(|| {
                TalkError::Config(
                    "OpenAI provider selected but providers.openai is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.openai.api_key is required".to_string(),
                ));
            }
            if let Some(m) = model {
                cfg.realtime_model = m.to_string();
            }
            Ok(Box::new(OpenAIRealtimeTranscriber::new(cfg)))
        }
        // Parakeet is one-shot-only; reject realtime selection cleanly.
        Provider::Parakeet => Err(TalkError::Config(
            "parakeet provider has no realtime mode; use one-shot transcription (omit --realtime)"
                .to_string(),
        )),
    }
}

// ── Mock ─────────────────────────────────────────────────────────────

/// Mock one-shot transcriber for testing.
///
/// Returns a hardcoded transcription result without making any API calls.
pub struct MockOneShotTranscriber {
    /// The text to return when transcribe is called.
    pub response_text: String,
}

impl MockOneShotTranscriber {
    /// Create a new mock transcriber with the given response text.
    pub fn new(response_text: impl Into<String>) -> Self {
        Self {
            response_text: response_text.into(),
        }
    }
}

#[async_trait]
impl OneShotTranscriber for MockOneShotTranscriber {
    async fn validate(&self) -> Result<(), TalkError> {
        Ok(())
    }

    async fn fetch_transcription(
        &self,
        body: TranscriptionBody,
    ) -> Result<TranscriptionResult, TalkError> {
        if let TranscriptionBody::Pipe { mut chunks, .. } = body {
            while chunks.recv().await.is_some() {}
        }

        Ok(TranscriptionResult {
            text: self.response_text.clone(),
            metadata: TranscriptionMetadata::default(),
            diarization: None,
            segments: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute path to a shipped test fixture.
    fn fixture(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests");
        p.push("fixtures");
        p.push(name);
        p
    }

    /// Read the channel count + input sample rate out of an OGG/Opus
    /// byte stream's OpusHead packet (RFC 7845).  Returns
    /// `(channels, declared_rate)`.  Note: Opus always declares 48000
    /// in OpusHead regardless of the encoder's internal rate, so the
    /// meaningful assertion for "16 kHz mono" is the channel count
    /// here plus the decoder rate verified separately.
    fn opus_head_channels(ogg: &[u8]) -> u8 {
        // Find "OpusHead" magic; channel count is the byte right after
        // the 8-magic + 1 version bytes = offset+9.
        let pos = ogg
            .windows(8)
            .position(|w| w == b"OpusHead")
            .expect("OpusHead present");
        ogg[pos + 9]
    }

    #[test]
    fn test_normalize_stereo_44k_m4a_to_mono_ogg() {
        // The stereo 44.1 kHz AAC fixture must come out as a mono OGG.
        let path = fixture("sine_440_0.5s_stereo.m4a");
        assert!(path.exists(), "fixture missing: {}", path.display());

        let (bytes, name) = normalize_file_for_upload(&path).expect("normalization should succeed");

        // Filename advertises ogg.
        assert!(name.ends_with(".ogg"), "expected .ogg name, got {name}");
        // Valid OGG/Opus stream, mono.
        assert_eq!(&bytes[0..4], b"OggS", "should be an OGG stream");
        assert_eq!(opus_head_channels(&bytes), 1, "must be downmixed to mono");

        // Decode it back as 16 kHz mono — proves the encoder produced a
        // stream a 16 kHz mono Opus decoder accepts and that it carries
        // real audio (non-silent).
        let mut decoder = opus::Decoder::new(16_000, opus::Channels::Mono)
            .expect("decoder creation should succeed");
        // Pull the first audio packet out of the OGG container.
        let mut packet_reader =
            ogg::reading::PacketReader::new(std::io::Cursor::new(bytes.clone()));
        // Skip the two header packets (OpusHead, OpusTags).
        let _ = packet_reader.read_packet_expected().expect("OpusHead");
        let _ = packet_reader.read_packet_expected().expect("OpusTags");
        let first_audio = packet_reader
            .read_packet_expected()
            .expect("at least one audio packet");
        let mut out = vec![0i16; 16_000]; // up to 1s
        let decoded = decoder
            .decode(&first_audio.data, &mut out, false)
            .expect("decode should succeed");
        assert!(decoded > 0, "decoded frame should be non-empty");
    }

    #[test]
    fn test_normalize_mono_m4a_stays_mono_ogg() {
        let path = fixture("sine_440_0.5s_mono.m4a");
        assert!(path.exists(), "fixture missing: {}", path.display());

        let (bytes, name) = normalize_file_for_upload(&path).expect("normalization should succeed");
        assert!(name.ends_with(".ogg"));
        assert_eq!(&bytes[0..4], b"OggS");
        assert_eq!(opus_head_channels(&bytes), 1);
    }

    #[test]
    fn test_normalize_missing_file_errors() {
        let err = normalize_file_for_upload(&PathBuf::from("/nonexistent/audio.m4a"))
            .expect_err("missing file should error");
        assert!(
            err.to_string().contains("not found"),
            "expected not-found error, got: {err}"
        );
    }

    #[test]
    fn test_normalize_undecodable_falls_back_to_raw_bytes() {
        // A file with an unsupported extension cannot be decoded; the
        // normalizer must fall back to uploading the raw bytes rather
        // than failing (never break a previously-working upload).
        let dir = tempfile::TempDir::new().expect("tmp dir");
        let path = dir.path().join("weird.bin");
        std::fs::write(&path, b"not really audio").expect("write");

        let (bytes, name) = normalize_file_for_upload(&path).expect("fallback should succeed");
        assert_eq!(bytes, b"not really audio");
        assert_eq!(name, "weird.bin", "fallback keeps original file name");
    }

    #[tokio::test]
    async fn test_mock_transcriber_returns_text() {
        let mock = MockOneShotTranscriber::new("Hello, world!");
        let result = mock
            .fetch_transcription(TranscriptionBody::File(PathBuf::from("/tmp/test.wav")))
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "Hello, world!");
    }

    #[tokio::test]
    async fn test_mock_transcriber_stream() {
        let mock = MockOneShotTranscriber::new("Streamed transcription");
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        // Send some fake audio data
        tx.send(vec![0u8; 100]).await.unwrap();
        tx.send(vec![1u8; 200]).await.unwrap();
        drop(tx); // Close the channel

        let result = mock
            .fetch_transcription(TranscriptionBody::Pipe {
                chunks: rx,
                file_name: "test.wav".to_string(),
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "Streamed transcription");
    }

    #[tokio::test]
    async fn test_mock_transcriber_ignores_path() {
        let mock = MockOneShotTranscriber::new("Fixed response");
        let result1 = mock
            .fetch_transcription(TranscriptionBody::File(PathBuf::from("/path/one.wav")))
            .await;
        let result2 = mock
            .fetch_transcription(TranscriptionBody::File(PathBuf::from("/path/two.wav")))
            .await;

        assert_eq!(result1.unwrap().text, "Fixed response");
        assert_eq!(result2.unwrap().text, "Fixed response");
    }

    #[test]
    fn test_provider_from_str() {
        assert_eq!("mistral".parse::<Provider>().unwrap(), Provider::Mistral);
        assert_eq!("openai".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert_eq!("OpenAI".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert!("unknown".parse::<Provider>().is_err());
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(Provider::Mistral.to_string(), "mistral");
        assert_eq!(Provider::OpenAI.to_string(), "openai");
    }

    #[test]
    fn test_format_plain_text_without_diarization() {
        let result = TranscriptionResult {
            text: "Hello world.".to_string(),
            metadata: Default::default(),
            diarization: None,
            segments: None,
        };
        assert_eq!(format_transcription_output(&result, false), "Hello world.");
    }

    #[test]
    fn test_format_diarized_output() {
        let result = TranscriptionResult {
            text: "Hello. I am fine.".to_string(),
            metadata: Default::default(),
            diarization: Some(vec![
                DiarizationSegment {
                    speaker: "SPEAKER_00".to_string(),
                    start: 0.0,
                    end: 1.5,
                    text: "Hello.".to_string(),
                },
                DiarizationSegment {
                    speaker: "SPEAKER_01".to_string(),
                    start: 1.5,
                    end: 3.0,
                    text: "I am fine.".to_string(),
                },
            ]),
            segments: None,
        };
        assert_eq!(
            format_transcription_output(&result, false),
            "[SPEAKER_00] Hello.\n[SPEAKER_01] I am fine."
        );
    }

    #[test]
    fn test_format_diarized_merges_same_speaker() {
        let result = TranscriptionResult {
            text: "Hello. How are you? I am fine.".to_string(),
            metadata: Default::default(),
            diarization: Some(vec![
                DiarizationSegment {
                    speaker: "SPEAKER_00".to_string(),
                    start: 0.0,
                    end: 1.0,
                    text: "Hello.".to_string(),
                },
                DiarizationSegment {
                    speaker: "SPEAKER_00".to_string(),
                    start: 1.0,
                    end: 2.0,
                    text: "How are you?".to_string(),
                },
                DiarizationSegment {
                    speaker: "SPEAKER_01".to_string(),
                    start: 2.0,
                    end: 3.5,
                    text: "I am fine.".to_string(),
                },
            ]),
            segments: None,
        };
        assert_eq!(
            format_transcription_output(&result, false),
            "[SPEAKER_00] Hello. How are you?\n[SPEAKER_01] I am fine."
        );
    }

    #[test]
    fn test_format_diarized_empty_segments() {
        let result = TranscriptionResult {
            text: "Hello world.".to_string(),
            metadata: Default::default(),
            diarization: Some(vec![]),
            segments: None,
        };
        // Empty segments → fall back to plain text
        assert_eq!(format_transcription_output(&result, false), "Hello world.");
    }

    #[test]
    fn test_format_diarized_with_timestamps() {
        let result = TranscriptionResult {
            text: "Hello. I am fine.".to_string(),
            metadata: Default::default(),
            diarization: Some(vec![
                DiarizationSegment {
                    speaker: "speaker_1".to_string(),
                    start: 0.0,
                    end: 1.5,
                    text: "Hello.".to_string(),
                },
                DiarizationSegment {
                    speaker: "speaker_2".to_string(),
                    start: 1.5,
                    end: 3.0,
                    text: "I am fine.".to_string(),
                },
            ]),
            segments: None,
        };
        assert_eq!(
            format_transcription_output(&result, true),
            "[00:00:00] speaker_1 Hello.\n[00:00:01] speaker_2 I am fine."
        );
    }

    #[test]
    fn test_format_diarized_with_timestamps_merges_same_speaker() {
        let result = TranscriptionResult {
            text: "Hello. How are you? I am fine.".to_string(),
            metadata: Default::default(),
            diarization: Some(vec![
                DiarizationSegment {
                    speaker: "speaker_1".to_string(),
                    start: 0.0,
                    end: 1.0,
                    text: "Hello.".to_string(),
                },
                DiarizationSegment {
                    speaker: "speaker_1".to_string(),
                    start: 1.0,
                    end: 2.0,
                    text: "How are you?".to_string(),
                },
                DiarizationSegment {
                    speaker: "speaker_2".to_string(),
                    start: 2.0,
                    end: 3.5,
                    text: "I am fine.".to_string(),
                },
            ]),
            segments: None,
        };
        assert_eq!(
            format_transcription_output(&result, true),
            "[00:00:00] speaker_1 Hello. How are you?\n[00:00:02] speaker_2 I am fine."
        );
    }

    #[test]
    fn test_format_timestamp_helper() {
        assert_eq!(format_timestamp(0.0), "00:00:00");
        assert_eq!(format_timestamp(1.5), "00:00:01");
        assert_eq!(format_timestamp(61.0), "00:01:01");
        assert_eq!(format_timestamp(3661.0), "01:01:01");
    }

    #[test]
    fn test_parse_transcript_segments_voxtral_shape() {
        // Real Voxtral response shape: segments with start/end/text
        // and speaker_id: null when diarization is not requested.
        let raw = serde_json::json!([
            {
                "text": "Du coup je viens de corriger.",
                "start": 1.2,
                "end": 10.7,
                "type": "transcription_segment",
                "speaker_id": null
            },
            {
                "text": " Le premier c'était un bug.",
                "start": 11.9,
                "end": 24.5,
                "type": "transcription_segment",
                "speaker_id": null
            }
        ]);
        let slice = raw.as_array().expect("array literal is array");
        let parsed = parse_transcript_segments(slice).expect("some segments");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].start, 1.2);
        assert_eq!(parsed[0].end, 10.7);
        assert_eq!(parsed[0].text, "Du coup je viens de corriger.");
        assert_eq!(parsed[1].start, 11.9);
        assert_eq!(parsed[1].end, 24.5);
    }

    #[test]
    fn test_parse_transcript_segments_whisper_verbose_json_shape() {
        // Whisper verbose_json shape adds extra fields we ignore:
        // `id`, `seek`, `tokens`, `temperature`, `avg_logprob`,
        // `compression_ratio`, `no_speech_prob`.  Only start/end/text
        // matter for us.
        let raw = serde_json::json!([
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 3.2,
                "text": " Hello world.",
                "tokens": [50364, 2425, 1002, 13, 50524],
                "temperature": 0.0,
                "avg_logprob": -0.3,
                "compression_ratio": 1.1,
                "no_speech_prob": 0.01
            }
        ]);
        let slice = raw.as_array().expect("array literal is array");
        let parsed = parse_transcript_segments(slice).expect("some segments");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].start, 0.0);
        assert_eq!(parsed[0].end, 3.2);
        assert_eq!(parsed[0].text, " Hello world.");
    }

    #[test]
    fn test_parse_transcript_segments_skips_malformed() {
        // Segments without start/end, or with empty text, are skipped.
        // Returns None when nothing usable survives.
        let raw = serde_json::json!([
            { "text": "no timing here" },
            { "start": 0.0, "text": "no end here" },
            { "start": 0.0, "end": 1.0, "text": "" },
            { "start": 5.0, "end": 6.0 }
        ]);
        let slice = raw.as_array().expect("array literal is array");
        assert!(parse_transcript_segments(slice).is_none());
    }

    #[test]
    fn test_parse_transcript_segments_mixed_good_and_bad() {
        // One good segment among several malformed ones survives.
        let raw = serde_json::json!([
            { "text": "no timing" },
            { "start": 1.0, "end": 2.5, "text": "valid" },
            { "start": 0.0, "end": 1.0, "text": "" }
        ]);
        let slice = raw.as_array().expect("array literal is array");
        let parsed = parse_transcript_segments(slice).expect("some segments");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].text, "valid");
    }

    #[test]
    fn test_parse_transcript_segments_empty_input() {
        let parsed = parse_transcript_segments(&[]);
        assert!(parsed.is_none());
    }

    #[test]
    fn test_transcription_result_default_has_no_segments() {
        let result = TranscriptionResult::default();
        assert!(result.segments.is_none());
        assert!(result.diarization.is_none());
        assert_eq!(result.text, "");
    }

    fn minimal_config() -> Config {
        use tempfile::NamedTempFile;
        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test
"#;
        let mut file = NamedTempFile::new().expect("tmp file");
        std::io::Write::write_all(&mut file, yaml.as_bytes()).expect("write");
        Config::load(Some(file.path())).expect("load")
    }

    #[tokio::test]
    async fn test_produce_transcript_returns_existing_pick() {
        let dir = tempfile::TempDir::new().expect("tmp dir");
        let audio_path = dir.path().join("has-pick.ogg");
        std::fs::write(&audio_path, b"fake").expect("write audio");
        crate::recording_cache::write_pick(
            &audio_path,
            "openai",
            "whisper-1",
            false,
            "cached text",
        )
        .expect("write pick");

        let config = minimal_config();
        let result = produce_transcript(
            &audio_path,
            &config,
            Provider::OpenAI,
            None,
            &(std::sync::Arc::new(crate::telemetry::NoOpSink)
                as std::sync::Arc<dyn crate::telemetry::TelemetrySink>),
        )
        .await;
        assert_eq!(result.unwrap(), "cached text");
    }

    #[tokio::test]
    async fn test_produce_transcript_returns_in_progress_when_locked() {
        let dir = tempfile::TempDir::new().expect("tmp dir");
        let audio_path = dir.path().join("locked.ogg");
        std::fs::write(&audio_path, b"fake").expect("write audio");
        crate::recording_cache::acquire_pick_lock(&audio_path).expect("lock");

        let config = minimal_config();
        let result = produce_transcript(
            &audio_path,
            &config,
            Provider::OpenAI,
            None,
            &(std::sync::Arc::new(crate::telemetry::NoOpSink)
                as std::sync::Arc<dyn crate::telemetry::TelemetrySink>),
        )
        .await;
        assert!(matches!(result, Err(TalkError::TranscriptInProgress)));
    }

    #[tokio::test]
    async fn test_produce_transcript_prefers_pick_over_lock_only_when_lock_absent() {
        // When both pick and lock exist, lock wins -> InProgress.
        // When only pick exists, pick wins -> return text.
        let dir = tempfile::TempDir::new().expect("tmp dir");
        let audio_path = dir.path().join("pick-only.ogg");
        std::fs::write(&audio_path, b"fake").expect("write audio");
        crate::recording_cache::write_pick(&audio_path, "openai", "whisper-1", false, "x")
            .expect("write pick");

        let config = minimal_config();
        let result = produce_transcript(
            &audio_path,
            &config,
            Provider::OpenAI,
            None,
            &(std::sync::Arc::new(crate::telemetry::NoOpSink)
                as std::sync::Arc<dyn crate::telemetry::TelemetrySink>),
        )
        .await;
        assert_eq!(result.unwrap(), "x");
    }

    /// Factory wiring: with the `parakeet` feature enabled (the
    /// default), `create_oneshot_transcriber(.., Provider::Parakeet, ..)`
    /// must return `Ok(boxed)` without touching the filesystem or the
    /// network.  The expensive model load is deferred to
    /// `validate` / `fetch_transcription`.
    #[cfg(feature = "parakeet")]
    #[test]
    fn create_oneshot_transcriber_parakeet_constructs_without_io() {
        use crate::config::{ParakeetConfig, ParakeetVariant, ProvidersConfig};

        let tmp = tempfile::TempDir::new().expect("tmp dir");
        let parakeet_cfg = ParakeetConfig {
            variant: ParakeetVariant::Int8,
            // Empty dir — must NOT error at construction time.
            model_dir: Some(tmp.path().to_path_buf()),
            num_threads: 1,
            model: None,
        };
        let config = Config {
            output_dir: tmp.path().to_path_buf(),
            providers: ProvidersConfig {
                mistral: None,
                openai: None,
                parakeet: Some(parakeet_cfg),
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };

        let t = create_oneshot_transcriber(
            &config,
            Provider::Parakeet,
            None,
            false,
            RequestTimeoutPolicy::Proportional,
        )
        .expect("parakeet construction must succeed");
        // We only need to confirm the boxed value exists — the
        // concrete type stays private behind the trait.
        let _: Box<dyn OneShotTranscriber> = t;
    }
}
