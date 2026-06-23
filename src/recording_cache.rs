//! Recording cache management for talk-rs.
//!
//! Keeps the last N recordings in `~/.cache/talk-rs/recordings/` with
//! companion YAML metadata files.  Each recording is an OGG file named
//! with an ISO-8601 timestamp; the metadata file encodes the provider,
//! model, and mode in its filename and contains the transcript plus
//! recording metadata.

use crate::config::Provider;
use crate::daemon::cache_dir;
use crate::error::TalkError;
use crate::transcription::{
    MistralProviderMetadata, OpenAIProviderMetadata, OpenAIRealtimeMetadata,
    ProviderSpecificMetadata, TokenUsage, TranscriptionMetadata, TranscriptionResult,
};
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

/// Maximum number of recordings to keep in the cache.
const MAX_CACHED_RECORDINGS: usize = 10;
const LAST_RECORDING_POINTER: &str = "last_recording.ogg";
const LAST_METADATA_POINTER: &str = "last_metadata.yml";
const LAST_PASTE_STATE_FILE: &str = "last_paste.yml";

/// Metadata associated with a cached recording.
#[derive(Debug, Serialize, Deserialize)]
pub struct RecordingMetadata {
    /// Filename of the cached audio recording (basename only).
    pub recording: String,
    /// Transcription provider used.
    pub provider: String,
    /// Model name used for transcription.
    pub model: String,
    /// Whether realtime mode was used.
    pub realtime: bool,
    /// The transcription result.
    pub transcript: String,
    /// ISO-8601 timestamp of when the recording was made.
    pub timestamp: String,
    /// Optional API metadata captured during transcription.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<CommonMetadata>,
    /// Provider-specific metadata payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_api: Option<ProviderApiMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<CommonSegment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diarization: Option<Vec<CommonDiarizationSegment>>,
}

/// Minimal metadata used for retry/replacement flows.
#[derive(Debug, Deserialize)]
pub struct RecordingMetadataBrief {
    pub transcript: String,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

/// State of the last pasted transcription.
#[derive(Debug, Serialize, Deserialize)]
pub struct LastPasteState {
    pub timestamp: String,
    pub char_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_id: Option<String>,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PickerSelectionMetadata {
    provider: String,
    model: String,
    streaming: bool,
    /// The final transcription text chosen by the user (may have been
    /// edited in the text area).  This is the authoritative text for
    /// the recording — takes priority over provider sidecars.
    #[serde(default)]
    text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommonMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_latency_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_elapsed_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_processing_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub word_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<CommonTokenUsage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommonTokenUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommonSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommonDiarizationSegment {
    pub speaker: String,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProviderApiMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai: Option<OpenAIApiMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mistral: Option<MistralApiMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIApiMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub rate_limit_headers: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unknown_event_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realtime: Option<OpenAIRealtimeApiMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIRealtimeApiMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub event_counts: std::collections::BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_rate_limits: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub ws_upgrade_headers: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MistralApiMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unknown_event_types: Vec<String>,
}

impl RecordingMetadata {
    /// Convert to a TranscriptionResult (for cache reads).
    pub fn into_transcription_result(self) -> TranscriptionResult {
        let segments = self.segments.map(|segs| {
            segs.into_iter()
                .map(|s| crate::transcription::TranscriptSegment {
                    start: s.start,
                    end: s.end,
                    text: s.text,
                })
                .collect()
        });

        let diarization = self.diarization.map(|segs| {
            segs.into_iter()
                .map(|s| crate::transcription::DiarizationSegment {
                    speaker: s.speaker,
                    start: s.start,
                    end: s.end,
                    text: s.text,
                })
                .collect()
        });

        let metadata = self
            .metadata
            .map(|cm| TranscriptionMetadata {
                request_latency_ms: cm.request_latency_ms,
                session_elapsed_ms: cm.session_elapsed_ms,
                request_id: cm.request_id,
                provider_processing_ms: cm.provider_processing_ms,
                detected_language: cm.detected_language,
                audio_seconds: cm.audio_seconds,
                segment_count: cm.segment_count,
                word_count: cm.word_count,
                token_usage: cm.token_usage.map(|tu| TokenUsage {
                    input_tokens: tu.input_tokens,
                    output_tokens: tu.output_tokens,
                    total_tokens: tu.total_tokens,
                }),
                provider_specific: None,
            })
            .unwrap_or_default();

        TranscriptionResult {
            text: self.transcript,
            metadata,
            diarization,
            segments,
        }
    }
}

/// Transcription sidecar cache.
///
/// Abstract cache for transcription results. The storage mechanism
/// (currently YAML files next to the audio) is an implementation
/// detail hidden behind this API.
pub struct TranscriptionCache;

impl TranscriptionCache {
    /// Look up a cached transcription for the given audio file,
    /// provider, and model.
    pub fn get(audio_path: &Path, provider: Provider, model: &str) -> Option<TranscriptionResult> {
        let dir = audio_path.parent()?;
        let stem = audio_path.file_stem()?.to_str()?;
        // `oneshot` is the current mode token; `batch` is the legacy
        // alias kept in the read list so pre-rename sidecars written
        // before the batch->one-shot vocabulary change are still found.
        for mode in &["oneshot", "batch", "realtime"] {
            let safe_model = model.replace(['/', ' '], "-");
            let filename = format!("{}_{}_{}_{}.yml", stem, provider, safe_model, mode);
            let path = dir.join(&filename);
            if path.exists() {
                match Self::read_sidecar(&path) {
                    Ok(result) => return Some(result),
                    Err(e) => {
                        log::warn!("failed to read cached sidecar {}: {}", path.display(), e);
                        continue;
                    }
                }
            }
        }
        None
    }

    /// Store a transcription result next to the audio file.
    pub fn store(
        audio_path: &Path,
        provider: Provider,
        model: &str,
        realtime: bool,
        result: &TranscriptionResult,
    ) -> Result<PathBuf, TalkError> {
        let dir = audio_path
            .parent()
            .ok_or_else(|| TalkError::Config("audio path has no parent directory".to_string()))?;
        let stem = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| TalkError::Config("audio path has no file stem".to_string()))?;
        let audio_filename = audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| TalkError::Config("audio path has no filename".to_string()))?;
        let text = crate::transcription::format_transcription_output(result, false);
        write_metadata_to_dir(
            dir,
            stem,
            provider,
            model,
            realtime,
            &text,
            audio_filename,
            &result.metadata,
            result.segments.as_deref(),
            result.diarization.as_deref(),
        )
    }

    /// Read a YAML sidecar file and convert to TranscriptionResult.
    fn read_sidecar(path: &Path) -> Result<TranscriptionResult, TalkError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| TalkError::Config(format!("failed to read {}: {}", path.display(), e)))?;
        let meta: RecordingMetadata = serde_yaml::from_str(&content)
            .map_err(|e| TalkError::Config(format!("failed to parse {}: {}", path.display(), e)))?;
        Ok(meta.into_transcription_result())
    }
}

fn is_token_usage_empty(v: &TokenUsage) -> bool {
    v.input_tokens.is_none() && v.output_tokens.is_none() && v.total_tokens.is_none()
}

fn metadata_has_common_fields(metadata: &TranscriptionMetadata) -> bool {
    metadata.request_latency_ms.is_some()
        || metadata.session_elapsed_ms.is_some()
        || metadata.request_id.is_some()
        || metadata.provider_processing_ms.is_some()
        || metadata.detected_language.is_some()
        || metadata.audio_seconds.is_some()
        || metadata.segment_count.is_some()
        || metadata.word_count.is_some()
        || metadata
            .token_usage
            .as_ref()
            .is_some_and(|usage| !is_token_usage_empty(usage))
}

fn common_metadata_from_transcription(metadata: &TranscriptionMetadata) -> Option<CommonMetadata> {
    if !metadata_has_common_fields(metadata) {
        return None;
    }

    let token_usage = metadata
        .token_usage
        .as_ref()
        .filter(|usage| !is_token_usage_empty(usage))
        .map(|usage| CommonTokenUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
        });

    Some(CommonMetadata {
        request_latency_ms: metadata.request_latency_ms,
        session_elapsed_ms: metadata.session_elapsed_ms,
        request_id: metadata.request_id.clone(),
        provider_processing_ms: metadata.provider_processing_ms,
        detected_language: metadata.detected_language.clone(),
        audio_seconds: metadata.audio_seconds,
        segment_count: metadata.segment_count,
        word_count: metadata.word_count,
        token_usage,
    })
}

fn openai_provider_metadata(data: &OpenAIProviderMetadata) -> Option<OpenAIApiMetadata> {
    if data.model.is_none()
        && data.usage_raw.is_none()
        && data.rate_limit_headers.is_empty()
        && data.unknown_event_types.is_empty()
        && data
            .realtime
            .as_ref()
            .is_none_or(openai_realtime_metadata_is_empty)
    {
        return None;
    }

    Some(OpenAIApiMetadata {
        model: data.model.clone(),
        usage: data.usage_raw.clone(),
        rate_limit_headers: data.rate_limit_headers.clone(),
        unknown_event_types: data.unknown_event_types.clone(),
        realtime: data.realtime.as_ref().and_then(openai_realtime_metadata),
    })
}

fn openai_realtime_metadata_is_empty(data: &OpenAIRealtimeMetadata) -> bool {
    data.session_id.is_none()
        && data.conversation_id.is_none()
        && data.event_counts.is_empty()
        && data.last_rate_limits.is_none()
        && data.ws_upgrade_headers.is_empty()
}

fn openai_realtime_metadata(data: &OpenAIRealtimeMetadata) -> Option<OpenAIRealtimeApiMetadata> {
    if openai_realtime_metadata_is_empty(data) {
        return None;
    }

    Some(OpenAIRealtimeApiMetadata {
        session_id: data.session_id.clone(),
        conversation_id: data.conversation_id.clone(),
        event_counts: data.event_counts.clone(),
        last_rate_limits: data.last_rate_limits.clone(),
        ws_upgrade_headers: data.ws_upgrade_headers.clone(),
    })
}

fn mistral_provider_metadata(data: &MistralProviderMetadata) -> Option<MistralApiMetadata> {
    if data.model.is_none() && data.usage_raw.is_none() && data.unknown_event_types.is_empty() {
        return None;
    }

    Some(MistralApiMetadata {
        model: data.model.clone(),
        usage: data.usage_raw.clone(),
        unknown_event_types: data.unknown_event_types.clone(),
    })
}

fn provider_api_metadata_from_transcription(
    metadata: &TranscriptionMetadata,
) -> Option<ProviderApiMetadata> {
    let (openai, mistral) = match metadata.provider_specific.as_ref() {
        Some(ProviderSpecificMetadata::OpenAI(data)) => (openai_provider_metadata(data), None),
        Some(ProviderSpecificMetadata::Mistral(data)) => (None, mistral_provider_metadata(data)),
        None => (None, None),
    };

    if openai.is_none() && mistral.is_none() {
        None
    } else {
        Some(ProviderApiMetadata { openai, mistral })
    }
}

pub(crate) fn common_segments_from_result(
    segments: Option<&[crate::transcription::TranscriptSegment]>,
) -> Option<Vec<CommonSegment>> {
    let segs = segments?;
    if segs.is_empty() {
        return None;
    }
    Some(
        segs.iter()
            .map(|s| CommonSegment {
                start: s.start,
                end: s.end,
                text: s.text.clone(),
            })
            .collect(),
    )
}

pub(crate) fn common_diarization_from_result(
    diarization: Option<&[crate::transcription::DiarizationSegment]>,
) -> Option<Vec<CommonDiarizationSegment>> {
    let segs = diarization?;
    if segs.is_empty() {
        return None;
    }
    Some(
        segs.iter()
            .map(|s| CommonDiarizationSegment {
                speaker: s.speaker.clone(),
                start: s.start,
                end: s.end,
                text: s.text.clone(),
            })
            .collect(),
    )
}

/// Get the recordings cache directory (`~/.cache/talk-rs/recordings/`).
pub fn recordings_dir() -> Result<PathBuf, TalkError> {
    Ok(cache_dir()?.join("recordings"))
}

/// Ensure the recordings cache directory exists.
fn ensure_recordings_dir() -> Result<PathBuf, TalkError> {
    let dir = recordings_dir()?;
    fs::create_dir_all(&dir).map_err(|e| {
        TalkError::Config(format!(
            "failed to create recordings directory {}: {}",
            dir.display(),
            e
        ))
    })?;
    Ok(dir)
}

/// Generate a timestamped OGG path in the recordings cache directory.
///
/// Returns `(ogg_path, timestamp_string)` so the same timestamp can be
/// used for both the OGG and metadata filenames.
///
/// The timestamp is an ISO 8601 local datetime with numeric timezone
/// offset (e.g. `2026-04-11T13-15-52+0200`).  Colons in the time portion
/// are replaced by dashes so the filename is filesystem-safe, and the
/// timezone offset disambiguates recordings made across DST transitions
/// or from hosts in different timezones.  This format matches the
/// `memo` tool's naming scheme.
pub fn generate_recording_path() -> Result<(PathBuf, String), TalkError> {
    let dir = ensure_recordings_dir()?;
    let now = Local::now();
    let ts = now.format("%Y-%m-%dT%H-%M-%S%z").to_string();
    let ogg_path = dir.join(format!("{}.ogg", ts));
    Ok((ogg_path, ts))
}

/// Build the metadata YAML filename from components.
///
/// Format: `{timestamp}_{provider}_{model}_{mode}.yml`
/// where mode is "realtime" or "oneshot".
fn metadata_filename(timestamp: &str, provider: Provider, model: &str, realtime: bool) -> String {
    let mode = if realtime { "realtime" } else { "oneshot" };
    // Sanitise model name: replace `/` and spaces with `-`
    let safe_model = model.replace(['/', ' '], "-");
    format!("{}_{}_{}_{}.yml", timestamp, provider, safe_model, mode)
}

/// Per-model lock filename.
///
/// Format: `{stem}_{provider}_{model}_{mode}_lock.yml`.
fn model_lock_filename(stem: &str, provider: Provider, model: &str, realtime: bool) -> String {
    let mode = if realtime { "realtime" } else { "oneshot" };
    let safe_model = model.replace(['/', ' '], "-");
    format!("{}_{}_{}_{}-lock.yml", stem, provider, safe_model, mode)
}

fn model_lock_path(
    audio_path: &Path,
    provider: Provider,
    model: &str,
    realtime: bool,
) -> Result<PathBuf, TalkError> {
    let dir = audio_path
        .parent()
        .ok_or_else(|| TalkError::Config("audio path has no parent directory".to_string()))?;
    let stem = audio_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| TalkError::Config("audio path has no file stem".to_string()))?;
    Ok(dir.join(model_lock_filename(stem, provider, model, realtime)))
}

/// Public wrapper for [`model_lock_path`] used by
/// [`crate::transcription::jobs`].  Exposes the lock-path
/// computation so the jobs module can write the YAML payload at
/// the same location the legacy `acquire_model_lock` would have
/// used.
pub fn model_lock_path_public(
    audio_path: &Path,
    provider: Provider,
    model: &str,
    realtime: bool,
) -> Result<PathBuf, TalkError> {
    model_lock_path(audio_path, provider, model, realtime)
}

/// Acquire the per-model lock for a (recording, provider, model, mode)
/// tuple.  Returns `Err(TalkError::ModelInProgress)` if the lock is
/// already held by another process.
pub fn acquire_model_lock(
    audio_path: &Path,
    provider: Provider,
    model: &str,
    realtime: bool,
) -> Result<(), TalkError> {
    let path = model_lock_path(audio_path, provider, model, realtime)?;
    if path.exists() {
        return Err(TalkError::ModelInProgress);
    }
    fs::write(&path, b"")
        .map_err(|e| TalkError::Config(format!("failed to write {}: {}", path.display(), e)))
}

/// Release the per-model lock.  Idempotent: missing lock is not an
/// error.
pub fn release_model_lock(
    audio_path: &Path,
    provider: Provider,
    model: &str,
    realtime: bool,
) -> Result<(), TalkError> {
    let path = model_lock_path(audio_path, provider, model, realtime)?;
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(TalkError::Config(format!(
            "failed to remove {}: {}",
            path.display(),
            e
        ))),
    }
}

/// Check whether a per-model lock currently exists without acquiring
/// it.  Used by probe logic.
pub fn is_model_locked(audio_path: &Path, provider: Provider, model: &str, realtime: bool) -> bool {
    model_lock_path(audio_path, provider, model, realtime)
        .map(|p| p.exists())
        .unwrap_or(false)
}

fn pick_filename(stem: &str) -> String {
    format!("{stem}.pick.yml")
}

fn pick_lock_filename(stem: &str) -> String {
    format!("{stem}.pick-lock.yml")
}

fn pick_path(audio_path: &Path) -> Result<PathBuf, TalkError> {
    let dir = audio_path
        .parent()
        .ok_or_else(|| TalkError::Config("audio path has no parent directory".to_string()))?;
    let stem = audio_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| TalkError::Config("audio path has no file stem".to_string()))?;
    Ok(dir.join(pick_filename(stem)))
}

fn pick_lock_path(audio_path: &Path) -> Result<PathBuf, TalkError> {
    let dir = audio_path
        .parent()
        .ok_or_else(|| TalkError::Config("audio path has no parent directory".to_string()))?;
    let stem = audio_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| TalkError::Config("audio path has no file stem".to_string()))?;
    Ok(dir.join(pick_lock_filename(stem)))
}

fn recording_sidecar_dirs(audio_path: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(audio_dir) = audio_path.parent() {
        dirs.push(audio_dir.to_path_buf());
    }
    if let Ok(cache_dir) = recordings_dir() {
        if !dirs.iter().any(|dir| dir == &cache_dir) {
            dirs.push(cache_dir);
        }
    }
    dirs
}

pub fn list_sidecars_for_audio(audio_path: &Path) -> Vec<(Provider, String, String, bool)> {
    let stem = match audio_path.file_stem().and_then(|s| s.to_str()) {
        Some(stem) if !stem.is_empty() => stem,
        _ => return Vec::new(),
    };

    let sidecar_prefix = format!("{stem}_");
    let selection_name = pick_filename(stem);
    let mut seen: HashSet<(String, String, bool)> = HashSet::new();
    let mut entries = Vec::new();

    for dir in recording_sidecar_dirs(audio_path) {
        let read_dir = match fs::read_dir(&dir) {
            Ok(read_dir) => read_dir,
            Err(err) => {
                log::debug!("failed to scan sidecars in {}: {}", dir.display(), err);
                continue;
            }
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(name) => name,
                None => continue,
            };
            if name == selection_name
                || !name.starts_with(&sidecar_prefix)
                || path.extension().and_then(|ext| ext.to_str()) != Some("yml")
            {
                continue;
            }

            let content = match fs::read_to_string(&path) {
                Ok(content) => content,
                Err(err) => {
                    log::debug!("failed to read sidecar {}: {}", path.display(), err);
                    continue;
                }
            };
            let metadata = match serde_yaml::from_str::<RecordingMetadata>(&content) {
                Ok(metadata) => metadata,
                Err(err) => {
                    log::debug!("skipping non-recording sidecar {}: {}", path.display(), err);
                    continue;
                }
            };
            let provider = match metadata.provider.parse::<Provider>() {
                Ok(provider) => provider,
                Err(err) => {
                    log::debug!(
                        "skipping sidecar with invalid provider {}: {}",
                        path.display(),
                        err
                    );
                    continue;
                }
            };

            let key = (
                provider.to_string(),
                metadata.model.clone(),
                metadata.realtime,
            );
            if seen.insert(key) {
                entries.push((
                    provider,
                    metadata.model,
                    metadata.transcript,
                    metadata.realtime,
                ));
            }
        }
    }

    entries
}

/// Status of a recording's authoritative transcription.
///
/// This is the return type of [`get_transcript`] and encapsulates
/// the three possible states the pick file system can be in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptStatus {
    /// Pick file exists.  Text may be empty (valid result for silent
    /// recordings).
    Available(String),
    /// Pick lock file exists.  Another process is currently
    /// producing the authoritative transcript.
    InProgress,
    /// Neither pick file nor lock file exists.  No transcript.
    NotAvailable,
}

/// Check the authoritative transcript status for a recording.
///
/// Checks the pick lock first (`<stem>.pick-lock.yml`) then the pick
/// file (`<stem>.pick.yml`).  If a lock is present the returned
/// status is [`TranscriptStatus::InProgress`] regardless of whether
/// a pick file also exists -- the production is overwriting.
///
/// This is the single source of truth for "does this recording have
/// a transcript?".  No sidecar fallback.  No API call.
pub fn get_transcript(audio_path: &Path) -> TranscriptStatus {
    // Lock takes priority over pick: if a lock is present, a
    // producer is mid-flight and any pick on disk is stale.
    if let Ok(lock) = pick_lock_path(audio_path) {
        if lock.exists() {
            return TranscriptStatus::InProgress;
        }
    }
    match read_pick(audio_path) {
        Some((_, _, _, text)) => TranscriptStatus::Available(text),
        None => TranscriptStatus::NotAvailable,
    }
}

/// Write the pick file for a recording.
///
/// The pick file is the authoritative transcription -- possibly
/// user-edited.  Overwrites any existing pick file.
pub fn write_pick(
    audio_path: &Path,
    provider: &str,
    model: &str,
    streaming: bool,
    text: &str,
) -> Result<(), TalkError> {
    let path = pick_path(audio_path)?;
    let selection = PickerSelectionMetadata {
        provider: provider.to_string(),
        model: model.to_string(),
        streaming,
        text: text.to_string(),
    };
    let yaml = serde_yaml::to_string(&selection)
        .map_err(|e| TalkError::Config(format!("failed to serialise pick: {}", e)))?;
    fs::write(&path, yaml)
        .map_err(|e| TalkError::Config(format!("failed to write {}: {}", path.display(), e)))
}

/// Write the pick file for a recording only when no pick already
/// exists.
///
/// Used by the `dictate` command after a successful streaming
/// transcription: the streaming result becomes the authoritative
/// transcript unless the user already edited one.  Produces no
/// error if a pick is already present; the existing pick is
/// preserved.
pub fn write_pick_if_absent(
    audio_path: &Path,
    provider: &str,
    model: &str,
    streaming: bool,
    text: &str,
) -> Result<(), TalkError> {
    if read_pick(audio_path).is_some() {
        return Ok(());
    }
    write_pick(audio_path, provider, model, streaming, text)
}

/// Read the full pick metadata for a recording.
///
/// Returns `(provider, model, streaming, text)` when the pick file
/// exists and parses correctly.  `None` if the file is missing or
/// malformed.
pub fn read_pick(audio_path: &Path) -> Option<(Provider, String, bool, String)> {
    let path = pick_path(audio_path).ok()?;
    let content = fs::read_to_string(&path).ok()?;
    let selection = serde_yaml::from_str::<PickerSelectionMetadata>(&content).ok()?;
    let provider = selection.provider.parse::<Provider>().ok()?;
    Some((
        provider,
        selection.model,
        selection.streaming,
        selection.text,
    ))
}

/// Acquire the pick-level lock for a recording.
///
/// Writes `<stem>.pick-lock.yml` with empty contents.  Returns an
/// error if the lock already exists (another producer is in
/// progress).
pub fn acquire_pick_lock(audio_path: &Path) -> Result<(), TalkError> {
    let path = pick_lock_path(audio_path)?;
    if path.exists() {
        return Err(TalkError::TranscriptInProgress);
    }
    fs::write(&path, b"")
        .map_err(|e| TalkError::Config(format!("failed to write {}: {}", path.display(), e)))
}

/// Release the pick-level lock for a recording.
///
/// Removes `<stem>.pick-lock.yml` if it exists.  Idempotent: missing
/// lock is not an error.
pub fn release_pick_lock(audio_path: &Path) -> Result<(), TalkError> {
    let path = pick_lock_path(audio_path)?;
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(TalkError::Config(format!(
            "failed to remove {}: {}",
            path.display(),
            e
        ))),
    }
}

/// Write a metadata YAML file for a cached recording.
#[allow(clippy::too_many_arguments)]
pub fn write_metadata_to_dir(
    dir: &Path,
    timestamp: &str,
    provider: Provider,
    model: &str,
    realtime: bool,
    transcript: &str,
    audio_filename: &str,
    transcription_metadata: &TranscriptionMetadata,
    segments: Option<&[crate::transcription::TranscriptSegment]>,
    diarization: Option<&[crate::transcription::DiarizationSegment]>,
) -> Result<PathBuf, TalkError> {
    let meta = RecordingMetadata {
        recording: audio_filename.to_string(),
        provider: provider.to_string(),
        model: model.to_string(),
        realtime,
        transcript: transcript.to_string(),
        timestamp: timestamp.to_string(),
        metadata: common_metadata_from_transcription(transcription_metadata),
        provider_api: provider_api_metadata_from_transcription(transcription_metadata),
        segments: common_segments_from_result(segments),
        diarization: common_diarization_from_result(diarization),
    };

    let yaml = serde_yaml::to_string(&meta)
        .map_err(|e| TalkError::Config(format!("failed to serialise recording metadata: {}", e)))?;

    let filename = metadata_filename(timestamp, provider, model, realtime);
    let meta_path = dir.join(&filename);

    fs::write(&meta_path, yaml).map_err(|e| {
        TalkError::Config(format!(
            "failed to write metadata file {}: {}",
            meta_path.display(),
            e
        ))
    })?;

    log::debug!("wrote recording metadata: {}", meta_path.display());
    Ok(meta_path)
}

/// Write a metadata YAML file for a cached recording.
#[allow(clippy::too_many_arguments)]
pub fn write_metadata(
    timestamp: &str,
    provider: Provider,
    model: &str,
    realtime: bool,
    transcript: &str,
    audio_filename: &str,
    transcription_metadata: &TranscriptionMetadata,
    segments: Option<&[crate::transcription::TranscriptSegment]>,
    diarization: Option<&[crate::transcription::DiarizationSegment]>,
) -> Result<PathBuf, TalkError> {
    let dir = ensure_recordings_dir()?;
    let meta_path = write_metadata_to_dir(
        &dir,
        timestamp,
        provider,
        model,
        realtime,
        transcript,
        audio_filename,
        transcription_metadata,
        segments,
        diarization,
    )?;

    let audio_path = dir.join(audio_filename);
    if let Err(e) = write_last_pointers(&audio_path, &meta_path) {
        log::warn!("failed to update last recording pointers: {}", e);
    }

    Ok(meta_path)
}

pub(crate) fn write_last_pointers(
    audio_path: &std::path::Path,
    meta_path: &std::path::Path,
) -> Result<(), TalkError> {
    let dir = ensure_recordings_dir()?;
    let audio_link = dir.join(LAST_RECORDING_POINTER);
    let yml_link = dir.join(LAST_METADATA_POINTER);

    if audio_link.exists() {
        fs::remove_file(&audio_link).map_err(|e| {
            TalkError::Config(format!("failed to replace {}: {}", audio_link.display(), e))
        })?;
    }
    if yml_link.exists() {
        fs::remove_file(&yml_link).map_err(|e| {
            TalkError::Config(format!("failed to replace {}: {}", yml_link.display(), e))
        })?;
    }

    symlink(audio_path, &audio_link).map_err(|e| {
        TalkError::Config(format!(
            "failed to create pointer {}: {}",
            audio_link.display(),
            e
        ))
    })?;
    symlink(meta_path, &yml_link).map_err(|e| {
        TalkError::Config(format!(
            "failed to create pointer {}: {}",
            yml_link.display(),
            e
        ))
    })?;

    Ok(())
}

fn resolve_pointer(name: &str) -> Result<PathBuf, TalkError> {
    let dir = recordings_dir()?;
    let link = dir.join(name);
    if !link.exists() {
        return Err(TalkError::Config(format!(
            "cache pointer not found: {}",
            link.display()
        )));
    }
    let target = fs::read_link(&link).map_err(|e| {
        TalkError::Config(format!("failed to read pointer {}: {}", link.display(), e))
    })?;
    if target.is_absolute() {
        Ok(target)
    } else {
        Ok(dir.join(target))
    }
}

pub fn last_recording_path() -> Result<PathBuf, TalkError> {
    resolve_pointer(LAST_RECORDING_POINTER)
}

pub fn last_metadata_path() -> Result<PathBuf, TalkError> {
    resolve_pointer(LAST_METADATA_POINTER)
}

pub fn latest_recording_path() -> Result<PathBuf, TalkError> {
    let dir = recordings_dir()?;
    let mut oggs: Vec<PathBuf> = Vec::new();
    let entries = fs::read_dir(&dir).map_err(|e| {
        TalkError::Config(format!(
            "failed to read recordings directory {}: {}",
            dir.display(),
            e
        ))
    })?;

    for entry in entries {
        let entry = entry
            .map_err(|e| TalkError::Config(format!("failed to read directory entry: {}", e)))?;
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == LAST_RECORDING_POINTER {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("ogg") && !path.is_symlink() {
            oggs.push(path);
        }
    }

    oggs.sort();
    oggs.pop().ok_or_else(|| {
        TalkError::Config(format!("no cached recordings found in {}", dir.display()))
    })
}

pub fn metadata_path_for_recording(
    audio_path: &std::path::Path,
) -> Result<Option<PathBuf>, TalkError> {
    let stem = audio_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if stem.is_empty() {
        return Ok(None);
    }

    // Search for companion YAMLs matching the stem in a directory.
    let find_yml = |dir: &std::path::Path| -> Result<Vec<PathBuf>, TalkError> {
        let mut ymls = Vec::new();
        let entries = fs::read_dir(dir).map_err(|e| {
            TalkError::Config(format!("failed to read directory {}: {}", dir.display(), e))
        })?;
        for entry in entries {
            let entry = entry
                .map_err(|e| TalkError::Config(format!("failed to read directory entry: {}", e)))?;
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with(stem) && path.extension().and_then(|e| e.to_str()) == Some("yml") {
                ymls.push(path);
            }
        }
        ymls.sort();
        Ok(ymls)
    };

    // First look next to the audio file itself.
    if let Some(audio_dir) = audio_path.parent() {
        let ymls = find_yml(audio_dir)?;
        if let Some(last) = ymls.last() {
            return Ok(Some(last.clone()));
        }
    }

    // Fall back to the recordings cache directory.
    let cache_dir = recordings_dir()?;
    let ymls = find_yml(&cache_dir)?;
    Ok(ymls.last().cloned())
}

pub fn read_metadata_brief(path: &std::path::Path) -> Result<RecordingMetadataBrief, TalkError> {
    let content = fs::read_to_string(path).map_err(|e| {
        TalkError::Config(format!("failed to read metadata {}: {}", path.display(), e))
    })?;
    serde_yaml::from_str::<RecordingMetadataBrief>(&content).map_err(|e| {
        TalkError::Config(format!(
            "failed to parse metadata {}: {}",
            path.display(),
            e
        ))
    })
}

pub fn write_last_paste_state(window_id: Option<&str>, text: &str) -> Result<PathBuf, TalkError> {
    let dir = ensure_recordings_dir()?;
    let state = LastPasteState {
        timestamp: Local::now().format("%Y-%m-%dT%H-%M-%S%z").to_string(),
        char_count: text.chars().count(),
        window_id: window_id.map(ToString::to_string),
        text: text.to_string(),
    };
    let yaml = serde_yaml::to_string(&state)
        .map_err(|e| TalkError::Config(format!("failed to serialise last_paste state: {}", e)))?;
    let path = dir.join(LAST_PASTE_STATE_FILE);
    fs::write(&path, yaml)
        .map_err(|e| TalkError::Config(format!("failed to write {}: {}", path.display(), e)))?;
    Ok(path)
}

pub fn read_last_paste_state() -> Result<Option<LastPasteState>, TalkError> {
    let dir = recordings_dir()?;
    let path = dir.join(LAST_PASTE_STATE_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&path)
        .map_err(|e| TalkError::Config(format!("failed to read {}: {}", path.display(), e)))?;
    let state = serde_yaml::from_str::<LastPasteState>(&content)
        .map_err(|e| TalkError::Config(format!("failed to parse {}: {}", path.display(), e)))?;
    Ok(Some(state))
}

/// Rotate the cache: keep only the most recent `MAX_CACHED_RECORDINGS`
/// recording pairs (OGG + YAML).
///
/// Recordings are identified by their `.ogg` extension.  Each OGG is
/// paired with zero or more `.yml` files sharing the same timestamp
/// prefix.  The oldest pairs beyond the limit are deleted.
pub fn rotate_cache() -> Result<(), TalkError> {
    let dir = match recordings_dir() {
        Ok(d) if d.exists() => d,
        _ => return Ok(()), // Nothing to rotate
    };

    // Collect OGG files sorted by name (which is timestamp-based,
    // so lexicographic order == chronological order).
    let mut oggs: Vec<PathBuf> = Vec::new();
    let entries = fs::read_dir(&dir).map_err(|e| {
        TalkError::Config(format!(
            "failed to read recordings directory {}: {}",
            dir.display(),
            e
        ))
    })?;

    for entry in entries {
        let entry = entry
            .map_err(|e| TalkError::Config(format!("failed to read directory entry: {}", e)))?;
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == LAST_RECORDING_POINTER {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("ogg") && !path.is_symlink() {
            oggs.push(path);
        }
    }

    oggs.sort();

    // If within limit, nothing to do
    if oggs.len() <= MAX_CACHED_RECORDINGS {
        return Ok(());
    }

    // Remove oldest entries beyond the limit
    let to_remove = oggs.len() - MAX_CACHED_RECORDINGS;
    for ogg_path in &oggs[..to_remove] {
        // Extract the timestamp prefix from the OGG filename
        let stem = ogg_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

        // Delete the OGG file
        if let Err(e) = fs::remove_file(ogg_path) {
            log::warn!("failed to remove cached OGG {}: {}", ogg_path.display(), e);
        } else {
            log::debug!("rotated out cached OGG: {}", ogg_path.display());
        }

        // Delete all matching YAML files (same timestamp prefix)
        if !stem.is_empty() {
            if let Ok(entries) = fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if name.starts_with(stem)
                        && path.extension().and_then(|e| e.to_str()) == Some("yml")
                    {
                        if let Err(e) = fs::remove_file(&path) {
                            log::warn!(
                                "failed to remove cached metadata {}: {}",
                                path.display(),
                                e
                            );
                        } else {
                            log::debug!("rotated out cached metadata: {}", path.display());
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_metadata(transcript: &str, provider: &str, model: &str, realtime: bool) -> String {
        serde_yaml::to_string(&RecordingMetadata {
            recording: "sample.ogg".to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
            realtime,
            transcript: transcript.to_string(),
            timestamp: "2026-02-18T12-33-45".to_string(),
            metadata: None,
            provider_api: None,
            segments: None,
            diarization: None,
        })
        .expect("serialise metadata")
    }

    fn sample_result() -> TranscriptionResult {
        TranscriptionResult {
            text: "hello world".to_string(),
            metadata: TranscriptionMetadata {
                request_latency_ms: Some(123),
                session_elapsed_ms: Some(456),
                request_id: Some("req_123".to_string()),
                provider_processing_ms: Some(78),
                detected_language: Some("en".to_string()),
                audio_seconds: Some(2.5),
                segment_count: Some(2),
                word_count: Some(2),
                token_usage: Some(TokenUsage {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                    total_tokens: Some(30),
                }),
                provider_specific: Some(ProviderSpecificMetadata::OpenAI(OpenAIProviderMetadata {
                    model: Some("whisper-1".to_string()),
                    usage_raw: Some(serde_json::json!({"total_tokens": 30})),
                    rate_limit_headers: std::collections::BTreeMap::from([(
                        "x-ratelimit-limit-requests".to_string(),
                        "5000".to_string(),
                    )]),
                    unknown_event_types: vec!["mystery".to_string()],
                    realtime: None,
                })),
            },
            diarization: None,
            segments: Some(vec![
                crate::transcription::TranscriptSegment {
                    start: 0.0,
                    end: 1.0,
                    text: "hello".to_string(),
                },
                crate::transcription::TranscriptSegment {
                    start: 1.0,
                    end: 2.0,
                    text: "world".to_string(),
                },
            ]),
        }
    }

    /// Override the recordings directory for testing by creating files
    /// directly in a temp directory and testing rotation logic.

    #[test]
    fn test_metadata_filename_oneshot() {
        let name = metadata_filename("2026-02-18T12-33-45", Provider::OpenAI, "whisper-1", false);
        assert_eq!(name, "2026-02-18T12-33-45_openai_whisper-1_oneshot.yml");
    }

    #[test]
    fn test_metadata_filename_realtime() {
        let name = metadata_filename(
            "2026-02-18T12-33-45",
            Provider::OpenAI,
            "gpt-4o-mini-transcribe",
            true,
        );
        assert_eq!(
            name,
            "2026-02-18T12-33-45_openai_gpt-4o-mini-transcribe_realtime.yml"
        );
    }

    #[test]
    fn test_metadata_filename_mistral() {
        let name = metadata_filename(
            "2026-02-18T12-33-45",
            Provider::Mistral,
            "voxtral-mini-latest",
            false,
        );
        assert_eq!(
            name,
            "2026-02-18T12-33-45_mistral_voxtral-mini-latest_oneshot.yml"
        );
    }

    #[test]
    fn test_metadata_filename_sanitises_slashes() {
        let name = metadata_filename("2026-02-18T12-33-45", Provider::OpenAI, "org/model", false);
        assert_eq!(name, "2026-02-18T12-33-45_openai_org-model_oneshot.yml");
    }

    #[test]
    fn test_recording_metadata_serialisation() {
        let meta = RecordingMetadata {
            recording: "2026-02-18T12-33-45.ogg".to_string(),
            provider: "openai".to_string(),
            model: "whisper-1".to_string(),
            realtime: false,
            transcript: "Hello world.".to_string(),
            timestamp: "2026-02-18T12-33-45".to_string(),
            metadata: None,
            provider_api: None,
            segments: None,
            diarization: None,
        };
        let yaml = serde_yaml::to_string(&meta).expect("serialise");
        assert!(yaml.contains("recording: 2026-02-18T12-33-45.ogg"));
        assert!(yaml.contains("provider: openai"));
        assert!(yaml.contains("model: whisper-1"));
        assert!(yaml.contains("realtime: false"));
        assert!(yaml.contains("transcript: Hello world."));
        assert!(yaml.contains("timestamp: 2026-02-18T12-33-45"));
    }

    #[test]
    fn test_recording_metadata_serialisation_with_provider_metadata() {
        let meta = RecordingMetadata {
            recording: "2026-02-18T12-33-45.ogg".to_string(),
            provider: "openai".to_string(),
            model: "gpt-4o-transcribe".to_string(),
            realtime: false,
            transcript: "Bonjour".to_string(),
            timestamp: "2026-02-18T12-33-45".to_string(),
            metadata: Some(CommonMetadata {
                request_latency_ms: Some(123),
                session_elapsed_ms: None,
                request_id: Some("req_123".to_string()),
                provider_processing_ms: Some(45),
                detected_language: Some("fr".to_string()),
                audio_seconds: Some(3.2),
                segment_count: Some(2),
                word_count: Some(4),
                token_usage: Some(CommonTokenUsage {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                    total_tokens: Some(30),
                }),
            }),
            provider_api: Some(ProviderApiMetadata {
                openai: Some(OpenAIApiMetadata {
                    model: Some("gpt-4o-transcribe".to_string()),
                    usage: Some(serde_json::json!({"input_tokens": 10})),
                    rate_limit_headers: std::collections::BTreeMap::new(),
                    unknown_event_types: Vec::new(),
                    realtime: None,
                }),
                mistral: None,
            }),
            segments: None,
            diarization: None,
        };

        let yaml = serde_yaml::to_string(&meta).expect("serialise");
        assert!(yaml.contains("metadata:"));
        assert!(yaml.contains("request_latency_ms: 123"));
        assert!(yaml.contains("provider_api:"));
        assert!(yaml.contains("openai:"));
        assert!(yaml.contains("request_id: req_123"));
    }

    #[test]
    fn test_recording_metadata_serialize_with_segments() {
        let meta = RecordingMetadata {
            recording: "2026-02-18T12-33-45.ogg".to_string(),
            provider: "openai".to_string(),
            model: "whisper-1".to_string(),
            realtime: false,
            transcript: "Hello world.".to_string(),
            timestamp: "2026-02-18T12-33-45".to_string(),
            metadata: None,
            provider_api: None,
            diarization: None,
            segments: Some(vec![CommonSegment {
                start: 1.2,
                end: 10.7,
                text: "Hello world.".to_string(),
            }]),
        };

        let yaml = serde_yaml::to_string(&meta).expect("serialise");
        assert!(yaml.contains("segments:\n- start: 1.2\n  end: 10.7\n  text:"));
    }

    #[test]
    fn test_recording_metadata_serialize_without_segments() {
        let meta = RecordingMetadata {
            recording: "2026-02-18T12-33-45.ogg".to_string(),
            provider: "openai".to_string(),
            model: "whisper-1".to_string(),
            realtime: false,
            transcript: "Hello world.".to_string(),
            timestamp: "2026-02-18T12-33-45".to_string(),
            metadata: None,
            provider_api: None,
            segments: None,
            diarization: None,
        };

        let yaml = serde_yaml::to_string(&meta).expect("serialise");
        assert!(!yaml.contains("segments"));
    }

    #[test]
    fn test_transcription_cache_round_trip() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("sample.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");
        let result = sample_result();

        let path =
            TranscriptionCache::store(&audio_path, Provider::OpenAI, "whisper-1", false, &result)
                .expect("store sidecar");

        assert!(path.exists());

        let cached = TranscriptionCache::get(&audio_path, Provider::OpenAI, "whisper-1")
            .expect("read cached sidecar");

        assert_eq!(cached.text, result.text);
        assert_eq!(cached.segments, result.segments);
        assert_eq!(
            cached.metadata.request_latency_ms,
            result.metadata.request_latency_ms
        );
        assert_eq!(
            cached.metadata.session_elapsed_ms,
            result.metadata.session_elapsed_ms
        );
        assert_eq!(cached.metadata.request_id, result.metadata.request_id);
        assert_eq!(
            cached.metadata.provider_processing_ms,
            result.metadata.provider_processing_ms
        );
        assert_eq!(
            cached.metadata.detected_language,
            result.metadata.detected_language
        );
        assert_eq!(cached.metadata.audio_seconds, result.metadata.audio_seconds);
        assert_eq!(cached.metadata.segment_count, result.metadata.segment_count);
        assert_eq!(cached.metadata.word_count, result.metadata.word_count);
        assert!(cached.metadata.token_usage.is_some());
        let token_usage = cached
            .metadata
            .token_usage
            .expect("token usage survives round-trip");
        assert_eq!(token_usage.input_tokens, Some(10));
        assert_eq!(token_usage.output_tokens, Some(20));
        assert_eq!(token_usage.total_tokens, Some(30));
        assert!(cached.metadata.provider_specific.is_none());
    }

    #[test]
    fn test_transcription_cache_get_miss_returns_none() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("missing.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");

        let cached = TranscriptionCache::get(&audio_path, Provider::Mistral, "voxtral-mini-latest");

        assert!(cached.is_none());
    }

    #[test]
    fn test_rotate_cache_removes_oldest() {
        let dir = TempDir::new().expect("create temp dir");
        let rec_dir = dir.path();

        // Create 12 fake OGG + YML pairs
        for i in 0..12 {
            let ts = format!("2026-02-{:02}T12-00-00", i + 1);
            let ogg = rec_dir.join(format!("{}.ogg", ts));
            let yml = rec_dir.join(format!("{}_openai_whisper-1_batch.yml", ts));
            fs::write(&ogg, "fake ogg").expect("write ogg");
            fs::write(&yml, "fake yml").expect("write yml");
        }

        // Manually run rotation on this directory
        rotate_in_dir(rec_dir, MAX_CACHED_RECORDINGS).expect("rotate");

        // Count remaining OGG files
        let remaining_oggs: Vec<_> = fs::read_dir(rec_dir)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|ext| ext.to_str()) == Some("ogg"))
            .collect();

        assert_eq!(remaining_oggs.len(), MAX_CACHED_RECORDINGS);

        // Verify the oldest 2 were removed
        assert!(!rec_dir.join("2026-02-01T12-00-00.ogg").exists());
        assert!(!rec_dir
            .join("2026-02-01T12-00-00_openai_whisper-1_batch.yml")
            .exists());
        assert!(!rec_dir.join("2026-02-02T12-00-00.ogg").exists());
        assert!(!rec_dir
            .join("2026-02-02T12-00-00_openai_whisper-1_batch.yml")
            .exists());

        // Verify the newest are still there
        assert!(rec_dir.join("2026-02-12T12-00-00.ogg").exists());
        assert!(rec_dir
            .join("2026-02-12T12-00-00_openai_whisper-1_batch.yml")
            .exists());
    }

    #[test]
    fn test_rotate_cache_removes_pick_selection_sidecar() {
        let dir = TempDir::new().expect("create temp dir");
        let rec_dir = dir.path();

        for i in 0..12 {
            let ts = format!("2026-02-{:02}T12-00-00", i + 1);
            fs::write(rec_dir.join(format!("{ts}.ogg")), "fake ogg").expect("write ogg");
            fs::write(
                rec_dir.join(format!("{ts}.pick.yml")),
                "provider: openai\nmodel: whisper-1\nstreaming: false\n",
            )
            .expect("write selection");
        }

        rotate_in_dir(rec_dir, MAX_CACHED_RECORDINGS).expect("rotate");

        assert!(!rec_dir.join("2026-02-01T12-00-00.pick.yml").exists());
        assert!(!rec_dir.join("2026-02-02T12-00-00.pick.yml").exists());
        assert!(rec_dir.join("2026-02-12T12-00-00.pick.yml").exists());
    }

    #[test]
    fn test_rotate_cache_under_limit_is_noop() {
        let dir = TempDir::new().expect("create temp dir");
        let rec_dir = dir.path();

        // Create 5 OGG files (under the limit)
        for i in 0..5 {
            let ts = format!("2026-02-{:02}T12-00-00", i + 1);
            let ogg = rec_dir.join(format!("{}.ogg", ts));
            fs::write(&ogg, "fake ogg").expect("write ogg");
        }

        rotate_in_dir(rec_dir, MAX_CACHED_RECORDINGS).expect("rotate");

        let remaining: Vec<_> = fs::read_dir(rec_dir)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .collect();

        assert_eq!(remaining.len(), 5);
    }

    #[test]
    fn test_rotate_cache_empty_dir() {
        let dir = TempDir::new().expect("create temp dir");
        rotate_in_dir(dir.path(), MAX_CACHED_RECORDINGS).expect("rotate");
    }

    #[test]
    fn test_read_metadata_brief_transcript() {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("meta.yml");
        let content = "recording: x.ogg\nprovider: openai\nmodel: whisper-1\nrealtime: false\ntranscript: hello world\ntimestamp: 2026-01-01T00-00-00\n";
        fs::write(&path, content).expect("write metadata");

        let brief = read_metadata_brief(&path).expect("read brief");
        assert_eq!(brief.transcript, "hello world");
        assert_eq!(brief.provider.as_deref(), Some("openai"));
        assert_eq!(brief.model.as_deref(), Some("whisper-1"));
    }

    #[test]
    fn test_read_metadata_brief_without_provider_model() {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("meta.yml");
        let content = "transcript: legacy entry\n";
        fs::write(&path, content).expect("write metadata");

        let brief = read_metadata_brief(&path).expect("read brief");
        assert_eq!(brief.transcript, "legacy entry");
        assert_eq!(brief.provider, None);
        assert_eq!(brief.model, None);
    }

    #[test]
    fn test_list_sidecars_for_audio_reads_audio_dir_and_skips_pick_file() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("sample.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");
        fs::write(
            dir.path().join("sample_openai_whisper-1_batch.yml"),
            sample_metadata("hello", "openai", "whisper-1", false),
        )
        .expect("write sidecar");
        fs::write(
            dir.path().join("sample.pick.yml"),
            "provider: mistral\nmodel: voxtral-mini-2507\nstreaming: false\n",
        )
        .expect("write selection");

        let sidecars = list_sidecars_for_audio(&audio_path);

        assert_eq!(sidecars.len(), 1);
        assert_eq!(
            sidecars[0],
            (
                Provider::OpenAI,
                "whisper-1".to_string(),
                "hello".to_string(),
                false
            )
        );
    }

    #[test]
    fn test_list_sidecars_for_audio_uses_recordings_dir_as_fallback() {
        let dir = TempDir::new().expect("create temp dir");
        let recordings = recordings_dir().expect("recordings dir");
        fs::create_dir_all(&recordings).expect("create recordings dir");
        let audio_path = dir.path().join("fallback.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");
        let fallback_sidecar = recordings.join("fallback_mistral_voxtral-mini-2507_batch.yml");
        fs::write(
            &fallback_sidecar,
            sample_metadata("bonjour", "mistral", "voxtral-mini-2507", false),
        )
        .expect("write fallback sidecar");

        let sidecars = list_sidecars_for_audio(&audio_path);

        assert_eq!(sidecars.len(), 1);
        assert_eq!(
            sidecars[0],
            (
                Provider::Mistral,
                "voxtral-mini-2507".to_string(),
                "bonjour".to_string(),
                false
            )
        );

        let _ = fs::remove_file(fallback_sidecar);
    }

    #[test]
    fn test_list_sidecars_for_audio_deduplicates_provider_model_mode() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("dup.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");
        fs::write(
            dir.path().join("dup_openai_whisper-1_batch.yml"),
            sample_metadata("first", "openai", "whisper-1", false),
        )
        .expect("write first sidecar");
        fs::write(
            dir.path().join("dup_openai_whisper-1_realtime.yml"),
            sample_metadata("stream", "openai", "whisper-1", true),
        )
        .expect("write realtime sidecar");
        fs::write(
            dir.path().join("dup_openai_whisper-1_batch-copy.yml"),
            sample_metadata("second", "openai", "whisper-1", false),
        )
        .expect("write duplicate sidecar");

        let sidecars = list_sidecars_for_audio(&audio_path);

        assert_eq!(sidecars.len(), 2);
        assert!(
            sidecars.contains(&(
                Provider::OpenAI,
                "whisper-1".to_string(),
                "first".to_string(),
                false
            )) || sidecars.contains(&(
                Provider::OpenAI,
                "whisper-1".to_string(),
                "second".to_string(),
                false
            ))
        );
        assert!(sidecars.contains(&(
            Provider::OpenAI,
            "whisper-1".to_string(),
            "stream".to_string(),
            true
        )));
    }

    #[test]
    fn test_pick_round_trip() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("select.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");

        write_pick(&audio_path, "openai", "whisper-1", true, "hello world").expect("write pick");

        assert_eq!(
            read_pick(&audio_path),
            Some((
                Provider::OpenAI,
                "whisper-1".to_string(),
                true,
                "hello world".to_string()
            ))
        );
    }

    #[test]
    fn test_get_transcript_not_available_when_no_files() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("nothing.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");

        assert_eq!(get_transcript(&audio_path), TranscriptStatus::NotAvailable);
    }

    #[test]
    fn test_get_transcript_available_when_pick_exists() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("has-pick.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");
        write_pick(&audio_path, "openai", "whisper-1", false, "hello").expect("write pick");

        assert_eq!(
            get_transcript(&audio_path),
            TranscriptStatus::Available("hello".to_string())
        );
    }

    #[test]
    fn test_get_transcript_available_with_empty_text() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("silent.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");
        write_pick(&audio_path, "openai", "whisper-1", false, "").expect("write pick");

        assert_eq!(
            get_transcript(&audio_path),
            TranscriptStatus::Available(String::new())
        );
    }

    #[test]
    fn test_get_transcript_in_progress_when_lock_exists() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("locked.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");
        acquire_pick_lock(&audio_path).expect("acquire lock");

        assert_eq!(get_transcript(&audio_path), TranscriptStatus::InProgress);
    }

    #[test]
    fn test_get_transcript_lock_takes_priority_over_pick() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("both.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");
        // Production is overwriting: pick exists from previous run,
        // lock means a new producer is mid-flight.
        write_pick(&audio_path, "openai", "whisper-1", false, "stale").expect("write pick");
        acquire_pick_lock(&audio_path).expect("acquire lock");

        assert_eq!(get_transcript(&audio_path), TranscriptStatus::InProgress);
    }

    #[test]
    fn test_acquire_pick_lock_fails_when_already_held() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("lock-test.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");
        acquire_pick_lock(&audio_path).expect("first acquire");

        let result = acquire_pick_lock(&audio_path);
        assert!(matches!(result, Err(TalkError::TranscriptInProgress)));
    }

    #[test]
    fn test_release_pick_lock_removes_file() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("release-test.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");
        acquire_pick_lock(&audio_path).expect("acquire");
        assert_eq!(get_transcript(&audio_path), TranscriptStatus::InProgress);

        release_pick_lock(&audio_path).expect("release");
        assert_eq!(get_transcript(&audio_path), TranscriptStatus::NotAvailable);
    }

    #[test]
    fn test_release_pick_lock_is_idempotent() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("no-lock.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");

        // No lock to begin with — should not error.
        release_pick_lock(&audio_path).expect("release missing lock");
    }

    /// Testable rotation function that operates on an arbitrary directory.
    fn rotate_in_dir(dir: &std::path::Path, max: usize) -> Result<(), TalkError> {
        let mut oggs: Vec<PathBuf> = Vec::new();
        let entries = fs::read_dir(dir).map_err(|e| {
            TalkError::Config(format!(
                "failed to read recordings directory {}: {}",
                dir.display(),
                e
            ))
        })?;

        for entry in entries {
            let entry = entry
                .map_err(|e| TalkError::Config(format!("failed to read directory entry: {}", e)))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("ogg") {
                oggs.push(path);
            }
        }

        oggs.sort();

        if oggs.len() <= max {
            return Ok(());
        }

        let to_remove = oggs.len() - max;
        for ogg_path in &oggs[..to_remove] {
            let stem = ogg_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

            let _ = fs::remove_file(ogg_path);

            if !stem.is_empty() {
                if let Ok(entries) = fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if name.starts_with(stem)
                            && path.extension().and_then(|e| e.to_str()) == Some("yml")
                        {
                            let _ = fs::remove_file(&path);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_acquire_model_lock_fails_when_already_held() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("mlock.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");

        acquire_model_lock(&audio_path, Provider::OpenAI, "whisper-1", false)
            .expect("first acquire");
        let result = acquire_model_lock(&audio_path, Provider::OpenAI, "whisper-1", false);
        assert!(matches!(result, Err(TalkError::ModelInProgress)));
    }

    #[test]
    fn test_acquire_model_lock_independent_per_model() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("independent.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");

        acquire_model_lock(&audio_path, Provider::OpenAI, "whisper-1", false)
            .expect("lock whisper");
        // Different model — should succeed.
        acquire_model_lock(&audio_path, Provider::OpenAI, "gpt-4o", false).expect("lock gpt-4o");
        // Same model but realtime mode — different lock.
        acquire_model_lock(&audio_path, Provider::OpenAI, "whisper-1", true)
            .expect("lock whisper realtime");
    }

    #[test]
    fn test_release_model_lock_is_idempotent() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("rel.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");

        release_model_lock(&audio_path, Provider::OpenAI, "whisper-1", false)
            .expect("release missing lock");
    }

    #[test]
    fn test_is_model_locked() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("check.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");

        assert!(!is_model_locked(
            &audio_path,
            Provider::OpenAI,
            "whisper-1",
            false
        ));
        acquire_model_lock(&audio_path, Provider::OpenAI, "whisper-1", false).expect("acquire");
        assert!(is_model_locked(
            &audio_path,
            Provider::OpenAI,
            "whisper-1",
            false
        ));
        release_model_lock(&audio_path, Provider::OpenAI, "whisper-1", false).expect("release");
        assert!(!is_model_locked(
            &audio_path,
            Provider::OpenAI,
            "whisper-1",
            false
        ));
    }

    #[test]
    fn test_transcription_cache_round_trip_with_diarization() {
        let dir = TempDir::new().expect("create temp dir");
        let audio_path = dir.path().join("sample.ogg");
        fs::write(&audio_path, b"fake ogg").expect("write audio");

        let mut result = sample_result();
        result.diarization = Some(vec![
            crate::transcription::DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start: 0.0,
                end: 2.5,
                text: "Bonjour tout le monde".to_string(),
            },
            crate::transcription::DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start: 2.5,
                end: 5.0,
                text: "Hello, comment ca va ?".to_string(),
            },
        ]);

        let path = TranscriptionCache::store(
            &audio_path,
            Provider::Mistral,
            "voxtral-mini-2602",
            false,
            &result,
        )
        .expect("store sidecar");

        assert!(path.exists());

        let cached = TranscriptionCache::get(&audio_path, Provider::Mistral, "voxtral-mini-2602")
            .expect("read cached sidecar");

        // When diarization is present, store() formats text with speakers.
        // We verify the diarization segments survive round-trip — that's the real bug.
        let diarization = cached
            .diarization
            .expect("diarization should survive round-trip");
        assert_eq!(diarization.len(), 2);
        assert_eq!(diarization[0].speaker, "SPEAKER_00");
        assert_eq!(diarization[0].text, "Bonjour tout le monde");
        assert_eq!(diarization[1].speaker, "SPEAKER_01");
        assert_eq!(diarization[1].text, "Hello, comment ca va ?");

        // Verify YAML contains speaker labels
        let yaml = fs::read_to_string(&path).expect("read yaml");
        assert!(yaml.contains("speaker: SPEAKER_00"));
        assert!(yaml.contains("speaker: SPEAKER_01"));
    }
}
