//! Recording cache management for talk-rs.
//!
//! Keeps the last N recordings in `~/.cache/talk-rs/recordings/` with
//! companion YAML metadata files.  Each recording is a WAV file named
//! with an ISO-8601 timestamp; the metadata file encodes the provider,
//! model, and mode in its filename and contains the transcript plus
//! recording metadata.

use crate::core::config::Provider;
use crate::core::daemon::cache_dir;
use crate::core::error::TalkError;
use crate::core::transcription::{
    MistralProviderMetadata, OpenAIProviderMetadata, OpenAIRealtimeMetadata,
    ProviderSpecificMetadata, TokenUsage, TranscriptionMetadata,
};
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::fs::symlink;
use std::path::PathBuf;

/// Maximum number of recordings to keep in the cache.
const MAX_CACHED_RECORDINGS: usize = 10;
const LAST_RECORDING_POINTER: &str = "last_recording.wav";
const LAST_METADATA_POINTER: &str = "last_metadata.yml";
const LAST_PASTE_STATE_FILE: &str = "last_paste.yml";

/// Metadata associated with a cached recording.
#[derive(Debug, Serialize)]
pub struct RecordingMetadata {
    /// Filename of the WAV recording (basename only).
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<CommonMetadata>,
    /// Provider-specific metadata payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_api: Option<ProviderApiMetadata>,
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

#[derive(Debug, Serialize)]
pub struct CommonMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_elapsed_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_processing_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<CommonTokenUsage>,
}

#[derive(Debug, Serialize)]
pub struct CommonTokenUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ProviderApiMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai: Option<OpenAIApiMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mistral: Option<MistralApiMetadata>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIApiMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub rate_limit_headers: std::collections::BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unknown_event_types: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub realtime: Option<OpenAIRealtimeApiMetadata>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIRealtimeApiMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub event_counts: std::collections::BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_rate_limits: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub ws_upgrade_headers: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct MistralApiMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unknown_event_types: Vec<String>,
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

/// Generate a timestamped WAV path in the recordings cache directory.
///
/// Returns `(wav_path, timestamp_string)` so the same timestamp can be
/// used for both the WAV and metadata filenames.
pub fn generate_recording_path() -> Result<(PathBuf, String), TalkError> {
    let dir = ensure_recordings_dir()?;
    let now = Local::now();
    let ts = now.format("%Y-%m-%dT%H-%M-%S").to_string();
    let wav_path = dir.join(format!("{}.wav", ts));
    Ok((wav_path, ts))
}

/// Build the metadata YAML filename from components.
///
/// Format: `{timestamp}_{provider}_{model}_{mode}.yml`
/// where mode is "realtime" or "batch".
fn metadata_filename(timestamp: &str, provider: Provider, model: &str, realtime: bool) -> String {
    let mode = if realtime { "realtime" } else { "batch" };
    // Sanitise model name: replace `/` and spaces with `-`
    let safe_model = model.replace(['/', ' '], "-");
    format!("{}_{}_{}_{}.yml", timestamp, provider, safe_model, mode)
}

/// Write a metadata YAML file for a cached recording.
pub fn write_metadata(
    timestamp: &str,
    provider: Provider,
    model: &str,
    realtime: bool,
    transcript: &str,
    wav_filename: &str,
    transcription_metadata: &TranscriptionMetadata,
) -> Result<PathBuf, TalkError> {
    let dir = ensure_recordings_dir()?;

    let meta = RecordingMetadata {
        recording: wav_filename.to_string(),
        provider: provider.to_string(),
        model: model.to_string(),
        realtime,
        transcript: transcript.to_string(),
        timestamp: timestamp.to_string(),
        metadata: common_metadata_from_transcription(transcription_metadata),
        provider_api: provider_api_metadata_from_transcription(transcription_metadata),
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

    let wav_path = dir.join(wav_filename);
    if let Err(e) = write_last_pointers(&wav_path, &meta_path) {
        log::warn!("failed to update last recording pointers: {}", e);
    }

    log::debug!("wrote recording metadata: {}", meta_path.display());
    Ok(meta_path)
}

fn write_last_pointers(
    wav_path: &std::path::Path,
    meta_path: &std::path::Path,
) -> Result<(), TalkError> {
    let dir = ensure_recordings_dir()?;
    let wav_link = dir.join(LAST_RECORDING_POINTER);
    let yml_link = dir.join(LAST_METADATA_POINTER);

    if wav_link.exists() {
        fs::remove_file(&wav_link).map_err(|e| {
            TalkError::Config(format!("failed to replace {}: {}", wav_link.display(), e))
        })?;
    }
    if yml_link.exists() {
        fs::remove_file(&yml_link).map_err(|e| {
            TalkError::Config(format!("failed to replace {}: {}", yml_link.display(), e))
        })?;
    }

    symlink(wav_path, &wav_link).map_err(|e| {
        TalkError::Config(format!(
            "failed to create pointer {}: {}",
            wav_link.display(),
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
    let mut wavs: Vec<PathBuf> = Vec::new();
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
        if path.extension().and_then(|e| e.to_str()) == Some("wav") && !path.is_symlink() {
            wavs.push(path);
        }
    }

    wavs.sort();
    wavs.pop().ok_or_else(|| {
        TalkError::Config(format!("no cached recordings found in {}", dir.display()))
    })
}

pub fn metadata_path_for_recording(
    wav_path: &std::path::Path,
) -> Result<Option<PathBuf>, TalkError> {
    let dir = recordings_dir()?;
    let stem = wav_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if stem.is_empty() {
        return Ok(None);
    }
    let mut ymls: Vec<PathBuf> = Vec::new();
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
        if name.starts_with(stem) && path.extension().and_then(|e| e.to_str()) == Some("yml") {
            ymls.push(path);
        }
    }
    ymls.sort();
    Ok(ymls.pop())
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
        timestamp: Local::now().format("%Y-%m-%dT%H-%M-%S").to_string(),
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
/// recording pairs (WAV + YAML).
///
/// Recordings are identified by their `.wav` extension.  Each WAV is
/// paired with zero or more `.yml` files sharing the same timestamp
/// prefix.  The oldest pairs beyond the limit are deleted.
pub fn rotate_cache() -> Result<(), TalkError> {
    let dir = match recordings_dir() {
        Ok(d) if d.exists() => d,
        _ => return Ok(()), // Nothing to rotate
    };

    // Collect WAV files sorted by name (which is timestamp-based,
    // so lexicographic order == chronological order).
    let mut wavs: Vec<PathBuf> = Vec::new();
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
        if path.extension().and_then(|e| e.to_str()) == Some("wav") && !path.is_symlink() {
            wavs.push(path);
        }
    }

    wavs.sort();

    // If within limit, nothing to do
    if wavs.len() <= MAX_CACHED_RECORDINGS {
        return Ok(());
    }

    // Remove oldest entries beyond the limit
    let to_remove = wavs.len() - MAX_CACHED_RECORDINGS;
    for wav_path in &wavs[..to_remove] {
        // Extract the timestamp prefix from the WAV filename
        let stem = wav_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

        // Delete the WAV file
        if let Err(e) = fs::remove_file(wav_path) {
            log::warn!("failed to remove cached WAV {}: {}", wav_path.display(), e);
        } else {
            log::debug!("rotated out cached WAV: {}", wav_path.display());
        }

        // Delete all matching YAML files (same timestamp prefix)
        if !stem.is_empty() {
            let yml_prefix = format!("{}_", stem);
            if let Ok(entries) = fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if name.starts_with(&yml_prefix)
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

    /// Override the recordings directory for testing by creating files
    /// directly in a temp directory and testing rotation logic.

    #[test]
    fn test_metadata_filename_batch() {
        let name = metadata_filename("2026-02-18T12-33-45", Provider::OpenAI, "whisper-1", false);
        assert_eq!(name, "2026-02-18T12-33-45_openai_whisper-1_batch.yml");
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
            "2026-02-18T12-33-45_mistral_voxtral-mini-latest_batch.yml"
        );
    }

    #[test]
    fn test_metadata_filename_sanitises_slashes() {
        let name = metadata_filename("2026-02-18T12-33-45", Provider::OpenAI, "org/model", false);
        assert_eq!(name, "2026-02-18T12-33-45_openai_org-model_batch.yml");
    }

    #[test]
    fn test_recording_metadata_serialisation() {
        let meta = RecordingMetadata {
            recording: "2026-02-18T12-33-45.wav".to_string(),
            provider: "openai".to_string(),
            model: "whisper-1".to_string(),
            realtime: false,
            transcript: "Hello world.".to_string(),
            timestamp: "2026-02-18T12-33-45".to_string(),
            metadata: None,
            provider_api: None,
        };
        let yaml = serde_yaml::to_string(&meta).expect("serialise");
        assert!(yaml.contains("recording: 2026-02-18T12-33-45.wav"));
        assert!(yaml.contains("provider: openai"));
        assert!(yaml.contains("model: whisper-1"));
        assert!(yaml.contains("realtime: false"));
        assert!(yaml.contains("transcript: Hello world."));
        assert!(yaml.contains("timestamp: 2026-02-18T12-33-45"));
    }

    #[test]
    fn test_recording_metadata_serialisation_with_provider_metadata() {
        let meta = RecordingMetadata {
            recording: "2026-02-18T12-33-45.wav".to_string(),
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
        };

        let yaml = serde_yaml::to_string(&meta).expect("serialise");
        assert!(yaml.contains("metadata:"));
        assert!(yaml.contains("request_latency_ms: 123"));
        assert!(yaml.contains("provider_api:"));
        assert!(yaml.contains("openai:"));
        assert!(yaml.contains("request_id: req_123"));
    }

    #[test]
    fn test_rotate_cache_removes_oldest() {
        let dir = TempDir::new().expect("create temp dir");
        let rec_dir = dir.path();

        // Create 12 fake WAV + YML pairs
        for i in 0..12 {
            let ts = format!("2026-02-{:02}T12-00-00", i + 1);
            let wav = rec_dir.join(format!("{}.wav", ts));
            let yml = rec_dir.join(format!("{}_openai_whisper-1_batch.yml", ts));
            fs::write(&wav, "fake wav").expect("write wav");
            fs::write(&yml, "fake yml").expect("write yml");
        }

        // Manually run rotation on this directory
        rotate_in_dir(rec_dir, MAX_CACHED_RECORDINGS).expect("rotate");

        // Count remaining WAV files
        let remaining_wavs: Vec<_> = fs::read_dir(rec_dir)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|ext| ext.to_str()) == Some("wav"))
            .collect();

        assert_eq!(remaining_wavs.len(), MAX_CACHED_RECORDINGS);

        // Verify the oldest 2 were removed
        assert!(!rec_dir.join("2026-02-01T12-00-00.wav").exists());
        assert!(!rec_dir
            .join("2026-02-01T12-00-00_openai_whisper-1_batch.yml")
            .exists());
        assert!(!rec_dir.join("2026-02-02T12-00-00.wav").exists());
        assert!(!rec_dir
            .join("2026-02-02T12-00-00_openai_whisper-1_batch.yml")
            .exists());

        // Verify the newest are still there
        assert!(rec_dir.join("2026-02-12T12-00-00.wav").exists());
        assert!(rec_dir
            .join("2026-02-12T12-00-00_openai_whisper-1_batch.yml")
            .exists());
    }

    #[test]
    fn test_rotate_cache_under_limit_is_noop() {
        let dir = TempDir::new().expect("create temp dir");
        let rec_dir = dir.path();

        // Create 5 WAV files (under the limit)
        for i in 0..5 {
            let ts = format!("2026-02-{:02}T12-00-00", i + 1);
            let wav = rec_dir.join(format!("{}.wav", ts));
            fs::write(&wav, "fake wav").expect("write wav");
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
        let content = "recording: x.wav\nprovider: openai\nmodel: whisper-1\nrealtime: false\ntranscript: hello world\ntimestamp: 2026-01-01T00-00-00\n";
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

    /// Testable rotation function that operates on an arbitrary directory.
    fn rotate_in_dir(dir: &std::path::Path, max: usize) -> Result<(), TalkError> {
        let mut wavs: Vec<PathBuf> = Vec::new();
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
            if path.extension().and_then(|e| e.to_str()) == Some("wav") {
                wavs.push(path);
            }
        }

        wavs.sort();

        if wavs.len() <= max {
            return Ok(());
        }

        let to_remove = wavs.len() - max;
        for wav_path in &wavs[..to_remove] {
            let stem = wav_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

            let _ = fs::remove_file(wav_path);

            if !stem.is_empty() {
                let yml_prefix = format!("{}_", stem);
                if let Ok(entries) = fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if name.starts_with(&yml_prefix)
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
}
