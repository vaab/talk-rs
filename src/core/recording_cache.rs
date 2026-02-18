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
use chrono::Local;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

/// Maximum number of recordings to keep in the cache.
const MAX_CACHED_RECORDINGS: usize = 10;

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
) -> Result<PathBuf, TalkError> {
    let dir = ensure_recordings_dir()?;

    let meta = RecordingMetadata {
        recording: wav_filename.to_string(),
        provider: provider.to_string(),
        model: model.to_string(),
        realtime,
        transcript: transcript.to_string(),
        timestamp: timestamp.to_string(),
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
        if path.extension().and_then(|e| e.to_str()) == Some("wav") {
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
