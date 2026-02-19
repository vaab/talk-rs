//! Picker transcription result cache.
//!
//! When the picker window (`--pick`) fires parallel transcription
//! requests, the results are persisted to disk so that reopening the
//! picker for the **same audio file** returns instantly without any
//! API calls.
//!
//! The cache also remembers which entry the user last selected so
//! that it appears pre-selected the next time the picker opens.
//!
//! Cache files live in `~/.cache/talk-rs/picker-results/` and are
//! named after the audio file stem (e.g.
//! `20250219-081500.json` for `20250219-081500.wav`).

use crate::core::daemon::cache_dir;
use crate::core::error::TalkError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level cache structure for one audio file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PickerCache {
    /// All transcription results.
    pub results: Vec<CachedResult>,
    /// The (provider, model) pair the user last selected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected: Option<SelectedEntry>,
}

/// Identifies the last-selected picker entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedEntry {
    pub provider: String,
    pub model: String,
}

/// A single cached transcription result for one (provider, model) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    pub provider: String,
    pub model: String,
    pub text: String,
}

/// Return the cache file path for a given audio file.
///
/// The cache is stored alongside other talk-rs caches in a dedicated
/// `picker-results/` subdirectory, named after the audio file stem.
fn cache_path_for(audio_path: &Path) -> Result<PathBuf, TalkError> {
    let stem = audio_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    Ok(cache_dir()?
        .join("picker-results")
        .join(format!("{stem}.json")))
}

/// Read previously cached picker data for an audio file.
///
/// Returns a default (empty) [`PickerCache`] when no cache exists or
/// on any read/parse error — a missing cache simply means every
/// provider will be queried again.
///
/// Transparently migrates the old flat-array format to the current
/// [`PickerCache`] struct.
pub fn read(audio_path: &Path) -> PickerCache {
    let cache_path = match cache_path_for(audio_path) {
        Ok(p) => p,
        Err(e) => {
            log::debug!("picker cache path error: {}", e);
            return PickerCache::default();
        }
    };
    if !cache_path.exists() {
        return PickerCache::default();
    }
    let contents = match std::fs::read_to_string(&cache_path) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("picker cache read error: {}", e);
            return PickerCache::default();
        }
    };

    // Current format (object with `results` + optional `selected`).
    if let Ok(cache) = serde_json::from_str::<PickerCache>(&contents) {
        log::info!(
            "loaded {} cached picker result(s) for {}",
            cache.results.len(),
            audio_path.display(),
        );
        return cache;
    }

    // Legacy format: bare JSON array of results (auto-migrate).
    if let Ok(results) = serde_json::from_str::<Vec<CachedResult>>(&contents) {
        log::info!(
            "migrated {} legacy picker result(s) for {}",
            results.len(),
            audio_path.display(),
        );
        return PickerCache {
            results,
            selected: None,
        };
    }

    log::warn!("picker cache parse error for {}", audio_path.display());
    PickerCache::default()
}

// ── Write helpers ───────────────────────────────────────────────────

/// Ensure the parent directory exists, then write `cache` as JSON.
fn write_to_disk(audio_path: &Path, cache: &PickerCache) -> Result<(), TalkError> {
    let cache_path = cache_path_for(audio_path)?;
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            TalkError::Config(format!(
                "failed to create picker cache directory {}: {}",
                parent.display(),
                e,
            ))
        })?;
    }
    let json = serde_json::to_string_pretty(cache)
        .map_err(|e| TalkError::Config(format!("failed to serialise picker cache: {e}")))?;
    std::fs::write(&cache_path, json).map_err(|e| {
        TalkError::Config(format!(
            "failed to write picker cache {}: {}",
            cache_path.display(),
            e,
        ))
    })?;
    log::info!(
        "wrote {} picker result(s) to {}",
        cache.results.len(),
        cache_path.display(),
    );
    Ok(())
}

/// Persist transcription results, preserving the existing `selected`
/// marker.
///
/// Only successful transcriptions (non-empty text) should be passed.
pub fn write_results(audio_path: &Path, results: &[CachedResult]) -> Result<(), TalkError> {
    if results.is_empty() {
        return Ok(());
    }
    let mut cache = read(audio_path);
    cache.results = results.to_vec();
    write_to_disk(audio_path, &cache)
}

/// Record which (provider, model) the user last selected, preserving
/// the existing results.
pub fn write_selected(audio_path: &Path, provider: &str, model: &str) -> Result<(), TalkError> {
    let mut cache = read(audio_path);
    cache.selected = Some(SelectedEntry {
        provider: provider.to_string(),
        model: model.to_string(),
    });
    write_to_disk(audio_path, &cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_missing_file_returns_default() {
        let cache = read(Path::new("/tmp/nonexistent-audio-file.wav"));
        assert!(cache.results.is_empty());
        assert!(cache.selected.is_none());
    }

    #[test]
    fn test_roundtrip_new_format() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let cache_file = dir.path().join("test.json");

        let cache = PickerCache {
            results: vec![
                CachedResult {
                    provider: "openai".into(),
                    model: "whisper-1".into(),
                    text: "hello world".into(),
                },
                CachedResult {
                    provider: "mistral".into(),
                    model: "voxtral-mini-2507".into(),
                    text: "bonjour monde".into(),
                },
            ],
            selected: Some(SelectedEntry {
                provider: "mistral".into(),
                model: "voxtral-mini-2507".into(),
            }),
        };

        let json = serde_json::to_string_pretty(&cache).expect("serialise");
        std::fs::write(&cache_file, &json).expect("write");

        let read_back: PickerCache =
            serde_json::from_str(&std::fs::read_to_string(&cache_file).expect("read"))
                .expect("parse");
        assert_eq!(read_back.results.len(), 2);
        assert_eq!(read_back.results[0].text, "hello world");
        let sel = read_back.selected.expect("selected present");
        assert_eq!(sel.provider, "mistral");
        assert_eq!(sel.model, "voxtral-mini-2507");
    }

    #[test]
    fn test_legacy_format_migration() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let cache_file = dir.path().join("legacy.json");

        // Write old flat-array format.
        let legacy = vec![CachedResult {
            provider: "openai".into(),
            model: "whisper-1".into(),
            text: "legacy text".into(),
        }];
        let json = serde_json::to_string_pretty(&legacy).expect("serialise");
        std::fs::write(&cache_file, &json).expect("write");

        // New format parse should fail (it's an array, not an object).
        let contents = std::fs::read_to_string(&cache_file).expect("read");
        assert!(serde_json::from_str::<PickerCache>(&contents).is_err());
        // Legacy parse should succeed.
        let results: Vec<CachedResult> = serde_json::from_str(&contents).expect("legacy parse");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "legacy text");
    }

    #[test]
    fn test_read_corrupt_file_returns_default() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let cache_file = dir.path().join("corrupt.json");
        std::fs::write(&cache_file, "not valid json{{{").expect("write");

        let contents = std::fs::read_to_string(&cache_file).expect("read");
        assert!(serde_json::from_str::<PickerCache>(&contents).is_err());
        assert!(serde_json::from_str::<Vec<CachedResult>>(&contents).is_err());
    }

    #[test]
    fn test_selected_none_omitted_in_json() {
        let cache = PickerCache {
            results: vec![CachedResult {
                provider: "openai".into(),
                model: "whisper-1".into(),
                text: "test".into(),
            }],
            selected: None,
        };
        let json = serde_json::to_string(&cache).expect("serialise");
        assert!(!json.contains("selected"));
    }

    #[test]
    fn test_write_results_empty_is_noop() {
        let result = write_results(Path::new("/tmp/noop.wav"), &[]);
        assert!(result.is_ok());
    }
}
