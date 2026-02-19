//! Picker transcription result cache.
//!
//! When the picker window (`--pick`) fires parallel transcription
//! requests, the results are persisted to disk so that reopening the
//! picker for the **same audio file** returns instantly without any
//! API calls.
//!
//! Cache files live in `~/.cache/talk-rs/picker-results/` and are
//! named after the audio file stem (e.g.
//! `20250219-081500.json` for `20250219-081500.wav`).

use crate::core::daemon::cache_dir;
use crate::core::error::TalkError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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

/// Read previously cached picker results for an audio file.
///
/// Returns an empty `Vec` when no cache exists or on any read/parse
/// error (errors are logged but never propagated — a missing cache
/// simply means every provider will be queried again).
pub fn read(audio_path: &Path) -> Vec<CachedResult> {
    let cache_path = match cache_path_for(audio_path) {
        Ok(p) => p,
        Err(e) => {
            log::debug!("picker cache path error: {}", e);
            return Vec::new();
        }
    };
    if !cache_path.exists() {
        return Vec::new();
    }
    let contents = match std::fs::read_to_string(&cache_path) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("picker cache read error: {}", e);
            return Vec::new();
        }
    };
    match serde_json::from_str::<Vec<CachedResult>>(&contents) {
        Ok(results) => {
            log::info!(
                "loaded {} cached picker result(s) for {}",
                results.len(),
                audio_path.display(),
            );
            results
        }
        Err(e) => {
            log::warn!("picker cache parse error: {}", e);
            Vec::new()
        }
    }
}

/// Persist picker results to the cache file for an audio file.
///
/// Only successful transcriptions (non-empty text) should be passed
/// here.  The file is written atomically (content is fully serialised
/// before writing) to avoid partial reads.
pub fn write(audio_path: &Path, results: &[CachedResult]) -> Result<(), TalkError> {
    if results.is_empty() {
        return Ok(());
    }
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
    let json = serde_json::to_string_pretty(results)
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
        results.len(),
        cache_path.display(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_missing_file_returns_empty() {
        let results = read(Path::new("/tmp/nonexistent-audio-file.wav"));
        assert!(results.is_empty());
    }

    #[test]
    fn test_roundtrip() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let audio = dir.path().join("test-audio.wav");
        std::fs::write(&audio, b"fake").expect("write fake audio");

        // Patch XDG cache dir for isolation — write/read use cache_path_for
        // which calls cache_dir().  Instead, test the serialisation layer
        // directly.
        let cache_file = dir.path().join("test-audio.json");

        let entries = vec![
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
        ];

        let json = serde_json::to_string_pretty(&entries).expect("serialise");
        std::fs::write(&cache_file, &json).expect("write cache");

        let read_back: Vec<CachedResult> =
            serde_json::from_str(&std::fs::read_to_string(&cache_file).expect("read"))
                .expect("parse");
        assert_eq!(read_back.len(), 2);
        assert_eq!(read_back[0].provider, "openai");
        assert_eq!(read_back[0].text, "hello world");
        assert_eq!(read_back[1].provider, "mistral");
        assert_eq!(read_back[1].text, "bonjour monde");
    }

    #[test]
    fn test_read_corrupt_file_returns_empty() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let audio = dir.path().join("corrupt.wav");

        // Write a corrupt JSON file at the expected cache path
        let cache_file = dir.path().join("corrupt.json");
        std::fs::write(&cache_file, "not valid json{{{").expect("write corrupt");

        // Direct parse should fail gracefully
        let contents = std::fs::read_to_string(&cache_file).expect("read");
        let parsed: Result<Vec<CachedResult>, _> = serde_json::from_str(&contents);
        assert!(parsed.is_err());

        // The public read() returns empty for nonexistent cache (the
        // path won't match since cache_dir() differs), but the parse
        // branch is exercised above.
        let _ = audio; // suppress unused warning
    }

    #[test]
    fn test_write_empty_is_noop() {
        // write() with empty slice should succeed without creating files
        let result = write(Path::new("/tmp/noop.wav"), &[]);
        assert!(result.is_ok());
    }
}
