//! Recording entry listing and file operations for the recordings browser.

use super::audio::{ogg_duration_secs, wav_duration_secs};
use crate::core::config::Config;
use crate::core::error::TalkError;
use crate::core::recording_cache;
use std::path::PathBuf;

/// Entry for one cached recording (WAV or OGG).
pub(super) struct RecordingEntry {
    pub(super) path: PathBuf,
    pub(super) date_label: String,
    pub(super) duration_label: String,
    pub(super) size_label: String,
    pub(super) transcript_preview: String,
}

/// Parse a date label from a timestamp-based filename stem.
///
/// Expected format: `2026-02-18T12-33-45` → `"2026-02-18 12:33:45"`.
fn date_label_from_stem(stem: &str) -> String {
    if stem.len() >= 19 {
        let date_part = &stem[..10];
        let time_part = stem[11..19].replace('-', ":");
        format!("{} {}", date_part, time_part)
    } else {
        stem.to_string()
    }
}

/// Format seconds into `M:SS` or `H:MM:SS`.
pub(super) fn format_duration(secs: f64) -> String {
    let total = secs.round() as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{}:{:02}:{:02}", h, m, s)
    } else {
        format!("{}:{:02}", m, s)
    }
}

/// Format byte count into human-readable size.
pub(super) fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{} KB", bytes / 1_000)
    } else {
        format!("{} B", bytes)
    }
}

/// Gather OGG recordings (actual `talk-rs record` output), sorted newest-first.
///
/// Reads `output_dir` from the user configuration file.
pub(super) fn list_ogg_recordings() -> Result<Vec<RecordingEntry>, TalkError> {
    let config = Config::load(None)?;
    let dir = config.output_dir;
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&dir).map_err(|e| {
        TalkError::Config(format!(
            "failed to read recordings directory {}: {}",
            dir.display(),
            e
        ))
    })?;

    let mut oggs: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| TalkError::Config(format!("failed to read directory entry: {}", e)))?;
        let path = entry.path();

        // Skip symlinks
        if path.is_symlink() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("ogg") {
            oggs.push(path);
        }
    }

    oggs.sort();
    oggs.reverse();

    let mut result = Vec::with_capacity(oggs.len());
    for ogg_path in oggs {
        let stem = ogg_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let date_label = date_label_from_stem(stem);

        let duration_label = ogg_duration_secs(&ogg_path)
            .map(format_duration)
            .unwrap_or_else(|| "?:??".to_string());

        let size_label = std::fs::metadata(&ogg_path)
            .map(|m| format_size(m.len()))
            .unwrap_or_else(|_| "?".to_string());

        result.push(RecordingEntry {
            path: ogg_path,
            date_label,
            duration_label,
            size_label,
            transcript_preview: String::new(),
        });
    }

    Ok(result)
}

/// Gather WAV dictation cache entries (with companion YML), sorted newest-first.
pub(super) fn list_wav_recordings() -> Result<Vec<RecordingEntry>, TalkError> {
    let dir = recording_cache::recordings_dir()?;
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&dir).map_err(|e| {
        TalkError::Config(format!(
            "failed to read recordings directory {}: {}",
            dir.display(),
            e
        ))
    })?;

    let mut wavs: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| TalkError::Config(format!("failed to read directory entry: {}", e)))?;
        let path = entry.path();

        // Skip symlinks (last_recording.wav, last_metadata.yml)
        if path.is_symlink() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("wav") {
            wavs.push(path);
        }
    }

    // Sort lexicographically (timestamp-based names → chronological)
    wavs.sort();
    // Reverse for newest-first
    wavs.reverse();

    let mut result = Vec::with_capacity(wavs.len());
    for wav_path in wavs {
        let stem = wav_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let date_label = date_label_from_stem(stem);

        let duration_label = wav_duration_secs(&wav_path)
            .map(format_duration)
            .unwrap_or_else(|| "?:??".to_string());

        let size_label = std::fs::metadata(&wav_path)
            .map(|m| format_size(m.len()))
            .unwrap_or_else(|_| "?".to_string());

        // Try to read transcript from companion metadata YAML
        let transcript_preview = match recording_cache::metadata_path_for_recording(&wav_path) {
            Ok(Some(meta_path)) => recording_cache::read_metadata_brief(&meta_path)
                .map(|b| {
                    // Single line, truncated preview (char-safe)
                    let line = b.transcript.replace('\n', " ");
                    if line.chars().count() > 200 {
                        let truncated: String = line.chars().take(200).collect();
                        format!("{truncated}…")
                    } else {
                        line
                    }
                })
                .unwrap_or_default(),
            _ => String::new(),
        };

        result.push(RecordingEntry {
            path: wav_path,
            date_label,
            duration_label,
            size_label,
            transcript_preview,
        });
    }

    Ok(result)
}

/// Delete a recording and its companion metadata YAML files.
///
/// For WAV files, also removes matching `*_<model>.yml` companion files.
/// For OGG files, only the single file is removed (no companions).
pub(super) fn delete_recording(file_path: &std::path::Path) -> Result<(), TalkError> {
    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let stem = file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

    // Delete the file itself
    if let Err(e) = std::fs::remove_file(file_path) {
        log::warn!("failed to remove {}: {}", file_path.display(), e);
    }

    // For WAV files, also delete matching YAML metadata files
    if ext == "wav" && !stem.is_empty() {
        if let Ok(dir) = recording_cache::recordings_dir() {
            let yml_prefix = format!("{}_", stem);
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if name.starts_with(&yml_prefix)
                        && path.extension().and_then(|e| e.to_str()) == Some("yml")
                    {
                        if let Err(e) = std::fs::remove_file(&path) {
                            log::warn!("failed to remove metadata {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Open the system file manager with `file_path` highlighted.
///
/// Uses GTK's [`FileLauncher`](gtk4::FileLauncher) which passes the
/// activation token so the file manager window is raised to the front.
pub(super) fn open_in_file_manager(file_path: &std::path::Path, parent_window: &gtk4::Window) {
    let gio_file = gtk4::gio::File::for_path(file_path);
    let launcher = gtk4::FileLauncher::new(Some(&gio_file));
    launcher.open_containing_folder(
        Some(parent_window),
        gtk4::gio::Cancellable::NONE,
        |result| {
            if let Err(e) = result {
                log::warn!("failed to open file manager: {}", e);
            }
        },
    );
}
