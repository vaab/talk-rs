//! Recording entry listing and file operations for the recordings browser.

use super::audio::ogg_duration_secs;
use crate::config::Config;
use crate::error::TalkError;
use crate::recording_cache;
use std::path::PathBuf;

/// Entry for one cached recording.
pub(super) struct RecordingEntry {
    pub(super) path: PathBuf,
    pub(super) date_label: String,
    pub(super) duration_label: String,
    pub(super) size_label: String,
    /// Full single-line transcript (newlines collapsed to spaces), used
    /// for the copy-to-clipboard action. Empty when no transcript is
    /// available.
    pub(super) transcript_full: String,
    /// Display-ready preview of the transcript: same as
    /// `transcript_full` when short, otherwise truncated to
    /// `TRANSCRIPT_PREVIEW_CHARS` chars with a trailing ellipsis.
    pub(super) transcript_preview: String,
}

/// Maximum number of characters shown in the transcript preview label
/// before truncation + ellipsis.
const TRANSCRIPT_PREVIEW_CHARS: usize = 200;

/// Build the (full, preview) transcript pair used by the recordings
/// browser.
///
/// - `full` is the transcript with newlines collapsed to spaces, used
///   for the clipboard copy action. It is never truncated.
/// - `preview` is the same string truncated to
///   [`TRANSCRIPT_PREVIEW_CHARS`] characters with a trailing ellipsis
///   when longer, used for the GTK display label.
///
/// Both values are empty iff `raw` is empty.
fn transcript_variants(raw: &str) -> (String, String) {
    let full = raw.replace('\n', " ");
    let preview = if full.chars().count() > TRANSCRIPT_PREVIEW_CHARS {
        let truncated: String = full.chars().take(TRANSCRIPT_PREVIEW_CHARS).collect();
        format!("{truncated}…")
    } else {
        full.clone()
    };
    (full, preview)
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
    let t = std::time::Instant::now();
    let config = Config::load(None)?;
    log::debug!("list_ogg: config load {:.0?}", t.elapsed());
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

        // Try to read transcript from companion metadata YAML
        let (transcript_full, transcript_preview) =
            match crate::recording_cache::metadata_path_for_recording(&ogg_path) {
                Ok(Some(meta_path)) => crate::recording_cache::read_metadata_brief(&meta_path)
                    .map(|b| transcript_variants(&b.transcript))
                    .unwrap_or_default(),
                _ => (String::new(), String::new()),
            };

        result.push(RecordingEntry {
            path: ogg_path,
            date_label,
            duration_label,
            size_label,
            transcript_full,
            transcript_preview,
        });
    }

    log::debug!(
        "list_ogg: {} entries, total {:.0?}",
        result.len(),
        t.elapsed(),
    );
    Ok(result)
}

/// Gather dictation cache entries (with companion YML), sorted newest-first.
pub(super) fn list_cache_recordings() -> Result<Vec<RecordingEntry>, TalkError> {
    let t = std::time::Instant::now();
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

    let mut oggs: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| TalkError::Config(format!("failed to read directory entry: {}", e)))?;
        let path = entry.path();

        // Skip symlinks (last_recording.ogg, last_metadata.yml)
        if path.is_symlink() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("ogg") {
            oggs.push(path);
        }
    }

    // Sort lexicographically (timestamp-based names → chronological)
    oggs.sort();
    // Reverse for newest-first
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

        // Try to read transcript from companion metadata YAML
        let (transcript_full, transcript_preview) =
            match recording_cache::metadata_path_for_recording(&ogg_path) {
                Ok(Some(meta_path)) => recording_cache::read_metadata_brief(&meta_path)
                    .map(|b| transcript_variants(&b.transcript))
                    .unwrap_or_default(),
                _ => (String::new(), String::new()),
            };

        result.push(RecordingEntry {
            path: ogg_path,
            date_label,
            duration_label,
            size_label,
            transcript_full,
            transcript_preview,
        });
    }

    log::debug!(
        "list_cache: {} entries, total {:.0?}",
        result.len(),
        t.elapsed(),
    );
    Ok(result)
}

/// Delete a recording and its companion metadata YAML files.
///
/// For cache audio files, also removes matching `*_<model>.yml`
/// companion files.
pub(super) fn delete_recording(file_path: &std::path::Path) -> Result<(), TalkError> {
    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let stem = file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

    // Delete the file itself
    if let Err(e) = std::fs::remove_file(file_path) {
        log::warn!("failed to remove {}: {}", file_path.display(), e);
    }

    // For cache audio files, also delete matching YAML metadata
    // files and the waterfall spectrogram cache (.wf).
    // Keep `wav` for backward compatibility with older cache entries.
    if (ext == "wav" || ext == "ogg") && !stem.is_empty() {
        if let Ok(dir) = recording_cache::recordings_dir() {
            // Remove companion YAML files (<stem>_<model>.yml).
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

            // Remove waterfall spectrogram cache (<stem>.wf).
            let wf_path = dir.join(format!("{}.wf", stem));
            if wf_path.exists() {
                if let Err(e) = std::fs::remove_file(&wf_path) {
                    log::warn!(
                        "failed to remove waterfall cache {}: {}",
                        wf_path.display(),
                        e
                    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_variants_empty_stays_empty() {
        let (full, preview) = transcript_variants("");
        assert_eq!(full, "");
        assert_eq!(preview, "");
    }

    #[test]
    fn transcript_variants_short_text_is_not_truncated() {
        let raw = "Hello, world!";
        let (full, preview) = transcript_variants(raw);
        assert_eq!(full, "Hello, world!");
        assert_eq!(preview, "Hello, world!");
        assert!(!preview.contains('…'));
    }

    #[test]
    fn transcript_variants_collapses_newlines_to_spaces() {
        let raw = "first line\nsecond line\nthird";
        let (full, preview) = transcript_variants(raw);
        assert_eq!(full, "first line second line third");
        assert_eq!(preview, "first line second line third");
    }

    #[test]
    fn transcript_variants_boundary_exactly_preview_chars() {
        // Exactly TRANSCRIPT_PREVIEW_CHARS chars: must NOT be truncated.
        let raw: String = "a".repeat(TRANSCRIPT_PREVIEW_CHARS);
        let (full, preview) = transcript_variants(&raw);
        assert_eq!(full.chars().count(), TRANSCRIPT_PREVIEW_CHARS);
        assert_eq!(preview.chars().count(), TRANSCRIPT_PREVIEW_CHARS);
        assert_eq!(full, raw);
        assert_eq!(preview, raw);
        assert!(!preview.contains('…'));
    }

    #[test]
    fn transcript_variants_long_text_is_truncated_with_ellipsis() {
        // TRANSCRIPT_PREVIEW_CHARS + 1 chars: must be truncated.
        let raw: String = "b".repeat(TRANSCRIPT_PREVIEW_CHARS + 1);
        let (full, preview) = transcript_variants(&raw);

        // Full is untouched.
        assert_eq!(full.chars().count(), TRANSCRIPT_PREVIEW_CHARS + 1);
        assert_eq!(full, raw);

        // Preview is TRANSCRIPT_PREVIEW_CHARS chars + ellipsis.
        assert_eq!(preview.chars().count(), TRANSCRIPT_PREVIEW_CHARS + 1);
        assert!(preview.ends_with('…'));
        let without_ellipsis: String = preview.chars().take(TRANSCRIPT_PREVIEW_CHARS).collect();
        assert_eq!(without_ellipsis, "b".repeat(TRANSCRIPT_PREVIEW_CHARS));
    }

    #[test]
    fn transcript_variants_very_long_text_preserves_full() {
        // Simulate a realistic long transcript.
        let raw: String = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        let (full, preview) = transcript_variants(&raw);

        assert_eq!(full, raw);
        assert!(full.chars().count() > TRANSCRIPT_PREVIEW_CHARS);
        assert!(preview.ends_with('…'));
        // Preview should be exactly TRANSCRIPT_PREVIEW_CHARS chars from `full` plus ellipsis.
        let expected_prefix: String = full.chars().take(TRANSCRIPT_PREVIEW_CHARS).collect();
        assert!(preview.starts_with(&expected_prefix));
    }

    #[test]
    fn transcript_variants_multibyte_chars_counted_correctly() {
        // 201 CJK characters — each is one char but multiple bytes.
        // A byte-based truncation would panic or split a code point;
        // a char-based truncation is safe and produces exactly
        // TRANSCRIPT_PREVIEW_CHARS + 1 chars (with the ellipsis).
        let raw: String = "漢".repeat(TRANSCRIPT_PREVIEW_CHARS + 1);
        let (full, preview) = transcript_variants(&raw);

        assert_eq!(full.chars().count(), TRANSCRIPT_PREVIEW_CHARS + 1);
        assert_eq!(preview.chars().count(), TRANSCRIPT_PREVIEW_CHARS + 1);
        assert!(preview.ends_with('…'));
        let without_ellipsis: String = preview.chars().take(TRANSCRIPT_PREVIEW_CHARS).collect();
        assert_eq!(without_ellipsis, "漢".repeat(TRANSCRIPT_PREVIEW_CHARS));
    }

    #[test]
    fn transcript_variants_long_text_with_newlines() {
        // Long text with embedded newlines: newlines must be collapsed
        // to spaces first, then truncation is applied to the single-line
        // form. This mirrors the real data flow from YAML metadata.
        let long_line = "word ".repeat(60); // 300 chars
        let raw = format!("{long_line}\n{long_line}");
        let (full, preview) = transcript_variants(&raw);

        // No newlines in either output.
        assert!(!full.contains('\n'));
        assert!(!preview.contains('\n'));

        // Full retains both lines joined by a space.
        assert!(full.chars().count() > TRANSCRIPT_PREVIEW_CHARS);

        // Preview is truncated + ellipsis.
        assert!(preview.ends_with('…'));
        assert_eq!(preview.chars().count(), TRANSCRIPT_PREVIEW_CHARS + 1);
    }
}
