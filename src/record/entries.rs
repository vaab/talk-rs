//! Recording entry listing and file operations for the recordings browser.

use super::audio::{m4a_duration_secs, ogg_duration_secs};
use crate::config::Config;
use crate::error::TalkError;
use crate::recording_cache;
use std::path::{Path, PathBuf};

/// File extensions that the recordings browser considers playable
/// audio.  Used both by [`collect_audio_recursive`] and by
/// [`list_cache_recordings`] so listing and cache stay in sync.
///
/// Currently:
/// * `ogg` — talk-rs's own Opus output (and dictation cache).
/// * `m4a` / `mp4` / `aac` — imported audio (e.g. iPhone voice memos),
///   decoded via the `symphonia` AAC/MP4 backend in
///   [`super::audio::read_m4a_as_f32`].
///
/// `wav` is intentionally excluded from the listing: it is only used
/// for legacy cache entries that the cleanup path in
/// [`delete_recording`] still handles, but no current code path writes
/// or browses `.wav` recordings.
const AUDIO_EXTENSIONS: &[&str] = &["ogg", "m4a", "mp4", "aac"];

/// Return `true` when `path` has an extension this UI considers a
/// listable audio recording.  Comparison is case-insensitive to be
/// kind to files imported from external sources (e.g. `.M4A` from a
/// camera).
fn has_audio_extension(path: &Path) -> bool {
    let ext = match path.extension().and_then(|e| e.to_str()) {
        Some(e) => e.to_ascii_lowercase(),
        None => return false,
    };
    AUDIO_EXTENSIONS.iter().any(|&candidate| candidate == ext)
}

/// Compute the duration of an audio file in seconds, dispatching on
/// file extension.  Returns `None` when the format is unknown or the
/// header cannot be parsed.
///
/// This is the listing-time peer of [`super::audio::read_audio_as_i16`]:
/// it picks the right O(1) duration probe per format instead of fully
/// decoding the file, which keeps the recordings browser snappy even
/// with many entries.
fn audio_duration_secs(path: &Path) -> Option<f64> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())?
        .to_ascii_lowercase();
    match ext.as_str() {
        "ogg" | "opus" => ogg_duration_secs(path),
        "m4a" | "mp4" | "aac" => m4a_duration_secs(path),
        _ => None,
    }
}

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
    /// Pick-file status for this recording.  Drives the UI: show
    /// transcript text, "(no text)" placeholder, "transcription
    /// ongoing" indicator, or audio player bar.
    pub(super) status: crate::recording_cache::TranscriptStatus,
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

/// Recursively collect every supported audio file (see
/// [`AUDIO_EXTENSIONS`]) under `dir` into `out`.
///
/// Used by [`list_ogg_recordings`] so the recordings browser still works
/// after the archival directory was namespaced into `YYYY/MM/`
/// subdirectories.  Flat top-level files (pre-migration or user-placed)
/// are also picked up, so the reader tolerates both layouts at once.
///
/// Symlinks are skipped defensively (both directory and file symlinks) to
/// avoid cycles and to match the existing flat reader's behaviour.
/// Errors on a subtree are logged and the walk continues — a single
/// unreadable nested directory must not break the whole listing.
fn collect_audio_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), TalkError> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
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

        // Skip symlinks (both file and directory) to avoid cycles and
        // accidental escape from the recordings root.
        if path.is_symlink() {
            continue;
        }

        if path.is_dir() {
            if let Err(err) = collect_audio_recursive(&path, out) {
                log::warn!("list_audio: skipping subtree {}: {}", path.display(), err);
            }
        } else if has_audio_extension(&path) {
            out.push(path);
        }
    }

    Ok(())
}

/// Collect supported audio files directly inside `dir` without descending.
fn collect_audio_flat(dir: &Path) -> Result<Vec<PathBuf>, TalkError> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
        TalkError::Config(format!(
            "failed to read recordings directory {}: {}",
            dir.display(),
            e
        ))
    })?;

    let mut audio = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| TalkError::Config(format!("failed to read directory entry: {}", e)))?;
        let path = entry.path();
        if !path.is_symlink() && path.is_file() && has_audio_extension(&path) {
            audio.push(path);
        }
    }
    Ok(audio)
}

/// Sort timestamp-bearing recording basenames newest-first.
fn sort_recording_paths_newest_first(audio: &mut [PathBuf]) {
    audio.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RecordingNavigation {
    pub(crate) previous: Option<PathBuf>,
    pub(crate) next: Option<PathBuf>,
}

fn navigation_in_collection(
    current: &Path,
    mut paths: Vec<PathBuf>,
) -> Option<RecordingNavigation> {
    sort_recording_paths_newest_first(&mut paths);
    let current_index = paths.iter().position(|path| {
        std::fs::canonicalize(path)
            .map(|canonical| canonical == current)
            .unwrap_or(false)
    })?;

    Some(RecordingNavigation {
        previous: current_index
            .checked_sub(1)
            .and_then(|index| paths.get(index).cloned()),
        next: paths.get(current_index + 1).cloned(),
    })
}

fn canonical_root(path: &Path) -> Option<PathBuf> {
    std::fs::canonicalize(path)
        .ok()
        .filter(|root| root.is_dir())
}

fn recording_navigation_in_roots(
    current: &Path,
    cache_root: &Path,
    output_root: &Path,
) -> Result<Option<RecordingNavigation>, TalkError> {
    let current = match std::fs::canonicalize(current) {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };

    // Cache wins deliberately if roots overlap. Its collection is flat;
    // output recordings use the browser's recursive walk.
    if let Some(cache_root) = canonical_root(cache_root) {
        if current.starts_with(&cache_root) {
            if let Some(navigation) =
                navigation_in_collection(&current, collect_audio_flat(&cache_root)?)
            {
                return Ok(Some(navigation));
            }
        }
    }

    if let Some(output_root) = canonical_root(output_root) {
        if current.starts_with(&output_root) {
            let mut audio = Vec::new();
            collect_audio_recursive(&output_root, &mut audio)?;
            return Ok(navigation_in_collection(&current, audio));
        }
    }

    Ok(None)
}

/// Resolve the current recording's immediate neighbours in the same
/// collection used by the recordings browser.
pub(crate) fn recording_navigation(
    current: &Path,
    output_root: &Path,
) -> Result<Option<RecordingNavigation>, TalkError> {
    let cache_root = recording_cache::recordings_dir()?;
    recording_navigation_in_roots(current, &cache_root, output_root)
}

/// Gather audio recordings (actual `talk-rs record` output plus any
/// imported audio in the same tree), sorted newest-first.
///
/// Reads `output_dir` from the user configuration file and walks it
/// recursively, so files living in `YYYY/MM/` subdirectories are listed
/// alongside any legacy flat files.  All extensions in
/// [`AUDIO_EXTENSIONS`] are listed — historically that was only `.ogg`
/// (talk-rs's own Opus output), but the browser now also surfaces
/// `.m4a` / `.mp4` / `.aac` files dropped into the directory (e.g.
/// iPhone voice memos imported manually), so they can be played and
/// transcribed alongside native recordings.
pub(super) fn list_ogg_recordings() -> Result<Vec<RecordingEntry>, TalkError> {
    let t = std::time::Instant::now();
    let config = Config::load(None)?;
    log::debug!("list_audio: config load {:.0?}", t.elapsed());
    list_ogg_recordings_in_dir(&config.output_dir, &config)
}

fn list_ogg_recordings_in_dir(
    dir: &Path,
    config: &Config,
) -> Result<Vec<RecordingEntry>, TalkError> {
    let t = std::time::Instant::now();
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut audio: Vec<PathBuf> = Vec::new();
    collect_audio_recursive(dir, &mut audio)?;

    // Sort by file name (the timestamp-bearing basename) rather than by
    // full path.  This keeps chronological ordering correct when flat
    // (`<dir>/2026-04-05T…`) and nested (`<dir>/2026/04/2026-04-10T…`)
    // files coexist during a transition: path-based sorting would place
    // `2026-04-05T…` before `2026/04/2026-04-10T…` because `-` < `/` in
    // ASCII, producing an out-of-order result.  Sorting by file name
    // alone ignores the directory prefix and yields the right order.
    sort_recording_paths_newest_first(&mut audio);

    let mut result = Vec::with_capacity(audio.len());
    for audio_path in audio {
        let stem = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let date_label = date_label_from_stem(stem);

        let duration_label = audio_duration_secs(&audio_path)
            .map(format_duration)
            .unwrap_or_else(|| "?:??".to_string());

        let size_label = std::fs::metadata(&audio_path)
            .map(|m| format_size(m.len()))
            .unwrap_or_else(|_| "?".to_string());

        // Read transcript via the waterfall: pick file first, then
        // default-provider/model sidecar (no API call).  Keeps the
        // pick-lock state visible separately so the UI can still
        // show "transcription ongoing".
        let status = match crate::recording_cache::get_transcript(&audio_path) {
            status @ crate::recording_cache::TranscriptStatus::InProgress => status,
            _ => match crate::transcription::read_cached_transcript(&audio_path, config) {
                Some(text) => crate::recording_cache::TranscriptStatus::Available(text),
                None => crate::recording_cache::TranscriptStatus::NotAvailable,
            },
        };
        let (transcript_full, transcript_preview) = match &status {
            crate::recording_cache::TranscriptStatus::Available(text) => transcript_variants(text),
            _ => (String::new(), String::new()),
        };

        result.push(RecordingEntry {
            path: audio_path,
            date_label,
            duration_label,
            size_label,
            transcript_full,
            transcript_preview,
            status,
        });
    }

    log::debug!(
        "list_audio: {} entries, total {:.0?}",
        result.len(),
        t.elapsed(),
    );
    Ok(result)
}

/// Gather dictation cache entries (with companion YML), sorted newest-first.
pub(super) fn list_cache_recordings() -> Result<Vec<RecordingEntry>, TalkError> {
    let config = Config::load(None)?;
    let dir = recording_cache::recordings_dir()?;
    list_cache_recordings_in_dir(&dir, &config)
}

fn list_cache_recordings_in_dir(
    dir: &Path,
    config: &Config,
) -> Result<Vec<RecordingEntry>, TalkError> {
    let t = std::time::Instant::now();
    if !dir.exists() {
        return Ok(Vec::new());
    }

    // The cache is intentionally flat and symlinks such as
    // `last_recording.ogg` are not separate browser entries.
    let mut audio = collect_audio_flat(dir)?;
    sort_recording_paths_newest_first(&mut audio);

    let mut result = Vec::with_capacity(audio.len());
    for audio_path in audio {
        let stem = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let date_label = date_label_from_stem(stem);

        let duration_label = audio_duration_secs(&audio_path)
            .map(format_duration)
            .unwrap_or_else(|| "?:??".to_string());

        let size_label = std::fs::metadata(&audio_path)
            .map(|m| format_size(m.len()))
            .unwrap_or_else(|_| "?".to_string());

        // Read transcript via the waterfall: pick → default-model
        // sidecar (no API call).
        let status = match recording_cache::get_transcript(&audio_path) {
            status @ recording_cache::TranscriptStatus::InProgress => status,
            _ => match crate::transcription::read_cached_transcript(&audio_path, config) {
                Some(text) => recording_cache::TranscriptStatus::Available(text),
                None => recording_cache::TranscriptStatus::NotAvailable,
            },
        };
        let (transcript_full, transcript_preview) = match &status {
            recording_cache::TranscriptStatus::Available(text) => transcript_variants(text),
            _ => (String::new(), String::new()),
        };

        result.push(RecordingEntry {
            path: audio_path,
            date_label,
            duration_label,
            size_label,
            transcript_full,
            transcript_preview,
            status,
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
    delete_recording_in_dir(file_path, recording_cache::recordings_dir().ok().as_deref())
}

fn delete_recording_in_dir(file_path: &Path, dir: Option<&Path>) -> Result<(), TalkError> {
    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let stem = file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

    // Delete the file itself
    if let Err(e) = std::fs::remove_file(file_path) {
        log::warn!("failed to remove {}: {}", file_path.display(), e);
    }

    // For cache audio files, also delete matching YAML metadata
    // files and the waterfall spectrogram cache (.wf).
    // Keep `wav` for backward compatibility with older cache entries;
    // include every extension recognised by [`AUDIO_EXTENSIONS`] so
    // m4a / mp4 / aac imports get the same cleanup treatment as
    // native `.ogg` recordings.
    let is_known_audio = ext == "wav" || AUDIO_EXTENSIONS.contains(&ext);
    if is_known_audio && !stem.is_empty() {
        if let Some(dir) = dir {
            // Remove companion YAML files (<stem>_*.yml and <stem>.pick.yml).
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if name.starts_with(stem)
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
    use crate::config::{MistralConfig, ProvidersConfig};
    use crate::recording_cache::TranscriptionCache;
    use crate::transcription::TranscriptionResult;

    fn listing_config(root: &Path) -> Config {
        Config {
            output_dir: root.to_path_buf(),
            providers: ProvidersConfig {
                mistral: Some(MistralConfig {
                    api_key: "test-key".into(),
                    url: None,
                    model: "voxtral-mini-2507".into(),
                    context_bias: None,
                    tts_model: "voxtral-mini-tts-latest".into(),
                    tts_voice: None,
                    tts_voices: None,
                }),
                openai: None,
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        }
    }

    fn cached_text(path: &Path, text: &str) {
        TranscriptionCache::store(
            path,
            crate::config::Provider::Mistral,
            "voxtral-mini-2507",
            false,
            &TranscriptionResult {
                text: text.into(),
                ..TranscriptionResult::default()
            },
        )
        .expect("write sidecar");
    }

    #[test]
    fn assembled_output_entries_mix_flat_imports_and_nested_recordings_newest_first() {
        let temp = tempfile::tempdir().expect("tempdir");
        let old = temp.path().join("2026-04-01T10-00-00+0200.m4a");
        let nested = temp.path().join("2026/04");
        std::fs::create_dir_all(&nested).expect("nested dir");
        let new = nested.join("2026-04-02T10-00-00+0200.ogg");
        std::fs::write(&old, b"import").expect("import");
        std::fs::write(&new, b"recording").expect("recording");
        cached_text(&old, "sidecar text");
        recording_cache::write_pick(&old, "mistral", "voxtral-mini-2507", false, "edited\ntext")
            .expect("pick");

        let rows = list_ogg_recordings_in_dir(temp.path(), &listing_config(temp.path()))
            .expect("assembled rows");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].path, new);
        assert_eq!(rows[0].date_label, "2026-04-02 10:00:00");
        assert_eq!(
            rows[0].status,
            recording_cache::TranscriptStatus::NotAvailable
        );
        assert_eq!(rows[0].size_label, "9 B");
        assert_eq!(rows[1].path, old);
        assert_eq!(
            rows[1].status,
            recording_cache::TranscriptStatus::Available("edited\ntext".into())
        );
        assert_eq!(rows[1].transcript_full, "edited text");
        assert_eq!(rows[1].transcript_preview, "edited text");
    }

    #[test]
    fn assembled_cache_entry_prefers_active_pick_lock_over_sidecar_and_pick() {
        let temp = tempfile::tempdir().expect("tempdir");
        let audio = temp.path().join("2026-04-02T10-00-00+0200.ogg");
        std::fs::write(&audio, b"audio").expect("audio");
        cached_text(&audio, "sidecar text");
        recording_cache::write_pick(&audio, "mistral", "voxtral-mini-2507", false, "old pick")
            .expect("pick");
        recording_cache::acquire_pick_lock(&audio).expect("pick lock");

        let rows = list_cache_recordings_in_dir(temp.path(), &listing_config(temp.path()))
            .expect("assembled rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].status,
            recording_cache::TranscriptStatus::InProgress
        );
        assert_eq!(
            (&*rows[0].transcript_full, &*rows[0].transcript_preview),
            ("", "")
        );
    }

    #[test]
    fn assembled_cache_entry_falls_back_to_default_sidecar() {
        let temp = tempfile::tempdir().expect("tempdir");
        let audio = temp.path().join("2026-04-02T10-00-00+0200.ogg");
        std::fs::write(&audio, b"audio").expect("audio");
        cached_text(&audio, "sidecar text");

        let rows = list_cache_recordings_in_dir(temp.path(), &listing_config(temp.path()))
            .expect("assembled rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].status,
            recording_cache::TranscriptStatus::Available("sidecar text".into())
        );
        assert_eq!(rows[0].transcript_full, "sidecar text");
    }

    #[test]
    fn deletion_removes_audio_sidecars_and_waterfall() {
        let temp = tempfile::tempdir().expect("tempdir");
        let audio = temp.path().join("memo.ogg");
        let pick = temp.path().join("memo.pick.yml");
        let sidecar = temp.path().join("memo_mistral_model_oneshot.yml");
        let waterfall = temp.path().join("memo.wf");
        for path in [&audio, &pick, &sidecar, &waterfall] {
            std::fs::write(path, b"fixture").expect("fixture");
        }
        delete_recording_in_dir(&audio, Some(temp.path())).expect("delete");
        for path in [&audio, &pick, &sidecar, &waterfall] {
            assert!(!path.exists(), "{} should be removed", path.display());
        }
    }

    #[test]
    #[ignore = "BUG: deletion matches YAML sidecars by raw stem prefix and removes another recording's metadata"]
    fn deletion_preserves_similarly_prefixed_recording() {
        let temp = tempfile::tempdir().expect("tempdir");
        let audio = temp.path().join("memo.ogg");
        let other = temp.path().join("memo-extra.ogg");
        let other_pick = temp.path().join("memo-extra.pick.yml");
        for path in [&audio, &other, &other_pick] {
            std::fs::write(path, b"fixture").expect("fixture");
        }
        delete_recording_in_dir(&audio, Some(temp.path())).expect("delete");
        assert!(other.exists());
        assert!(other_pick.exists(), "another recording's pick must survive");
    }

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

    // ---- collect_audio_recursive ----

    use tempfile::TempDir;

    /// Basename helper for assertions.
    fn basename(p: &Path) -> &str {
        p.file_name().and_then(|n| n.to_str()).unwrap_or("")
    }

    #[test]
    fn collect_audio_flat_layout() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path();
        std::fs::write(dir.join("a.ogg"), b"").unwrap();
        std::fs::write(dir.join("b.ogg"), b"").unwrap();
        std::fs::write(dir.join("ignored.txt"), b"").unwrap();
        std::fs::write(dir.join("also-ignored.wf"), b"").unwrap();

        let mut out = Vec::new();
        collect_audio_recursive(dir, &mut out).expect("collect");

        let mut names: Vec<_> = out.iter().map(|p| basename(p).to_string()).collect();
        names.sort();
        assert_eq!(names, vec!["a.ogg", "b.ogg"]);
    }

    #[test]
    fn collect_audio_nested_layout() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path();

        // Build `<dir>/2026/04/` and `<dir>/2025/12/`.
        let nested_2026_04 = dir.join("2026").join("04");
        let nested_2025_12 = dir.join("2025").join("12");
        std::fs::create_dir_all(&nested_2026_04).unwrap();
        std::fs::create_dir_all(&nested_2025_12).unwrap();

        std::fs::write(nested_2026_04.join("2026-04-10T08-23-15+0200.ogg"), b"").unwrap();
        std::fs::write(nested_2025_12.join("2025-12-27T04-31-23+0100.ogg"), b"").unwrap();

        // Sidecars that must NOT be picked up.
        std::fs::write(
            nested_2026_04.join("2026-04-10T08-23-15+0200-voxtral.json"),
            b"",
        )
        .unwrap();
        std::fs::write(nested_2026_04.join("2026-04-10T08-23-15+0200.txt"), b"").unwrap();

        let mut out = Vec::new();
        collect_audio_recursive(dir, &mut out).expect("collect");

        assert_eq!(out.len(), 2, "should find exactly 2 nested .ogg files");
        let mut names: Vec<_> = out.iter().map(|p| basename(p).to_string()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "2025-12-27T04-31-23+0100.ogg".to_string(),
                "2026-04-10T08-23-15+0200.ogg".to_string(),
            ]
        );
    }

    #[test]
    fn collect_audio_mixed_flat_and_nested() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path();

        // Legacy flat file (pre-migration or user-placed).
        std::fs::write(dir.join("2026-04-05T10-00-00+0200.ogg"), b"").unwrap();

        // Post-migration nested file.
        let nested = dir.join("2026").join("04");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("2026-04-10T10-00-00+0200.ogg"), b"").unwrap();

        let mut out = Vec::new();
        collect_audio_recursive(dir, &mut out).expect("collect");

        assert_eq!(out.len(), 2, "should find both flat and nested files");

        // Sort by file name (same rule list_ogg_recordings uses) and
        // verify chronological order.  Path-based sorting would produce
        // the wrong order here because `-` < `/` in ASCII.
        out.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
        let names: Vec<_> = out.iter().map(|p| basename(p).to_string()).collect();
        assert_eq!(
            names,
            vec![
                "2026-04-05T10-00-00+0200.ogg".to_string(),
                "2026-04-10T10-00-00+0200.ogg".to_string(),
            ],
            "file-name-based sort must place the Apr-5 flat file before the Apr-10 nested file"
        );
    }

    #[test]
    fn collect_audio_skips_symlinks() {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path();

        // Real file.
        std::fs::write(dir.join("real.ogg"), b"").unwrap();

        // File symlink pointing at the real file — must be skipped.
        std::os::unix::fs::symlink(dir.join("real.ogg"), dir.join("link.ogg")).unwrap();

        // Directory symlink pointing at `.` — would cause infinite
        // recursion if followed.  Must be skipped.
        std::os::unix::fs::symlink(dir, dir.join("self-link")).unwrap();

        let mut out = Vec::new();
        collect_audio_recursive(dir, &mut out).expect("collect");

        assert_eq!(out.len(), 1, "symlinks must be skipped");
        assert_eq!(basename(&out[0]), "real.ogg");
    }

    #[test]
    fn collect_audio_empty_directory() {
        let tmp = TempDir::new().expect("tempdir");
        let mut out = Vec::new();
        collect_audio_recursive(tmp.path(), &mut out).expect("collect");
        assert!(out.is_empty());
    }

    #[test]
    fn collect_audio_deep_nesting() {
        // Sanity check: the walker handles more than two levels (e.g.
        // `year/month/day/` if the layout ever gets deeper).
        let tmp = TempDir::new().expect("tempdir");
        let deep = tmp.path().join("2026").join("04").join("10");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(deep.join("memo.ogg"), b"").unwrap();

        let mut out = Vec::new();
        collect_audio_recursive(tmp.path(), &mut out).expect("collect");

        assert_eq!(out.len(), 1);
        assert_eq!(basename(&out[0]), "memo.ogg");
    }

    #[test]
    fn collect_audio_picks_up_m4a_files() {
        // The recordings browser must list `.m4a` files (e.g. iPhone
        // voice memos dropped into the recordings directory), not
        // just talk-rs's own `.ogg` output.
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path();
        std::fs::write(dir.join("memo.m4a"), b"").unwrap();
        std::fs::write(dir.join("clip.MP4"), b"").unwrap(); // case-insensitive
        std::fs::write(dir.join("stream.aac"), b"").unwrap();
        std::fs::write(dir.join("ignored.txt"), b"").unwrap();
        std::fs::write(dir.join("also-ignored.wav"), b"").unwrap(); // wav is NOT listable

        let mut out = Vec::new();
        collect_audio_recursive(dir, &mut out).expect("collect");

        let mut names: Vec<_> = out.iter().map(|p| basename(p).to_string()).collect();
        names.sort();
        assert_eq!(names, vec!["clip.MP4", "memo.m4a", "stream.aac"]);
    }

    #[test]
    fn collect_audio_mixed_ogg_and_m4a() {
        // Real-world layout: a recordings directory containing both
        // native `.ogg` recordings and imported `.m4a` files.  Both
        // must be listed alongside one another.
        let tmp = TempDir::new().expect("tempdir");
        let nested = tmp.path().join("2026").join("04");
        std::fs::create_dir_all(&nested).unwrap();

        std::fs::write(nested.join("2026-04-10T08-23-15+0200.ogg"), b"").unwrap();
        std::fs::write(nested.join("imported-memo.m4a"), b"").unwrap();

        let mut out = Vec::new();
        collect_audio_recursive(tmp.path(), &mut out).expect("collect");

        let mut names: Vec<_> = out.iter().map(|p| basename(p).to_string()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "2026-04-10T08-23-15+0200.ogg".to_string(),
                "imported-memo.m4a".to_string(),
            ]
        );
    }

    #[test]
    fn has_audio_extension_is_case_insensitive() {
        assert!(has_audio_extension(Path::new("foo.M4A")));
        assert!(has_audio_extension(Path::new("foo.m4a")));
        assert!(has_audio_extension(Path::new("foo.OGG")));
        assert!(has_audio_extension(Path::new("foo.mp4")));
        assert!(has_audio_extension(Path::new("foo.aac")));
        assert!(!has_audio_extension(Path::new("foo.wav"))); // not listable
        assert!(!has_audio_extension(Path::new("foo.txt")));
        assert!(!has_audio_extension(Path::new("foo")));
    }

    #[test]
    fn audio_duration_secs_dispatches_correctly() {
        // Unknown extension → None (not an error).
        let tmp = TempDir::new().expect("tempdir");
        let unknown = tmp.path().join("clip.flac");
        std::fs::write(&unknown, b"x").unwrap();
        assert_eq!(audio_duration_secs(&unknown), None);

        // No extension → None.
        let bare = tmp.path().join("noext");
        std::fs::write(&bare, b"x").unwrap();
        assert_eq!(audio_duration_secs(&bare), None);
    }

    #[test]
    fn recording_navigation_uses_flat_cache_newest_first() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = tmp.path().join("cache");
        let output = tmp.path().join("output");
        std::fs::create_dir_all(cache.join("ignored-nested")).expect("create cache");
        std::fs::create_dir_all(&output).expect("create output");

        let newest = cache.join("2026-04-03T10-00-00+0200.ogg");
        let current = cache.join("2026-04-02T10-00-00+0200.ogg");
        let oldest = cache.join("2026-04-01T10-00-00+0200.ogg");
        for path in [&newest, &current, &oldest] {
            std::fs::write(path, b"").expect("write audio");
        }
        std::fs::write(
            cache.join("ignored-nested/2026-04-04T10-00-00+0200.ogg"),
            b"",
        )
        .expect("write nested cache audio");

        let navigation = recording_navigation_in_roots(&current, &cache, &output)
            .expect("resolve navigation")
            .expect("cache collection");

        assert_eq!(navigation.previous.as_deref(), Some(newest.as_path()));
        assert_eq!(navigation.next.as_deref(), Some(oldest.as_path()));
    }

    #[test]
    fn recording_navigation_uses_recursive_output_and_disables_boundaries() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = tmp.path().join("cache");
        let output = tmp.path().join("output");
        std::fs::create_dir_all(&cache).expect("create cache");
        std::fs::create_dir_all(output.join("2026/04")).expect("create output");

        let newest = output.join("2026/04/2026-04-03T10-00-00+0200.ogg");
        let oldest = output.join("2026-04-01T10-00-00+0200.ogg");
        std::fs::write(&newest, b"").expect("write newest");
        std::fs::write(&oldest, b"").expect("write oldest");

        let newest_navigation = recording_navigation_in_roots(&newest, &cache, &output)
            .expect("resolve newest")
            .expect("output collection");
        assert_eq!(newest_navigation.previous, None);
        assert_eq!(newest_navigation.next.as_deref(), Some(oldest.as_path()));

        let oldest_navigation = recording_navigation_in_roots(&oldest, &cache, &output)
            .expect("resolve oldest")
            .expect("output collection");
        assert_eq!(
            oldest_navigation.previous.as_deref(),
            Some(newest.as_path())
        );
        assert_eq!(oldest_navigation.next, None);
    }

    #[test]
    fn recording_navigation_falls_through_to_output_nested_under_cache() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = tmp.path().join("cache");
        let output = cache.join("output").join("2026").join("04");
        std::fs::create_dir_all(&output).expect("create nested output");

        let newest = output.join("2026-04-03T10-00-00+0200.ogg");
        let oldest = output.join("2026-04-01T10-00-00+0200.ogg");
        std::fs::write(&newest, b"").expect("write newest");
        std::fs::write(&oldest, b"").expect("write oldest");

        let newest_navigation = recording_navigation_in_roots(&newest, &cache, &output)
            .expect("resolve newest")
            .expect("nested output collection");
        assert_eq!(newest_navigation.previous, None);
        assert_eq!(newest_navigation.next.as_deref(), Some(oldest.as_path()));

        let oldest_navigation = recording_navigation_in_roots(&oldest, &cache, &output)
            .expect("resolve oldest")
            .expect("nested output collection");
        assert_eq!(
            oldest_navigation.previous.as_deref(),
            Some(newest.as_path())
        );
        assert_eq!(oldest_navigation.next, None);
    }

    #[test]
    fn recording_navigation_canonicalizes_current_and_rejects_foreign_files() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = tmp.path().join("cache");
        let output = tmp.path().join("output");
        let foreign_dir = tmp.path().join("foreign");
        std::fs::create_dir_all(&cache).expect("create cache");
        std::fs::create_dir_all(&output).expect("create output");
        std::fs::create_dir_all(&foreign_dir).expect("create foreign");

        let current = cache.join("2026-04-02T10-00-00+0200.ogg");
        let older = cache.join("2026-04-01T10-00-00+0200.ogg");
        std::fs::write(&current, b"").expect("write current");
        std::fs::write(&older, b"").expect("write older");
        let retry_last = cache.join("last_recording.ogg");
        std::os::unix::fs::symlink(&current, &retry_last).expect("create retry-last symlink");

        let navigation = recording_navigation_in_roots(&retry_last, &cache, &output)
            .expect("resolve symlink")
            .expect("cache collection");
        assert_eq!(navigation.previous, None);
        assert_eq!(navigation.next.as_deref(), Some(older.as_path()));

        let foreign = foreign_dir.join("2026-04-04T10-00-00+0200.ogg");
        std::fs::write(&foreign, b"").expect("write foreign");
        assert_eq!(
            recording_navigation_in_roots(&foreign, &cache, &output).expect("resolve foreign"),
            None
        );
    }
}
