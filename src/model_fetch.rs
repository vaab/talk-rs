//! Shared model-fetch machinery for the local on-device backends.
//!
//! Both the local ASR backend (`parakeet`) and the local TTS backend
//! (`kokoro`) need the exact same lifecycle for their multi-hundred-
//! megabyte model artifacts:
//!
//! 1. **Presence check** — are all required files on disk and non-empty?
//! 2. **Consent** — never download silently; prompt on a TTY, or log +
//!    proceed when non-interactive (selecting a local backend *is* the
//!    consent for unattended callers).
//! 3. **Download** — stream a `.tar.bz2` with progress logging.
//! 4. **Extract + verify + atomic promote** — untar into a staging dir
//!    next to the target, verify the required files, then a single
//!    atomic rename so the target is either complete or absent.
//! 5. **Concurrency** — two processes racing on the same target dir
//!    serialise on a sibling `<dir>.lock` flock.
//!
//! This module is the extraction of what used to live only in
//! `transcription::parakeet::model`.  Parakeet now delegates its
//! download/extract to [`download_and_install`] and its consent flow to
//! [`ensure_with_cli_consent`], so the Kokoro backend reuses the exact
//! same, already-battle-tested code.  The behaviour observed by the
//! parakeet tests is preserved — the generic helpers here take the
//! parakeet-specific parameters (URL, inner-dir name, required files,
//! size label) via [`ModelSpec`].
//!
//! Gated on `any(feature = "parakeet", feature = "kokoro")` because it
//! needs `tar` / `bzip2` which both features pull.

use crate::error::TalkError;
use fs2::FileExt;
use futures::StreamExt;
use std::fs::OpenOptions;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

/// Progress-log granularity for the streaming download.
const PROGRESS_LOG_EVERY_BYTES: u64 = 32 * 1024 * 1024; // 32 MiB

/// Static description of a downloadable model artifact.
///
/// Everything the generic fetch machinery needs that differs between
/// backends.  A backend builds one of these and hands it to
/// [`is_present`] / [`download_and_install`] / [`ensure_with_cli_consent`].
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Human-facing backend name for log/error copy (e.g. `"Parakeet"`,
    /// `"Kokoro"`).
    pub display_name: &'static str,
    /// Tarball asset URL (`.tar.bz2`).
    pub tarball_url: &'static str,
    /// Top-level directory the archive nests every entry under.  The
    /// extractor prefers this exact name but falls back to "the single
    /// non-hidden subdir" for re-packaged variants.
    pub inner_dir: &'static str,
    /// Required files (relative to the model dir) that must all exist
    /// and be non-empty for the model to count as present.
    pub required_files: &'static [&'static str],
    /// Approximate download size for the consent prompt (e.g.
    /// `"~640 MB"`).
    pub approx_size: &'static str,
    /// Short manual-install hint listing the essential files, appended
    /// to failure/consent messages (e.g.
    /// `"encoder.int8.onnx/decoder.int8.onnx/joiner.int8.onnx/tokens.txt"`).
    pub manual_files_hint: &'static str,
}

impl ModelSpec {
    /// Format the manual-fallback error message.  Surfaced verbatim by
    /// any download/extract/verify failure so a locked-down user can
    /// install the model by hand.
    fn manual_fallback_message(&self, cause: &str, dir: &Path) -> String {
        format!(
            "Failed to download {} model ({}). To install manually: \
             download {}, extract it, and place {} into {}.",
            self.display_name,
            cause,
            self.tarball_url,
            self.manual_files_hint,
            dir.display()
        )
    }
}

/// True iff every required file for `spec` exists under `dir` AND is
/// non-empty.  Non-empty matters because an interrupted download could
/// leave a zero-byte placeholder; we must not treat that as "present".
pub fn is_present(dir: &Path, spec: &ModelSpec) -> bool {
    spec.required_files.iter().all(|name| {
        let p = dir.join(name);
        match std::fs::metadata(&p) {
            Ok(m) => m.is_file() && m.len() > 0,
            Err(_) => false,
        }
    })
}

/// Check that the model is present under `dir`, WITHOUT downloading.
///
/// This is the consent-safe presence gate: it performs no network I/O
/// and never fetches anything.  Callers that want the model downloaded
/// must obtain explicit user consent and then call
/// [`download_and_install`].
///
/// # Errors
///
/// Returns a [`TalkError::Config`] whose message names the model
/// directory and the exact URL for a manual install.
pub fn ensure_present(dir: &Path, spec: &ModelSpec) -> Result<(), TalkError> {
    if is_present(dir, spec) {
        return Ok(());
    }
    Err(TalkError::Config(format!(
        "{} model not found in {}. The model is not downloaded \
         automatically; it must be fetched with explicit consent. \
         To install manually: download {}, extract it, and place \
         {} into {}.",
        spec.display_name,
        dir.display(),
        spec.tarball_url,
        spec.manual_files_hint,
        dir.display()
    )))
}

/// RAII unlock wrapper around an `fs2::FileExt::lock_exclusive`.
struct LockGuard<'a>(&'a std::fs::File);

impl Drop for LockGuard<'_> {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(self.0);
    }
}

/// Download and install the model under `dir`.
///
/// **This always performs the network download** (after a re-check
/// under the lock).  Callers MUST have obtained user consent before
/// invoking it — there is no silent-download path (see
/// [`ensure_present`]).
///
/// Atomicity + concurrency contract identical to the original
/// parakeet `download_model`: on any failure `dir` is left
/// present-and-complete or absent, never half-populated; concurrent
/// callers serialise on `<dir>.lock`.
pub async fn download_and_install(dir: &Path, spec: &ModelSpec) -> Result<(), TalkError> {
    // 1. Fast path: already complete (idempotent).
    if is_present(dir, spec) {
        return Ok(());
    }

    // 2. Make sure the parent exists, then take the sibling flock.
    let parent = dir.parent().ok_or_else(|| {
        TalkError::Config(format!(
            "{} model_dir has no parent directory: {}",
            spec.display_name,
            dir.display()
        ))
    })?;
    std::fs::create_dir_all(parent).map_err(|e| {
        TalkError::Config(format!(
            "Failed to create {} model parent dir {}: {}",
            spec.display_name,
            parent.display(),
            e
        ))
    })?;

    let lock_path = sibling_lock_path(dir);
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| {
            TalkError::Config(format!(
                "Failed to open {} model lock file {}: {}",
                spec.display_name,
                lock_path.display(),
                e
            ))
        })?;
    lock_file.lock_exclusive().map_err(|e| {
        TalkError::Config(format!(
            "Failed to acquire exclusive flock on {}: {}",
            lock_path.display(),
            e
        ))
    })?;
    let _lock_guard = LockGuard(&lock_file);

    // 3. Re-check under the lock — another process may have completed
    //    while we waited.
    if is_present(dir, spec) {
        return Ok(());
    }

    // 4. Stream the tarball to a temp file in the same parent dir so
    //    the eventual rename stays on a single filesystem.
    let tarball_tmp = parent.join(format!(
        "{}.download.tmp",
        dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("talk-rs-model")
    ));
    // Best-effort wipe any leftover from a prior aborted run.
    let _ = std::fs::remove_file(&tarball_tmp);

    if let Err(e) = download_to(spec.tarball_url, &tarball_tmp).await {
        let _ = std::fs::remove_file(&tarball_tmp);
        return Err(TalkError::Config(
            spec.manual_fallback_message(&e.to_string(), dir),
        ));
    }

    // 5. Extract + verify + atomic-promote (pure, testable).
    let res = install_from_tarball(&tarball_tmp, dir, spec);
    // Tarball is no longer needed regardless of outcome.
    let _ = std::fs::remove_file(&tarball_tmp);

    if let Err(e) = res {
        return Err(TalkError::Config(
            spec.manual_fallback_message(&e.to_string(), dir),
        ));
    }

    Ok(())
}

/// Path of the sibling lock file used to serialise concurrent
/// download callers.  Lives next to `dir`, not inside it, so it
/// survives a `remove_dir_all(dir)` cleanup.
fn sibling_lock_path(dir: &Path) -> PathBuf {
    let mut p = dir.to_path_buf();
    let name = p
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("talk-rs-model");
    let lock_name = format!("{}.lock", name);
    p.set_file_name(lock_name);
    p
}

/// Streaming download of `url` into `dest` via async reqwest.
async fn download_to(url: &str, dest: &Path) -> Result<(), TalkError> {
    log::info!("model fetch: downloading {} -> {}", url, dest.display());

    let response = reqwest::get(url)
        .await
        .map_err(|e| TalkError::Transcription(format!("HTTP GET failed: {}", e)))?;

    let status = response.status();
    if !status.is_success() {
        return Err(TalkError::Transcription(format!(
            "HTTP {} from {}",
            status, url
        )));
    }

    let total_hint = response.content_length();
    let mut out = std::fs::File::create(dest).map_err(|e| {
        TalkError::Transcription(format!("create temp tarball {}: {}", dest.display(), e))
    })?;

    let mut stream = response.bytes_stream();
    let mut written: u64 = 0;
    let mut next_log_at: u64 = PROGRESS_LOG_EVERY_BYTES;

    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|e| TalkError::Transcription(format!("HTTP stream error: {}", e)))?;
        out.write_all(&chunk).map_err(|e| {
            TalkError::Transcription(format!("write to temp tarball {}: {}", dest.display(), e))
        })?;
        written += chunk.len() as u64;
        if written >= next_log_at {
            match total_hint {
                Some(total) => log::info!("model fetch: downloaded {} / {} bytes", written, total),
                None => log::info!("model fetch: downloaded {} bytes", written),
            }
            next_log_at = next_log_at.saturating_add(PROGRESS_LOG_EVERY_BYTES);
        }
    }

    out.flush().map_err(|e| {
        TalkError::Transcription(format!("flush temp tarball {}: {}", dest.display(), e))
    })?;
    drop(out);

    log::info!("model fetch: download complete ({} bytes)", written);
    Ok(())
}

/// Extract `tarball` (a `.tar.bz2`), verify the required files, and
/// atomically promote them so `dir` directly contains the files (no
/// nested subdir).
///
/// **Pure** (no network, no globals).  This is the testable seam.
///
/// Atomicity guarantee: on any failure, `dir` is left absent (or
/// untouched if it didn't exist).  Staging temp dir is cleaned up.
pub fn install_from_tarball(tarball: &Path, dir: &Path, spec: &ModelSpec) -> Result<(), TalkError> {
    let parent = dir.parent().ok_or_else(|| {
        TalkError::Transcription(format!("install target {} has no parent", dir.display()))
    })?;
    std::fs::create_dir_all(parent).map_err(|e| {
        TalkError::Transcription(format!("create parent {}: {}", parent.display(), e))
    })?;

    // Staging dir lives next to `dir` on the same filesystem so the
    // final rename is atomic and cross-FS safe.
    let staging = parent.join(format!(
        "{}.staging.tmp",
        dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("talk-rs-model")
    ));
    // Wipe any leftover staging from a prior aborted run.
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| {
        TalkError::Transcription(format!("create staging dir {}: {}", staging.display(), e))
    })?;

    let cleanup_staging = |s: &Path| {
        let _ = std::fs::remove_dir_all(s);
    };

    // Decompress + untar straight into staging.
    let tarball_file = std::fs::File::open(tarball).map_err(|e| {
        cleanup_staging(&staging);
        TalkError::Transcription(format!("open tarball {}: {}", tarball.display(), e))
    })?;
    let decoder = bzip2::read::BzDecoder::new(tarball_file);
    let mut archive = tar::Archive::new(decoder);
    if let Err(e) = archive.unpack(&staging) {
        cleanup_staging(&staging);
        return Err(TalkError::Transcription(format!(
            "extract tarball into {}: {}",
            staging.display(),
            e
        )));
    }

    // Locate the inner dir.
    let inner_dir = locate_inner_dir(&staging, spec)?;

    // Verify all required files are present and non-empty.
    for name in spec.required_files {
        let p = inner_dir.join(name);
        let md = std::fs::metadata(&p).map_err(|e| {
            cleanup_staging(&staging);
            TalkError::Transcription(format!("extracted archive missing {}: {}", p.display(), e))
        })?;
        if !md.is_file() || md.len() == 0 {
            cleanup_staging(&staging);
            return Err(TalkError::Transcription(format!(
                "extracted archive has empty or non-file entry: {}",
                p.display()
            )));
        }
    }

    // Atomic promote: remove any partial existing dir, then rename.
    if let Err(e) = std::fs::remove_dir_all(dir) {
        if e.kind() != std::io::ErrorKind::NotFound {
            cleanup_staging(&staging);
            return Err(TalkError::Transcription(format!(
                "remove pre-existing model dir {}: {}",
                dir.display(),
                e
            )));
        }
    }

    if let Err(e) = std::fs::rename(&inner_dir, dir) {
        cleanup_staging(&staging);
        return Err(TalkError::Transcription(format!(
            "promote {} -> {}: {}",
            inner_dir.display(),
            dir.display(),
            e
        )));
    }

    // Inner dir is gone (renamed); clean up the now-empty staging
    // skeleton best-effort.
    cleanup_staging(&staging);

    Ok(())
}

/// Find the single nested top-level dir inside `staging`.
///
/// Preference order:
/// 1. The known upstream name (`spec.inner_dir`) if present.
/// 2. The lone non-hidden subdirectory, when staging contains exactly
///    one such dir (covers a hypothetical re-packaging).
fn locate_inner_dir(staging: &Path, spec: &ModelSpec) -> Result<PathBuf, TalkError> {
    let preferred = staging.join(spec.inner_dir);
    if preferred.is_dir() {
        return Ok(preferred);
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(staging).map_err(|e| {
        TalkError::Transcription(format!("read staging dir {}: {}", staging.display(), e))
    })?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            candidates.push(path);
        }
    }

    if candidates.len() == 1 {
        return Ok(candidates.remove(0));
    }

    Err(TalkError::Transcription(format!(
        "could not locate {} model dir inside extracted archive at {} \
         (expected `{}/` or a single top-level dir, found {} dirs)",
        spec.display_name,
        staging.display(),
        spec.inner_dir,
        candidates.len()
    )))
}

/// Ensure the model is present, asking for consent on the terminal
/// when it is not.
///
/// Behaviour when the model is absent:
///
/// * **Interactive (stdin is a TTY):** prints a `[y/N]` prompt to
///   stderr and reads the answer.  Anything other than `y`/`yes`
///   aborts with a [`TalkError::Config`] (no download).
/// * **Non-interactive (piped / no TTY):** logs a clear message and
///   proceeds — for unattended callers *selecting the local backend
///   is itself the consent*.
///
/// A no-op (returns `Ok`) when the model is already present.
pub async fn ensure_with_cli_consent(dir: &Path, spec: &ModelSpec) -> Result<(), TalkError> {
    if is_present(dir, spec) {
        return Ok(());
    }

    if std::io::stdin().is_terminal() {
        eprint!(
            "The {} model ({}) is not installed at {}.\n\
             Download it now? [y/N] ",
            spec.display_name,
            spec.approx_size,
            dir.display()
        );
        let _ = std::io::stderr().flush();

        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .map_err(TalkError::Io)?;
        let answer = answer.trim().to_ascii_lowercase();
        if answer != "y" && answer != "yes" {
            return Err(TalkError::Config(format!(
                "{} model download declined; nothing was downloaded. \
                 Install it later by re-running and accepting the prompt, \
                 or place the files manually in {}.",
                spec.display_name,
                dir.display()
            )));
        }
        eprintln!(
            "Downloading {} model to {} …",
            spec.display_name,
            dir.display()
        );
    } else {
        eprintln!(
            "{} model not found at {}; downloading ({}) — selecting this \
             local backend implies consent. Choose a different provider to \
             avoid this.",
            spec.display_name,
            dir.display(),
            spec.approx_size,
        );
    }

    download_and_install(dir, spec).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use bzip2::Compression;
    use tempfile::TempDir;

    const TEST_SPEC: ModelSpec = ModelSpec {
        display_name: "Test",
        tarball_url: "https://example.com/test-model.tar.bz2",
        inner_dir: "test-model-inner",
        required_files: &["a.onnx", "b.onnx", "tokens.txt"],
        approx_size: "~1 MB",
        manual_files_hint: "a.onnx/b.onnx/tokens.txt",
    };

    /// Build a synthetic `.tar.bz2` matching the upstream layout:
    /// every entry nested under `spec.inner_dir/`.
    fn make_synthetic_tarball(
        path: &Path,
        entries: &[(&str, &[u8])],
        inner: &str,
    ) -> Result<(), TalkError> {
        let file = std::fs::File::create(path).map_err(|e| {
            TalkError::Transcription(format!("create fixture {}: {}", path.display(), e))
        })?;
        let encoder = bzip2::write::BzEncoder::new(file, Compression::fast());
        let mut builder = tar::Builder::new(encoder);

        for (name, bytes) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            let entry_path = format!("{}/{}", inner, name);
            builder
                .append_data(&mut header, &entry_path, *bytes)
                .map_err(|e| TalkError::Transcription(format!("append {}: {}", entry_path, e)))?;
        }

        let encoder = builder
            .into_inner()
            .map_err(|e| TalkError::Transcription(format!("close tar: {}", e)))?;
        encoder
            .finish()
            .map_err(|e| TalkError::Transcription(format!("close bz2: {}", e)))?;
        Ok(())
    }

    fn populate_dir(dir: &Path, spec: &ModelSpec) {
        std::fs::create_dir_all(dir).unwrap();
        for name in spec.required_files {
            let mut f = std::fs::File::create(dir.join(name)).unwrap();
            f.write_all(b"dummy-bytes").unwrap();
        }
    }

    #[test]
    fn is_present_empty_dir_is_false() {
        let tmp = TempDir::new().unwrap();
        assert!(!is_present(tmp.path(), &TEST_SPEC));
    }

    #[test]
    fn is_present_complete_dir_is_true() {
        let tmp = TempDir::new().unwrap();
        populate_dir(tmp.path(), &TEST_SPEC);
        assert!(is_present(tmp.path(), &TEST_SPEC));
    }

    #[test]
    fn is_present_zero_byte_file_is_false() {
        let tmp = TempDir::new().unwrap();
        populate_dir(tmp.path(), &TEST_SPEC);
        std::fs::File::create(tmp.path().join("tokens.txt")).unwrap();
        assert!(!is_present(tmp.path(), &TEST_SPEC));
    }

    #[test]
    fn ensure_present_errors_without_download_when_absent() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("m");
        let err = ensure_present(&dir, &TEST_SPEC).expect_err("absent model must error");
        let msg = err.to_string();
        assert!(msg.contains("not downloaded automatically") || msg.contains("explicit consent"));
        assert!(msg.contains(TEST_SPEC.tarball_url));
        assert!(!dir.exists());
    }

    #[test]
    fn install_from_tarball_promotes_and_flattens() {
        let tmp = TempDir::new().unwrap();
        let tarball = tmp.path().join("model.tar.bz2");
        make_synthetic_tarball(
            &tarball,
            &[
                ("a.onnx", b"A"),
                ("b.onnx", b"B"),
                ("tokens.txt", b"T"),
                ("README.md", b"R"),
            ],
            TEST_SPEC.inner_dir,
        )
        .unwrap();

        let dir = tmp.path().join("model");
        install_from_tarball(&tarball, &dir, &TEST_SPEC).expect("install must succeed");

        assert!(is_present(&dir, &TEST_SPEC));
        assert!(!dir.join(TEST_SPEC.inner_dir).exists(), "nesting flattened");
        assert!(!tmp.path().join("model.staging.tmp").exists());
    }

    #[test]
    fn install_from_tarball_missing_file_leaves_dir_absent() {
        let tmp = TempDir::new().unwrap();
        let tarball = tmp.path().join("model.tar.bz2");
        make_synthetic_tarball(
            &tarball,
            &[("a.onnx", b"A"), ("tokens.txt", b"T")],
            TEST_SPEC.inner_dir,
        )
        .unwrap();

        let dir = tmp.path().join("model");
        let err =
            install_from_tarball(&tarball, &dir, &TEST_SPEC).expect_err("missing file must fail");
        assert!(err.to_string().contains("b.onnx"));
        assert!(!dir.exists(), "failed install must not leave a dir");
    }

    #[test]
    fn install_from_tarball_single_subdir_fallback() {
        // A re-packaged tarball with a different top-level name still
        // works because we fall back to the single non-hidden subdir.
        let tmp = TempDir::new().unwrap();
        let tarball = tmp.path().join("model.tar.bz2");
        make_synthetic_tarball(
            &tarball,
            &[("a.onnx", b"A"), ("b.onnx", b"B"), ("tokens.txt", b"T")],
            "some-other-name",
        )
        .unwrap();

        let dir = tmp.path().join("model");
        install_from_tarball(&tarball, &dir, &TEST_SPEC).expect("fallback install must succeed");
        assert!(is_present(&dir, &TEST_SPEC));
    }

    #[test]
    fn sibling_lock_path_is_sibling_not_child() {
        let dir = Path::new("/tmp/some/kokoro-model");
        let lock = sibling_lock_path(dir);
        assert_eq!(lock, Path::new("/tmp/some/kokoro-model.lock"));
    }

    #[tokio::test]
    async fn download_and_install_fast_path_when_present() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("m");
        populate_dir(&dir, &TEST_SPEC);
        // No network in the test runner; the fast path must
        // short-circuit before any HTTP call.
        let start = std::time::Instant::now();
        download_and_install(&dir, &TEST_SPEC)
            .await
            .expect("fast path must succeed");
        assert!(start.elapsed() < std::time::Duration::from_millis(100));
    }

    #[test]
    fn manual_fallback_message_contains_url_and_dir() {
        let msg = TEST_SPEC.manual_fallback_message("boom", Path::new("/some/where/m"));
        assert!(msg.contains(TEST_SPEC.tarball_url));
        assert!(msg.contains("/some/where/m"));
        assert!(msg.contains("a.onnx"));
        assert!(msg.contains("boom"));
    }
}
