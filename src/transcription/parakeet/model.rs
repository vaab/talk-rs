//! Parakeet model lifecycle: presence check, download, extract,
//! atomic promote.
//!
//! # Why this lives in its own module
//!
//! Local-inference backends have a concept the remote-API providers
//! don't: a multi-hundred-megabyte artifact that must be present on
//! disk before the first `BatchTranscriber::validate()` succeeds.
//! Keeping the I/O-heavy lifecycle (download streaming, bzip2 +
//! tar extraction, file-locking, atomic rename) isolated from the
//! Phase-3 transcriber keeps the inference path thin and makes the
//! tricky bits (concurrency, atomicity, corruption) directly
//! unit-testable.
//!
//! # Atomicity contract
//!
//! [`download_model`] and the lower-level [`install_from_tarball`]
//! guarantee that on any failure path the final `dir` is either
//! present-and-complete (4 non-empty files, [`is_present`] returns
//! true) or absent — never half-populated.  A crashed/interrupted
//! run must leave the next run with a clean retry.
//!
//! # Consent contract
//!
//! There is NO silent auto-download.  [`ensure_present`] (used by the
//! transcribe pipeline's `validate`) only *checks* for the model and
//! errors if it is absent.  [`download_model`] performs the actual
//! fetch and is invoked ONLY by entry surfaces that have first
//! obtained explicit user consent.
//!
//! # Concurrency contract
//!
//! Two concurrent processes calling [`download_model`] on the same
//! `dir` serialise on a sibling lock file (`<dir>.lock`,
//! `fs2::FileExt::lock_exclusive`).  The loser re-checks presence
//! under the lock and returns immediately when the winner finished.
//!
//! # Variant scope (v1)
//!
//! INT8 is the only prebuilt variant exposed by upstream
//! sherpa-onnx releases at the time of writing.  FP32 returns a
//! clear "not available as a prebuilt download" error pointing the
//! user at manual installation; no FP32 URL is attempted.

use crate::config::ParakeetVariant;
use crate::error::TalkError;
use fs2::FileExt;
use futures::StreamExt;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Tarball asset URL for the INT8 model.  Verified in Phase 0 of the
/// parakeet-local-backend plan.
const INT8_TARBALL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2";

/// Top-level directory name inside the INT8 tarball.  Every entry in
/// the archive is nested under this dir; the extractor flattens
/// the nesting so the final `dir` directly contains the model
/// files.
const INT8_TARBALL_INNER_DIR: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";

/// The 4 files a complete INT8 model dir must contain.
const MODEL_FILES_INT8: &[&str] = &[
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
];

/// Progress-log granularity for the streaming download.
const PROGRESS_LOG_EVERY_BYTES: u64 = 32 * 1024 * 1024; // 32 MiB

/// Returns the set of required files for `variant`.  INT8 today;
/// FP32 has no on-disk presence concept here because there is no
/// prebuilt tarball — manual installs of FP32 would have to mirror
/// this layout, but Phase 2 does not download FP32.
fn required_files(variant: ParakeetVariant) -> &'static [&'static str] {
    match variant {
        ParakeetVariant::Int8 => MODEL_FILES_INT8,
        // No FP32 prebuilt — but for `is_present` we still allow
        // a manually-placed FP32 model: the unquantized filenames
        // drop the `.int8` infix.  This is forward-looking and
        // makes `is_present(.., Fp32)` meaningful for manual
        // installs even though we never download FP32 ourselves.
        ParakeetVariant::Fp32 => &["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"],
    }
}

/// True iff every required file for `variant` exists under `dir`
/// AND is non-empty.  Non-empty matters because an interrupted
/// download could in principle leave a zero-byte placeholder; we
/// must not treat that as "present".
pub fn is_present(dir: &Path, variant: ParakeetVariant) -> bool {
    let files = required_files(variant);
    files.iter().all(|name| {
        let p = dir.join(name);
        match std::fs::metadata(&p) {
            Ok(m) => m.is_file() && m.len() > 0,
            Err(_) => false,
        }
    })
}

/// Format the manual-fallback error message.  Surfaced verbatim by
/// any download/extract/verify failure so a locked-down user can
/// install the model by hand.
fn manual_fallback_message(cause: &str, dir: &Path) -> String {
    format!(
        "Failed to download Parakeet model ({}). To install manually: \
         download {}, extract it, and place encoder.int8.onnx/\
         decoder.int8.onnx/joiner.int8.onnx/tokens.txt into {}.",
        cause,
        INT8_TARBALL_URL,
        dir.display()
    )
}

/// RAII unlock wrapper around an `fs2::FileExt::lock_exclusive`,
/// mirroring `transport::validate_cache::LockGuard` exactly so we
/// release the flock on any early return.
struct LockGuard<'a>(&'a std::fs::File);

impl Drop for LockGuard<'_> {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(self.0);
    }
}

/// Check that the model is present under `dir`, WITHOUT downloading.
///
/// This is the consent-safe presence gate: it performs no network
/// I/O and never fetches anything.  Callers that want the model
/// downloaded must obtain explicit user consent and then call
/// [`download_model`].
///
/// # Errors
///
/// Returns a [`TalkError::Config`] whose message names the model
/// directory and the exact `wget` URL for a manual install, so the
/// caller (or the user reading the error) knows how to provision the
/// model.  This is the error surfaces match on to trigger their own
/// consent flow.
pub fn ensure_present(dir: &Path, variant: ParakeetVariant) -> Result<(), TalkError> {
    if is_present(dir, variant) {
        return Ok(());
    }
    Err(TalkError::Config(format!(
        "Parakeet model not found in {}. The model is not downloaded \
         automatically; it must be fetched with explicit consent. \
         To install manually: download {}, extract it, and place \
         encoder.int8.onnx/decoder.int8.onnx/joiner.int8.onnx/tokens.txt \
         into {}.",
        dir.display(),
        INT8_TARBALL_URL,
        dir.display()
    )))
}

/// Download and install the model under `dir`.
///
/// **This always performs the network download** (after a
/// re-check under the lock).  It is the *explicit, consented* half of
/// the old `ensure_model`: callers MUST have obtained user consent
/// before invoking it — there is no silent-download path left in the
/// transcribe pipeline (see [`ensure_present`], which `validate`
/// uses).
///
/// See module docs for the atomicity + concurrency contract.
///
/// # Errors
///
/// * FP32 variant — always errors with the manual-install message
///   (no prebuilt tarball).
/// * Network / extract / verify failure — errors with the
///   manual-`wget`-fallback message that names the exact URL +
///   target dir.
pub async fn download_model(dir: &Path, variant: ParakeetVariant) -> Result<(), TalkError> {
    // 1. Fast path: already complete (idempotent — safe if two
    //    surfaces both decided to download).
    if is_present(dir, variant) {
        return Ok(());
    }

    // 2. FP32 has no prebuilt tarball.
    if variant == ParakeetVariant::Fp32 {
        return Err(TalkError::Config(format!(
            "Parakeet fp32 variant is not available as a prebuilt download; \
             place the model files manually in {} or use the int8 variant.",
            dir.display()
        )));
    }

    // 3. Make sure the parent exists, then take the sibling flock.
    let parent = dir.parent().ok_or_else(|| {
        TalkError::Config(format!(
            "Parakeet model_dir has no parent directory: {}",
            dir.display()
        ))
    })?;
    std::fs::create_dir_all(parent).map_err(|e| {
        TalkError::Config(format!(
            "Failed to create Parakeet model parent dir {}: {}",
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
                "Failed to open Parakeet model lock file {}: {}",
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

    // 4. Re-check under the lock — another process may have
    //    completed while we waited.
    if is_present(dir, variant) {
        return Ok(());
    }

    // 5. Stream the tarball to a temp file in the same parent dir
    //    so the eventual rename stays on a single filesystem.
    let tarball_tmp = parent.join(format!(
        "{}.download.tmp",
        dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("parakeet-model")
    ));
    // Best-effort wipe any leftover from a prior aborted run.
    let _ = std::fs::remove_file(&tarball_tmp);

    if let Err(e) = download_to(INT8_TARBALL_URL, &tarball_tmp).await {
        let _ = std::fs::remove_file(&tarball_tmp);
        return Err(TalkError::Config(manual_fallback_message(
            &e.to_string(),
            dir,
        )));
    }

    // 6–10. Extract + verify + atomic-promote (pure, testable).
    let res = install_from_tarball(&tarball_tmp, dir);
    // Tarball is no longer needed regardless of outcome.
    let _ = std::fs::remove_file(&tarball_tmp);

    if let Err(e) = res {
        // install_from_tarball already cleaned its own staging.
        // Wrap into the manual-fallback message so callers always
        // see the actionable text on a failed download path.
        return Err(TalkError::Config(manual_fallback_message(
            &e.to_string(),
            dir,
        )));
    }

    Ok(())
}

/// Path of the sibling lock file used to serialise concurrent
/// `ensure_model` callers.  Lives next to `dir`, not inside it, so
/// it survives a `remove_dir_all(dir)` cleanup without us having to
/// re-create it.
fn sibling_lock_path(dir: &Path) -> PathBuf {
    let mut p = dir.to_path_buf();
    let name = p
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("parakeet-model");
    let lock_name = format!("{}.lock", name);
    p.set_file_name(lock_name);
    p
}

/// Streaming download of `url` into `dest` via async reqwest.
///
/// Errors map straight to `TalkError::Transcription` strings; the
/// caller wraps them in the manual-fallback message.
async fn download_to(url: &str, dest: &Path) -> Result<(), TalkError> {
    log::info!("Parakeet model: downloading {} -> {}", url, dest.display());

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
                Some(total) => {
                    log::info!("Parakeet model: downloaded {} / {} bytes", written, total)
                }
                None => log::info!("Parakeet model: downloaded {} bytes", written),
            }
            next_log_at = next_log_at.saturating_add(PROGRESS_LOG_EVERY_BYTES);
        }
    }

    out.flush().map_err(|e| {
        TalkError::Transcription(format!("flush temp tarball {}: {}", dest.display(), e))
    })?;
    drop(out);

    log::info!("Parakeet model: download complete ({} bytes)", written);
    Ok(())
}

/// Extract `tarball` (a `.tar.bz2`), verify the 4 expected INT8
/// files, and atomically promote them so `dir` directly contains
/// the files (no nested subdir).
///
/// **Pure** (no network, no globals).  This is the testable seam:
/// unit tests build a tiny synthetic `.tar.bz2` and exercise
/// extract+verify+promote+atomicity directly without involving the
/// HTTP path.
///
/// Atomicity guarantee: on any failure, `dir` is left absent (or
/// untouched if it didn't exist).  Staging temp dir is cleaned up.
pub fn install_from_tarball(tarball: &Path, dir: &Path) -> Result<(), TalkError> {
    let parent = dir.parent().ok_or_else(|| {
        TalkError::Transcription(format!("install target {} has no parent", dir.display()))
    })?;
    std::fs::create_dir_all(parent).map_err(|e| {
        TalkError::Transcription(format!("create parent {}: {}", parent.display(), e))
    })?;

    // Staging dir lives next to `dir` on the same filesystem so
    // the final rename is atomic and cross-FS safe.
    let staging = parent.join(format!(
        "{}.staging.tmp",
        dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("parakeet-model")
    ));
    // Wipe any leftover staging from a prior aborted run.
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| {
        TalkError::Transcription(format!("create staging dir {}: {}", staging.display(), e))
    })?;

    // Helper to wipe staging on any failure path.
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

    // Locate the inner dir.  The tarball is documented to nest
    // every entry under `INT8_TARBALL_INNER_DIR/`; prefer that exact
    // path, but fall back to "the single non-hidden subdir" so a
    // re-packaged variant with a different top-level name still
    // works.
    let inner_dir = locate_inner_dir(&staging)?;

    // Verify all 4 required files are present and non-empty.
    for name in MODEL_FILES_INT8 {
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
    // The remove is best-effort — if `dir` doesn't exist (the
    // common case), `remove_dir_all` returns NotFound which we
    // ignore.  After the rename, `dir` directly contains the 4
    // files.
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
/// 1. The known upstream name (`INT8_TARBALL_INNER_DIR`) if present.
/// 2. The lone non-hidden subdirectory, when staging contains
///    exactly one such dir (covers a hypothetical re-packaging).
///
/// Returns an error when neither matches — that's a corrupt /
/// unrecognised tarball layout.
fn locate_inner_dir(staging: &Path) -> Result<PathBuf, TalkError> {
    let preferred = staging.join(INT8_TARBALL_INNER_DIR);
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
        "could not locate Parakeet model dir inside extracted archive at {} \
         (expected `{}/` or a single top-level dir, found {} dirs)",
        staging.display(),
        INT8_TARBALL_INNER_DIR,
        candidates.len()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bzip2::Compression;
    use tempfile::TempDir;

    /// Build a synthetic `.tar.bz2` matching the upstream layout:
    /// every entry nested under `INT8_TARBALL_INNER_DIR/`.  The
    /// `entries` argument is `(filename, bytes)` pairs.  Missing
    /// a required filename produces a corrupt/incomplete fixture.
    fn make_synthetic_tarball(path: &Path, entries: &[(&str, &[u8])]) -> Result<(), TalkError> {
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
            let entry_path = format!("{}/{}", INT8_TARBALL_INNER_DIR, name);
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

    /// Helper: populate `dir` with all 4 INT8 files (non-empty).
    fn populate_int8_dir(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        for name in MODEL_FILES_INT8 {
            let mut f = std::fs::File::create(dir.join(name)).unwrap();
            f.write_all(b"dummy-bytes").unwrap();
        }
    }

    #[test]
    fn is_present_empty_dir_is_false() {
        let tmp = TempDir::new().unwrap();
        assert!(!is_present(tmp.path(), ParakeetVariant::Int8));
    }

    #[test]
    fn is_present_complete_dir_is_true() {
        let tmp = TempDir::new().unwrap();
        populate_int8_dir(tmp.path());
        assert!(is_present(tmp.path(), ParakeetVariant::Int8));
    }

    #[test]
    fn is_present_missing_one_file_is_false() {
        let tmp = TempDir::new().unwrap();
        populate_int8_dir(tmp.path());
        std::fs::remove_file(tmp.path().join("joiner.int8.onnx")).unwrap();
        assert!(!is_present(tmp.path(), ParakeetVariant::Int8));
    }

    #[test]
    fn is_present_zero_byte_file_is_false() {
        let tmp = TempDir::new().unwrap();
        populate_int8_dir(tmp.path());
        // Truncate one file to zero bytes — must NOT count as present.
        std::fs::File::create(tmp.path().join("tokens.txt")).unwrap();
        assert!(!is_present(tmp.path(), ParakeetVariant::Int8));
    }

    #[tokio::test]
    async fn download_model_fp32_errors_without_network() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("p");
        let err = download_model(&dir, ParakeetVariant::Fp32)
            .await
            .expect_err("fp32 must not download");
        let msg = err.to_string();
        assert!(
            msg.contains("fp32"),
            "expected 'fp32' in error, got: {}",
            msg
        );
        assert!(
            msg.contains("manually"),
            "expected 'manually' in error, got: {}",
            msg
        );
        // dir must NOT have been created.
        assert!(!dir.exists(), "fp32 error path must not create model dir");
    }

    #[tokio::test]
    async fn download_model_fast_path_when_already_present() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("p");
        populate_int8_dir(&dir);

        // No network is available in the offline test runner; the
        // fast path must short-circuit before any HTTP call.  If
        // this hangs or fails, the fast path is broken.
        let start = std::time::Instant::now();
        download_model(&dir, ParakeetVariant::Int8)
            .await
            .expect("fast path must succeed");
        let elapsed = start.elapsed();
        // Generous bound — disk metadata only; even on slow CI
        // this is sub-millisecond.  100 ms is "definitely not a
        // network round trip".
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "fast path took {:?} — probably hit the network",
            elapsed
        );
    }

    /// `ensure_present` is the consent-safe gate: it must NEVER touch
    /// the network and must return Ok only when the files exist.
    #[test]
    fn ensure_present_ok_when_files_exist() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("p");
        populate_int8_dir(&dir);
        ensure_present(&dir, ParakeetVariant::Int8).expect("present dir must be Ok");
    }

    /// When the model is absent, `ensure_present` returns an error
    /// that points the user at a manual install (the URL + dir) and
    /// makes clear nothing was downloaded.  It must NOT create the
    /// dir or hit the network.
    #[test]
    fn ensure_present_errors_without_download_when_absent() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("p");
        let err = ensure_present(&dir, ParakeetVariant::Int8)
            .expect_err("absent model must error, not download");
        let msg = err.to_string();
        assert!(
            msg.contains("not downloaded automatically") || msg.contains("explicit consent"),
            "error must make clear there is no auto-download, got: {}",
            msg
        );
        assert!(
            msg.contains(INT8_TARBALL_URL),
            "error must name the manual-install URL, got: {}",
            msg
        );
        // Presence check must be side-effect free.
        assert!(!dir.exists(), "ensure_present must not create the dir");
    }

    #[test]
    fn install_from_tarball_promotes_and_flattens() {
        let staging_tmp = TempDir::new().unwrap();
        let tarball = staging_tmp.path().join("model.tar.bz2");
        make_synthetic_tarball(
            &tarball,
            &[
                ("encoder.int8.onnx", b"E"),
                ("decoder.int8.onnx", b"D"),
                ("joiner.int8.onnx", b"J"),
                ("tokens.txt", b"T"),
                // A bonus entry we should pass through without
                // tripping verification (it's just extra payload).
                ("README.md", b"R"),
            ],
        )
        .unwrap();

        let dir = staging_tmp.path().join("model");
        install_from_tarball(&tarball, &dir).expect("install must succeed");

        // Final dir directly contains the 4 files — no nested
        // sherpa-onnx-... subdir.
        assert!(is_present(&dir, ParakeetVariant::Int8));
        for name in MODEL_FILES_INT8 {
            assert!(dir.join(name).is_file(), "{} should be present", name);
        }
        // The nested dir name must NOT exist as a subdir of dir.
        assert!(
            !dir.join(INT8_TARBALL_INNER_DIR).exists(),
            "nesting must be flattened"
        );
        // Staging temp must be cleaned up.
        assert!(
            !staging_tmp.path().join("model.staging.tmp").exists(),
            "staging dir should be cleaned up"
        );
    }

    #[test]
    fn install_from_tarball_corrupt_missing_file_leaves_dir_absent() {
        let staging_tmp = TempDir::new().unwrap();
        let tarball = staging_tmp.path().join("model.tar.bz2");
        // Missing `joiner.int8.onnx` — incomplete tarball.
        make_synthetic_tarball(
            &tarball,
            &[
                ("encoder.int8.onnx", b"E"),
                ("decoder.int8.onnx", b"D"),
                ("tokens.txt", b"T"),
            ],
        )
        .unwrap();

        let dir = staging_tmp.path().join("model");
        let err = install_from_tarball(&tarball, &dir).expect_err("missing file must fail");
        assert!(
            err.to_string().contains("joiner.int8.onnx"),
            "error should name the missing file: {}",
            err
        );

        // Atomicity: no half-populated dir left behind.
        assert!(
            !dir.exists(),
            "failed install must NOT leave a half-populated dir"
        );
        assert!(
            !staging_tmp.path().join("model.staging.tmp").exists(),
            "staging dir should be cleaned up on failure"
        );
    }

    #[test]
    fn install_from_tarball_rejects_empty_file() {
        let staging_tmp = TempDir::new().unwrap();
        let tarball = staging_tmp.path().join("model.tar.bz2");
        // All 4 names present, but one is zero-length — corrupt.
        make_synthetic_tarball(
            &tarball,
            &[
                ("encoder.int8.onnx", b"E"),
                ("decoder.int8.onnx", b"D"),
                ("joiner.int8.onnx", b""), // empty
                ("tokens.txt", b"T"),
            ],
        )
        .unwrap();

        let dir = staging_tmp.path().join("model");
        let err = install_from_tarball(&tarball, &dir).expect_err("empty file must fail");
        assert!(
            err.to_string().contains("empty"),
            "error should mention 'empty': {}",
            err
        );
        assert!(!dir.exists(), "atomicity violated");
    }

    #[test]
    fn install_from_tarball_overwrites_partial_existing_dir() {
        // Simulate: a previous run crashed mid-install and left a
        // stale half-populated dir.  A subsequent successful
        // install must atomically replace it.
        let staging_tmp = TempDir::new().unwrap();
        let tarball = staging_tmp.path().join("model.tar.bz2");
        make_synthetic_tarball(
            &tarball,
            &[
                ("encoder.int8.onnx", b"NEW-E"),
                ("decoder.int8.onnx", b"NEW-D"),
                ("joiner.int8.onnx", b"NEW-J"),
                ("tokens.txt", b"NEW-T"),
            ],
        )
        .unwrap();

        let dir = staging_tmp.path().join("model");
        std::fs::create_dir_all(&dir).unwrap();
        // Stale file from a prior aborted run.
        std::fs::write(dir.join("stale.txt"), b"old").unwrap();

        install_from_tarball(&tarball, &dir).expect("install must succeed");

        assert!(is_present(&dir, ParakeetVariant::Int8));
        // Stale leftover must be gone — we did a clean replace.
        assert!(!dir.join("stale.txt").exists(), "stale file survived");
        // Confirm new content landed.
        let content = std::fs::read(dir.join("encoder.int8.onnx")).unwrap();
        assert_eq!(content, b"NEW-E");
    }

    #[test]
    fn manual_fallback_message_contains_url_and_dir() {
        let dir = Path::new("/some/where/parakeet");
        let msg = manual_fallback_message("boom", dir);
        assert!(msg.contains(INT8_TARBALL_URL));
        assert!(msg.contains("/some/where/parakeet"));
        assert!(msg.contains("encoder.int8.onnx"));
        assert!(msg.contains("boom"));
    }

    #[test]
    fn sibling_lock_path_is_sibling_not_child() {
        let dir = Path::new("/tmp/some/parakeet-model");
        let lock = sibling_lock_path(dir);
        assert_eq!(lock, Path::new("/tmp/some/parakeet-model.lock"));
    }
}
