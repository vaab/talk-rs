//! Parakeet model lifecycle: presence check, download, extract,
//! atomic promote.
//!
//! # Why this lives in its own module
//!
//! Local-inference backends have a concept the remote-API providers
//! don't: a multi-hundred-megabyte artifact that must be present on
//! disk before the first `OneShotTranscriber::validate()` succeeds.
//!
//! The generic, backend-agnostic machinery (download streaming,
//! bzip2 + tar extraction, file-locking, atomic rename) now lives in
//! [`crate::model_fetch`] and is SHARED with the Kokoro TTS backend.
//! This module keeps only the parakeet-specific bits: the
//! variant-keyed file layout (INT8 vs FP32) and the FP32
//! "no-prebuilt-tarball" guard.  It builds a
//! [`crate::model_fetch::ModelSpec`] for the INT8 variant and
//! delegates download/extract to the shared helper.
//!
//! # Atomicity contract
//!
//! [`download_model`] delegates to
//! [`crate::model_fetch::download_and_install`], which guarantees that
//! on any failure path the final `dir` is either present-and-complete
//! or absent — never half-populated.
//!
//! # Consent contract
//!
//! There is NO silent auto-download.  [`ensure_present`] (used by the
//! transcribe pipeline's `validate`) only *checks* for the model and
//! errors if it is absent.  [`download_model`] performs the actual
//! fetch and is invoked ONLY by entry surfaces that have first
//! obtained explicit user consent.
//!
//! # Variant scope (v1)
//!
//! INT8 is the only prebuilt variant exposed by upstream sherpa-onnx
//! releases.  FP32 returns a clear "not available as a prebuilt
//! download" error pointing the user at manual installation.

use crate::config::ParakeetVariant;
use crate::error::TalkError;
use crate::model_fetch::ModelSpec;
use std::path::Path;

/// Tarball asset URL for the INT8 model.  Verified in Phase 0 of the
/// parakeet-local-backend plan.
const INT8_TARBALL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2";

/// Top-level directory name inside the INT8 tarball.
const INT8_TARBALL_INNER_DIR: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";

/// The 4 files a complete INT8 model dir must contain.
const MODEL_FILES_INT8: &[&str] = &[
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
];

/// The 4 files a complete FP32 (unquantized) model dir must contain.
const MODEL_FILES_FP32: &[&str] = &["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"];

/// The shared [`ModelSpec`] describing the INT8 tarball for the
/// generic fetch machinery.  Exposed to the sibling `consent` module
/// so the CLI-consent flow delegates to
/// [`crate::model_fetch::ensure_with_cli_consent`].
pub(super) const INT8_SPEC: ModelSpec = ModelSpec {
    display_name: "Parakeet",
    tarball_url: INT8_TARBALL_URL,
    inner_dir: INT8_TARBALL_INNER_DIR,
    required_files: MODEL_FILES_INT8,
    approx_size: "~640 MB",
    manual_files_hint: "encoder.int8.onnx/decoder.int8.onnx/joiner.int8.onnx/tokens.txt",
};

/// Returns the set of required files for `variant`.
fn required_files(variant: ParakeetVariant) -> &'static [&'static str] {
    match variant {
        ParakeetVariant::Int8 => MODEL_FILES_INT8,
        // No FP32 prebuilt — but for `is_present` we still allow a
        // manually-placed FP32 model with the unquantized filenames.
        ParakeetVariant::Fp32 => MODEL_FILES_FP32,
    }
}

/// True iff every required file for `variant` exists under `dir` AND
/// is non-empty.
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

/// Check that the model is present under `dir`, WITHOUT downloading.
///
/// This is the consent-safe presence gate: it performs no network I/O
/// and never fetches anything.
///
/// # Errors
///
/// Returns a [`TalkError::Config`] whose message names the model
/// directory and the exact URL for a manual install.
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
/// **This always performs the network download** (after a re-check
/// under the lock).  Delegates the mechanics to
/// [`crate::model_fetch::download_and_install`]; only the FP32 guard is
/// parakeet-specific.
///
/// # Errors
///
/// * FP32 variant — always errors with the manual-install message
///   (no prebuilt tarball).
/// * Network / extract / verify failure — errors with the
///   manual-`wget`-fallback message that names the exact URL + target
///   dir.
pub async fn download_model(dir: &Path, variant: ParakeetVariant) -> Result<(), TalkError> {
    // Fast path: already complete (idempotent).
    if is_present(dir, variant) {
        return Ok(());
    }

    // FP32 has no prebuilt tarball.
    if variant == ParakeetVariant::Fp32 {
        return Err(TalkError::Config(format!(
            "Parakeet fp32 variant is not available as a prebuilt download; \
             place the model files manually in {} or use the int8 variant.",
            dir.display()
        )));
    }

    crate::model_fetch::download_and_install(dir, &INT8_SPEC).await
}

/// Extract `tarball` (a `.tar.bz2`), verify the 4 expected INT8 files,
/// and atomically promote them.  Thin wrapper over
/// [`crate::model_fetch::install_from_tarball`] with the INT8 spec.
///
/// **Pure** (no network, no globals).
pub fn install_from_tarball(tarball: &Path, dir: &Path) -> Result<(), TalkError> {
    crate::model_fetch::install_from_tarball(tarball, dir, &INT8_SPEC)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bzip2::Compression;
    use std::io::Write;
    use std::path::Path;
    use tempfile::TempDir;

    /// Build a synthetic `.tar.bz2` matching the upstream layout.
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
        assert!(!dir.exists(), "fp32 error path must not create model dir");
    }

    #[tokio::test]
    async fn download_model_fast_path_when_already_present() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("p");
        populate_int8_dir(&dir);

        let start = std::time::Instant::now();
        download_model(&dir, ParakeetVariant::Int8)
            .await
            .expect("fast path must succeed");
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "fast path took {:?} — probably hit the network",
            elapsed
        );
    }

    #[test]
    fn ensure_present_ok_when_files_exist() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("p");
        populate_int8_dir(&dir);
        ensure_present(&dir, ParakeetVariant::Int8).expect("present dir must be Ok");
    }

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
                ("README.md", b"R"),
            ],
        )
        .unwrap();

        let dir = staging_tmp.path().join("model");
        install_from_tarball(&tarball, &dir).expect("install must succeed");

        assert!(is_present(&dir, ParakeetVariant::Int8));
        for name in MODEL_FILES_INT8 {
            assert!(dir.join(name).is_file(), "{} should be present", name);
        }
        assert!(
            !dir.join(INT8_TARBALL_INNER_DIR).exists(),
            "nesting must be flattened"
        );
        assert!(
            !staging_tmp.path().join("model.staging.tmp").exists(),
            "staging dir should be cleaned up"
        );
    }

    #[test]
    fn install_from_tarball_corrupt_missing_file_leaves_dir_absent() {
        let staging_tmp = TempDir::new().unwrap();
        let tarball = staging_tmp.path().join("model.tar.bz2");
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
        std::fs::write(dir.join("stale.txt"), b"old").unwrap();

        install_from_tarball(&tarball, &dir).expect("install must succeed");

        assert!(is_present(&dir, ParakeetVariant::Int8));
        assert!(!dir.join("stale.txt").exists(), "stale file survived");
        let content = std::fs::read(dir.join("encoder.int8.onnx")).unwrap();
        assert_eq!(content, b"NEW-E");
    }
}
