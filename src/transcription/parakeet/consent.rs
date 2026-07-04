//! Download-consent helpers for the local Parakeet backend.
//!
//! The transcribe pipeline NEVER downloads the model on its own
//! (`model::ensure_present` only checks; it never fetches).  Instead,
//! each entry surface — the `transcribe` CLI, `dictate --toggle`, the
//! `--pick` picker — is responsible for obtaining explicit user
//! consent and then calling [`model::download_model`].
//!
//! This module holds the shared, surface-agnostic pieces:
//!
//! * [`resolve`] — turn a [`Config`] into the Parakeet model
//!   directory + variant + a presence flag, so a surface can decide
//!   whether a consent prompt is even needed.
//! * [`ensure_with_cli_consent`] — the terminal/CLI consent flow
//!   (interactive y/n on a TTY; a clear stderr log + proceed when
//!   non-interactive, since *selecting Parakeet is the consent* for
//!   unattended callers).
//!
//! The GTK picker drives its own dialog-based consent (it cannot
//! block on stdin) and the toggle overlay drives a progress badge;
//! both call [`model::download_model`] directly after consent, reusing
//! [`resolve`] for the presence decision.

use std::path::PathBuf;

use crate::config::{Config, ParakeetVariant};
use crate::error::TalkError;

use super::model;

/// Resolved Parakeet model location + whether it is already present.
pub struct ModelStatus {
    /// The directory the model lives in (resolved from config /
    /// XDG default).
    pub model_dir: PathBuf,
    /// The quantization variant selected.
    pub variant: ParakeetVariant,
    /// `true` when all required model files are already on disk, so
    /// no download (and therefore no consent) is needed.
    pub present: bool,
}

/// Resolve the Parakeet model directory + variant from config and
/// report whether the model is already present on disk.
///
/// Uses defaults when no `providers.parakeet:` block exists (the
/// backend is zero-config). Returns an error only when the model
/// directory cannot be resolved at all (e.g. no home directory).
pub fn resolve(config: &Config) -> Result<ModelStatus, TalkError> {
    let cfg = config.providers.parakeet.clone().unwrap_or_default();
    let model_dir = cfg.resolved_model_dir()?;
    let variant = cfg.resolved_variant();
    let present = model::is_present(&model_dir, variant);
    Ok(ModelStatus {
        model_dir,
        variant,
        present,
    })
}

/// Ensure the Parakeet model is present, asking for consent on the
/// terminal when it is not.
///
/// Behaviour when the model is absent:
///
/// * **Interactive (stdin is a TTY):** prints a `[y/N]` prompt to
///   stderr and reads the answer. Anything other than `y`/`yes`
///   aborts with a [`TalkError::Config`] (no download).
/// * **Non-interactive (piped / no TTY):** logs a clear message to
///   stderr and proceeds with the download — for unattended callers
///   *selecting Parakeet is itself the consent* (user-confirmed
///   policy). This keeps scripts/cron working without a hung prompt.
///
/// A no-op (returns `Ok`) when the model is already present.
///
/// Delegates the interactive `[y/N]` prompt / non-interactive
/// proceed-with-log mechanics to the shared
/// [`crate::model_fetch::ensure_with_cli_consent`], passing the
/// parakeet INT8 [`ModelSpec`](crate::model_fetch::ModelSpec).  This
/// is behaviour-preserving: the prompt copy is templated from the
/// spec's `display_name` / `approx_size`, and the download still runs
/// through [`model::download_model`]'s atomic + concurrency path.
pub async fn ensure_with_cli_consent(config: &Config) -> Result<(), TalkError> {
    let status = resolve(config)?;
    if status.present {
        return Ok(());
    }

    // FP32 has no prebuilt tarball; surface the clear error rather than
    // prompting to download something that cannot be fetched.
    if status.variant == ParakeetVariant::Fp32 {
        return model::download_model(&status.model_dir, status.variant).await;
    }

    crate::model_fetch::ensure_with_cli_consent(&status.model_dir, &model::INT8_SPEC).await
}
