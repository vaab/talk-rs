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

use std::io::{IsTerminal, Write};
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
pub async fn ensure_with_cli_consent(config: &Config) -> Result<(), TalkError> {
    let status = resolve(config)?;
    if status.present {
        return Ok(());
    }

    let dir = &status.model_dir;

    if std::io::stdin().is_terminal() {
        // Interactive: ask before downloading ~640 MB.
        eprint!(
            "The Parakeet speech model (~640 MB) is not installed at {}.\n\
             Download it now? [y/N] ",
            dir.display()
        );
        // Flush so the prompt shows before we block on stdin.
        let _ = std::io::stderr().flush();

        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .map_err(TalkError::Io)?;
        let answer = answer.trim().to_ascii_lowercase();
        if answer != "y" && answer != "yes" {
            return Err(TalkError::Config(format!(
                "Parakeet model download declined; nothing was downloaded. \
                 Install it later by re-running and accepting the prompt, \
                 or place the files manually in {}.",
                dir.display()
            )));
        }
        eprintln!("Downloading Parakeet model to {} …", dir.display());
    } else {
        // Non-interactive: selecting Parakeet is the consent. Make the
        // download loud on stderr so it is never a silent surprise.
        eprintln!(
            "Parakeet model not found at {}; downloading (~640 MB) — \
             selecting the parakeet provider implies consent. Set a \
             different provider to avoid this.",
            dir.display()
        );
    }

    model::download_model(dir, status.variant).await
}
