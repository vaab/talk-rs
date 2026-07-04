//! Download-consent helper for the local Kokoro TTS backend.
//!
//! The synthesis-side sibling of
//! [`crate::transcription::parakeet::consent`].  The `speak` action
//! calls [`ensure_with_cli_consent`] before synthesis so the model is
//! fetched (with explicit consent) exactly once; thereafter it is a
//! no-op.  The interactive `[y/N]` prompt / non-interactive
//! proceed-with-log mechanics live in the shared
//! [`crate::model_fetch`] helper.

use crate::config::Config;
use crate::error::TalkError;

use super::model;

/// Resolved Kokoro model location + whether it is already present.
pub struct ModelStatus {
    /// The directory the model lives in (resolved from config / XDG
    /// default).
    pub model_dir: std::path::PathBuf,
    /// `true` when the core model files are already on disk.
    pub present: bool,
}

/// Resolve the Kokoro model directory from config and report whether
/// the model is already present on disk.  Zero-config: uses defaults
/// when no `providers.kokoro:` block exists.
pub fn resolve(config: &Config) -> Result<ModelStatus, TalkError> {
    let cfg = config.providers.kokoro.clone().unwrap_or_default();
    let model_dir = cfg.resolved_model_dir()?;
    let present = model::is_present(&model_dir);
    Ok(ModelStatus { model_dir, present })
}

/// Ensure the Kokoro model is present, asking for consent on the
/// terminal when it is not (a no-op when already present).  Delegates
/// to [`model::ensure_with_cli_consent`].
pub async fn ensure_with_cli_consent(config: &Config) -> Result<(), TalkError> {
    let status = resolve(config)?;
    if status.present {
        return Ok(());
    }
    model::ensure_with_cli_consent(&status.model_dir).await
}
