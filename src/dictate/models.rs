//! Model catalog: provider/model resolution and known model lists.

use crate::config::{Config, Provider};

/// Resolve the effective provider from CLI override or config default.
pub(super) fn resolve_provider(cli_provider: Option<Provider>, config: &Config) -> Provider {
    if let Some(p) = cli_provider {
        return p;
    }
    config
        .transcription
        .as_ref()
        .map(|t| t.default_provider)
        .unwrap_or(Provider::Mistral)
}

/// Resolve the effective model name from CLI override or config default.
pub(super) fn resolve_model(
    cli_model: Option<&str>,
    config: &Config,
    provider: Provider,
    realtime: bool,
) -> String {
    if let Some(m) = cli_model {
        return m.to_string();
    }
    match provider {
        Provider::Mistral => config
            .providers
            .mistral
            .as_ref()
            .map(|c| c.model.clone())
            .unwrap_or_else(|| "voxtral-mini-2507".to_string()),
        Provider::OpenAI => {
            if realtime {
                config
                    .providers
                    .openai
                    .as_ref()
                    .map(|c| c.realtime_model.clone())
                    .unwrap_or_else(|| "gpt-realtime-whisper".to_string())
            } else {
                config
                    .providers
                    .openai
                    .as_ref()
                    .map(|c| c.model.clone())
                    .unwrap_or_else(|| "whisper-1".to_string())
            }
        }
        // Parakeet has only a one-shot model; the `realtime` flag is
        // irrelevant.  Mirrors `ParakeetConfig::resolved_model_name`.
        Provider::Parakeet => config
            .providers
            .parakeet
            .as_ref()
            .map(|c| c.resolved_model_name())
            .unwrap_or_else(|| "parakeet-tdt-0.6b-v3-int8".to_string()),
    }
}

/// Known OpenAI models for the `/v1/audio/transcriptions` endpoint.
const OPENAI_TRANSCRIPTION_MODELS: &[&str] =
    &["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"];

/// Known Mistral models for the `/v1/audio/transcriptions` endpoint.
///
/// `voxtral-mini-latest` aliases to `voxtral-mini-2602` for
/// transcription; we use explicit version names so the user can
/// compare results between generations.
const MISTRAL_TRANSCRIPTION_MODELS: &[&str] = &["voxtral-mini-2507", "voxtral-mini-2602"];

/// Known local Parakeet models.  v1 ships INT8 only; FP32 is a
/// config-accepted variant whose download is deferred to a later
/// phase.
const PARAKEET_TRANSCRIPTION_MODELS: &[&str] = &["parakeet-tdt-0.6b-v3-int8"];

/// Known Mistral models that support realtime (WebSocket) transcription.
const MISTRAL_REALTIME_MODELS: &[&str] = &["voxtral-mini-transcribe-realtime-2602"];

/// Known OpenAI models that support realtime (WebSocket) transcription.
///
/// After the 2026-02-27 Realtime API GA cutover, the historical
/// list ``["gpt-4o-mini-transcribe", "gpt-4o-transcribe"]`` is
/// no longer accepted by the GA endpoint — those names return
/// ``invalid_model`` from the realtime transcription path.  The
/// GA model for streaming transcription is
/// ``"gpt-realtime-whisper"``.
const OPENAI_REALTIME_MODELS: &[&str] = &["gpt-realtime-whisper"];

/// Push all known realtime transcription models for `provider` into
/// `out`.  The third element of each tuple is `true` (streaming).
fn add_known_realtime_models(out: &mut Vec<(Provider, String, bool)>, provider: Provider) {
    let models = match provider {
        Provider::OpenAI => OPENAI_REALTIME_MODELS,
        Provider::Mistral => MISTRAL_REALTIME_MODELS,
        // Parakeet has no realtime mode.
        Provider::Parakeet => &[],
    };
    for m in models {
        out.push((provider, (*m).to_string(), true));
    }
}

pub(super) fn build_retry_candidates(
    config: &Config,
    cli_provider: Option<Provider>,
    cli_model: Option<&str>,
) -> Vec<(Provider, String, bool)> {
    let mut out: Vec<(Provider, String, bool)> = Vec::new();

    // If the user explicitly specified a model, include it even if it
    // is not in the known list (e.g. a dated snapshot or new model).
    if let (Some(provider), Some(model)) = (cli_provider, cli_model) {
        out.push((provider, model.to_string(), false));
    }

    match cli_provider {
        Some(provider) => {
            // Specific provider requested: add all known one-shot + realtime models.
            add_known_models_with_streaming(&mut out, provider);
            add_known_realtime_models(&mut out, provider);
        }
        None => {
            // No provider filter: add all known models for every
            // provider, plus the config defaults (in case the user
            // configured a model we do not list).
            add_known_models_with_streaming(&mut out, Provider::OpenAI);
            add_known_models_with_streaming(&mut out, Provider::Mistral);
            add_known_realtime_models(&mut out, Provider::OpenAI);
            add_known_realtime_models(&mut out, Provider::Mistral);
            out.push((
                Provider::OpenAI,
                resolve_model(None, config, Provider::OpenAI, false),
                false,
            ));
            out.push((
                Provider::Mistral,
                resolve_model(None, config, Provider::Mistral, false),
                false,
            ));
            // Parakeet is the only LOCAL backend and selecting it can
            // trigger a multi-hundred-MB model download.  It must never
            // be surfaced without the user opting in, so only include it
            // in the no-filter picker when a `providers.parakeet:` block
            // exists in config.  An explicit `--provider parakeet` still
            // works (handled by the `Some(provider)` arm above).  No
            // realtime entry — the local backend has no streaming mode.
            if config.providers.parakeet.is_some() {
                add_known_models_with_streaming(&mut out, Provider::Parakeet);
                out.push((
                    Provider::Parakeet,
                    resolve_model(None, config, Provider::Parakeet, false),
                    false,
                ));
            }
        }
    }

    out.sort_by(|a, b| {
        a.0.to_string()
            .cmp(&b.0.to_string())
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    out.dedup();
    out
}

/// Push all known one-shot transcription models for `provider` into `out`
/// as `(provider, model, streaming=false)` triples.
fn add_known_models_with_streaming(out: &mut Vec<(Provider, String, bool)>, provider: Provider) {
    let models = match provider {
        Provider::OpenAI => OPENAI_TRANSCRIPTION_MODELS,
        Provider::Mistral => MISTRAL_TRANSCRIPTION_MODELS,
        Provider::Parakeet => PARAKEET_TRANSCRIPTION_MODELS,
    };
    for m in models {
        out.push((provider, (*m).to_string(), false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ParakeetConfig, ParakeetVariant, ProvidersConfig};
    use std::path::PathBuf;

    /// Minimal `Config` for candidate-building tests, with the
    /// `parakeet` provider section optionally present.
    fn config_with_parakeet(parakeet: Option<ParakeetConfig>) -> Config {
        Config {
            output_dir: PathBuf::from("/tmp/talk-rs-test"),
            providers: ProvidersConfig {
                mistral: None,
                openai: None,
                parakeet,
            },
            indicators: None,
            transcription: None,
            paste: None,
            audio: None,
            recording: None,
        }
    }

    fn parakeet_section() -> ParakeetConfig {
        ParakeetConfig {
            variant: ParakeetVariant::Int8,
            model_dir: None,
            num_threads: 2,
            model: None,
        }
    }

    /// The local Parakeet backend must NOT appear in the no-filter
    /// `--pick` candidate list unless the user has explicitly opted in
    /// with a `providers.parakeet:` block.  Otherwise the picker would
    /// dangle an option that can trigger a multi-hundred-MB download.
    #[test]
    fn no_filter_omits_parakeet_when_unconfigured() {
        let config = config_with_parakeet(None);
        let candidates = build_retry_candidates(&config, None, None);
        assert!(
            !candidates.iter().any(|(p, _, _)| *p == Provider::Parakeet),
            "Parakeet must not be an auto-candidate when unconfigured; got: {:?}",
            candidates
        );
    }

    /// When the user HAS configured `providers.parakeet`, the no-filter
    /// picker includes it (explicit opt-in).
    #[test]
    fn no_filter_includes_parakeet_when_configured() {
        let config = config_with_parakeet(Some(parakeet_section()));
        let candidates = build_retry_candidates(&config, None, None);
        assert!(
            candidates.iter().any(|(p, _, _)| *p == Provider::Parakeet),
            "Parakeet should be an auto-candidate when configured; got: {:?}",
            candidates
        );
        // Still never offered as a realtime (streaming) candidate.
        assert!(
            !candidates
                .iter()
                .any(|(p, _, streaming)| *p == Provider::Parakeet && *streaming),
            "Parakeet has no realtime mode; must not appear as streaming"
        );
    }

    /// An explicit `--provider parakeet` works regardless of whether a
    /// config section exists (the `Some(provider)` arm bypasses the
    /// opt-in gate).
    #[test]
    fn explicit_provider_parakeet_yields_candidates_without_config() {
        let config = config_with_parakeet(None);
        let candidates = build_retry_candidates(&config, Some(Provider::Parakeet), None);
        assert!(
            candidates.iter().any(|(p, _, _)| *p == Provider::Parakeet),
            "explicit --provider parakeet must yield Parakeet candidates; got: {:?}",
            candidates
        );
    }
}
