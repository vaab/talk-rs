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
                    .unwrap_or_else(|| "gpt-4o-mini-transcribe".to_string())
            } else {
                config
                    .providers
                    .openai
                    .as_ref()
                    .map(|c| c.model.clone())
                    .unwrap_or_else(|| "whisper-1".to_string())
            }
        }
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

/// Known Mistral models that support realtime (WebSocket) transcription.
const MISTRAL_REALTIME_MODELS: &[&str] = &["voxtral-mini-transcribe-realtime-2602"];

/// Known OpenAI models that support realtime (WebSocket) transcription.
const OPENAI_REALTIME_MODELS: &[&str] = &["gpt-4o-mini-transcribe", "gpt-4o-transcribe"];

/// Push all known realtime transcription models for `provider` into
/// `out`.  The third element of each tuple is `true` (streaming).
fn add_known_realtime_models(out: &mut Vec<(Provider, String, bool)>, provider: Provider) {
    let models = match provider {
        Provider::OpenAI => OPENAI_REALTIME_MODELS,
        Provider::Mistral => MISTRAL_REALTIME_MODELS,
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
            // Specific provider requested: add all known batch + realtime models.
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

/// Push all known batch transcription models for `provider` into `out`
/// as `(provider, model, streaming=false)` triples.
fn add_known_models_with_streaming(out: &mut Vec<(Provider, String, bool)>, provider: Provider) {
    let models = match provider {
        Provider::OpenAI => OPENAI_TRANSCRIPTION_MODELS,
        Provider::Mistral => MISTRAL_TRANSCRIPTION_MODELS,
    };
    for m in models {
        out.push((provider, (*m).to_string(), false));
    }
}
