//! Single catalog of known transcription models and their capabilities.

use crate::config::Provider;

pub(crate) const MISTRAL_TRANSCRIPTION_MODELS: &[&str] =
    &["voxtral-mini-2507", "voxtral-mini-2602"];
pub(crate) const MISTRAL_REALTIME_MODELS: &[&str] = &["voxtral-mini-transcribe-realtime-2602"];
pub(crate) const PARAKEET_TRANSCRIPTION_MODELS: &[&str] = &["parakeet-tdt-0.6b-v3-int8"];
pub(crate) const OPENAI_BATCH_MODELS: &[&str] = &[
    "gpt-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "whisper-1",
];
pub(crate) const OPENAI_REALTIME_MODELS: &[&str] = &["gpt-live-transcribe", "gpt-realtime-whisper"];

pub(crate) fn known_models(provider: Provider, realtime: bool) -> &'static [&'static str] {
    match (provider, realtime) {
        (Provider::Mistral, false) => MISTRAL_TRANSCRIPTION_MODELS,
        (Provider::Mistral, true) => MISTRAL_REALTIME_MODELS,
        (Provider::OpenAI, false) => OPENAI_BATCH_MODELS,
        (Provider::OpenAI, true) => OPENAI_REALTIME_MODELS,
        (Provider::Parakeet, false) => PARAKEET_TRANSCRIPTION_MODELS,
        (Provider::Parakeet, true) => &[],
    }
}

pub(crate) fn supports(
    provider: Provider,
    model: &str,
    capability: &str,
    declared: &[String],
) -> bool {
    if provider.is_local() {
        return false;
    }
    let known = known_models(provider, false).contains(&model)
        || known_models(provider, true).contains(&model)
        || (provider == Provider::Mistral && model == "voxtral-mini-latest");
    match capability {
        "diarize" => match provider {
            Provider::Mistral if model == "voxtral-mini-latest" => true,
            Provider::Mistral
                if model
                    .strip_prefix("voxtral-mini-")
                    .and_then(|version| version.parse::<u32>().ok())
                    .is_some_and(|version| version >= 2602) =>
            {
                true
            }
            _ => !known && declared.iter().any(|s| s == capability),
        },
        "realtime" => {
            known_models(provider, true).contains(&model)
                || (!known && declared.iter().any(|s| s == capability))
        }
        _ => false,
    }
}
