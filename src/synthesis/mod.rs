//! Speech synthesis (text-to-speech) interfaces and implementations.
//!
//! The synthesis-side mirror of [`crate::transcription`].  Where
//! transcription is *audio in, text out*, synthesis is *text in, audio
//! out*.  A single trait models the one mode of operation the `speak`
//! command needs:
//!
//! - [`OneShotSynthesizer`]: text in, full PCM buffer out.
//!
//! Two backends implement it, exactly paralleling the transcription
//! providers:
//!
//! - [`mistral`] — remote Voxtral TTS (`POST /v1/audio/speech`), the
//!   synthesis sibling of [`crate::transcription::mistral`].  No build
//!   feature (always available).
//! - [`kokoro`] — local Kokoro multi-lang TTS via sherpa-onnx, the
//!   synthesis sibling of [`crate::transcription::parakeet`].  Gated on
//!   the `kokoro` build feature.
//!
//! The [`create_oneshot_synthesizer`] factory dispatches on
//! [`SynthesisProvider`] with cfg-gated arms mirroring
//! [`crate::transcription::create_oneshot_transcriber`]: the
//! `#[cfg(feature = "kokoro")]` arm constructs, the
//! `#[cfg(not(feature = "kokoro"))]` arm returns a clear rebuild
//! instruction.

use crate::config::{Config, SynthesisProvider};
use crate::error::TalkError;
use async_trait::async_trait;

pub mod lang;
pub mod mistral;
pub mod mistral_presets;
pub mod resolve;

#[cfg(feature = "kokoro")]
pub mod kokoro;

pub use lang::detect_lang;
pub use resolve::{guard_voice_lang, primary_subtag, LangSource, VoiceMeta};

/// A single one-shot synthesis request.
///
/// Every field but `text` is optional; each backend resolves unset
/// fields against its own config defaults.
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    /// The text to synthesize.  Must be non-empty (the CLI rejects
    /// empty input before reaching a backend).
    pub text: String,
    /// Voice selector: a Kokoro voice name (`af_heart`, `ff_siwis`, …)
    /// or a Mistral preset voice id (UUID).  When `None` the backend
    /// falls back to its config default (and then a per-language
    /// default for Kokoro).
    pub voice: Option<String>,
    /// Speech rate multiplier (`1.0` = normal).  Honoured by Kokoro;
    /// ignored by the Mistral endpoint (which has no speed knob in v1).
    pub speed: Option<f32>,
    /// Phonemization language (`en` / `fr`).  Honoured by Kokoro (it
    /// selects the phonemizer + default voice); ignored by Mistral
    /// (voice choice implies language server-side).
    pub lang: Option<String>,
}

/// The result of a one-shot synthesis: mono `i16` PCM plus its sample
/// rate.
///
/// `i16` PCM is the codebase's canonical audio representation (audio
/// capture, the WAV/OGG writers, and Parakeet all speak `i16`), so the
/// synthesis result meets the playback / WAV-writer paths in their
/// native form with no extra conversion.
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Mono 16-bit signed PCM samples.
    pub pcm: Vec<i16>,
    /// Sample rate of `pcm` in Hz (Kokoro: 24000; Mistral Voxtral WAV:
    /// 24000).
    pub sample_rate: u32,
}

impl SynthesisResult {
    /// Duration of the synthesized audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.pcm.len() as f64 / self.sample_rate as f64
        }
    }
}

/// One-shot synthesis: text in, full PCM out.
///
/// All implementations must be `Send + Sync` for use in async
/// contexts.  Mirrors [`crate::transcription::OneShotTranscriber`].
#[async_trait]
pub(crate) trait OneShotSynthesizer: Send + Sync {
    /// Pre-flight check: verify the backend is usable (API key present,
    /// model files on disk, …) before doing real work.
    ///
    /// Implementations should keep this cheap and side-effect-free —
    /// e.g. the Mistral backend only checks that an API key is present
    /// (it does NOT call the API), and Kokoro checks the model files
    /// are on disk.
    async fn validate(&self) -> Result<(), TalkError>;

    /// Synthesize `req.text` into a [`SynthesisResult`].
    async fn synthesize(&self, req: SynthesisRequest) -> Result<SynthesisResult, TalkError>;
}

/// Synthesize `request` with `provider`, returning the audio.
///
/// THE public one-shot synthesis entry point, mirroring
/// [`crate::transcription::transcribe_audio`].  Constructs the backend
/// via [`create_oneshot_synthesizer`], runs its `validate` pre-flight,
/// then `synthesize`.  Callers that need the lower-level trait object
/// (e.g. to reuse it across requests) can use the crate-internal
/// factory directly.
pub async fn synthesize(
    config: &Config,
    provider: SynthesisProvider,
    request: SynthesisRequest,
) -> Result<SynthesisResult, TalkError> {
    let synthesizer = create_oneshot_synthesizer(config, provider)?;
    synthesizer.validate().await?;
    synthesizer.synthesize(request).await
}

/// Create a one-shot synthesizer for the given provider.
///
/// Mirrors [`crate::transcription::create_oneshot_transcriber`]: the
/// Kokoro arm is cfg-gated so a `--no-default-features` (or
/// `--features playback` without `kokoro`) build still compiles and
/// gives a clear rebuild instruction when the user asks for Kokoro.
pub(crate) fn create_oneshot_synthesizer(
    config: &Config,
    provider: SynthesisProvider,
) -> Result<Box<dyn OneShotSynthesizer>, TalkError> {
    match provider {
        SynthesisProvider::Mistral => {
            let cfg = config.providers.mistral.clone().ok_or_else(|| {
                TalkError::Config(
                    "Mistral synthesis selected but providers.mistral is not configured"
                        .to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.mistral.api_key is required".to_string(),
                ));
            }
            Ok(Box::new(mistral::MistralOneShotSynthesizer::new(cfg)?))
        }
        // The local Kokoro backend.  Construction is cheap (path
        // resolution only); the heavy model load happens on first
        // `synthesize`.  `validate` only CHECKS model presence — it
        // never downloads.  The download is an explicit, consented step
        // driven by the `speak` action before synthesis begins.
        #[cfg(feature = "kokoro")]
        SynthesisProvider::Kokoro => {
            let cfg = config.providers.kokoro.clone().unwrap_or_default();
            Ok(Box::new(kokoro::KokoroOneShotSynthesizer::new(cfg)?))
        }
        #[cfg(not(feature = "kokoro"))]
        SynthesisProvider::Kokoro => Err(TalkError::Config(
            "talk-rs was built without the 'kokoro' feature; rebuild with \
             --features kokoro to enable the local Kokoro TTS backend"
                .to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesis_result_duration() {
        let r = SynthesisResult {
            pcm: vec![0i16; 24_000],
            sample_rate: 24_000,
        };
        assert!((r.duration_secs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn synthesis_result_duration_zero_rate_is_zero() {
        let r = SynthesisResult {
            pcm: vec![0i16; 100],
            sample_rate: 0,
        };
        assert_eq!(r.duration_secs(), 0.0);
    }
}
