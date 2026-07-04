//! Local Kokoro TTS `OneShotSynthesizer` via sherpa-onnx.
//!
//! The synthesis sibling of
//! [`crate::transcription::parakeet::ParakeetOneShotTranscriber`]: a
//! fresh `OfflineTts` engine per call, the blocking C synthesis work
//! offloaded to [`tokio::task::spawn_blocking`].
//!
//! # Language selection (config-driven, agnostic)
//!
//! The language is resolved as: request `--lang` > config
//! `providers.kokoro.lang` > the stock model's *baked* language
//! (discovered from ONNX metadata — see
//! [`super::model::baked_language`]).  No language is hardcoded here.
//! [`super::model::ensure_lang_model`] returns the correct
//! `model.onnx` / `model-<lang>.onnx` for the resolved language,
//! deriving the per-language model on demand.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use sherpa_onnx::{GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig};

use crate::config::KokoroConfig;
use crate::error::TalkError;

use super::model;
use super::voices;
use crate::synthesis::{OneShotSynthesizer, SynthesisRequest, SynthesisResult};

/// Local Kokoro one-shot synthesizer.
///
/// Construction is cheap (path resolution only); the heavy
/// `OfflineTts::create` happens lazily inside [`Self::synthesize`].
pub struct KokoroOneShotSynthesizer {
    model_dir: PathBuf,
    /// Default voice name from config (`providers.kokoro.voice`).
    config_voice: Option<String>,
    /// Default language from config (`providers.kokoro.lang`).
    config_lang: Option<String>,
    num_threads: usize,
}

impl KokoroOneShotSynthesizer {
    /// Build a synthesizer from a parsed [`KokoroConfig`].
    pub fn new(cfg: KokoroConfig) -> Result<Self, TalkError> {
        let model_dir = cfg.resolved_model_dir()?;
        Ok(Self {
            model_dir,
            config_voice: cfg.voice.clone(),
            config_lang: cfg.lang.clone(),
            num_threads: cfg.resolved_num_threads(),
        })
    }

    /// Resolve the effective language for a request.
    ///
    /// request `--lang` > config `lang` (unless `auto`) > the stock
    /// model's baked language (discovered, never hardcoded).
    ///
    /// The `speak` command normally pre-resolves the language (incl.
    /// auto-detection) and always sets `req.lang`, so this is the
    /// fallback path for direct callers.  A config `lang: auto` (the
    /// new default) is treated as "unset" here — auto-detection is a
    /// higher-level concern the backend does not perform — so it falls
    /// through to the baked language rather than passing the literal
    /// string `auto` to espeak-ng.
    fn resolve_lang(&self, req: &SynthesisRequest) -> Result<String, TalkError> {
        if let Some(l) = &req.lang {
            let l = l.trim();
            if !l.is_empty() && !l.eq_ignore_ascii_case("auto") {
                return Ok(l.to_string());
            }
        }
        if let Some(l) = &self.config_lang {
            let l = l.trim();
            if !l.is_empty() && !l.eq_ignore_ascii_case("auto") {
                return Ok(l.to_string());
            }
        }
        // Fall back to whatever the stock model was built for.
        model::baked_language(&self.model_dir)
    }

    /// Resolve the effective speaker id for a request + language.
    ///
    /// request `--voice` > config `voice` > per-language default
    /// (prefix-derived, agnostic).
    fn resolve_sid(&self, req: &SynthesisRequest, lang: &str) -> Result<i32, TalkError> {
        if let Some(v) = &req.voice {
            return voices::name_to_sid(v);
        }
        if let Some(v) = &self.config_voice {
            return voices::name_to_sid(v);
        }
        Ok(voices::default_sid_for_lang(lang))
    }
}

#[async_trait]
impl OneShotSynthesizer for KokoroOneShotSynthesizer {
    /// Pre-flight: check the model is present on disk — WITHOUT
    /// downloading.  Consent-safe: no network I/O.  The download is an
    /// explicit, consented action driven by the `speak` action.
    async fn validate(&self) -> Result<(), TalkError> {
        model::ensure_present(&self.model_dir)
    }

    async fn synthesize(&self, req: SynthesisRequest) -> Result<SynthesisResult, TalkError> {
        let lang = self.resolve_lang(&req)?;
        let sid = self.resolve_sid(&req, &lang)?;
        let speed = req.speed.unwrap_or(1.0);
        let text = req.text.clone();

        // Resolve (and lazily derive) the per-language model file.
        let model_path = model::ensure_lang_model(&self.model_dir, &lang)?;
        let is_stock = model_path == self.model_dir.join(model::MODEL_EN);

        let model_dir = self.model_dir.clone();
        let num_threads = self.num_threads as i32;

        // sherpa-onnx synthesis is synchronous C work — offload it.
        let result = tokio::task::spawn_blocking(move || -> Result<SynthesisResult, TalkError> {
            run_synthesis(
                &model_dir,
                &model_path,
                is_stock,
                num_threads,
                sid,
                speed,
                &text,
            )
        })
        .await
        .map_err(|e| {
            TalkError::Transcription(format!("kokoro synthesis task panicked: {}", e))
        })??;

        Ok(result)
    }
}

/// Build a TTS engine for `model_path` and synthesize `text`.
///
/// Pure / synchronous — meant for [`tokio::task::spawn_blocking`].
///
/// `is_stock` selects the lexicon set: the stock (baked-language) model
/// uses the shipped language-specific lexicons; a derived-language
/// model uses the empty lexicon and relies purely on espeak-ng, exactly
/// as validated in the POC.
fn run_synthesis(
    model_dir: &Path,
    model_path: &Path,
    is_stock: bool,
    num_threads: i32,
    sid: i32,
    speed: f32,
    text: &str,
) -> Result<SynthesisResult, TalkError> {
    let dir = path_to_string(model_dir)?;
    let model = path_to_string(model_path)?;

    // Lexicon selection:
    // - stock (baked-language) model → the shipped lexicons.
    // - derived-language model → the empty lexicon (espeak-ng only).
    let lexicon = if is_stock {
        format!("{dir}/lexicon-us-en.txt,{dir}/lexicon-zh.txt")
    } else {
        format!("{}/{}", dir, model::EMPTY_LEXICON)
    };

    let config = OfflineTtsConfig {
        model: sherpa_onnx::OfflineTtsModelConfig {
            kokoro: OfflineTtsKokoroModelConfig {
                model: Some(model),
                voices: Some(format!("{dir}/voices.bin")),
                tokens: Some(format!("{dir}/tokens.txt")),
                data_dir: Some(format!("{dir}/espeak-ng-data")),
                dict_dir: Some(format!("{dir}/dict")),
                lexicon: Some(lexicon),
                length_scale: 1.0,
                ..Default::default()
            },
            num_threads,
            debug: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let tts = OfflineTts::create(&config).ok_or_else(|| {
        TalkError::Transcription(format!(
            "kokoro: failed to create OfflineTts (model_dir={})",
            dir
        ))
    })?;

    let gen_config = GenerationConfig {
        sid,
        speed,
        ..Default::default()
    };

    let audio = tts
        .generate_with_config(text, &gen_config, None::<fn(&[f32], f32) -> bool>)
        .ok_or_else(|| TalkError::Transcription("kokoro: generation failed".to_string()))?;

    let sample_rate = audio.sample_rate() as u32;
    let pcm = f32_to_i16(audio.samples());

    if pcm.is_empty() {
        return Err(TalkError::Transcription(
            "kokoro: synthesis produced no audio (empty text or model issue)".to_string(),
        ));
    }

    Ok(SynthesisResult { pcm, sample_rate })
}

/// Convert normalised f32 samples in `[-1.0, 1.0]` to `i16` PCM.
///
/// Clamps out-of-range values and rounds to nearest — the inverse of
/// the parakeet `pcm_i16_to_f32_normalised` conversion.
fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| {
            let clamped = s.clamp(-1.0, 1.0);
            // 32767 keeps +1.0 in range; symmetric rounding.
            (clamped * 32767.0).round() as i16
        })
        .collect()
}

/// Convert a [`Path`] to an owned `String` for the sherpa-onnx config.
fn path_to_string(p: &Path) -> Result<String, TalkError> {
    p.to_str().map(|s| s.to_string()).ok_or_else(|| {
        TalkError::Config(format!(
            "kokoro: model path is not valid UTF-8: {}",
            p.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_to_i16_boundary_values() {
        let f = vec![-1.0f32, 0.0, 1.0, -2.0, 2.0];
        let out = f32_to_i16(&f);
        assert_eq!(out[0], -32767);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 32767);
        // Out-of-range values clamp.
        assert_eq!(out[3], -32767);
        assert_eq!(out[4], 32767);
    }

    #[test]
    fn f32_to_i16_empty() {
        assert!(f32_to_i16(&[]).is_empty());
    }

    fn synthesizer_with_dir(dir: &Path) -> KokoroOneShotSynthesizer {
        KokoroOneShotSynthesizer {
            model_dir: dir.to_path_buf(),
            config_voice: None,
            config_lang: None,
            num_threads: 2,
        }
    }

    #[test]
    fn resolve_sid_prefers_request_voice() {
        let s = synthesizer_with_dir(Path::new("/tmp/x"));
        let req = SynthesisRequest {
            text: "hi".into(),
            voice: Some("am_michael".into()),
            speed: None,
            lang: None,
        };
        assert_eq!(s.resolve_sid(&req, "en").unwrap(), 16);
    }

    #[test]
    fn resolve_sid_falls_back_to_lang_default() {
        let s = synthesizer_with_dir(Path::new("/tmp/x"));
        let req = SynthesisRequest {
            text: "salut".into(),
            voice: None,
            speed: None,
            lang: Some("fr".into()),
        };
        // fr → ff_siwis (sid 30), prefix-derived.
        assert_eq!(s.resolve_sid(&req, "fr").unwrap(), 30);
    }

    #[test]
    fn resolve_sid_uses_config_voice_over_lang_default() {
        let mut s = synthesizer_with_dir(Path::new("/tmp/x"));
        s.config_voice = Some("af_bella".into());
        let req = SynthesisRequest {
            text: "hi".into(),
            voice: None,
            speed: None,
            lang: None,
        };
        assert_eq!(s.resolve_sid(&req, "en").unwrap(), 2);
    }

    #[test]
    fn resolve_lang_prefers_request_then_config() {
        let mut s = synthesizer_with_dir(Path::new("/tmp/x"));
        s.config_lang = Some("de".into());
        // Request wins.
        let req = SynthesisRequest {
            text: "hi".into(),
            voice: None,
            speed: None,
            lang: Some("it".into()),
        };
        assert_eq!(s.resolve_lang(&req).unwrap(), "it");
        // No request → config.
        let req2 = SynthesisRequest {
            text: "hi".into(),
            voice: None,
            speed: None,
            lang: None,
        };
        assert_eq!(s.resolve_lang(&req2).unwrap(), "de");
    }
}
