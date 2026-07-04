//! `speak` command implementation.
//!
//! Synthesizes speech from text using the configured synthesis backend
//! and either plays it through the default output device or saves it as
//! a WAV file.  The synthesis-side mirror of [`crate::transcribe`].
//!
//! Text resolution order: positional argument > `--file` > stdin (when
//! stdin is not a TTY).  Provider resolution: `--provider` > config
//! `speak.default_provider` > Kokoro-if-configured-else-Mistral.

use std::io::{IsTerminal, Read};
use std::path::PathBuf;

use crate::config::{Config, MistralConfig, SynthesisProvider};
use crate::error::TalkError;
use crate::synthesis::resolve::{guard_voice_lang, LangSource, VoiceMeta};
use crate::synthesis::{self, detect_lang, SynthesisRequest};

/// Options for the `speak` command (parsed from the CLI).
#[derive(Debug, Default, Clone)]
pub struct SpeakOpts {
    /// Text to speak (positional argument).
    pub text: Option<String>,
    /// Read text from this file instead of the positional argument.
    pub file: Option<PathBuf>,
    /// Synthesis provider override.
    pub provider: Option<SynthesisProvider>,
    /// Voice name / id override.
    pub voice: Option<String>,
    /// Phonemization language override (`--lang`).  Highest-priority
    /// input to the language-resolution chain (see [`resolve_lang`]).
    pub lang: Option<String>,
    /// Speech rate multiplier (Kokoro).
    pub speed: Option<f32>,
    /// Save to this WAV file instead of playing.
    pub output: Option<PathBuf>,
    /// Bypass the voice↔language mismatch guard (`--force`).  Bypasses
    /// ONLY that guard — every other error still fires.
    pub force: bool,
}

/// The concrete language + voice a request resolved to, plus the
/// provenance needed by the mismatch guard.
///
/// Produced by [`resolve_request`] as the single cross-provider
/// resolution step, then folded into the [`SynthesisRequest`] handed to
/// a backend (whose own `resolve_*` methods then see already-concrete
/// values).
#[derive(Debug)]
struct ResolvedRequest {
    /// The concrete language code the backend should phonemize / voice.
    lang: String,
    /// The concrete voice id / name, or `None` to let the backend pick
    /// its own (only Kokoro does — Mistral always resolves a concrete
    /// voice or errors).
    voice: Option<String>,
}

/// Resolve the text to speak from (in priority order) the positional
/// argument, a `--file`, or stdin.
///
/// Errors when no source yields non-empty text.  Stdin is only read
/// when it is NOT a TTY (so an interactive `talk-rs speak` with no
/// argument fails fast with a helpful error rather than hanging).
fn resolve_text(text: &Option<String>, file: &Option<PathBuf>) -> Result<String, TalkError> {
    if let Some(t) = text {
        let t = t.trim();
        if !t.is_empty() {
            return Ok(t.to_string());
        }
        return Err(TalkError::Config(
            "speak: the provided text is empty".to_string(),
        ));
    }

    if let Some(path) = file {
        let content = std::fs::read_to_string(path).map_err(|e| {
            TalkError::Config(format!(
                "speak: failed to read --file {}: {}",
                path.display(),
                e
            ))
        })?;
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err(TalkError::Config(format!(
                "speak: --file {} is empty",
                path.display()
            )));
        }
        return Ok(trimmed.to_string());
    }

    // Fall back to stdin, but only if it is piped (not a TTY).
    if !std::io::stdin().is_terminal() {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(TalkError::Io)?;
        let trimmed = buf.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    Err(TalkError::Config(
        "speak: no text provided. Pass text as an argument, use --file <PATH>, \
         or pipe text via stdin."
            .to_string(),
    ))
}

/// Resolve the effective synthesis provider.
///
/// `--provider` > config `speak.default_provider` >
/// Kokoro-if-configured-else-Mistral.
fn resolve_provider(config: &Config, cli_provider: Option<SynthesisProvider>) -> SynthesisProvider {
    if let Some(p) = cli_provider {
        return p;
    }
    if let Some(p) = config.speak.as_ref().and_then(|s| s.default_provider) {
        return p;
    }
    // No explicit default: prefer the local Kokoro backend when it is
    // configured, otherwise fall back to the remote Mistral provider.
    if config.providers.kokoro.is_some() {
        SynthesisProvider::Kokoro
    } else {
        SynthesisProvider::Mistral
    }
}

/// The concrete provider config-language, if it names a real language
/// (i.e. NOT `auto` and not empty).  Returns `None` for `auto` / unset
/// so the caller falls through to auto-detection.
fn concrete_config_lang(raw: Option<&str>) -> Option<String> {
    let l = raw?.trim();
    if l.is_empty() || l.eq_ignore_ascii_case("auto") {
        None
    } else {
        Some(l.to_ascii_lowercase())
    }
}

/// The provider's config `lang` field (Kokoro has one; Mistral does
/// not), as a concrete language or `None`.
fn provider_config_lang(config: &Config, provider: SynthesisProvider) -> Option<String> {
    match provider {
        SynthesisProvider::Kokoro => concrete_config_lang(
            config
                .providers
                .kokoro
                .as_ref()
                .and_then(|k| k.lang.as_deref()),
        ),
        // Mistral has no phonemization-language config knob; a Mistral
        // voice implies its language server-side.  Language resolution
        // for Mistral therefore relies on --lang or auto-detection.
        SynthesisProvider::Mistral => None,
    }
}

/// The shared language-resolution chain (single place, both providers):
///
/// `--lang` (CLI) > concrete provider config `lang` > auto-detect from
/// the text (`lang: auto` or unset).  Returns the resolved code plus
/// [`LangSource`] provenance for the guard's error message.  Auto
/// detection always yields *something* for non-empty text; an empty
/// classification (empty text — impossible here, the CLI rejects it)
/// falls back to English.
fn resolve_lang(
    cli_lang: Option<&str>,
    config: &Config,
    provider: SynthesisProvider,
    text: &str,
) -> (String, LangSource) {
    if let Some(l) = cli_lang {
        let l = l.trim();
        if !l.is_empty() {
            return (l.to_ascii_lowercase(), LangSource::Cli);
        }
    }
    if let Some(l) = provider_config_lang(config, provider) {
        return (l, LangSource::Config);
    }
    match detect_lang(text) {
        Some(l) => (l, LangSource::Detected),
        None => ("en".to_string(), LangSource::Detected),
    }
}

/// Resolve the Mistral voice + its guard metadata for a resolved
/// language.
///
/// Chain: `--voice` > `tts_voice` (pin) > `tts_voices[lang]` > built-in
/// default table > error.  Returns `(voice_id, explicit_meta)` where
/// `explicit_meta` is `Some` ONLY for a user-pinned voice whose
/// language is known (so the guard can check it); auto-selected voices
/// (map / default) return `None` and never trip the guard.
fn resolve_mistral_voice(
    cli_voice: Option<&str>,
    cfg: &MistralConfig,
    lang: &str,
) -> Result<(String, Option<VoiceMeta>), TalkError> {
    use synthesis::mistral_presets;

    // Explicit pins (CLI or config) — subject to the guard.
    if let Some(v) = cli_voice {
        let v = v.trim().to_string();
        let meta = mistral_presets::voice_language(&v);
        return Ok((v, meta));
    }
    if let Some(v) = &cfg.tts_voice {
        let v = v.trim().to_string();
        let meta = mistral_presets::voice_language(&v);
        return Ok((v, meta));
    }

    // Auto-selected — matches the language by construction, no guard.
    if let Some(map) = &cfg.tts_voices {
        let primary = synthesis::primary_subtag(lang);
        if let Some(v) = map
            .get(&primary)
            .or_else(|| map.get(lang))
            .map(|s| s.trim().to_string())
        {
            return Ok((v, None));
        }
    }
    if let Some(v) = mistral_presets::default_voice_for_lang(lang) {
        return Ok((v.to_string(), None));
    }

    let supported: Vec<&str> = mistral_presets::supported_langs().collect();
    Err(TalkError::Config(format!(
        "no Mistral voice for language '{}': pass --voice <id>, set \
         providers.mistral.tts_voice, or add providers.mistral.tts_voices.{} . \
         Built-in defaults exist for: {}. List all voices with \
         `GET /v1/audio/voices?voice_type=preset`.",
        lang,
        synthesis::primary_subtag(lang),
        supported.join(", ")
    )))
}

/// The subset of language primary-subtags Mistral's built-in default
/// table can voice (used to decide whether a *detected* language is
/// speakable before falling back to English).
fn mistral_can_voice(lang: &str) -> bool {
    let primary = synthesis::primary_subtag(lang);
    synthesis::mistral_presets::supported_langs().any(|l| l == primary)
}

/// The shared cross-provider request resolution: resolve the language,
/// apply the "detected language the provider can't voice → fall back to
/// en" rule, resolve the voice, then run the mismatch guard.
///
/// This is the *single* place the guard runs, so both providers get
/// identical behaviour.  It returns a [`ResolvedRequest`] with concrete
/// `lang` + `voice` for the backend.
fn resolve_request(
    config: &Config,
    provider: SynthesisProvider,
    opts: &SpeakOpts,
    text: &str,
) -> Result<ResolvedRequest, TalkError> {
    let (mut lang, lang_source) = resolve_lang(opts.lang.as_deref(), config, provider, text);

    let cli_voice = opts
        .voice
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty());

    let (voice, explicit_meta): (Option<String>, Option<VoiceMeta>) = match provider {
        SynthesisProvider::Mistral => {
            // Fall back to English when auto-detection picked a language
            // Mistral cannot voice (its preset defaults cover en/fr).
            // A user-pinned voice or explicit --lang is respected as-is.
            if lang_source == LangSource::Detected
                && cli_voice.is_none()
                && !mistral_can_voice(&lang)
            {
                log::info!(
                    "speak: detected language '{}' has no Mistral voice; falling back to 'en'",
                    lang
                );
                lang = "en".to_string();
            }
            let cfg = config.providers.mistral.as_ref().ok_or_else(|| {
                TalkError::Config(
                    "Mistral synthesis selected but providers.mistral is not configured"
                        .to_string(),
                )
            })?;
            let (v, meta) = resolve_mistral_voice(cli_voice, cfg, &lang)?;
            (Some(v), meta)
        }
        SynthesisProvider::Kokoro => {
            resolve_kokoro_voice(cli_voice, config, &mut lang, lang_source)?
        }
    };

    guard_voice_lang(explicit_meta.as_ref(), &lang, lang_source, opts.force)?;

    Ok(ResolvedRequest { lang, voice })
}

/// Kokoro voice + guard-metadata resolution.  Split out (and cfg-gated
/// helper below) so the non-kokoro build still links.
#[cfg(feature = "kokoro")]
fn resolve_kokoro_voice(
    cli_voice: Option<&str>,
    config: &Config,
    lang: &mut String,
    lang_source: LangSource,
) -> Result<(Option<String>, Option<VoiceMeta>), TalkError> {
    use crate::synthesis::kokoro::voices;

    let kcfg = config.providers.kokoro.clone().unwrap_or_default();

    // Explicit pins (CLI --voice or config voice) — subject to the guard.
    if let Some(v) = cli_voice {
        let meta = voices::voice_language(v);
        return Ok((Some(v.to_string()), meta));
    }
    if let Some(v) = &kcfg.voice {
        let meta = voices::voice_language(v);
        return Ok((Some(v.clone()), meta));
    }

    // Auto-selected per-language default.  When auto-detection picked a
    // language Kokoro has no speaker for, fall back to English (Kokoro
    // can still derive a model, but with no per-prefix voice there is no
    // sensible default speaker).
    if lang_source == LangSource::Detected && voices::default_voice_for_lang(lang).is_none() {
        log::info!(
            "speak: detected language '{}' has no Kokoro speaker; falling back to 'en'",
            lang
        );
        *lang = "en".to_string();
    }
    // Voice left as None: the backend's own per-language default (sid)
    // kicks in.  Auto-selected ⇒ no guard.
    Ok((None, None))
}

/// Kokoro resolution stub for the non-kokoro build: the factory rejects
/// the provider before we get here, so this only needs to compile.
#[cfg(not(feature = "kokoro"))]
fn resolve_kokoro_voice(
    _cli_voice: Option<&str>,
    _config: &Config,
    _lang: &mut String,
    _lang_source: LangSource,
) -> Result<(Option<String>, Option<VoiceMeta>), TalkError> {
    Ok((None, None))
}

/// Run the `speak` command.
pub async fn speak(opts: SpeakOpts) -> Result<(), TalkError> {
    let text = resolve_text(&opts.text, &opts.file)?;
    let config = Config::load(None)?;
    let provider = resolve_provider(&config, opts.provider);

    // Kokoro is a local backend whose model must be downloaded once.
    // The synthesis pipeline never downloads silently, so obtain
    // consent here (TTY prompt, or stderr-log + proceed when piped)
    // before synthesis reaches `validate`.  No-op once installed.
    #[cfg(feature = "kokoro")]
    if provider == SynthesisProvider::Kokoro {
        crate::synthesis::kokoro::consent::ensure_with_cli_consent(&config).await?;
    }

    // Shared cross-provider resolution: language chain + voice chain +
    // mismatch guard (the single place the guard runs).
    let resolved = resolve_request(&config, provider, &opts, &text)?;

    let req = SynthesisRequest {
        text,
        voice: resolved.voice,
        speed: opts.speed,
        lang: Some(resolved.lang),
    };

    let result = synthesis::synthesize(&config, provider, req).await?;

    log::info!(
        "speak: synthesized {:.2}s of audio ({} samples @ {} Hz) via {}",
        result.duration_secs(),
        result.pcm.len(),
        result.sample_rate,
        provider
    );

    match opts.output {
        Some(path) => {
            save_wav(&path, &result.pcm, result.sample_rate)?;
            println!("Saved synthesized audio to: {}", path.display());
            Ok(())
        }
        None => play(&result.pcm, result.sample_rate),
    }
}

/// Save mono `i16` PCM to a WAV file using the shared [`WavWriter`].
fn save_wav(path: &std::path::Path, pcm: &[i16], sample_rate: u32) -> Result<(), TalkError> {
    use crate::audio::{AudioWriter, WavWriter};
    use crate::config::AudioConfig;
    use std::io::Write;

    let cfg = AudioConfig {
        sample_rate,
        channels: 1,
        bitrate: 0, // unused for WAV
    };
    let mut writer = WavWriter::new(cfg);
    let mut out = writer.header()?;
    out.extend_from_slice(&writer.write_pcm(pcm)?);
    // WavWriter::finalize returns a corrected 44-byte header that must
    // overwrite the placeholder header at offset 0.
    let final_header = writer.finalize()?;
    out[..final_header.len()].copy_from_slice(&final_header);

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(TalkError::Io)?;
        }
    }
    let mut file = std::fs::File::create(path).map_err(TalkError::Io)?;
    file.write_all(&out).map_err(TalkError::Io)?;
    file.sync_all().map_err(TalkError::Io)?;
    Ok(())
}

/// Play mono `i16` PCM through the default output device, blocking
/// until playback finishes.
#[cfg(feature = "playback")]
fn play(pcm: &[i16], sample_rate: u32) -> Result<(), TalkError> {
    let player = crate::audio::AudioPlayer::new()?;
    player.play_pcm_blocking(pcm, sample_rate)
}

/// Playback fallback for builds without the `playback` feature: there
/// is no output device support, so instruct the user to save to a file.
#[cfg(not(feature = "playback"))]
fn play(_pcm: &[i16], _sample_rate: u32) -> Result<(), TalkError> {
    Err(TalkError::Config(
        "talk-rs was built without the 'playback' feature; audio playback is \
         unavailable. Rebuild with --features playback, or use -o <PATH> to \
         save a WAV file instead."
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{KokoroConfig, ProvidersConfig};
    use std::collections::HashMap;

    /// Minimal Mistral config for resolution tests.
    fn mistral_cfg(
        tts_voice: Option<&str>,
        tts_voices: Option<HashMap<String, String>>,
    ) -> MistralConfig {
        MistralConfig {
            api_key: "k".to_string(),
            url: None,
            model: "m".to_string(),
            context_bias: None,
            tts_model: "voxtral-mini-tts-latest".to_string(),
            tts_voice: tts_voice.map(str::to_string),
            tts_voices,
        }
    }

    /// A `Config` carrying only a Mistral provider.
    fn config_mistral(m: MistralConfig) -> Config {
        Config {
            output_dir: std::path::PathBuf::from("/tmp"),
            providers: ProvidersConfig {
                mistral: Some(m),
                openai: None,
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        }
    }

    /// A `Config` carrying only a Kokoro provider (with a given `lang`).
    fn config_kokoro(lang: Option<&str>, voice: Option<&str>) -> Config {
        Config {
            output_dir: std::path::PathBuf::from("/tmp"),
            providers: ProvidersConfig {
                mistral: None,
                openai: None,
                parakeet: None,
                kokoro: Some(KokoroConfig {
                    variant: None,
                    model_dir: None,
                    voice: voice.map(str::to_string),
                    num_threads: None,
                    lang: lang.map(str::to_string),
                }),
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        }
    }

    // ── Language-resolution precedence ──────────────────────────────

    #[test]
    fn lang_cli_beats_auto_detection() {
        let cfg = config_kokoro(None, None);
        // Text is clearly French, but --lang en must win.
        let (lang, src) = resolve_lang(
            Some("en"),
            &cfg,
            SynthesisProvider::Kokoro,
            "Bonjour tout le monde comment allez vous",
        );
        assert_eq!(lang, "en");
        assert_eq!(src, LangSource::Cli);
    }

    #[test]
    fn lang_concrete_config_beats_auto_detection() {
        // Kokoro config lang: fr; English text — config must win.
        let cfg = config_kokoro(Some("fr"), None);
        let (lang, src) = resolve_lang(
            None,
            &cfg,
            SynthesisProvider::Kokoro,
            "The quick brown fox jumps over the lazy dog.",
        );
        assert_eq!(lang, "fr");
        assert_eq!(src, LangSource::Config);
    }

    #[test]
    fn lang_auto_used_when_config_is_auto() {
        let cfg = config_kokoro(Some("auto"), None);
        let (lang, src) = resolve_lang(
            None,
            &cfg,
            SynthesisProvider::Kokoro,
            "Bonjour tout le monde, je suis ravi de vous voir",
        );
        assert_eq!(lang, "fr");
        assert_eq!(src, LangSource::Detected);
    }

    #[test]
    fn lang_auto_used_when_config_absent() {
        let cfg = config_kokoro(None, None);
        let (lang, src) = resolve_lang(
            None,
            &cfg,
            SynthesisProvider::Kokoro,
            "The quick brown fox jumps over the lazy dog.",
        );
        assert_eq!(lang, "en");
        assert_eq!(src, LangSource::Detected);
    }

    // ── Mistral voice resolution ────────────────────────────────────

    #[test]
    fn mistral_tts_voices_map_selects_by_lang() {
        let mut map = HashMap::new();
        map.insert("en".to_string(), "en-voice".to_string());
        map.insert("fr".to_string(), "fr-voice".to_string());
        let cfg = mistral_cfg(None, Some(map));
        let (voice, meta) = resolve_mistral_voice(None, &cfg, "fr").unwrap();
        assert_eq!(voice, "fr-voice");
        // Auto-selected ⇒ no guard metadata.
        assert!(meta.is_none());
    }

    #[test]
    fn mistral_builtin_default_when_map_absent() {
        let cfg = mistral_cfg(None, None);
        let (voice, meta) = resolve_mistral_voice(None, &cfg, "fr").unwrap();
        // Marie - Neutral (fr).
        assert_eq!(voice, "5a271406-039d-46fe-835b-fbbb00eaf08d");
        assert!(meta.is_none());
        let (voice_en, _) = resolve_mistral_voice(None, &cfg, "en").unwrap();
        // Paul - Neutral (en).
        assert_eq!(voice_en, "c69964a6-ab8b-4f8a-9465-ec0925096ec8");
    }

    #[test]
    fn mistral_pin_returns_guard_metadata_for_known_preset() {
        // Pinned Marie (fr).
        let cfg = mistral_cfg(Some("5a271406-039d-46fe-835b-fbbb00eaf08d"), None);
        let (voice, meta) = resolve_mistral_voice(None, &cfg, "en").unwrap();
        assert_eq!(voice, "5a271406-039d-46fe-835b-fbbb00eaf08d");
        let meta = meta.expect("known preset ⇒ guard metadata");
        assert_eq!(meta.name, "Marie - Neutral");
        assert_eq!(meta.lang, "fr_fr");
    }

    #[test]
    fn mistral_cli_voice_beats_pin_and_map() {
        let mut map = HashMap::new();
        map.insert("fr".to_string(), "fr-voice".to_string());
        let cfg = mistral_cfg(Some("pinned"), Some(map));
        let (voice, _) = resolve_mistral_voice(Some("cli-voice"), &cfg, "fr").unwrap();
        assert_eq!(voice, "cli-voice");
    }

    #[test]
    fn mistral_unknown_uuid_has_no_guard_metadata() {
        let cfg = mistral_cfg(Some("00000000-0000-0000-0000-000000000000"), None);
        let (_, meta) = resolve_mistral_voice(None, &cfg, "fr").unwrap();
        // Custom voice ⇒ unknowable language ⇒ no guard.
        assert!(meta.is_none());
    }

    #[test]
    fn mistral_no_voice_for_unsupported_lang_errors() {
        let cfg = mistral_cfg(None, None);
        let err = resolve_mistral_voice(None, &cfg, "de").expect_err("no de voice");
        assert!(err
            .to_string()
            .contains("no Mistral voice for language 'de'"));
    }

    // ── End-to-end resolve_request + guard ──────────────────────────

    #[test]
    fn resolve_request_mistral_auto_french_picks_marie() {
        let cfg = config_mistral(mistral_cfg(None, None));
        let opts = SpeakOpts::default();
        let r = resolve_request(
            &cfg,
            SynthesisProvider::Mistral,
            &opts,
            "Bonjour tout le monde, je suis ravi de vous voir",
        )
        .unwrap();
        assert_eq!(r.lang, "fr");
        assert_eq!(
            r.voice.as_deref(),
            Some("5a271406-039d-46fe-835b-fbbb00eaf08d")
        );
    }

    #[test]
    fn resolve_request_mistral_auto_english_picks_paul() {
        let cfg = config_mistral(mistral_cfg(None, None));
        let opts = SpeakOpts::default();
        let r = resolve_request(
            &cfg,
            SynthesisProvider::Mistral,
            &opts,
            "The quick brown fox jumps over the lazy dog.",
        )
        .unwrap();
        assert_eq!(r.lang, "en");
        assert_eq!(
            r.voice.as_deref(),
            Some("c69964a6-ab8b-4f8a-9465-ec0925096ec8")
        );
    }

    #[test]
    fn resolve_request_mistral_pin_mismatch_errors() {
        // Pin Paul (en) but text is French → mismatch guard fires.
        let cfg = config_mistral(mistral_cfg(
            Some("c69964a6-ab8b-4f8a-9465-ec0925096ec8"),
            None,
        ));
        let opts = SpeakOpts::default();
        let err = resolve_request(
            &cfg,
            SynthesisProvider::Mistral,
            &opts,
            "Bonjour tout le monde, je suis ravi de vous voir",
        )
        .expect_err("mismatch must error");
        let msg = err.to_string();
        assert!(msg.contains("Paul - Neutral"), "msg: {msg}");
        assert!(msg.contains("en_us"), "msg: {msg}");
        assert!(msg.contains("'fr'"), "msg: {msg}");
        assert!(msg.contains("--force"), "msg: {msg}");
    }

    #[test]
    fn resolve_request_mistral_pin_mismatch_force_proceeds() {
        let cfg = config_mistral(mistral_cfg(
            Some("c69964a6-ab8b-4f8a-9465-ec0925096ec8"),
            None,
        ));
        let opts = SpeakOpts {
            force: true,
            ..Default::default()
        };
        let r = resolve_request(
            &cfg,
            SynthesisProvider::Mistral,
            &opts,
            "Bonjour tout le monde, je suis ravi de vous voir",
        )
        .expect("--force bypasses the guard");
        // The pinned voice is honoured despite the mismatch.
        assert_eq!(
            r.voice.as_deref(),
            Some("c69964a6-ab8b-4f8a-9465-ec0925096ec8")
        );
    }

    #[test]
    fn resolve_request_mistral_custom_voice_skips_guard() {
        // A custom (non-preset) UUID pin: language unknowable → no guard
        // even though the text is French.
        let cfg = config_mistral(mistral_cfg(
            Some("00000000-0000-0000-0000-000000000000"),
            None,
        ));
        let opts = SpeakOpts::default();
        let r = resolve_request(
            &cfg,
            SynthesisProvider::Mistral,
            &opts,
            "Bonjour tout le monde, je suis ravi de vous voir",
        )
        .expect("custom voice skips guard");
        assert_eq!(
            r.voice.as_deref(),
            Some("00000000-0000-0000-0000-000000000000")
        );
    }

    #[cfg(feature = "kokoro")]
    #[test]
    fn resolve_request_kokoro_pin_mismatch_errors() {
        // Pin an English voice (af_heart) but --lang fr → guard fires.
        let cfg = config_kokoro(None, Some("af_heart"));
        let opts = SpeakOpts {
            lang: Some("fr".to_string()),
            ..Default::default()
        };
        let err = resolve_request(&cfg, SynthesisProvider::Kokoro, &opts, "anything")
            .expect_err("prefix mismatch must error");
        let msg = err.to_string();
        assert!(msg.contains("af_heart"), "msg: {msg}");
        assert!(msg.contains("--force"), "msg: {msg}");
    }

    #[cfg(feature = "kokoro")]
    #[test]
    fn resolve_request_kokoro_auto_french_selects_default_voice() {
        // No pin, French text → lang fr, voice left None (backend picks
        // ff_siwis via default_sid_for_lang), no guard.
        let cfg = config_kokoro(None, None);
        let opts = SpeakOpts::default();
        let r = resolve_request(
            &cfg,
            SynthesisProvider::Kokoro,
            &opts,
            "Bonjour tout le monde, je suis ravi de vous voir",
        )
        .unwrap();
        assert_eq!(r.lang, "fr");
        assert!(r.voice.is_none());
    }

    #[cfg(feature = "kokoro")]
    #[test]
    fn resolve_request_kokoro_pin_match_ok() {
        // Pin ff_siwis (fr) with --lang fr → no mismatch.
        let cfg = config_kokoro(None, Some("ff_siwis"));
        let opts = SpeakOpts {
            lang: Some("fr".to_string()),
            ..Default::default()
        };
        let r = resolve_request(&cfg, SynthesisProvider::Kokoro, &opts, "anything").unwrap();
        assert_eq!(r.lang, "fr");
        assert_eq!(r.voice.as_deref(), Some("ff_siwis"));
    }

    #[test]
    fn resolve_text_positional_wins() {
        let t = resolve_text(&Some("hello".to_string()), &None).unwrap();
        assert_eq!(t, "hello");
    }

    #[test]
    fn resolve_text_trims_positional() {
        let t = resolve_text(&Some("  hi \n".to_string()), &None).unwrap();
        assert_eq!(t, "hi");
    }

    #[test]
    fn resolve_text_empty_positional_errors() {
        let err = resolve_text(&Some("   ".to_string()), &None).expect_err("empty");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn resolve_text_from_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let p = tmp.path().join("t.txt");
        std::fs::write(&p, "from file\n").unwrap();
        let t = resolve_text(&None, &Some(p)).unwrap();
        assert_eq!(t, "from file");
    }

    #[test]
    fn resolve_text_empty_file_errors() {
        let tmp = tempfile::TempDir::new().unwrap();
        let p = tmp.path().join("t.txt");
        std::fs::write(&p, "   \n").unwrap();
        let err = resolve_text(&None, &Some(p)).expect_err("empty file");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn resolve_text_missing_file_errors() {
        let err = resolve_text(&None, &Some(PathBuf::from("/nonexistent/xyz.txt")))
            .expect_err("missing file");
        assert!(err.to_string().contains("failed to read"));
    }

    #[test]
    fn save_wav_roundtrips_via_parser() {
        // Save a small PCM buffer and re-parse it with the mistral WAV
        // parser to confirm a valid WAV was produced.
        let tmp = tempfile::TempDir::new().unwrap();
        let p = tmp.path().join("out.wav");
        let pcm = vec![0i16, 1000, -1000, 32767, -32768];
        save_wav(&p, &pcm, 24_000).unwrap();

        let bytes = std::fs::read(&p).unwrap();
        let (parsed, rate) =
            crate::synthesis::mistral::parse_wav_pcm_i16(&bytes).expect("valid WAV");
        assert_eq!(rate, 24_000);
        assert_eq!(parsed, pcm);
    }

    #[test]
    fn save_wav_creates_parent_dirs() {
        let tmp = tempfile::TempDir::new().unwrap();
        let p = tmp.path().join("nested/dir/out.wav");
        save_wav(&p, &[1i16, 2, 3], 24_000).unwrap();
        assert!(p.exists());
    }
}
