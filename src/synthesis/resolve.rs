//! Shared language / voice resolution + the voice↔language mismatch
//! guard for the `speak` command.
//!
//! This is the *single* place that owns the cross-provider policy so
//! neither the Mistral nor the Kokoro backend duplicates it:
//!
//! * [`LangSource`] records HOW the effective language was decided
//!   (CLI `--lang`, provider config, or auto-detection) so the guard's
//!   error can tell the user which remedy applies.
//! * [`VoiceMeta`] is the provider-agnostic "what language does this
//!   voice speak" fact the guard compares against.  A provider returns
//!   `None` for a voice whose language it cannot know (e.g. a Mistral
//!   *custom* UUID not in the embedded preset table) — such voices skip
//!   the guard silently.
//! * [`primary_subtag`] normalises `en_us` / `en-GB` / `fr_FR` to their
//!   primary subtag so the comparison is `en`-vs-`en`, not
//!   `en_us`-vs-`en_gb`.
//! * [`guard_voice_lang`] is the guard itself: it fires ONLY for an
//!   explicitly-chosen voice whose known language conflicts with the
//!   resolved language, and is bypassed by `--force`.

use crate::error::TalkError;

/// How the effective synthesis language was resolved.
///
/// Carried into [`guard_voice_lang`] so the mismatch error names the
/// right remedy: an auto-detected wrong guess is fixed with `--lang`,
/// whereas an explicit `--lang` / config value that conflicts with a
/// pinned voice is fixed by changing the voice (or `--force`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LangSource {
    /// From the CLI `--lang` flag.
    Cli,
    /// From a concrete provider config `lang` (e.g. `kokoro.lang: fr`).
    Config,
    /// Auto-detected from the input text (`lang: auto` / unset).
    Detected,
}

impl LangSource {
    /// Human-readable phrase describing the source, embedded verbatim
    /// in the mismatch error ("resolved lang 'fr' (…)").
    pub fn describe(self) -> &'static str {
        match self {
            LangSource::Cli => "from --lang",
            LangSource::Config => "from config",
            LangSource::Detected => "auto-detected from text",
        }
    }
}

/// A provider-agnostic description of a *known* voice: its display name
/// and the language it speaks (as a raw code — `en_us`, `fr`, …).
///
/// Providers build this from their embedded voice metadata: the Mistral
/// preset UUID table and the Kokoro speaker-prefix table.  A voice the
/// provider cannot map to a language yields `None` from the provider's
/// lookup (not a `VoiceMeta` with an empty language), so the guard can
/// distinguish "unknown voice ⇒ skip" from "known voice ⇒ check".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceMeta {
    /// Human-facing voice name (e.g. `"Marie - Neutral"`, `"ff_siwis"`).
    pub name: String,
    /// The voice's language code, possibly region-qualified
    /// (`"en_us"`, `"fr"`).  Compared by primary subtag only.
    pub lang: String,
}

/// The primary language subtag of a possibly region-qualified code.
///
/// Splits on the first `-` or `_` and lowercases: `en_us` → `en`,
/// `fr-FR` → `fr`, `en` → `en`.  Used so `en_us` and `en_gb` both
/// match a resolved `en`.
pub fn primary_subtag(code: &str) -> String {
    code.split(['-', '_'])
        .next()
        .unwrap_or(code)
        .to_ascii_lowercase()
}

/// The voice↔language mismatch guard.
///
/// Fires — returning a [`TalkError::Config`] — when ALL of:
///
/// * `explicit_voice` is `Some` (the voice was pinned by the user via
///   `--voice` / config, NOT auto-selected from a per-language map or
///   default), AND
/// * the provider knows that voice's language (`VoiceMeta`), AND
/// * that language's primary subtag differs from `resolved_lang`'s, AND
/// * `force` is `false`.
///
/// The error names the voice + its language, the resolved language +
/// how it was resolved, and the three remedies (`--lang`, a matching
/// `--voice`, `--force`).  Auto-selected voices pass `None` and thus
/// never trip the guard (they cannot mismatch by construction).
pub fn guard_voice_lang(
    explicit_voice: Option<&VoiceMeta>,
    resolved_lang: &str,
    lang_source: LangSource,
    force: bool,
) -> Result<(), TalkError> {
    if force {
        return Ok(());
    }
    let Some(voice) = explicit_voice else {
        return Ok(());
    };
    let voice_primary = primary_subtag(&voice.lang);
    let resolved_primary = primary_subtag(resolved_lang);
    if voice_primary == resolved_primary {
        return Ok(());
    }

    // Tailor the first remedy: `--lang` only helps when the language
    // was auto-detected (the user can pin the language the detector got
    // wrong); when the language came from --lang/config the conflict is
    // with the *voice*, so lead with changing the voice.
    let lang_remedy = match lang_source {
        LangSource::Detected => format!(
            "pass --lang {} (or the correct language) if auto-detection guessed wrong",
            voice_primary
        ),
        LangSource::Cli | LangSource::Config => format!(
            "change the resolved language (currently {}) to {}",
            resolved_primary, voice_primary
        ),
    };

    Err(TalkError::Config(format!(
        "voice/language mismatch: the voice '{name}' speaks {voice_lang} \
         but the resolved language is '{resolved}' ({how}).\n\
         Remedies:\n  \
         - {lang_remedy};\n  \
         - pass a --voice whose language is '{resolved}';\n  \
         - pass --force to synthesize anyway with '{name}'.",
        name = voice.name,
        voice_lang = voice.lang,
        resolved = resolved_primary,
        how = lang_source.describe(),
        lang_remedy = lang_remedy,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(name: &str, lang: &str) -> VoiceMeta {
        VoiceMeta {
            name: name.to_string(),
            lang: lang.to_string(),
        }
    }

    #[test]
    fn primary_subtag_strips_region_and_lowercases() {
        assert_eq!(primary_subtag("en_us"), "en");
        assert_eq!(primary_subtag("en-GB"), "en");
        assert_eq!(primary_subtag("fr_FR"), "fr");
        assert_eq!(primary_subtag("fr"), "fr");
        assert_eq!(primary_subtag("ZH"), "zh");
    }

    #[test]
    fn guard_passes_when_no_explicit_voice() {
        // Auto-selected voice: explicit_voice is None → never guards.
        guard_voice_lang(None, "fr", LangSource::Detected, false).expect("no voice, no guard");
    }

    #[test]
    fn guard_passes_when_languages_match_by_primary_subtag() {
        let v = meta("Paul - Neutral", "en_us");
        guard_voice_lang(Some(&v), "en", LangSource::Detected, false).expect("en_us matches en");
    }

    #[test]
    fn guard_passes_when_forced() {
        let v = meta("Paul - Neutral", "en_us");
        guard_voice_lang(Some(&v), "fr", LangSource::Detected, true).expect("force bypasses");
    }

    #[test]
    fn guard_errors_on_mismatch_naming_everything() {
        let v = meta("Paul - Neutral", "en_us");
        let err = guard_voice_lang(Some(&v), "fr", LangSource::Detected, false)
            .expect_err("must error on mismatch");
        let msg = err.to_string();
        // Voice name.
        assert!(msg.contains("Paul - Neutral"), "msg: {msg}");
        // Voice's language.
        assert!(msg.contains("en_us"), "msg: {msg}");
        // Resolved language + how it was resolved.
        assert!(msg.contains("'fr'"), "msg: {msg}");
        assert!(msg.contains("auto-detected from text"), "msg: {msg}");
        // All three remedies.
        assert!(msg.contains("--lang"), "msg: {msg}");
        assert!(msg.contains("--voice"), "msg: {msg}");
        assert!(msg.contains("--force"), "msg: {msg}");
    }

    #[test]
    fn guard_error_remedy_reflects_lang_source_cli() {
        let v = meta("ff_siwis", "fr");
        let err = guard_voice_lang(Some(&v), "en", LangSource::Cli, false)
            .expect_err("must error on mismatch");
        let msg = err.to_string();
        assert!(msg.contains("from --lang"), "msg: {msg}");
        // CLI/config source leads with changing the resolved language.
        assert!(msg.contains("change the resolved language"), "msg: {msg}");
    }

    #[test]
    fn guard_error_remedy_reflects_lang_source_config() {
        let v = meta("ff_siwis", "fr");
        let err = guard_voice_lang(Some(&v), "en", LangSource::Config, false)
            .expect_err("must error on mismatch");
        assert!(err.to_string().contains("from config"));
    }
}
