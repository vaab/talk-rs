//! Kokoro speaker table + voice-name → speaker-id (sid) mapping.
//!
//! The `kokoro-multi-lang-v1_0` model ships 53 speakers.  sherpa-onnx
//! selects a speaker by integer `sid` in the [`GenerationConfig`];
//! humans want to say `--voice af_heart`, so this module owns the
//! bidirectional mapping.
//!
//! Prefix legend (from the upstream model card):
//! `af/am` US · `bf/bm` GB · `ef/em` ES · `ff` FR · `hf/hm` HI ·
//! `if/im` IT · `jf/jm` JA · `pf/pm` PT-BR · `zf/zm` ZH.
//!
//! [`GenerationConfig`]: sherpa_onnx::GenerationConfig

use crate::error::TalkError;

/// The 53-speaker table for `kokoro-multi-lang-v1_0`, lifted verbatim
/// from the validated POC.
pub(crate) const SPEAKERS: &[(i32, &str)] = &[
    (0, "af_alloy"),
    (1, "af_aoede"),
    (2, "af_bella"),
    (3, "af_heart"),
    (4, "af_jessica"),
    (5, "af_kore"),
    (6, "af_nicole"),
    (7, "af_nova"),
    (8, "af_river"),
    (9, "af_sarah"),
    (10, "af_sky"),
    (11, "am_adam"),
    (12, "am_echo"),
    (13, "am_eric"),
    (14, "am_fenrir"),
    (15, "am_liam"),
    (16, "am_michael"),
    (17, "am_onyx"),
    (18, "am_puck"),
    (19, "am_santa"),
    (20, "bf_alice"),
    (21, "bf_emma"),
    (22, "bf_isabella"),
    (23, "bf_lily"),
    (24, "bm_daniel"),
    (25, "bm_fable"),
    (26, "bm_george"),
    (27, "bm_lewis"),
    (28, "ef_dora"),
    (29, "em_alex"),
    (30, "ff_siwis"),
    (31, "hf_alpha"),
    (32, "hf_beta"),
    (33, "hm_omega"),
    (34, "hm_psi"),
    (35, "if_sara"),
    (36, "im_nicola"),
    (37, "jf_alpha"),
    (38, "jf_gongitsune"),
    (39, "jf_nezumi"),
    (40, "jf_tebukuro"),
    (41, "jm_kumo"),
    (42, "pf_dora"),
    (43, "pm_alex"),
    (44, "pm_santa"),
    (45, "zf_xiaobei"),
    (46, "zf_xiaoni"),
    (47, "zf_xiaoxiao"),
    (48, "zf_xiaoyi"),
    (49, "zm_yunjian"),
    (50, "zm_yunxi"),
    (51, "zm_yunxia"),
    (52, "zm_yunyang"),
];

/// Map a language *primary subtag* to the Kokoro voice-name prefix(es)
/// that speak it, derived from the upstream naming convention.
///
/// This is data, not policy: it simply records which two-letter voice
/// prefixes the model uses for each language so the default-voice
/// lookup can stay language-agnostic (no `if lang == "fr"` branches in
/// the resolution logic).  A language absent from this table has no
/// per-language default and falls back to the first speaker overall.
const LANG_VOICE_PREFIXES: &[(&str, &[&str])] = &[
    ("en", &["af", "am", "bf", "bm"]), // US + GB English
    ("es", &["ef", "em"]),             // Spanish
    ("fr", &["ff"]),                   // French
    ("hi", &["hf", "hm"]),             // Hindi
    ("it", &["if", "im"]),             // Italian
    ("ja", &["jf", "jm"]),             // Japanese
    ("pt", &["pf", "pm"]),             // Portuguese (BR)
    ("zh", &["zf", "zm"]),             // Chinese
];

/// Resolve a voice *name* to its integer speaker id.
///
/// Accepts either a known voice name (`af_heart`) or a bare integer
/// string (`"3"`, for power users who know the sid).  Returns a clear
/// error naming a few valid voices when the name is unknown.
pub(crate) fn name_to_sid(name: &str) -> Result<i32, TalkError> {
    let trimmed = name.trim();
    // Bare integer sid form.
    if let Ok(sid) = trimmed.parse::<i32>() {
        if SPEAKERS.iter().any(|(i, _)| *i == sid) {
            return Ok(sid);
        }
        return Err(TalkError::Config(format!(
            "kokoro voice sid {} out of range (valid: 0-{})",
            sid,
            SPEAKERS.len() - 1
        )));
    }
    // Named form.
    if let Some((sid, _)) = SPEAKERS.iter().find(|(_, n)| *n == trimmed) {
        return Ok(*sid);
    }
    Err(TalkError::Config(format!(
        "unknown kokoro voice '{}'. Examples: af_heart, am_michael, \
         ff_siwis. See the model card for all 53 voices.",
        trimmed
    )))
}

/// Resolve the default sid for a language when no voice is supplied.
///
/// Language-agnostic: looks up the voice-name prefixes for the
/// language's primary subtag in [`LANG_VOICE_PREFIXES`] and returns the
/// first speaker whose name carries one of those prefixes.  When the
/// language is unknown (no prefix mapping, or no matching speaker), it
/// falls back to the first speaker in the table (sid 0) rather than
/// hardcoding any particular language's voice.
pub(crate) fn default_sid_for_lang(lang: &str) -> i32 {
    let primary = lang.split('-').next().unwrap_or(lang).to_ascii_lowercase();
    if let Some((_, prefixes)) = LANG_VOICE_PREFIXES.iter().find(|(l, _)| *l == primary) {
        for (sid, name) in SPEAKERS {
            let name_prefix = name.split('_').next().unwrap_or("");
            if prefixes.contains(&name_prefix) {
                return *sid;
            }
        }
    }
    // Unknown language: fall back to the first speaker.
    SPEAKERS.first().map(|(sid, _)| *sid).unwrap_or(0)
}

/// The language primary-subtag a Kokoro voice *name* speaks, derived
/// from its two-letter prefix (`af`/`am`/`bf`/`bm` → `en`, `ff` → `fr`,
/// …).  Returns `None` for an unknown voice name or a bare-integer sid
/// form (the guard skips voices whose language it cannot know).
///
/// This is the Kokoro side of the mismatch-guard's
/// [`crate::synthesis::resolve::VoiceMeta`] lookup.
pub(crate) fn voice_language(name: &str) -> Option<crate::synthesis::resolve::VoiceMeta> {
    let trimmed = name.trim();
    // Resolve a bare-integer sid to its canonical name first, so
    // `--voice 30` guards exactly like `--voice ff_siwis`.
    let resolved_name = if let Ok(sid) = trimmed.parse::<i32>() {
        SPEAKERS.iter().find(|(i, _)| *i == sid).map(|(_, n)| *n)?
    } else if SPEAKERS.iter().any(|(_, n)| *n == trimmed) {
        trimmed
    } else {
        return None;
    };

    let prefix = resolved_name.split('_').next().unwrap_or("");
    let lang = LANG_VOICE_PREFIXES
        .iter()
        .find(|(_, prefixes)| prefixes.contains(&prefix))
        .map(|(l, _)| *l)?;
    Some(crate::synthesis::resolve::VoiceMeta {
        name: resolved_name.to_string(),
        lang: lang.to_string(),
    })
}

/// The default voice *name* for a language when no voice is supplied —
/// the name form of [`default_sid_for_lang`], returned only when the
/// language actually has a per-prefix speaker (so callers can tell
/// "Kokoro can voice this language" from "fall back to English").
pub(crate) fn default_voice_for_lang(lang: &str) -> Option<&'static str> {
    let primary = lang
        .split(['-', '_'])
        .next()
        .unwrap_or(lang)
        .to_ascii_lowercase();
    let (_, prefixes) = LANG_VOICE_PREFIXES.iter().find(|(l, _)| *l == primary)?;
    SPEAKERS
        .iter()
        .find(|(_, name)| prefixes.contains(&name.split('_').next().unwrap_or("")))
        .map(|(_, name)| *name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_53_speakers_with_contiguous_ids() {
        assert_eq!(SPEAKERS.len(), 53);
        for (idx, (sid, _)) in SPEAKERS.iter().enumerate() {
            assert_eq!(*sid, idx as i32, "sid must equal index");
        }
    }

    #[test]
    fn name_to_sid_known_names() {
        assert_eq!(name_to_sid("af_heart").unwrap(), 3);
        assert_eq!(name_to_sid("am_michael").unwrap(), 16);
        assert_eq!(name_to_sid("ff_siwis").unwrap(), 30);
        assert_eq!(name_to_sid("zm_yunyang").unwrap(), 52);
    }

    #[test]
    fn name_to_sid_trims_whitespace() {
        assert_eq!(name_to_sid("  af_heart \n").unwrap(), 3);
    }

    #[test]
    fn name_to_sid_bare_integer() {
        assert_eq!(name_to_sid("0").unwrap(), 0);
        assert_eq!(name_to_sid("52").unwrap(), 52);
    }

    #[test]
    fn name_to_sid_bare_integer_out_of_range() {
        let err = name_to_sid("53").expect_err("out of range");
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn name_to_sid_unknown_name_errors() {
        let err = name_to_sid("no_such_voice").expect_err("unknown");
        let msg = err.to_string();
        assert!(msg.contains("no_such_voice"));
        assert!(msg.contains("af_heart"));
    }

    #[test]
    fn default_sid_by_lang_is_prefix_derived() {
        // en → first en-prefixed voice (af_alloy, sid 0).
        assert_eq!(default_sid_for_lang("en"), 0);
        // fr → the only ff-prefixed voice (ff_siwis, sid 30).
        assert_eq!(default_sid_for_lang("fr"), 30);
        // zh → first zh-prefixed voice (zf_xiaobei, sid 45).
        assert_eq!(default_sid_for_lang("zh"), 45);
        // Region subtag is stripped: fr-FR behaves like fr.
        assert_eq!(default_sid_for_lang("fr-FR"), 30);
        // Unknown language falls back to the first speaker (sid 0),
        // NOT a hardcoded per-language voice.
        assert_eq!(default_sid_for_lang("xx"), 0);
    }

    #[test]
    fn voice_language_derives_prefix_language() {
        // English prefixes.
        assert_eq!(voice_language("af_heart").unwrap().lang, "en");
        assert_eq!(voice_language("am_michael").unwrap().lang, "en");
        assert_eq!(voice_language("bf_alice").unwrap().lang, "en");
        // French.
        let ff = voice_language("ff_siwis").unwrap();
        assert_eq!(ff.lang, "fr");
        assert_eq!(ff.name, "ff_siwis");
        // Other languages.
        assert_eq!(voice_language("zf_xiaobei").unwrap().lang, "zh");
        assert_eq!(voice_language("jf_alpha").unwrap().lang, "ja");
    }

    #[test]
    fn voice_language_bare_sid_resolves_to_name_and_lang() {
        // sid 30 == ff_siwis (fr).
        let v = voice_language("30").unwrap();
        assert_eq!(v.name, "ff_siwis");
        assert_eq!(v.lang, "fr");
    }

    #[test]
    fn voice_language_unknown_is_none() {
        assert!(voice_language("no_such_voice").is_none());
        // Out-of-range sid.
        assert!(voice_language("999").is_none());
    }

    #[test]
    fn default_voice_for_lang_names() {
        assert_eq!(default_voice_for_lang("en"), Some("af_alloy"));
        assert_eq!(default_voice_for_lang("fr"), Some("ff_siwis"));
        assert_eq!(default_voice_for_lang("fr-FR"), Some("ff_siwis"));
        assert_eq!(default_voice_for_lang("zh"), Some("zf_xiaobei"));
        // A language Kokoro has no speaker for.
        assert_eq!(default_voice_for_lang("de"), None);
        assert_eq!(default_voice_for_lang("xx"), None);
    }
}
