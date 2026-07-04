//! Embedded Mistral / Voxtral **preset** voice metadata + per-language
//! voice resolution.
//!
//! The `speak` command's mismatch guard needs to know the language of a
//! pinned Mistral voice, but a Mistral voice is an opaque UUID on the
//! wire.  The set of *preset* voices (as returned by
//! `GET /v1/audio/voices?voice_type=preset`) is small and stable, so we
//! embed the full table (uuid → display name + language) here.  A UUID
//! **absent** from this table is treated as a user *custom* voice of
//! unknowable language — the guard skips it silently (per the spec).
//!
//! The table also powers the built-in per-language default voices
//! ([`default_voice_for_lang`]) used when neither `--voice`,
//! `providers.mistral.tts_voice`, nor `providers.mistral.tts_voices`
//! selects a voice.

use crate::synthesis::resolve::{primary_subtag, VoiceMeta};

/// One embedded preset voice: `(uuid, display_name, language)`.
///
/// `language` is region-qualified (`"en_us"`, `"en_gb"`, `"fr_fr"`) to
/// mirror the API's own naming; the guard compares by primary subtag,
/// so `en_us` and `en_gb` both match a resolved `en`.
struct Preset {
    uuid: &'static str,
    name: &'static str,
    lang: &'static str,
}

/// The full preset voice table (30 voices across Paul/Oliver/Jane/Marie).
///
/// Sourced verbatim from the `GET /v1/audio/voices?voice_type=preset`
/// listing.  Names carry the emotion variant so the guard's error is
/// unambiguous (e.g. `"Paul - Neutral"`).
const PRESETS: &[Preset] = &[
    // Paul — en_us (8 emotion variants).
    Preset {
        uuid: "530e2e20-58e2-45d8-b0a5-4594f4915944",
        name: "Paul - Sad",
        lang: "en_us",
    },
    Preset {
        uuid: "c69964a6-ab8b-4f8a-9465-ec0925096ec8",
        name: "Paul - Neutral",
        lang: "en_us",
    },
    Preset {
        uuid: "1024d823-a11e-43ee-bf3d-d440dccc0577",
        name: "Paul - Happy",
        lang: "en_us",
    },
    Preset {
        uuid: "1f017bcb-02e5-460d-989b-db065c0c6122",
        name: "Paul - Frustrated",
        lang: "en_us",
    },
    Preset {
        uuid: "5940190b-f58a-4c3e-8264-a40d63fd6883",
        name: "Paul - Excited",
        lang: "en_us",
    },
    Preset {
        uuid: "98559b22-62b5-4a64-a7cd-fc78ca41faa8",
        name: "Paul - Confident",
        lang: "en_us",
    },
    Preset {
        uuid: "01d985cd-5e0c-4457-bfd8-80ba31a5bc03",
        name: "Paul - Cheerful",
        lang: "en_us",
    },
    Preset {
        uuid: "cb891218-482c-4392-9878-91e8d999d57a",
        name: "Paul - Angry",
        lang: "en_us",
    },
    // Oliver — en_gb (7 variants).
    Preset {
        uuid: "e3596645-b1af-469e-b857-f18ddedc7652",
        name: "Oliver - Neutral",
        lang: "en_gb",
    },
    Preset {
        uuid: "d4101b8f-12c3-450d-a812-7d700b3a3245",
        name: "Oliver - Sad",
        lang: "en_gb",
    },
    Preset {
        uuid: "e8e5b1de-493c-4061-8414-e2170f9f4b6f",
        name: "Oliver - Excited",
        lang: "en_gb",
    },
    Preset {
        uuid: "390c8a2b-60a6-4882-8437-c49a8bd33b63",
        name: "Oliver - Curious",
        lang: "en_gb",
    },
    Preset {
        uuid: "8169ab87-bc99-4669-a5ec-6855860ace24",
        name: "Oliver - Confident",
        lang: "en_gb",
    },
    Preset {
        uuid: "5ad5d44e-6b4e-4a57-a8a8-4cae088034ed",
        name: "Oliver - Cheerful",
        lang: "en_gb",
    },
    Preset {
        uuid: "862274a7-8333-48f7-b668-f19c932999e0",
        name: "Oliver - Angry",
        lang: "en_gb",
    },
    // Jane — en_gb (9 variants).
    Preset {
        uuid: "82c99ee6-f932-423f-a4a3-d403c8914b8d",
        name: "Jane - Neutral",
        lang: "en_gb",
    },
    Preset {
        uuid: "c7a8eb83-5247-4540-89f3-6650d349100d",
        name: "Jane - Sad",
        lang: "en_gb",
    },
    Preset {
        uuid: "cbe96cf0-85ec-4a10-accb-0b35c93b6dfd",
        name: "Jane - Confident",
        lang: "en_gb",
    },
    Preset {
        uuid: "5de47977-6e47-4266-a938-3bc1d76b4676",
        name: "Jane - Curious",
        lang: "en_gb",
    },
    Preset {
        uuid: "60844938-221d-4d1e-8233-34203f787d9f",
        name: "Jane - Frustrated",
        lang: "en_gb",
    },
    Preset {
        uuid: "a3e41ea8-020b-44c0-8d8b-f6cc03524e31",
        name: "Jane - Sarcasm",
        lang: "en_gb",
    },
    Preset {
        uuid: "7d0a90a3-c211-4489-aaa0-61269299edc7",
        name: "Jane - Confused",
        lang: "en_gb",
    },
    Preset {
        uuid: "e7168caa-f7ed-4e1c-98a1-434251f4f2b0",
        name: "Jane - Jealousy",
        lang: "en_gb",
    },
    Preset {
        uuid: "230ccacf-8800-4aa0-8ac2-8d004f1d9fb7",
        name: "Jane - Shameful",
        lang: "en_gb",
    },
    // Marie — fr_fr (6 variants).
    Preset {
        uuid: "5a271406-039d-46fe-835b-fbbb00eaf08d",
        name: "Marie - Neutral",
        lang: "fr_fr",
    },
    Preset {
        uuid: "49d024dd-981b-4462-bb17-74d381eb8fd7",
        name: "Marie - Happy",
        lang: "fr_fr",
    },
    Preset {
        uuid: "4adeb2c6-25a3-44bc-8100-5234dfc1193b",
        name: "Marie - Sad",
        lang: "fr_fr",
    },
    Preset {
        uuid: "2f62b1af-aea3-4079-9d10-7ca665ee7243",
        name: "Marie - Excited",
        lang: "fr_fr",
    },
    Preset {
        uuid: "e0580ce5-e63c-4cbe-88c8-a983b80c5f1f",
        name: "Marie - Curious",
        lang: "fr_fr",
    },
    Preset {
        uuid: "a7c07cdc-1c35-4d87-a938-c610a654f600",
        name: "Marie - Angry",
        lang: "fr_fr",
    },
];

/// Built-in default preset voice per language primary-subtag.
///
/// Used as the final fallback in the Mistral voice-resolution chain
/// (after `--voice`, `tts_voice`, `tts_voices`).  Deliberately small:
/// only the languages Voxtral TTS ships preset voices for (English →
/// Paul Neutral, French → Marie Neutral).
const DEFAULT_VOICE_BY_LANG: &[(&str, &str)] = &[
    ("en", "c69964a6-ab8b-4f8a-9465-ec0925096ec8"), // Paul - Neutral (en_us)
    ("fr", "5a271406-039d-46fe-835b-fbbb00eaf08d"), // Marie - Neutral (fr_fr)
];

/// Look up the [`VoiceMeta`] for a preset voice UUID.
///
/// Returns `None` for any UUID not in the embedded preset table — i.e.
/// a user *custom* voice of unknowable language, which the mismatch
/// guard skips silently.
pub fn voice_language(uuid: &str) -> Option<VoiceMeta> {
    let trimmed = uuid.trim();
    PRESETS
        .iter()
        .find(|p| p.uuid == trimmed)
        .map(|p| VoiceMeta {
            name: p.name.to_string(),
            lang: p.lang.to_string(),
        })
}

/// The built-in default preset voice UUID for `lang`'s primary subtag,
/// if one exists (`en` → Paul, `fr` → Marie).
pub fn default_voice_for_lang(lang: &str) -> Option<&'static str> {
    let primary = primary_subtag(lang);
    DEFAULT_VOICE_BY_LANG
        .iter()
        .find(|(l, _)| *l == primary)
        .map(|(_, uuid)| *uuid)
}

/// The set of language primary-subtags the built-in default table can
/// voice — used by the resolver to decide whether a detected language
/// is speakable by Mistral before falling back to English.
pub fn supported_langs() -> impl Iterator<Item = &'static str> {
    DEFAULT_VOICE_BY_LANG.iter().map(|(l, _)| *l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_table_is_complete() {
        // 8 Paul + 7 Oliver + 9 Jane + 6 Marie = 30 voices.
        assert_eq!(PRESETS.len(), 30);
    }

    #[test]
    fn every_uuid_is_unique() {
        let mut seen = std::collections::HashSet::new();
        for p in PRESETS {
            assert!(seen.insert(p.uuid), "duplicate uuid {}", p.uuid);
        }
    }

    #[test]
    fn voice_language_known_presets() {
        let paul = voice_language("c69964a6-ab8b-4f8a-9465-ec0925096ec8").expect("paul");
        assert_eq!(paul.name, "Paul - Neutral");
        assert_eq!(paul.lang, "en_us");

        let marie = voice_language("5a271406-039d-46fe-835b-fbbb00eaf08d").expect("marie");
        assert_eq!(marie.name, "Marie - Neutral");
        assert_eq!(marie.lang, "fr_fr");
    }

    #[test]
    fn voice_language_trims() {
        assert!(voice_language("  5a271406-039d-46fe-835b-fbbb00eaf08d \n").is_some());
    }

    #[test]
    fn voice_language_unknown_uuid_is_none() {
        // A custom (non-preset) voice → unknowable language.
        assert!(voice_language("00000000-0000-0000-0000-000000000000").is_none());
    }

    #[test]
    fn default_voice_for_lang_en_fr() {
        assert_eq!(
            default_voice_for_lang("en"),
            Some("c69964a6-ab8b-4f8a-9465-ec0925096ec8")
        );
        assert_eq!(
            default_voice_for_lang("en_us"),
            Some("c69964a6-ab8b-4f8a-9465-ec0925096ec8")
        );
        assert_eq!(
            default_voice_for_lang("fr"),
            Some("5a271406-039d-46fe-835b-fbbb00eaf08d")
        );
        // A language with no preset voice.
        assert_eq!(default_voice_for_lang("de"), None);
    }
}
