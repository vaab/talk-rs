//! Automatic language detection for the `speak` command.
//!
//! A thin, provider-agnostic wrapper around [`whichlang`] (quickwit-oss,
//! zero runtime deps).  whichlang classifies text into one of 16
//! languages; this module maps that classification to the lowercase
//! ISO 639-1 two-letter codes the synthesis backends speak (`"en"`,
//! `"fr"`, …) so the rest of the pipeline never touches the whichlang
//! enum directly.
//!
//! # Contract
//!
//! whichlang *always* returns a language — there is no "unknown"
//! variant, and it does not expose a confidence score, so even a
//! single word yields a guess.  We deliberately add **no length
//! heuristics**: per the design, we trust the detector and let the
//! downstream resolution chain (config `lang`, `--lang`, and the
//! provider's per-language voice fallback) handle anything the guess
//! gets wrong.  [`detect_lang`] therefore returns `Option` only to
//! model the *empty-input* edge (no text ⇒ nothing to detect), never a
//! low-confidence result.

use whichlang::{detect_language, Lang};

/// Map a [`whichlang::Lang`] to its lowercase ISO 639-1 two-letter
/// code.
///
/// whichlang's own `three_letter_code()` yields ISO 639-3 (`eng`,
/// `fra`, …); the synthesis backends key off the two-letter primary
/// subtag (`en`, `fr`, …), so we map explicitly here rather than
/// truncating the three-letter code (which would be wrong for several
/// languages).
fn lang_to_iso_639_1(lang: Lang) -> &'static str {
    match lang {
        Lang::Ara => "ar",
        Lang::Cmn => "zh", // Mandarin Chinese
        Lang::Deu => "de",
        Lang::Eng => "en",
        Lang::Fra => "fr",
        Lang::Hin => "hi",
        Lang::Ita => "it",
        Lang::Jpn => "ja",
        Lang::Kor => "ko",
        Lang::Nld => "nl",
        Lang::Por => "pt",
        Lang::Rus => "ru",
        Lang::Spa => "es",
        Lang::Swe => "sv",
        Lang::Tur => "tr",
        Lang::Vie => "vi",
    }
}

/// Detect the language of `text`, returning its lowercase ISO 639-1
/// code (`"en"`, `"fr"`, …).
///
/// Returns `None` only when `text` is empty / whitespace-only (nothing
/// to classify).  For any non-empty input whichlang always yields a
/// language, so a `Some` is returned even for very short strings — see
/// the module docs for why we trust that unconditionally.
pub fn detect_lang(text: &str) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }
    Some(lang_to_iso_639_1(detect_language(text)).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_english_sentence() {
        assert_eq!(
            detect_lang("The quick brown fox jumps over the lazy dog."),
            Some("en".to_string())
        );
    }

    #[test]
    fn detects_french_accented_sentence() {
        assert_eq!(
            detect_lang("Bonjour, comment allez-vous aujourd'hui ? J'espère que tout va bien."),
            Some("fr".to_string())
        );
    }

    #[test]
    fn detects_french_without_accents() {
        // No accents at all — whichlang must still pick French from the
        // word shapes / n-grams alone.
        assert_eq!(
            detect_lang("Salut ca va bien merci et toi comment vas tu aujourd hui"),
            Some("fr".to_string())
        );
    }

    #[test]
    fn detects_simple_greeting() {
        // The canonical acceptance example.
        assert_eq!(
            detect_lang("Bonjour tout le monde, je suis ravi de vous voir"),
            Some("fr".to_string())
        );
    }

    #[test]
    fn empty_input_yields_none() {
        assert_eq!(detect_lang(""), None);
        assert_eq!(detect_lang("   \n\t "), None);
    }

    #[test]
    fn maps_every_whichlang_variant_to_two_letters() {
        // Guard: every mapped code is exactly two lowercase ASCII
        // letters (no ISO 639-3 leakage).
        for lang in [
            Lang::Ara,
            Lang::Cmn,
            Lang::Deu,
            Lang::Eng,
            Lang::Fra,
            Lang::Hin,
            Lang::Ita,
            Lang::Jpn,
            Lang::Kor,
            Lang::Nld,
            Lang::Por,
            Lang::Rus,
            Lang::Spa,
            Lang::Swe,
            Lang::Tur,
            Lang::Vie,
        ] {
            let code = lang_to_iso_639_1(lang);
            assert_eq!(code.len(), 2, "{:?} -> {}", lang, code);
            assert!(
                code.chars().all(|c| c.is_ascii_lowercase()),
                "{:?} -> {}",
                lang,
                code
            );
        }
    }
}
