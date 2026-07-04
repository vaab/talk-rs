//! Kokoro model lifecycle: presence check, download, extract, and the
//! per-language ONNX metadata patch.
//!
//! The download / extract / atomic-promote / consent machinery is
//! SHARED with the parakeet backend via [`crate::model_fetch`] — this
//! module only supplies the Kokoro-specific [`ModelSpec`] and the
//! per-language model derivation.
//!
//! # Language support: the `voice` metadata patch (language-agnostic)
//!
//! Kokoro's phonemization language is baked into the ONNX model
//! metadata under the `voice` key (NOT selectable per request).  The
//! stock `model.onnx` ships with one baked language (currently
//! `en-us`); feeding it another language's text produces English
//! pronunciation of foreign words.  To synthesize a *different*
//! language we derive a sibling `model-<lang>.onnx` whose `voice`
//! metadata reads the requested language code (e.g. `fr`).
//!
//! This module is **agnostic about which languages exist**: the set of
//! languages, and the default, come from config / the request, never
//! from a hardcoded en/fr table.  The one thing we cannot invent is the
//! language *code* espeak-ng expects — the caller supplies it verbatim
//! (`--lang fr`, `--lang de`, …) and it is written into the metadata
//! as-is.
//!
//! The ONNX file is a protobuf.  The `voice` metadata lives in a
//! `metadata_props` entry (`StringStringEntryProto { key, value }`)
//! serialized as:
//!
//! ```text
//! 0x72 <entry_len> 0x0a 0x05 "voice" 0x12 <value_len> <value>
//! ```
//!
//! We do a **length-aware, verified byte patch**: locate the single
//! `voice` entry, read its current value, and rewrite the value to the
//! target language code, recomputing BOTH single-byte varint length
//! prefixes (entry length and value length).  Language codes and the
//! surrounding lengths are all short (< 128 bytes), so the varints stay
//! single-byte and the rewrite is trivially correct.  We refuse to
//! patch (returning a clear error) unless exactly one `voice` entry is
//! present, so a future model layout change fails loudly rather than
//! silently corrupting the file.
//!
//! An empty lexicon file (`empty_lexicon.txt`) is also created for use
//! by non-baked languages, since the shipped lexicons are language
//! specific and a derived-language model relies purely on espeak-ng.

use crate::error::TalkError;
use crate::model_fetch::ModelSpec;
use std::path::Path;

/// Tarball asset URL for `kokoro-multi-lang-v1_0` (~350 MB).
const KOKORO_TARBALL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2";

/// Top-level directory name inside the tarball.
const KOKORO_TARBALL_INNER_DIR: &str = "kokoro-multi-lang-v1_0";

/// The core files a complete Kokoro model dir must contain.  The
/// `espeak-ng-data/` and `dict/` directories and the lexicons are also
/// shipped but are not part of the presence check; these three regular
/// files are the load-bearing minimum.
const MODEL_FILES: &[&str] = &["model.onnx", "voices.bin", "tokens.txt"];

/// Name of the stock model file inside the dir.
pub(crate) const MODEL_EN: &str = "model.onnx";
/// Name of the empty lexicon used for derived-language models.
pub(crate) const EMPTY_LEXICON: &str = "empty_lexicon.txt";

/// The shared [`ModelSpec`] describing the Kokoro tarball.
pub(super) const KOKORO_SPEC: ModelSpec = ModelSpec {
    display_name: "Kokoro",
    tarball_url: KOKORO_TARBALL_URL,
    inner_dir: KOKORO_TARBALL_INNER_DIR,
    required_files: MODEL_FILES,
    approx_size: "~350 MB",
    manual_files_hint: "model.onnx/voices.bin/tokens.txt (plus espeak-ng-data/ and dict/)",
};

/// Filename of the derived model for language `lang` (e.g.
/// `model-fr.onnx`).  The stock model (`model.onnx`) already carries
/// its own baked language, so it is used directly for that language.
pub(crate) fn model_filename_for_lang(lang: &str) -> String {
    format!("model-{}.onnx", lang)
}

/// True iff the core Kokoro model files are present and non-empty.
pub(crate) fn is_present(dir: &Path) -> bool {
    crate::model_fetch::is_present(dir, &KOKORO_SPEC)
}

/// Presence gate (no download).  Errors with a manual-install message
/// naming the URL + dir when absent.
pub(crate) fn ensure_present(dir: &Path) -> Result<(), TalkError> {
    crate::model_fetch::ensure_present(dir, &KOKORO_SPEC)
}

/// Ensure the model is present, prompting for consent on a TTY (or
/// proceeding with a stderr log when non-interactive).
///
/// This performs ONLY the base-model download — it is deliberately
/// language-agnostic and does NOT pre-generate any per-language model.
/// Derived-language models are created lazily, on demand, by
/// [`ensure_lang_model`] the first time a given language is requested.
pub(crate) async fn ensure_with_cli_consent(dir: &Path) -> Result<(), TalkError> {
    crate::model_fetch::ensure_with_cli_consent(dir, &KOKORO_SPEC).await
}

/// The baked language of the stock `model.onnx`, read from its ONNX
/// `voice` metadata.  Everything about language selection keys off this
/// discovered value rather than a hardcoded assumption.
pub(crate) fn baked_language(dir: &Path) -> Result<String, TalkError> {
    let en_model = dir.join(MODEL_EN);
    let bytes = std::fs::read(&en_model).map_err(|e| {
        TalkError::Config(format!(
            "failed to read Kokoro model {}: {}",
            en_model.display(),
            e
        ))
    })?;
    read_voice_metadata(&bytes)
}

/// Resolve the on-disk model file to use for the requested `lang`,
/// generating a derived `model-<lang>.onnx` if necessary.
///
/// * When `lang` equals the stock model's baked language, the stock
///   `model.onnx` is used directly (no derivation).
/// * Otherwise a `model-<lang>.onnx` is derived from the stock model by
///   rewriting the `voice` metadata to `lang`, created lazily and
///   cached on disk.  An `empty_lexicon.txt` is ensured alongside.
///
/// Returns the absolute path of the model file to load.
pub(crate) fn ensure_lang_model(dir: &Path, lang: &str) -> Result<std::path::PathBuf, TalkError> {
    let baked = baked_language(dir)?;

    // Stock model already speaks this language: use it directly.
    // Match on the full baked code (`en-us`) or its primary subtag
    // (`en`), so `--lang en` maps to the `en-us` stock model.
    if lang == baked || baked.split('-').next() == Some(lang) {
        return Ok(dir.join(MODEL_EN));
    }

    // Ensure the empty lexicon exists (derived-language models rely on
    // espeak-ng, not the shipped language-specific lexicons).
    let empty_lexicon = dir.join(EMPTY_LEXICON);
    if !empty_lexicon.exists() {
        std::fs::write(&empty_lexicon, b"").map_err(|e| {
            TalkError::Config(format!(
                "failed to create {}: {}",
                empty_lexicon.display(),
                e
            ))
        })?;
    }

    let derived = dir.join(model_filename_for_lang(lang));
    if derived.exists() {
        return Ok(derived);
    }

    let en_model = dir.join(MODEL_EN);
    let bytes = std::fs::read(&en_model).map_err(|e| {
        TalkError::Config(format!(
            "failed to read stock Kokoro model {} for language patch: {}",
            en_model.display(),
            e
        ))
    })?;
    let patched = patch_voice_metadata(&bytes, lang)?;

    // Write to a temp sibling then atomic-rename so a crash mid-write
    // never leaves a truncated derived model.
    let tmp = dir.join(format!("model-{}.onnx.tmp", lang));
    std::fs::write(&tmp, &patched)
        .map_err(|e| TalkError::Config(format!("failed to write {}: {}", tmp.display(), e)))?;
    std::fs::rename(&tmp, &derived).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        TalkError::Config(format!(
            "failed to promote {} -> {}: {}",
            tmp.display(),
            derived.display(),
            e
        ))
    })?;
    log::info!(
        "kokoro: generated {} for language '{}'",
        derived.display(),
        lang
    );
    Ok(derived)
}

/// Key marker for the `voice` metadata entry:
/// `0x0a 0x05 "voice" 0x12` — i.e. field 1 (key) = `"voice"` followed
/// by the field-2 (value) tag.  The value's own single-byte length
/// varint immediately follows this marker.
const VOICE_KEY_MARKER: &[u8] = b"\x0a\x05voice\x12";

/// Full-entry tag byte preceding the key marker:
/// `0x72` = field 14 (`metadata_props`), wire type 2, whose next byte
/// is the entry-length varint.
const METADATA_ENTRY_TAG: u8 = 0x72;

/// Locate the single `voice` metadata entry and return
/// `(entry_tag_index, value_start, value_len)`.
///
/// `entry_tag_index` points at the `0x72` entry tag; `value_start`
/// points at the first byte of the value string; `value_len` is the
/// value's length (from its single-byte varint).  Errors when the
/// marker is absent, ambiguous, or uses a multi-byte length varint
/// (which our single-byte rewrite does not handle).
fn locate_voice_entry(bytes: &[u8]) -> Result<(usize, usize, usize), TalkError> {
    let mut hits = bytes
        .windows(VOICE_KEY_MARKER.len())
        .enumerate()
        .filter(|(_, w)| *w == VOICE_KEY_MARKER)
        .map(|(i, _)| i);

    let marker = hits.next().ok_or_else(|| {
        TalkError::Config(
            "kokoro language patch: could not locate the `voice` ONNX metadata \
             entry (unexpected model layout). Provide a hand-patched \
             model-<lang>.onnx (voice metadata = the language code) next to \
             model.onnx."
                .to_string(),
        )
    })?;
    if hits.next().is_some() {
        return Err(TalkError::Config(
            "kokoro language patch: the `voice` ONNX metadata entry appears \
             more than once; refusing to patch ambiguously."
                .to_string(),
        ));
    }

    // The value length varint sits immediately after the marker.
    let value_len_idx = marker + VOICE_KEY_MARKER.len();
    let value_len_byte = *bytes.get(value_len_idx).ok_or_else(|| {
        TalkError::Config("kokoro language patch: truncated voice metadata".to_string())
    })?;
    // We only handle single-byte varints (< 128) — language codes are
    // tiny, so any real value fits.
    if value_len_byte >= 0x80 {
        return Err(TalkError::Config(
            "kokoro language patch: multi-byte value length not supported".to_string(),
        ));
    }
    let value_len = value_len_byte as usize;
    let value_start = value_len_idx + 1;

    // Walk back from the marker to the entry tag (`0x72`) so we can fix
    // the entry-length varint too.  The marker is preceded by the
    // entry-length byte, which is preceded by the tag.
    if marker < 2 {
        return Err(TalkError::Config(
            "kokoro language patch: voice entry too close to start of file".to_string(),
        ));
    }
    let entry_tag_idx = marker - 2;
    if bytes[entry_tag_idx] != METADATA_ENTRY_TAG {
        return Err(TalkError::Config(
            "kokoro language patch: unexpected metadata entry framing".to_string(),
        ));
    }

    if value_start + value_len > bytes.len() {
        return Err(TalkError::Config(
            "kokoro language patch: voice value runs past end of file".to_string(),
        ));
    }

    Ok((entry_tag_idx, value_start, value_len))
}

/// Read the current `voice` metadata value (the baked language code).
pub(crate) fn read_voice_metadata(bytes: &[u8]) -> Result<String, TalkError> {
    let (_, value_start, value_len) = locate_voice_entry(bytes)?;
    let raw = &bytes[value_start..value_start + value_len];
    String::from_utf8(raw.to_vec()).map_err(|_| {
        TalkError::Config("kokoro language patch: voice metadata is not UTF-8".to_string())
    })
}

/// Rewrite the ONNX `voice` metadata to `lang`, recomputing the entry
/// and value length varints.
///
/// Language-agnostic: `lang` is written verbatim as the new value.
/// The entry-length varint is adjusted by the value-length delta.  The
/// key (`voice`) length is unchanged.  Requires `lang` to be short
/// enough (< 128 bytes plus the fixed 8-byte key framing) that the
/// entry length stays a single-byte varint — trivially true for any
/// real language code.
pub(crate) fn patch_voice_metadata(bytes: &[u8], lang: &str) -> Result<Vec<u8>, TalkError> {
    if lang.is_empty() {
        return Err(TalkError::Config(
            "kokoro language patch: empty language code".to_string(),
        ));
    }
    if lang.len() >= 0x80 {
        return Err(TalkError::Config(
            "kokoro language patch: language code too long".to_string(),
        ));
    }

    let (entry_tag_idx, value_start, old_value_len) = locate_voice_entry(bytes)?;
    let entry_len_idx = entry_tag_idx + 1;
    let old_entry_len = bytes[entry_len_idx];
    if old_entry_len >= 0x80 {
        return Err(TalkError::Config(
            "kokoro language patch: multi-byte entry length not supported".to_string(),
        ));
    }

    let new_value_len = lang.len();
    // new_entry_len = old_entry_len - old_value_len + new_value_len.
    let new_entry_len_i = old_entry_len as isize - old_value_len as isize + new_value_len as isize;
    if !(0..0x80).contains(&new_entry_len_i) {
        return Err(TalkError::Config(
            "kokoro language patch: patched entry length out of single-byte range".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(bytes.len() + new_value_len);
    // Everything up to and including the entry tag.
    out.extend_from_slice(&bytes[..=entry_tag_idx]);
    // New entry length.
    out.push(new_entry_len_i as u8);
    // The key framing `0x0a 0x05 "voice" 0x12` (VOICE_KEY_MARKER).
    out.extend_from_slice(VOICE_KEY_MARKER);
    // New value length + value bytes.
    out.push(new_value_len as u8);
    out.extend_from_slice(lang.as_bytes());
    // Everything after the old value.
    out.extend_from_slice(&bytes[value_start + old_value_len..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact stock `voice = en-us` entry bytes.
    const VOICE_EN_ENTRY: &[u8] = b"\x72\x0e\x0a\x05voice\x12\x05en-us";

    fn synthetic_model() -> Vec<u8> {
        let mut v = b"onnx-header-noise".to_vec();
        v.extend_from_slice(VOICE_EN_ENTRY);
        v.extend_from_slice(b"onnx-trailing-noise");
        v
    }

    #[test]
    fn read_voice_metadata_returns_baked_language() {
        let m = synthetic_model();
        assert_eq!(read_voice_metadata(&m).unwrap(), "en-us");
    }

    #[test]
    fn patch_voice_metadata_to_fr() {
        let m = synthetic_model();
        let patched = patch_voice_metadata(&m, "fr").expect("patch");
        // fr is 3 bytes shorter than en-us.
        assert_eq!(patched.len(), m.len() - 3);
        assert_eq!(read_voice_metadata(&patched).unwrap(), "fr");
        // Surrounding noise preserved.
        assert!(patched.starts_with(b"onnx-header-noise"));
        assert!(patched.ends_with(b"onnx-trailing-noise"));
    }

    #[test]
    fn patch_voice_metadata_to_longer_code() {
        // A longer code (e.g. "de-de", same length as en-us) round-trips.
        let m = synthetic_model();
        let patched = patch_voice_metadata(&m, "de-de").expect("patch");
        assert_eq!(patched.len(), m.len()); // same length
        assert_eq!(read_voice_metadata(&patched).unwrap(), "de-de");
    }

    #[test]
    fn patch_voice_metadata_to_short_and_back_roundtrips_framing() {
        let m = synthetic_model();
        let de = patch_voice_metadata(&m, "de").expect("patch de");
        // Re-patch the derived model to another language.
        let it = patch_voice_metadata(&de, "it").expect("patch it");
        assert_eq!(read_voice_metadata(&it).unwrap(), "it");
    }

    #[test]
    fn patch_voice_metadata_errors_when_absent() {
        let buf = b"no voice metadata entry here".to_vec();
        let err = patch_voice_metadata(&buf, "fr").expect_err("must error");
        assert!(err.to_string().contains("could not locate"));
    }

    #[test]
    fn patch_voice_metadata_errors_when_ambiguous() {
        let mut buf = Vec::new();
        buf.extend_from_slice(VOICE_EN_ENTRY);
        buf.extend_from_slice(b"----");
        buf.extend_from_slice(VOICE_EN_ENTRY);
        let err = patch_voice_metadata(&buf, "fr").expect_err("must error");
        assert!(err.to_string().contains("more than once"));
    }

    #[test]
    fn patch_voice_metadata_rejects_empty_lang() {
        let m = synthetic_model();
        let err = patch_voice_metadata(&m, "").expect_err("must error");
        assert!(err.to_string().contains("empty language code"));
    }

    #[test]
    fn ensure_lang_model_stock_for_baked_language() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join(MODEL_EN), synthetic_model()).unwrap();
        std::fs::write(dir.join("voices.bin"), b"x").unwrap();
        std::fs::write(dir.join("tokens.txt"), b"x").unwrap();

        // `en` maps to the baked `en-us` stock model (primary subtag).
        let p = ensure_lang_model(dir, "en").expect("resolve en");
        assert_eq!(p, dir.join(MODEL_EN));
        // Full baked code also maps to the stock model.
        let p2 = ensure_lang_model(dir, "en-us").expect("resolve en-us");
        assert_eq!(p2, dir.join(MODEL_EN));
    }

    #[test]
    fn ensure_lang_model_derives_and_caches() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join(MODEL_EN), synthetic_model()).unwrap();

        let p = ensure_lang_model(dir, "fr").expect("derive fr");
        assert_eq!(p, dir.join("model-fr.onnx"));
        assert!(dir.join(EMPTY_LEXICON).exists());
        assert_eq!(
            read_voice_metadata(&std::fs::read(&p).unwrap()).unwrap(),
            "fr"
        );
        // Second call is cached (no re-derivation error).
        let p2 = ensure_lang_model(dir, "fr").expect("cached fr");
        assert_eq!(p, p2);
    }

    #[test]
    fn kokoro_spec_has_expected_url_and_inner_dir() {
        assert!(KOKORO_SPEC
            .tarball_url
            .ends_with("kokoro-multi-lang-v1_0.tar.bz2"));
        assert_eq!(KOKORO_SPEC.inner_dir, "kokoro-multi-lang-v1_0");
    }
}
