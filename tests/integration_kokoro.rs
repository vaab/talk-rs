//! Integration test for the local Kokoro TTS synthesis backend.
//!
//! **Gated** on an environment variable so CI without the (~350 MB)
//! model stays green:
//!
//! * `TALK_RS_TEST_KOKORO_MODEL_DIR` — absolute path to an extracted
//!   `kokoro-multi-lang-v1_0` model directory (must contain
//!   `model.onnx`, `voices.bin`, `tokens.txt`, `espeak-ng-data/`,
//!   `dict/`).  A working local copy exists at
//!   `/tmp/kokoro-poc/kokoro-multi-lang-v1_0`.
//!
//! When unset, the test prints a skip banner and returns successfully
//! (no-op).
//!
//! Public-API path: drives the same
//! [`talk_rs::synthesis::create_oneshot_synthesizer`] factory the
//! `speak` command uses, with a `Config` whose
//! `providers.kokoro.model_dir` points at the env-supplied dir.  The
//! model files are already on disk, so `validate` short-circuits and no
//! network I/O happens.

#![cfg(feature = "kokoro")]

use std::path::PathBuf;

use talk_rs::config::{Config, KokoroConfig, ProvidersConfig, SynthesisProvider};
use talk_rs::synthesis::{synthesize, SynthesisRequest};

/// Return `Some(path)` if env var is set to a non-empty value.
fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

fn config_with_kokoro(model_dir: &std::path::Path, output_dir: &std::path::Path) -> Config {
    Config {
        output_dir: output_dir.to_path_buf(),
        providers: ProvidersConfig {
            mistral: None,
            openai: None,
            parakeet: None,
            kokoro: Some(KokoroConfig {
                variant: None,
                model_dir: Some(model_dir.to_path_buf()),
                voice: None,
                num_threads: Some(2),
                lang: None,
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

#[tokio::test]
async fn kokoro_synthesizes_english_speech() {
    let Some(model_dir) = env_path("TALK_RS_TEST_KOKORO_MODEL_DIR") else {
        eprintln!(
            "SKIP: kokoro_synthesizes_english_speech — \
             TALK_RS_TEST_KOKORO_MODEL_DIR not set. \
             Point it at an extracted kokoro-multi-lang-v1_0 model dir to enable."
        );
        return;
    };
    assert!(
        model_dir.is_dir(),
        "TALK_RS_TEST_KOKORO_MODEL_DIR does not point at a directory: {}",
        model_dir.display()
    );
    assert!(
        model_dir.join("model.onnx").is_file(),
        "model dir must contain a pre-installed model.onnx (in {}). \
         The synthesis pipeline does not auto-download in tests.",
        model_dir.display()
    );

    let tmp = tempfile::TempDir::new().expect("create tempdir");
    let cfg = config_with_kokoro(&model_dir, tmp.path());

    let result = synthesize(
        &cfg,
        SynthesisProvider::Kokoro,
        SynthesisRequest {
            text: "The quick brown fox jumps over the lazy dog.".to_string(),
            voice: Some("af_heart".to_string()),
            speed: None,
            lang: Some("en".to_string()),
        },
    )
    .await
    .expect("kokoro synthesis must succeed");

    eprintln!(
        "[kokoro] synthesized {} samples @ {} Hz ({:.2}s)",
        result.pcm.len(),
        result.sample_rate,
        result.duration_secs()
    );

    // Kokoro outputs 24 kHz audio.
    assert_eq!(result.sample_rate, 24_000);
    assert!(
        !result.pcm.is_empty(),
        "synthesis produced no audio — model may be misloaded"
    );
    // A ~9-word sentence should be at least ~0.5s of audio and not
    // absurdly long.
    let dur = result.duration_secs();
    assert!(
        (0.5..30.0).contains(&dur),
        "implausible synthesized duration {:.2}s",
        dur
    );
    // The audio must carry real signal (not pure silence).
    let peak = result
        .pcm
        .iter()
        .map(|s| s.unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(peak > 100, "synthesized audio is silent (peak={})", peak);
}

#[tokio::test]
async fn kokoro_synthesizes_french_via_derived_model() {
    let Some(model_dir) = env_path("TALK_RS_TEST_KOKORO_MODEL_DIR") else {
        eprintln!(
            "SKIP: kokoro_synthesizes_french_via_derived_model — \
             TALK_RS_TEST_KOKORO_MODEL_DIR not set."
        );
        return;
    };
    if !model_dir.join("model.onnx").is_file() {
        eprintln!("SKIP: model.onnx missing in {}", model_dir.display());
        return;
    }

    let tmp = tempfile::TempDir::new().expect("create tempdir");
    let cfg = config_with_kokoro(&model_dir, tmp.path());

    // French drives the on-demand `model-fr.onnx` derivation + the
    // ff_siwis default voice.  This exercises the language-agnostic
    // ONNX voice-metadata patch end-to-end.
    let result = synthesize(
        &cfg,
        SynthesisProvider::Kokoro,
        SynthesisRequest {
            text: "Bonjour, comment allez-vous aujourd'hui ?".to_string(),
            voice: None, // → ff_siwis default for fr
            speed: None,
            lang: Some("fr".to_string()),
        },
    )
    .await
    .expect("french synthesis must succeed");

    eprintln!(
        "[kokoro-fr] synthesized {} samples @ {} Hz ({:.2}s)",
        result.pcm.len(),
        result.sample_rate,
        result.duration_secs()
    );

    assert_eq!(result.sample_rate, 24_000);
    assert!(!result.pcm.is_empty(), "french synthesis produced no audio");
    let peak = result
        .pcm
        .iter()
        .map(|s| s.unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(peak > 100, "french audio is silent (peak={})", peak);
}
