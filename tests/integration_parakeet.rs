//! Integration test for the local Parakeet transcription backend.
//!
//! This test is **gated** on two environment variables so CI without
//! the (~640 MB) model stays green:
//!
//! * `TALK_RS_TEST_PARAKEET_MODEL_DIR` — absolute path to an
//!   extracted sherpa-onnx Parakeet TDT v3 INT8 model directory
//!   (must contain `encoder.int8.onnx`, `decoder.int8.onnx`,
//!   `joiner.int8.onnx`, `tokens.txt`).
//! * `TALK_RS_TEST_PARAKEET_WAV` — absolute path to a real-speech
//!   WAV file (16 kHz mono recommended; sherpa auto-resamples).
//!   Defaults to `<model_dir>/test_wavs/fr.wav` when unset (the
//!   French sample bundled with the upstream tarball, which
//!   transcribes to "Ne vous demandez pas ce que votre pays peut
//!   faire pour vous. Demandez-vous plutôt ce que vous pouvez
//!   faire pour lui.").
//!
//! When either is unset and no fallback `test_wavs/fr.wav` exists,
//! the test prints a skip banner and returns successfully (no-op).
//!
//! Public-API path: drives the same `transcribe_audio` entry point
//! the CLI's `transcribe` command uses — no `pub(crate)` factory or
//! trait calls are needed.  We construct a minimal `Config` with
//! `providers.parakeet.model_dir` pointing at the env-supplied dir
//! and let `transcribe_audio` do the rest (cache lookup, `validate`
//! → `ensure_model`, inference).  The model files are already on
//! disk, so `ensure_model` short-circuits and no network I/O
//! happens during the test.

#![cfg(feature = "parakeet")]

use std::path::{Path, PathBuf};

use talk_rs::config::{Config, ParakeetConfig, ParakeetVariant, Provider, ProvidersConfig};
use talk_rs::transcription::{transcribe_audio, RequestTimeoutPolicy, TranscribeOptions};

/// Return `Some(path)` if env var is set to a non-empty value.
fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

#[tokio::test]
async fn parakeet_transcribes_real_french_speech() {
    let Some(model_dir) = env_path("TALK_RS_TEST_PARAKEET_MODEL_DIR") else {
        eprintln!(
            "SKIP: parakeet_transcribes_real_french_speech — \
             TALK_RS_TEST_PARAKEET_MODEL_DIR not set. \
             Point it at an extracted sherpa-onnx Parakeet INT8 model dir to enable."
        );
        return;
    };
    assert!(
        model_dir.is_dir(),
        "TALK_RS_TEST_PARAKEET_MODEL_DIR does not point at a directory: {}",
        model_dir.display()
    );
    // The model is NEVER auto-downloaded by the transcribe pipeline
    // (consent-gated — see `model::ensure_present`).  This test
    // therefore requires the model to be PRE-INSTALLED at the env dir;
    // point it at an already-extracted INT8 model.  We assert the
    // encoder file exists to fail fast with a clear message rather
    // than surfacing the "model not found / consent required" error
    // from deep in `transcribe_audio`.
    assert!(
        model_dir.join("encoder.int8.onnx").is_file(),
        "TALK_RS_TEST_PARAKEET_MODEL_DIR must contain a pre-installed \
         INT8 model (encoder.int8.onnx missing in {}). The transcribe \
         pipeline does not auto-download; install the model first.",
        model_dir.display()
    );

    let wav = env_path("TALK_RS_TEST_PARAKEET_WAV")
        .or_else(|| {
            let fallback = model_dir.join("test_wavs").join("fr.wav");
            fallback.is_file().then_some(fallback)
        })
        .unwrap_or_else(|| {
            panic!(
                "TALK_RS_TEST_PARAKEET_WAV not set and no bundled \
                 <model_dir>/test_wavs/fr.wav fallback found under {}",
                model_dir.display()
            )
        });
    assert!(
        wav.is_file(),
        "speech fixture not a regular file: {}",
        wav.display()
    );

    // Stage the WAV in a tempdir so the transcription sidecar cache
    // (written next to the audio file by `transcribe_audio`) does
    // not leak into the shared model dir or repo tree.
    let tmp = tempfile::TempDir::new().expect("create tempdir");
    let staged = tmp.path().join("fixture.wav");
    std::fs::copy(&wav, &staged).expect("copy WAV fixture into tempdir");

    let cfg = build_parakeet_config(tmp.path(), &model_dir);

    let sink: std::sync::Arc<dyn talk_rs::telemetry::TelemetrySink> =
        std::sync::Arc::new(talk_rs::telemetry::NoOpSink);

    let result = transcribe_audio(
        &staged,
        &cfg,
        Provider::Parakeet,
        None,
        false,
        TranscribeOptions {
            allow_api: true,
            policy: RequestTimeoutPolicy::Proportional,
            cancel_token: None,
            skip_legacy_lock: false,
        },
        &sink,
    )
    .await
    .expect("parakeet transcription must succeed");

    let text = result.text.trim().to_string();
    eprintln!("[parakeet] transcribed text: {:?}", text);

    assert!(
        !text.is_empty(),
        "transcription returned empty text — audio fixture may be silent or model misloaded"
    );

    // Stable substring assertions only — quantization / version drift
    // can shift punctuation and casing of the full sentence, but
    // these tokens are essential to the French phrase.
    let lower = text.to_lowercase();
    assert!(
        lower.contains("demandez"),
        "expected 'demandez' (case-insensitive) in transcript, got: {:?}",
        text
    );
    assert!(
        lower.contains("vous"),
        "expected 'vous' (case-insensitive) in transcript, got: {:?}",
        text
    );
}

/// Minimal `Config` with the parakeet provider wired to `model_dir`
/// and `output_dir` pointing at a throwaway tempdir.
fn build_parakeet_config(output_dir: &Path, model_dir: &Path) -> Config {
    Config {
        output_dir: output_dir.to_path_buf(),
        providers: ProvidersConfig {
            mistral: None,
            openai: None,
            parakeet: Some(ParakeetConfig {
                variant: ParakeetVariant::Int8,
                model_dir: Some(model_dir.to_path_buf()),
                num_threads: 2,
                model: None,
            }),
        },
        indicators: None,
        transcription: None,
        paste: None,
        audio: None,
        recording: None,
    }
}
