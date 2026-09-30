//! Local-model cuts: `parakeet-warm` and `speak-streaming-playback`.
//!
//! Gated on the same variables as the existing model integration tests
//! (`TALK_RS_TEST_PARAKEET_MODEL_DIR`, `TALK_RS_TEST_KOKORO_MODEL_DIR`);
//! models are never downloaded.  Both run the real binary.

use std::time::Duration;

use crate::require_counters;
use crate::support::provider::{MockProvider, ProxyOptions};
use crate::support::runner::{self, Sandbox};
use crate::support::{gate, Metrics};

fn parakeet_config(
    output: &std::path::Path,
    model_dir: &str,
    cloud_url: &str,
    chain: bool,
) -> String {
    let mut yaml = format!(
        "output_dir: {}\nproviders:\n  mistral:\n    api_key: fake\n    url: {cloud_url}\n    model: voxtral-mini-2602\n  parakeet:\n    model_dir: {model_dir}\n    num_threads: 2\nindicators:\n  boop_interval_ms: 0\n  visual_overlay: false\n",
        output.display()
    );
    if chain {
        yaml.push_str(
            "transcription:\n  chains:\n    perf:\n      - mistral/voxtral-mini-2602\n      - parakeet\n",
        );
    }
    yaml
}

async fn parakeet_dictation(cut: &'static str, statuses: &[u16], chain: bool) {
    let Some(model_dir) = gate("TALK_RS_TEST_PARAKEET_MODEL_DIR") else {
        return;
    };
    let sandbox = Sandbox::new();
    let provider = MockProvider::start(statuses, Duration::ZERO, ProxyOptions::default()).await;
    sandbox.write_config(&parakeet_config(
        &sandbox.output_dir(),
        &model_dir,
        &provider.url(),
        chain,
    ));
    let speech = std::path::Path::new(&model_dir).join("test_wavs/en.wav");
    // Replayed at real time; en.wav is 24 kHz mono, the paced source
    // takes its native rate and dictate resamples to 16 kHz.
    let mut args = vec![
        "dictate",
        "--no-sounds",
        "--no-overlay",
        "--no-paste",
        "--no-bt-auto-switch",
        "--target-window",
        "0",
    ];
    if chain {
        args.extend(["--chain", "perf"]);
    } else {
        args.extend(["--provider", "parakeet"]);
    }
    let mut cmd = sandbox.command(&args);
    cmd.env("TALK_RS_PERF_PACED_INPUT", &speech);
    let (report, _) = runner::run_with_sigint(
        cmd,
        &sandbox.log_path(),
        "recording — waiting for shutdown signal",
        Duration::from_secs(3),
        Duration::from_secs(300),
    )
    .await;
    report.assert_success();
    assert!(!report.stdout.trim().is_empty(), "no transcript delivered");
    let mut m = Metrics::new("parakeet-warm", cut);
    m.counters(&report, &["recognizer_creates"])
        .set(
            "stop_transcription_done_ms",
            report.stop_ms("transcription_done") as f64,
        )
        .set("child_maxrss_kb", report.child_maxrss_kb as f64)
        .set("child_cpu_ms", report.child_cpu_ms)
        .write();
}

/// Parakeet as the primary provider, 3 s live dictation.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs TALK_RS_TEST_PARAKEET_MODEL_DIR"]
async fn perf_parakeet_primary_3s() {
    require_counters!();
    parakeet_dictation("parakeet-primary-3s", &[200], false).await;
}

/// Chain: cloud answers 503 twice (live + nothing else), Parakeet last.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs TALK_RS_TEST_PARAKEET_MODEL_DIR"]
async fn perf_parakeet_chain_fallback_3s() {
    require_counters!();
    parakeet_dictation("chain-cloud-busy-parakeet-last", &[503], true).await;
}

/// Chain where the cloud succeeds: Parakeet must cost nothing
/// (`recognizer_creates == 0`, peak RSS unchanged).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs TALK_RS_TEST_PARAKEET_MODEL_DIR"]
async fn perf_parakeet_chain_cloud_ok_3s() {
    require_counters!();
    parakeet_dictation("chain-cloud-ok", &[200], true).await;
}

const PARAGRAPH: &str = "Speech synthesis turns written text into audio. \
    This paragraph is long enough to take several seconds to synthesize, \
    which is exactly the situation where starting playback early matters. \
    A listener should hear the first sentence while the rest is still \
    being generated, instead of waiting in silence for the whole text. \
    The measurement records when synthesis starts, when the first audio \
    reaches the player, and when synthesis finishes. \
    Saving to a file must keep producing exactly the same samples.";

/// `speak` of a multi-sentence paragraph with Kokoro: time to first
/// audio versus total synthesis, and `-o` output identity.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs TALK_RS_TEST_KOKORO_MODEL_DIR"]
async fn perf_speak_kokoro_paragraph() {
    require_counters!();
    let Some(model_dir) = gate("TALK_RS_TEST_KOKORO_MODEL_DIR") else {
        return;
    };
    let config = |sandbox: &Sandbox| {
        format!(
            "output_dir: {}\nproviders:\n  kokoro:\n    model_dir: {model_dir}\n    num_threads: 2\n",
            sandbox.output_dir().display()
        )
    };

    // Play to the (null) output device: first-audio mark.
    let sandbox = Sandbox::new();
    sandbox.write_config(&config(&sandbox));
    let played = runner::run(
        sandbox.command(&["speak", "--provider", "kokoro", "--lang", "en", PARAGRAPH]),
        &sandbox.log_path(),
        Duration::from_secs(600),
    )
    .await;
    played.assert_success();
    let start = played.marks["speak_synthesis_start"] as f64;
    let first_audio = played.marks["speak_first_audio"] as f64 - start;
    let synthesis = played.marks["speak_synthesis_done"] as f64 - start;

    // Save to a file: the WAV must stay complete.  Kokoro is not
    // bit-deterministic across runs (a few samples of jitter), so the
    // guard is the audio duration, not a hash.
    let out = sandbox.path().join("a.wav");
    let out_arg = out.to_string_lossy().into_owned();
    let saved = runner::run(
        sandbox.command(&[
            "speak",
            "--provider",
            "kokoro",
            "--lang",
            "en",
            "-o",
            &out_arg,
            PARAGRAPH,
        ]),
        &sandbox.log_path(),
        Duration::from_secs(600),
    )
    .await;
    saved.assert_success();
    let wav = std::fs::read(&out).expect("speak -o output");
    assert!(
        wav.len() > 44 + 24_000 * 2 * 5,
        "paragraph shorter than 5 s"
    );

    Metrics::new("speak-streaming-playback", "kokoro-paragraph")
        .counters(&played, &["tts_creates"])
        .set("first_audio_ms", first_audio)
        .set("synthesis_ms", synthesis)
        .set("first_audio_ratio", first_audio / synthesis.max(1.0))
        .set("wav_seconds", (wav.len() - 44) as f64 / 2.0 / 24_000.0)
        .set("child_maxrss_kb", played.child_maxrss_kb as f64)
        .write();
}
