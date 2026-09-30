//! Local-model cuts: `parakeet-warm` and `speak-streaming-playback`.
//!
//! Gated on the same variables as the existing model integration tests
//! (`TALK_RS_TEST_PARAKEET_MODEL_DIR`, `TALK_RS_TEST_KOKORO_MODEL_DIR`).
//! The directories are validated with the production presence checks
//! before anything runs, network egress is blocked by the sandbox, and
//! each cut asserts the model tree was left unchanged: models are never
//! downloaded or modified by a measurement.

use std::time::Duration;

use crate::require_counters;
use crate::support::provider::{MockProvider, ProxyOptions};
use crate::support::runner::{self, RunReport, Sandbox};
use crate::support::{model_gate, Metrics, ModelDir};
use talk_rs::perf_counters::LocalModel;

fn parakeet_config(
    output: &std::path::Path,
    model: &ModelDir,
    cloud_url: &str,
    chain: bool,
) -> String {
    let mut yaml = format!(
        "output_dir: {}\nproviders:\n  mistral:\n    api_key: fake\n    url: {cloud_url}\n    model: voxtral-mini-2602\n  parakeet:\n    model_dir: {}\n    num_threads: 2\nindicators:\n  boop_interval_ms: 0\n  visual_overlay: false\n",
        output.display(),
        model.path()
    );
    if chain {
        yaml.push_str(
            "transcription:\n  chains:\n    perf:\n      - mistral/voxtral-mini-2602\n      - parakeet\n",
        );
    }
    yaml
}

/// Reference transcript of the model's bundled `test_wavs/en.wav`
/// (checked word-wise, case- and punctuation-insensitive).
const EN_REFERENCE: &str =
    "ask not what your country can do for you ask what you can do for your country";

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// Word recall of `got` against `reference` (order-insensitive).
fn word_recall(reference: &str, got: &str) -> f64 {
    let got = words(got);
    let reference = words(reference);
    let hits = reference.iter().filter(|w| got.contains(w)).count();
    hits as f64 / reference.len().max(1) as f64
}

async fn parakeet_dictation(
    model: &ModelDir,
    statuses: &[u16],
    chain: bool,
) -> (RunReport, MockProvider) {
    let sandbox = Sandbox::new();
    let provider = MockProvider::start(statuses, Duration::ZERO, ProxyOptions::default()).await;
    sandbox.write_config(&parakeet_config(
        &sandbox.output_dir(),
        model,
        &provider.url(),
        chain,
    ));
    let speech = model.dir.join("test_wavs/en.wav");
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
    // en.wav (3.85 s at 24 kHz) replayed at real-time pace, stopped
    // just after its end.
    cmd.env("TALK_RS_PERF_PACED_INPUT", &speech);
    let (report, _) = runner::run_with_sigint(
        cmd,
        &sandbox.log_path(),
        "recording — waiting for shutdown signal",
        Duration::from_millis(3_900),
        Duration::from_secs(300),
    )
    .await;
    report.assert_success();
    model.assert_unchanged();
    (report, provider)
}

fn parakeet_metrics(cut: &'static str, report: &RunReport) {
    let mark = |n: &str| report.marks.get(n).map(|&v| v as f64);
    let mut m = Metrics::new("parakeet-warm", cut);
    m.counters(report, &["recognizer_creates"])
        .set(
            "stop_transcription_done_ms",
            report.stop_ms("transcription_done") as f64,
        )
        .set(
            "start_capture_started_ms",
            report.start.get("capture_started").copied().unwrap_or(0) as f64,
        )
        .set("child_maxrss_kb", report.child_maxrss_kb as f64)
        .set("child_cpu_ms", report.child_cpu_ms);
    if let (Some(init), Some(ready), Some(decoded)) = (
        mark("parakeet_init_start"),
        mark("parakeet_ready"),
        mark("parakeet_decoded"),
    ) {
        m.set("recognizer_init_ms", ready - init)
            .set("inference_ms", decoded - ready);
    }
    m.write();
}

/// Parakeet as the primary provider, 3.9 s live dictation of real
/// speech: the transcript must be the reference sentence.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs TALK_RS_TEST_PARAKEET_MODEL_DIR"]
async fn perf_parakeet_primary_3s() {
    require_counters!();
    let Some(model) = model_gate("TALK_RS_TEST_PARAKEET_MODEL_DIR", LocalModel::Parakeet) else {
        return;
    };
    let (report, _) = parakeet_dictation(&model, &[200], false).await;
    let recall = word_recall(EN_REFERENCE, &report.stdout);
    assert!(
        recall >= 0.8,
        "wrong transcript ({recall:.2}): {:?}",
        report.stdout
    );
    assert_eq!(
        report.counter("recognizer_creates"),
        1,
        "one recognizer per dictation"
    );
    parakeet_metrics("parakeet-primary-3s", &report);
}

/// Chain: the cloud answers 503 to the live upload, Parakeet (last)
/// transcribes the cached recording.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs TALK_RS_TEST_PARAKEET_MODEL_DIR"]
async fn perf_parakeet_chain_fallback_3s() {
    require_counters!();
    let Some(model) = model_gate("TALK_RS_TEST_PARAKEET_MODEL_DIR", LocalModel::Parakeet) else {
        return;
    };
    let (report, provider) = parakeet_dictation(&model, &[503], true).await;
    let recall = word_recall(EN_REFERENCE, &report.stdout);
    assert!(
        recall >= 0.8,
        "wrong transcript ({recall:.2}): {:?}",
        report.stdout
    );
    assert_eq!(provider.posts().await.len(), 1, "one paid cloud attempt");
    assert_eq!(report.counter("recognizer_creates"), 1);
    parakeet_metrics("chain-cloud-busy-parakeet-last", &report);
}

/// Chain where the cloud succeeds: Parakeet must cost nothing.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs TALK_RS_TEST_PARAKEET_MODEL_DIR"]
async fn perf_parakeet_chain_cloud_ok_3s() {
    require_counters!();
    let Some(model) = model_gate("TALK_RS_TEST_PARAKEET_MODEL_DIR", LocalModel::Parakeet) else {
        return;
    };
    let (report, _) = parakeet_dictation(&model, &[200], true).await;
    assert_eq!(report.stdout.trim(), crate::support::provider::TRANSCRIPT);
    parakeet_metrics("chain-cloud-ok", &report);
}

const PARAGRAPH: &str = "Speech synthesis turns written text into audio. \
    This paragraph is long enough to take several seconds to synthesize, \
    which is exactly the situation where starting playback early matters. \
    A listener should hear the first sentence while the rest is still \
    being generated, instead of waiting in silence for the whole text. \
    The measurement records when synthesis starts, when the first audio \
    reaches the player, and when synthesis finishes. \
    Saving to a file must keep producing exactly the same samples.";

/// Parse a mono 16-bit PCM WAV (validating its header); returns the
/// samples as f32 and the sample rate.
fn read_wav(bytes: &[u8]) -> (Vec<f32>, u32) {
    assert!(
        bytes.len() > 44 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a WAV"
    );
    let mut pos = 12;
    let (mut rate, mut channels, mut bits) = (0u32, 0u16, 0u16);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().expect("size")) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        if id == b"fmt " {
            channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes(body[4..8].try_into().expect("rate"));
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            assert_eq!((channels, bits), (1, 16), "mono 16-bit PCM");
            assert_eq!(size, body.len(), "data chunk complete");
            let pcm = body
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
            return (pcm, rate);
        }
        pos += 8 + size + (size & 1);
    }
    panic!("WAV without data chunk");
}

/// 10 ms RMS envelope of `pcm` at `rate`.
fn envelope(pcm: &[f32], rate: u32) -> Vec<f64> {
    pcm.chunks((rate / 100) as usize)
        .map(|c| (c.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / c.len() as f64).sqrt())
        .collect()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (ma, mb) = (
        a.iter().sum::<f64>() / n as f64,
        b.iter().sum::<f64>() / n as f64,
    );
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        c += (a[i] - ma) * (b[i] - mb);
        va += (a[i] - ma).powi(2);
        vb += (b[i] - mb).powi(2);
    }
    c / (va.sqrt() * vb.sqrt()).max(1e-12)
}

/// `speak` of a paragraph with Kokoro, played through the capturing
/// sink: time from synthesis start to the first MEANINGFUL sample the
/// output consumed, and the complete played audio compared with the
/// same text saved with `-o` (content envelope; Kokoro is not bit-
/// deterministic across runs).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs TALK_RS_TEST_KOKORO_MODEL_DIR"]
async fn perf_speak_kokoro_paragraph() {
    require_counters!();
    let Some(model) = model_gate("TALK_RS_TEST_KOKORO_MODEL_DIR", LocalModel::Kokoro) else {
        return;
    };
    let sandbox = Sandbox::new();
    sandbox.write_config(&format!(
        "output_dir: {}\nproviders:\n  kokoro:\n    model_dir: {}\n    num_threads: 2\n",
        sandbox.output_dir().display(),
        model.path()
    ));

    let sink = sandbox.path().join("sink.log");
    let mut cmd = sandbox.command(&["speak", "--provider", "kokoro", "--lang", "en", PARAGRAPH]);
    cmd.env("TALK_RS_PERF_AUDIO_SINK", &sink);
    let t_spawn_epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_millis();
    let played = runner::run(cmd, &sandbox.log_path(), Duration::from_secs(600)).await;
    played.assert_success();
    let events = std::fs::read_to_string(&sink).expect("sink events");
    let first = events
        .lines()
        .find_map(|l| l.strip_prefix("first-audio "))
        .expect("sound reached the output");
    let field = |k: &str| {
        first
            .split_whitespace()
            .find_map(|kv| kv.strip_prefix(k))
            .expect("first-audio field")
            .to_string()
    };
    let first_epoch: u128 = field("epoch=").parse().expect("epoch");
    let consumed: u64 = events
        .lines()
        .filter_map(|l| l.strip_prefix("consumed "))
        .filter_map(|v| v.parse().ok())
        .max()
        .expect("consumed count");
    let played_pcm: Vec<f32> = std::fs::read(sink.with_extension("pcm"))
        .expect("sink pcm")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(played_pcm.len() as u64, consumed, "sink bookkeeping");

    let start = played.marks["speak_synthesis_start"] as f64;
    let synthesis_ms = played.marks["speak_synthesis_done"] as f64 - start;
    let handoff_ms = played.marks["speak_handoff"] as f64 - start;
    // The sink's wall clock and the child's mark clock share the spawn
    // origin within a few ms: express first sound on the mark clock.
    let first_sound_ms = first_epoch.saturating_sub(t_spawn_epoch) as f64 - start;

    // Save the same text with -o: complete, mono, 24 kHz WAV whose
    // content matches what was played (whole paragraph, full drain).
    let out = sandbox.path().join("out.wav");
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
    model.assert_unchanged();
    let (saved_pcm, rate) = read_wav(&std::fs::read(&out).expect("speak -o output"));
    let saved_s = saved_pcm.len() as f64 / rate as f64;
    let played_s = played_pcm.len() as f64 / 48_000.0;
    assert!(
        saved_s > 15.0,
        "paragraph shorter than expected: {saved_s:.1}s"
    );
    assert!(
        (played_s - saved_s).abs() / saved_s < 0.05,
        "played {played_s:.2}s vs saved {saved_s:.2}s: playback incomplete"
    );
    let r = correlation(&envelope(&played_pcm, 48_000), &envelope(&saved_pcm, rate));
    assert!(
        r > 0.8,
        "played audio does not match the synthesized paragraph (r={r:.3})"
    );

    Metrics::new("speak-streaming-playback", "kokoro-paragraph")
        .counters(&played, &["tts_creates"])
        .set("first_sound_ms", first_sound_ms)
        .set("handoff_ms", handoff_ms)
        .set("synthesis_ms", synthesis_ms)
        .set("first_sound_ratio", first_sound_ms / synthesis_ms.max(1.0))
        .set("played_seconds", played_s)
        .set("saved_seconds", saved_s)
        .set("played_vs_saved_envelope_r", r)
        .set("child_maxrss_kb", played.child_maxrss_kb as f64)
        .write();
}
