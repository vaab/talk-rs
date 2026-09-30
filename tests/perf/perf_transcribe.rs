//! Items `upload-normalize-once` and `http-client-pool-prewarm` on the
//! file-transcription paths (`transcribe`, chain fallback).
//!
//! Upload guards check the normalisation CONTRACT, not only that
//! attempts agree with each other: every upload must be a strict,
//! finalized 16 kHz mono Ogg Opus stream carrying the input's audio
//! (content envelope against the input decoded independently), so a
//! blind pass-through of a 48 kHz recording, a truncated or silent
//! upload, or a stale prepared artifact fails the cut.  Upload work is
//! counted where it happens (`upload_encodes`, `audio_file_decodes`),
//! after any cache decision.  Timing metrics are medians over
//! `TALK_RS_PERF_REPEAT` runs.

use std::time::Duration;

use crate::require_counters;
use crate::support::audio::{decode_ogg_strict, envelope_correlation};
use crate::support::fixtures::{self, OggProfile};
use crate::support::provider::{multipart_file, MockProvider, ProxyOptions, TRANSCRIPT};
use crate::support::runner::{self, RunReport, Sandbox};
use crate::support::{config_yaml, repeats, Metrics, CHAIN_3};

const BUSY_BUSY_OK: [u16; 3] = [503, 503, 200];

/// 16 kHz reference of an input file, decoded by the test itself.
fn reference_16k(path: &std::path::Path) -> Vec<i16> {
    let pcm48 = decode_ogg_strict(&std::fs::read(path).expect("input"), 48_000);
    pcm48.iter().step_by(3).copied().collect()
}

/// Every upload is the upload profile carrying `reference`.
fn assert_upload_profile(what: &str, upload: &[u8], reference: &[i16]) {
    let head = upload
        .windows(8)
        .position(|w| w == b"OpusHead")
        .expect("OpusHead");
    assert_eq!(upload[head + 9], 1, "{what}: mono");
    let decoded = decode_ogg_strict(upload, 16_000);
    assert!(
        decoded.len().abs_diff(reference.len()) <= 320,
        "{what}: {} vs {} samples",
        decoded.len(),
        reference.len()
    );
    let r = envelope_correlation(reference, &decoded, 16_000);
    assert!(r > 0.95, "{what}: not the input audio (r={r:.3})");
}

async fn transcribe_chain_2min() -> (RunReport, MockProvider, Vec<i16>) {
    let sandbox = Sandbox::new();
    let provider =
        MockProvider::start(&BUSY_BUSY_OK, Duration::ZERO, ProxyOptions::default()).await;
    sandbox.write_config(&config_yaml(
        &sandbox.output_dir(),
        &provider.url(),
        &CHAIN_3,
    ));
    let input = sandbox.output_dir().join("2026-09-30T10-00-00+0200.ogg");
    let cached = fixtures::cached_ogg("record-2min", 120.0, OggProfile::Recording);
    std::fs::copy(&cached, &input).expect("stage input");
    let reference = reference_16k(&input);
    let input_arg = input.to_string_lossy().into_owned();
    let report = runner::run(
        sandbox.command(&["transcribe", "--chain", "perf", &input_arg]),
        &sandbox.log_path(),
        Duration::from_secs(600),
    )
    .await;
    report.assert_success();
    drop(sandbox);
    (report, provider, reference)
}

/// `transcribe --chain perf` of a 2-minute `talk-rs record` OGG
/// (48 kHz, Audio profile) where the first two chain entries answer
/// 503.  Every entry re-normalizes the same file today.
#[tokio::test(flavor = "multi_thread")]
async fn perf_transcribe_fallback_chain_2min() {
    require_counters!();
    let mut busy_to_final = Vec::new();
    let mut walls = Vec::new();
    let mut first = None;
    for _ in 0..repeats() {
        let (report, provider, reference) = transcribe_chain_2min().await;
        assert_eq!(report.stdout.trim(), TRANSCRIPT);
        let posts = provider.posts().await;
        assert_eq!(posts.len(), 3);
        for (i, post) in posts.iter().enumerate() {
            assert_upload_profile(&format!("entry {i}"), &multipart_file(post), &reference);
        }
        let times = provider.post_times();
        busy_to_final.push(times[2].duration_since(times[0]).as_secs_f64() * 1000.0);
        walls.push(report.wall_ms);
        if first.is_none() {
            first = Some((report, provider));
        }
    }
    let (report, provider) = first.expect("one run");
    Metrics::new("upload-normalize-once", "transcribe-fallback-chain-3-2min")
        .counters(
            &report,
            &[
                "normalize_calls",
                "upload_encodes",
                "audio_file_decodes",
                "opus_frames_encoded",
            ],
        )
        .timing("first_busy_to_final_post_ms", &busy_to_final)
        .timing("wall_ms", &walls)
        .set("child_cpu_ms", report.child_cpu_ms)
        .set("child_maxrss_kb", report.child_maxrss_kb as f64)
        .write();
    Metrics::new(
        "http-client-pool-prewarm",
        "transcribe-fallback-chain-3-2min",
    )
    .counters(&report, &["http_client_builds"])
    .set(
        "requests",
        provider
            .server
            .received_requests()
            .await
            .unwrap_or_default()
            .len() as f64,
    )
    .set("tcp_connections", provider.proxy.accepted() as f64)
    .write();
}

/// `transcribe` of a file that talk-rs itself wrote into its dictation
/// cache (16 kHz mono Voip, the exact upload profile): the re-encode is
/// pure waste when provenance is proven.  The upload must still be the
/// complete recording.
#[tokio::test(flavor = "multi_thread")]
async fn perf_transcribe_own_cache_recording_2min() {
    require_counters!();
    let sandbox = Sandbox::new();
    let provider = MockProvider::start(&[200], Duration::ZERO, ProxyOptions::default()).await;
    sandbox.write_config(&config_yaml(&sandbox.output_dir(), &provider.url(), &[]));
    let recordings = sandbox.cache_dir().join("recordings");
    std::fs::create_dir_all(&recordings).expect("cache recordings dir");
    let input = recordings.join("2026-09-30T10-00-00+0200.ogg");
    fixtures::write_ogg(&input, 120.0, OggProfile::Transcription);
    let reference = decode_ogg_strict(&std::fs::read(&input).expect("input"), 16_000);

    let input_arg = input.to_string_lossy().into_owned();
    let report = runner::run(
        sandbox.command(&["transcribe", "--provider", "mistral", &input_arg]),
        &sandbox.log_path(),
        Duration::from_secs(600),
    )
    .await;
    report.assert_success();
    assert_eq!(report.stdout.trim(), TRANSCRIPT);
    let posts = provider.posts().await;
    assert_eq!(posts.len(), 1);
    let upload = multipart_file(&posts[0]);
    assert_upload_profile("own cache", &upload, &reference);
    let original = std::fs::read(&input).expect("input bytes");

    Metrics::new("upload-normalize-once", "transcribe-own-cache-ogg-2min")
        .counters(
            &report,
            &["normalize_calls", "upload_encodes", "audio_file_decodes"],
        )
        .set("upload_is_file_bytes", u8::from(upload == original))
        .set("child_cpu_ms", report.child_cpu_ms)
        .write();
}

/// Negative provenance: a 48 kHz `record`-profile OGG placed in the
/// dictation cache directory with a cache-style name.  Location and
/// extension prove nothing: it must still be converted.
#[tokio::test(flavor = "multi_thread")]
async fn perf_transcribe_foreign_ogg_in_cache_dir_is_converted() {
    require_counters!();
    let sandbox = Sandbox::new();
    let provider = MockProvider::start(&[200], Duration::ZERO, ProxyOptions::default()).await;
    sandbox.write_config(&config_yaml(&sandbox.output_dir(), &provider.url(), &[]));
    let recordings = sandbox.cache_dir().join("recordings");
    std::fs::create_dir_all(&recordings).expect("cache recordings dir");
    let input = recordings.join("2026-09-30T11-00-00+0200.ogg");
    fixtures::write_ogg(&input, 20.0, OggProfile::Recording);
    let reference = reference_16k(&input);
    let input_arg = input.to_string_lossy().into_owned();
    let report = runner::run(
        sandbox.command(&["transcribe", "--provider", "mistral", &input_arg]),
        &sandbox.log_path(),
        Duration::from_secs(600),
    )
    .await;
    report.assert_success();
    let upload = multipart_file(&provider.posts().await[0]);
    assert_ne!(
        upload,
        std::fs::read(&input).expect("input"),
        "48 kHz input uploaded as is"
    );
    assert_upload_profile("foreign ogg", &upload, &reference);
}

/// Environment marker selecting the heartbeat worker role of this test
/// binary (see `perf_normalize_blocks_runtime_worker`).
const HEARTBEAT_WORKER: &str = "TALK_RS_PERF_HEARTBEAT_WORKER";

/// Normalization runs on a tokio worker today.  A 10 ms heartbeat on a
/// single-worker runtime measures how long the worker is blocked while
/// `transcribe_audio` prepares a 2-minute upload.
///
/// Isolation: the measurement runs in a child process of this test
/// binary with a CLEARED environment (no inherited provider keys/URLs,
/// private HOME/XDG, private validate cache, egress blocked), and the
/// config is deserialised directly — `Config::load` would apply
/// `TALK_RS_*` environment overrides.  The heartbeat starts, and is
/// observed ticking, before the transcription is spawned.
#[test]
fn perf_normalize_blocks_runtime_worker() {
    if std::env::var_os(HEARTBEAT_WORKER).is_some() {
        heartbeat_worker();
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let result = dir.path().join("gap-ms");
    let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "perf_transcribe::perf_normalize_blocks_runtime_worker",
            "--nocapture",
        ])
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", dir.path())
        .env("XDG_CONFIG_HOME", dir.path().join("config"))
        .env("XDG_CACHE_HOME", dir.path().join("cache"))
        .env("XDG_DATA_HOME", dir.path().join("data"))
        .env(
            "TALK_RS_VALIDATE_CACHE_PATH",
            dir.path().join("validate-cache.yaml"),
        )
        .env("HTTP_PROXY", "http://127.0.0.1:9")
        .env("HTTPS_PROXY", "http://127.0.0.1:9")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env(HEARTBEAT_WORKER, &result)
        .status()
        .expect("spawn heartbeat worker");
    assert!(status.success(), "heartbeat worker failed");
    let max_gap_ms: f64 = std::fs::read_to_string(&result)
        .expect("worker result")
        .trim()
        .parse()
        .expect("gap value");
    Metrics::new("upload-normalize-once", "runtime-heartbeat-2min")
        .set("heartbeat_max_gap_ms", max_gap_ms)
        .write();
}

fn heartbeat_worker() {
    let result = std::path::PathBuf::from(std::env::var_os(HEARTBEAT_WORKER).expect("marker"));
    let root = result.parent().expect("worker dir").to_path_buf();
    let input = root.join("memo.ogg");
    fixtures::write_ogg(&input, 120.0, OggProfile::Recording);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("runtime");
    let max_gap_ms = runtime.block_on(async {
        let provider = MockProvider::start(&[200], Duration::ZERO, ProxyOptions::default()).await;
        // Deserialised directly: no environment overrides can apply.
        let config: talk_rs::config::Config =
            serde_yaml::from_str(&config_yaml(&root, &provider.url(), &[])).expect("config");
        let sink: std::sync::Arc<dyn talk_rs::telemetry::TelemetrySink> =
            std::sync::Arc::new(talk_rs::telemetry::NoOpSink);

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ticks = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (flag, count) = (std::sync::Arc::clone(&stop), std::sync::Arc::clone(&ticks));
        let heartbeat = tokio::spawn(async move {
            let mut last = std::time::Instant::now();
            let mut max_gap = Duration::ZERO;
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_millis(10)).await;
                let now = std::time::Instant::now();
                max_gap = max_gap.max(now - last);
                last = now;
                count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            max_gap
        });
        // Synchronised start: the heartbeat must be ticking first.
        while ticks.load(std::sync::atomic::Ordering::Relaxed) < 5 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        // Spawned, so it competes with the heartbeat for the single
        // worker thread (as provider calls do inside talk-rs).
        let result = tokio::spawn(async move {
            talk_rs::transcription::transcribe_audio(
                &input,
                &config,
                talk_rs::config::Provider::Mistral,
                None,
                false,
                talk_rs::transcription::TranscribeOptions {
                    allow_api: true,
                    ..Default::default()
                },
                &sink,
            )
            .await
        })
        .await
        .expect("transcription task")
        .expect("transcription");
        assert_eq!(result.text, TRANSCRIPT);
        assert_eq!(
            provider.posts().await.len(),
            1,
            "the mock provider was used"
        );
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        heartbeat.await.expect("heartbeat").as_secs_f64() * 1000.0
    });
    std::fs::write(&result, format!("{max_gap_ms}\n")).expect("write worker result");
}

/// `transcribe` of a 20-minute recording: absolute cost of preparing a
/// long upload (decode + re-encode) for one provider attempt.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "long: 20-minute fixture"]
async fn perf_transcribe_long_20min() {
    require_counters!();
    let sandbox = Sandbox::new();
    let provider = MockProvider::start(&[200], Duration::ZERO, ProxyOptions::default()).await;
    sandbox.write_config(&config_yaml(&sandbox.output_dir(), &provider.url(), &[]));
    // Hard-link the shared cached fixture into the sandbox: talk-rs
    // writes transcription sidecars next to its input, which must not
    // land beside the shared copy (the next run would be a cache hit).
    let input = sandbox.output_dir().join("2026-09-30T10-00-00+0200.ogg");
    let cached = fixtures::cached_ogg("record-20min", 1200.0, OggProfile::Recording);
    std::fs::hard_link(&cached, &input)
        .or_else(|_| std::fs::copy(&cached, &input).map(|_| ()))
        .expect("stage 20 min fixture");
    let input_arg = input.to_string_lossy().into_owned();
    let report = runner::run(
        sandbox.command(&["transcribe", "--provider", "mistral", &input_arg]),
        &sandbox.log_path(),
        Duration::from_secs(1800),
    )
    .await;
    report.assert_success();
    assert_eq!(report.stdout.trim(), TRANSCRIPT);
    let posts = provider.posts().await;
    let decoded = decode_ogg_strict(&multipart_file(&posts[0]), 16_000);
    let seconds = decoded.len() as f64 / 16_000.0;
    assert!((seconds - 1200.0).abs() < 0.1, "upload is {seconds}s");
    let started = report.started_at.expect("spawn instant");
    let post_received_ms = provider
        .post_times()
        .first()
        .map_or(0.0, |t| t.duration_since(started).as_secs_f64() * 1000.0);

    Metrics::new("upload-normalize-once", "transcribe-long-20min")
        .counters(
            &report,
            &["normalize_calls", "upload_encodes", "audio_file_decodes"],
        )
        .set("post_received_after_start_ms", post_received_ms)
        .set("wall_ms", report.wall_ms)
        .set("child_cpu_ms", report.child_cpu_ms)
        .set("child_maxrss_kb", report.child_maxrss_kb as f64)
        .write();
}
