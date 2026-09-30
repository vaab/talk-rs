//! Items `upload-normalize-once` and `http-client-pool-prewarm` on the
//! file-transcription paths (`transcribe`, chain fallback).

use std::time::Duration;

use crate::require_counters;
use crate::support::fixtures::{self, OggProfile};
use crate::support::provider::{multipart_file, MockProvider, ProxyOptions, TRANSCRIPT};
use crate::support::runner::{self, Sandbox};
use crate::support::{config_yaml, Metrics, CHAIN_3};

const BUSY_BUSY_OK: [u16; 3] = [503, 503, 200];

/// `transcribe --chain perf` of a 2-minute `talk-rs record` OGG
/// (48 kHz, Audio profile) where the first two chain entries answer
/// 503.  Every entry re-normalizes the same file today.
#[tokio::test(flavor = "multi_thread")]
async fn perf_transcribe_fallback_chain_2min() {
    require_counters!();
    let sandbox = Sandbox::new();
    let provider =
        MockProvider::start(&BUSY_BUSY_OK, Duration::ZERO, ProxyOptions::default()).await;
    sandbox.write_config(&config_yaml(
        &sandbox.output_dir(),
        &provider.url(),
        &CHAIN_3,
    ));
    let input = sandbox.output_dir().join("2026-09-30T10-00-00+0200.ogg");
    fixtures::write_ogg(&input, 120.0, OggProfile::Recording);

    let input_arg = input.to_string_lossy().into_owned();
    let report = runner::run(
        sandbox.command(&["transcribe", "--chain", "perf", &input_arg]),
        &sandbox.log_path(),
        Duration::from_secs(600),
    )
    .await;
    report.assert_success();

    // Invariants: the third entry's transcript is delivered, every
    // entry uploaded the same 2-minute audio.
    assert_eq!(report.stdout.trim(), TRANSCRIPT);
    let posts = provider.posts().await;
    assert_eq!(posts.len(), 3);
    let uploads: Vec<Vec<u8>> = posts.iter().map(multipart_file).collect();
    for upload in &uploads {
        let seconds = fixtures::ogg_duration(upload);
        assert!((seconds - 120.0).abs() < 0.1, "upload is {seconds}s");
        assert_eq!(
            fixtures::ogg_without_serials(upload),
            fixtures::ogg_without_serials(&uploads[0]),
            "chain entries uploaded different audio"
        );
    }
    let times = provider.post_times();
    let first_busy_to_final_post = times[2].duration_since(times[0]).as_secs_f64() * 1000.0;

    Metrics::new("upload-normalize-once", "transcribe-fallback-chain-3-2min")
        .counters(&report, &["normalize_calls", "opus_frames_encoded"])
        .set("first_busy_to_final_post_ms", first_busy_to_final_post)
        .set("wall_ms", report.wall_ms)
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
/// pure waste when provenance is proven.
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
    let seconds = fixtures::ogg_duration(&upload);
    assert!((seconds - 120.0).abs() < 0.1, "upload is {seconds}s");
    let original = std::fs::read(&input).expect("input bytes");

    Metrics::new("upload-normalize-once", "transcribe-own-cache-ogg-2min")
        .counters(&report, &["normalize_calls", "opus_frames_encoded"])
        .set("upload_is_file_bytes", u8::from(upload == original))
        .set("wall_ms", report.wall_ms)
        .set("child_cpu_ms", report.child_cpu_ms)
        .write();
}

/// Normalization runs on a tokio worker today.  A 10 ms heartbeat on a
/// single-worker runtime measures how long the worker is blocked while
/// `transcribe_audio` prepares a 2-minute upload.
#[test]
fn perf_normalize_blocks_runtime_worker() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = dir.path().join("memo.ogg");
    fixtures::write_ogg(&input, 120.0, OggProfile::Recording);
    let root = dir.path().to_path_buf();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("runtime");
    let max_gap_ms = runtime.block_on(async {
        let validate_cache = root.join("validate-cache.yaml");
        // Only this test reads the variable in-process (children get
        // their own isolated value from the sandbox runner).
        std::env::set_var("TALK_RS_VALIDATE_CACHE_PATH", &validate_cache);
        let provider = MockProvider::start(&[200], Duration::ZERO, ProxyOptions::default()).await;
        let config_path = root.join("config.yaml");
        std::fs::write(&config_path, config_yaml(&root, &provider.url(), &[])).expect("config");
        let config = talk_rs::config::Config::load(Some(&config_path)).expect("config");
        let sink: std::sync::Arc<dyn talk_rs::telemetry::TelemetrySink> =
            std::sync::Arc::new(talk_rs::telemetry::NoOpSink);

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = std::sync::Arc::clone(&stop);
        let heartbeat = tokio::spawn(async move {
            let mut last = std::time::Instant::now();
            let mut max_gap = Duration::ZERO;
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_millis(10)).await;
                let now = std::time::Instant::now();
                max_gap = max_gap.max(now - last);
                last = now;
            }
            max_gap
        });
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
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        heartbeat.await.expect("heartbeat").as_secs_f64() * 1000.0
    });

    Metrics::new("upload-normalize-once", "runtime-heartbeat-2min")
        .set("heartbeat_max_gap_ms", max_gap_ms)
        .write();
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
    let seconds = fixtures::ogg_duration(&multipart_file(&posts[0]));
    assert!((seconds - 1200.0).abs() < 0.1, "upload is {seconds}s");
    let started = report.started_at.expect("spawn instant");
    let post_received_ms = provider
        .post_times()
        .first()
        .map_or(0.0, |t| t.duration_since(started).as_secs_f64() * 1000.0);

    Metrics::new("upload-normalize-once", "transcribe-long-20min")
        .counters(&report, &["normalize_calls", "opus_frames_encoded"])
        .set("post_received_after_start_ms", post_received_ms)
        .set("wall_ms", report.wall_ms)
        .set("child_cpu_ms", report.child_cpu_ms)
        .set("child_maxrss_kb", report.child_maxrss_kb as f64)
        .write();
}
