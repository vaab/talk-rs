//! Live one-shot dictation cuts, driven through the real binary with
//! the paced capture seam (`TALK_RS_PERF_PACED_INPUT`): the recording
//! runs at real-time pace and is stopped by SIGINT, exactly like the
//! toggle shortcut.  Covers items `http-client-pool-prewarm`,
//! `single-opus-encode`, `upload-normalize-once` (fallback) and the
//! overlay's FFT work.
//!
//! Every cut checks what the user gets, not only the work done: the
//! provider's transcript is printed, the cache OGG is a well-formed,
//! finalized stream that decodes to the dictated audio, and every
//! upload decodes to that same audio (content envelope against the
//! paced reference).  Timing metrics are medians over
//! `TALK_RS_PERF_REPEAT` runs (default 5).

use std::time::{Duration, Instant};

use crate::require_counters;
use crate::support::audio::{decode_ogg_strict, envelope_correlation};
use crate::support::fixtures;
use crate::support::provider::{multipart_file, MockProvider, ProxyOptions, TRANSCRIPT};
use crate::support::runner::{self, RunReport, Sandbox};
use crate::support::{config_yaml, repeats, Metrics, CHAIN_3};

/// Opus frames per second of 16 kHz audio (20 ms frames).
const FRAMES_PER_SEC: f64 = 50.0;
/// The paced input loops this 4 s speech-like clip.
const LOOP_SECONDS: f64 = 4.0;

struct Dictation {
    report: RunReport,
    provider: MockProvider,
    sandbox: Sandbox,
    sigint_at: Instant,
    /// SIGINT → the mock provider received the (first) POST.
    sigint_to_first_post_ms: f64,
}

async fn dictate(seconds: f64, statuses: &[u16], chain: &[&str], proxy: ProxyOptions) -> Dictation {
    let sandbox = Sandbox::new();
    let provider = MockProvider::start(statuses, Duration::ZERO, proxy).await;
    sandbox.write_config(&config_yaml(&sandbox.output_dir(), &provider.url(), chain));
    let input = sandbox.path().join("speech.wav");
    fixtures::write_wav(&input, &fixtures::speech_pcm(LOOP_SECONDS, 16_000), 16_000);
    let mut args = vec![
        "dictate",
        "--no-sounds",
        "--no-overlay",
        "--no-paste",
        "--no-bt-auto-switch",
        "--target-window",
        "0",
    ];
    if !chain.is_empty() {
        args.extend(["--chain", "perf"]);
    }
    let mut cmd = sandbox.command(&args);
    cmd.env("TALK_RS_PERF_PACED_INPUT", &input);
    let (report, sigint_at) = runner::run_with_sigint(
        cmd,
        &sandbox.log_path(),
        "recording — waiting for shutdown signal",
        Duration::from_secs_f64(seconds),
        Duration::from_secs(300),
    )
    .await;
    let first_post = provider.post_times().first().copied();
    let sigint_to_first_post_ms = first_post.map_or(f64::NAN, |t| {
        t.duration_since(sigint_at).as_secs_f64() * 1000.0
    });
    Dictation {
        report,
        provider,
        sandbox,
        sigint_at,
        sigint_to_first_post_ms,
    }
}

/// The paced source's audio for a recording of `samples` samples.
fn looped_reference(samples: usize) -> Vec<i16> {
    let clip = fixtures::speech_pcm(LOOP_SECONDS, 16_000);
    (0..samples).map(|i| clip[i % clip.len()]).collect()
}

/// Invariants every dictation cut checks: success, the provider's
/// transcript printed, a complete cache recording of the dictated
/// length (± 1 s of SIGINT jitter) holding the paced speech, and every
/// upload carrying that same recording.  Returns (recorded seconds,
/// cache PCM).
fn assert_delivered(d: &Dictation, seconds: f64) -> (f64, Vec<i16>) {
    d.report.assert_success();
    assert_eq!(d.report.stdout.trim(), TRANSCRIPT);
    let cache = fixtures::only_ogg_in(&d.sandbox.cache_dir().join("recordings"));
    let cached = decode_ogg_strict(&std::fs::read(&cache).expect("cache ogg"), 16_000);
    let recorded = cached.len() as f64 / 16_000.0;
    assert!(
        (recorded - seconds).abs() < 1.0,
        "cache OGG holds {recorded:.2}s for a {seconds}s dictation"
    );
    let r = envelope_correlation(&looped_reference(cached.len()), &cached, 16_000);
    assert!(
        r > 0.9,
        "cache OGG does not hold the dictated speech (r={r:.3})"
    );
    (recorded, cached)
}

/// Every upload decodes to the cached recording (same length, same
/// content).
async fn assert_uploads_are_recording(d: &Dictation, cached: &[i16]) {
    for (i, post) in d.provider.posts().await.iter().enumerate() {
        let uploaded = decode_ogg_strict(&multipart_file(post), 16_000);
        assert!(
            uploaded.len().abs_diff(cached.len()) <= 320,
            "upload {i}: {} samples vs cache {}",
            uploaded.len(),
            cached.len()
        );
        let r = envelope_correlation(cached, &uploaded, 16_000);
        assert!(r > 0.95, "upload {i} is not the recording (r={r:.3})");
    }
}

fn stop_metrics(m: &mut Metrics, runs: &[Dictation]) {
    for step in ["capture_stopped", "ogg_flushed", "transcription_done"] {
        let samples: Vec<f64> = runs.iter().map(|d| d.report.stop_ms(step) as f64).collect();
        m.timing(&format!("stop_{step}_ms"), &samples);
    }
    let posts: Vec<f64> = runs.iter().map(|d| d.sigint_to_first_post_ms).collect();
    m.timing("sigint_to_first_post_ms", &posts)
        .set("child_cpu_ms", runs[0].report.child_cpu_ms)
        .set("child_maxrss_kb", runs[0].report.child_maxrss_kb as f64);
}

/// Short dictation (3 s), provider answers at once.
#[tokio::test(flavor = "multi_thread")]
async fn perf_dictate_short_3s() {
    require_counters!();
    let mut runs = Vec::new();
    for _ in 0..repeats() {
        let d = dictate(3.0, &[200], &[], ProxyOptions::default()).await;
        let (_, cached) = assert_delivered(&d, 3.0);
        assert_eq!(d.provider.posts().await.len(), 1);
        assert_uploads_are_recording(&d, &cached).await;
        runs.push(d);
    }
    let d = &runs[0];
    let recorded = assert_delivered(d, 3.0).0;
    let mut m = Metrics::new("http-client-pool-prewarm", "dictate-short-3s");
    stop_metrics(&mut m, &runs);
    m.counters(&d.report, &["http_client_builds"])
        .set("tcp_connections", d.provider.proxy.accepted() as f64)
        .write();
    Metrics::new("single-opus-encode", "dictate-short-3s")
        .counters(&d.report, &["opus_frames_encoded"])
        .set(
            "encode_passes",
            d.report.counter("opus_frames_encoded") as f64 / (recorded * FRAMES_PER_SEC),
        )
        .write();
}

/// Short dictation over an emulated slow link: every new connection is
/// delayed 150 ms by the proxy (application-side emulation of TCP+TLS
/// round trips).  Prewarm is credited only when the connection opened
/// BEFORE the stop gesture is the one that carries the upload.
#[tokio::test(flavor = "multi_thread")]
async fn perf_dictate_short_3s_slow_connect() {
    require_counters!();
    let proxy = ProxyOptions {
        accept_delay: Duration::from_millis(150),
        idle_close: None,
    };
    let mut runs = Vec::new();
    let mut prewarmed = Vec::new();
    let mut capture_start = Vec::new();
    for _ in 0..repeats() {
        let d = dictate(3.0, &[200], &[], proxy).await;
        let (_, cached) = assert_delivered(&d, 3.0);
        assert_eq!(d.provider.posts().await.len(), 1, "exactly one paid upload");
        assert_uploads_are_recording(&d, &cached).await;
        let (_, accepted) = d
            .provider
            .proxy
            .first_post_connection()
            .expect("POST socket");
        prewarmed.push(f64::from(u8::from(accepted < d.sigint_at)));
        capture_start.push(d.report.start.get("capture_started").copied().unwrap_or(0) as f64);
        runs.push(d);
    }
    let mut m = Metrics::new("http-client-pool-prewarm", "dictate-short-3s-connect-150ms");
    stop_metrics(&mut m, &runs);
    m.set(
        "upload_on_prewarmed_connection",
        if prewarmed.iter().all(|&v| v == 1.0) {
            1.0
        } else {
            0.0
        },
    )
    .set(
        "extra_connections",
        runs[0].provider.proxy.accepted().saturating_sub(1) as f64,
    )
    .timing("start_capture_started_ms", &capture_start)
    .write();
}

/// Long dictation (2 min).  The proxy drops connections idle for 30 s,
/// like a server dropping keep-alives: a prewarmed connection must not
/// turn into a failed or retried upload.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "long: 2-minute real-time recording"]
async fn perf_dictate_long_2min() {
    require_counters!();
    let proxy = ProxyOptions {
        accept_delay: Duration::ZERO,
        idle_close: Some(Duration::from_secs(30)),
    };
    let d = dictate(120.0, &[200], &[], proxy).await;
    let (recorded, cached) = assert_delivered(&d, 120.0);
    assert_eq!(d.provider.posts().await.len(), 1, "exactly one paid upload");
    assert_uploads_are_recording(&d, &cached).await;
    assert_eq!(d.report.counter("data_retries"), 0, "no data retry charged");
    let mut m = Metrics::new("single-opus-encode", "dictate-long-2min");
    stop_metrics(&mut m, std::slice::from_ref(&d));
    m.counters(
        &d.report,
        &[
            "opus_frames_encoded",
            "http_client_builds",
            "connection_retries",
        ],
    )
    .set(
        "encode_passes",
        d.report.counter("opus_frames_encoded") as f64 / (recorded * FRAMES_PER_SEC),
    )
    .write();
}

/// Fallback: the first chain entry answers 503 to the live upload, the
/// second 503 again on the file retry, the third succeeds.  Every entry
/// must upload the complete recording.
#[tokio::test(flavor = "multi_thread")]
async fn perf_dictate_fallback_chain_3() {
    require_counters!();
    let mut runs = Vec::new();
    let mut busy_to_final = Vec::new();
    for _ in 0..repeats() {
        let d = dictate(3.0, &[503, 503, 200], &CHAIN_3, ProxyOptions::default()).await;
        let (_, cached) = assert_delivered(&d, 3.0);
        assert_eq!(d.provider.posts().await.len(), 3);
        assert_uploads_are_recording(&d, &cached).await;
        let times = d.provider.post_times();
        busy_to_final.push(times[2].duration_since(times[0]).as_secs_f64() * 1000.0);
        runs.push(d);
    }
    let d = &runs[0];
    let mut m = Metrics::new("upload-normalize-once", "dictate-fallback-chain-3");
    stop_metrics(&mut m, &runs);
    m.counters(
        &d.report,
        &["normalize_calls", "upload_encodes", "audio_file_decodes"],
    )
    .timing("first_busy_to_final_post_ms", &busy_to_final)
    .write();
    Metrics::new("http-client-pool-prewarm", "dictate-fallback-chain-3")
        .counters(&d.report, &["http_client_builds"])
        .set("tcp_connections", d.provider.proxy.accepted() as f64)
        .write();
}

/// Overlay FFT work while recording: 5 s paced dictation with the
/// recording badge on an isolated X display, per visualizer mode.
/// `fft_calls / overlay_frames` is the FFT rate per rendered frame.
async fn overlay_fft(viz: Option<&str>, cut: &'static str) {
    let Some(display) = crate::support::display::IsolatedDisplay::start() else {
        eprintln!("SKIP: neither Xvfb nor weston available");
        return;
    };
    let sandbox = Sandbox::new();
    let provider = MockProvider::start(&[200], Duration::ZERO, ProxyOptions::default()).await;
    let yaml = config_yaml(&sandbox.output_dir(), &provider.url(), &[])
        .replace("visual_overlay: false", "visual_overlay: true");
    sandbox.write_config(&yaml);
    let input = sandbox.path().join("speech.wav");
    fixtures::write_wav(&input, &fixtures::speech_pcm(4.0, 16_000), 16_000);
    let mut args = vec![
        "dictate",
        "--no-sounds",
        "--no-paste",
        "--no-bt-auto-switch",
        "--no-auto-pause",
        "--target-window",
        "0",
    ];
    if let Some(mode) = viz {
        args.extend(["--viz", mode]);
    }
    let mut cmd = sandbox.command(&args);
    cmd.env("TALK_RS_PERF_PACED_INPUT", &input)
        .env("DISPLAY", &display.display)
        .env("GDK_BACKEND", "x11");
    let (report, _) = runner::run_with_sigint(
        cmd,
        &sandbox.log_path(),
        "recording — waiting for shutdown signal",
        Duration::from_secs(5),
        Duration::from_secs(120),
    )
    .await;
    drop(display);
    report.assert_success();
    assert_eq!(report.stdout.trim(), TRANSCRIPT);
    let frames = report.counter("overlay_frames");
    assert!(frames > 100, "overlay rendered only {frames} frames");
    let ffts = report.counter("fft_calls");
    Metrics::new("overlay-fft-gating", cut)
        .set("overlay_frames", frames as f64)
        .set("fft_calls", ffts as f64)
        .set("fft_per_frame", ffts as f64 / frames as f64)
        .set("child_cpu_ms", report.child_cpu_ms)
        .write();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs an isolated X display (Xvfb or weston)"]
async fn perf_overlay_fft_viz_none() {
    require_counters!();
    overlay_fft(None, "recording-5s-viz-none").await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs an isolated X display (Xvfb or weston)"]
async fn perf_overlay_fft_viz_amplitude() {
    require_counters!();
    overlay_fft(Some("amplitude"), "recording-5s-viz-amplitude").await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs an isolated X display (Xvfb or weston)"]
async fn perf_overlay_fft_viz_waterfall() {
    require_counters!();
    overlay_fft(Some("waterfall"), "recording-5s-viz-waterfall").await;
}
