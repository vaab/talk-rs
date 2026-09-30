//! Live one-shot dictation cuts, driven through the real binary with
//! the paced capture seam (`TALK_RS_PERF_PACED_INPUT`): the recording
//! runs at real-time pace and is stopped by SIGINT, exactly like the
//! toggle shortcut.  Covers items `http-client-pool-prewarm`,
//! `single-opus-encode`, `upload-normalize-once` (fallback) and the
//! stop→transcript latency every dictation pays.

use std::time::Duration;

use crate::require_counters;
use crate::support::fixtures;
use crate::support::provider::{multipart_file, MockProvider, ProxyOptions, TRANSCRIPT};
use crate::support::runner::{self, RunReport, Sandbox};
use crate::support::{config_yaml, Metrics, CHAIN_3};

/// Opus frames per second of 16 kHz audio (20 ms frames).
const FRAMES_PER_SEC: f64 = 50.0;

struct Dictation {
    report: RunReport,
    provider: MockProvider,
    sandbox: Sandbox,
    /// SIGINT → the mock provider received the (first) POST.
    sigint_to_first_post_ms: f64,
    /// Proxy accepted the first connection before SIGINT.
    connected_before_stop: bool,
}

async fn dictate(seconds: f64, statuses: &[u16], chain: &[&str], proxy: ProxyOptions) -> Dictation {
    let sandbox = Sandbox::new();
    let provider = MockProvider::start(statuses, Duration::ZERO, proxy).await;
    sandbox.write_config(&config_yaml(&sandbox.output_dir(), &provider.url(), chain));
    let input = sandbox.path().join("speech.wav");
    // A 4 s loop is replayed at real-time pace for as long as we record.
    fixtures::write_wav(&input, &fixtures::speech_pcm(4.0, 16_000), 16_000);

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
    let connected_before_stop = provider
        .proxy
        .accept_times()
        .first()
        .is_some_and(|t| *t < sigint_at);
    Dictation {
        report,
        provider,
        sandbox,
        sigint_to_first_post_ms,
        connected_before_stop,
    }
}

/// Invariants every dictation cut checks: success, the provider's
/// transcript printed, a complete cache recording of the dictated
/// length (± one chunk of SIGINT jitter).
fn assert_delivered(d: &Dictation, seconds: f64) -> f64 {
    d.report.assert_success();
    assert_eq!(d.report.stdout.trim(), TRANSCRIPT);
    let cache = fixtures::only_ogg_in(&d.sandbox.cache_dir().join("recordings"));
    let recorded = fixtures::ogg_duration(&std::fs::read(&cache).expect("cache ogg"));
    assert!(
        (recorded - seconds).abs() < 1.0,
        "cache OGG holds {recorded:.2}s for a {seconds}s dictation"
    );
    recorded
}

fn stop_metrics(m: &mut Metrics, d: &Dictation) {
    for step in ["capture_stopped", "ogg_flushed", "transcription_done"] {
        m.set(&format!("stop_{step}_ms"), d.report.stop_ms(step) as f64);
    }
    m.set("sigint_to_first_post_ms", d.sigint_to_first_post_ms)
        .set("child_cpu_ms", d.report.child_cpu_ms)
        .set("child_maxrss_kb", d.report.child_maxrss_kb as f64);
}

/// Short dictation (3 s), provider answers at once.
#[tokio::test(flavor = "multi_thread")]
async fn perf_dictate_short_3s() {
    require_counters!();
    let d = dictate(3.0, &[200], &[], ProxyOptions::default()).await;
    let recorded = assert_delivered(&d, 3.0);
    let mut m = Metrics::new("http-client-pool-prewarm", "dictate-short-3s");
    stop_metrics(&mut m, &d);
    m.counters(&d.report, &["http_client_builds"])
        .set("tcp_connections", d.provider.proxy.accepted() as f64)
        .set("connected_before_stop", u8::from(d.connected_before_stop))
        .write();
    Metrics::new("single-opus-encode", "dictate-short-3s")
        .counters(&d.report, &["opus_frames_encoded"])
        .set(
            "encode_passes",
            d.report.counter("opus_frames_encoded") as f64 / (recorded * FRAMES_PER_SEC),
        )
        .set("child_cpu_ms", d.report.child_cpu_ms)
        .write();
}

/// Short dictation over an emulated slow link: the proxy delays every
/// new connection by 150 ms (TCP + TLS round trips).  A connection
/// opened while recording (prewarm) takes this off the stop path.
#[tokio::test(flavor = "multi_thread")]
async fn perf_dictate_short_3s_slow_connect() {
    require_counters!();
    let proxy = ProxyOptions {
        accept_delay: Duration::from_millis(150),
        idle_close: None,
    };
    let d = dictate(3.0, &[200], &[], proxy).await;
    assert_delivered(&d, 3.0);
    let mut m = Metrics::new("http-client-pool-prewarm", "dictate-short-3s-connect-150ms");
    stop_metrics(&mut m, &d);
    m.set("connected_before_stop", u8::from(d.connected_before_stop))
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
    let recorded = assert_delivered(&d, 120.0);
    assert_eq!(d.provider.posts().await.len(), 1, "exactly one upload");
    let upload = multipart_file(&d.provider.posts().await[0]);
    let uploaded = fixtures::ogg_duration(&upload);
    assert!(
        (uploaded - recorded).abs() < 0.1,
        "upload holds {uploaded}s"
    );
    let mut m = Metrics::new("single-opus-encode", "dictate-long-2min");
    stop_metrics(&mut m, &d);
    m.counters(&d.report, &["opus_frames_encoded", "http_client_builds"])
        .set(
            "encode_passes",
            d.report.counter("opus_frames_encoded") as f64 / (recorded * FRAMES_PER_SEC),
        )
        .write();
}

/// Fallback: the first chain entry answers 503 to the live upload, the
/// second 503 again on the file retry, the third succeeds.  The cache
/// OGG is re-normalized for every file-backed entry today.
#[tokio::test(flavor = "multi_thread")]
async fn perf_dictate_fallback_chain_3() {
    require_counters!();
    let d = dictate(3.0, &[503, 503, 200], &CHAIN_3, ProxyOptions::default()).await;
    let recorded = assert_delivered(&d, 3.0);
    let posts = d.provider.posts().await;
    assert_eq!(posts.len(), 3);
    for post in &posts {
        let uploaded = fixtures::ogg_duration(&multipart_file(post));
        assert!(
            (uploaded - recorded).abs() < 0.1,
            "upload holds {uploaded}s"
        );
    }
    let times = d.provider.post_times();
    let busy_to_final_ms = times[2].duration_since(times[0]).as_secs_f64() * 1000.0;
    let mut m = Metrics::new("upload-normalize-once", "dictate-fallback-chain-3");
    stop_metrics(&mut m, &d);
    m.counters(&d.report, &["normalize_calls", "opus_frames_encoded"])
        .set("first_busy_to_final_post_ms", busy_to_final_ms)
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
