//! Observational work counters for the performance harness.
//!
//! Every function in this module is a no-op unless the crate is built
//! with the `perf-counters` cargo feature (or as the crate's own unit
//! tests, following the `tail_samples_relocated` precedent in
//! `audio::writer`).  Production sites call [`incr`], [`add`],
//! [`gauge_inc`] and [`gauge_dec`] unconditionally; otherwise these
//! compile to empty inline functions, so the release binary pays
//! nothing.
//!
//! When enabled:
//!
//! - counters are process-wide `AtomicU64`s, readable in-process via
//!   [`snapshot`], mirrored per thread (`thread_snapshot`, unit tests
//!   only) so a test can measure its own work while other tests run
//!   in parallel;
//! - (feature only) the CLI calls [`init`] at startup and [`emit`]
//!   before returning,
//!   which logs one `perf-counter: <name>=<value>` line per counter
//!   (plus `perf-mark:` lines recorded with [`mark`]);
//! - `SIGUSR2` makes a running process dump the same lines, so a
//!   long-lived process (the recordings browser) can be sampled
//!   without exiting.
//!
//! Counters observe; they never change behaviour.  Gauges track a
//! current level and the maximum level reached (`<name>_max`).

/// Monotonic work counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Counter {
    /// `reqwest::Client` constructions for provider HTTP requests.
    HttpClientBuilds,
    /// Whole-file upload normalizations (decode + 16 kHz re-encode).
    NormalizeCalls,
    /// Opus frames encoded by `OggOpusWriter` (all writers, all paths).
    OpusFramesEncoded,
    /// `compute_spectrum` calls (overlay frames and waterfall columns).
    FftCalls,
    /// Overlay render-loop frames (denominator for per-frame FFT cost).
    OverlayFrames,
    /// Parakeet `OfflineRecognizer` constructions.
    RecognizerCreates,
    /// Kokoro `OfflineTts` constructions.
    TtsCreates,
    /// `<stem>.pick.yml` reads.
    PickReads,
    /// Full recordings-browser directory listings.
    ListCalls,
    /// Settle sleeps performed by `ensure_focus` before checking focus.
    FocusSleeps,
    /// Invocations of recordings-browser playback-progress tick callbacks.
    PlayerTickCallbacks,
    /// Waterfall computations that had to decode the audio (cache miss).
    WaterfallDecodes,
    /// Whole-file decodes performed to start playback.
    PlaybackDecodes,
}

impl Counter {
    /// Every counter, in emission order.
    pub const ALL: [Counter; 13] = [
        Counter::HttpClientBuilds,
        Counter::NormalizeCalls,
        Counter::OpusFramesEncoded,
        Counter::FftCalls,
        Counter::OverlayFrames,
        Counter::RecognizerCreates,
        Counter::TtsCreates,
        Counter::PickReads,
        Counter::ListCalls,
        Counter::FocusSleeps,
        Counter::PlayerTickCallbacks,
        Counter::WaterfallDecodes,
        Counter::PlaybackDecodes,
    ];

    /// Stable snake_case name used in `perf-counter:` lines.
    pub fn name(self) -> &'static str {
        match self {
            Counter::HttpClientBuilds => "http_client_builds",
            Counter::NormalizeCalls => "normalize_calls",
            Counter::OpusFramesEncoded => "opus_frames_encoded",
            Counter::FftCalls => "fft_calls",
            Counter::OverlayFrames => "overlay_frames",
            Counter::RecognizerCreates => "recognizer_creates",
            Counter::TtsCreates => "tts_creates",
            Counter::PickReads => "pick_reads",
            Counter::ListCalls => "list_calls",
            Counter::FocusSleeps => "focus_sleeps",
            Counter::PlayerTickCallbacks => "player_tick_callbacks",
            Counter::WaterfallDecodes => "waterfall_decodes",
            Counter::PlaybackDecodes => "playback_decodes",
        }
    }
}

/// Level gauges; each also records the maximum level reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gauge {
    /// Recurring playback-progress timer sources currently registered
    /// in the recordings browser.  Incremented when a source is
    /// registered, decremented when it is removed.
    PlayerTickSourcesLive,
    /// Waterfall computations currently running on worker threads.
    WaterfallWorkersInflight,
}

impl Gauge {
    /// Every gauge, in emission order.
    pub const ALL: [Gauge; 2] = [
        Gauge::PlayerTickSourcesLive,
        Gauge::WaterfallWorkersInflight,
    ];

    /// Stable snake_case name; the maximum is emitted as `<name>_max`.
    pub fn name(self) -> &'static str {
        match self {
            Gauge::PlayerTickSourcesLive => "player_tick_sources_live",
            Gauge::WaterfallWorkersInflight => "waterfall_workers_inflight",
        }
    }
}

#[cfg(any(test, feature = "perf-counters"))]
mod enabled {
    use super::{Counter, Gauge};
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
    #[cfg(feature = "perf-counters")]
    use std::sync::OnceLock;
    #[cfg(feature = "perf-counters")]
    use std::time::Instant;

    thread_local! {
        static THREAD_COUNTERS: [std::cell::Cell<u64>; Counter::ALL.len()] =
            const { [const { std::cell::Cell::new(0) }; Counter::ALL.len()] };
    }

    static COUNTERS: [AtomicU64; Counter::ALL.len()] =
        [const { AtomicU64::new(0) }; Counter::ALL.len()];
    static GAUGES: [AtomicI64; Gauge::ALL.len()] = [const { AtomicI64::new(0) }; Gauge::ALL.len()];
    static GAUGE_MAX: [AtomicI64; Gauge::ALL.len()] =
        [const { AtomicI64::new(0) }; Gauge::ALL.len()];
    /// Largest GTK main-loop tick gap (ms) since the last dump.
    static GTK_STALL_MAX_MS: AtomicU64 = AtomicU64::new(0);
    #[cfg(feature = "perf-counters")]
    static START: OnceLock<Instant> = OnceLock::new();

    fn counter_index(counter: Counter) -> usize {
        Counter::ALL
            .iter()
            .position(|c| *c == counter)
            .unwrap_or_default()
    }

    fn gauge_index(gauge: Gauge) -> usize {
        Gauge::ALL
            .iter()
            .position(|g| *g == gauge)
            .unwrap_or_default()
    }

    pub fn add(counter: Counter, n: u64) {
        let i = counter_index(counter);
        COUNTERS[i].fetch_add(n, Ordering::Relaxed);
        THREAD_COUNTERS.with(|c| c[i].set(c[i].get() + n));
    }

    #[cfg(test)]
    pub fn thread_value(counter: Counter) -> u64 {
        THREAD_COUNTERS.with(|c| c[counter_index(counter)].get())
    }

    pub fn gauge_inc(gauge: Gauge) {
        let i = gauge_index(gauge);
        let now = GAUGES[i].fetch_add(1, Ordering::Relaxed) + 1;
        GAUGE_MAX[i].fetch_max(now, Ordering::Relaxed);
    }

    pub fn gauge_dec(gauge: Gauge) {
        GAUGES[gauge_index(gauge)].fetch_sub(1, Ordering::Relaxed);
    }

    pub fn snapshot() -> Vec<(String, u64)> {
        let mut out: Vec<(String, u64)> = Counter::ALL
            .iter()
            .map(|c| {
                (
                    c.name().to_string(),
                    COUNTERS[counter_index(*c)].load(Ordering::Relaxed),
                )
            })
            .collect();
        for g in Gauge::ALL {
            let i = gauge_index(g);
            out.push((
                g.name().to_string(),
                GAUGES[i].load(Ordering::Relaxed).max(0) as u64,
            ));
            out.push((
                format!("{}_max", g.name()),
                GAUGE_MAX[i].load(Ordering::Relaxed).max(0) as u64,
            ));
        }
        out.push((
            "gtk_stall_max_ms".to_string(),
            GTK_STALL_MAX_MS.load(Ordering::Relaxed),
        ));
        #[cfg(feature = "perf-counters")]
        out.extend(process_resources());
        out
    }

    /// Resource usage of this process so far, from `/proc/self`:
    /// peak RSS (`VmHWM`), live threads and user+system CPU time.
    #[cfg(feature = "perf-counters")]
    fn process_resources() -> Vec<(String, u64)> {
        let mut out = Vec::new();
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for (key, name) in [("VmHWM:", "vm_hwm_kb"), ("Threads:", "threads")] {
                if let Some(value) = status
                    .lines()
                    .find_map(|l| l.strip_prefix(key))
                    .and_then(|v| v.split_whitespace().next())
                    .and_then(|v| v.parse::<u64>().ok())
                {
                    out.push((name.to_string(), value));
                }
            }
        }
        if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
            // Fields after the parenthesised command name; utime and
            // stime are the 12th and 13th of those, in USER_HZ (100/s
            // on Linux).
            let ticks: u64 = stat
                .rfind(')')
                .map(|close| {
                    stat[close + 1..]
                        .split_whitespace()
                        .skip(11)
                        .take(2)
                        .filter_map(|v| v.parse::<u64>().ok())
                        .sum()
                })
                .unwrap_or_default();
            out.push(("cpu_ms".to_string(), ticks * 10));
        }
        out
    }

    #[cfg(feature = "perf-counters")]
    pub fn elapsed_ms() -> u128 {
        START.get_or_init(Instant::now).elapsed().as_millis()
    }

    #[cfg(feature = "perf-counters")]
    pub fn mark(name: &str) {
        log::info!("perf-mark: {}=+{}ms", name, elapsed_ms());
    }

    #[cfg(feature = "perf-counters")]
    pub fn emit() {
        for (name, value) in snapshot() {
            log::info!("perf-counter: {}={}", name, value);
        }
        // The stall maximum is a per-window measurement: each dump
        // closes one observation window.
        GTK_STALL_MAX_MS.store(0, Ordering::Relaxed);
    }

    #[cfg(feature = "perf-counters")]
    pub fn init() {
        let _ = START.get_or_init(Instant::now);
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async {
                use tokio::signal::unix::{signal, SignalKind};
                let Ok(mut usr2) = signal(SignalKind::user_defined2()) else {
                    return;
                };
                while usr2.recv().await.is_some() {
                    mark("dump");
                    emit();
                }
            });
        }
    }

    #[cfg(all(feature = "ui", feature = "perf-counters"))]
    pub fn install_gtk_stall_probe() {
        use gtk4::glib;
        if std::env::var_os("TALK_RS_PERF_GTK_PROBE").is_none() {
            return;
        }
        let last = std::cell::Cell::new(Instant::now());
        glib::timeout_add_local(std::time::Duration::from_millis(16), move || {
            let now = Instant::now();
            let gap = now.duration_since(last.replace(now)).as_millis() as u64;
            GTK_STALL_MAX_MS.fetch_max(gap, Ordering::Relaxed);
            glib::ControlFlow::Continue
        });
    }
}

/// Increment `counter` by one.
#[inline(always)]
pub fn incr(counter: Counter) {
    add(counter, 1);
}

/// Increment `counter` by `n`.
#[inline(always)]
pub fn add(counter: Counter, n: u64) {
    #[cfg(any(test, feature = "perf-counters"))]
    enabled::add(counter, n);
    #[cfg(not(any(test, feature = "perf-counters")))]
    let _ = (counter, n);
}

/// Raise `gauge` by one, updating its recorded maximum.
#[inline(always)]
pub fn gauge_inc(gauge: Gauge) {
    #[cfg(any(test, feature = "perf-counters"))]
    enabled::gauge_inc(gauge);
    #[cfg(not(any(test, feature = "perf-counters")))]
    let _ = gauge;
}

/// Lower `gauge` by one.
#[inline(always)]
pub fn gauge_dec(gauge: Gauge) {
    #[cfg(any(test, feature = "perf-counters"))]
    enabled::gauge_dec(gauge);
    #[cfg(not(any(test, feature = "perf-counters")))]
    let _ = gauge;
}

/// Current value of every counter and gauge (empty when disabled).
pub fn snapshot() -> Vec<(String, u64)> {
    #[cfg(any(test, feature = "perf-counters"))]
    return enabled::snapshot();
    #[cfg(not(any(test, feature = "perf-counters")))]
    Vec::new()
}

/// Work done by the calling thread so far for `counter` (unit tests).
#[cfg(test)]
pub(crate) fn thread_value(counter: Counter) -> u64 {
    enabled::thread_value(counter)
}

/// Append one measurement to `$TALK_RS_PERF_OUT/<item>--<cut>.yaml`
/// (no-op when the variable is unset).  Used by the crate's own
/// performance tests; `tests/perf_support` has the same format.
#[cfg(test)]
pub(crate) fn record_metrics(item: &str, cut: &str, metrics: &[(&str, f64)]) {
    let Some(dir) = std::env::var_os("TALK_RS_PERF_OUT") else {
        return;
    };
    let mut body = format!("item: {item}\ncut: {cut}\nmetrics:\n");
    for (name, value) in metrics {
        let name = name.replace('_', "-");
        if value.fract() == 0.0 {
            body.push_str(&format!("  {name}: {}\n", *value as i64));
        } else {
            body.push_str(&format!("  {name}: {value:.3}\n"));
        }
    }
    let dir = std::path::PathBuf::from(dir);
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join(format!("{item}--{cut}.yaml")), body);
}

/// Log `perf-mark: <name>=+<ms>` relative to [`init`].
#[inline(always)]
pub fn mark(name: &str) {
    #[cfg(feature = "perf-counters")]
    enabled::mark(name);
    #[cfg(not(feature = "perf-counters"))]
    let _ = name;
}

/// Log one `perf-counter:` line per counter and gauge.
#[inline(always)]
pub fn emit() {
    #[cfg(feature = "perf-counters")]
    enabled::emit();
}

/// Start the process clock used by [`mark`] and install the
/// `SIGUSR2` dump handler.  Must be called inside a tokio runtime.
#[inline(always)]
pub fn init() {
    #[cfg(feature = "perf-counters")]
    enabled::init();
}

/// Record the largest gap between 16 ms GTK main-loop ticks, reported
/// as `gtk_stall_max_ms`.  Active only when `TALK_RS_PERF_GTK_PROBE`
/// is set; must be called on the GTK main thread.
#[cfg(feature = "ui")]
#[inline(always)]
pub fn install_gtk_stall_probe() {
    #[cfg(feature = "perf-counters")]
    enabled::install_gtk_stall_probe();
}

/// A live-capture replacement and its sample rate.
#[cfg(feature = "capture")]
pub type PacedCapture = (Box<dyn crate::audio::AudioCapture>, u32);

/// Hidden paced live-capture source for the performance harness.
///
/// When `TALK_RS_PERF_PACED_INPUT` names a 16-bit mono WAV file, the
/// live dictation path records from that file at real-time pace (one
/// 20 ms chunk every 20 ms, looping) instead of opening the
/// microphone, so the live one-shot path (SIGINT stop, live upload,
/// fallback) can be measured without audio hardware.  Returns the
/// capture and its sample rate.  Always `None` without the feature.
#[cfg(feature = "capture")]
pub fn paced_input() -> Result<Option<PacedCapture>, crate::error::TalkError> {
    #[cfg(feature = "perf-counters")]
    {
        let Some(path) = std::env::var_os("TALK_RS_PERF_PACED_INPUT") else {
            return Ok(None);
        };
        let source = paced::PacedWavSource::open(std::path::Path::new(&path))?;
        let rate = source.sample_rate;
        log::info!(
            "perf: paced input {} at {} Hz",
            std::path::Path::new(&path).display(),
            rate
        );
        Ok(Some((Box::new(source), rate)))
    }
    #[cfg(not(feature = "perf-counters"))]
    Ok(None)
}

#[cfg(all(feature = "perf-counters", feature = "capture"))]
mod paced {
    use crate::audio::{AudioCapture, CHANNEL_CAPACITY, CHUNK_DURATION_MS};
    use crate::error::TalkError;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tokio::sync::mpsc;

    pub(super) struct PacedWavSource {
        samples: Arc<Vec<i16>>,
        pub(super) sample_rate: u32,
        running: Arc<AtomicBool>,
    }

    impl PacedWavSource {
        pub(super) fn open(path: &std::path::Path) -> Result<Self, TalkError> {
            let bytes = std::fs::read(path)
                .map_err(|e| TalkError::Audio(format!("paced input {}: {e}", path.display())))?;
            let bad = || TalkError::Audio(format!("paced input {}: not a PCM WAV", path.display()));
            if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
                return Err(bad());
            }
            let mut pos = 12;
            let mut rate = None;
            let mut channels = 1u16;
            while pos + 8 <= bytes.len() {
                let id = &bytes[pos..pos + 4];
                let size = u32::from_le_bytes([
                    bytes[pos + 4],
                    bytes[pos + 5],
                    bytes[pos + 6],
                    bytes[pos + 7],
                ]) as usize;
                let body = pos + 8;
                let end = (body + size).min(bytes.len());
                if id == b"fmt " && end >= body + 8 {
                    channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
                    rate = Some(u32::from_le_bytes([
                        bytes[body + 4],
                        bytes[body + 5],
                        bytes[body + 6],
                        bytes[body + 7],
                    ]));
                } else if id == b"data" {
                    if channels != 1 {
                        return Err(bad());
                    }
                    let samples = bytes[body..end]
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]))
                        .collect::<Vec<_>>();
                    return Ok(Self {
                        samples: Arc::new(samples),
                        sample_rate: rate.ok_or_else(bad)?,
                        running: Arc::new(AtomicBool::new(false)),
                    });
                }
                pos = body + size + (size & 1);
            }
            Err(bad())
        }
    }

    impl AudioCapture for PacedWavSource {
        fn start(&mut self) -> Result<mpsc::Receiver<Vec<i16>>, TalkError> {
            let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
            let samples = Arc::clone(&self.samples);
            let running = Arc::clone(&self.running);
            let chunk = (self.sample_rate as usize * CHUNK_DURATION_MS as usize) / 1000;
            running.store(true, Ordering::Release);
            tokio::spawn(async move {
                let mut ticker =
                    tokio::time::interval(std::time::Duration::from_millis(CHUNK_DURATION_MS));
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Burst);
                let mut pos = 0usize;
                while !samples.is_empty() {
                    ticker.tick().await;
                    if !running.load(Ordering::Acquire) {
                        break;
                    }
                    let out: Vec<i16> = (0..chunk)
                        .map(|i| samples[(pos + i) % samples.len()])
                        .collect();
                    pos = (pos + chunk) % samples.len();
                    if tx.send(out).await.is_err() {
                        break;
                    }
                }
            });
            Ok(rx)
        }

        fn stop(&mut self) -> Result<(), TalkError> {
            self.running.store(false, Ordering::Release);
            Ok(())
        }
    }
}
