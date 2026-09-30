//! Reusable audio playback through the default output device via cpal.
//!
//! [`AudioPlayer`] owns a single continuous cpal output stream (playing
//! silence when idle) and a shared playback buffer.  It is the shared
//! core behind two consumers:
//!
//! * The `speak` command (via [`AudioPlayer::play_pcm_blocking`]):
//!   play a mono `i16` PCM buffer at a given sample rate, blocking
//!   until playback finishes.
//! * The recordings browser's `WavPlayer` (in
//!   `crate::record::player`), which delegates its stream + transport
//!   controls (play / pause / seek / progress) to an `AudioPlayer` via
//!   [`AudioPlayer::load_f32`] and the control methods here.
//!
//! Extracting this from `record/player.rs` lets the `speak` command
//! reuse the exact playback path without pulling in the `ui` feature —
//! it lives under the `playback` feature (cpal only).

use crate::error::TalkError;

/// Shared state between the GUI/caller thread and the cpal output
/// callback.
struct PlaybackState {
    /// Mono samples at the device's native sample rate.
    samples: Vec<f32>,
    position: usize,
    paused: bool,
}

/// Plays mono audio through cpal's default output device.
///
/// The output stream runs continuously (outputting silence when idle).
/// Load audio with [`load_f32`](AudioPlayer::load_f32) (samples already
/// at the device rate) or synthesize-and-play in one call with
/// [`play_pcm_blocking`](AudioPlayer::play_pcm_blocking).
pub struct AudioPlayer {
    state: std::sync::Arc<std::sync::Mutex<PlaybackState>>,
    device_sample_rate: u32,
    // Dropping this stops the stream (a cpal stream, or the harness's
    // capturing sink thread).
    _stream: OutputStream,
}

/// What drives [`fill_output`]: the cpal device, or (performance
/// harness only) a paced capturing sink.  Held only for its `Drop`
/// (stopping the stream / joining the sink thread), never read.
#[allow(dead_code)] // variants are kept alive for their Drop, not read
enum OutputStream {
    Device(cpal::Stream),
    #[cfg(feature = "perf-counters")]
    Capture(perf_sink::CaptureSink),
}

/// The output callback body, shared by every output backend so that
/// what a capturing sink consumes is exactly what the device would.
///
/// Fills `output` (interleaved, `channels` wide) from the shared
/// playback state and returns the mono frames taken from the loaded
/// samples (0 when paused, idle, or the state lock was busy).
fn fill_output(
    state: &std::sync::Mutex<PlaybackState>,
    output: &mut [f32],
    channels: usize,
) -> usize {
    let Ok(mut guard) = state.try_lock() else {
        output.fill(0.0);
        return 0;
    };
    if guard.paused {
        output.fill(0.0);
        return 0;
    }
    let frames = output.len() / channels;
    let mut consumed = 0;
    for frame_idx in 0..frames {
        let sample = if guard.position < guard.samples.len() {
            let s = guard.samples[guard.position];
            guard.position += 1;
            consumed += 1;
            s
        } else {
            0.0
        };
        for ch in 0..channels {
            output[frame_idx * channels + ch] = sample;
        }
    }
    consumed
}

impl AudioPlayer {
    /// Open the default output device and start a silent stream.
    pub fn new() -> Result<Self, TalkError> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let state = std::sync::Arc::new(std::sync::Mutex::new(PlaybackState {
            samples: Vec::new(),
            position: 0,
            paused: false,
        }));

        #[cfg(feature = "perf-counters")]
        if let Some(sink) = perf_sink::CaptureSink::from_env(std::sync::Arc::clone(&state))? {
            return Ok(Self {
                state,
                device_sample_rate: sink.sample_rate,
                _stream: OutputStream::Capture(sink),
            });
        }

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| TalkError::Audio("no default audio output device".to_string()))?;
        let config = device
            .default_output_config()
            .map_err(|e| TalkError::Audio(format!("output config: {}", e)))?;

        let device_sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let state_cb = std::sync::Arc::clone(&state);

        let stream = device
            .build_output_stream(
                &cpal::StreamConfig {
                    channels: config.channels(),
                    sample_rate: config.sample_rate(),
                    buffer_size: cpal::BufferSize::Default,
                },
                move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    fill_output(&state_cb, output, channels);
                },
                |err| log::error!("audio output error: {}", err),
                None,
            )
            .map_err(|e| TalkError::Audio(format!("output stream: {}", e)))?;

        stream
            .play()
            .map_err(|e| TalkError::Audio(format!("start output stream: {}", e)))?;

        Ok(Self {
            state,
            device_sample_rate,
            _stream: OutputStream::Device(stream),
        })
    }

    /// The device's native output sample rate in Hz.
    pub fn device_sample_rate(&self) -> u32 {
        self.device_sample_rate
    }

    /// Load mono `f32` samples that are ALREADY at the device sample
    /// rate, and start playback from the beginning.
    ///
    /// Callers that have samples at a different rate should resample to
    /// [`device_sample_rate`](AudioPlayer::device_sample_rate) first,
    /// or use [`play_pcm_blocking`](AudioPlayer::play_pcm_blocking)
    /// which resamples for them.
    pub fn load_f32(&self, samples: Vec<f32>) {
        crate::perf_counters::gauge_set(
            crate::perf_counters::Gauge::PlayerRetainedSamples,
            samples.len() as i64,
        );
        if let Ok(mut guard) = self.state.lock() {
            guard.samples = samples;
            guard.position = 0;
            guard.paused = false;
        }
    }

    /// Play a mono `i16` PCM buffer at `sample_rate`, blocking the
    /// calling thread until playback finishes.
    ///
    /// The PCM is converted to `f32` and resampled to the device rate.
    /// Returns once every sample has been consumed by the output
    /// callback (plus a short drain margin so the tail is not clipped).
    pub fn play_pcm_blocking(&self, pcm: &[i16], sample_rate: u32) -> Result<(), TalkError> {
        if pcm.is_empty() {
            return Ok(());
        }
        let mono: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
        let resampled = resample_linear_mono(&mono, sample_rate, self.device_sample_rate);
        self.load_f32(resampled);

        // Block until the output callback has consumed all samples.
        // Poll the shared position rather than computing a fixed sleep,
        // so an under-run or device stall does not clip the tail.
        let poll = std::time::Duration::from_millis(20);
        loop {
            std::thread::sleep(poll);
            let done = self
                .state
                .lock()
                .map(|g| g.position >= g.samples.len() || g.samples.is_empty())
                .unwrap_or(true);
            if done {
                break;
            }
        }
        // Small fixed drain margin so the device flushes its own
        // output buffer before we clear the samples.
        std::thread::sleep(std::time::Duration::from_millis(120));
        self.stop();
        Ok(())
    }

    /// Stop playback immediately and clear the buffer.
    pub fn stop(&self) {
        crate::perf_counters::gauge_set(crate::perf_counters::Gauge::PlayerRetainedSamples, 0);
        if let Ok(mut guard) = self.state.lock() {
            guard.samples.clear();
            guard.position = 0;
            guard.paused = false;
        }
    }

    /// `true` when all samples have been consumed (or nothing loaded).
    pub fn is_finished(&self) -> bool {
        self.state
            .lock()
            .map(|g| g.samples.is_empty() || g.position >= g.samples.len())
            .unwrap_or(true)
    }

    /// Playback progress as a fraction (0.0–1.0).
    pub fn progress(&self) -> f64 {
        self.state
            .lock()
            .map(|g| {
                if g.samples.is_empty() {
                    0.0
                } else {
                    g.position as f64 / g.samples.len() as f64
                }
            })
            .unwrap_or(0.0)
    }

    /// Seek to a position expressed as a fraction (0.0–1.0).
    pub fn seek(&self, fraction: f64) {
        let fraction = fraction.clamp(0.0, 1.0);
        if let Ok(mut guard) = self.state.lock() {
            if !guard.samples.is_empty() {
                guard.position = (fraction * guard.samples.len() as f64) as usize;
            }
        }
    }

    /// Pause playback (position is preserved).
    pub fn pause(&self) {
        if let Ok(mut guard) = self.state.lock() {
            guard.paused = true;
        }
    }

    /// Resume playback from the current position.
    pub fn resume(&self) {
        if let Ok(mut guard) = self.state.lock() {
            guard.paused = false;
        }
    }

    /// `true` when playback is paused.
    pub fn is_paused(&self) -> bool {
        self.state.lock().map(|g| g.paused).unwrap_or(false)
    }

    /// `true` when audio is loaded (samples are present).
    pub fn has_audio(&self) -> bool {
        self.state
            .lock()
            .map(|g| !g.samples.is_empty())
            .unwrap_or(false)
    }

    /// Total duration of loaded audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.state
            .lock()
            .map(|g| {
                if g.samples.is_empty() {
                    0.0
                } else {
                    g.samples.len() as f64 / self.device_sample_rate as f64
                }
            })
            .unwrap_or(0.0)
    }
}

/// Performance-harness output sink (feature `perf-counters` only).
///
/// With `TALK_RS_PERF_AUDIO_SINK=<path>` set, [`AudioPlayer::new`]
/// opens no device: a thread calls [`fill_output`] every 10 ms of wall
/// clock with a 10 ms buffer at 48 kHz mono — the pace of a real
/// device — and appends a line per event to `<path>`:
///
/// - `first-audio +<ms> epoch=<unix ms> sample=<n>`: the first
///   consumed frame whose magnitude exceeds 1e-3 (meaningful sound,
///   not leading silence), with milliseconds since the sink opened,
///   the wall clock, and its index among consumed samples;
/// - `consumed <total>` every second and on drop: frames consumed.
///
/// The sink also writes every consumed sample (f32 LE) to
/// `<path>.pcm` so a test can compare what was played with what was
/// synthesized or decoded.
#[cfg(feature = "perf-counters")]
mod perf_sink {
    use super::{fill_output, PlaybackState};
    use crate::error::TalkError;
    use std::io::Write;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    pub(super) struct CaptureSink {
        pub(super) sample_rate: u32,
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl CaptureSink {
        pub(super) fn from_env(
            state: Arc<Mutex<PlaybackState>>,
        ) -> Result<Option<Self>, TalkError> {
            let Some(path) = std::env::var_os("TALK_RS_PERF_AUDIO_SINK") else {
                return Ok(None);
            };
            let path = std::path::PathBuf::from(path);
            let open = |p: &std::path::Path| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .map_err(|e| TalkError::Audio(format!("perf sink {}: {e}", p.display())))
            };
            let mut events = open(&path)?;
            let mut pcm = open(&path.with_extension("pcm"))?;
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            const RATE: u32 = 48_000;
            let thread = std::thread::spawn(move || {
                let started = std::time::Instant::now();
                let period = std::time::Duration::from_millis(10);
                let mut buffer = vec![0.0f32; (RATE / 100) as usize];
                let mut total = 0u64;
                let mut heard = false;
                let mut tick = 0u64;
                let mut next = started;
                while !flag.load(Ordering::Acquire) {
                    let consumed = fill_output(&state, &mut buffer, 1);
                    if consumed > 0 {
                        let bytes: Vec<u8> = buffer[..consumed]
                            .iter()
                            .flat_map(|s| s.to_le_bytes())
                            .collect();
                        let _ = pcm.write_all(&bytes);
                        if !heard {
                            if let Some(i) = buffer[..consumed].iter().position(|s| s.abs() > 1e-3)
                            {
                                heard = true;
                                let epoch = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.as_millis())
                                    .unwrap_or_default();
                                let _ = writeln!(
                                    events,
                                    "first-audio +{}ms epoch={epoch} sample={}",
                                    started.elapsed().as_millis(),
                                    total + i as u64
                                );
                            }
                        }
                        total += consumed as u64;
                    }
                    tick += 1;
                    if tick.is_multiple_of(100) {
                        let _ = writeln!(events, "consumed {total}");
                    }
                    next += period;
                    if let Some(wait) = next.checked_duration_since(std::time::Instant::now()) {
                        std::thread::sleep(wait);
                    }
                }
                let _ = writeln!(events, "consumed {total}");
                let _ = pcm.flush();
            });
            Ok(Some(Self {
                sample_rate: RATE,
                stop,
                thread: Some(thread),
            }))
        }
    }

    impl Drop for CaptureSink {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

/// Linear-interpolation resample of a mono `f32` buffer from
/// `src_rate` to `target_rate`.
///
/// Self-contained (no dependency on the `record`-only resample helper)
/// so `AudioPlayer` stays usable in a `playback`-only build.
fn resample_linear_mono(mono: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
    if src_rate == 0 || src_rate == target_rate || mono.is_empty() {
        return mono.to_vec();
    }
    let ratio = target_rate as f64 / src_rate as f64;
    let new_len = (mono.len() as f64 * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let s0 = mono.get(idx).copied().unwrap_or(0.0);
        let s1 = mono.get(idx + 1).copied().unwrap_or(s0);
        out.push(s0 + (s1 - s0) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_same_rate_is_identity() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0];
        let out = resample_linear_mono(&samples, 24_000, 24_000);
        assert_eq!(out, samples);
    }

    #[test]
    fn resample_empty_is_empty() {
        assert!(resample_linear_mono(&[], 24_000, 48_000).is_empty());
    }

    #[test]
    fn resample_upsample_doubles_length_approx() {
        let samples = vec![0.0f32, 1.0, 0.0, -1.0];
        let out = resample_linear_mono(&samples, 24_000, 48_000);
        // Upsampling 2x roughly doubles the sample count.
        assert!(out.len() >= samples.len() * 2 - 1);
    }

    #[test]
    fn resample_zero_src_rate_is_identity() {
        let samples = vec![0.1f32, 0.2];
        let out = resample_linear_mono(&samples, 0, 48_000);
        assert_eq!(out, samples);
    }
}
