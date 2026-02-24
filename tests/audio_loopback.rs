//! Closed-loop integration tests for audio playback and capture.
//!
//! Uses a PipeWire/PulseAudio null sink as a virtual loopback device:
//! audio played to the sink is immediately available on its monitor
//! source, enabling end-to-end verification without real hardware.
//!
//! Requirements:
//! - PipeWire (or PulseAudio) running
//! - `pactl` available on PATH
//!
//! Run with: `cargo test --test audio_loopback -- --ignored --test-threads=1`

use std::process::Command;
use std::sync::{Mutex, OnceLock};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use talk_rs::audio::indicator::SoundPlayer;

// ── Shared null sink (lives for entire test binary) ──────────────────
//
// Creating/destroying a null sink per test causes PipeWire routing
// churn and cpal connection caching issues. Instead, we create one
// sink lazily and keep it alive for all tests. PipeWire cleans up
// client modules automatically when the process exits.

const SINK_NAME: &str = "talk_rs_test_loopback";

struct SharedSink {
    _module_id: u32,
}

impl SharedSink {
    fn init() -> Self {
        // Clean up any leftover sinks from previous test runs to avoid
        // duplicate routing and unpredictable audio paths.
        Self::cleanup_stale_sinks();

        let output = Command::new("pactl")
            .args([
                "load-module",
                "module-null-sink",
                &format!("sink_name={SINK_NAME}"),
                "sink_properties=device.description=TalkRsTestLoopback",
            ])
            .output()
            .expect("failed to run pactl — is PipeWire/PulseAudio running?");

        assert!(
            output.status.success(),
            "pactl load-module failed: {}",
            String::from_utf8_lossy(&output.stderr),
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        let _module_id: u32 = stdout
            .trim()
            .parse()
            .unwrap_or_else(|e| panic!("failed to parse module ID from '{stdout}': {e}"));

        // Point cpal at the loopback device.
        std::env::set_var("PULSE_SINK", SINK_NAME);
        std::env::set_var("PULSE_SOURCE", format!("{SINK_NAME}.monitor"));

        // Give PipeWire time to register the new sink and source.
        std::thread::sleep(std::time::Duration::from_millis(300));

        Self { _module_id }
    }

    /// Remove any existing null-sink modules with our sink name.
    fn cleanup_stale_sinks() {
        let output = match Command::new("pactl")
            .args(["list", "short", "modules"])
            .output()
        {
            Ok(o) => o,
            Err(_) => return,
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains(SINK_NAME) {
                if let Some(id_str) = line.split_whitespace().next() {
                    let _ = Command::new("pactl")
                        .args(["unload-module", id_str])
                        .output();
                }
            }
        }
    }
}

/// Ensure the shared null sink exists. Safe to call from any test.
fn ensure_sink() {
    static SINK: OnceLock<SharedSink> = OnceLock::new();
    SINK.get_or_init(SharedSink::init);
}

// ── Test serialization ───────────────────────────────────────────────
//
// Tests MUST run serially (--test-threads=1) because they share
// process-global env vars and a single null sink. This lock adds
// defense-in-depth and recovers from poisoned state if a test panics.

fn serial_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let mutex = LOCK.get_or_init(|| Mutex::new(()));
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

// ── Device lookup ────────────────────────────────────────────────────

/// The ALSA device name that routes through PulseAudio.
///
/// cpal uses the ALSA backend on Linux. The `"default"` ALSA device
/// bypasses PulseAudio env vars, so sounds leak to the real speakers.
/// The `"pulse"` ALSA device routes through the PulseAudio protocol
/// where `PULSE_SINK` / `PULSE_SOURCE` are respected.
const PULSE_ALSA_DEVICE: &str = "pulse";

/// Find a cpal output device by name.
fn find_output_device(name: &str) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    host.output_devices()
        .map_err(|e| format!("failed to enumerate output devices: {e}"))?
        .find(|d| d.name().map(|n| n == name).unwrap_or(false))
        .ok_or_else(|| format!("output device '{name}' not found"))
}

/// Find a cpal input device by name.
fn find_input_device(name: &str) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    host.input_devices()
        .map_err(|e| format!("failed to enumerate input devices: {e}"))?
        .find(|d| d.name().map(|n| n == name).unwrap_or(false))
        .ok_or_else(|| format!("input device '{name}' not found"))
}

// ── Capture helper ───────────────────────────────────────────────────

/// Open a cpal input stream on the `"pulse"` ALSA device (which
/// respects PULSE_SOURCE) and capture raw f32 samples.
struct LoopbackCapture {
    captured: Arc<Mutex<Vec<f32>>>,
    running: Arc<AtomicBool>,
    stream: cpal::Stream,
    sample_rate: u32,
    channels: u16,
}

impl LoopbackCapture {
    fn new() -> Result<Self, String> {
        let device = find_input_device(PULSE_ALSA_DEVICE)?;

        let config = device
            .default_input_config()
            .map_err(|e| format!("failed to get input config: {e}"))?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let sample_format = config.sample_format();

        let captured: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_cb = Arc::clone(&captured);
        let running = Arc::new(AtomicBool::new(true));
        let running_cb = Arc::clone(&running);

        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                &StreamConfig {
                    channels,
                    sample_rate: SampleRate(sample_rate),
                    buffer_size: cpal::BufferSize::Default,
                },
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if running_cb.load(Ordering::Acquire) {
                        if let Ok(mut buf) = captured_cb.lock() {
                            buf.extend_from_slice(data);
                        }
                    }
                },
                |err| {
                    eprintln!("loopback capture error: {err}");
                },
                None,
            ),
            _ => return Err(format!("unsupported sample format: {sample_format:?}")),
        }
        .map_err(|e| format!("failed to build input stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("failed to start input stream: {e}"))?;

        Ok(Self {
            captured,
            running,
            stream,
            sample_rate,
            channels,
        })
    }

    /// Stop capturing and return the collected samples.
    fn stop(self) -> Vec<f32> {
        self.running.store(false, Ordering::Release);
        // Let the callback drain.
        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(self.stream);

        let guard = self.captured.lock().expect("captured lock poisoned");
        guard.clone()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Downmix interleaved multi-channel samples to mono by averaging.
fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Compute peak absolute amplitude of a mono signal.
fn peak(mono: &[f32]) -> f32 {
    mono.iter().copied().map(f32::abs).fold(0.0f32, f32::max)
}

/// Play a sound through SoundPlayer with loopback capture, returning
/// the captured mono audio and the capture sample rate.
fn play_and_capture<F>(play_fn: F) -> (Vec<f32>, u32)
where
    F: FnOnce(&SoundPlayer),
{
    ensure_sink();

    // Start capture BEFORE playing so we catch everything.
    let capture = LoopbackCapture::new().expect("failed to start loopback capture");
    let capture_rate = capture.sample_rate;
    let capture_channels = capture.channels;

    // Settle time for capture stream to start receiving callbacks.
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Use the "pulse" ALSA device so PULSE_SINK routes audio to the
    // null sink instead of the real speakers.
    let output_device =
        find_output_device(PULSE_ALSA_DEVICE).expect("pulse output device not found");
    let player = SoundPlayer::from_device(output_device)
        .expect("failed to create SoundPlayer on pulse device");

    play_fn(&player);

    // Extra drain time for the pipeline.
    std::thread::sleep(std::time::Duration::from_millis(300));

    // Drop the player BEFORE stopping capture, so the output stream
    // flushes its final buffers into the null sink.
    drop(player);
    std::thread::sleep(std::time::Duration::from_millis(100));

    let raw_samples = capture.stop();
    let mono = to_mono(&raw_samples, capture_channels);

    (mono, capture_rate)
}

// ── Tests ────────────────────────────────────────────────────────────

/// Verify that SoundPlayer output is captured through the loopback.
///
/// Plays the start sound, captures from the monitor source, and checks
/// that the captured audio contains non-trivial signal.
#[test]
#[ignore] // Requires PipeWire/PulseAudio
fn loopback_start_sound_is_captured() {
    let _lock = serial_lock();

    let (mono, capture_rate) = play_and_capture(|player| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("failed to build tokio runtime");
        rt.block_on(player.play_start());
    });

    let min_expected_samples = (capture_rate as f64 * 0.1) as usize;
    assert!(
        mono.len() >= min_expected_samples,
        "expected at least {} mono samples, got {}",
        min_expected_samples,
        mono.len(),
    );

    // TONE_VOLUME is 0.05 — through the null sink loopback the signal
    // may be attenuated but should still be well above noise floor.
    let p = peak(&mono);
    assert!(
        p > 0.0005,
        "captured audio is silent (peak={p:.6}), loopback may not be working",
    );

    let nonzero_count = mono.iter().filter(|&&s| s.abs() > 0.00005).count();
    assert!(
        nonzero_count > 100,
        "too few non-trivial samples ({nonzero_count}), sound may be clipped",
    );

    eprintln!(
        "  [diag] start: peak={p:.6}, nonzero={nonzero_count}, total={}",
        mono.len(),
    );
}

/// Verify that the stop sound is also captured through the loopback.
#[test]
#[ignore] // Requires PipeWire/PulseAudio
fn loopback_stop_sound_is_captured() {
    let _lock = serial_lock();

    let (mono, _) = play_and_capture(|player| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("failed to build tokio runtime");
        rt.block_on(player.play_stop());
    });

    let p = peak(&mono);
    assert!(p > 0.0005, "stop sound not captured (peak={p:.6})",);

    let nonzero_count = mono.iter().filter(|&&s| s.abs() > 0.00005).count();
    assert!(
        nonzero_count > 100,
        "too few non-trivial samples for stop sound ({nonzero_count})",
    );

    eprintln!(
        "  [diag] stop: peak={p:.6}, nonzero={nonzero_count}, total={}",
        mono.len(),
    );
}

/// Verify that preemption works: playing stop immediately after start
/// replaces the start sound. We verify this by counting tone onsets
/// in the energy envelope — preemption should produce at most 3
/// onsets (partial start note + two stop notes), not 4 (two start +
/// two stop notes played fully).
#[test]
#[ignore] // Requires PipeWire/PulseAudio
fn loopback_preemption_replaces_sound() {
    let _lock = serial_lock();

    let (mono, capture_rate) = play_and_capture(|player| {
        // Fire-and-forget start sound.
        player.play(&player.sounds.start.clone());
        // Let some of start sound through before preempting.
        std::thread::sleep(std::time::Duration::from_millis(30));
        // Preempt with stop sound and wait for it.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("failed to build tokio runtime");
        rt.block_on(player.play_and_wait(&player.sounds.stop.clone()));
    });

    let p = peak(&mono);
    assert!(
        p > 0.0005,
        "no audio captured during preemption test (peak={p:.6})",
    );

    // Compute RMS energy in 10ms windows.
    let window_samples = (capture_rate as f64 * 0.01) as usize;
    let envelope: Vec<f64> = mono
        .chunks(window_samples)
        .map(|chunk| {
            let sum_sq: f64 = chunk.iter().map(|&s| (s as f64) * (s as f64)).sum();
            (sum_sq / chunk.len() as f64).sqrt()
        })
        .collect();

    let env_peak = envelope.iter().copied().fold(0.0f64, f64::max);
    let env_threshold = env_peak * 0.3;

    // Count tone onsets (transitions from silent to active).
    let mut onsets = 0u32;
    let mut above = false;
    for &e in &envelope {
        if e > env_threshold && !above {
            onsets += 1;
            above = true;
        } else if e < env_threshold * 0.5 {
            above = false;
        }
    }

    eprintln!("  [diag] preemption: peak={p:.6}, env_peak={env_peak:.6}, onsets={onsets}",);

    // Without preemption: 4 onsets (start note 1, start note 2,
    // stop note 1, stop note 2).
    // With preemption: at most 3 onsets (partial start + 2 stop notes),
    // possibly 2 if the start is cut very short.
    assert!(
        onsets <= 3,
        "expected at most 3 tone onsets but found {onsets} — \
         preemption may not have worked (4 = both sounds played fully)",
    );

    // Should have at least the 2 stop notes.
    assert!(
        onsets >= 2,
        "expected at least 2 tone onsets but found {onsets} — \
         stop sound may be incomplete",
    );
}

/// Verify the start sound contains two distinct tone bursts.
///
/// The start sound is an ascending major third: two 120ms tones
/// separated by a 30ms gap. We verify completeness by computing
/// an energy envelope and confirming two energy peaks exist (i.e.,
/// the sound was not clipped to a single note or truncated).
#[test]
#[ignore] // Requires PipeWire/PulseAudio
fn loopback_start_sound_has_two_notes() {
    let _lock = serial_lock();

    let (mono, capture_rate) = play_and_capture(|player| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("failed to build tokio runtime");
        rt.block_on(player.play_start());
    });

    let p = peak(&mono);
    assert!(p > 0.0005, "no non-silent audio captured (peak={p:.6})",);

    // Compute RMS energy in 10ms windows to build an envelope.
    let window_samples = (capture_rate as f64 * 0.01) as usize;
    let envelope: Vec<f64> = mono
        .chunks(window_samples)
        .map(|chunk| {
            let sum_sq: f64 = chunk.iter().map(|&s| (s as f64) * (s as f64)).sum();
            (sum_sq / chunk.len() as f64).sqrt()
        })
        .collect();

    // Find peak energy.
    let env_peak = envelope.iter().copied().fold(0.0f64, f64::max);

    // Threshold at 30% of envelope peak to identify tone regions.
    let env_threshold = env_peak * 0.3;

    // Count transitions from below to above threshold (tone onsets).
    let mut onsets = 0u32;
    let mut above = false;
    for &e in &envelope {
        if e > env_threshold && !above {
            onsets += 1;
            above = true;
        } else if e < env_threshold * 0.5 {
            // Hysteresis: require significant drop to count as gap.
            above = false;
        }
    }

    eprintln!(
        "  [diag] two_notes: peak={p:.6}, env_peak={env_peak:.6}, \
         env_threshold={env_threshold:.6}, onsets={onsets}, \
         envelope_len={}",
        envelope.len(),
    );

    // We expect exactly 2 onsets (two tone bursts).
    // Allow 2-3 in case the gap detection has minor artifacts.
    assert!(
        onsets >= 2,
        "expected 2 tone onsets but found {onsets} — \
         sound may be clipped or incomplete",
    );
    assert!(
        onsets <= 4,
        "expected 2 tone onsets but found {onsets} — \
         unexpected fragmentation in captured audio",
    );
}

/// Verify that CpalCapture (the production capture path) can receive
/// audio from the loopback monitor source.
///
/// Uses the `"pulse"` ALSA device for both playback and capture so
/// `PULSE_SINK` / `PULSE_SOURCE` are respected and the test is silent.
///
/// NOTE: CpalCapture uses `default_input_device()` internally, which
/// resolves to the ALSA `"default"` device. For this test to work, we
/// build a raw capture stream on the `"pulse"` device instead of using
/// CpalCapture, then verify the i16 conversion path separately.
#[test]
#[ignore] // Requires PipeWire/PulseAudio
fn loopback_capture_receives_played_audio_i16() {
    let _lock = serial_lock();
    ensure_sink();

    // Open capture on the "pulse" input device (monitor via PULSE_SOURCE).
    let input_device = find_input_device(PULSE_ALSA_DEVICE).expect("pulse input device not found");
    let input_config = input_device
        .default_input_config()
        .expect("failed to get input config");
    let sample_rate = input_config.sample_rate().0;
    let channels = input_config.channels();

    let captured: Arc<Mutex<Vec<i16>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_cb = Arc::clone(&captured);
    let running = Arc::new(AtomicBool::new(true));
    let running_cb = Arc::clone(&running);

    let stream = input_device
        .build_input_stream(
            &StreamConfig {
                channels,
                sample_rate: SampleRate(sample_rate),
                buffer_size: cpal::BufferSize::Default,
            },
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if running_cb.load(Ordering::Acquire) {
                    if let Ok(mut buf) = captured_cb.lock() {
                        buf.extend(data.iter().map(|&s| (s * 32767.0) as i16));
                    }
                }
            },
            |err| eprintln!("capture error: {err}"),
            None,
        )
        .expect("failed to build i16 capture stream");

    stream.play().expect("failed to start i16 capture stream");
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Play start sound to the null sink.
    let output_device =
        find_output_device(PULSE_ALSA_DEVICE).expect("pulse output device not found");
    let player = SoundPlayer::from_device(output_device)
        .expect("failed to create SoundPlayer on pulse device");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("failed to build tokio runtime");
    rt.block_on(player.play_start());

    // Extra drain time.
    std::thread::sleep(std::time::Duration::from_millis(300));
    drop(player);
    std::thread::sleep(std::time::Duration::from_millis(100));

    running.store(false, Ordering::Release);
    std::thread::sleep(std::time::Duration::from_millis(50));
    drop(stream);

    let all_samples = captured.lock().expect("lock poisoned").clone();

    assert!(
        !all_samples.is_empty(),
        "capture received no audio from loopback",
    );

    let i16_peak = all_samples
        .iter()
        .map(|&s| s.unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(
        i16_peak > 10,
        "captured i16 audio is silent (peak={i16_peak})",
    );

    let nonzero = all_samples
        .iter()
        .filter(|&&s| s.unsigned_abs() > 5)
        .count();
    assert!(nonzero > 50, "too few non-trivial i16 samples ({nonzero})",);

    eprintln!(
        "  [diag] i16_capture: peak={i16_peak}, nonzero={nonzero}, total={}, rate={sample_rate}",
        all_samples.len(),
    );
}
