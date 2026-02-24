//! Sound indicators for recording feedback.
//!
//! Provides a single-channel [`SoundPlayer`] that plays short synthesized tones
//! with preemption: starting a new sound immediately replaces any in-progress one.
//! This avoids overlapping sounds (e.g., boop + stop playing simultaneously).
//!
//! Tone parameters are inspired by 0k-memo:
//! - **Start**: ascending major third (364 Hz → 458 Hz)
//! - **Stop**: descending major third (458 Hz → 364 Hz)
//! - **Boop**: soft double-pulse at 364 Hz (recording heartbeat)

use crate::error::TalkError;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

/// Maximum wall-clock time for warmup/drain before giving up.
const WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

/// Duration of silence (in seconds) to pump through the pipeline before
/// considering the output device primed. Covers PulseAudio's default
/// ring-buffer latency on Linux.
const WARMUP_SECS: f64 = 0.20;

/// Duration of silence (in seconds) to let the device drain after the
/// last sample has been consumed by the callback. Ensures the hardware
/// actually emits the final audio before we move on.
const DRAIN_SECS: f64 = 0.15;

// ── Tone parameters ──────────────────────────────────────────────────

/// Base frequency for indicator tones (Hz).
const BASE_FREQ: f32 = 364.0;
/// Major-third above the base (Hz).
const THIRD_FREQ: f32 = 458.0;

/// Duration of a single tone burst (seconds).
const TONE_DURATION: f32 = 0.12;
/// Half-sine fade-in duration (seconds).
const FADE_IN: f32 = 0.01;
/// Half-sine fade-out duration (seconds).
const FADE_OUT: f32 = 0.02;

/// Gap between the two notes in start/stop sounds (seconds).
const NOTE_GAP: f32 = 0.03;
/// Gap between the two pulses in a boop (seconds).
const BOOP_GAP: f32 = 0.045;

/// Volume for start/stop tones (0.0–1.0).
const TONE_VOLUME: f32 = 0.05;
/// Volume for boop tones (much quieter).
const BOOP_VOLUME: f32 = 0.01;

// ── Tone synthesis ───────────────────────────────────────────────────

/// Generate a sine-wave tone with half-sine fade envelope.
fn generate_tone(freq: f32, duration_secs: f32, volume: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let fade_in_samples = (FADE_IN * sample_rate as f32) as usize;
    let fade_out_samples = (FADE_OUT * sample_rate as f32) as usize;

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let sine = (2.0 * std::f32::consts::PI * freq * t).sin();

            // Half-sine envelope
            let envelope = if i < fade_in_samples {
                // Attack: half-sine curve 0→1
                let phase = i as f32 / fade_in_samples as f32;
                (std::f32::consts::FRAC_PI_2 * phase).sin()
            } else if i >= num_samples - fade_out_samples {
                // Release: half-sine curve 1→0
                let remaining = (num_samples - i) as f32 / fade_out_samples as f32;
                (std::f32::consts::FRAC_PI_2 * remaining).sin()
            } else {
                1.0
            };

            sine * envelope * volume
        })
        .collect()
}

/// Generate silence of a given duration.
fn generate_silence(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    vec![0.0; (duration_secs * sample_rate as f32) as usize]
}

/// Pre-rendered indicator sounds.
pub struct IndicatorSounds {
    pub start: Vec<f32>,
    pub stop: Vec<f32>,
    pub boop: Vec<f32>,
}

impl IndicatorSounds {
    /// Synthesize all indicator sounds at the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        let start = Self::build_start(sample_rate);
        let stop = Self::build_stop(sample_rate);
        let boop = Self::build_boop(sample_rate);
        Self { start, stop, boop }
    }

    /// Ascending major third: 364 Hz → 458 Hz.
    fn build_start(sr: u32) -> Vec<f32> {
        let mut samples = generate_tone(BASE_FREQ, TONE_DURATION, TONE_VOLUME, sr);
        samples.extend(generate_silence(NOTE_GAP, sr));
        samples.extend(generate_tone(THIRD_FREQ, TONE_DURATION, TONE_VOLUME, sr));
        samples
    }

    /// Descending major third: 458 Hz → 364 Hz.
    fn build_stop(sr: u32) -> Vec<f32> {
        let mut samples = generate_tone(THIRD_FREQ, TONE_DURATION, TONE_VOLUME, sr);
        samples.extend(generate_silence(NOTE_GAP, sr));
        samples.extend(generate_tone(BASE_FREQ, TONE_DURATION, TONE_VOLUME, sr));
        samples
    }

    /// Double-pulse boop at base frequency (soft).
    fn build_boop(sr: u32) -> Vec<f32> {
        let mut samples = generate_tone(BASE_FREQ, TONE_DURATION, BOOP_VOLUME, sr);
        samples.extend(generate_silence(BOOP_GAP, sr));
        samples.extend(generate_tone(BASE_FREQ, TONE_DURATION, BOOP_VOLUME, sr));
        samples
    }
}

// ── Playback state (shared between caller and cpal callback) ─────────

struct PlaybackState {
    /// Current sample buffer being played.
    samples: Vec<f32>,
    /// Read position within the buffer.
    position: usize,
}

impl PlaybackState {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            position: 0,
        }
    }

    /// Replace the current sound, restarting from the beginning.
    fn replace(&mut self, samples: Vec<f32>) {
        self.samples = samples;
        self.position = 0;
    }

    /// Fill `output` with mono samples, advancing the read position.
    /// Outputs silence once the buffer is exhausted.
    #[cfg(test)]
    fn fill(&mut self, output: &mut [f32]) {
        for sample in output.iter_mut() {
            if self.position < self.samples.len() {
                *sample = self.samples[self.position];
                self.position += 1;
            } else {
                *sample = 0.0;
            }
        }
    }
}

// ── Output callback ──────────────────────────────────────────────────

/// Fill an output buffer from the playback state.
///
/// Reads mono samples from `state`, duplicates them across `channels`,
/// and increments `frames_output` by the number of frames processed.
/// If the state lock is contended, outputs silence for this callback
/// (~5 ms at typical buffer sizes) — inaudible.
///
/// This is the core logic of the cpal output callback, extracted as a
/// free function for testability.
fn fill_output_buffer(
    output: &mut [f32],
    channels: usize,
    state: &Mutex<PlaybackState>,
    frames_output: &AtomicU64,
) {
    let frames = output.len() / channels;

    if let Ok(mut guard) = state.try_lock() {
        for frame in 0..frames {
            let sample = if guard.position < guard.samples.len() {
                let s = guard.samples[guard.position];
                guard.position += 1;
                s
            } else {
                0.0
            };
            for ch in 0..channels {
                output[frame * channels + ch] = sample;
            }
        }
    } else {
        // Lock contended — output silence.
        for sample in output.iter_mut() {
            *sample = 0.0;
        }
    }

    // Unconditionally count frames — used for warmup timing.
    frames_output.fetch_add(frames as u64, Ordering::Release);
}

// ── Playback completion ──────────────────────────────────────────────

/// Wait until the callback has consumed `target_len` samples, then
/// sleep for the pipeline drain period.
///
/// Uses position tracking for completion detection with a wall-clock
/// deadline as fallback. If the callback stalls (device sleeping,
/// PulseAudio corked), the wall-clock limit ensures we never block
/// longer than the sound's actual duration plus drain overhead — the
/// pipeline continues even if a sound is lost.
async fn wait_for_playback(state: &Mutex<PlaybackState>, sample_rate: u32, target_len: usize) {
    if target_len == 0 {
        return;
    }

    let duration_secs = target_len as f64 / f64::from(sample_rate);
    // Wall-clock upper bound: actual duration + drain + small margin.
    // This is the MAXIMUM time we block — roughly matching what a
    // simple sleep-based approach would wait, so the pipeline is never
    // stalled even when the audio device is unresponsive.
    let wall_limit = std::time::Duration::from_secs_f64(duration_secs + DRAIN_SECS + 0.05);
    let deadline = tokio::time::Instant::now() + wall_limit;

    loop {
        let consumed = state.lock().map(|g| g.position).unwrap_or(target_len);
        if consumed >= target_len {
            // Callback consumed everything — drain for pipeline latency.
            tokio::time::sleep(std::time::Duration::from_secs_f64(DRAIN_SECS)).await;
            log::debug!(
                "play_and_wait: playback complete ({} samples), drained {:.0}ms",
                target_len,
                DRAIN_SECS * 1000.0,
            );
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            log::warn!(
                "play_and_wait: wall-clock timeout ({:.2}s), position {}/{}",
                wall_limit.as_secs_f64(),
                consumed,
                target_len,
            );
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
}

// ── SoundPlayer ──────────────────────────────────────────────────────

/// Single-channel audio player with preemption.
///
/// Owns a persistent cpal output stream. Calling [`play`](SoundPlayer::play)
/// immediately replaces any in-progress sound — no overlap, no mixing.
pub struct SoundPlayer {
    state: Arc<Mutex<PlaybackState>>,
    /// Pre-rendered indicator sounds.
    pub sounds: IndicatorSounds,
    /// Sample rate of the output device (for duration calculations).
    sample_rate: u32,
    // The stream is kept alive by this field; dropping SoundPlayer stops it.
    _stream: cpal::Stream,
}

impl SoundPlayer {
    /// Create a new player using the default output device.
    ///
    /// After starting the stream, blocks until the output pipeline has
    /// processed at least [`WARMUP_SECS`] worth of frames. This
    /// deterministically primes PulseAudio's internal ring-buffer so the
    /// first real sound is not clipped.
    pub fn new() -> Result<Self, TalkError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| TalkError::Audio("no default audio output device found".to_string()))?;
        Self::from_device(device)
    }

    /// Create a new player targeting a specific output device.
    ///
    /// Useful for tests (routing to a virtual null sink) or for
    /// selecting a non-default output device.
    pub fn from_device(device: cpal::Device) -> Result<Self, TalkError> {
        let config = device
            .default_output_config()
            .map_err(|e| TalkError::Audio(format!("failed to get output device config: {}", e)))?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let sounds = IndicatorSounds::new(sample_rate);

        let state = Arc::new(Mutex::new(PlaybackState::new()));
        let state_cb = Arc::clone(&state);

        // Frame counter for warmup detection — only needed during
        // construction, not stored on Self.
        let frames_output = Arc::new(AtomicU64::new(0));
        let frames_output_cb = Arc::clone(&frames_output);

        let stream = device
            .build_output_stream(
                &cpal::StreamConfig {
                    channels: config.channels(),
                    sample_rate: config.sample_rate(),
                    buffer_size: cpal::BufferSize::Default,
                },
                move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    fill_output_buffer(output, channels, &state_cb, &frames_output_cb);
                },
                |err| {
                    log::error!("audio output error: {}", err);
                },
                None,
            )
            .map_err(|e| TalkError::Audio(format!("failed to create output stream: {}", e)))?;

        stream
            .play()
            .map_err(|e| TalkError::Audio(format!("failed to start output stream: {}", e)))?;

        // Poll until the callback has output enough frames to fill
        // PulseAudio's pipeline, ensuring the first real sound is not
        // lost to device startup latency.
        let warmup_frames = (WARMUP_SECS * f64::from(sample_rate)) as u64;
        let deadline = std::time::Instant::now() + WAIT_TIMEOUT;
        loop {
            let emitted = frames_output.load(Ordering::Acquire);
            if emitted >= warmup_frames {
                log::debug!(
                    "audio output pipeline warm ({} frames emitted, needed {})",
                    emitted,
                    warmup_frames,
                );
                break;
            }
            if std::time::Instant::now() >= deadline {
                log::warn!(
                    "audio output warmup timed out after {}ms ({} / {} frames) \
                     — start sound may be clipped",
                    WAIT_TIMEOUT.as_millis(),
                    emitted,
                    warmup_frames,
                );
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        Ok(Self {
            state,
            sounds,
            sample_rate,
            _stream: stream,
        })
    }

    /// Play a sound, immediately preempting any in-progress sound.
    pub fn play(&self, samples: &[f32]) {
        if let Ok(mut guard) = self.state.lock() {
            guard.replace(samples.to_vec());
        }
    }

    /// Play a sound and wait until the output device has fully emitted it.
    ///
    /// Uses position tracking to detect when the callback has consumed
    /// all samples, then sleeps for [`DRAIN_SECS`] to let the output
    /// pipeline flush. A wall-clock deadline ensures this never blocks
    /// longer than the sound's natural duration plus drain overhead,
    /// even if the audio callback stalls.
    pub async fn play_and_wait(&self, samples: &[f32]) {
        self.play(samples);
        wait_for_playback(&self.state, self.sample_rate, samples.len()).await;
    }

    /// Play the start sound (ascending major third) and wait.
    pub async fn play_start(&self) {
        // Clone to avoid borrow issues — these are small buffers
        let samples = self.sounds.start.clone();
        self.play_and_wait(&samples).await;
    }

    /// Play the stop sound (descending major third), preempting any boop.
    pub async fn play_stop(&self) {
        let samples = self.sounds.stop.clone();
        self.play_and_wait(&samples).await;
    }

    /// Start a background boop loop. Returns a cancellation token to stop it.
    ///
    /// The boop plays every `interval` and is immediately preempted if a
    /// new sound (e.g., stop) is played on this player.
    ///
    /// The loop runs in a tokio task that only holds the shared playback
    /// state (not the cpal stream), keeping it `Send`.
    pub fn start_boop_loop(&self, interval: std::time::Duration) -> CancellationToken {
        let token = CancellationToken::new();
        let token_clone = token.clone();
        let state = Arc::clone(&self.state);
        let boop_samples = self.sounds.boop.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {
                        if let Ok(mut guard) = state.lock() {
                            guard.replace(boop_samples.clone());
                        }
                    }
                    _ = token_clone.cancelled() => {
                        break;
                    }
                }
            }
        });

        token
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    // ── Tone synthesis tests ─────────────────────────────────────────

    #[test]
    fn test_generate_tone_length() {
        let samples = generate_tone(440.0, 0.12, 1.0, 44_100);
        let expected = (0.12 * 44_100.0) as usize;
        assert_eq!(samples.len(), expected);
    }

    #[test]
    fn test_generate_tone_not_silent() {
        let samples = generate_tone(440.0, 0.12, 1.0, 44_100);
        let max = samples.iter().copied().fold(0.0_f32, f32::max);
        assert!(max > 0.5, "tone should have significant amplitude");
    }

    #[test]
    fn test_generate_tone_fade_envelope() {
        let samples = generate_tone(440.0, 0.12, 1.0, 44_100);
        assert!(
            samples[0].abs() < 0.05,
            "first sample should be near zero (fade-in)"
        );
        assert!(
            samples.last().copied().unwrap_or(1.0).abs() < 0.05,
            "last sample should be near zero (fade-out)"
        );
    }

    #[test]
    fn test_generate_tone_volume_scaling() {
        let loud = generate_tone(440.0, 0.12, 1.0, 44_100);
        let quiet = generate_tone(440.0, 0.12, 0.01, 44_100);

        let max_loud = loud.iter().copied().fold(0.0_f32, f32::max);
        let max_quiet = quiet.iter().copied().fold(0.0_f32, f32::max);

        assert!(
            max_loud > max_quiet * 50.0,
            "loud tone should be much louder"
        );
    }

    #[test]
    fn test_generate_silence() {
        let silence = generate_silence(0.05, 44_100);
        assert_eq!(silence.len(), (0.05 * 44_100.0) as usize);
        assert!(silence.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_indicator_sounds_structure() {
        let sounds = IndicatorSounds::new(44_100);

        assert_eq!(sounds.start.len(), sounds.stop.len());

        let expected_start_len =
            (TONE_DURATION * 44_100.0) as usize * 2 + (NOTE_GAP * 44_100.0) as usize;
        assert_eq!(sounds.start.len(), expected_start_len);

        let expected_boop_len =
            (TONE_DURATION * 44_100.0) as usize * 2 + (BOOP_GAP * 44_100.0) as usize;
        assert_eq!(sounds.boop.len(), expected_boop_len);
    }

    #[test]
    fn test_start_sound_ascending() {
        let sounds = IndicatorSounds::new(44_100);
        let tone_len = (TONE_DURATION * 44_100.0) as usize;

        let first_tone = &sounds.start[..tone_len];
        let second_tone_start = tone_len + (NOTE_GAP * 44_100.0) as usize;
        let second_tone = &sounds.start[second_tone_start..];

        let crossings = |s: &[f32]| -> usize {
            s.windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count()
        };

        assert!(
            crossings(second_tone) > crossings(first_tone),
            "start sound should be ascending (low→high)"
        );
    }

    #[test]
    fn test_stop_sound_descending() {
        let sounds = IndicatorSounds::new(44_100);
        let tone_len = (TONE_DURATION * 44_100.0) as usize;

        let first_tone = &sounds.stop[..tone_len];
        let second_tone_start = tone_len + (NOTE_GAP * 44_100.0) as usize;
        let second_tone = &sounds.stop[second_tone_start..];

        let crossings = |s: &[f32]| -> usize {
            s.windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count()
        };

        assert!(
            crossings(first_tone) > crossings(second_tone),
            "stop sound should be descending (high→low)"
        );
    }

    #[test]
    fn test_boop_is_quieter_than_start() {
        let sounds = IndicatorSounds::new(44_100);

        let max_start = sounds.start.iter().copied().fold(0.0_f32, f32::max);
        let max_boop = sounds.boop.iter().copied().fold(0.0_f32, f32::max);

        assert!(
            max_start > max_boop * 3.0,
            "boop should be much quieter than start"
        );
    }

    // ── PlaybackState unit tests ─────────────────────────────────────

    #[test]
    fn test_playback_state_replace_preempts() {
        let mut state = PlaybackState::new();

        state.replace(vec![1.0, 2.0, 3.0]);
        let mut out = [0.0; 2];
        state.fill(&mut out);
        assert_eq!(out, [1.0, 2.0]);
        assert_eq!(state.position, 2);

        // Replace mid-playback — position resets
        state.replace(vec![9.0, 8.0]);
        let mut out2 = [0.0; 2];
        state.fill(&mut out2);
        assert_eq!(out2, [9.0, 8.0]);
    }

    #[test]
    fn test_playback_state_silence_after_exhausted() {
        let mut state = PlaybackState::new();
        state.replace(vec![1.0]);

        let mut out = [0.0; 3];
        state.fill(&mut out);
        assert_eq!(out, [1.0, 0.0, 0.0]);
    }

    // ── fill_output_buffer tests ─────────────────────────────────────

    #[test]
    fn test_fill_output_buffer_plays_samples() {
        let state = Arc::new(Mutex::new(PlaybackState::new()));
        let frames_output = Arc::new(AtomicU64::new(0));

        state.lock().unwrap().replace(vec![0.5, -0.3, 0.7]);

        let mut output = vec![0.0f32; 4]; // 4 frames, mono
        fill_output_buffer(&mut output, 1, &state, &frames_output);

        assert_eq!(output, [0.5, -0.3, 0.7, 0.0]);
        assert_eq!(state.lock().unwrap().position, 3);
    }

    #[test]
    fn test_fill_output_buffer_multichannel_duplication() {
        let state = Arc::new(Mutex::new(PlaybackState::new()));
        let frames_output = Arc::new(AtomicU64::new(0));

        state.lock().unwrap().replace(vec![0.5, -0.3]);

        // 2 frames × 2 channels = 4 samples in output buffer
        let mut output = vec![0.0f32; 4];
        fill_output_buffer(&mut output, 2, &state, &frames_output);

        // Each mono sample duplicated to both channels
        assert_eq!(output, [0.5, 0.5, -0.3, -0.3]);
        // frames_output counts FRAMES, not samples
        assert_eq!(frames_output.load(Ordering::Acquire), 2);
    }

    #[test]
    fn test_fill_output_buffer_increments_frames_output() {
        let state = Arc::new(Mutex::new(PlaybackState::new()));
        let frames_output = Arc::new(AtomicU64::new(0));

        let mut output = vec![0.0f32; 8]; // 8 frames, mono
        fill_output_buffer(&mut output, 1, &state, &frames_output);
        assert_eq!(frames_output.load(Ordering::Acquire), 8);

        // Another call accumulates
        let mut output2 = vec![0.0f32; 4];
        fill_output_buffer(&mut output2, 1, &state, &frames_output);
        assert_eq!(frames_output.load(Ordering::Acquire), 12);
    }

    #[test]
    fn test_fill_output_buffer_contended_lock_outputs_silence() {
        let state = Arc::new(Mutex::new(PlaybackState::new()));
        let frames_output = Arc::new(AtomicU64::new(0));

        state.lock().unwrap().replace(vec![1.0, 2.0, 3.0]);

        // Hold the lock to simulate contention
        let _guard = state.lock().unwrap();

        let mut output = vec![0.0f32; 3];
        fill_output_buffer(&mut output, 1, &state, &frames_output);

        // Should be all silence because lock was contended
        assert_eq!(output, [0.0, 0.0, 0.0]);
        // But frames_output STILL increments (unconditional)
        assert_eq!(frames_output.load(Ordering::Acquire), 3);
    }

    #[test]
    fn test_fill_output_buffer_silence_after_samples_exhausted() {
        let state = Arc::new(Mutex::new(PlaybackState::new()));
        let frames_output = Arc::new(AtomicU64::new(0));

        state.lock().unwrap().replace(vec![1.0]);

        let mut output = vec![0.0f32; 4];
        fill_output_buffer(&mut output, 1, &state, &frames_output);

        assert_eq!(output, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(state.lock().unwrap().position, 1);
    }

    // ── CallbackDriver: simulates cpal output in a thread ────────────
    //
    // Calls fill_output_buffer() periodically, capturing all output
    // samples. This lets us test the full playback pipeline (warmup,
    // play, drain, preemption) without real audio hardware.

    struct CallbackDriver {
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<Vec<f32>>>,
    }

    impl CallbackDriver {
        fn start(
            state: Arc<Mutex<PlaybackState>>,
            frames_output: Arc<AtomicU64>,
            channels: usize,
            frames_per_callback: usize,
            interval: std::time::Duration,
        ) -> Self {
            let stop = Arc::new(AtomicBool::new(false));
            let stop_cb = Arc::clone(&stop);

            let thread = std::thread::spawn(move || {
                let mut captured = Vec::new();
                while !stop_cb.load(Ordering::Acquire) {
                    let mut buffer = vec![0.0f32; frames_per_callback * channels];
                    fill_output_buffer(&mut buffer, channels, &state, &frames_output);
                    captured.extend_from_slice(&buffer);
                    std::thread::sleep(interval);
                }
                captured
            });

            Self {
                stop,
                thread: Some(thread),
            }
        }

        fn stop(mut self) -> Vec<f32> {
            self.stop.store(true, Ordering::Release);
            self.thread
                .take()
                .expect("thread already joined")
                .join()
                .expect("callback thread panicked")
        }
    }

    impl Drop for CallbackDriver {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    // ── wait_for_playback tests ──────────────────────────────────────

    /// With an active callback, wait_for_playback detects completion
    /// via position tracking and returns output matching the input.
    #[tokio::test]
    async fn test_wait_for_playback_completes_with_active_callback() {
        let state = Arc::new(Mutex::new(PlaybackState::new()));
        let frames_output = Arc::new(AtomicU64::new(0));
        let sample_rate = 48_000u32;

        // 4800 samples = 100 ms at 48 kHz
        let input = vec![0.5f32; 4800];
        let target_len = input.len();
        state.lock().unwrap().replace(input);

        // 256 frames every 5 ms ≈ 51 200 frames/s
        let driver = CallbackDriver::start(
            Arc::clone(&state),
            Arc::clone(&frames_output),
            1,
            256,
            std::time::Duration::from_millis(5),
        );

        let start = std::time::Instant::now();
        wait_for_playback(&state, sample_rate, target_len).await;
        let elapsed = start.elapsed();

        let pos = state.lock().unwrap().position;
        assert!(
            pos >= target_len,
            "position {} should be >= {}",
            pos,
            target_len,
        );

        // ~100 ms playback + 150 ms drain = ~250 ms
        assert!(
            elapsed.as_millis() >= 100,
            "should include drain time, took {:?}",
            elapsed,
        );
        assert!(
            elapsed.as_millis() < 1000,
            "should complete well within 1 s, took {:?}",
            elapsed,
        );

        // Verify captured output matches input
        let output = driver.stop();
        for (i, &s) in output.iter().take(target_len).enumerate() {
            assert!(
                (s - 0.5).abs() < f32::EPSILON,
                "output[{}] should be 0.5, got {}",
                i,
                s,
            );
        }
    }

    /// Without a callback, position never advances. Must fall back to
    /// the wall-clock deadline (duration + drain + margin) instead of
    /// blocking for WAIT_TIMEOUT (1 s).
    #[tokio::test]
    async fn test_wait_for_playback_wall_clock_fallback_without_callback() {
        let state = Arc::new(Mutex::new(PlaybackState::new()));

        // 4800 samples = 100 ms at 48 kHz
        state.lock().unwrap().replace(vec![0.5f32; 4800]);

        let start = std::time::Instant::now();
        wait_for_playback(&state, 48_000, 4800).await;
        let elapsed = start.elapsed();

        // Wall-clock limit: 100 + 150 + 50 = 300 ms
        assert!(
            elapsed.as_millis() < 600,
            "wall-clock fallback should fire within 600 ms, took {:?}",
            elapsed,
        );
        assert!(
            elapsed.as_millis() >= 200,
            "should wait at least duration + drain, took {:?}",
            elapsed,
        );
    }

    #[tokio::test]
    async fn test_wait_for_playback_empty_returns_immediately() {
        let state = Arc::new(Mutex::new(PlaybackState::new()));

        let start = std::time::Instant::now();
        wait_for_playback(&state, 48_000, 0).await;
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 10,
            "empty playback should return immediately, took {:?}",
            elapsed,
        );
    }

    /// Preempting a playing sound replaces it mid-stream.
    #[tokio::test]
    async fn test_preemption_replaces_in_progress_sound() {
        let state = Arc::new(Mutex::new(PlaybackState::new()));
        let frames_output = Arc::new(AtomicU64::new(0));

        // Load a long sound (10 000 samples)
        state.lock().unwrap().replace(vec![1.0f32; 10_000]);

        let driver = CallbackDriver::start(
            Arc::clone(&state),
            Arc::clone(&frames_output),
            1,
            256,
            std::time::Duration::from_millis(2),
        );

        // Let it play some of the first sound
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Preempt with a short replacement
        state.lock().unwrap().replace(vec![-0.5f32; 100]);

        // Wait for the short replacement
        wait_for_playback(&state, 48_000, 100).await;

        let output = driver.stop();

        let has_original = output.iter().any(|&s| (s - 1.0).abs() < f32::EPSILON);
        let has_replacement = output.iter().any(|&s| (s - (-0.5)).abs() < f32::EPSILON);
        assert!(has_original, "should have played some of the original");
        assert!(has_replacement, "should have played the replacement");
    }

    /// The key correctness property: play_and_wait never blocks longer
    /// than (duration + DRAIN_SECS + margin), regardless of whether
    /// the callback is running.
    #[tokio::test]
    async fn test_wait_for_playback_timing_is_bounded() {
        let state = Arc::new(Mutex::new(PlaybackState::new()));

        // Test several durations WITHOUT a callback — worst case
        for &duration_ms in &[50u64, 100, 200, 300] {
            let n_samples = (48_000u64 * duration_ms / 1000) as usize;
            state.lock().unwrap().replace(vec![0.1f32; n_samples]);

            let start = std::time::Instant::now();
            wait_for_playback(&state, 48_000, n_samples).await;
            let elapsed = start.elapsed();

            // Upper bound: duration + DRAIN(150) + margin(50) + scheduling(100)
            let upper_ms = duration_ms + 150 + 50 + 100;
            assert!(
                elapsed.as_millis() < upper_ms as u128,
                "{} ms sound: elapsed {:?} should be < {} ms",
                duration_ms,
                elapsed,
                upper_ms,
            );
        }
    }
}
