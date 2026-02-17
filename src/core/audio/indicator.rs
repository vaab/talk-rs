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

use crate::core::error::TalkError;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Condvar, Mutex};
use tokio_util::sync::CancellationToken;

/// Maximum time to wait for the audio output device to start pulling
/// samples before giving up.
const WARMUP_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

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
    pub fn new() -> Result<Self, TalkError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| TalkError::Audio("no default audio output device found".to_string()))?;

        let config = device
            .default_output_config()
            .map_err(|e| TalkError::Audio(format!("failed to get output device config: {}", e)))?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let sounds = IndicatorSounds::new(sample_rate);

        let state = Arc::new(Mutex::new(PlaybackState::new()));
        let state_cb = Arc::clone(&state);

        // Warmup synchronisation: the output callback signals when the
        // audio device is actually pulling samples so we can guarantee
        // the pipeline is live before playing any sound.
        let warmup_pair = Arc::new((Mutex::new(false), Condvar::new()));
        let warmup_cb = Arc::clone(&warmup_pair);

        let stream = device
            .build_output_stream(
                &cpal::StreamConfig {
                    channels: config.channels(),
                    sample_rate: config.sample_rate(),
                    buffer_size: cpal::BufferSize::Default,
                },
                move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // Signal warmup on first callback invocation.
                    {
                        let (lock, cvar) = &*warmup_cb;
                        if let Ok(mut ready) = lock.lock() {
                            if !*ready {
                                *ready = true;
                                cvar.notify_one();
                            }
                        }
                    }

                    // try_lock: if the caller is swapping buffers right now,
                    // output silence for this callback (~5ms) — inaudible.
                    if let Ok(mut guard) = state_cb.try_lock() {
                        // Mono samples → duplicate to all channels
                        let frames = output.len() / channels;
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
                        // Lock contended — output silence
                        for sample in output.iter_mut() {
                            *sample = 0.0;
                        }
                    }
                },
                |err| {
                    log::error!("audio output error: {}", err);
                },
                None, // no timeout
            )
            .map_err(|e| TalkError::Audio(format!("failed to create output stream: {}", e)))?;

        stream
            .play()
            .map_err(|e| TalkError::Audio(format!("failed to start output stream: {}", e)))?;

        // Block until the audio device has actually invoked the callback
        // at least once, proving the output pipeline is live. This avoids
        // the first sound being partially lost to device startup latency.
        {
            let (lock, cvar) = &*warmup_pair;
            let guard = lock
                .lock()
                .map_err(|_| TalkError::Audio("warmup lock poisoned".to_string()))?;
            if !*guard {
                let (guard, timeout_result) = cvar
                    .wait_timeout(guard, WARMUP_TIMEOUT)
                    .map_err(|_| TalkError::Audio("warmup condvar poisoned".to_string()))?;
                if !*guard && timeout_result.timed_out() {
                    log::warn!(
                        "audio output warmup timed out after {}ms — start sound may be clipped",
                        WARMUP_TIMEOUT.as_millis()
                    );
                }
            }
            log::debug!("audio output pipeline warm");
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

    /// Play a sound and wait for it to finish.
    pub async fn play_and_wait(&self, samples: &[f32]) {
        let duration_secs = samples.len() as f32 / self.sample_rate as f32;
        self.play(samples);
        // Add a small margin to ensure playback completes
        let wait = std::time::Duration::from_secs_f32(duration_secs + 0.05);
        tokio::time::sleep(wait).await;
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
        // First sample should be near zero (fade in)
        assert!(
            samples[0].abs() < 0.05,
            "first sample should be near zero (fade-in)"
        );
        // Last sample should be near zero (fade out)
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

        // Start and stop should be same length (two tones + gap)
        assert_eq!(sounds.start.len(), sounds.stop.len());

        // Boop should be same structure (two tones + gap) but different gap
        let expected_start_len =
            (TONE_DURATION * 44_100.0) as usize * 2 + (NOTE_GAP * 44_100.0) as usize;
        assert_eq!(sounds.start.len(), expected_start_len);

        let expected_boop_len =
            (TONE_DURATION * 44_100.0) as usize * 2 + (BOOP_GAP * 44_100.0) as usize;
        assert_eq!(sounds.boop.len(), expected_boop_len);
    }

    #[test]
    fn test_start_sound_ascending() {
        // Start sound: first half should be lower freq than second half
        let sounds = IndicatorSounds::new(44_100);
        let tone_len = (TONE_DURATION * 44_100.0) as usize;

        // Count zero-crossings as a proxy for frequency
        let first_tone = &sounds.start[..tone_len];
        let second_tone_start = tone_len + (NOTE_GAP * 44_100.0) as usize;
        let second_tone = &sounds.start[second_tone_start..];

        let crossings = |s: &[f32]| -> usize {
            s.windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count()
        };

        let first_crossings = crossings(first_tone);
        let second_crossings = crossings(second_tone);

        // 458 Hz has more zero-crossings than 364 Hz in the same duration
        assert!(
            second_crossings > first_crossings,
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

        let first_crossings = crossings(first_tone);
        let second_crossings = crossings(second_tone);

        assert!(
            first_crossings > second_crossings,
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

    #[test]
    fn test_playback_state_replace_preempts() {
        let mut state = PlaybackState::new();

        // Load first sound
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
}
