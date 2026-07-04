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
//!   [`crate::record::player`]), which delegates its stream + transport
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
    // Dropping this stops the stream.
    _stream: cpal::Stream,
}

impl AudioPlayer {
    /// Open the default output device and start a silent stream.
    pub fn new() -> Result<Self, TalkError> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| TalkError::Audio("no default audio output device".to_string()))?;
        let config = device
            .default_output_config()
            .map_err(|e| TalkError::Audio(format!("output config: {}", e)))?;

        let device_sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        let state = std::sync::Arc::new(std::sync::Mutex::new(PlaybackState {
            samples: Vec::new(),
            position: 0,
            paused: false,
        }));
        let state_cb = std::sync::Arc::clone(&state);

        let stream = device
            .build_output_stream(
                &cpal::StreamConfig {
                    channels: config.channels(),
                    sample_rate: config.sample_rate(),
                    buffer_size: cpal::BufferSize::Default,
                },
                move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if let Ok(mut guard) = state_cb.try_lock() {
                        if guard.paused {
                            for s in output.iter_mut() {
                                *s = 0.0;
                            }
                            return;
                        }
                        let frames = output.len() / channels;
                        for frame_idx in 0..frames {
                            let sample = if guard.position < guard.samples.len() {
                                let s = guard.samples[guard.position];
                                guard.position += 1;
                                s
                            } else {
                                0.0
                            };
                            for ch in 0..channels {
                                output[frame_idx * channels + ch] = sample;
                            }
                        }
                    } else {
                        for s in output.iter_mut() {
                            *s = 0.0;
                        }
                    }
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
            _stream: stream,
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
