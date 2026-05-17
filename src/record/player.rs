//! Audio playback for the recordings browser via cpal.

use super::audio::{read_m4a_as_f32, read_ogg_as_f32, read_wav_as_f32};
use crate::error::TalkError;

/// Shared state between the GUI thread and the `cpal` output callback.
struct WavPlaybackState {
    samples: Vec<f32>,
    position: usize,
    paused: bool,
}

/// Plays audio files (WAV or OGG) through `cpal`'s default output device.
///
/// Created once when the recordings window opens. The `cpal` output
/// stream runs continuously (outputting silence when idle). Calling
/// [`play`](WavPlayer::play) loads an audio file and starts from the
/// beginning; [`stop`](WavPlayer::stop) clears the buffer.
// Named WavPlayer for backwards compatibility; handles both WAV and OGG.
pub(crate) struct WavPlayer {
    state: std::sync::Arc<std::sync::Mutex<WavPlaybackState>>,
    device_sample_rate: u32,
    // Dropping this stops the stream.
    _stream: cpal::Stream,
}

impl WavPlayer {
    /// Open the default output device and start a silent stream.
    pub(crate) fn new() -> Result<Self, TalkError> {
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

        let state = std::sync::Arc::new(std::sync::Mutex::new(WavPlaybackState {
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

    /// Load an audio file (WAV, OGG Opus, or M4A/MP4/AAC) and start
    /// playing it from the beginning.
    ///
    /// Dispatches on the file extension (case-insensitive) to the
    /// matching decoder in [`super::audio`].  Unknown extensions fall
    /// through to the WAV parser for backwards compatibility — this
    /// errors loudly on non-PCM data rather than silently producing
    /// garbage, which matches the long-standing behaviour for
    /// unsupported formats.
    pub(crate) fn play(&self, audio_path: &std::path::Path) -> Result<(), TalkError> {
        let ext = audio_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let samples = match ext.as_str() {
            "ogg" | "opus" => read_ogg_as_f32(audio_path, self.device_sample_rate)?,
            "m4a" | "mp4" | "aac" => read_m4a_as_f32(audio_path, self.device_sample_rate)?,
            _ => read_wav_as_f32(audio_path, self.device_sample_rate)?,
        };
        if let Ok(mut guard) = self.state.lock() {
            guard.samples = samples;
            guard.position = 0;
            guard.paused = false;
        }
        Ok(())
    }

    /// Stop playback immediately.
    pub(crate) fn stop(&self) {
        if let Ok(mut guard) = self.state.lock() {
            guard.samples.clear();
            guard.position = 0;
            guard.paused = false;
        }
    }

    /// `true` when all samples have been consumed (or nothing loaded).
    pub(crate) fn is_finished(&self) -> bool {
        self.state
            .lock()
            .map(|g| g.samples.is_empty() || g.position >= g.samples.len())
            .unwrap_or(true)
    }

    /// Playback progress as a fraction (0.0–1.0).
    ///
    /// Returns 0.0 when nothing is loaded or playback hasn't started.
    pub(crate) fn progress(&self) -> f64 {
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
    pub(crate) fn seek(&self, fraction: f64) {
        let fraction = fraction.clamp(0.0, 1.0);
        if let Ok(mut guard) = self.state.lock() {
            if !guard.samples.is_empty() {
                guard.position = (fraction * guard.samples.len() as f64) as usize;
            }
        }
    }

    /// Pause playback (position is preserved).
    pub(crate) fn pause(&self) {
        if let Ok(mut guard) = self.state.lock() {
            guard.paused = true;
        }
    }

    /// Resume playback from the current position.
    pub(crate) fn resume(&self) {
        if let Ok(mut guard) = self.state.lock() {
            guard.paused = false;
        }
    }

    /// `true` when playback is paused.
    pub(crate) fn is_paused(&self) -> bool {
        self.state.lock().map(|g| g.paused).unwrap_or(false)
    }

    /// `true` when audio is loaded (samples are present).
    pub(crate) fn has_audio(&self) -> bool {
        self.state
            .lock()
            .map(|g| !g.samples.is_empty())
            .unwrap_or(false)
    }

    /// Total duration of loaded audio in seconds.
    ///
    /// Returns 0.0 when nothing is loaded.
    pub(crate) fn duration_secs(&self) -> f64 {
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
