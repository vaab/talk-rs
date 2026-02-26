//! Audio playback for the recordings browser via cpal.

use super::audio::{read_ogg_as_f32, read_wav_as_f32};
use crate::error::TalkError;

/// Shared state between the GUI thread and the `cpal` output callback.
struct WavPlaybackState {
    samples: Vec<f32>,
    position: usize,
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

    /// Load an audio file (WAV or OGG) and start playing it from the beginning.
    pub(crate) fn play(&self, audio_path: &std::path::Path) -> Result<(), TalkError> {
        let ext = audio_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let samples = match ext {
            "ogg" => read_ogg_as_f32(audio_path, self.device_sample_rate)?,
            _ => read_wav_as_f32(audio_path, self.device_sample_rate)?,
        };
        if let Ok(mut guard) = self.state.lock() {
            guard.samples = samples;
            guard.position = 0;
        }
        Ok(())
    }

    /// Stop playback immediately.
    pub(crate) fn stop(&self) {
        if let Ok(mut guard) = self.state.lock() {
            guard.samples.clear();
            guard.position = 0;
        }
    }

    /// `true` when all samples have been consumed (or nothing loaded).
    pub(crate) fn is_finished(&self) -> bool {
        self.state
            .lock()
            .map(|g| g.samples.is_empty() || g.position >= g.samples.len())
            .unwrap_or(true)
    }
}
