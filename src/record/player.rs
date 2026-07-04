//! Audio playback for the recordings browser.
//!
//! Thin wrapper over the shared [`crate::audio::AudioPlayer`]: it adds
//! the file-decode dispatch (WAV / OGG / M4A → f32 at the device rate)
//! and forwards every transport control to the shared player.  The cpal
//! output-stream machinery itself lives in `audio::player` so the
//! `speak` command can reuse the same playback path without the `ui`
//! feature.

use super::audio::{read_m4a_as_f32, read_ogg_as_f32, read_wav_as_f32};
use crate::audio::AudioPlayer;
use crate::error::TalkError;

/// Plays audio files (WAV, OGG Opus, or M4A/MP4/AAC) through the
/// default output device.
///
/// Created once when the recordings window opens.  Delegates the
/// continuous cpal output stream and all transport controls to a
/// shared [`AudioPlayer`]; this type only owns the file-format decode
/// dispatch.
// Named WavPlayer for backwards compatibility; handles WAV/OGG/M4A.
pub(crate) struct WavPlayer {
    inner: AudioPlayer,
}

impl WavPlayer {
    /// Open the default output device and start a silent stream.
    pub(crate) fn new() -> Result<Self, TalkError> {
        Ok(Self {
            inner: AudioPlayer::new()?,
        })
    }

    /// Load an audio file (WAV, OGG Opus, or M4A/MP4/AAC) and start
    /// playing it from the beginning.
    ///
    /// Dispatches on the file extension (case-insensitive) to the
    /// matching decoder in [`super::audio`].  Unknown extensions fall
    /// through to the WAV parser for backwards compatibility.
    pub(crate) fn play(&self, audio_path: &std::path::Path) -> Result<(), TalkError> {
        let ext = audio_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let rate = self.inner.device_sample_rate();
        let samples = match ext.as_str() {
            "ogg" | "opus" => read_ogg_as_f32(audio_path, rate)?,
            "m4a" | "mp4" | "aac" => read_m4a_as_f32(audio_path, rate)?,
            _ => read_wav_as_f32(audio_path, rate)?,
        };
        self.inner.load_f32(samples);
        Ok(())
    }

    /// Stop playback immediately.
    pub(crate) fn stop(&self) {
        self.inner.stop();
    }

    /// `true` when all samples have been consumed (or nothing loaded).
    pub(crate) fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Playback progress as a fraction (0.0–1.0).
    pub(crate) fn progress(&self) -> f64 {
        self.inner.progress()
    }

    /// Seek to a position expressed as a fraction (0.0–1.0).
    pub(crate) fn seek(&self, fraction: f64) {
        self.inner.seek(fraction);
    }

    /// Pause playback (position is preserved).
    pub(crate) fn pause(&self) {
        self.inner.pause();
    }

    /// Resume playback from the current position.
    pub(crate) fn resume(&self) {
        self.inner.resume();
    }

    /// `true` when playback is paused.
    pub(crate) fn is_paused(&self) -> bool {
        self.inner.is_paused()
    }

    /// `true` when audio is loaded (samples are present).
    pub(crate) fn has_audio(&self) -> bool {
        self.inner.has_audio()
    }

    /// Total duration of loaded audio in seconds.
    pub(crate) fn duration_secs(&self) -> f64 {
        self.inner.duration_secs()
    }
}
