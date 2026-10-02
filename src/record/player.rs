//! Audio playback for the recordings browser.
//!
//! Thin wrapper over the shared [`crate::audio::AudioPlayer`]: it adds
//! the file-decode dispatch (WAV / OGG / M4A → f32 at the device rate)
//! and forwards every transport control to the shared player.  The cpal
//! output-stream machinery itself lives in `audio::player` so the
//! `speak` command can reuse the same playback path without the `ui`
//! feature.
//!
//! Decoding is available in two shapes: [`WavPlayer::play`] decodes on
//! the calling thread and loads the result (the picker's single
//! recording), and [`WavPlayer::load_in_background`] decodes on a
//! worker thread and hands back a [`PendingLoad`] that the caller polls
//! from its own loop (the recordings browser, whose GTK thread must not
//! block for the seconds a long recording takes to decode).

use super::audio::{read_m4a_as_f32, read_ogg_as_f32, read_wav_as_f32};
use crate::audio::AudioPlayer;
use crate::error::TalkError;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

type DecodeRequest = (
    PathBuf,
    u32,
    std::sync::mpsc::Sender<Result<Vec<f32>, TalkError>>,
);

#[derive(Default)]
struct DecodeState {
    latest: Option<DecodeRequest>,
    shutdown: bool,
}

/// One decoder per player; pending handovers replace, rather than stack up.
struct DecodeQueue {
    shared: Arc<(Mutex<DecodeState>, Condvar)>,
}

impl DecodeQueue {
    fn new() -> Result<Self, TalkError> {
        let shared = Arc::new((Mutex::new(DecodeState::default()), Condvar::new()));
        let worker = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("record-playback-decode".into())
            .spawn(move || Self::run(worker))
            .map_err(|e| TalkError::Audio(format!("start playback decoder: {e}")))?;
        Ok(Self { shared })
    }

    fn request(&self, path: &Path, rate: u32) -> PendingLoad {
        let (tx, rx) = std::sync::mpsc::channel();
        let (state, wake) = &*self.shared;
        if let Ok(mut guard) = state.lock() {
            guard.latest = Some((path.to_path_buf(), rate, tx));
            wake.notify_one();
        }
        PendingLoad::new(rx)
    }

    fn run(shared: Arc<(Mutex<DecodeState>, Condvar)>) {
        loop {
            let (state, wake) = &*shared;
            let job = {
                let Ok(mut guard) = state.lock() else {
                    return;
                };
                while guard.latest.is_none() && !guard.shutdown {
                    guard = match wake.wait(guard) {
                        Ok(guard) => guard,
                        Err(_) => return,
                    };
                }
                if guard.shutdown {
                    return;
                }
                guard.latest.take()
            };
            if let Some((path, rate, tx)) = job {
                let result = std::panic::catch_unwind(|| WavPlayer::decode(&path, rate))
                    .unwrap_or_else(|_| {
                        Err(TalkError::Audio(format!(
                            "playback decoder panicked: {}",
                            path.display()
                        )))
                    });
                let _ = tx.send(result);
            }
        }
    }
}

impl Drop for DecodeQueue {
    fn drop(&mut self) {
        let (state, wake) = &*self.shared;
        if let Ok(mut guard) = state.lock() {
            guard.shutdown = true;
            guard.latest = None;
            wake.notify_one();
        }
    }
}

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
    decoder: DecodeQueue,
}

/// A decode running on a worker thread, started by
/// [`WavPlayer::load_in_background`].
///
/// Poll it with [`PendingLoad::try_take`]; dropping it abandons the
/// result (the worker finishes and its samples are discarded), which is
/// how a superseded Play request is cancelled.
pub(crate) struct PendingLoad {
    rx: std::sync::mpsc::Receiver<Result<Vec<f32>, TalkError>>,
}

/// What a poll of a [`PendingLoad`] found.
pub(crate) enum LoadPoll {
    /// Still decoding.
    Pending,
    /// Decoded; the samples are at the device rate, ready to load.
    Ready(Vec<f32>),
    /// The decode failed (or its worker vanished).
    Failed(TalkError),
}

impl PendingLoad {
    /// Build from a channel.  Production uses
    /// [`WavPlayer::load_in_background`]; tests feed the sender
    /// themselves to script when (and whether) a decode completes.
    pub(crate) fn new(rx: std::sync::mpsc::Receiver<Result<Vec<f32>, TalkError>>) -> Self {
        Self { rx }
    }

    /// Non-blocking check of the decode.
    pub(crate) fn try_take(&self) -> LoadPoll {
        match self.rx.try_recv() {
            Ok(Ok(samples)) => LoadPoll::Ready(samples),
            Ok(Err(e)) => LoadPoll::Failed(e),
            Err(std::sync::mpsc::TryRecvError::Empty) => LoadPoll::Pending,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                LoadPoll::Failed(TalkError::Audio("playback decode worker vanished".into()))
            }
        }
    }
}

impl WavPlayer {
    /// Open the default output device and start a silent stream.
    pub(crate) fn new() -> Result<Self, TalkError> {
        Ok(Self {
            inner: AudioPlayer::new()?,
            decoder: DecodeQueue::new()?,
        })
    }

    /// Decode an audio file (WAV, OGG Opus, or M4A/MP4/AAC) to mono
    /// `f32` at `rate`: the whole work [`play`](Self::play) does before
    /// loading, usable from any thread.
    ///
    /// Dispatches on the file extension (case-insensitive) to the
    /// matching decoder in [`super::audio`].  Unknown extensions fall
    /// through to the WAV parser for backwards compatibility.
    pub(crate) fn decode(audio_path: &Path, rate: u32) -> Result<Vec<f32>, TalkError> {
        let ext = audio_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        crate::perf_counters::incr(crate::perf_counters::Counter::PlaybackDecodes);
        match ext.as_str() {
            "ogg" | "opus" => read_ogg_as_f32(audio_path, rate),
            "m4a" | "mp4" | "aac" => read_m4a_as_f32(audio_path, rate),
            _ => read_wav_as_f32(audio_path, rate),
        }
    }

    /// Load an audio file and start playing it from the beginning,
    /// decoding on the calling thread.
    pub(crate) fn play(&self, audio_path: &Path) -> Result<(), TalkError> {
        let samples = Self::decode(audio_path, self.inner.device_sample_rate())?;
        self.inner.load_f32(samples);
        Ok(())
    }

    /// Start decoding `audio_path` on a worker thread.  Nothing is
    /// loaded yet: poll the returned [`PendingLoad`] and pass its
    /// samples to [`load`](Self::load).
    pub(crate) fn load_in_background(&self, audio_path: &Path) -> PendingLoad {
        self.decoder
            .request(audio_path, self.inner.device_sample_rate())
    }

    /// Load already-decoded samples (device rate) and start playing
    /// them from the beginning.
    pub(crate) fn load_at(&self, samples: Vec<f32>, fraction: f64, paused: bool) {
        self.inner.load_f32_at(samples, fraction, paused);
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
