//! Audio capture interfaces and implementations.

use crate::error::TalkError;
use tokio::sync::mpsc;

#[cfg(feature = "capture")]
pub mod bt_link;
#[cfg(feature = "capture")]
pub mod bt_profile;
#[cfg(feature = "capture")]
pub mod cpal_capture;
pub mod encoder;
pub mod file_source;
#[cfg(feature = "capture")]
pub mod indicator;
pub mod mock;
#[cfg(feature = "capture")]
pub mod monitor_capture;
#[cfg(test)]
pub(crate) mod perf_audio;
#[cfg(feature = "capture")]
pub mod pipewire_capture;
#[cfg(feature = "playback")]
pub mod player;
#[cfg(feature = "capture")]
pub mod recording_feedback;
pub mod resample;
pub mod ring_buffer;
pub mod tee;
pub mod writer;

pub use encoder::{AudioEncoder, MockEncoder, OpusEncoder};
#[cfg(feature = "playback")]
pub use player::AudioPlayer;
pub use writer::{AudioWriter, OggOpusWriter, WavWriter};

pub(crate) const CHUNK_DURATION_MS: u64 = 20;
pub(crate) const CHANNEL_CAPACITY: usize = 25;

/// Error limit for live capture to prove readiness after `start()`.
///
/// This is NOT a readiness assumption: readiness is only ever the proof
/// carried by [`CaptureReady`].  When the proof does not arrive within
/// this bound, recording fails loudly instead of showing the cue.
pub const CAPTURE_READY_LIMIT: std::time::Duration = std::time::Duration::from_secs(5);

/// Proof that a live capture is delivering audio from its device.
///
/// Produced only once the capture stream, explicitly targeted at
/// `source`, reached the streaming state and delivered a buffer, with —
/// for a Bluetooth HFP microphone — the voice link to the headset up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureReady {
    /// Device (PipeWire node / PulseAudio source) name the stream is linked to.
    pub source: String,
    /// Instant at which readiness was proven (a buffer was delivered).
    pub first_buffer_at: std::time::Instant,
}

/// Channel on which a live capture publishes its readiness proof (or the
/// reason it could not be obtained).
pub type CaptureReadyReceiver = tokio::sync::oneshot::Receiver<Result<CaptureReady, TalkError>>;

/// How a capture source establishes that recording has really begun.
pub enum CaptureReadiness {
    /// Not a live device (file replay, mock): there is no device start-up
    /// to prove; every sample is produced by the source itself.
    NotApplicable,
    /// Live device: wait for the proof on this channel.
    Pending(CaptureReadyReceiver),
}

impl CaptureReadiness {
    /// Wait for the readiness proof, bounded by `limit` as an error limit.
    ///
    /// Returns `Ok(None)` for non-live sources, `Ok(Some(proof))` once a
    /// live capture proved it is delivering audio, and an error naming
    /// the missing proof otherwise.
    pub async fn wait(self, limit: std::time::Duration) -> Result<Option<CaptureReady>, TalkError> {
        let rx = match self {
            Self::NotApplicable => return Ok(None),
            Self::Pending(rx) => rx,
        };
        match tokio::time::timeout(limit, rx).await {
            Ok(Ok(Ok(ready))) => Ok(Some(ready)),
            Ok(Ok(Err(error))) => Err(error),
            Ok(Err(_)) => Err(TalkError::Audio(
                "capture ended before proving readiness (no buffer delivered)".into(),
            )),
            Err(_) => Err(TalkError::Audio(format!(
                "capture did not prove readiness within {:?}: the stream never \
                 reached the streaming state with a delivered buffer",
                limit
            ))),
        }
    }
}

/// Trait for audio capture from system devices.
pub trait AudioCapture: Send {
    /// Start audio capture and return a receiver for PCM samples.
    ///
    /// The receiver yields chunks of i16 PCM samples, typically 20ms each.
    fn start(&mut self) -> Result<mpsc::Receiver<Vec<i16>>, TalkError>;

    /// Take how this capture proves it is really recording.
    ///
    /// Called once, after [`start`](Self::start).  Live device captures
    /// MUST return [`CaptureReadiness::Pending`]; only sources that
    /// produce samples themselves (files, mocks) may return
    /// [`CaptureReadiness::NotApplicable`].
    fn readiness(&mut self) -> CaptureReadiness;

    /// Stop audio capture and clean up resources.
    fn stop(&mut self) -> Result<(), TalkError>;
}
