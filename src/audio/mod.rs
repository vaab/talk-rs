//! Audio capture interfaces and implementations.

use crate::error::TalkError;
use tokio::sync::mpsc;

pub mod cpal_capture;
pub mod encoder;
pub mod file_source;
pub mod indicator;
pub mod mock;
pub mod monitor_capture;
pub mod pipewire_capture;
pub mod resample;
pub mod writer;

pub use encoder::{AudioEncoder, MockEncoder, OpusEncoder};
pub use writer::{AudioWriter, OggOpusWriter, WavWriter};

pub(crate) const CHUNK_DURATION_MS: u64 = 20;
pub(crate) const CHANNEL_CAPACITY: usize = 25;

/// Trait for audio capture from system devices.
pub trait AudioCapture: Send {
    /// Start audio capture and return a receiver for PCM samples.
    ///
    /// The receiver yields chunks of i16 PCM samples, typically 20ms each.
    fn start(&mut self) -> Result<mpsc::Receiver<Vec<i16>>, TalkError>;

    /// Stop audio capture and clean up resources.
    fn stop(&mut self) -> Result<(), TalkError>;
}
