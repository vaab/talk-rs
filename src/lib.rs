//! Voice dictation for Linux: record audio, transcribe it, and paste the
//! result into the focused application. The crate also provides text-to-speech
//! through the `speak` workflow.
//!
//! Public entry points for library consumers include:
//! - [`config`] for loading and validating application configuration.
//! - [`transcription`] for batch and realtime speech-to-text providers.
//! - [`synthesis`] and [`speak`] for text-to-speech providers and workflows.
//! - [`audio`] for capture, encoding, decoding, and playback primitives.
//! - [`recording_cache`] for recording metadata and transcript caching.
//! - [`telemetry`] for observing transcription pipeline events.
//!
//! For headless builds (cloud transcription only, no desktop stack), see
//! the "Cargo feature flags" section of the
//! [README](https://github.com/vaab/talk-rs#cargo-feature-flags) and
//! depend on the crate with `default-features = false`.

pub mod audio;
#[cfg(all(feature = "capture", feature = "ui"))]
pub mod cli;
pub mod clipboard;
pub mod config;
pub mod daemon;
#[cfg(all(feature = "capture", feature = "ui"))]
pub mod dictate;
pub mod error;
#[cfg(feature = "ui")]
pub mod gtk_theme;
#[cfg(any(feature = "parakeet", feature = "kokoro"))]
pub mod model_fetch;
#[cfg(feature = "ui")]
pub mod paste;
pub mod record;
pub mod recording_cache;
pub mod speak;
pub mod synthesis;
pub mod telemetry;
pub mod transcribe;
pub mod transcription;
#[cfg(feature = "ui")]
pub mod widgets;
#[cfg(feature = "ui")]
pub mod x11;

#[cfg(feature = "ui")]
pub use clipboard::X11Clipboard;
pub use clipboard::{Clipboard, MockClipboard};
pub use transcription::{
    MistralOneShotTranscriber, MockOneShotTranscriber, OpenAIOneShotTranscriber,
    OpenAIRealtimeTranscriber,
};
