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
#[cfg(feature = "ui")]
pub mod paste;
pub mod record;
pub mod recording_cache;
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
