pub mod audio;
pub mod clipboard;
pub mod config;
pub mod error;
pub mod transcription;

pub use clipboard::{Clipboard, MockClipboard, X11Clipboard};
pub use transcription::{MistralTranscriber, MockTranscriber, Transcriber};
