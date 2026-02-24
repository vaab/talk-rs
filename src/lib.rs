pub mod audio;
pub mod cli;
pub mod clipboard;
pub mod config;
pub mod daemon;
pub mod dictate;
pub mod error;
pub mod gtk_theme;
pub mod monitor;
pub mod overlay;
pub mod paste;
pub mod record;
pub mod recording_cache;
pub mod transcribe;
pub mod transcription;
pub mod visualizer;
pub mod x11;

pub use clipboard::{Clipboard, MockClipboard, X11Clipboard};
pub use transcription::{
    BatchTranscriber, MistralBatchTranscriber, MockBatchTranscriber, OpenAIBatchTranscriber,
    OpenAIRealtimeTranscriber,
};
