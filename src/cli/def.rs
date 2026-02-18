use crate::core::config::Provider;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "talk-rs", version, about = "Talk CLI")]
pub struct Cli {
    /// Increase logging verbosity (-v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Record audio from the system and save to a file
    Record {
        /// Output file path (optional, defaults to memo-YYYY-MM-DD-HH-MM-SS.ogg)
        #[arg(value_name = "FILE")]
        file: Option<String>,
    },
    /// Transcribe an audio file to text
    Transcribe {
        /// Input audio file path
        #[arg(value_name = "INPUT")]
        input: String,
        /// Output file path (optional, defaults to stdout)
        #[arg(value_name = "OUTPUT")]
        output: Option<String>,
        /// Transcription provider (mistral or openai)
        #[arg(long, value_parser = clap::value_parser!(Provider))]
        provider: Option<Provider>,
        /// Model name (overrides config default for the chosen provider)
        #[arg(long)]
        model: Option<String>,
    },
    /// Record, transcribe, and paste text into the focused application
    Dictate {
        /// Save audio recording to this file path
        #[arg(long, value_name = "PATH")]
        save: Option<String>,
        /// Feed a pre-recorded WAV file instead of live microphone capture
        #[arg(long, value_name = "FILE")]
        input_audio_file: Option<String>,
        /// Transcription provider (mistral or openai)
        #[arg(long, value_parser = clap::value_parser!(Provider))]
        provider: Option<Provider>,
        /// Model name (overrides config default for the chosen provider)
        #[arg(long)]
        model: Option<String>,
        /// Use realtime mode (stream audio via WebSocket, get incremental text)
        #[arg(long)]
        realtime: bool,
        /// Toggle daemon mode: first call starts recording, second call stops
        #[arg(long)]
        toggle: bool,
        /// Disable sound indicators (start/stop/boop)
        #[arg(long)]
        no_sounds: bool,
        /// Disable visual overlay indicator
        #[arg(long)]
        no_overlay: bool,
        /// Show amplitude history visualizer (left of badge)
        #[arg(long)]
        amplitude: bool,
        /// Show spectrum visualizer (right of badge)
        #[arg(long)]
        spectrum: bool,
        /// Run as daemon process (internal, used by --toggle)
        #[arg(long, hide = true)]
        daemon: bool,
        /// Target window ID for paste (internal, used by --toggle)
        #[arg(long, hide = true)]
        target_window: Option<String>,
    },
}
