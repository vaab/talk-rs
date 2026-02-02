use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "talk-rs", version, about = "Talk CLI")]
pub struct Cli {
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
    },
    /// Record, transcribe, and paste text into the focused application
    Dictate {
        /// Output file path to save the audio recording (optional)
        #[arg(value_name = "FILE")]
        file: Option<String>,
        /// Enable chunked mode: split recording into chunks for separate transcription
        #[arg(long)]
        chunked: bool,
        /// Chunk duration in seconds (requires --chunked, overrides config value)
        #[arg(short = 'n', long = "chunk-seconds", requires = "chunked")]
        chunk_seconds: Option<u64>,
    },
}
