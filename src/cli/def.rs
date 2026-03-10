use crate::config::Provider;
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
        /// Output file path (defaults to <output_dir>/memo-YYYY-MM-DD-HH-MM-SS.ogg)
        #[arg(value_name = "FILE")]
        file: Option<String>,
        /// Mix system audio (monitor) with microphone input
        #[arg(long)]
        monitor: bool,
        /// Open GTK recordings browser to manage cached recordings
        #[arg(long)]
        ui: bool,
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
        /// Enable speaker diarization (identify who is speaking)
        #[arg(long)]
        diarize: bool,
    },
    /// Record, transcribe, and paste text into the focused application
    Dictate {
        /// Save audio recording to this file path
        #[arg(long, value_name = "PATH")]
        save: Option<String>,
        /// Write recording metadata YAML to this file path
        #[arg(long, value_name = "FILE")]
        output_yaml: Option<String>,
        /// Feed a pre-recorded WAV file instead of live microphone capture
        #[arg(long, value_name = "FILE")]
        input_audio_file: Option<String>,
        /// Reuse the last cached recording as input audio
        #[arg(long)]
        retry_last: bool,
        /// Offer multiple candidate transcriptions in a GTK picker window
        #[arg(long)]
        pick: bool,
        /// Delete previous pasted text length before inserting selected text
        #[arg(long)]
        replace_last_paste: bool,
        /// Transcription provider (mistral or openai)
        #[arg(long, value_parser = clap::value_parser!(Provider))]
        provider: Option<Provider>,
        /// Model name (overrides config default for the chosen provider)
        #[arg(long)]
        model: Option<String>,
        /// Enable speaker diarization (identify who is speaking)
        #[arg(long)]
        diarize: bool,
        /// Use realtime mode (stream audio via WebSocket, get incremental text)
        #[arg(long)]
        realtime: bool,
        /// Toggle daemon mode: first call starts recording, second call stops
        #[arg(long)]
        toggle: bool,
        /// Disable sound indicators (start/stop/boop)
        #[arg(long)]
        no_sounds: bool,
        /// Disable periodic boop sounds during recording
        #[arg(long)]
        no_boop: bool,
        /// Disable chunked pasting (paste all text in one shot)
        #[arg(long)]
        no_chunk_paste: bool,
        /// Mix system audio (monitor) with microphone input
        #[arg(long)]
        monitor: bool,
        /// Disable visual overlay indicator
        #[arg(long)]
        no_overlay: bool,
        /// Disable auto-pause during silence (forward all audio to transcription)
        #[arg(long)]
        no_auto_pause: bool,
        /// Visualizer inside the recording badge (waterfall, amplitude, spectrum)
        #[arg(long, value_parser = clap::value_parser!(crate::config::VizMode))]
        viz: Option<crate::config::VizMode>,
        /// Use monochrome colors for the visualizer (theme-aware)
        #[arg(long)]
        mono: bool,
        /// Run as daemon process (internal, used by --toggle)
        #[arg(long, hide = true)]
        daemon: bool,
        /// Target window ID for paste (internal, used by --toggle)
        #[arg(long, hide = true)]
        target_window: Option<String>,
    },
}
