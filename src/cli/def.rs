use crate::config::{Provider, SynthesisProvider};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "talk-rs", version, about = "Talk CLI")]
pub struct Cli {
    /// Increase logging verbosity (-v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Write logs to a file (in addition to stderr)
    #[arg(long, value_name = "PATH", env = "TALK_RS_LOG_FILE")]
    pub log_file: Option<String>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Record audio from the system and save to a file
    Record {
        /// Output file path (defaults to <output_dir>/YYYY/MM/YYYY-MM-DDTHH-MM-SS+ZZZZ.ogg)
        #[arg(value_name = "FILE")]
        file: Option<String>,
        /// Mix system audio (monitor) with microphone input
        #[arg(long)]
        monitor: bool,
        /// Disable sound indicators (start/stop/boop)
        #[arg(long, conflicts_with = "ui")]
        no_sounds: bool,
        /// Disable periodic boop sounds (keep start/stop tones)
        #[arg(long, conflicts_with = "ui")]
        no_boop: bool,
        /// Disable visual overlay indicator
        #[arg(long, conflicts_with = "ui")]
        no_overlay: bool,
        /// Visualizer inside the recording badge (waterfall, amplitude, spectrum)
        #[arg(
            long,
            value_parser = clap::value_parser!(crate::config::VizMode),
            conflicts_with = "ui"
        )]
        viz: Option<crate::config::VizMode>,
        /// Use monochrome colors for the visualizer (theme-aware)
        #[arg(long, conflicts_with = "ui")]
        mono: bool,
        /// Open GTK recordings browser to manage cached recordings
        #[arg(long)]
        ui: bool,
        /// Toggle background recording: first call starts, second call stops
        #[arg(long, conflicts_with = "ui")]
        toggle: bool,
        /// Disable auto-switching of a Bluetooth headset to HFP for
        /// the duration of the recording (overrides config
        /// `audio.bt_auto_switch`)
        #[arg(long)]
        no_bt_auto_switch: bool,
        /// Run as recording daemon (internal, used by --toggle)
        #[arg(long, hide = true)]
        daemon: bool,
    },
    /// Transcribe an audio file to text
    Transcribe {
        /// Input audio file path
        #[arg(value_name = "INPUT")]
        input: String,
        /// Output file path (optional, defaults to stdout)
        #[arg(value_name = "OUTPUT")]
        output: Option<String>,
        /// Transcription provider (mistral, openai, or parakeet)
        #[arg(long, value_parser = clap::value_parser!(Provider))]
        provider: Option<Provider>,
        /// Model name (overrides config default for the chosen provider)
        #[arg(long)]
        model: Option<String>,
        /// Enable speaker diarization (identify who is speaking)
        #[arg(long)]
        diarize: bool,
        /// Include timestamps in the output (HH:MM:SS prefix)
        #[arg(long)]
        timestamp: bool,
    },
    /// Synthesize speech from text and play it (or save to a WAV file)
    Speak {
        /// Text to speak.  When omitted, text is read from `--file` or,
        /// failing that, from stdin (when stdin is not a TTY).
        #[arg(value_name = "TEXT")]
        text: Option<String>,
        /// Read the text to speak from this file instead of the
        /// positional argument
        #[arg(short = 'f', long, value_name = "PATH")]
        file: Option<String>,
        /// Synthesis provider (kokoro or mistral)
        #[arg(long, value_parser = clap::value_parser!(SynthesisProvider))]
        provider: Option<SynthesisProvider>,
        /// Voice: a Kokoro voice name (e.g. af_heart, ff_siwis) or a
        /// Mistral preset voice id
        #[arg(long, value_name = "NAME_OR_ID")]
        voice: Option<String>,
        /// Phonemization language for Kokoro (e.g. en, fr).  Ignored by
        /// the Mistral provider (voice implies language)
        #[arg(long, value_name = "LANG")]
        lang: Option<String>,
        /// Speech rate multiplier (Kokoro only; 1.0 = normal)
        #[arg(long, value_name = "FACTOR")]
        speed: Option<f32>,
        /// Save synthesized audio to this WAV file instead of playing
        /// it through the speakers
        #[arg(short = 'o', long, value_name = "PATH")]
        output: Option<String>,
        /// Bypass the voice/language mismatch guard: synthesize even
        /// when an explicitly-chosen voice's language differs from the
        /// resolved (auto-detected / --lang / config) language
        #[arg(long)]
        force: bool,
    },
    /// Record, transcribe, and paste text into the focused application
    Dictate {
        /// Save audio recording to this file path
        #[arg(long, value_name = "PATH")]
        save: Option<String>,
        /// Write recording metadata YAML to this file path
        #[arg(long, value_name = "FILE")]
        output_yaml: Option<String>,
        /// Feed a pre-recorded audio file instead of live microphone capture
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
        /// Transcription provider (mistral, openai, or parakeet)
        #[arg(long, value_parser = clap::value_parser!(Provider))]
        provider: Option<Provider>,
        /// Model name (overrides config default for the chosen provider)
        #[arg(long)]
        model: Option<String>,
        /// Enable speaker diarization (identify who is speaking)
        #[arg(long)]
        diarize: bool,
        /// Include timestamps in the output (HH:MM:SS prefix)
        #[arg(long)]
        timestamp: bool,
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
        /// Skip pasting transcription into the focused application
        #[arg(long)]
        no_paste: bool,
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
        /// Audio format for one-shot uploads (wav or ogg)
        #[arg(long, value_enum, default_value_t)]
        upload_format: crate::transcription::UploadFormat,
        /// Disable auto-switching of a Bluetooth headset to HFP for
        /// the duration of the recording (overrides config
        /// `audio.bt_auto_switch`)
        #[arg(long)]
        no_bt_auto_switch: bool,
        /// Run as daemon process (internal, used by --toggle)
        #[arg(long, hide = true)]
        daemon: bool,
        /// Target window ID for paste (internal, used by --toggle)
        #[arg(long, hide = true)]
        target_window: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_toggle_parses_public_options_and_hidden_child_mode() {
        let cli = Cli::try_parse_from([
            "talk-rs",
            "-vv",
            "record",
            "--toggle",
            "--monitor",
            "--no-sounds",
            "--no-boop",
            "--no-overlay",
            "--viz",
            "waterfall",
            "--mono",
            "--no-bt-auto-switch",
            "meeting.ogg",
        ])
        .expect("record toggle arguments should parse");

        assert_eq!(cli.verbose, 2);
        match cli.command {
            Commands::Record {
                file,
                monitor,
                ui,
                no_sounds,
                no_boop,
                no_overlay,
                viz,
                mono,
                no_bt_auto_switch,
                toggle,
                daemon,
            } => {
                assert_eq!(file.as_deref(), Some("meeting.ogg"));
                assert!(monitor);
                assert!(!ui);
                assert!(no_sounds);
                assert!(no_boop);
                assert!(no_overlay);
                assert_eq!(viz, Some(crate::config::VizMode::Waterfall));
                assert!(mono);
                assert!(no_bt_auto_switch);
                assert!(toggle);
                assert!(!daemon);
            }
            other => panic!("expected record command, got {other:?}"),
        }

        let child = Cli::try_parse_from(["talk-rs", "record", "--daemon"])
            .expect("hidden record child mode should parse");
        match child.command {
            Commands::Record { daemon, .. } => assert!(daemon),
            other => panic!("expected record command, got {other:?}"),
        }
    }

    #[test]
    fn record_toggle_conflicts_with_ui() {
        let error = Cli::try_parse_from(["talk-rs", "record", "--toggle", "--ui"])
            .expect_err("record toggle and UI must conflict");

        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn record_feedback_flags_conflict_with_ui() {
        for flag in [
            "--no-sounds",
            "--no-boop",
            "--no-overlay",
            "--viz",
            "--mono",
        ] {
            let mut args = vec!["talk-rs", "record", "--ui", flag];
            if flag == "--viz" {
                args.push("waterfall");
            }
            let error = Cli::try_parse_from(args)
                .expect_err("recording feedback flags must conflict with the recordings UI");
            assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
        }
    }
}
