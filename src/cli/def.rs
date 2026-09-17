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
        /// Language for Kokoro phonemization; for Mistral, selects/validates the
        /// preset voice (auto-detected from the text when omitted)
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

    #[test]
    fn global_options_apply_before_the_subcommand() {
        let cli = Cli::try_parse_from([
            "talk-rs",
            "-vvv",
            "--log-file",
            "/tmp/talk.log",
            "transcribe",
            "in.ogg",
        ])
        .expect("global options should parse");

        assert_eq!(cli.verbose, 3);
        assert_eq!(cli.log_file.as_deref(), Some("/tmp/talk.log"));
        assert!(matches!(cli.command, Commands::Transcribe { .. }));
    }

    #[test]
    fn transcribe_parses_every_documented_option() {
        let cli = Cli::try_parse_from([
            "talk-rs",
            "transcribe",
            "--provider",
            "parakeet",
            "--model",
            "parakeet-tdt",
            "--diarize",
            "--timestamp",
            "in.m4a",
            "out.txt",
        ])
        .expect("transcribe arguments should parse");

        match cli.command {
            Commands::Transcribe {
                input,
                output,
                provider,
                model,
                diarize,
                timestamp,
            } => {
                assert_eq!(input, "in.m4a");
                assert_eq!(output.as_deref(), Some("out.txt"));
                assert_eq!(provider, Some(Provider::Parakeet));
                assert_eq!(model.as_deref(), Some("parakeet-tdt"));
                assert!(diarize);
                assert!(timestamp);
            }
            other => panic!("expected transcribe command, got {other:?}"),
        }
    }

    #[test]
    fn transcribe_requires_an_input_file() {
        let error = Cli::try_parse_from(["talk-rs", "transcribe"])
            .expect_err("transcribe without input must fail");
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn provider_rejects_unknown_names() {
        let error =
            Cli::try_parse_from(["talk-rs", "transcribe", "--provider", "whisperx", "a.ogg"])
                .expect_err("unknown provider must be rejected at parse time");
        assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn speak_parses_every_documented_option() {
        let cli = Cli::try_parse_from([
            "talk-rs",
            "speak",
            "--provider",
            "mistral",
            "--voice",
            "alice",
            "--lang",
            "fr",
            "--speed",
            "1.25",
            "-f",
            "msg.txt",
            "-o",
            "out.wav",
            "--force",
            "Bonjour",
        ])
        .expect("speak arguments should parse");

        match cli.command {
            Commands::Speak {
                text,
                file,
                provider,
                voice,
                lang,
                speed,
                output,
                force,
            } => {
                assert_eq!(text.as_deref(), Some("Bonjour"));
                assert_eq!(file.as_deref(), Some("msg.txt"));
                assert_eq!(provider, Some(SynthesisProvider::Mistral));
                assert_eq!(voice.as_deref(), Some("alice"));
                assert_eq!(lang.as_deref(), Some("fr"));
                assert_eq!(speed, Some(1.25));
                assert_eq!(output.as_deref(), Some("out.wav"));
                assert!(force);
            }
            other => panic!("expected speak command, got {other:?}"),
        }
    }

    #[test]
    fn speak_text_is_optional_so_stdin_can_be_used() {
        let cli = Cli::try_parse_from(["talk-rs", "speak"]).expect("bare speak should parse");
        match cli.command {
            Commands::Speak { text, file, .. } => {
                assert!(text.is_none());
                assert!(file.is_none());
            }
            other => panic!("expected speak command, got {other:?}"),
        }
    }

    #[test]
    fn dictate_parses_every_documented_option() {
        let cli = Cli::try_parse_from([
            "talk-rs",
            "dictate",
            "--save",
            "rec.ogg",
            "--output-yaml",
            "meta.yml",
            "--input-audio-file",
            "in.wav",
            "--retry-last",
            "--pick",
            "--replace-last-paste",
            "--provider",
            "openai",
            "--model",
            "gpt-transcribe",
            "--diarize",
            "--timestamp",
            "--realtime",
            "--toggle",
            "--no-sounds",
            "--no-boop",
            "--no-chunk-paste",
            "--no-paste",
            "--monitor",
            "--no-overlay",
            "--no-auto-pause",
            "--viz",
            "spectrum",
            "--mono",
            "--upload-format",
            "ogg",
            "--no-bt-auto-switch",
        ])
        .expect("dictate arguments should parse");

        match cli.command {
            Commands::Dictate {
                save,
                output_yaml,
                input_audio_file,
                retry_last,
                pick,
                replace_last_paste,
                provider,
                model,
                diarize,
                timestamp,
                realtime,
                toggle,
                no_sounds,
                no_boop,
                no_chunk_paste,
                no_paste,
                monitor,
                no_overlay,
                no_auto_pause,
                viz,
                mono,
                upload_format,
                no_bt_auto_switch,
                daemon,
                target_window,
            } => {
                assert_eq!(save.as_deref(), Some("rec.ogg"));
                assert_eq!(output_yaml.as_deref(), Some("meta.yml"));
                assert_eq!(input_audio_file.as_deref(), Some("in.wav"));
                assert!(retry_last);
                assert!(pick);
                assert!(replace_last_paste);
                assert_eq!(provider, Some(Provider::OpenAI));
                assert_eq!(model.as_deref(), Some("gpt-transcribe"));
                assert!(diarize);
                assert!(timestamp);
                assert!(realtime);
                assert!(toggle);
                assert!(no_sounds);
                assert!(no_boop);
                assert!(no_chunk_paste);
                assert!(no_paste);
                assert!(monitor);
                assert!(no_overlay);
                assert!(no_auto_pause);
                assert_eq!(viz, Some(crate::config::VizMode::Spectrum));
                assert!(mono);
                assert_eq!(upload_format, crate::transcription::UploadFormat::Ogg);
                assert!(no_bt_auto_switch);
                assert!(!daemon);
                assert!(target_window.is_none());
            }
            other => panic!("expected dictate command, got {other:?}"),
        }
    }

    #[test]
    fn dictate_defaults_to_wav_uploads_and_no_flags() {
        let cli = Cli::try_parse_from(["talk-rs", "dictate"]).expect("bare dictate should parse");
        match cli.command {
            Commands::Dictate {
                upload_format,
                realtime,
                toggle,
                provider,
                ..
            } => {
                assert_eq!(upload_format, crate::transcription::UploadFormat::Wav);
                assert!(!realtime);
                assert!(!toggle);
                assert!(provider.is_none());
            }
            other => panic!("expected dictate command, got {other:?}"),
        }
    }
}
