//! Transcribe command implementation.
//!
//! Transcribes an audio file to text using the configured transcription backend.
//! Supports writing output to stdout or to a file.

use crate::config::{Config, Provider};
use crate::error::TalkError;
use crate::transcription;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

/// Parse command-line arguments for the transcribe command.
///
/// # Arguments
/// * `args` - Command-line arguments: [input_file, optional_output_file]
///
/// # Returns
/// A tuple of (input_path, optional_output_path)
pub fn parse_args(args: &[String]) -> Result<(PathBuf, Option<PathBuf>), TalkError> {
    match args.len() {
        1 => {
            // Input file only, output to stdout
            Ok((PathBuf::from(&args[0]), None))
        }
        2 => {
            // Input file and output file
            Ok((PathBuf::from(&args[0]), Some(PathBuf::from(&args[1]))))
        }
        _ => Err(TalkError::Audio(
            "transcribe command requires 1 or 2 arguments (input_file [output_file])".to_string(),
        )),
    }
}

/// Transcribe an audio file to text.
///
/// Follows the waterfall architecture from `doc/plan/plan.md`:
///
/// - **Specific options on CLI** (`--provider`, `--model`, or
///   `--diarize`): call Layer 3 (`transcribe_audio`) directly.  No
///   pick I/O -- the user asked for a specific transcription, not
///   the authoritative one.
/// - **No specific options**: call Layer 2 (`produce_transcript`),
///   which checks the pick file first, transcribes via the default
///   model on miss, and writes the pick file.  On
///   `TranscriptInProgress`, polls `get_transcript()` every 1 s for
///   up to 5 s.
pub async fn transcribe(
    args: Vec<String>,
    cli_provider: Option<Provider>,
    cli_model: Option<String>,
    diarize: bool,
    timestamp: bool,
) -> Result<(), TalkError> {
    let (input_path, output_path) = parse_args(&args)?;
    if !input_path.exists() {
        return Err(TalkError::Audio(format!(
            "Input file not found: {}",
            input_path.display()
        )));
    }
    let config = Config::load(None)?;
    let provider = cli_provider
        .or_else(|| config.transcription.as_ref().map(|t| t.default_provider))
        .unwrap_or(Provider::Mistral);

    // Parakeet is a local backend whose model must be downloaded once.
    // The transcribe pipeline never downloads silently, so obtain
    // consent here (TTY prompt, or stderr-log + proceed when piped)
    // before transcription reaches `validate`.  No-op once installed.
    #[cfg(feature = "parakeet")]
    if provider == Provider::Parakeet {
        crate::transcription::parakeet::consent::ensure_with_cli_consent(&config).await?;
    }

    let specific_options = cli_provider.is_some() || cli_model.is_some() || diarize;
    let sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink> =
        std::sync::Arc::new(crate::telemetry::NoOpSink);

    let output_text = if specific_options {
        // Specific options -> Layer 3 directly, no pick I/O.
        // CLI is an autonomous caller (no human watching a GTK
        // window), so use `Proportional` so a hung server cannot
        // wedge the CLI invocation indefinitely.
        let result = transcription::transcribe_audio(
            &input_path,
            &config,
            provider,
            cli_model.as_deref(),
            diarize,
            transcription::TranscribeOptions {
                allow_api: true,
                policy: transcription::RequestTimeoutPolicy::Proportional,
                cancel_token: None,
                skip_legacy_lock: false,
            },
            &sink,
        )
        .await?;
        transcription::format_transcription_output(&result, timestamp)
    } else {
        // Default options -> Layer 2 (uses pick file as cache).
        produce_or_wait(&input_path, &config, provider, &sink).await?
    };

    match output_path {
        Some(output_file) => {
            let mut file = tokio::fs::File::create(&output_file)
                .await
                .map_err(TalkError::Io)?;
            file.write_all(output_text.as_bytes())
                .await
                .map_err(TalkError::Io)?;
            file.sync_all().await.map_err(TalkError::Io)?;
            println!("Transcription saved to: {}", output_file.display());
        }
        None => println!("{}", output_text),
    }

    Ok(())
}

/// Call [`transcription::produce_transcript`] and, on
/// [`TalkError::TranscriptInProgress`], poll
/// [`recording_cache::get_transcript`] every 1 s for up to 5 s.
async fn produce_or_wait(
    input_path: &std::path::Path,
    config: &Config,
    provider: Provider,
    sink: &std::sync::Arc<dyn crate::telemetry::TelemetrySink>,
) -> Result<String, TalkError> {
    use crate::recording_cache::{self, TranscriptStatus};

    match transcription::produce_transcript(input_path, config, provider, None, sink).await {
        Ok(text) => return Ok(text),
        Err(TalkError::TranscriptInProgress) => {}
        Err(e) => return Err(e),
    }

    // Poll up to 5 times with 1 s interval.
    for _ in 0..5 {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        match recording_cache::get_transcript(input_path) {
            TranscriptStatus::Available(text) => return Ok(text),
            TranscriptStatus::NotAvailable => {
                // Lock was released without producing a pick — retry.
                return Box::pin(produce_or_wait(input_path, config, provider, sink)).await;
            }
            TranscriptStatus::InProgress => continue,
        }
    }
    Err(TalkError::TranscriptInProgress)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn default_transcription_uses_authoritative_pick_without_provider_request() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let audio = dir.path().join("memo.ogg");
        std::fs::write(&audio, b"recording").expect("test audio");
        crate::recording_cache::write_pick(&audio, "openai", "whisper-1", false, "edited text")
            .expect("saved pick");
        let config_file = dir.path().join("config.yaml");
        std::fs::write(
            &config_file,
            format!("output_dir: {}\nproviders: {{}}\n", dir.path().display()),
        )
        .expect("test config");
        let config = Config::load(Some(&config_file)).expect("load isolated config");
        let sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink> =
            std::sync::Arc::new(crate::telemetry::NoOpSink);

        let output = produce_or_wait(&audio, &config, Provider::Mistral, &sink)
            .await
            .expect("pick bypasses absent provider configuration");
        assert_eq!(output, "edited text");
    }

    #[tokio::test]
    async fn transcribe_rejects_missing_input_before_loading_user_config() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let missing = dir.path().join("missing.ogg");
        let error = transcribe(
            vec![missing.display().to_string()],
            None,
            None,
            false,
            false,
        )
        .await
        .expect_err("missing recording must be rejected");
        assert_eq!(
            error.to_string(),
            format!("Audio error: Input file not found: {}", missing.display())
        );
    }

    #[test]
    fn test_parse_args_input_only() {
        let args = vec!["audio.ogg".to_string()];
        let result = parse_args(&args).expect("parse should succeed");

        assert_eq!(result.0, PathBuf::from("audio.ogg"));
        assert_eq!(result.1, None);
    }

    #[test]
    fn test_parse_args_input_and_output() {
        let args = vec!["audio.ogg".to_string(), "transcript.txt".to_string()];
        let result = parse_args(&args).expect("parse should succeed");

        assert_eq!(result.0, PathBuf::from("audio.ogg"));
        assert_eq!(result.1, Some(PathBuf::from("transcript.txt")));
    }

    #[test]
    fn test_parse_args_with_paths() {
        let args = vec![
            "/tmp/audio.ogg".to_string(),
            "/tmp/transcript.txt".to_string(),
        ];
        let result = parse_args(&args).expect("parse should succeed");

        assert_eq!(result.0, PathBuf::from("/tmp/audio.ogg"));
        assert_eq!(result.1, Some(PathBuf::from("/tmp/transcript.txt")));
    }

    #[test]
    fn test_parse_args_no_args() {
        let args = vec![];
        let result = parse_args(&args);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires 1 or 2 arguments"));
    }

    #[test]
    fn test_parse_args_too_many_args() {
        let args = vec![
            "audio.ogg".to_string(),
            "transcript.txt".to_string(),
            "extra.txt".to_string(),
        ];
        let result = parse_args(&args);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires 1 or 2 arguments"));
    }
}
