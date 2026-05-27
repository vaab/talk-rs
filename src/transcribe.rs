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

    #[tokio::test]
    async fn test_transcribe_pipeline_with_mock_transcriber() {
        use crate::config::Provider;
        use crate::recording_cache;
        use crate::transcription::{BatchTranscriber, MockBatchTranscriber, TranscriptionBody};
        use std::fs;
        use tempfile::TempDir;

        // Create temporary directory for test files
        let temp_dir = TempDir::new().expect("create temp dir");

        // Create a temporary input audio file
        let input_path = temp_dir.path().join("test-audio.ogg");
        fs::write(&input_path, b"fake audio data").expect("write input file");

        // Create output path
        let output_path = temp_dir.path().join("transcript.txt");

        // Create mock transcriber
        let mock = MockBatchTranscriber::new("This is a test transcription");

        // Transcribe using mock
        let transcription = mock
            .fetch_transcription(TranscriptionBody::File(input_path.clone()))
            .await
            .expect("transcribe should succeed");

        let output_text = transcription::format_transcription_output(&transcription, false);
        let sidecar_path = recording_cache::write_metadata_to_dir(
            temp_dir.path(),
            "test-audio",
            Provider::Mistral,
            "mock-model",
            false,
            &output_text,
            "test-audio.ogg",
            &transcription.metadata,
            transcription.segments.as_deref(),
            transcription.diarization.as_deref(),
        )
        .expect("write sidecar metadata");

        // Write output (simulating what transcribe() does)
        let mut file = tokio::fs::File::create(&output_path)
            .await
            .expect("create output file");
        file.write_all(output_text.as_bytes())
            .await
            .expect("write to file");
        file.sync_all().await.expect("sync file");

        // Verify output file was created and has correct content
        let content = fs::read_to_string(&output_path).expect("read output file");
        assert_eq!(content, "This is a test transcription");

        let sidecar = fs::read_to_string(&sidecar_path).expect("read sidecar file");
        assert!(sidecar.contains("recording: test-audio.ogg"));
        assert!(sidecar.contains("provider: mistral"));
        assert!(sidecar.contains("model: mock-model"));
        assert!(sidecar.contains("transcript: This is a test transcription"));
    }
}
