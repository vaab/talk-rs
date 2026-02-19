//! Transcribe command implementation.
//!
//! Transcribes an audio file to text using the configured transcription backend.
//! Supports writing output to stdout or to a file.

use crate::core::config::{Config, Provider};
use crate::core::error::TalkError;
use crate::core::transcription;
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
/// # Arguments
/// * `args` - Command-line arguments (input file, optional output file)
///
/// # Flow
/// 1. Parse arguments to get input and optional output file paths
/// 2. Load configuration (Mistral API key)
/// 3. Initialize MistralTranscriber
/// 4. Call transcribe_file() on the input audio
/// 5. Write result to stdout (if no output file) or to output file
pub async fn transcribe(
    args: Vec<String>,
    cli_provider: Option<Provider>,
    cli_model: Option<String>,
) -> Result<(), TalkError> {
    // Parse arguments
    let (input_path, output_path) = parse_args(&args)?;

    // Verify input file exists
    if !input_path.exists() {
        return Err(TalkError::Audio(format!(
            "Input file not found: {}",
            input_path.display()
        )));
    }

    // Load configuration
    let config = Config::load(None)?;

    // Resolve provider from CLI override or config default
    let provider = cli_provider
        .or_else(|| config.transcription.as_ref().map(|t| t.default_provider))
        .unwrap_or(Provider::Mistral);

    // Create transcriber via factory
    let transcriber =
        transcription::create_batch_transcriber(&config, provider, cli_model.as_deref())?;

    // Pre-flight: verify provider connectivity and model validity.
    log::info!("validating {} provider configuration", provider);
    transcriber.validate().await?;

    // Transcribe the file
    let transcription = transcriber.transcribe_file(&input_path).await?;

    // Write output
    match output_path {
        Some(output_file) => {
            // Write to file
            let mut file = tokio::fs::File::create(&output_file)
                .await
                .map_err(TalkError::Io)?;
            file.write_all(transcription.text.as_bytes())
                .await
                .map_err(TalkError::Io)?;
            file.sync_all().await.map_err(TalkError::Io)?;
            println!("Transcription saved to: {}", output_file.display());
        }
        None => {
            // Write to stdout
            println!("{}", transcription.text);
        }
    }

    Ok(())
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
        use crate::core::transcription::{BatchTranscriber, MockBatchTranscriber};
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
            .transcribe_file(&input_path)
            .await
            .expect("transcribe should succeed");

        // Write output (simulating what transcribe() does)
        let mut file = tokio::fs::File::create(&output_path)
            .await
            .expect("create output file");
        file.write_all(transcription.text.as_bytes())
            .await
            .expect("write to file");
        file.sync_all().await.expect("sync file");

        // Verify output file was created and has correct content
        let content = fs::read_to_string(&output_path).expect("read output file");
        assert_eq!(content, "This is a test transcription");
    }
}
