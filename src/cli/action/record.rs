//! Record command implementation.
//!
//! Captures audio from the system, encodes it with Opus, and writes to a file.
//! Supports graceful shutdown via SIGINT (Ctrl+C).

use crate::core::audio::cpal_capture::CpalCapture;
use crate::core::audio::{AudioCapture, AudioEncoder, OpusEncoder};
use crate::core::config::Config;
use crate::core::error::TalkError;
use chrono::Local;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

/// Parse command-line arguments for the record command.
///
/// Returns the output file path. If not provided, generates a timestamp-based filename.
pub fn parse_args(args: &[String]) -> Result<PathBuf, TalkError> {
    match args.len() {
        0 => {
            // Generate default filename with timestamp
            let now = Local::now();
            let filename = now.format("memo-%Y-%m-%d-%H-%M-%S.ogg").to_string();
            Ok(PathBuf::from(filename))
        }
        1 => {
            // Use provided filename
            Ok(PathBuf::from(&args[0]))
        }
        _ => Err(TalkError::Audio(
            "record command takes at most one argument (output file path)".to_string(),
        )),
    }
}

/// Record audio from the system and write to a file.
///
/// # Arguments
/// * `args` - Command-line arguments (optional output file path)
///
/// # Flow
/// 1. Parse arguments to get output file path
/// 2. Load configuration (audio settings)
/// 3. Initialize CpalCapture and OpusEncoder
/// 4. Spawn async task to read from capture channel, encode, and write to file
/// 5. Wait for SIGINT (Ctrl+C) to gracefully shutdown
/// 6. Flush encoder and close file
pub async fn record(args: Vec<String>) -> Result<(), TalkError> {
    // Parse arguments
    let output_path = parse_args(&args)?;

    // Load configuration
    let config = Config::load(None)?;

    // Initialize audio capture
    let mut capture = CpalCapture::new(config.audio.clone());
    let mut rx = capture.start()?;

    // Initialize audio encoder
    let mut encoder = OpusEncoder::new(config.audio.clone())?;

    // Create output file
    let mut file = tokio::fs::File::create(&output_path)
        .await
        .map_err(TalkError::Io)?;

    // Spawn async task to read from capture channel, encode, and write to file
    let encode_task = tokio::spawn(async move {
        while let Some(pcm_chunk) = rx.recv().await {
            match encoder.encode(&pcm_chunk) {
                Ok(encoded_data) => {
                    if !encoded_data.is_empty() {
                        if let Err(err) = file.write_all(&encoded_data).await {
                            eprintln!("Error writing to file: {}", err);
                            return Err(TalkError::Io(err));
                        }
                    }
                }
                Err(err) => {
                    eprintln!("Error encoding audio: {}", err);
                    return Err(err);
                }
            }
        }

        // Flush encoder to write remaining data
        match encoder.flush() {
            Ok(remaining_data) => {
                if !remaining_data.is_empty() {
                    if let Err(err) = file.write_all(&remaining_data).await {
                        eprintln!("Error writing flushed data: {}", err);
                        return Err(TalkError::Io(err));
                    }
                }
            }
            Err(err) => {
                eprintln!("Error flushing encoder: {}", err);
                return Err(err);
            }
        }

        // Close file
        if let Err(err) = file.sync_all().await {
            eprintln!("Error syncing file: {}", err);
            return Err(TalkError::Io(err));
        }

        Ok::<(), TalkError>(())
    });

    // Wait for SIGINT (Ctrl+C)
    tokio::signal::ctrl_c()
        .await
        .map_err(|err| TalkError::Audio(format!("Failed to listen for Ctrl+C: {}", err)))?;

    println!("Stopping recording...");

    // Stop capture (closes channel)
    capture.stop()?;

    // Wait for encode task to complete
    match encode_task.await {
        Ok(Ok(())) => {
            println!("Recording saved to: {}", output_path.display());
            Ok(())
        }
        Ok(Err(err)) => Err(err),
        Err(err) => Err(TalkError::Audio(format!("Encode task panicked: {}", err))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_no_args() {
        let args = vec![];
        let result = parse_args(&args).expect("parse should succeed");

        // Should generate a filename with memo- prefix and .ogg extension
        let filename = result.file_name().expect("should have filename");
        let filename_str = filename.to_string_lossy();
        assert!(filename_str.starts_with("memo-"));
        assert!(filename_str.ends_with(".ogg"));
    }

    #[test]
    fn test_parse_args_with_filename() {
        let args = vec!["my-recording.ogg".to_string()];
        let result = parse_args(&args).expect("parse should succeed");

        assert_eq!(result, PathBuf::from("my-recording.ogg"));
    }

    #[test]
    fn test_parse_args_with_path() {
        let args = vec!["/tmp/my-recording.ogg".to_string()];
        let result = parse_args(&args).expect("parse should succeed");

        assert_eq!(result, PathBuf::from("/tmp/my-recording.ogg"));
    }

    #[test]
    fn test_parse_args_too_many_args() {
        let args = vec!["file1.ogg".to_string(), "file2.ogg".to_string()];
        let result = parse_args(&args);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at most one argument"));
    }

    #[tokio::test]
    async fn test_record_pipeline_with_mock_capture() {
        use crate::core::audio::{mock::MockAudioCapture, AudioCapture};
        use crate::core::config::AudioConfig;
        use std::fs;
        use tempfile::TempDir;

        // Create temporary directory for test output
        let temp_dir = TempDir::new().expect("create temp dir");
        let output_path = temp_dir.path().join("test-recording.ogg");

        // Use test audio config instead of loading from file
        let audio_config = AudioConfig {
            sample_rate: 16_000,
            channels: 1,
            bitrate: 32_000,
        };

        // Initialize mock capture
        let mut capture =
            MockAudioCapture::new(audio_config.sample_rate, audio_config.channels, 440.0);
        let mut rx = capture.start().expect("start capture");

        // Initialize encoder
        let mut encoder = OpusEncoder::new(audio_config).expect("create encoder");

        // Create output file
        let mut file = tokio::fs::File::create(&output_path)
            .await
            .expect("create file");

        // Simulate encoding a few chunks
        for _ in 0..3 {
            if let Some(pcm_chunk) = rx.recv().await {
                let encoded_data = encoder.encode(&pcm_chunk).expect("encode");
                if !encoded_data.is_empty() {
                    file.write_all(&encoded_data).await.expect("write to file");
                }
            }
        }

        // Flush encoder
        let remaining_data = encoder.flush().expect("flush");
        if !remaining_data.is_empty() {
            file.write_all(&remaining_data)
                .await
                .expect("write flushed data");
        }

        file.sync_all().await.expect("sync file");

        // Stop capture
        capture.stop().expect("stop capture");

        // Verify file was created and has content
        let metadata = fs::metadata(&output_path).expect("get file metadata");
        assert!(metadata.len() > 0, "output file should have content");
    }
}
