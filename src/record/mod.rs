//! Record command implementation.
//!
//! Captures audio from the system, encodes it with Opus, and writes to a file.
//! Supports graceful shutdown via SIGINT (Ctrl+C).

pub(crate) mod ui;

use crate::core::audio::cpal_capture::CpalCapture;
use crate::core::audio::monitor_capture::MonitorCapture;
use crate::core::audio::{AudioCapture, AudioWriter, OggOpusWriter, WavWriter};
use crate::core::config::{AudioConfig, Config};
use crate::core::error::TalkError;
use chrono::Local;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use tokio::io::{AsyncSeekExt, AsyncWriteExt};

/// Generate a default timestamped filename for a recording.
fn default_filename() -> String {
    let now = Local::now();
    now.format("memo-%Y-%m-%d-%H-%M-%S.ogg").to_string()
}

/// Resolve the output file path from CLI arguments and the configured
/// `output_dir`.
///
/// - No arguments → `<output_dir>/memo-YYYY-MM-DD-HH-MM-SS.ogg`
/// - One argument → used as-is
/// - More than one → error
fn resolve_output_path(args: &[String], output_dir: &Path) -> Result<PathBuf, TalkError> {
    match args.len() {
        0 => Ok(output_dir.join(default_filename())),
        1 => Ok(PathBuf::from(&args[0])),
        _ => Err(TalkError::Audio(
            "record command takes at most one argument (output file path)".to_string(),
        )),
    }
}

/// Parse command-line arguments for the record command.
///
/// Returns the output file path. If not provided, generates a
/// timestamp-based filename inside the configured `output_dir`.
pub fn parse_args(args: &[String]) -> Result<PathBuf, TalkError> {
    let config = Config::load(None)?;
    resolve_output_path(args, &config.output_dir)
}

fn create_writer(path: &Path, config: AudioConfig) -> Result<Box<dyn AudioWriter>, TalkError> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("wav") => Ok(Box::new(WavWriter::new(config))),
        _ => Ok(Box::new(OggOpusWriter::new(config)?)),
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
pub async fn record(args: Vec<String>, monitor: bool) -> Result<(), TalkError> {
    // Parse arguments
    let output_path = parse_args(&args)?;

    // Initialize audio capture
    let mut capture: Box<dyn AudioCapture> = if monitor {
        log::info!("recording with mic+monitor (PipeWire)");
        Box::new(MonitorCapture::new(AudioConfig::new()))
    } else {
        Box::new(CpalCapture::new(AudioConfig::new()))
    };
    let mut rx = capture.start()?;

    // Initialize audio writer
    let mut writer = create_writer(&output_path, AudioConfig::new())?;
    let is_wav = matches!(
        output_path.extension().and_then(|e| e.to_str()),
        Some("wav")
    );

    // Create output file
    let mut file = tokio::fs::File::create(&output_path)
        .await
        .map_err(TalkError::Io)?;

    // Spawn async task to read from capture channel, encode, and write to file
    let encode_task = tokio::spawn(async move {
        let header = match writer.header() {
            Ok(bytes) => bytes,
            Err(err) => {
                log::error!("error creating header: {}", err);
                return Err(err);
            }
        };
        if let Err(err) = file.write_all(&header).await {
            log::error!("error writing header: {}", err);
            return Err(TalkError::Io(err));
        }

        while let Some(pcm_chunk) = rx.recv().await {
            let encoded_data = match writer.write_pcm(&pcm_chunk) {
                Ok(data) => data,
                Err(err) => {
                    log::error!("error encoding audio: {}", err);
                    return Err(err);
                }
            };
            if !encoded_data.is_empty() {
                if let Err(err) = file.write_all(&encoded_data).await {
                    log::error!("error writing to file: {}", err);
                    return Err(TalkError::Io(err));
                }
            }
        }

        // Finalize writer to write remaining data
        match writer.finalize() {
            Ok(remaining_data) => {
                if !remaining_data.is_empty() {
                    if is_wav {
                        if let Err(err) = file.seek(SeekFrom::Start(0)).await {
                            log::error!("error seeking to start: {}", err);
                            return Err(TalkError::Io(err));
                        }
                    }
                    if let Err(err) = file.write_all(&remaining_data).await {
                        log::error!("error writing flushed data: {}", err);
                        return Err(TalkError::Io(err));
                    }
                }
            }
            Err(err) => {
                log::error!("error finalizing writer: {}", err);
                return Err(err);
            }
        }

        // Close file
        if let Err(err) = file.sync_all().await {
            log::error!("error syncing file: {}", err);
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
    fn test_resolve_output_path_no_args_uses_output_dir() {
        let output_dir = PathBuf::from("/tmp/test-output");
        let args: Vec<String> = vec![];
        let result = resolve_output_path(&args, &output_dir).expect("resolve should succeed");

        // Path must live inside output_dir.
        assert_eq!(
            result.parent().expect("should have parent"),
            output_dir,
            "default recording should be placed in output_dir"
        );

        // Filename must follow the memo-YYYY-MM-DD-HH-MM-SS.ogg pattern.
        let filename = result
            .file_name()
            .expect("should have filename")
            .to_string_lossy();
        assert!(
            filename.starts_with("memo-"),
            "filename should start with memo-"
        );
        assert!(filename.ends_with(".ogg"), "filename should end with .ogg");
    }

    #[test]
    fn test_resolve_output_path_with_filename() {
        let output_dir = PathBuf::from("/tmp/test-output");
        let args = vec!["my-recording.ogg".to_string()];
        let result = resolve_output_path(&args, &output_dir).expect("resolve should succeed");

        assert_eq!(result, PathBuf::from("my-recording.ogg"));
    }

    #[test]
    fn test_resolve_output_path_with_absolute_path() {
        let output_dir = PathBuf::from("/tmp/test-output");
        let args = vec!["/tmp/my-recording.ogg".to_string()];
        let result = resolve_output_path(&args, &output_dir).expect("resolve should succeed");

        assert_eq!(result, PathBuf::from("/tmp/my-recording.ogg"));
    }

    #[test]
    fn test_resolve_output_path_too_many_args() {
        let output_dir = PathBuf::from("/tmp/test-output");
        let args = vec!["file1.ogg".to_string(), "file2.ogg".to_string()];
        let result = resolve_output_path(&args, &output_dir);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at most one argument"));
    }

    #[tokio::test]
    async fn test_record_pipeline_with_mock_capture() {
        use crate::core::audio::{
            mock::MockAudioCapture, AudioCapture, AudioWriter, OggOpusWriter,
        };
        use crate::core::config::AudioConfig;
        use std::fs;
        use tempfile::TempDir;

        // Create temporary directory for test output
        let temp_dir = TempDir::new().expect("create temp dir");
        let output_path = temp_dir.path().join("test-recording.ogg");

        // Use test audio config instead of loading from file
        let audio_config = AudioConfig::new();

        // Initialize mock capture
        let mut capture =
            MockAudioCapture::new(audio_config.sample_rate, audio_config.channels, 440.0);
        let mut rx = capture.start().expect("start capture");

        // Initialize writer
        let mut writer = OggOpusWriter::new(audio_config).expect("create writer");

        // Create output file
        let mut file = tokio::fs::File::create(&output_path)
            .await
            .expect("create file");

        // Write header
        let header = writer.header().expect("header");
        file.write_all(&header).await.expect("write header");

        // Simulate encoding a few chunks
        for _ in 0..3 {
            if let Some(pcm_chunk) = rx.recv().await {
                let encoded_data = writer.write_pcm(&pcm_chunk).expect("encode");
                if !encoded_data.is_empty() {
                    file.write_all(&encoded_data).await.expect("write to file");
                }
            }
        }

        // Finalize writer
        let remaining_data = writer.finalize().expect("finalize");
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

        let bytes = fs::read(&output_path).expect("read output file");
        assert!(bytes.starts_with(b"OggS"), "output should start with OggS");
        let has_opus_head = bytes
            .windows(b"OpusHead".len())
            .any(|window| window == b"OpusHead");
        assert!(has_opus_head, "output should contain OpusHead");
    }
}
