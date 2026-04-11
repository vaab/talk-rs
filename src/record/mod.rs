//! Record command implementation.
//!
//! Captures audio from the system, encodes it with Opus, and writes to a file.
//! Supports graceful shutdown via SIGINT (Ctrl+C).

pub(crate) mod audio;
mod entries;
pub(crate) mod player;
pub(crate) mod ui;

use crate::audio::cpal_capture::CpalCapture;
use crate::audio::monitor_capture::MonitorCapture;
use crate::audio::{AudioCapture, AudioWriter, OggOpusWriter, WavWriter};
use crate::config::{AudioConfig, Config};
use crate::error::TalkError;
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
/// - No arguments → `<output_dir>/YYYY/MM/memo-YYYY-MM-DD-HH-MM-SS.ogg`
///   (auto-namespaced by year and month to keep the flat directory from
///   growing unbounded).
/// - One argument → used as-is
/// - More than one → error
///
/// This is a pure computation: it does not create the directory.  The
/// caller is responsible for calling [`std::fs::create_dir_all`] on the
/// parent before opening the file.
fn resolve_output_path(args: &[String], output_dir: &Path) -> Result<PathBuf, TalkError> {
    match args.len() {
        0 => {
            let now = Local::now();
            let year = now.format("%Y").to_string();
            let month = now.format("%m").to_string();
            Ok(output_dir.join(year).join(month).join(default_filename()))
        }
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

    // Ensure the parent directory exists.  For auto-generated paths this
    // creates the `YYYY/MM/` subdirectory; for user-provided paths it
    // creates any missing intermediate directories (principle of least
    // surprise when the user passes a nested path).
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await.map_err(|err| {
                TalkError::Io(std::io::Error::new(
                    err.kind(),
                    format!(
                        "failed to create recording directory {}: {}",
                        parent.display(),
                        err
                    ),
                ))
            })?;
        }
    }

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
    fn test_resolve_output_path_no_args_nests_by_year_and_month() {
        let output_dir = PathBuf::from("/tmp/test-output");
        let args: Vec<String> = vec![];
        let result = resolve_output_path(&args, &output_dir).expect("resolve should succeed");

        // Path must live inside output_dir/YYYY/MM/.
        let month_dir = result.parent().expect("should have month parent");
        let year_dir = month_dir.parent().expect("should have year parent");
        let root = year_dir.parent().expect("should have root parent");

        assert_eq!(
            root, output_dir,
            "root above the YYYY/MM subdirs should be output_dir"
        );

        let year_name = year_dir
            .file_name()
            .expect("year dir name")
            .to_string_lossy();
        let month_name = month_dir
            .file_name()
            .expect("month dir name")
            .to_string_lossy();

        assert_eq!(year_name.len(), 4, "year segment should be 4 digits");
        assert!(
            year_name.chars().all(|c| c.is_ascii_digit()),
            "year segment should be all digits, got: {}",
            year_name
        );
        assert_eq!(month_name.len(), 2, "month segment should be 2 digits");
        assert!(
            month_name.chars().all(|c| c.is_ascii_digit()),
            "month segment should be all digits, got: {}",
            month_name
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

        // Sanity: the year/month in the directory path should match the
        // year/month embedded in the filename.
        assert!(
            filename.contains(&format!("memo-{}-{}-", year_name, month_name)),
            "filename {} should carry the same year-month as its parent dirs {}/{}",
            filename,
            year_name,
            month_name
        );
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
        use crate::audio::{mock::MockAudioCapture, AudioCapture, AudioWriter, OggOpusWriter};
        use crate::config::AudioConfig;
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
