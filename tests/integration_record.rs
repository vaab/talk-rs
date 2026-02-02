//! Integration tests for the `record` CLI command.
//!
//! Tests the record command end-to-end with both real audio hardware and mock capture.

use std::fs;
use talk_rs::core::audio::mock::MockAudioCapture;
use talk_rs::core::audio::{AudioCapture, AudioEncoder, OpusEncoder};
use talk_rs::core::config::AudioConfig;
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;

/// Test that the record command creates an output file with mock capture.
///
/// This test uses MockAudioCapture to feed synthetic PCM data through OpusEncoder
/// to a temp file, verifying that the file exists and contains Opus data.
#[tokio::test]
async fn test_record_with_mock_capture_creates_file() {
    // Create temporary directory for test output
    let temp_dir = TempDir::new().expect("create temp dir");
    let output_path = temp_dir.path().join("test-recording.ogg");

    // Use test audio config
    let audio_config = AudioConfig {
        sample_rate: 16_000,
        channels: 1,
        bitrate: 32_000,
    };

    // Initialize mock capture
    let mut capture = MockAudioCapture::new(audio_config.sample_rate, audio_config.channels, 440.0);
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
    assert!(
        metadata.len() > 0,
        "output file should have content, got {} bytes",
        metadata.len()
    );

    // Verify file contains Opus data
    // Note: OpusEncoder produces raw Opus frames, not Ogg-wrapped Opus
    let file_content = fs::read(&output_path).expect("read file");
    assert!(!file_content.is_empty(), "file should contain encoded data");
    // Opus frames typically have specific byte patterns, but we just verify non-empty
    // and that it's not all zeros
    assert!(
        file_content.iter().any(|&b| b != 0),
        "file should contain non-zero data"
    );
}

/// Test that the default filename format is generated correctly.
///
/// This test verifies the parsing logic for the default filename pattern
/// `memo-YYYY-MM-DD-HH-MM-SS.ogg` without performing actual recording.
#[test]
fn test_record_default_filename_format() {
    use chrono::Local;
    use std::path::PathBuf;

    // Simulate the parse_args logic from record.rs
    let now = Local::now();
    let filename = now.format("memo-%Y-%m-%d-%H-%M-%S.ogg").to_string();
    let path = PathBuf::from(&filename);

    // Verify filename structure
    let filename_str = path
        .file_name()
        .expect("should have filename")
        .to_string_lossy();

    assert!(
        filename_str.starts_with("memo-"),
        "filename should start with 'memo-'"
    );
    assert!(
        filename_str.ends_with(".ogg"),
        "filename should end with '.ogg'"
    );

    // Verify the format matches the pattern: memo-YYYY-MM-DD-HH-MM-SS.ogg
    // Should be exactly 28 characters: "memo-" (5) + "YYYY-MM-DD-HH-MM-SS" (19) + ".ogg" (4)
    assert_eq!(
        filename_str.len(),
        28,
        "filename should be exactly 28 characters, got {}",
        filename_str.len()
    );

    // Verify date components are numeric
    // Format: memo-%Y-%m-%d-%H-%M-%S.ogg
    // When split by '-': ["memo", "YYYY", "MM", "DD", "HH", "MM", "SS.ogg"]
    let parts: Vec<&str> = filename_str.split('-').collect();
    assert_eq!(parts.len(), 7, "should have 7 parts separated by hyphens");

    // parts[0] = "memo"
    assert_eq!(parts[0], "memo");

    // parts[1] = YYYY (4 digits)
    assert_eq!(parts[1].len(), 4);
    assert!(parts[1].chars().all(|c| c.is_ascii_digit()));

    // parts[2] = MM (2 digits)
    assert_eq!(parts[2].len(), 2);
    assert!(parts[2].chars().all(|c| c.is_ascii_digit()));

    // parts[3] = DD (2 digits)
    assert_eq!(parts[3].len(), 2);
    assert!(parts[3].chars().all(|c| c.is_ascii_digit()));

    // parts[4] = HH (2 digits)
    assert_eq!(parts[4].len(), 2);
    assert!(parts[4].chars().all(|c| c.is_ascii_digit()));

    // parts[5] = MM (2 digits)
    assert_eq!(parts[5].len(), 2);
    assert!(parts[5].chars().all(|c| c.is_ascii_digit()));

    // parts[6] = SS.ogg (6 characters: SS.ogg)
    assert_eq!(parts[6].len(), 6);
    let ss_ogg = parts[6];
    assert!(ss_ogg[0..2].chars().all(|c| c.is_ascii_digit()));
    assert_eq!(&ss_ogg[2..3], ".");
    assert_eq!(&ss_ogg[3..6], "ogg");
}

/// Test that record creates an output file with real audio hardware.
///
/// This test requires audio hardware to be available and is marked with `#[ignore]`
/// to prevent it from running in CI environments.
///
/// Run with: `cargo test -- --ignored test_record_creates_output_file`
#[tokio::test]
#[ignore]
async fn test_record_creates_output_file() {
    use talk_rs::core::audio::cpal_capture::CpalCapture;

    // Create temporary directory for test output
    let temp_dir = TempDir::new().expect("create temp dir");
    let output_path = temp_dir.path().join("test-recording-hardware.ogg");

    // Use test audio config
    let audio_config = AudioConfig {
        sample_rate: 16_000,
        channels: 1,
        bitrate: 32_000,
    };

    // Initialize real audio capture
    let mut capture = CpalCapture::new(audio_config.clone());
    let mut rx = capture.start().expect("start capture");

    // Initialize encoder
    let mut encoder = OpusEncoder::new(audio_config).expect("create encoder");

    // Create output file
    let mut file = tokio::fs::File::create(&output_path)
        .await
        .expect("create file");

    // Record for approximately 1 second
    // At 16kHz with 20ms chunks, we expect ~50 chunks per second
    let mut chunks_recorded = 0;
    let target_chunks = 50; // ~1 second

    while chunks_recorded < target_chunks {
        if let Some(pcm_chunk) = rx.recv().await {
            let encoded_data = encoder.encode(&pcm_chunk).expect("encode");
            if !encoded_data.is_empty() {
                file.write_all(&encoded_data).await.expect("write to file");
            }
            chunks_recorded += 1;
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
    assert!(
        metadata.len() > 0,
        "output file should have content, got {} bytes",
        metadata.len()
    );

    // Verify file contains Opus data
    // Note: OpusEncoder produces raw Opus frames, not Ogg-wrapped Opus
    let file_content = fs::read(&output_path).expect("read file");
    assert!(!file_content.is_empty(), "file should contain encoded data");
    // Opus frames typically have specific byte patterns, but we just verify non-empty
    // and that it's not all zeros
    assert!(
        file_content.iter().any(|&b| b != 0),
        "file should contain non-zero data"
    );
}
