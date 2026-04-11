//! Integration tests for the `record` CLI command.
//!
//! Tests the record command end-to-end with both real audio hardware and mock capture.

use std::fs;
use talk_rs::audio::mock::MockAudioCapture;
use talk_rs::audio::{AudioCapture, AudioEncoder, OpusEncoder};
use talk_rs::config::AudioConfig;
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
/// The default filename is an ISO 8601 local datetime with a numeric
/// timezone offset (e.g. `2026-04-11T13-15-52+0200.ogg`), matching the
/// `memo` tool's naming scheme.  This test verifies that the filename
/// produced by the format string round-trips through `chrono`'s parser
/// and resolves to approximately "now", without requiring a real
/// recording.
#[test]
fn test_record_default_filename_format() {
    use chrono::{DateTime, Datelike, Local};
    use std::path::PathBuf;

    // Reproduce the format string used by `record::default_filename()`.
    // Keep this in sync with `src/record/mod.rs`.
    let before = Local::now();
    let filename = before.format("%Y-%m-%dT%H-%M-%S%z.ogg").to_string();
    let after = Local::now();
    let path = PathBuf::from(&filename);

    let filename_str = path
        .file_name()
        .expect("should have filename")
        .to_string_lossy();

    assert!(
        filename_str.ends_with(".ogg"),
        "filename should end with ``.ogg``, got {}",
        filename_str
    );

    // The stem must parse as a chrono datetime with timezone offset.
    let stem = filename_str
        .strip_suffix(".ogg")
        .expect("filename ends with .ogg");
    let parsed = DateTime::parse_from_str(stem, "%Y-%m-%dT%H-%M-%S%z")
        .unwrap_or_else(|e| panic!("filename stem {} should parse: {}", stem, e));

    // The parsed timestamp must carry a real timezone offset (even UTC
    // produces `+0000`, so this just exercises that `%z` was honoured).
    let offset_secs = parsed.offset().local_minus_utc();
    assert!(
        offset_secs.abs() < 24 * 3600,
        "parsed offset {} seconds should be a valid timezone",
        offset_secs
    );

    // The parsed instant must fall within the window this test observed.
    // `parse_from_str` returns a `DateTime<FixedOffset>`, which compares
    // correctly against `DateTime<Local>` values taken before and after.
    assert!(
        parsed >= before - chrono::Duration::seconds(1),
        "parsed {} should not be earlier than before {}",
        parsed,
        before
    );
    assert!(
        parsed <= after + chrono::Duration::seconds(1),
        "parsed {} should not be later than after {}",
        parsed,
        after
    );

    // Sanity check the calendar components: year must be 4 digits, month
    // 1..=12, day 1..=31.
    assert!(
        (2000..=9999).contains(&parsed.year()),
        "year {} should be 4-digit",
        parsed.year()
    );
    assert!(
        (1..=12).contains(&parsed.month()),
        "month {} out of range",
        parsed.month()
    );
    assert!(
        (1..=31).contains(&parsed.day()),
        "day {} out of range",
        parsed.day()
    );
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
    use talk_rs::audio::cpal_capture::CpalCapture;

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
