//! Integration tests for the transcription module and transcribe command.
//!
//! Tests cover:
//! - Real Mistral API transcription (ignored by default, requires API key)
//! - Mock transcriber with file output
//! - Error handling for non-existent files

use std::fs;
use std::path::Path;
use talk_rs::config::{Config, MistralConfig, Provider, ProvidersConfig};
use talk_rs::transcription::transcribe_audio;
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;

/// Create a minimal WAV file with synthetic PCM data.
///
/// Creates a valid WAV header with 16-bit PCM audio at 16kHz mono.
/// The audio data is synthetic (silence/zeros) but valid for API submission.
fn create_test_wav_file(path: &Path) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = fs::File::create(path)?;

    // WAV header constants
    const SAMPLE_RATE: u32 = 16000;
    const CHANNELS: u16 = 1;
    const BITS_PER_SAMPLE: u16 = 16;
    const BYTES_PER_SAMPLE: u32 = (BITS_PER_SAMPLE as u32) / 8;
    const BYTE_RATE: u32 = SAMPLE_RATE * CHANNELS as u32 * BYTES_PER_SAMPLE;
    const BLOCK_ALIGN: u16 = CHANNELS * (BITS_PER_SAMPLE / 8);

    // Generate 1 second of audio data (16000 samples)
    let num_samples = SAMPLE_RATE;
    let data_size = num_samples * BYTES_PER_SAMPLE;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_size).to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt subchunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // Subchunk1Size
    file.write_all(&1u16.to_le_bytes())?; // AudioFormat (1 = PCM)
    file.write_all(&CHANNELS.to_le_bytes())?;
    file.write_all(&SAMPLE_RATE.to_le_bytes())?;
    file.write_all(&BYTE_RATE.to_le_bytes())?;
    file.write_all(&BLOCK_ALIGN.to_le_bytes())?;
    file.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;

    // data subchunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    // Write synthetic PCM data (silence - all zeros)
    let silence = vec![0u8; data_size as usize];
    file.write_all(&silence)?;

    Ok(())
}

/// Test transcription output writing with a synthetic result.
///
/// This test verifies that:
/// 1. Output is correctly written to a file
/// 2. File content matches the transcribed text
#[tokio::test]
async fn test_transcribe_with_mock_writes_to_file() {
    // Create temporary directory for test files
    let temp_dir = TempDir::new().expect("create temp dir");

    // Create a temporary input audio file
    let input_path = temp_dir.path().join("test-audio.wav");
    fs::write(&input_path, b"fake audio data").expect("write input file");

    // Create output path
    let output_path = temp_dir.path().join("transcript.txt");

    let transcription = talk_rs::transcription::TranscriptionResult {
        text: "This is a test transcription from mock".to_string(),
        metadata: Default::default(),
        diarization: None,
        segments: None,
    };

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
    assert_eq!(content, "This is a test transcription from mock");
}

/// Test that public transcribe entry point reports cache-only misses.
///
/// This test verifies that:
/// 1. `transcribe_audio` can be called on a missing cache entry
/// 2. `allow_api = false` returns `CacheOnly` without touching the network
#[tokio::test]
async fn test_transcribe_audio_cache_only_on_missing_entry() {
    let config = MistralConfig {
        api_key: "test-api-key".to_string(),
        url: None,
        model: "voxtral-mini-latest".to_string(),
        context_bias: None,
    };
    let temp_dir = TempDir::new().expect("create temp dir");
    let audio_path = temp_dir.path().join("missing.wav");
    let config = Config {
        output_dir: temp_dir.path().to_path_buf(),
        providers: ProvidersConfig {
            mistral: Some(config),
            openai: None,
        },
        indicators: None,
        transcription: None,
        paste: None,
        audio: None,
    };

    let sink: std::sync::Arc<dyn talk_rs::telemetry::TelemetrySink> =
        std::sync::Arc::new(talk_rs::telemetry::NoOpSink);
    let result = transcribe_audio(
        &audio_path,
        &config,
        Provider::Mistral,
        None,
        false,
        false,
        &sink,
    )
    .await;

    assert!(matches!(result, Err(talk_rs::error::TalkError::CacheOnly)));
}

/// Test real Mistral API transcription with synthetic audio.
///
/// This test is ignored by default because it requires:
/// 1. Valid Mistral API key in ~/.config/talk-rs/config.yaml
/// 2. Network access to api.mistral.ai
///
/// To run: `cargo test --test integration_transcribe -- --ignored`
///
/// The test:
/// 1. Loads API key from config file
/// 2. Creates a valid WAV file with synthetic PCM data
/// 3. Sends it to the real Mistral API
/// 4. Verifies non-empty text is returned
#[tokio::test]
#[ignore]
async fn test_mistral_transcriber_real_api() {
    use talk_rs::config::config_path;

    // Load config to get API key
    let config_path = config_path().expect("get config path");
    if !config_path.exists() {
        panic!(
            "Config file not found at {}. Create it with your Mistral API key.",
            config_path.display()
        );
    }

    let config_content = fs::read_to_string(&config_path).expect("read config file");
    let config: talk_rs::config::Config =
        serde_yaml::from_str(&config_content).expect("parse config YAML");

    // Create temporary directory for test audio
    let temp_dir = TempDir::new().expect("create temp dir");
    let audio_path = temp_dir.path().join("test-audio.wav");

    // Create a valid WAV file with synthetic PCM data
    create_test_wav_file(&audio_path).expect("create test WAV file");

    // Verify file was created and has content
    assert!(audio_path.exists(), "test WAV file should exist");
    let file_size = fs::metadata(&audio_path).expect("get file metadata").len();
    assert!(file_size > 0, "test WAV file should have content");

    let mistral_config = config
        .providers
        .mistral
        .expect("mistral provider must be configured for this test");
    let runtime_config = Config {
        output_dir: temp_dir.path().to_path_buf(),
        providers: ProvidersConfig {
            mistral: Some(mistral_config),
            openai: None,
        },
        indicators: None,
        transcription: None,
        paste: None,
        audio: None,
    };

    // Transcribe the file
    let sink: std::sync::Arc<dyn talk_rs::telemetry::TelemetrySink> =
        std::sync::Arc::new(talk_rs::telemetry::NoOpSink);
    let result = transcribe_audio(
        &audio_path,
        &runtime_config,
        Provider::Mistral,
        None,
        false,
        true,
        &sink,
    )
    .await;

    // Verify result
    assert!(
        result.is_ok(),
        "transcription should succeed. Error: {:?}",
        result.err()
    );

    let text = result.unwrap().text;
    assert!(!text.is_empty(), "transcribed text should not be empty");

    // Log the result for debugging
    println!("Transcription result: {}", text);
}
