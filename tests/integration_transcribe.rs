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

#[test]
fn transcribe_cli_reports_missing_input_without_loading_user_configuration() {
    let dir = TempDir::new().expect("tempdir");
    let missing = dir.path().join("absent.ogg");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_talk-rs"))
        .arg("transcribe")
        .arg(&missing)
        .env("XDG_CONFIG_HOME", dir.path())
        .env_remove("TALK_RS_LOG_FILE")
        .output()
        .expect("run isolated CLI");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stdout, b"");
    assert_eq!(
        String::from_utf8(output.stderr).expect("UTF-8 error"),
        format!(
            "error: Audio error: Input file not found: {}\n",
            missing.display()
        )
    );
}

#[test]
fn transcribe_cli_writes_saved_pick_to_requested_output_file() {
    let dir = TempDir::new().expect("tempdir");
    let config_dir = dir.path().join("config").join("talk-rs");
    fs::create_dir_all(&config_dir).expect("isolated config dir");
    fs::write(
        config_dir.join("config.yaml"),
        format!("output_dir: {}\nproviders: {{}}\n", dir.path().display()),
    )
    .expect("isolated config");
    let audio = dir.path().join("memo.ogg");
    fs::write(&audio, b"audio").expect("fixture audio");
    talk_rs::recording_cache::write_pick(
        &audio,
        "openai",
        "whisper-1",
        false,
        "edited transcript\nsecond line",
    )
    .expect("saved edit");
    let destination = dir.path().join("transcript.txt");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_talk-rs"))
        .arg("transcribe")
        .arg(&audio)
        .arg(&destination)
        .env("XDG_CONFIG_HOME", dir.path().join("config"))
        .env("XDG_CACHE_HOME", dir.path().join("cache"))
        .env_remove("TALK_RS_LOG_FILE")
        .output()
        .expect("run isolated CLI");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stderr, b"");
    assert_eq!(
        String::from_utf8(output.stdout).expect("UTF-8 success"),
        format!("Transcription saved to: {}\n", destination.display())
    );
    assert_eq!(
        fs::read_to_string(destination).expect("saved output"),
        "edited transcript\nsecond line"
    );
}

#[test]
fn explicit_model_cli_reads_its_sidecar_instead_of_authoritative_pick() {
    let dir = TempDir::new().expect("tempdir");
    let config_dir = dir.path().join("config").join("talk-rs");
    fs::create_dir_all(&config_dir).expect("isolated config dir");
    fs::write(
        config_dir.join("config.yaml"),
        format!("output_dir: {}\nproviders: {{}}\n", dir.path().display()),
    )
    .expect("isolated config");
    let audio = dir.path().join("memo.ogg");
    fs::write(&audio, b"audio").expect("fixture audio");
    talk_rs::recording_cache::write_pick(&audio, "mistral", "chosen", false, "edited pick")
        .expect("saved edit");
    talk_rs::recording_cache::TranscriptionCache::store(
        &audio,
        Provider::OpenAI,
        "whisper-1",
        false,
        &talk_rs::transcription::TranscriptionResult {
            text: "model-specific output".into(),
            ..Default::default()
        },
    )
    .expect("saved model sidecar");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_talk-rs"))
        .args(["transcribe", "--provider", "openai", "--model", "whisper-1"])
        .arg(&audio)
        .env("XDG_CONFIG_HOME", dir.path().join("config"))
        .env("XDG_CACHE_HOME", dir.path().join("cache"))
        .env_remove("TALK_RS_LOG_FILE")
        .output()
        .expect("run isolated CLI");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stderr, b"");
    assert_eq!(output.stdout, b"model-specific output\n");
    assert_eq!(
        talk_rs::recording_cache::read_pick(&audio),
        Some((
            Provider::Mistral,
            "chosen".into(),
            false,
            "edited pick".into()
        ))
    );
}

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
        tts_model: "voxtral-mini-tts-latest".to_string(),
        tts_voice: None,
        tts_voices: None,
    };
    let temp_dir = TempDir::new().expect("create temp dir");
    let audio_path = temp_dir.path().join("missing.wav");
    let config = Config {
        output_dir: temp_dir.path().to_path_buf(),
        providers: ProvidersConfig {
            mistral: Some(config),
            openai: None,
            parakeet: None,
            kokoro: None,
        },
        indicators: None,
        transcription: None,
        speak: None,
        paste: None,
        audio: None,
        recording: None,
    };

    let sink: std::sync::Arc<dyn talk_rs::telemetry::TelemetrySink> =
        std::sync::Arc::new(talk_rs::telemetry::NoOpSink);
    let result = transcribe_audio(
        &audio_path,
        &config,
        Provider::Mistral,
        None,
        false,
        talk_rs::transcription::TranscribeOptions {
            allow_api: false,
            policy: talk_rs::transcription::RequestTimeoutPolicy::Proportional,
            cancel_token: None,
            skip_legacy_lock: false,
        },
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
            parakeet: None,
            kokoro: None,
        },
        indicators: None,
        transcription: None,
        speak: None,
        paste: None,
        audio: None,
        recording: None,
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
        talk_rs::transcription::TranscribeOptions {
            allow_api: true,
            policy: talk_rs::transcription::RequestTimeoutPolicy::Proportional,
            cancel_token: None,
            skip_legacy_lock: false,
        },
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
