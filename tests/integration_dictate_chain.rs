//! Chain failures must still export the durable recording requested by --save.

use std::path::Path;
use std::time::Duration;

use indoc::formatdoc;
use talk_rs::audio::{AudioWriter, WavWriter};
use talk_rs::config::AudioConfig;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn write_input_wav(path: &Path) {
    let mut writer = WavWriter::new(AudioConfig::new());
    let mut bytes = writer.header().expect("WAV header");
    bytes.extend(writer.write_pcm(&vec![9000; 8000]).expect("WAV audio"));
    let header = writer.finalize().expect("final WAV header");
    bytes[..header.len()].copy_from_slice(&header);
    std::fs::write(path, bytes).expect("input recording");
}

async fn run_failing_chain(status: u16) -> (tempfile::TempDir, std::process::Output) {
    let dir = tempfile::tempdir().expect("isolated test directory");
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"id": "voxtral-mini-2602"}, {"id": "voxtral-mini-2507"}]
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(status).set_body_string("unavailable"))
        .mount(&server)
        .await;

    let config_dir = dir.path().join("config/talk-rs");
    std::fs::create_dir_all(&config_dir).expect("config directory");
    std::fs::write(
        config_dir.join("config.yaml"),
        formatdoc! {"
        output_dir: {}
        providers:
          mistral:
            api_key: fake
            url: {}
        transcription:
          chains:
            fail:
              - mistral/voxtral-mini-2602
              - mistral/voxtral-mini-2507
    ", dir.path().display(), server.uri()},
    )
    .expect("isolated config");
    let audio = dir.path().join("input.wav");
    write_input_wav(&audio);
    let save = dir.path().join("nested/saved.ogg");
    let output = tokio::time::timeout(
        Duration::from_secs(15),
        tokio::process::Command::new(env!("CARGO_BIN_EXE_talk-rs"))
            .args([
                "dictate",
                "--chain",
                "fail",
                "--no-sounds",
                "--no-overlay",
                "--no-paste",
                "--no-bt-auto-switch",
                "--target-window",
                "0",
            ])
            .arg("--input-audio-file")
            .arg(&audio)
            .arg("--save")
            .arg(&save)
            .env("XDG_CONFIG_HOME", dir.path().join("config"))
            .env("XDG_CACHE_HOME", dir.path().join("cache"))
            .env(
                "TALK_RS_VALIDATE_CACHE_PATH",
                dir.path().join("validate-cache.yaml"),
            )
            .env_remove("TALK_RS_PROVIDERS_MISTRAL_API_KEY")
            .env_remove("TALK_RS_PROVIDERS_MISTRAL_URL")
            .env_remove("TALK_RS_LOG_FILE")
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_DISPLAY")
            .output(),
    )
    .await
    .expect("dictation must finish")
    .expect("run isolated dictation");
    (dir, output)
}

#[tokio::test]
async fn chain_busy_exhaustion_still_saves_audio_and_returns_error() {
    let (dir, output) = run_failing_chain(429).await;
    let saved = std::fs::read(dir.path().join("nested/saved.ogg"))
        .expect("--save audio on busy exhaustion");
    assert!(saved.starts_with(b"OggS"));
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("status=429"));
}

#[tokio::test]
async fn chain_permanent_failure_still_saves_audio_and_returns_error() {
    let (dir, output) = run_failing_chain(401).await;
    let saved = std::fs::read(dir.path().join("nested/saved.ogg"))
        .expect("--save audio on permanent error");
    assert!(saved.starts_with(b"OggS"));
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("status=401"));
}
