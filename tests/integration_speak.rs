//! Integration tests for the `speak` (text-to-speech) surface.
//!
//! Covers the remote Mistral / Voxtral TTS provider end-to-end against
//! a wiremock-mocked `POST /v1/audio/speech` endpoint:
//!
//! 1. Build a tiny but valid 24 kHz mono 16-bit WAV in the test.
//! 2. base64-encode it and serve it as `{ "audio_data": "<b64>" }`.
//! 3. Drive the PUBLIC synthesis factory
//!    ([`talk_rs::synthesis::create_oneshot_synthesizer`]) with a
//!    `Config` whose `providers.mistral.url` points at the mock server,
//!    exactly as the `speak` command would.
//! 4. Assert the synthesized PCM round-trips, and that persisting it
//!    with the WAV writer produces a file the WAV parser accepts.
//!
//! No real network / API key is used — wiremock only.

use base64::Engine;
use talk_rs::config::{Config, MistralConfig, ProvidersConfig, SynthesisProvider};
use talk_rs::synthesis::{synthesize, SynthesisRequest};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Extract mono i16 PCM from a canonical (44-byte-header) WAV.  Local
/// to the test so we do not depend on any crate-private parser.
fn wav_data_pcm(bytes: &[u8]) -> Vec<i16> {
    assert_eq!(&bytes[0..4], b"RIFF", "not a RIFF file");
    assert_eq!(&bytes[8..12], b"WAVE", "not a WAVE file");
    // Locate the `data` chunk.
    let pos = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("data chunk present");
    let size = u32::from_le_bytes([
        bytes[pos + 4],
        bytes[pos + 5],
        bytes[pos + 6],
        bytes[pos + 7],
    ]) as usize;
    let start = pos + 8;
    bytes[start..start + size]
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// Build a canonical 44-byte-header mono 16-bit WAV around `samples`.
fn make_wav(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let channels: u16 = 1;
    let bits = 16u16;
    let byte_rate = sample_rate * channels as u32 * (bits as u32 / 8);
    let block_align = channels * (bits / 8);
    let data_size = (samples.len() * 2) as u32;

    let mut v = Vec::new();
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_size).to_le_bytes());
    v.extend_from_slice(b"WAVE");
    v.extend_from_slice(b"fmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes()); // PCM
    v.extend_from_slice(&channels.to_le_bytes());
    v.extend_from_slice(&sample_rate.to_le_bytes());
    v.extend_from_slice(&byte_rate.to_le_bytes());
    v.extend_from_slice(&block_align.to_le_bytes());
    v.extend_from_slice(&bits.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}

/// Minimal `Config` with a Mistral provider pointed at `url`.
fn config_with_mistral_url(url: &str, output_dir: &std::path::Path) -> Config {
    Config {
        output_dir: output_dir.to_path_buf(),
        providers: ProvidersConfig {
            mistral: Some(MistralConfig {
                api_key: "test-api-key".to_string(),
                url: Some(url.to_string()),
                model: "voxtral-mini-2507".to_string(),
                context_bias: None,
                tts_model: "voxtral-mini-tts-latest".to_string(),
                tts_voice: Some("test-voice-id".to_string()),
                tts_voices: None,
            }),
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
    }
}

#[tokio::test]
async fn speak_mistral_synthesizes_pcm_from_mocked_api() {
    let mock_server = MockServer::start().await;

    // The synthetic audio the "API" returns.
    let samples: Vec<i16> = vec![0, 1000, -1000, 32767, -32768, 250, -250];
    let wav = make_wav(&samples, 24_000);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);

    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(header("authorization", "Bearer test-api-key"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "audio_data": b64 })),
        )
        .mount(&mock_server)
        .await;

    let tmp = tempfile::TempDir::new().expect("tmp dir");
    let config = config_with_mistral_url(&mock_server.uri(), tmp.path());

    let result = synthesize(
        &config,
        SynthesisProvider::Mistral,
        SynthesisRequest {
            text: "hello world".to_string(),
            voice: None, // falls back to config tts_voice
            speed: None,
            lang: None,
        },
    )
    .await
    .expect("synthesis must succeed");

    assert_eq!(result.sample_rate, 24_000);
    assert_eq!(result.pcm, samples, "decoded PCM must match the mocked WAV");
    assert!(result.duration_secs() > 0.0);
}

#[tokio::test]
async fn speak_mistral_saved_wav_is_valid() {
    let mock_server = MockServer::start().await;

    let samples: Vec<i16> = (0..480)
        .map(|i| ((i as f32 * 0.05).sin() * 12000.0) as i16)
        .collect();
    let wav = make_wav(&samples, 24_000);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);

    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "audio_data": b64 })),
        )
        .mount(&mock_server)
        .await;

    let tmp = tempfile::TempDir::new().expect("tmp dir");
    let config = config_with_mistral_url(&mock_server.uri(), tmp.path());

    let result = synthesize(
        &config,
        SynthesisProvider::Mistral,
        SynthesisRequest {
            text: "save me to a file".to_string(),
            voice: Some("explicit-voice".to_string()),
            speed: None,
            lang: None,
        },
    )
    .await
    .expect("synthesis must succeed");

    // Persist via the shared WAV writer, then re-parse to prove the
    // produced file is a valid WAV carrying the right PCM.
    let out = tmp.path().join("out.wav");
    write_wav(&out, &result.pcm, result.sample_rate);

    let bytes = std::fs::read(&out).expect("read out.wav");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let parsed = wav_data_pcm(&bytes);
    assert_eq!(parsed, result.pcm);
}

#[tokio::test]
async fn speak_mistral_api_error_surfaces() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
        .mount(&mock_server)
        .await;

    let tmp = tempfile::TempDir::new().expect("tmp dir");
    let config = config_with_mistral_url(&mock_server.uri(), tmp.path());

    let err = synthesize(
        &config,
        SynthesisProvider::Mistral,
        SynthesisRequest {
            text: "boom".to_string(),
            voice: Some("v".to_string()),
            speed: None,
            lang: None,
        },
    )
    .await
    .expect_err("401 must surface as an error");
    assert!(err.to_string().contains("401"), "got: {}", err);
}

#[tokio::test]
async fn speak_mistral_request_carries_resolved_voice_id() {
    // Prove the resolved voice reaches the wire: the mock ONLY matches a
    // request whose JSON body carries the expected `voice_id`; any other
    // body 404s, failing synthesis.
    use wiremock::matchers::body_json;

    let mock_server = MockServer::start().await;
    let samples: Vec<i16> = vec![0, 5, -5];
    let wav = make_wav(&samples, 24_000);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);

    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(body_json(serde_json::json!({
            "model": "voxtral-mini-tts-latest",
            "input": "route me",
            "voice_id": "resolved-voice-xyz",
            "response_format": "wav",
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "audio_data": b64 })),
        )
        .mount(&mock_server)
        .await;

    let tmp = tempfile::TempDir::new().expect("tmp dir");
    let config = config_with_mistral_url(&mock_server.uri(), tmp.path());

    // A fully-resolved request (as `speak.rs` would build after the
    // shared resolution chain) carries the concrete voice id.
    let result = synthesize(
        &config,
        SynthesisProvider::Mistral,
        SynthesisRequest {
            text: "route me".to_string(),
            voice: Some("resolved-voice-xyz".to_string()),
            speed: None,
            lang: Some("en".to_string()),
        },
    )
    .await
    .expect("synthesis with matching voice_id must succeed");
    assert_eq!(result.pcm, samples);
}

/// Write mono i16 PCM to a WAV file using the public `WavWriter`.
fn write_wav(path: &std::path::Path, pcm: &[i16], sample_rate: u32) {
    use std::io::Write;
    use talk_rs::audio::{AudioWriter, WavWriter};
    use talk_rs::config::AudioConfig;

    let cfg = AudioConfig {
        sample_rate,
        channels: 1,
        bitrate: 0,
    };
    let mut writer = WavWriter::new(cfg);
    let mut out = writer.header().expect("header");
    out.extend_from_slice(&writer.write_pcm(pcm).expect("write pcm"));
    let final_header = writer.finalize().expect("finalize");
    out[..final_header.len()].copy_from_slice(&final_header);

    let mut file = std::fs::File::create(path).expect("create wav");
    file.write_all(&out).expect("write wav");
    file.sync_all().expect("sync wav");
}
