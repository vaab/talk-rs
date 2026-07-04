//! Mistral / Voxtral remote text-to-speech backend.
//!
//! The synthesis sibling of [`crate::transcription::mistral`].  Where
//! that module POSTs audio to `/v1/audio/transcriptions` and parses
//! text out, this one POSTs text to `/v1/audio/speech` and parses
//! audio out.
//!
//! # Wire contract (verified)
//!
//! ```text
//! POST {base}/v1/audio/speech
//! Authorization: Bearer <api_key>
//! Content-Type: application/json
//! { "model": "voxtral-mini-tts-latest",
//!   "input": "<text>",
//!   "voice_id": "<uuid>",
//!   "response_format": "wav" }
//!
//! 200 OK
//! { "audio_data": "<base64 WAV>" }
//! ```
//!
//! The returned WAV is 24 kHz mono 16-bit PCM.  We parse the WAV
//! header *properly* (locate the `data` chunk rather than assuming a
//! fixed 44-byte header) and extract the PCM samples into a
//! [`SynthesisResult`].

use base64::Engine;
use serde::Deserialize;
use std::time::Duration;

use crate::config::MistralConfig;
use crate::error::TalkError;

use super::{OneShotSynthesizer, SynthesisRequest, SynthesisResult};
use async_trait::async_trait;

/// Default API base URL for the Mistral API.
pub(crate) const API_BASE: &str = "https://api.mistral.ai";

/// Connect timeout for the speech request.  Mirrors the transport
/// layer's connect-timeout defence so a dead host fails fast.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Overall request timeout.  TTS of a short utterance is quick; a
/// generous cap keeps an unattended `speak` from hanging forever.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// JSON body sent to `/v1/audio/speech`.
#[derive(Debug, serde::Serialize)]
struct SpeechRequestBody<'a> {
    model: &'a str,
    input: &'a str,
    voice_id: &'a str,
    response_format: &'a str,
}

/// JSON response from `/v1/audio/speech`.
#[derive(Debug, Deserialize)]
struct SpeechResponseBody {
    /// Base64-encoded WAV audio.
    audio_data: String,
}

/// Remote Mistral / Voxtral one-shot synthesizer.
pub struct MistralOneShotSynthesizer {
    config: MistralConfig,
    /// Resolved `POST` endpoint (`{base}/v1/audio/speech`).
    endpoint: String,
}

impl MistralOneShotSynthesizer {
    /// Build a synthesizer from a parsed [`MistralConfig`].
    ///
    /// The endpoint is derived from `config.url` (if set) by appending
    /// `/v1/audio/speech`; otherwise the default base URL is used.
    pub fn new(config: MistralConfig) -> Result<Self, TalkError> {
        let base = config.url.as_deref().unwrap_or(API_BASE);
        let endpoint = format!("{}/v1/audio/speech", base.trim_end_matches('/'));
        Ok(Self { config, endpoint })
    }

    /// Build a synthesizer with an explicit endpoint (for testing).
    #[cfg(test)]
    pub fn with_endpoint(config: MistralConfig, endpoint: String) -> Result<Self, TalkError> {
        Ok(Self { config, endpoint })
    }

    /// Resolve the voice id to use: `--voice` > config `tts_voice` >
    /// error.
    fn resolve_voice(&self, req_voice: &Option<String>) -> Result<String, TalkError> {
        if let Some(v) = req_voice {
            return Ok(v.clone());
        }
        if let Some(v) = &self.config.tts_voice {
            return Ok(v.clone());
        }
        Err(TalkError::Config(
            "no voice selected for Mistral TTS: pass --voice <id> or set \
             providers.mistral.tts_voice. List available voices with \
             `GET /v1/audio/voices?voice_type=preset`."
                .to_string(),
        ))
    }
}

#[async_trait]
impl OneShotSynthesizer for MistralOneShotSynthesizer {
    /// Cheap pre-flight: only checks that an API key is present.  Does
    /// NOT call the API (per the plan — no network in `validate`).
    async fn validate(&self) -> Result<(), TalkError> {
        if self.config.api_key.is_empty() {
            return Err(TalkError::Config(
                "providers.mistral.api_key is required for Mistral TTS".to_string(),
            ));
        }
        Ok(())
    }

    async fn synthesize(&self, req: SynthesisRequest) -> Result<SynthesisResult, TalkError> {
        let voice = self.resolve_voice(&req.voice)?;

        let body = SpeechRequestBody {
            model: &self.config.tts_model,
            input: &req.text,
            voice_id: &voice,
            response_format: "wav",
        };

        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| TalkError::Transcription(format!("failed to build HTTP client: {}", e)))?;

        let response = client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| TalkError::Transcription(format!("Mistral TTS request failed: {}", e)))?;

        let status = response.status();
        let bytes = response.bytes().await.map_err(|e| {
            TalkError::Transcription(format!("Mistral TTS read body failed: {}", e))
        })?;

        if !status.is_success() {
            let body = String::from_utf8_lossy(&bytes);
            return Err(TalkError::Transcription(format!(
                "Mistral TTS API error ({}): {}",
                status.as_u16(),
                body
            )));
        }

        let parsed: SpeechResponseBody = serde_json::from_slice(&bytes).map_err(|e| {
            TalkError::Transcription(format!("failed to parse Mistral TTS response: {}", e))
        })?;

        let wav_bytes = base64::engine::general_purpose::STANDARD
            .decode(parsed.audio_data.trim())
            .map_err(|e| {
                TalkError::Transcription(format!(
                    "failed to base64-decode Mistral TTS audio: {}",
                    e
                ))
            })?;

        let (pcm, sample_rate) = parse_wav_pcm_i16(&wav_bytes)?;
        Ok(SynthesisResult { pcm, sample_rate })
    }
}

/// Parse a 16-bit PCM WAV byte stream into `(samples, sample_rate)`.
///
/// Robustly locates the `fmt ` and `data` chunks by walking the RIFF
/// chunk list — it does NOT assume the canonical 44-byte header, since
/// some encoders insert extra chunks (`LIST`, `fact`, …) before
/// `data`.  Returns downmixed-to-mono `i16` samples: multi-channel WAV
/// is averaged to mono so the result meets the codebase's mono PCM
/// convention.
///
/// # Errors
///
/// Returns a [`TalkError::Transcription`] on a truncated header, a
/// non-`WAVE` RIFF, a missing `fmt `/`data` chunk, or an unsupported
/// (non-16-bit-PCM) format.
pub(crate) fn parse_wav_pcm_i16(bytes: &[u8]) -> Result<(Vec<i16>, u32), TalkError> {
    let err = |m: &str| TalkError::Transcription(format!("invalid WAV: {}", m));

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(err("missing RIFF/WAVE header"));
    }

    let read_u16 = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let read_u32 = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);

    let mut pos = 12usize;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut fmt_seen = false;
    let mut data: Option<&[u8]> = None;

    // Walk the chunk list: each chunk is `<4-byte id><4-byte size><size bytes>`,
    // padded to an even boundary.
    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = read_u32(bytes, pos + 4) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(chunk_size);
        if body_end > bytes.len() {
            // Truncated final chunk — clamp so we can still surface a
            // `data` chunk that ran to EOF (some encoders under-report).
            if chunk_id == b"data" {
                data = Some(&bytes[body_start..bytes.len()]);
            }
            break;
        }

        if chunk_id == b"fmt " {
            if chunk_size < 16 {
                return Err(err("fmt chunk too small"));
            }
            let audio_format = read_u16(bytes, body_start);
            channels = read_u16(bytes, body_start + 2);
            sample_rate = read_u32(bytes, body_start + 4);
            bits_per_sample = read_u16(bytes, body_start + 14);
            // 1 = PCM; 0xFFFE = WAVE_FORMAT_EXTENSIBLE (still PCM for our use).
            if audio_format != 1 && audio_format != 0xFFFE {
                return Err(err(&format!(
                    "unsupported WAV audio format {} (only 16-bit PCM supported)",
                    audio_format
                )));
            }
            fmt_seen = true;
        } else if chunk_id == b"data" {
            data = Some(&bytes[body_start..body_end]);
        }

        // Advance, honouring the RIFF even-padding rule.
        pos = body_end + (chunk_size & 1);
    }

    if !fmt_seen {
        return Err(err("missing fmt chunk"));
    }
    if bits_per_sample != 16 {
        return Err(err(&format!(
            "unsupported bits_per_sample {} (only 16-bit PCM supported)",
            bits_per_sample
        )));
    }
    if channels == 0 {
        return Err(err("fmt chunk reports zero channels"));
    }
    if sample_rate == 0 {
        return Err(err("fmt chunk reports zero sample rate"));
    }
    let data = data.ok_or_else(|| err("missing data chunk"))?;

    // Interpret little-endian i16 samples.
    let mut interleaved: Vec<i16> = Vec::with_capacity(data.len() / 2);
    for frame in data.chunks_exact(2) {
        interleaved.push(i16::from_le_bytes([frame[0], frame[1]]));
    }

    // Downmix to mono when needed by averaging channels per frame.
    let pcm = if channels == 1 {
        interleaved
    } else {
        let ch = channels as usize;
        interleaved
            .chunks_exact(ch)
            .map(|frame| {
                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                (sum / ch as i32) as i16
            })
            .collect()
    };

    Ok((pcm, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal canonical 44-byte-header mono 16-bit WAV around
    /// `samples` at `sample_rate`.
    fn make_wav(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
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

    #[test]
    fn parse_canonical_mono_wav() {
        let samples = vec![0i16, 100, -100, 32767, -32768];
        let wav = make_wav(&samples, 24_000, 1);
        let (pcm, rate) = parse_wav_pcm_i16(&wav).expect("parse");
        assert_eq!(rate, 24_000);
        assert_eq!(pcm, samples);
    }

    #[test]
    fn parse_wav_with_extra_chunk_before_data() {
        // Insert a LIST chunk between fmt and data — the parser must
        // skip it and still find data (proving we don't assume a
        // fixed 44-byte header).
        let samples = vec![1i16, 2, 3, 4];
        let mut wav = make_wav(&samples, 24_000, 1);
        // Rebuild with an injected LIST chunk after the fmt chunk (the
        // fmt chunk ends at offset 36 in the canonical layout).
        let data_pos = wav
            .windows(4)
            .position(|w| w == b"data")
            .expect("data present");
        let list_chunk: &[u8] = b"LIST\x04\x00\x00\x00INFO";
        wav.splice(data_pos..data_pos, list_chunk.iter().copied());
        // RIFF size in header is now stale but the parser walks chunks,
        // not the RIFF size, so it still works.
        let (pcm, rate) = parse_wav_pcm_i16(&wav).expect("parse with extra chunk");
        assert_eq!(rate, 24_000);
        assert_eq!(pcm, samples);
    }

    #[test]
    fn parse_stereo_wav_downmixes_to_mono() {
        // Two channels: L=100, R=200 → mono avg 150.
        let interleaved = vec![100i16, 200, -100, -200];
        let wav = make_wav(&interleaved, 24_000, 2);
        let (pcm, _rate) = parse_wav_pcm_i16(&wav).expect("parse stereo");
        assert_eq!(pcm, vec![150i16, -150]);
    }

    #[test]
    fn parse_rejects_non_wav() {
        let err = parse_wav_pcm_i16(b"not a wav at all").expect_err("must reject");
        assert!(err.to_string().contains("RIFF/WAVE"));
    }

    #[test]
    fn parse_rejects_missing_data_chunk() {
        // fmt-only WAV, no data chunk.
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&28u32.to_le_bytes());
        v.extend_from_slice(b"WAVE");
        v.extend_from_slice(b"fmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&24_000u32.to_le_bytes());
        v.extend_from_slice(&48_000u32.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        let err = parse_wav_pcm_i16(&v).expect_err("must reject");
        assert!(err.to_string().contains("data chunk"));
    }

    #[test]
    fn endpoint_derivation() {
        let cfg = MistralConfig {
            api_key: "k".to_string(),
            url: None,
            model: "voxtral-mini-2507".to_string(),
            context_bias: None,
            tts_model: "voxtral-mini-tts-latest".to_string(),
            tts_voice: None,
            tts_voices: None,
        };
        let s = MistralOneShotSynthesizer::new(cfg).unwrap();
        assert_eq!(s.endpoint, "https://api.mistral.ai/v1/audio/speech");
    }

    #[test]
    fn endpoint_derivation_custom_url_trailing_slash() {
        let cfg = MistralConfig {
            api_key: "k".to_string(),
            url: Some("https://custom.example.com/".to_string()),
            model: "voxtral-mini-2507".to_string(),
            context_bias: None,
            tts_model: "voxtral-mini-tts-latest".to_string(),
            tts_voice: None,
            tts_voices: None,
        };
        let s = MistralOneShotSynthesizer::new(cfg).unwrap();
        assert_eq!(s.endpoint, "https://custom.example.com/v1/audio/speech");
    }

    #[test]
    fn resolve_voice_prefers_request_over_config() {
        let cfg = MistralConfig {
            api_key: "k".to_string(),
            url: None,
            model: "m".to_string(),
            context_bias: None,
            tts_model: "voxtral-mini-tts-latest".to_string(),
            tts_voice: Some("config-voice".to_string()),
            tts_voices: None,
        };
        let s = MistralOneShotSynthesizer::new(cfg).unwrap();
        assert_eq!(
            s.resolve_voice(&Some("cli-voice".to_string())).unwrap(),
            "cli-voice"
        );
        assert_eq!(s.resolve_voice(&None).unwrap(), "config-voice");
    }

    #[test]
    fn resolve_voice_errors_when_none_available() {
        let cfg = MistralConfig {
            api_key: "k".to_string(),
            url: None,
            model: "m".to_string(),
            context_bias: None,
            tts_model: "voxtral-mini-tts-latest".to_string(),
            tts_voice: None,
            tts_voices: None,
        };
        let s = MistralOneShotSynthesizer::new(cfg).unwrap();
        let err = s.resolve_voice(&None).expect_err("must error");
        assert!(err.to_string().contains("no voice selected"));
    }
}
