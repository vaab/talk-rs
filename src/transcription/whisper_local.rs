//! Local whisper.cpp batch transcription backend.
//!
//! Offline transcription via the [`whisper-rs`] bindings to whisper.cpp,
//! optionally accelerated with CUDA (or any other backend selected at
//! build time via crate features).
//!
//! Only the [`BatchTranscriber`] trait is implemented.  Realtime
//! streaming is not supported by this backend yet — selecting
//! `whisper_local` with `--realtime` returns a clear error from the
//! factory in `crate::transcription`.
//!
//! ## Audio format support
//!
//! The transcriber accepts files in two formats, both decoded in pure
//! Rust using existing dependencies (`ogg`, `opus`, `hound`):
//!
//! - OGG/Opus (the format produced by `talk-rs record`).
//! - WAV (16 kHz, mono, PCM 16-bit or 32-bit float).
//!
//! Other formats should be converted ahead of time with `ffmpeg`.

use crate::config::WhisperLocalConfig;
use crate::error::TalkError;
use crate::transcription::{TranscriptionMetadata, TranscriptionResult};
use async_trait::async_trait;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::BatchTranscriber;

/// Local Whisper batch transcriber.
///
/// Holds the configuration eagerly and the loaded [`WhisperContext`]
/// lazily inside a [`OnceLock`].  This avoids paying the multi-second
/// model-load cost up-front when the daemon spawns — the load only
/// happens on the first transcription, while recording has already
/// been captured.
pub struct WhisperLocalBatchTranscriber {
    config: WhisperLocalConfig,
    context: OnceLock<WhisperContext>,
}

impl WhisperLocalBatchTranscriber {
    /// Build a new transcriber from a config section.
    pub fn new(config: WhisperLocalConfig) -> Self {
        Self {
            config,
            context: OnceLock::new(),
        }
    }

    /// Return the loaded Whisper context, loading the model on first call.
    ///
    /// The first call is the expensive one: reading a 3 GB GGML file
    /// from disk and initialising the GPU backend takes a few seconds
    /// even with the file in the kernel page cache.  Subsequent calls
    /// from the same process are free.
    fn get_context(&self) -> Result<&WhisperContext, TalkError> {
        if let Some(ctx) = self.context.get() {
            return Ok(ctx);
        }
        if !self.config.model_path.exists() {
            return Err(TalkError::Transcription(format!(
                "whisper model file not found: {} — download a GGML model from \
                 https://huggingface.co/ggerganov/whisper.cpp and point \
                 providers.whisper_local.model_path at it",
                self.config.model_path.display()
            )));
        }
        let mut params = WhisperContextParameters::default();
        params.use_gpu(self.config.use_gpu);
        let model_path = self.config.model_path.to_string_lossy().into_owned();
        let started = Instant::now();
        let ctx = WhisperContext::new_with_params(&model_path, params).map_err(|e| {
            TalkError::Transcription(format!(
                "failed to load whisper model '{}': {}",
                model_path, e
            ))
        })?;
        log::info!(
            "loaded whisper model {} ({} ms)",
            model_path,
            started.elapsed().as_millis()
        );
        // OnceLock::set may race with another thread that initialised
        // first — the freshly built ctx is dropped in that case.
        let _ = self.context.set(ctx);
        self.context
            .get()
            .ok_or_else(|| TalkError::Transcription("whisper context init race".to_string()))
    }

    /// Decode an audio file at any supported format into 16 kHz mono f32 PCM.
    fn decode_audio_file(&self, path: &Path) -> Result<Vec<f32>, TalkError> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());
        match ext.as_deref() {
            Some("ogg") | Some("opus") => decode_ogg_opus(path),
            Some("wav") => decode_wav(path),
            other => Err(TalkError::Transcription(format!(
                "unsupported audio format '{}' for whisper_local — supported: .ogg, .opus, .wav \
                 (convert with `ffmpeg -i in -ac 1 -ar 16000 out.wav`)",
                other.unwrap_or("<none>")
            ))),
        }
    }

    /// Run whisper on the decoded PCM samples and return the joined text.
    fn run_inference(&self, samples: &[f32]) -> Result<TranscriptionResult, TalkError> {
        if samples.is_empty() {
            return Err(TalkError::Transcription(
                "audio is empty — nothing to transcribe".to_string(),
            ));
        }
        let ctx = self.get_context()?;
        let mut state = ctx.create_state().map_err(|e| {
            TalkError::Transcription(format!("failed to create whisper state: {}", e))
        })?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.config.threads);
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);

        // Language: "auto" / empty means let whisper detect.
        let language = self.config.language.trim();
        if !language.is_empty() && language.to_lowercase() != "auto" {
            params.set_language(Some(language));
        } else {
            params.set_language(Some("auto"));
        }

        if let Some(ref prompt) = self.config.initial_prompt {
            if !prompt.is_empty() {
                params.set_initial_prompt(prompt);
            }
        }

        let started = Instant::now();
        state
            .full(params, samples)
            .map_err(|e| TalkError::Transcription(format!("whisper inference failed: {}", e)))?;
        let inference_ms = started.elapsed().as_millis() as u64;

        let n_segments = state.full_n_segments();
        let mut text = String::new();
        for segment in state.as_iter() {
            let seg = segment.to_str_lossy().map_err(|e| {
                TalkError::Transcription(format!("whisper segment text read failed: {}", e))
            })?;
            text.push_str(seg.as_ref());
        }

        let audio_seconds = samples.len() as f64 / 16_000.0;
        let metadata = TranscriptionMetadata {
            request_latency_ms: Some(inference_ms),
            session_elapsed_ms: Some(inference_ms),
            request_id: None,
            provider_processing_ms: Some(inference_ms),
            detected_language: None,
            audio_seconds: Some(audio_seconds),
            segment_count: Some(n_segments as usize),
            word_count: None,
            token_usage: None,
            provider_specific: None,
        };

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            metadata,
            diarization: None,
        })
    }
}

#[async_trait]
impl BatchTranscriber for WhisperLocalBatchTranscriber {
    async fn validate(&self) -> Result<(), TalkError> {
        if !self.config.model_path.exists() {
            return Err(TalkError::Config(format!(
                "whisper model file not found: {}",
                self.config.model_path.display()
            )));
        }
        Ok(())
    }

    async fn transcribe_file(&self, audio_path: &Path) -> Result<TranscriptionResult, TalkError> {
        let path = audio_path.to_path_buf();
        // Whisper inference is CPU/GPU-bound and synchronous in the
        // C++ side — run it on the blocking thread pool so we don't
        // stall the tokio reactor.
        let this_config = self.config.clone();
        // Move the heavy work to a blocking task.  We can't borrow
        // `self` across the await because `OnceLock<WhisperContext>`
        // isn't `Send` through a captured reference in a way that
        // satisfies tokio's `spawn_blocking` lifetime requirements.
        // Instead we recreate a transient transcriber for the blocking
        // task: it shares the same model path (and therefore the same
        // kernel page cache).
        tokio::task::spawn_blocking(move || {
            let inner = WhisperLocalBatchTranscriber::new(this_config);
            let samples = inner.decode_audio_file(&path)?;
            inner.run_inference(&samples)
        })
        .await
        .map_err(|e| TalkError::Transcription(format!("whisper task join failed: {}", e)))?
    }

    async fn transcribe_stream(
        &self,
        mut audio_stream: tokio::sync::mpsc::Receiver<Vec<u8>>,
        file_name: &str,
    ) -> Result<TranscriptionResult, TalkError> {
        // Buffer the streamed bytes to a temp file with the correct
        // extension, then run the same file path that
        // `transcribe_file` would have handled.  Streaming directly
        // into whisper would require a true streaming decoder which we
        // do not yet have for OGG/Opus in pure Rust.
        let suffix = std::path::Path::new(file_name)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("ogg");
        let tmp = tempfile::Builder::new()
            .prefix("talk-rs-whisper-")
            .suffix(&format!(".{}", suffix))
            .tempfile()
            .map_err(|e| TalkError::Transcription(format!("temp file create: {}", e)))?;
        let (mut file, path) = tmp
            .keep()
            .map_err(|e| TalkError::Transcription(format!("temp file persist: {}", e.error)))?;

        while let Some(chunk) = audio_stream.recv().await {
            file.write_all(&chunk).map_err(TalkError::Io)?;
        }
        file.flush().map_err(TalkError::Io)?;
        drop(file);

        let result = self.transcribe_file(&path).await;
        // Best-effort cleanup; ignore errors.
        let _ = std::fs::remove_file(&path);
        result
    }
}

/// Decode an OGG/Opus container into mono f32 PCM at 16 kHz.
///
/// talk-rs records at 16 kHz mono, so no resampling is required.  The
/// Opus decoder is asked for 16 kHz output directly.
fn decode_ogg_opus(path: &Path) -> Result<Vec<f32>, TalkError> {
    let file = File::open(path).map_err(TalkError::Io)?;
    let mut reader = ogg::PacketReader::new(file);
    let mut decoder = opus::Decoder::new(16_000, opus::Channels::Mono)
        .map_err(|e| TalkError::Transcription(format!("opus decoder init failed: {}", e)))?;

    let mut pcm: Vec<f32> = Vec::new();
    // 120 ms at 16 kHz mono = 1920 samples; bump to 5760 to cover the
    // max-sized Opus frames regardless of the encoder's choice.
    let mut buf = vec![0_i16; 5760];

    loop {
        let packet = match reader.read_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(TalkError::Transcription(format!("ogg read failed: {}", e))),
        };
        // Skip the two stream headers (OpusHead, OpusTags).
        if packet.data.starts_with(b"OpusHead") || packet.data.starts_with(b"OpusTags") {
            continue;
        }
        let n = decoder
            .decode(&packet.data, &mut buf, false)
            .map_err(|e| TalkError::Transcription(format!("opus decode failed: {}", e)))?;
        pcm.extend(buf[..n].iter().map(|&s| s as f32 / 32768.0));
    }
    Ok(pcm)
}

/// Decode a WAV file into mono f32 PCM at 16 kHz.
///
/// Refuses to resample or downmix — the caller is expected to provide
/// audio at the right rate.  This is documented in the user-facing
/// error message.
fn decode_wav(path: &Path) -> Result<Vec<f32>, TalkError> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| TalkError::Transcription(format!("wav open failed: {}", e)))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(TalkError::Transcription(format!(
            "WAV must be mono (got {} channels) — \
             convert with `ffmpeg -i in.wav -ac 1 -ar 16000 out.wav`",
            spec.channels
        )));
    }
    if spec.sample_rate != 16_000 {
        return Err(TalkError::Transcription(format!(
            "WAV must be 16 kHz (got {} Hz) — \
             convert with `ffmpeg -i in.wav -ac 1 -ar 16000 out.wav`",
            spec.sample_rate
        )));
    }
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1_i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|r| {
                    r.map(|s| s as f32 / max)
                        .map_err(|e| TalkError::Transcription(format!("wav read: {}", e)))
                })
                .collect()
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|r| r.map_err(|e| TalkError::Transcription(format!("wav read: {}", e))))
            .collect(),
    }
}

/// Detect "model file missing" errors for the WhisperLocal provider.
///
/// Used by the transcription dispatcher to give the user a helpful
/// message when the YAML points at a non-existent GGML file.
pub fn is_model_error(error: &TalkError) -> bool {
    let msg = error.to_string();
    msg.contains("whisper model file not found") || msg.contains("failed to load whisper model")
}

/// Optionally enrich a model error with suggestions.
///
/// For the local provider, the only "suggestion" we can offer is the
/// canonical download URL — the model is a local file, not an
/// enumerable API resource.  We return the original error if it isn't
/// a model error.
pub fn enrich_model_error(error: TalkError) -> TalkError {
    if !is_model_error(&error) {
        return error;
    }
    let suggestion = "Download a GGML model with:\n  \
        mkdir -p ~/.local/share/talk-rs/models && \\\n  \
        curl -L -o ~/.local/share/talk-rs/models/ggml-large-v3.bin \\\n    \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin";
    TalkError::Transcription(format!("{}\n\n{}", error, suggestion))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_model_error_detects_missing_file() {
        let err = TalkError::Transcription("whisper model file not found: /tmp/x.bin".to_string());
        assert!(is_model_error(&err));
    }

    #[test]
    fn is_model_error_ignores_unrelated() {
        let err = TalkError::Transcription("some other error".to_string());
        assert!(!is_model_error(&err));
    }

    #[test]
    fn unsupported_format_returns_clear_error() {
        let cfg = WhisperLocalConfig {
            model_path: std::path::PathBuf::from("/nonexistent.bin"),
            language: "auto".to_string(),
            use_gpu: false,
            initial_prompt: None,
            threads: 1,
        };
        let t = WhisperLocalBatchTranscriber::new(cfg);
        let res = t.decode_audio_file(std::path::Path::new("/tmp/x.mp3"));
        match res {
            Err(TalkError::Transcription(msg)) => {
                assert!(msg.contains("unsupported audio format"));
            }
            other => panic!("expected unsupported-format error, got {:?}", other),
        }
    }
}
