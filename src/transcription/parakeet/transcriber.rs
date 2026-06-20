//! Phase 3: `BatchTranscriber` implementation for local Parakeet
//! inference via sherpa-onnx.
//!
//! See the module-level docs on [`crate::transcription::parakeet`]
//! for the design rationale (one recognizer per call, blocking work
//! offloaded to `spawn_blocking`).

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};

use crate::config::{ParakeetConfig, ParakeetVariant};
use crate::error::TalkError;
use crate::transcription::{
    BatchTranscriber, RequestTimeoutPolicy, TranscriptionBody, TranscriptionResult,
};

use super::model;

/// On-disk filenames for an INT8 model dir.
const FILENAMES_INT8: [&str; 4] = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
];

/// On-disk filenames for an FP32 (unquantized) model dir.
const FILENAMES_FP32: [&str; 4] = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"];

/// Resolve the 4 expected filenames for `variant`.
///
/// Mirrors the variant-keyed naming convention used by
/// [`super::model`] but lives here so the inference path can read
/// it without depending on `model`'s private layout helpers.
fn filenames_for(variant: ParakeetVariant) -> [&'static str; 4] {
    match variant {
        ParakeetVariant::Int8 => FILENAMES_INT8,
        ParakeetVariant::Fp32 => FILENAMES_FP32,
    }
}

/// Sample rate expected by Parakeet (NeMo TDT).  Audio is always
/// resampled / decoded to this rate by [`read_audio_as_i16`].
const PARAKEET_SAMPLE_RATE_HZ: i32 = 16_000;

/// Convert 16-bit PCM (signed) into normalised f32 in `[-1.0, 1.0)`.
///
/// Factored out so the conversion can be tested independently; the
/// boundary values (`i16::MIN` → `-1.0`, `i16::MAX` → just under
/// `+1.0`) are the easy ones to get wrong.
fn pcm_i16_to_f32_normalised(samples: &[i16]) -> Vec<f32> {
    // Divisor 32768.0 is the standard normalisation: it makes
    // `i16::MIN` map to exactly `-1.0` and keeps `i16::MAX` within
    // `[-1, 1)` (32767/32768).
    samples.iter().map(|&s| s as f32 / 32768.0).collect()
}

/// Local Parakeet TDT (ONNX, CPU) batch transcriber.
///
/// Construction is cheap (path resolution only); the heavy
/// `OfflineRecognizer::create` happens lazily inside
/// [`Self::fetch_transcription`].
pub struct ParakeetBatchTranscriber {
    model_dir: PathBuf,
    variant: ParakeetVariant,
    num_threads: i32,
}

impl ParakeetBatchTranscriber {
    /// Build a transcriber from a parsed [`ParakeetConfig`].
    ///
    /// The `policy` argument exists for symmetry with the remote
    /// providers' `with_policy` constructors — local inference has
    /// no per-request HTTP wall clock to cap, so it is accepted and
    /// ignored.  Future work could wire it into a soft inference
    /// deadline; out of scope for v1.
    pub fn with_policy(
        cfg: ParakeetConfig,
        _policy: RequestTimeoutPolicy,
    ) -> Result<Self, TalkError> {
        let model_dir = cfg.resolved_model_dir()?;
        let variant = cfg.resolved_variant();
        Ok(Self {
            model_dir,
            variant,
            num_threads: cfg.num_threads,
        })
    }

    /// Return the absolute paths of the 4 model files this
    /// transcriber will load at inference time.
    fn model_file_paths(&self) -> [PathBuf; 4] {
        let names = filenames_for(self.variant);
        [
            self.model_dir.join(names[0]),
            self.model_dir.join(names[1]),
            self.model_dir.join(names[2]),
            self.model_dir.join(names[3]),
        ]
    }
}

#[async_trait]
impl BatchTranscriber for ParakeetBatchTranscriber {
    /// Pre-flight: check the model is present on disk — **without
    /// downloading**.
    ///
    /// This is deliberately consent-safe: it performs NO network I/O.
    /// If the model is absent it returns an error (via
    /// [`model::ensure_present`]) rather than silently fetching
    /// ~640 MB.  Downloading the model is an explicit, user-consented
    /// action driven by each entry surface (the `transcribe` CLI,
    /// `dictate --toggle`, the `--pick` picker), which call
    /// [`model::download_model`] only after obtaining consent.  By the
    /// time `validate` runs in the transcribe pipeline the model is
    /// therefore already present on every supported path.
    async fn validate(&self) -> Result<(), TalkError> {
        model::ensure_present(&self.model_dir, self.variant)
    }

    async fn fetch_transcription(
        &self,
        body: TranscriptionBody,
    ) -> Result<TranscriptionResult, TalkError> {
        // 1. Obtain a 16 kHz mono i16 PCM buffer from either an
        //    on-disk file or a streaming channel of OGG bytes.
        let pcm_i16 = match body {
            TranscriptionBody::File(path) => {
                // `read_audio_as_i16` handles ogg / wav / m4a / mp4 /
                // aac and emits exactly 16 kHz mono — Parakeet's
                // native format.  No further resampling needed.
                crate::record::audio::read_audio_as_i16(&path)?
            }
            TranscriptionBody::Stream {
                mut chunks,
                file_name,
            } => {
                // Collect the encoded byte chunks (typically OGG)
                // and decode through the same `read_audio_as_i16`
                // path.  We materialise to a temp file rather than
                // adding an in-memory decode shim because the
                // existing decode helper is path-based and OGG /
                // M4A demuxing prefers seekable I/O.  The temp file
                // is best-effort removed afterwards.
                let mut bytes = Vec::new();
                while let Some(chunk) = chunks.recv().await {
                    bytes.extend_from_slice(&chunk);
                }
                log::info!(
                    "parakeet stream: collected {} bytes ({}) -> decoding via temp file",
                    bytes.len(),
                    file_name
                );
                write_temp_and_decode(&bytes, &file_name)?
            }
        };

        // 2. Normalise PCM to f32 in [-1, 1] for sherpa-onnx.
        let samples_f32 = pcm_i16_to_f32_normalised(&pcm_i16);

        // 3. Snapshot the small bits of `self` we need inside the
        //    blocking task — sherpa C calls are synchronous and
        //    must not run on the async executor.  `OfflineRecognizer`
        //    is `Send + Sync` (Phase 0 verified) so building it
        //    inside `spawn_blocking` is sound.
        let paths = self.model_file_paths();
        let num_threads = self.num_threads;
        let variant = self.variant;

        let text = tokio::task::spawn_blocking(move || -> Result<String, TalkError> {
            run_inference(&paths, num_threads, variant, &samples_f32)
        })
        .await
        .map_err(|e| {
            TalkError::Transcription(format!(
                "parakeet inference task panicked or was cancelled: {}",
                e
            ))
        })??;

        Ok(TranscriptionResult {
            text,
            ..Default::default()
        })
    }

    // `set_sink` and `set_cancel_token` use the default no-op
    // implementations from the trait.  Local inference v1 does not
    // emit telemetry events and is not cancellable mid-decode (the
    // sherpa C decode call is uninterruptible); wiring real
    // cancellation is future work.
}

/// Materialise streamed bytes into a temp file under
/// [`std::env::temp_dir`] and decode them via
/// [`crate::record::audio::read_audio_as_i16`].
///
/// `file_name` is honoured for its **extension** only (the decoder
/// dispatches by extension); a uniquely-suffixed basename keeps
/// concurrent dictations from colliding.  The file is removed on a
/// best-effort basis after decoding.
fn write_temp_and_decode(bytes: &[u8], file_name: &str) -> Result<Vec<i16>, TalkError> {
    let extension = Path::new(file_name)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("ogg");

    // Unique basename: pid + nanos.  No external dep, no temp-file
    // crate at runtime (tempfile is dev-only).
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "talk-rs-parakeet-{}-{}.{}",
        std::process::id(),
        nanos,
        extension
    ));

    std::fs::write(&path, bytes).map_err(|e| {
        TalkError::Transcription(format!(
            "parakeet stream: failed to write temp audio {}: {}",
            path.display(),
            e
        ))
    })?;

    let result = crate::record::audio::read_audio_as_i16(&path);

    // Best-effort cleanup regardless of decode outcome.
    if let Err(e) = std::fs::remove_file(&path) {
        log::debug!(
            "parakeet stream: failed to remove temp file {}: {}",
            path.display(),
            e
        );
    }

    result
}

/// Build a recognizer, feed the samples, decode, and return the
/// recognised text.
///
/// Pure / synchronous — meant to be called from inside
/// [`tokio::task::spawn_blocking`].
fn run_inference(
    paths: &[PathBuf; 4],
    num_threads: i32,
    variant: ParakeetVariant,
    samples_f32: &[f32],
) -> Result<String, TalkError> {
    let encoder = path_to_string(&paths[0])?;
    let decoder = path_to_string(&paths[1])?;
    let joiner = path_to_string(&paths[2])?;
    let tokens = path_to_string(&paths[3])?;

    let mut cfg = OfflineRecognizerConfig::default();
    cfg.model_config.transducer = OfflineTransducerModelConfig {
        encoder: Some(encoder),
        decoder: Some(decoder),
        joiner: Some(joiner),
    };
    cfg.model_config.tokens = Some(tokens);
    cfg.model_config.provider = Some("cpu".into());
    cfg.model_config.num_threads = num_threads;
    cfg.model_config.debug = false;
    // REQUIRED for NeMo Parakeet TDT: selects the matching
    // preprocessor (per-mel-bin global mean/var normalisation +
    // log-mel) and the TDT greedy decoder.  Omitting this returns
    // empty text without an error (Phase 0 verified).
    cfg.model_config.model_type = Some("nemo_transducer".into());

    let recognizer = OfflineRecognizer::create(&cfg).ok_or_else(|| {
        TalkError::Transcription(format!(
            "parakeet: failed to create recognizer (variant={}, model_dir contains: {:?})",
            variant,
            paths
                .iter()
                .map(|p| p.file_name().and_then(|n| n.to_str()).unwrap_or("?"))
                .collect::<Vec<_>>()
        ))
    })?;

    let stream = recognizer.create_stream();
    stream.accept_waveform(PARAKEET_SAMPLE_RATE_HZ, samples_f32);
    recognizer.decode(&stream);

    let result = stream.get_result().ok_or_else(|| {
        TalkError::Transcription(
            "parakeet: recognizer returned no result (decode produced null JSON)".to_string(),
        )
    })?;

    Ok(result.text)
}

/// Convert a [`Path`] to an owned `String` for the sherpa-onnx
/// config (which takes `String` paths, not `&Path`).
///
/// Returns a configuration error on non-UTF-8 paths rather than
/// silently lossy-converting them — a Parakeet model dir under a
/// non-UTF-8 path is a user-fixable problem.
fn path_to_string(p: &Path) -> Result<String, TalkError> {
    p.to_str().map(|s| s.to_string()).ok_or_else(|| {
        TalkError::Config(format!(
            "parakeet: model path is not valid UTF-8: {}",
            p.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ParakeetConfig;

    #[test]
    fn pcm_i16_to_f32_boundary_values() {
        let pcm = vec![i16::MIN, -1, 0, 1, i16::MAX];
        let out = pcm_i16_to_f32_normalised(&pcm);
        // i16::MIN / 32768 = -1.0 exactly.
        assert_eq!(out[0], -1.0);
        assert_eq!(out[2], 0.0);
        // i16::MAX / 32768 < 1.0 (32767/32768).
        assert!(out[4] < 1.0);
        assert!(out[4] > 0.999_9);
        // Every sample stays inside [-1, 1].
        for v in &out {
            assert!(*v >= -1.0 && *v <= 1.0, "sample out of range: {}", v);
        }
    }

    #[test]
    fn pcm_i16_to_f32_empty_is_empty() {
        let out = pcm_i16_to_f32_normalised(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn filenames_for_int8_matches_expected() {
        let names = filenames_for(ParakeetVariant::Int8);
        assert_eq!(names[0], "encoder.int8.onnx");
        assert_eq!(names[1], "decoder.int8.onnx");
        assert_eq!(names[2], "joiner.int8.onnx");
        assert_eq!(names[3], "tokens.txt");
    }

    #[test]
    fn filenames_for_fp32_drops_int8_infix() {
        let names = filenames_for(ParakeetVariant::Fp32);
        assert_eq!(names[0], "encoder.onnx");
        assert_eq!(names[1], "decoder.onnx");
        assert_eq!(names[2], "joiner.onnx");
        assert_eq!(names[3], "tokens.txt");
    }

    #[test]
    fn with_policy_succeeds_without_model_files_present() {
        // Construction must NOT load the model — that only happens
        // inside `fetch_transcription`.  An empty model_dir is fine
        // at build time so the failure surfaces at validate/fetch
        // with a clear message, not at construction.
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = ParakeetConfig {
            variant: ParakeetVariant::Int8,
            model_dir: Some(tmp.path().to_path_buf()),
            num_threads: 1,
            model: None,
        };
        let transcriber =
            ParakeetBatchTranscriber::with_policy(cfg, RequestTimeoutPolicy::Proportional)
                .expect("with_policy must not touch the filesystem");
        assert_eq!(transcriber.num_threads, 1);
        assert_eq!(transcriber.variant, ParakeetVariant::Int8);
        assert_eq!(transcriber.model_dir, tmp.path());
    }

    #[test]
    fn model_file_paths_int8_layout() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = ParakeetConfig {
            variant: ParakeetVariant::Int8,
            model_dir: Some(tmp.path().to_path_buf()),
            num_threads: 2,
            model: None,
        };
        let t =
            ParakeetBatchTranscriber::with_policy(cfg, RequestTimeoutPolicy::Proportional).unwrap();
        let paths = t.model_file_paths();
        assert!(paths[0].ends_with("encoder.int8.onnx"));
        assert!(paths[1].ends_with("decoder.int8.onnx"));
        assert!(paths[2].ends_with("joiner.int8.onnx"));
        assert!(paths[3].ends_with("tokens.txt"));
    }
}
