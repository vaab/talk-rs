//! Local Parakeet TDT transcription backend.
//!
//! Phase 3 wires the actual sherpa-onnx inference: file or
//! recorded-stream in, plain text out.  Two earlier phases handle
//! the rest of the lifecycle:
//!
//! * [`model`] — presence check + auto-download + atomic extract of
//!   the prebuilt sherpa-onnx Parakeet TDT v3 model.  Triggered from
//!   [`OneShotTranscriber::validate`] on first use.
//! * Config wiring ([`crate::config::ParakeetConfig`]) — resolves
//!   the model directory, variant, and thread count from YAML +
//!   environment.
//!
//! # Why a fresh recognizer per call (v1)
//!
//! The [`OneShotTranscriber`] trait has no construction hook for the
//! recognizer (its only state is the config), so we currently build
//! the `OfflineRecognizer` inside [`ParakeetOneShotTranscriber::fetch_transcription`].
//! For the `dictate` workflow that runs one transcription per
//! invocation, the ~1.7 s model-load cost is paid once per dictation
//! and is acceptable.  A future optimisation is to cache the
//! recognizer on `self` behind a `OnceCell`/`Mutex` — left out of
//! v1 to keep this module narrow and reviewable.
//!
//! # Blocking inside async
//!
//! sherpa-onnx's `create` / `accept_waveform` / `decode` /
//! `get_result` are synchronous C calls that can run for hundreds of
//! milliseconds.  We wrap them in
//! [`tokio::task::spawn_blocking`] so they cannot stall the async
//! executor.  `OfflineRecognizer` and `OfflineStream` are both
//! `Send + Sync` (Phase 0 verified), so moving them into the blocking
//! closure is sound.

#[cfg(feature = "parakeet")]
pub mod consent;

#[cfg(feature = "parakeet")]
pub mod model;

#[cfg(feature = "parakeet")]
mod transcriber;

#[cfg(feature = "parakeet")]
pub use transcriber::ParakeetOneShotTranscriber;
