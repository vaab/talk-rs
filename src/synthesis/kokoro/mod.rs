//! Local Kokoro multi-lang TTS backend (sherpa-onnx, CPU).
//!
//! The synthesis-side sibling of [`crate::transcription::parakeet`]:
//! on-device, keyed by model files on disk rather than an API key.
//! Three submodules, mirroring the parakeet layout:
//!
//! * [`model`] — presence check + auto-download (via the shared
//!   [`crate::model_fetch`] helper) + the language-agnostic ONNX
//!   `voice`-metadata patch that derives `model-<lang>.onnx` on demand.
//! * [`voices`] — the 53-speaker table and the voice-name → speaker-id
//!   mapping, plus the prefix-derived per-language default voice.
//! * [`synthesizer`] — the [`crate::synthesis::OneShotSynthesizer`]
//!   implementation wiring config + request + model + sherpa TTS.

pub mod consent;
pub mod model;
mod synthesizer;
pub(crate) mod voices;

pub use synthesizer::KokoroOneShotSynthesizer;
