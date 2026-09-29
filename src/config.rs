//! Configuration file handling for talk-rs.
//!
//! Config is stored in ~/.config/talk-rs/config.yaml

use crate::error::TalkError;
use crate::paste_config::{
    PasteNodeConfig, PasteTiming, DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_FETCH_TIMEOUT_MS,
    DEFAULT_RESTORE_SETTLE_MS, DEFAULT_TARGET_FETCH_RETRIES, DEFAULT_TARGET_QUIESCENCE_MS,
};
use directories::{ProjectDirs, UserDirs};
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Main configuration structure.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Output directory for recordings, screenshots, and clipboard saves.
    ///
    /// The `record` command saves files here by default when no explicit
    /// output path is given.
    ///
    /// Must be an absolute path: a relative value would resolve against
    /// the (unpredictable) process working directory at runtime, so it
    /// is rejected during validation.
    pub output_dir: PathBuf,

    /// Transcription providers configuration.
    pub providers: ProvidersConfig,

    /// Optional indicator settings.
    pub indicators: Option<IndicatorsConfig>,

    /// Optional transcription defaults (provider selection, etc.).
    pub transcription: Option<TranscriptionConfig>,

    /// Optional speech-synthesis defaults for the `speak` command
    /// (default provider selection).
    #[serde(default)]
    pub speak: Option<SpeakConfig>,

    /// Optional paste behaviour settings.
    #[serde(default)]
    pub paste: Option<PasteConfig>,

    /// Optional runtime audio settings (Bluetooth profile switching, …).
    ///
    /// Distinct from [`AudioConfig`] which is the internal codec /
    /// sample-rate / bitrate config (hardcoded, not user-facing).
    /// This section holds the user-facing audio toggles that have to
    /// be read from `config.yaml`.
    #[serde(default)]
    pub audio: Option<AudioSettings>,

    /// Optional human-facing recording-quality settings for the
    /// `record` command's `.ogg` output.
    ///
    /// This is deliberately separate from [`AudioConfig`]: that struct
    /// holds the *internal* characteristics required by the
    /// transcription providers (16 kHz mono — both Voxtral and Whisper
    /// downsample to 16 kHz mono internally), which are an
    /// implementation detail and must NOT be user-tunable.  This
    /// section, by contrast, controls the quality of recordings meant
    /// for a human to listen to / share, where higher fidelity is a
    /// legitimate preference.
    #[serde(default)]
    pub recording: Option<RecordingConfig>,
}

/// User-facing audio settings (loaded from `config.yaml`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioSettings {
    /// Auto-switch a connected Bluetooth headset to its Hands-Free
    /// Profile (HFP) for the duration of a recording, then restore
    /// the original profile (typically A2DP) afterwards.
    ///
    /// Default: `true`.  Override per-invocation with
    /// `--no-bt-auto-switch`, or globally via the
    /// `TALK_RS_AUDIO_BT_AUTO_SWITCH` environment variable.
    #[serde(default)]
    pub bt_auto_switch: Option<bool>,
}

impl AudioSettings {
    /// Resolved value of `bt_auto_switch`: `true` if unset (default-on).
    pub fn bt_auto_switch_enabled(&self) -> bool {
        self.bt_auto_switch.unwrap_or(true)
    }
}

/// Default sample rate for human-facing recordings (48 kHz, full-band
/// Opus, matches the PipeWire native rate so no resampling is needed).
pub const DEFAULT_RECORDING_SAMPLE_RATE: u32 = 48_000;
/// Default channel count for human-facing recordings (mono).  Most
/// dictation microphones are physically mono, and the transcription
/// providers downmix to mono regardless; stereo can be opted into for
/// genuine stereo sources.
pub const DEFAULT_RECORDING_CHANNELS: u8 = 1;
/// Default Opus bitrate for human-facing recordings (128 kbps —
/// transparent quality for voice and comfortable headroom for
/// music / ambient content).
pub const DEFAULT_RECORDING_BITRATE: u32 = 128_000;

/// Human-facing recording-quality settings (loaded from `config.yaml`).
///
/// Controls the `record` command's `.ogg` output.  Every field is
/// optional and falls back to a fidelity-oriented default; see
/// [`RecordingConfig::resolved`].
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordingConfig {
    /// Sample rate in Hz for the recorded `.ogg`.  Default: 48000.
    #[serde(default)]
    pub sample_rate: Option<u32>,

    /// Channel count for the recorded `.ogg` (1 = mono, 2 = stereo).
    /// Default: 1 (mono).
    #[serde(default)]
    pub channels: Option<u8>,

    /// Opus bitrate in bps for the recorded `.ogg`.  Default: 128000.
    #[serde(default)]
    pub bitrate: Option<u32>,
}

impl RecordingConfig {
    /// Resolve the recording quality into a concrete [`AudioConfig`],
    /// applying fidelity-oriented defaults for any unset field.
    ///
    /// Unlike [`AudioConfig::new`] (the hardcoded 16 kHz mono
    /// transcription profile), this produces the *human-facing*
    /// recording profile.
    pub fn resolved(&self) -> AudioConfig {
        AudioConfig {
            sample_rate: self.sample_rate.unwrap_or(DEFAULT_RECORDING_SAMPLE_RATE),
            channels: self.channels.unwrap_or(DEFAULT_RECORDING_CHANNELS),
            bitrate: self.bitrate.unwrap_or(DEFAULT_RECORDING_BITRATE),
        }
    }
}

/// Transcription provider identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// Mistral / Voxtral transcription.
    Mistral,
    /// OpenAI Whisper / GPT-4o transcription.
    #[serde(alias = "openai")]
    OpenAI,
    /// Local Parakeet TDT (ONNX, CPU).  Unlike [`Provider::Mistral`]
    /// and [`Provider::OpenAI`] (remote APIs keyed by `api_key`),
    /// Parakeet runs on-device and is keyed by model files on disk —
    /// the deliberate local-vs-remote divergence in the provider
    /// abstraction (see `parakeet-local-backend.md`).  Always present
    /// regardless of the `parakeet` build feature so a config with
    /// `provider: parakeet` always parses.
    Parakeet,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Mistral => write!(f, "mistral"),
            Provider::OpenAI => write!(f, "openai"),
            Provider::Parakeet => write!(f, "parakeet"),
        }
    }
}

impl Provider {
    pub fn is_local(self) -> bool {
        matches!(self, Self::Parakeet)
    }
}

impl std::str::FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mistral" => Ok(Provider::Mistral),
            "openai" => Ok(Provider::OpenAI),
            "parakeet" => Ok(Provider::Parakeet),
            other => Err(format!(
                "unknown provider '{}' (expected 'mistral', 'openai', or 'parakeet')",
                other
            )),
        }
    }
}

/// Speech-synthesis (text-to-speech) provider identifier.
///
/// The synthesis-side mirror of [`Provider`].  Like `Provider`, its
/// variants are ALWAYS present regardless of build features so a YAML
/// config naming `provider: kokoro` always parses — the feature gate
/// only affects whether the concrete backend can be *constructed*
/// (see `crate::synthesis::create_oneshot_synthesizer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SynthesisProvider {
    /// Local Kokoro multi-lang TTS (ONNX via sherpa-onnx, CPU).  The
    /// synthesis-side sibling of [`Provider::Parakeet`]: on-device,
    /// keyed by model files on disk rather than an API key.
    Kokoro,
    /// Remote Mistral / Voxtral TTS (`POST /v1/audio/speech`).  Shares
    /// the [`MistralConfig`] API key with the transcription surface.
    Mistral,
}

impl std::fmt::Display for SynthesisProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SynthesisProvider::Kokoro => write!(f, "kokoro"),
            SynthesisProvider::Mistral => write!(f, "mistral"),
        }
    }
}

impl std::str::FromStr for SynthesisProvider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "kokoro" => Ok(SynthesisProvider::Kokoro),
            "mistral" => Ok(SynthesisProvider::Mistral),
            other => Err(format!(
                "unknown synthesis provider '{}' (expected 'kokoro' or 'mistral')",
                other
            )),
        }
    }
}

/// Transcription providers configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvidersConfig {
    /// Mistral API configuration (optional — only required when using Mistral).
    #[serde(default)]
    pub mistral: Option<MistralConfig>,

    /// OpenAI API configuration (optional — only required when using OpenAI).
    #[serde(default)]
    pub openai: Option<OpenAIConfig>,

    /// Local Parakeet configuration (optional — only required when
    /// using the Parakeet provider; defaults are usable when present
    /// as an empty `parakeet: {}` block).
    #[serde(default)]
    pub parakeet: Option<ParakeetConfig>,

    /// Local Kokoro TTS configuration (optional — only required when
    /// using the Kokoro synthesis provider; defaults are usable when
    /// present as an empty `kokoro: {}` block).
    #[serde(default)]
    pub kokoro: Option<KokoroConfig>,
}

/// Mistral API configuration.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MistralConfig {
    /// API key for Mistral transcription service.
    #[serde(default)]
    pub api_key: String,

    /// Base URL for the Mistral API (defaults to `https://api.mistral.ai`).
    ///
    /// Override to point at a self-hosted or API-compatible endpoint.
    /// Only the base URL is required — path segments are appended
    /// automatically.
    #[serde(default)]
    pub url: Option<String>,

    /// Model name for transcription (defaults to "voxtral-mini-2507").
    #[serde(default = "default_mistral_model")]
    pub model: String,

    /// Context bias words/phrases for improved transcription accuracy.
    ///
    /// Comma-separated list of up to 100 words or phrases to guide
    /// the model toward correct spellings of names, technical terms,
    /// or domain-specific vocabulary.
    #[serde(default)]
    pub context_bias: Option<String>,

    /// Model name for text-to-speech synthesis via the Voxtral TTS
    /// endpoint (`POST /v1/audio/speech`).  Defaults to
    /// `voxtral-mini-tts-latest`.
    ///
    /// Deliberately co-located with the transcription fields rather
    /// than in a separate section: the Mistral API key is shared
    /// between speech-to-text and text-to-speech, so a single
    /// `providers.mistral` block keys both surfaces.
    #[serde(default = "default_mistral_tts_model")]
    pub tts_model: String,

    /// Default *pinned* TTS voice id (a Mistral preset voice UUID) used
    /// by the `speak` command when `--voice` is not supplied.  A pin
    /// takes precedence over the per-language [`Self::tts_voices`] map,
    /// so a pinned voice whose language conflicts with the resolved
    /// language trips the mismatch guard (see
    /// [`crate::synthesis::resolve::guard_voice_lang`]).  When unset the
    /// per-language map (then a built-in default table) is consulted;
    /// the available voices can be listed via `GET /v1/audio/voices`.
    #[serde(default)]
    pub tts_voice: Option<String>,

    /// Per-language TTS voices (ISO 639-1 code → preset voice UUID).
    ///
    /// Consulted after `--voice` and [`Self::tts_voice`] but before the
    /// built-in default table: the entry matching the resolved language
    /// is chosen automatically, so it can never trip the mismatch guard
    /// (an auto-selected voice matches the language by construction).
    /// Example: `{ en: "<paul-uuid>", fr: "<marie-uuid>" }`.
    #[serde(default)]
    pub tts_voices: Option<std::collections::HashMap<String, String>>,
}

fn default_mistral_model() -> String {
    "voxtral-mini-2507".to_string()
}

fn default_mistral_tts_model() -> String {
    "voxtral-mini-tts-latest".to_string()
}

/// Realtime transcription delay requested from OpenAI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIRealtimeDelay {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl std::fmt::Display for OpenAIRealtimeDelay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
        };
        write!(f, "{value}")
    }
}

impl std::str::FromStr for OpenAIRealtimeDelay {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "minimal" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::Xhigh),
            _ => Err(format!(
                "invalid OpenAI realtime delay '{value}' (expected minimal, low, medium, high, or xhigh)"
            )),
        }
    }
}

/// OpenAI API configuration.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct OpenAIConfig {
    /// API key for OpenAI transcription service.
    #[serde(default)]
    pub api_key: String,

    /// Base URL for the OpenAI API (defaults to `https://api.openai.com`).
    ///
    /// Override to point at a self-hosted or API-compatible endpoint.
    /// Only the base URL is required — path segments are appended
    /// automatically.
    #[serde(default)]
    pub url: Option<String>,

    /// Model name for one-shot transcription (defaults to "gpt-transcribe").
    #[serde(default = "default_openai_model")]
    pub model: String,

    /// Model name for realtime transcription (defaults to
    /// `"gpt-live-transcribe"`).
    #[serde(default = "default_openai_realtime_model")]
    pub realtime_model: String,

    /// Optional transcription prompt. Supported by the new models and
    /// retained for compatible legacy models.
    #[serde(default)]
    pub prompt: Option<String>,

    /// Optional expected vocabulary. Values are sent as repeated
    /// `keywords[]` fields by `gpt-transcribe` and as a JSON array by
    /// `gpt-live-transcribe`.
    #[serde(default)]
    pub keywords: Option<Vec<String>>,

    /// Optional expected languages. The new models accept multiple
    /// values; legacy models accept at most one unambiguous language.
    #[serde(default)]
    pub languages: Option<Vec<String>>,

    /// Optional delay profile for `gpt-live-transcribe` realtime sessions.
    #[serde(default)]
    pub realtime_delay: Option<OpenAIRealtimeDelay>,
}

fn default_openai_model() -> String {
    "gpt-transcribe".to_string()
}

fn default_openai_realtime_model() -> String {
    "gpt-live-transcribe".to_string()
}

/// Parakeet quantization variant.
///
/// Selects which prebuilt model tarball is fetched (in later phases)
/// and which on-disk filenames are expected.  `int8` is the
/// recommended CPU-friendly default; `fp32` is the higher-fidelity,
/// much larger option.  Both names match the sherpa-onnx release
/// asset naming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParakeetVariant {
    /// INT8 quantized (~640 MB).  Recommended; CPU-friendly.
    #[default]
    Int8,
    /// FP32 full precision (~2.3 GB).  Higher fidelity, much larger.
    Fp32,
}

impl std::fmt::Display for ParakeetVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParakeetVariant::Int8 => write!(f, "int8"),
            ParakeetVariant::Fp32 => write!(f, "fp32"),
        }
    }
}

impl std::str::FromStr for ParakeetVariant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "int8" => Ok(ParakeetVariant::Int8),
            "fp32" => Ok(ParakeetVariant::Fp32),
            other => Err(format!(
                "unknown parakeet variant '{}' (expected 'int8' or 'fp32')",
                other
            )),
        }
    }
}

/// Local Parakeet backend configuration.
///
/// Unlike [`MistralConfig`] / [`OpenAIConfig`] (remote APIs keyed by
/// `api_key`), Parakeet runs on-device: it is keyed by **model files
/// on disk**.  This is the deliberate local-vs-remote divergence in
/// the provider abstraction — see plan
/// `parakeet-local-backend.md`.  Every field is optional; the
/// default-constructed value is a usable INT8 configuration that
/// resolves its model directory to the XDG data dir.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParakeetConfig {
    /// Quantization variant.  `int8` (default) or `fp32`.  Selects
    /// which model tarball is fetched (in later phases) and which
    /// on-disk filenames are expected.
    #[serde(default)]
    pub variant: ParakeetVariant,

    /// Directory holding the model files
    /// (encoder/decoder/joiner + tokens.txt).  Defaults to the XDG
    /// data dir, suffixed by variant:
    /// `~/.local/share/talk-rs/models/parakeet-tdt-0.6b-v3-<variant>`.
    #[serde(default)]
    pub model_dir: Option<PathBuf>,

    /// Decode threads.  Default: 2.
    #[serde(default = "default_parakeet_threads")]
    pub num_threads: i32,

    /// Logical model name surfaced in cache keys / metadata.
    /// Defaults to `parakeet-tdt-0.6b-v3-<variant>`.
    #[serde(default)]
    pub model: Option<String>,
}

fn default_parakeet_threads() -> i32 {
    2
}

impl ParakeetConfig {
    /// Resolved variant (the field, with `Default` falling back to
    /// `Int8` when the YAML omits it).
    pub fn resolved_variant(&self) -> ParakeetVariant {
        self.variant
    }

    /// Resolved on-disk model directory.
    ///
    /// Pure path computation — does NOT touch the filesystem.  Returns
    /// the user-supplied `model_dir` with a leading tilde expanded when present, else
    /// constructs the XDG default
    /// `<data_dir>/models/parakeet-tdt-0.6b-v3-<variant>` (so the
    /// `int8` and `fp32` caches never collide).  Returns a
    /// [`TalkError::Config`] when the XDG data dir cannot be resolved,
    /// mirroring [`config_dir`].
    pub fn resolved_model_dir(&self) -> Result<PathBuf, TalkError> {
        if let Some(ref dir) = self.model_dir {
            return expand_tilde(dir);
        }
        let data_dir = ProjectDirs::from("org", "kalysto", "talk-rs")
            .map(|dirs| dirs.data_dir().to_path_buf())
            .ok_or_else(|| {
                TalkError::Config(
                    "Could not determine data directory for parakeet model_dir".to_string(),
                )
            })?;
        Ok(data_dir
            .join("models")
            .join(format!("parakeet-tdt-0.6b-v3-{}", self.variant)))
    }

    /// Resolved logical model name surfaced in cache keys / metadata.
    /// Returns `model` when set, else `parakeet-tdt-0.6b-v3-<variant>`.
    pub fn resolved_model_name(&self) -> String {
        if let Some(ref m) = self.model {
            return m.clone();
        }
        format!("parakeet-tdt-0.6b-v3-{}", self.variant)
    }
}

/// Local Kokoro TTS backend configuration.
///
/// The synthesis-side sibling of [`ParakeetConfig`]: like Parakeet,
/// Kokoro runs on-device and is keyed by **model files on disk**
/// rather than an API key.  Every field is optional; the
/// default-constructed value resolves its model directory to the XDG
/// data dir under `models/kokoro-multi-lang-v1_0`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KokoroConfig {
    /// Reserved for future multi-variant support (mirrors
    /// [`ParakeetConfig::variant`]).  Kokoro currently ships a single
    /// `kokoro-multi-lang-v1_0` tarball, so this is accepted-but-unused
    /// today; kept so a future quantized variant does not need a
    /// config-schema change.
    #[serde(default)]
    pub variant: Option<String>,

    /// Directory holding the model files (`model.onnx`, `voices.bin`,
    /// `tokens.txt`, `espeak-ng-data/`, `dict/`, lexicons).  Defaults
    /// to the XDG data dir:
    /// `~/.local/share/talk-rs/models/kokoro-multi-lang-v1_0`.
    #[serde(default)]
    pub model_dir: Option<PathBuf>,

    /// Default voice name (e.g. `af_heart`, `am_michael`, `ff_siwis`)
    /// used when `--voice` is not supplied.  When unset the backend
    /// picks a per-language default (`af_heart` for English,
    /// `ff_siwis` for French).
    #[serde(default)]
    pub voice: Option<String>,

    /// Inference threads.  Default: 4.
    #[serde(default)]
    pub num_threads: Option<usize>,

    /// Default phonemization language (`en` or `fr`) used when
    /// `--lang` is not supplied.  Defaults to `en`.
    #[serde(default)]
    pub lang: Option<String>,
}

impl KokoroConfig {
    /// Resolved on-disk model directory.
    ///
    /// Pure path computation — does NOT touch the filesystem.  Returns
    /// the user-supplied `model_dir` with a leading tilde expanded when present, else
    /// constructs the XDG default
    /// `<data_dir>/models/kokoro-multi-lang-v1_0`.  Mirrors
    /// [`ParakeetConfig::resolved_model_dir`].
    pub fn resolved_model_dir(&self) -> Result<PathBuf, TalkError> {
        if let Some(ref dir) = self.model_dir {
            return expand_tilde(dir);
        }
        let data_dir = ProjectDirs::from("org", "kalysto", "talk-rs")
            .map(|dirs| dirs.data_dir().to_path_buf())
            .ok_or_else(|| {
                TalkError::Config(
                    "Could not determine data directory for kokoro model_dir".to_string(),
                )
            })?;
        Ok(data_dir.join("models").join("kokoro-multi-lang-v1_0"))
    }

    /// Resolved inference thread count (defaults to 4).
    pub fn resolved_num_threads(&self) -> usize {
        self.num_threads.unwrap_or(4)
    }
}

/// Transcription defaults.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct TranscriptionConfig {
    /// Default transcription provider when `--provider` is not specified.
    #[serde(default = "default_provider")]
    pub default_provider: Provider,
    #[serde(default)]
    pub default_chain: Option<String>,
    #[serde(default)]
    pub chains: std::collections::BTreeMap<String, Vec<ChainEntryConfig>>,
    #[serde(default)]
    pub outage_memory: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ChainEntryConfig {
    Short(String),
    Full(ChainEntryDetails),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainEntryDetails {
    pub model: String,
    #[serde(default)]
    pub retries: Option<u32>,
    #[serde(default)]
    pub wait: Option<String>,
    #[serde(default)]
    pub languages: Option<Vec<String>>,
    #[serde(default)]
    pub supports: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ChainEntry {
    pub provider: Provider,
    pub model: String,
    /// Only a bare provider switches to its configured realtime model.
    pub realtime_model: Option<String>,
    pub retries: Option<u32>,
    pub wait: Option<std::time::Duration>,
    pub languages: Option<Vec<String>>,
    pub supports: Vec<String>,
}

impl ChainEntry {
    pub fn label(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedChain {
    pub name: String,
    pub entries: Vec<ChainEntry>,
    pub outage_memory: std::time::Duration,
}

fn parse_chain_duration(value: &str) -> Result<std::time::Duration, String> {
    let (number, multiplier) = match value.strip_suffix('s') {
        Some(number) => (number, 1),
        None => match value.strip_suffix('m') {
            Some(number) => (number, 60),
            None => return Err(format!("invalid duration \"{value}\" (use e.g. 30s or 2m)")),
        },
    };
    let seconds = number
        .parse::<u64>()
        .ok()
        .filter(|n| *n > 0)
        .and_then(|n| n.checked_mul(multiplier))
        .ok_or_else(|| format!("invalid duration \"{value}\" (use e.g. 30s or 2m)"))?;
    Ok(std::time::Duration::from_secs(seconds))
}

impl Config {
    pub fn resolve_chain(
        &self,
        name: Option<&str>,
        cli_provider: Option<Provider>,
        cli_model: Option<&str>,
    ) -> Result<Option<ResolvedChain>, TalkError> {
        let transcription = match &self.transcription {
            Some(config) => config,
            None => {
                return name.map_or(Ok(None), |name| {
                    Err(TalkError::Config(format!(
                        "unknown transcription chain \"{name}\""
                    )))
                })
            }
        };
        let selected = name.or_else(|| {
            if cli_provider.is_none() && cli_model.is_none() {
                transcription.default_chain.as_deref()
            } else {
                None
            }
        });
        let Some(name) = selected else {
            return Ok(None);
        };
        let specs = transcription
            .chains
            .get(name)
            .ok_or_else(|| TalkError::Config(format!("unknown transcription chain \"{name}\"")))?;
        if specs.is_empty() {
            return Err(TalkError::Config(format!(
                "chain \"{name}\" must have at least one entry"
            )));
        }
        let mut entries = Vec::with_capacity(specs.len());
        let mut seen = std::collections::HashSet::new();
        for (index, spec) in specs.iter().enumerate() {
            let (id, retries, wait, languages, supports) = match spec {
                ChainEntryConfig::Short(id) => (id.as_str(), None, None, None, Vec::new()),
                ChainEntryConfig::Full(details) => (
                    details.model.as_str(),
                    details.retries,
                    details.wait.as_deref(),
                    details.languages.clone(),
                    details.supports.clone(),
                ),
            };
            let contextual = |message| {
                TalkError::Config(format!(
                    "chain \"{name}\": entry {} \"{id}\": {message}",
                    index + 1
                ))
            };
            let (provider_name, model) =
                id.split_once('/').map_or((id, None), |(p, m)| (p, Some(m)));
            let provider: Provider = provider_name.parse().map_err(contextual)?;
            let realtime_model = if model.is_none() {
                match provider {
                    Provider::OpenAI => self
                        .providers
                        .openai
                        .as_ref()
                        .map(|p| p.realtime_model.clone()),
                    Provider::Mistral => {
                        Some(crate::transcription::catalog::MISTRAL_REALTIME_MODELS[0].to_string())
                    }
                    Provider::Parakeet => None,
                }
            } else {
                None
            };
            let model = match model {
                Some(model) if !model.is_empty() => model.to_string(),
                Some(_) => return Err(contextual("model name must not be empty".to_string())),
                None => match provider {
                    Provider::Mistral => self.providers.mistral.as_ref().map(|p| p.model.clone()),
                    Provider::OpenAI => self.providers.openai.as_ref().map(|p| p.model.clone()),
                    Provider::Parakeet => self
                        .providers
                        .parakeet
                        .as_ref()
                        .map(|p| p.resolved_model_name()),
                }
                .unwrap_or_default(),
            };
            let configured = match provider {
                Provider::Mistral => self
                    .providers
                    .mistral
                    .as_ref()
                    .is_some_and(|p| !p.api_key.trim().is_empty()),
                Provider::OpenAI => self
                    .providers
                    .openai
                    .as_ref()
                    .is_some_and(|p| !p.api_key.trim().is_empty()),
                Provider::Parakeet => self.providers.parakeet.is_some(),
            };
            if !configured {
                return Err(contextual(format!(
                    "provider {provider} is not configured (add providers.{provider}{} to config)",
                    if provider.is_local() { "" } else { ".api_key" }
                )));
            }
            let label = format!("{provider}/{model}");
            if !seen.insert(label.clone()) {
                return Err(TalkError::Config(format!(
                    "chain \"{name}\": duplicate entry \"{label}\""
                )));
            }
            if provider.is_local() && index + 1 < specs.len() {
                let next = match &specs[index + 1] {
                    ChainEntryConfig::Short(s) => s.as_str(),
                    ChainEntryConfig::Full(d) => d.model.as_str(),
                };
                return Err(TalkError::Config(format!("chain \"{name}\": entry {} \"{id}\" is local and always available, so entries after it ({next}) would never run — move it to the end of the chain", index + 1)));
            }
            let wait = wait
                .map(parse_chain_duration)
                .transpose()
                .map_err(contextual)?;
            if languages.as_ref().is_some_and(|values| {
                values.is_empty() || values.iter().any(|s| s.trim().is_empty())
            }) {
                return Err(contextual(
                    "languages must be a non-empty list of codes".to_string(),
                ));
            }
            if let Some(capability) = supports
                .iter()
                .find(|cap| !matches!(cap.as_str(), "diarize" | "realtime"))
            {
                return Err(contextual(format!(
                    "unknown capability \"{capability}\" (expected diarize or realtime)"
                )));
            }
            entries.push(ChainEntry {
                provider,
                model,
                realtime_model,
                retries,
                wait,
                languages,
                supports,
            });
        }
        let outage_memory = transcription
            .outage_memory
            .as_deref()
            .map(parse_chain_duration)
            .transpose()
            .map_err(TalkError::Config)?
            .unwrap_or(std::time::Duration::from_secs(180));
        Ok(Some(ResolvedChain {
            name: name.to_string(),
            entries,
            outage_memory,
        }))
    }
}

fn default_provider() -> Provider {
    Provider::Mistral
}

/// Speech-synthesis defaults for the `speak` command.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct SpeakConfig {
    /// Default synthesis provider when `--provider` is not specified.
    ///
    /// When unset the `speak` command falls back to Kokoro if a
    /// `providers.kokoro` section exists, else Mistral.
    #[serde(default)]
    pub default_provider: Option<SynthesisProvider>,
}

/// Audio configuration.
///
/// Hardcoded to the only sensible values for voice dictation:
/// 16 kHz mono at 32 kbps. Making these user-configurable was a footgun.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate in Hz (e.g., 16000).
    pub sample_rate: u32,

    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u8,

    /// Bitrate in bps for compressed formats.
    pub bitrate: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioConfig {
    /// Create a new `AudioConfig` with hardcoded voice-dictation defaults.
    pub fn new() -> Self {
        Self {
            sample_rate: 16_000,
            channels: 1,
            bitrate: 32_000,
        }
    }
}

/// Visualizer mode for the recording badge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VizMode {
    /// FFT spectrogram waterfall (time × frequency, opacity = magnitude).
    Waterfall,
    /// RMS amplitude history (symmetric bars, scrolling left).
    Amplitude,
    /// FFT frequency-domain bar chart.
    Spectrum,
}

impl std::fmt::Display for VizMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizMode::Waterfall => write!(f, "waterfall"),
            VizMode::Amplitude => write!(f, "amplitude"),
            VizMode::Spectrum => write!(f, "spectrum"),
        }
    }
}

impl std::str::FromStr for VizMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "waterfall" => Ok(VizMode::Waterfall),
            "amplitude" => Ok(VizMode::Amplitude),
            "spectrum" => Ok(VizMode::Spectrum),
            other => Err(format!(
                "unknown visualizer mode '{}' (expected: waterfall, amplitude, spectrum)",
                other
            )),
        }
    }
}

/// Indicator configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndicatorsConfig {
    /// Interval between boop sounds in milliseconds.
    #[serde(default = "default_boop_interval_ms")]
    pub boop_interval_ms: u64,

    /// Show visual indicator overlay.
    #[serde(default = "default_visual_overlay")]
    pub visual_overlay: bool,

    /// Visualizer mode rendered inside the recording badge.
    ///
    /// When absent, the badge shows only the red dot (no visualization).
    pub viz: Option<VizMode>,

    /// Render visualizers in monochrome unless the CLI forces it on.
    #[serde(default)]
    pub mono: bool,
}

impl Default for IndicatorsConfig {
    fn default() -> Self {
        Self {
            boop_interval_ms: default_boop_interval_ms(),
            visual_overlay: default_visual_overlay(),
            viz: None,
            mono: false,
        }
    }
}

fn default_boop_interval_ms() -> u64 {
    5_000
}

fn default_visual_overlay() -> bool {
    true
}

/// Keyboard shortcut used for pasting text into the target application.
#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
pub enum PasteShortcut {
    /// Default: Ctrl+Shift+V (paste from primary clipboard).
    #[default]
    #[serde(rename = "ctrl_shift_v")]
    CtrlShiftV,
    /// Ctrl+V (paste from regular clipboard, common in terminals / Emacs).
    #[serde(rename = "ctrl_v")]
    CtrlV,
}

/// Flat (pre-tree) paste configuration schema.
///
/// Kept as the BACKWARD-COMPAT surface for `paste:` sections that
/// predate the node-tree refactor.  New configs should prefer the
/// tree form via [`PasteConfig::Tree`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatPasteConfig {
    /// Maximum characters per clipboard paste chunk.
    ///
    /// Text longer than this is split on word boundaries into
    /// consecutive Ctrl+Shift+V keystrokes.  Set to `0` to disable
    /// chunking entirely (paste in one shot).
    #[serde(default = "default_paste_chunk_chars")]
    pub chunk_chars: usize,

    /// Keyboard shortcut to trigger paste in the target application.
    #[serde(default)]
    pub shortcut: PasteShortcut,

    /// **Backward-compat only.**  Was the pre-restore "settle" window
    /// in the legacy heuristic.  Replaced by the deterministic
    /// per-chunk target-confirmation gate inside the clipboard node;
    /// kept as an accepted-but-ignored field so existing YAML configs
    /// do not need editing.
    #[serde(default = "default_paste_restore_settle_ms")]
    pub restore_settle_ms: u64,

    /// Per-chunk ABORT deadline in milliseconds.  Default 500 (post
    /// retry work; was 300 under Phase 2, 400 under the legacy
    /// heuristic gate).  When the target does not fetch a chunk in
    /// time the clipboard node RETRIES (see `target_fetch_retries`)
    /// before failing loudly.
    #[serde(default = "default_paste_chunk_fetch_timeout_ms")]
    pub chunk_fetch_timeout_ms: u64,

    /// Number of automatic per-chunk retries on the deterministic
    /// target-confirmation path (default `2` = up to 3 attempts).
    /// See `crate::paste_config::DEFAULT_TARGET_FETCH_RETRIES`.
    #[serde(default = "default_paste_target_fetch_retries")]
    pub target_fetch_retries: u32,

    /// Quiet window used to learn the target's per-chunk fetch count.
    #[serde(default = "default_paste_target_quiescence_ms")]
    pub target_quiescence_ms: u64,
}

/// Paste behaviour configuration.
///
/// Two surface forms, both honoured at deserialise time:
///
/// * **Tree** (new): the YAML carries a `node:` key at the top level
///   and is parsed as a [`crate::paste_config::PasteNodeConfig`].
///   Enables the composable paste-node abstraction.
/// * **Flat** (legacy): the YAML carries the historical flat keys
///   (`chunk_chars`, `shortcut`, `restore_settle_ms`,
///   `chunk_fetch_timeout_ms`).  Behaves byte-for-byte identically
///   to pre-refactor `talk-rs`.
///
/// Selection is by the presence of a top-level `node:` key: with it the
/// section is parsed as a tree, without it as the flat form.  Dispatching
/// explicitly (instead of `#[serde(untagged)]` trial-and-error) lets the
/// chosen form's own error surface, so a misspelled key is reported by
/// name rather than as "did not match any variant".
///
/// Both forms parse in every feature set: headless builds share the
/// user's config file and must accept any valid `paste:` section even
/// though only `ui` builds can materialise it ([`PasteConfig::build_root`]).
#[derive(Debug, Clone)]
pub enum PasteConfig {
    /// New tree form.  Always carries a `node:` key.
    Tree(PasteNodeConfig),
    /// Legacy flat form.
    Flat(FlatPasteConfig),
}

impl<'de> Deserialize<'de> for PasteConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;

        let value = serde_yaml::Value::deserialize(deserializer)?;
        let is_tree = value
            .as_mapping()
            .is_some_and(|map| map.contains_key("node"));
        if is_tree {
            PasteNodeConfig::deserialize(value)
                .map(Self::Tree)
                .map_err(|err| D::Error::custom(format!("paste (tree form): {err}")))
        } else {
            FlatPasteConfig::deserialize(value)
                .map(Self::Flat)
                .map_err(|err| D::Error::custom(format!("paste (flat form): {err}")))
        }
    }
}

impl PasteConfig {
    /// Materialise this configuration into a runtime paste-node tree.
    ///
    /// * Tree form ⇒ build the tree as-declared.
    /// * Flat form with `chunk_chars > 0` ⇒
    ///   `chunk(chunk_chars) → clipboard(shortcut, restore_settle_ms, chunk_fetch_timeout_ms)`.
    /// * Flat form with `chunk_chars == 0` ⇒
    ///   `clipboard(shortcut, restore_settle_ms, chunk_fetch_timeout_ms)` (chunk wrapper skipped).
    ///
    /// `no_chunk_paste` (the CLI flag) strips every `Chunk` node from
    /// the resulting tree — identical effect to today's
    /// "set chunk_chars to 0" path on the flat form, and a natural
    /// extension to arbitrary trees.
    #[cfg(feature = "ui")]
    pub fn build_root(&self, no_chunk_paste: bool) -> Box<dyn crate::paste::PasteNode> {
        let tree = self.to_tree();
        crate::paste::build_root_from_config(&tree, no_chunk_paste)
    }

    /// Resolve the settle/timeout timing knobs for this configuration.
    pub fn timing(&self) -> PasteTiming {
        crate::paste_config::timing_from_tree(&self.to_tree())
    }

    /// Lower this configuration to its equivalent
    /// [`PasteNodeConfig`] tree.  Flat form is mapped according to
    /// [`PasteConfig::build_root`]'s contract.
    pub fn to_tree(&self) -> PasteNodeConfig {
        match self {
            Self::Tree(t) => t.clone(),
            Self::Flat(f) => flat_to_tree(f),
        }
    }
}

impl PasteConfig {
    /// Backward-compat accessor: the configured chunk size.  For tree
    /// configs we walk down to the first `Chunk` node; `0` if none.
    pub fn chunk_chars(&self) -> usize {
        fn find(cfg: &PasteNodeConfig) -> Option<usize> {
            match cfg {
                PasteNodeConfig::Chunk { chunk_chars, .. } => Some(*chunk_chars),
                PasteNodeConfig::DetectDisplayServer { x11, .. } => find(x11),
                PasteNodeConfig::MatchWmClass { default, .. } => find(default),
                PasteNodeConfig::MatchForegroundApp { default, .. } => find(default),
                PasteNodeConfig::Clipboard { .. } | PasteNodeConfig::XtestType {} => None,
            }
        }
        match self {
            Self::Flat(f) => f.chunk_chars,
            Self::Tree(t) => find(t).unwrap_or(0),
        }
    }

    /// Backward-compat accessor: the configured paste shortcut.  For
    /// tree configs we walk down to the first `Clipboard` node;
    /// defaults to [`PasteShortcut::default`] if none.
    pub fn shortcut(&self) -> PasteShortcut {
        fn find(cfg: &PasteNodeConfig) -> Option<PasteShortcut> {
            match cfg {
                PasteNodeConfig::Clipboard { shortcut, .. } => Some(*shortcut),
                PasteNodeConfig::Chunk { child, .. } => find(child),
                PasteNodeConfig::DetectDisplayServer { x11, .. } => find(x11),
                PasteNodeConfig::MatchWmClass { default, .. } => find(default),
                PasteNodeConfig::MatchForegroundApp { default, .. } => find(default),
                PasteNodeConfig::XtestType {} => None,
            }
        }
        match self {
            Self::Flat(f) => f.shortcut,
            Self::Tree(t) => find(t).unwrap_or_default(),
        }
    }

    /// Backward-compat accessor: settle window in ms.
    pub fn restore_settle_ms(&self) -> u64 {
        match self {
            Self::Flat(f) => f.restore_settle_ms,
            Self::Tree(_) => self.timing().restore_settle_ms,
        }
    }

    /// Backward-compat accessor: per-chunk fetch timeout in ms.
    pub fn chunk_fetch_timeout_ms(&self) -> u64 {
        match self {
            Self::Flat(f) => f.chunk_fetch_timeout_ms,
            Self::Tree(_) => self.timing().chunk_fetch_timeout_ms,
        }
    }
}

fn flat_to_tree(f: &FlatPasteConfig) -> PasteNodeConfig {
    let clipboard = PasteNodeConfig::Clipboard {
        shortcut: f.shortcut,
        restore_settle_ms: f.restore_settle_ms,
        chunk_fetch_timeout_ms: f.chunk_fetch_timeout_ms,
        target_quiescence_ms: f.target_quiescence_ms,
        target_fetch_retries: f.target_fetch_retries,
    };
    if f.chunk_chars == 0 {
        clipboard
    } else {
        PasteNodeConfig::Chunk {
            chunk_chars: f.chunk_chars,
            child: Box::new(clipboard),
        }
    }
}

fn default_paste_chunk_chars() -> usize {
    DEFAULT_CHUNK_CHARS
}

fn default_paste_restore_settle_ms() -> u64 {
    DEFAULT_RESTORE_SETTLE_MS
}

fn default_paste_chunk_fetch_timeout_ms() -> u64 {
    DEFAULT_CHUNK_FETCH_TIMEOUT_MS
}

fn default_paste_target_fetch_retries() -> u32 {
    DEFAULT_TARGET_FETCH_RETRIES
}

fn default_paste_target_quiescence_ms() -> u64 {
    DEFAULT_TARGET_QUIESCENCE_MS
}

/// Expand a leading `~` (or `~/…`) in a path to the user's home
/// directory.
///
/// Only a tilde that is the *first* path component is expanded:
/// - `~`            → `$HOME`
/// - `~/foo/bar`    → `$HOME/foo/bar`
///
/// A tilde anywhere else (e.g. `/tmp/~/x`) is left untouched, and so
/// is the `~user` form (expanding another user's home would require a
/// passwd lookup we deliberately do not perform).  Paths without a
/// leading tilde are returned unchanged.
fn expand_tilde(path: &Path) -> Result<PathBuf, TalkError> {
    let mut components = path.components();
    let first = match components.next() {
        Some(std::path::Component::Normal(part)) => part,
        // Absolute, root, `.`, `..`, etc. — nothing to expand.
        _ => return Ok(path.to_path_buf()),
    };

    if first != "~" {
        // Includes the `~user` form (e.g. `~alice`): not expanded.
        return Ok(path.to_path_buf());
    }

    let home = UserDirs::new()
        .map(|dirs| dirs.home_dir().to_path_buf())
        .ok_or_else(|| {
            TalkError::Config(
                "Could not determine home directory to expand '~' in configured path".to_string(),
            )
        })?;

    // Re-attach the remainder (everything after the leading `~`).
    let rest = components.as_path();
    if rest.as_os_str().is_empty() {
        Ok(home)
    } else {
        Ok(home.join(rest))
    }
}

/// Get the config directory (~/.config/talk-rs/).
pub fn config_dir() -> Result<PathBuf, TalkError> {
    ProjectDirs::from("org", "kalysto", "talk-rs")
        .map(|dirs| dirs.config_dir().to_path_buf())
        .ok_or_else(|| TalkError::Config("Could not determine config directory".to_string()))
}

/// Get the default config file path (~/.config/talk-rs/config.yaml).
pub fn config_path() -> Result<PathBuf, TalkError> {
    Ok(config_dir()?.join("config.yaml"))
}

impl Config {
    /// CLI monochrome mode forces the configured mode on.
    pub fn resolved_mono(&self, cli_mono: bool) -> bool {
        cli_mono || self.indicators.as_ref().is_some_and(|ind| ind.mono)
    }

    /// Load configuration from file.
    ///
    /// If `path` is provided, loads from that path.
    /// Otherwise, loads from the default XDG config location.
    ///
    /// Environment variables can override config values:
    /// - TALK_RS_OUTPUT_DIR
    /// - TALK_RS_PROVIDERS_MISTRAL_API_KEY
    /// - TALK_RS_PROVIDERS_MISTRAL_URL
    /// - TALK_RS_PROVIDERS_MISTRAL_MODEL
    /// - TALK_RS_PROVIDERS_MISTRAL_CONTEXT_BIAS
    /// - TALK_RS_PROVIDERS_OPENAI_API_KEY
    /// - TALK_RS_PROVIDERS_OPENAI_URL
    /// - TALK_RS_PROVIDERS_OPENAI_MODEL
    /// - TALK_RS_PROVIDERS_OPENAI_REALTIME_MODEL
    /// - TALK_RS_PROVIDERS_OPENAI_PROMPT
    /// - TALK_RS_PROVIDERS_OPENAI_KEYWORDS (comma-separated; whitespace and empty entries ignored)
    /// - TALK_RS_PROVIDERS_OPENAI_LANGUAGES (comma-separated; whitespace and empty entries ignored)
    /// - TALK_RS_PROVIDERS_OPENAI_REALTIME_DELAY
    /// - TALK_RS_PROVIDERS_PARAKEET_VARIANT
    /// - TALK_RS_PROVIDERS_PARAKEET_MODEL_DIR
    /// - TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS
    /// - TALK_RS_PROVIDERS_PARAKEET_MODEL
    /// - TALK_RS_PROVIDERS_MISTRAL_TTS_MODEL
    /// - TALK_RS_PROVIDERS_MISTRAL_TTS_VOICE
    /// - TALK_RS_PROVIDERS_KOKORO_VARIANT
    /// - TALK_RS_PROVIDERS_KOKORO_MODEL_DIR
    /// - TALK_RS_PROVIDERS_KOKORO_VOICE
    /// - TALK_RS_PROVIDERS_KOKORO_NUM_THREADS
    /// - TALK_RS_PROVIDERS_KOKORO_LANG
    /// - TALK_RS_INDICATORS_VIZ
    /// - TALK_RS_INDICATORS_MONO
    pub fn load(path: Option<&Path>) -> Result<Self, TalkError> {
        let config_path = match path {
            Some(path) => path.to_path_buf(),
            None => config_path()?,
        };

        let content = fs::read_to_string(&config_path).map_err(|err| {
            TalkError::Config(format!(
                "Failed to read config file {}: {}",
                config_path.display(),
                err
            ))
        })?;

        let mut config: Config = serde_yaml::from_str(&content).map_err(|err| {
            TalkError::Config(format!(
                "Failed to parse config file {}: {}",
                config_path.display(),
                err
            ))
        })?;

        if let Some(value) = env_var_string("TALK_RS_OUTPUT_DIR")? {
            config.output_dir = PathBuf::from(value);
        }

        // Indicators env var overrides.
        if let Some(value) = env_var_string("TALK_RS_INDICATORS_VIZ")? {
            let mode: VizMode = value.parse().map_err(TalkError::Config)?;
            config
                .indicators
                .get_or_insert_with(IndicatorsConfig::default)
                .viz = Some(mode);
        }
        if let Some(value) = env_var_string("TALK_RS_INDICATORS_MONO")? {
            let mono = parse_bool_env(&value).ok_or_else(|| {
                TalkError::Config(format!(
                    "TALK_RS_INDICATORS_MONO must be true/false/1/0/yes/no, got '{value}'"
                ))
            })?;
            config
                .indicators
                .get_or_insert_with(IndicatorsConfig::default)
                .mono = mono;
        }

        // Audio settings env var overrides.
        if let Some(value) = env_var_string("TALK_RS_AUDIO_BT_AUTO_SWITCH")? {
            let parsed = parse_bool_env(&value).ok_or_else(|| {
                TalkError::Config(format!(
                    "TALK_RS_AUDIO_BT_AUTO_SWITCH must be true/false/1/0/yes/no, got '{}'",
                    value
                ))
            })?;
            let audio = config.audio.get_or_insert_with(AudioSettings::default);
            audio.bt_auto_switch = Some(parsed);
        }

        // Recording-quality env var overrides (human-facing `record`
        // output).  Each parses to its numeric type and is stored on
        // the `recording` section, which is created on demand.
        if let Some(value) = env_var_string("TALK_RS_RECORDING_SAMPLE_RATE")? {
            let parsed = parse_u32_env("TALK_RS_RECORDING_SAMPLE_RATE", &value)?;
            let rec = config
                .recording
                .get_or_insert_with(RecordingConfig::default);
            rec.sample_rate = Some(parsed);
        }
        if let Some(value) = env_var_string("TALK_RS_RECORDING_CHANNELS")? {
            let parsed = parse_u32_env("TALK_RS_RECORDING_CHANNELS", &value)?;
            let channels = u8::try_from(parsed).map_err(|_| {
                TalkError::Config(format!(
                    "TALK_RS_RECORDING_CHANNELS must be 1 or 2, got '{}'",
                    value
                ))
            })?;
            if !matches!(channels, 1 | 2) {
                return Err(TalkError::Config(format!(
                    "TALK_RS_RECORDING_CHANNELS must be 1 or 2, got '{value}'"
                )));
            }
            let rec = config
                .recording
                .get_or_insert_with(RecordingConfig::default);
            rec.channels = Some(channels);
        }
        if let Some(value) = env_var_string("TALK_RS_RECORDING_BITRATE")? {
            let parsed = parse_u32_env("TALK_RS_RECORDING_BITRATE", &value)?;
            let rec = config
                .recording
                .get_or_insert_with(RecordingConfig::default);
            rec.bitrate = Some(parsed);
        }

        // Mistral env var overrides (only when the section exists).
        if let Some(ref mut mistral) = config.providers.mistral {
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_API_KEY")? {
                mistral.api_key = value;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_URL")? {
                mistral.url = Some(value);
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_MODEL")? {
                mistral.model = value;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_CONTEXT_BIAS")? {
                mistral.context_bias = Some(value);
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_TTS_MODEL")? {
                mistral.tts_model = value;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_TTS_VOICE")? {
                mistral.tts_voice = Some(value);
            }
        } else {
            // Allow creating the Mistral section purely from env vars.
            if let Some(api_key) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_API_KEY")? {
                config.providers.mistral = Some(MistralConfig {
                    api_key,
                    url: env_var_string("TALK_RS_PROVIDERS_MISTRAL_URL")?,
                    model: env_var_string("TALK_RS_PROVIDERS_MISTRAL_MODEL")?
                        .unwrap_or_else(default_mistral_model),
                    context_bias: env_var_string("TALK_RS_PROVIDERS_MISTRAL_CONTEXT_BIAS")?,
                    tts_model: env_var_string("TALK_RS_PROVIDERS_MISTRAL_TTS_MODEL")?
                        .unwrap_or_else(default_mistral_tts_model),
                    tts_voice: env_var_string("TALK_RS_PROVIDERS_MISTRAL_TTS_VOICE")?,
                    // The per-language `tts_voices` map is config-file
                    // only (a map has no single-string env form); a pure
                    // env-created section starts without it.
                    tts_voices: None,
                });
            }
        }

        // OpenAI env var overrides (only when the section exists).
        if let Some(ref mut openai) = config.providers.openai {
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_API_KEY")? {
                openai.api_key = value;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_URL")? {
                openai.url = Some(value);
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_MODEL")? {
                openai.model = value;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_REALTIME_MODEL")? {
                openai.realtime_model = value;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_PROMPT")? {
                openai.prompt = Some(value);
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_KEYWORDS")? {
                openai.keywords = parse_string_list_env(&value);
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_LANGUAGES")? {
                openai.languages = parse_string_list_env(&value);
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_REALTIME_DELAY")? {
                openai.realtime_delay = Some(value.parse().map_err(TalkError::Config)?);
            }
        } else {
            // Allow creating the OpenAI section purely from env vars.
            if let Some(api_key) = env_var_string("TALK_RS_PROVIDERS_OPENAI_API_KEY")? {
                config.providers.openai = Some(OpenAIConfig {
                    api_key,
                    url: env_var_string("TALK_RS_PROVIDERS_OPENAI_URL")?,
                    model: env_var_string("TALK_RS_PROVIDERS_OPENAI_MODEL")?
                        .unwrap_or_else(default_openai_model),
                    realtime_model: env_var_string("TALK_RS_PROVIDERS_OPENAI_REALTIME_MODEL")?
                        .unwrap_or_else(default_openai_realtime_model),
                    prompt: env_var_string("TALK_RS_PROVIDERS_OPENAI_PROMPT")?,
                    keywords: env_var_string("TALK_RS_PROVIDERS_OPENAI_KEYWORDS")?
                        .and_then(|value| parse_string_list_env(&value)),
                    languages: env_var_string("TALK_RS_PROVIDERS_OPENAI_LANGUAGES")?
                        .and_then(|value| parse_string_list_env(&value)),
                    realtime_delay: env_var_string("TALK_RS_PROVIDERS_OPENAI_REALTIME_DELAY")?
                        .map(|value| value.parse().map_err(TalkError::Config))
                        .transpose()?,
                });
            }
        }

        // Parakeet env var overrides (only when the section exists).
        if let Some(ref mut parakeet) = config.providers.parakeet {
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_PARAKEET_VARIANT")? {
                let variant: ParakeetVariant = value.parse().map_err(TalkError::Config)?;
                parakeet.variant = variant;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_PARAKEET_MODEL_DIR")? {
                parakeet.model_dir = Some(PathBuf::from(value));
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS")? {
                let parsed = parse_u32_env("TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS", &value)?;
                parakeet.num_threads = i32::try_from(parsed).map_err(|_| {
                    TalkError::Config(format!(
                        "TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS must fit in i32, got '{}'",
                        value
                    ))
                })?;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_PARAKEET_MODEL")? {
                parakeet.model = Some(value);
            }
        } else {
            // Allow creating the Parakeet section purely from env vars.
            // Any one of the four knobs (variant / model_dir /
            // num_threads / model) materialises the section, with
            // defaults for whichever knobs are unset — mirrors the
            // Mistral / OpenAI "create purely from env" branches.
            let variant_env = env_var_string("TALK_RS_PROVIDERS_PARAKEET_VARIANT")?;
            let model_dir_env = env_var_string("TALK_RS_PROVIDERS_PARAKEET_MODEL_DIR")?;
            let num_threads_env = env_var_string("TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS")?;
            let model_env = env_var_string("TALK_RS_PROVIDERS_PARAKEET_MODEL")?;
            if variant_env.is_some()
                || model_dir_env.is_some()
                || num_threads_env.is_some()
                || model_env.is_some()
            {
                let variant = match variant_env {
                    Some(v) => v.parse::<ParakeetVariant>().map_err(TalkError::Config)?,
                    None => ParakeetVariant::default(),
                };
                let num_threads = match num_threads_env {
                    Some(v) => {
                        let parsed = parse_u32_env("TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS", &v)?;
                        i32::try_from(parsed).map_err(|_| {
                            TalkError::Config(format!(
                                "TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS must fit in i32, got '{}'",
                                v
                            ))
                        })?
                    }
                    None => default_parakeet_threads(),
                };
                config.providers.parakeet = Some(ParakeetConfig {
                    variant,
                    model_dir: model_dir_env.map(PathBuf::from),
                    num_threads,
                    model: model_env,
                });
            }
        }

        // Kokoro env var overrides (only when the section exists).
        if let Some(ref mut kokoro) = config.providers.kokoro {
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_KOKORO_VARIANT")? {
                kokoro.variant = Some(value);
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_KOKORO_MODEL_DIR")? {
                kokoro.model_dir = Some(PathBuf::from(value));
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_KOKORO_VOICE")? {
                kokoro.voice = Some(value);
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_KOKORO_NUM_THREADS")? {
                let parsed = parse_u32_env("TALK_RS_PROVIDERS_KOKORO_NUM_THREADS", &value)?;
                kokoro.num_threads = Some(parsed as usize);
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_KOKORO_LANG")? {
                kokoro.lang = Some(value);
            }
        } else {
            // Allow creating the Kokoro section purely from env vars —
            // any one knob materialises it, mirroring the Parakeet
            // "create purely from env" branch.
            let variant_env = env_var_string("TALK_RS_PROVIDERS_KOKORO_VARIANT")?;
            let model_dir_env = env_var_string("TALK_RS_PROVIDERS_KOKORO_MODEL_DIR")?;
            let voice_env = env_var_string("TALK_RS_PROVIDERS_KOKORO_VOICE")?;
            let num_threads_env = env_var_string("TALK_RS_PROVIDERS_KOKORO_NUM_THREADS")?;
            let lang_env = env_var_string("TALK_RS_PROVIDERS_KOKORO_LANG")?;
            if variant_env.is_some()
                || model_dir_env.is_some()
                || voice_env.is_some()
                || num_threads_env.is_some()
                || lang_env.is_some()
            {
                let num_threads = match num_threads_env {
                    Some(v) => {
                        let parsed = parse_u32_env("TALK_RS_PROVIDERS_KOKORO_NUM_THREADS", &v)?;
                        Some(parsed as usize)
                    }
                    None => None,
                };
                config.providers.kokoro = Some(KokoroConfig {
                    variant: variant_env,
                    model_dir: model_dir_env.map(PathBuf::from),
                    voice: voice_env,
                    num_threads,
                    lang: lang_env,
                });
            }
        }

        // Expand a leading `~` in `output_dir` (from the file or the
        // TALK_RS_OUTPUT_DIR override) before validating, so the
        // documented `~/talk-rs-output` form resolves to an absolute
        // path under $HOME instead of a literal relative `~` directory.
        config.output_dir = expand_tilde(&config.output_dir)?;

        for (provider, section, env_key) in [
            (
                "mistral",
                config
                    .providers
                    .mistral
                    .as_ref()
                    .map(|p| p.api_key.as_str()),
                "TALK_RS_PROVIDERS_MISTRAL_API_KEY",
            ),
            (
                "openai",
                config.providers.openai.as_ref().map(|p| p.api_key.as_str()),
                "TALK_RS_PROVIDERS_OPENAI_API_KEY",
            ),
        ] {
            if section.is_some_and(str::is_empty) {
                return Err(TalkError::Config(format!(
                    "providers.{provider}.api_key is required in YAML or via {env_key}"
                )));
            }
        }

        validate_config(&config)?;

        Ok(config)
    }
}

fn env_var_string(key: &str) -> Result<Option<String>, TalkError> {
    match env::var(key) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => {
            Err(TalkError::Config(format!("{} must be valid UTF-8", key)))
        }
    }
}

/// Parse a boolean-ish env var value.  Accepts `true`/`false`,
/// `yes`/`no`, `1`/`0`, `on`/`off` (case-insensitive).  Returns
/// `None` for any other value (caller turns that into an error).
fn parse_bool_env(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "1" | "on" => Some(true),
        "false" | "no" | "0" | "off" => Some(false),
        _ => None,
    }
}

fn parse_string_list_env(value: &str) -> Option<Vec<String>> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

/// Parse an unsigned-integer env var value, producing a clear config
/// error (naming the variable) on failure.
fn parse_u32_env(key: &str, value: &str) -> Result<u32, TalkError> {
    value.trim().parse::<u32>().map_err(|_| {
        TalkError::Config(format!(
            "{} must be a positive integer, got '{}'",
            key, value
        ))
    })
}

fn validate_config(config: &Config) -> Result<(), TalkError> {
    if config.output_dir.as_os_str().is_empty() {
        return Err(TalkError::Config("output_dir is required".to_string()));
    }

    // `output_dir` must be absolute.  A relative path would resolve
    // against the process working directory at runtime, which is
    // unpredictable — the toggle daemon spawns with an inherited (and
    // effectively arbitrary) CWD, so recordings could land anywhere.
    // The documented contract (config.example.yaml / README) requires
    // an absolute path; enforce it here with a clear error rather than
    // failing silently later.
    if !config.output_dir.is_absolute() {
        return Err(TalkError::Config(format!(
            "output_dir must be an absolute path, got '{}'",
            config.output_dir.display()
        )));
    }

    if let Some(channels) = config.recording.as_ref().and_then(|rec| rec.channels) {
        if !matches!(channels, 1 | 2) {
            return Err(TalkError::Config(format!(
                "recording.channels must be 1 or 2, got {channels}"
            )));
        }
    }

    if let Some(transcription) = &config.transcription {
        if transcription.default_chain.is_some() || !transcription.chains.is_empty() {
            for name in transcription.chains.keys() {
                config.resolve_chain(Some(name), None, None)?;
            }
            if transcription.default_chain.is_some() {
                config.resolve_chain(None, None, None)?;
            }
        }
        if let Some(duration) = transcription.outage_memory.as_deref() {
            parse_chain_duration(duration).map_err(TalkError::Config)?;
        }
    }

    // Provider API keys are validated lazily — only when a provider is
    // actually used via the factory function.  This allows configs that
    // only define one provider to work without filling in keys for the
    // other.

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use std::error::Error;
    use std::ffi::OsString;
    use std::io::Write;
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use tempfile::NamedTempFile;

    #[test]
    fn chain_config_accepts_short_and_long_entries() -> Result<(), Box<dyn Error>> {
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers:
              mistral: {api_key: test-key}
              openai: {api_key: test-key}
              parakeet: {}
            transcription:
              default_chain: dictate
              outage_memory: 3m
              chains:
                dictate:
                  - mistral/voxtral-mini-2602
                  - model: openai/gpt-transcribe
                    retries: 2
                    wait: 30s
                    languages: [fr]
                  - model: mistral/voxtral-mini-2507
                    wait: 45s
                  - parakeet
                custom:
                  - model: mistral/some-future-model
                    supports: [diarize, realtime]
                live:
                  - openai
        "})?;
        let config = Config::load(Some(file.path()))?;
        let chain = config
            .resolve_chain(Some("dictate"), None, None)?
            .ok_or("chain missing")?;
        assert_eq!(chain.name, "dictate");
        assert_eq!(
            config
                .resolve_chain(None, None, None)?
                .ok_or("default missing")?
                .name,
            "dictate"
        );
        assert!(config
            .resolve_chain(None, Some(Provider::OpenAI), None)?
            .is_none());
        assert!(config
            .resolve_chain(None, None, Some("gpt-transcribe"))?
            .is_none());
        assert_eq!(
            chain
                .entries
                .iter()
                .map(|e| format!("{}/{}", e.provider, e.model))
                .collect::<Vec<_>>(),
            [
                "mistral/voxtral-mini-2602",
                "openai/gpt-transcribe",
                "mistral/voxtral-mini-2507",
                "parakeet/parakeet-tdt-0.6b-v3-int8"
            ]
        );
        assert_eq!(chain.entries[1].retries, Some(2));
        assert_eq!(
            chain.entries[1].wait,
            Some(std::time::Duration::from_secs(30))
        );
        assert_eq!(
            chain.entries[1].languages.as_deref(),
            Some(["fr".to_string()].as_slice())
        );
        assert_eq!(
            chain.entries[2].wait,
            Some(std::time::Duration::from_secs(45))
        );
        assert_eq!(
            config
                .resolve_chain(Some("custom"), None, None)?
                .ok_or("custom missing")?
                .entries[0]
                .supports,
            ["diarize", "realtime"]
        );
        assert_eq!(
            config
                .resolve_chain(Some("live"), None, None)?
                .ok_or("live missing")?
                .eligible(false, true, None)?
                .entries[0]
                .model,
            "gpt-live-transcribe"
        );
        Ok(())
    }

    #[test]
    fn chain_config_rejects_invalid_entries_with_exact_messages() -> Result<(), Box<dyn Error>> {
        let cases = [
            ("- parakeet\n      - openai/gpt-transcribe", "Configuration error: chain \"dictate\": entry 1 \"parakeet\" is local and always available, so entries after it (openai/gpt-transcribe) would never run — move it to the end of the chain"),
            ("- unknown/model", "Configuration error: chain \"dictate\": entry 1 \"unknown/model\": unknown provider 'unknown' (expected 'mistral', 'openai', or 'parakeet')"),
            ("[]", "Configuration error: chain \"dictate\" must have at least one entry"),
            ("- openai/gpt-transcribe\n      - openai/gpt-transcribe", "Configuration error: chain \"dictate\": duplicate entry \"openai/gpt-transcribe\""),
            ("- model: openai/gpt-transcribe\n        wait: never", "Configuration error: chain \"dictate\": entry 1 \"openai/gpt-transcribe\": invalid duration \"never\" (use e.g. 30s or 2m)"),
        ];
        for (entries, expected) in cases {
            let yaml = format!("output_dir: /tmp/test-output\nproviders:\n  openai: {{api_key: test-key}}\n  parakeet: {{}}\ntranscription:\n  chains:\n    dictate:\n      {entries}\n");
            let file = write_config(&yaml)?;
            assert_eq!(
                Config::load(Some(file.path()))
                    .expect_err("invalid chain")
                    .to_string(),
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn chain_config_rejects_unknown_defaults_and_unconfigured_provider(
    ) -> Result<(), Box<dyn Error>> {
        for (yaml, expected) in [
            (indoc! {"
                output_dir: /tmp/test-output
                providers: {}
                transcription:
                  default_chain: missing
            "}, "Configuration error: unknown transcription chain \"missing\""),
            (indoc! {"
                output_dir: /tmp/test-output
                providers: {}
                transcription:
                  chains:
                    dictate: [openai/gpt-transcribe]
            "}, "Configuration error: chain \"dictate\": entry 1 \"openai/gpt-transcribe\": provider openai is not configured (add providers.openai.api_key to config)"),
            (indoc! {"
                output_dir: /tmp/test-output
                providers: {}
                transcription:
                  outage_memory: endless
            "}, "Configuration error: invalid duration \"endless\" (use e.g. 30s or 2m)"),
        ] {
            let file = write_config(yaml)?;
            assert_eq!(Config::load(Some(file.path())).expect_err("invalid chain").to_string(), expected);
        }
        Ok(())
    }

    struct EnvGuard {
        key: String,
        value: Option<OsString>,
    }

    impl EnvGuard {
        fn set(key: &str, value: &str) -> Result<Self, Box<dyn Error>> {
            let previous = env::var_os(key);
            env::set_var(key, value);
            Ok(Self {
                key: key.to_string(),
                value: previous,
            })
        }

        fn clear(key: &str) -> Result<Self, Box<dyn Error>> {
            let previous = env::var_os(key);
            env::remove_var(key);
            Ok(Self {
                key: key.to_string(),
                value: previous,
            })
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.value {
                env::set_var(&self.key, value);
            } else {
                env::remove_var(&self.key);
            }
        }
    }

    fn env_lock() -> Result<MutexGuard<'static, ()>, Box<dyn Error>> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let mutex = ENV_LOCK.get_or_init(|| Mutex::new(()));
        mutex.lock().map_err(|_| "Env lock poisoned".into())
    }

    fn write_config(contents: &str) -> Result<NamedTempFile, Box<dyn Error>> {
        let mut file = NamedTempFile::new()?;
        file.write_all(contents.as_bytes())?;
        Ok(file)
    }

    #[test]
    fn unknown_config_keys_name_the_offending_field() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        for (section, yaml, key) in [
            (
                "root",
                indoc! {"
                output_dir: /tmp/test-output
                providers: {}
                output_directry: /tmp/typo
            "},
                "output_directry",
            ),
            (
                "providers",
                indoc! {"
                output_dir: /tmp/test-output
                providers:
                  mistral_api: {}
            "},
                "mistral_api",
            ),
            (
                "transcription",
                indoc! {"
                output_dir: /tmp/test-output
                providers: {}
                transcription:
                  default_provder: openai
            "},
                "default_provder",
            ),
            (
                "nested provider",
                indoc! {"
                output_dir: /tmp/test-output
                providers:
                  mistral:
                    api_key: test
                    modle: typo
            "},
                "modle",
            ),
            (
                "tree node",
                indoc! {"
                output_dir: /tmp/test-output
                providers: {}
                paste:
                  node: chunk
                  chunk_char: 80
                  child:
                    node: clipboard
            "},
                "chunk_char",
            ),
            (
                "nested tree node",
                indoc! {"
                output_dir: /tmp/test-output
                providers: {}
                paste:
                  node: match-wm-class
                  patterns:
                    - match: '*'
                      chld:
                        node: clipboard
                      child:
                        node: clipboard
                  default:
                    node: clipboard
            "},
                "chld",
            ),
        ] {
            let file = write_config(yaml)?;
            let err = Config::load(Some(file.path()))
                .err()
                .ok_or(format!("{section} accepted unknown key {key}"))?;
            assert!(err.to_string().contains(key), "{section}: {err}");
        }
        Ok(())
    }

    #[test]
    fn provider_sections_accept_env_only_keys_and_reject_missing_keys() -> Result<(), Box<dyn Error>>
    {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        for (provider, env_key) in [
            ("mistral", "TALK_RS_PROVIDERS_MISTRAL_API_KEY"),
            ("openai", "TALK_RS_PROVIDERS_OPENAI_API_KEY"),
        ] {
            let yaml = format!("output_dir: /tmp/test-output\nproviders:\n  {provider}:\n    model: custom-model\n");
            let file = write_config(&yaml)?;
            {
                let _key = EnvGuard::set(env_key, "env-secret")?;
                let config = Config::load(Some(file.path()))?;
                let actual = match provider {
                    "mistral" => config
                        .providers
                        .mistral
                        .as_ref()
                        .map(|p| p.api_key.as_str()),
                    _ => config.providers.openai.as_ref().map(|p| p.api_key.as_str()),
                };
                assert_eq!(actual, Some("env-secret"));
            }
            let err = Config::load(Some(file.path()))
                .err()
                .ok_or(format!("{provider} accepted absent API key"))?;
            let message = err.to_string();
            assert!(message.contains("api_key"), "{message}");
            assert!(message.contains(env_key), "{message}");
        }
        Ok(())
    }

    #[test]
    fn indicator_mono_resolves_yaml_env_and_cli() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _mono = EnvGuard::clear("TALK_RS_INDICATORS_MONO")?;
        for (yaml_mono, expected) in [("false", false), ("true", true)] {
            let file = write_config(&format!(
                "output_dir: /tmp/test-output\nproviders: {{}}\nindicators:\n  boop_interval_ms: 5000\n  visual_overlay: true\n  mono: {yaml_mono}\n"
            ))?;
            let config = Config::load(Some(file.path()))?;
            assert_eq!(config.resolved_mono(false), expected);
            assert!(config.resolved_mono(true));
            {
                let _override = EnvGuard::set("TALK_RS_INDICATORS_MONO", "true")?;
                assert!(Config::load(Some(file.path()))?.resolved_mono(false));
            }
        }
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers: {}
        "})?;
        assert!(!Config::load(Some(file.path()))?.resolved_mono(false));
        let _override = EnvGuard::set("TALK_RS_INDICATORS_MONO", "yes")?;
        assert!(Config::load(Some(file.path()))?.resolved_mono(false));
        Ok(())
    }

    #[test]
    fn indicator_mono_only_section_keeps_feedback_defaults() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers: {}
            indicators:
              mono: true
        "})?;
        let config = Config::load(Some(file.path()))?;
        let indicators = config.indicators.as_ref().ok_or("indicators missing")?;
        assert!(config.resolved_mono(false));
        assert_eq!(indicators.boop_interval_ms, 5_000);
        assert!(indicators.visual_overlay);
        Ok(())
    }

    #[test]
    fn indicator_viz_env_creates_missing_section() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _viz = EnvGuard::set("TALK_RS_INDICATORS_VIZ", "spectrum")?;
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers: {}
        "})?;
        let config = Config::load(Some(file.path()))?;
        assert_eq!(
            config.indicators.as_ref().and_then(|i| i.viz),
            Some(VizMode::Spectrum)
        );
        Ok(())
    }

    #[test]
    fn local_model_dirs_expand_leading_tilde() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let home = UserDirs::new()
            .ok_or("home unavailable")?
            .home_dir()
            .to_path_buf();
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers:
              parakeet:
                model_dir: ~/models/parakeet
              kokoro:
                model_dir: ~/models/kokoro
        "})?;
        let config = Config::load(Some(file.path()))?;
        assert_eq!(
            config
                .providers
                .parakeet
                .as_ref()
                .ok_or("parakeet missing")?
                .resolved_model_dir()?,
            home.join("models/parakeet")
        );
        assert_eq!(
            config
                .providers
                .kokoro
                .as_ref()
                .ok_or("kokoro missing")?
                .resolved_model_dir()?,
            home.join("models/kokoro")
        );
        Ok(())
    }

    #[test]
    fn recording_channels_reject_values_other_than_mono_or_stereo() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers: {}
            recording:
              channels: 3
        "})?;
        let err = Config::load(Some(file.path()))
            .err()
            .ok_or("YAML channels=3 accepted")?;
        assert!(err.to_string().contains("channels"), "{err}");

        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers: {}
        "})?;
        let _channels = EnvGuard::set("TALK_RS_RECORDING_CHANNELS", "3")?;
        let err = Config::load(Some(file.path()))
            .err()
            .ok_or("env channels=3 accepted")?;
        assert!(err.to_string().contains("CHANNELS"), "{err}");
        Ok(())
    }

    #[test]
    fn example_config_loads_without_unknown_keys() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let example = include_str!("../config.example.yaml");
        let file = write_config(&example.replace("<PLACEHOLDER>", "/tmp/test-output"))?;
        let config = Config::load(Some(file.path()))?;
        assert_eq!(config.output_dir, PathBuf::from("/tmp/test-output"));
        Ok(())
    }

    /// Clear all provider-related env vars to prevent cross-test leakage.
    fn clear_all_provider_env_vars() -> Result<Vec<EnvGuard>, Box<dyn Error>> {
        Ok(vec![
            EnvGuard::clear("TALK_RS_OUTPUT_DIR")?,
            EnvGuard::clear("TALK_RS_RECORDING_SAMPLE_RATE")?,
            EnvGuard::clear("TALK_RS_RECORDING_CHANNELS")?,
            EnvGuard::clear("TALK_RS_RECORDING_BITRATE")?,
            EnvGuard::clear("TALK_RS_INDICATORS_VIZ")?,
            EnvGuard::clear("TALK_RS_INDICATORS_MONO")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_API_KEY")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_URL")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_MODEL")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_CONTEXT_BIAS")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_TTS_MODEL")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_TTS_VOICE")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_API_KEY")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_URL")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_MODEL")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_REALTIME_MODEL")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_PROMPT")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_KEYWORDS")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_LANGUAGES")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_REALTIME_DELAY")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_PARAKEET_VARIANT")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_PARAKEET_MODEL_DIR")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_PARAKEET_MODEL")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_KOKORO_VARIANT")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_KOKORO_MODEL_DIR")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_KOKORO_VOICE")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_KOKORO_NUM_THREADS")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_KOKORO_LANG")?,
        ])
    }

    #[test]
    fn test_config_load_valid() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert_eq!(config.output_dir, PathBuf::from("/tmp/test-output"));
        let m = config.providers.mistral.as_ref().expect("mistral present");
        assert_eq!(m.api_key, "test-api-key");
        assert_eq!(m.model, "voxtral-mini-2507");
        assert!(m.context_bias.is_none());
        assert!(config.providers.openai.is_none());
        Ok(())
    }

    #[test]
    fn test_config_env_override() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _set_key = EnvGuard::set("TALK_RS_PROVIDERS_MISTRAL_API_KEY", "override-key")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let m = config.providers.mistral.as_ref().expect("mistral present");
        assert_eq!(m.api_key, "override-key");
        Ok(())
    }

    #[test]
    fn test_config_model_and_context_bias() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
    model: voxtral-mini-2602
    context_bias: "Kalysto,talk-rs,Voxtral"
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let m = config.providers.mistral.as_ref().expect("mistral present");
        assert_eq!(m.model, "voxtral-mini-2602");
        assert_eq!(m.context_bias.as_deref(), Some("Kalysto,talk-rs,Voxtral"));
        Ok(())
    }

    #[test]
    fn test_config_env_override_model_and_context_bias() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _set_model = EnvGuard::set("TALK_RS_PROVIDERS_MISTRAL_MODEL", "voxtral-mini-2602")?;
        let _set_bias = EnvGuard::set("TALK_RS_PROVIDERS_MISTRAL_CONTEXT_BIAS", "custom,terms")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let m = config.providers.mistral.as_ref().expect("mistral present");
        assert_eq!(m.model, "voxtral-mini-2602");
        assert_eq!(m.context_bias.as_deref(), Some("custom,terms"));
        Ok(())
    }

    #[test]
    fn test_config_missing_required_field() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        // Empty output_dir should still fail validation.
        let yaml = r#"
output_dir: ""
providers: {}
"#;
        let file = write_config(yaml)?;

        let result = Config::load(Some(file.path()));
        match result {
            Ok(_) => Err("Expected empty output_dir to fail".into()),
            Err(err) => {
                assert!(err.to_string().contains("output_dir is required"));
                Ok(())
            }
        }
    }

    #[test]
    fn test_config_relative_output_dir_rejected() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        // A relative output_dir must be rejected: it would otherwise
        // resolve against the process working directory at runtime
        // (unpredictable for the toggle daemon).  The documentation in
        // config.example.yaml / README states an absolute path is
        // required, so validation must enforce it.
        let yaml = r#"
output_dir: relative/path
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let result = Config::load(Some(file.path()));
        match result {
            Ok(_) => Err("Expected relative output_dir to fail".into()),
            Err(err) => {
                assert!(
                    err.to_string().contains("absolute"),
                    "error should mention 'absolute', got: {}",
                    err
                );
                Ok(())
            }
        }
    }

    #[test]
    fn test_config_relative_output_dir_via_env_rejected() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        // An absolute path in the file is overridden by a relative one
        // from the environment; the override must also be rejected.
        let _set_dir = EnvGuard::set("TALK_RS_OUTPUT_DIR", "relative/from/env")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let result = Config::load(Some(file.path()));
        match result {
            Ok(_) => Err("Expected relative output_dir from env to fail".into()),
            Err(err) => {
                assert!(
                    err.to_string().contains("absolute"),
                    "error should mention 'absolute', got: {}",
                    err
                );
                Ok(())
            }
        }
    }

    #[test]
    fn test_config_tilde_output_dir_expanded() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let home = directories::UserDirs::new()
            .map(|d| d.home_dir().to_path_buf())
            .ok_or("home dir unavailable")?;

        // `~/talk-rs-output` (the README minimal example) must expand to
        // an absolute path under $HOME, not be treated as a literal
        // relative directory named `~`.
        let yaml = r#"
output_dir: ~/talk-rs-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert_eq!(config.output_dir, home.join("talk-rs-output"));
        Ok(())
    }

    #[test]
    fn test_config_bare_tilde_output_dir_expanded() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let home = directories::UserDirs::new()
            .map(|d| d.home_dir().to_path_buf())
            .ok_or("home dir unavailable")?;

        // A bare `~` expands to $HOME itself.
        let yaml = r#"
output_dir: ~
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert_eq!(config.output_dir, home);
        Ok(())
    }

    #[test]
    fn test_config_quoted_bare_tilde_output_dir_expanded() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let home = UserDirs::new()
            .ok_or("home unavailable")?
            .home_dir()
            .to_path_buf();
        let file = write_config(indoc! {r#"
            output_dir: "~"
            providers: {}
        "#})?;
        assert_eq!(Config::load(Some(file.path()))?.output_dir, home);
        Ok(())
    }

    #[test]
    fn test_config_tilde_output_dir_via_env_expanded() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let home = directories::UserDirs::new()
            .map(|d| d.home_dir().to_path_buf())
            .ok_or("home dir unavailable")?;
        let _set_dir = EnvGuard::set("TALK_RS_OUTPUT_DIR", "~/from-env")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert_eq!(config.output_dir, home.join("from-env"));
        Ok(())
    }

    #[test]
    fn test_config_non_leading_tilde_not_expanded() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        // A tilde that is not the leading path component must be left
        // untouched.  `/tmp/~/x` is absolute and already valid, so the
        // value must pass through verbatim (no home substitution).
        let yaml = r#"
output_dir: /tmp/~/x
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert_eq!(config.output_dir, PathBuf::from("/tmp/~/x"));
        Ok(())
    }

    #[test]
    fn test_config_openai_provider() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  openai:
    api_key: sk-test-key
    model: gpt-4o-transcribe
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert!(config.providers.mistral.is_none());
        let o = config.providers.openai.as_ref().expect("openai present");
        assert_eq!(o.api_key, "sk-test-key");
        assert_eq!(o.model, "gpt-4o-transcribe");
        Ok(())
    }

    #[test]
    fn openai_omitted_migration_fields_preserve_defaults() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers:
              openai:
                api_key: sk-test-key
        "})?;

        let config = Config::load(Some(file.path()))?;
        let openai = config.providers.openai.as_ref().ok_or("openai missing")?;
        assert_eq!(openai.model, "gpt-transcribe");
        assert_eq!(openai.realtime_model, "gpt-live-transcribe");
        assert!(openai.prompt.is_none());
        assert!(openai.keywords.is_none());
        assert!(openai.languages.is_none());
        assert!(openai.realtime_delay.is_none());
        Ok(())
    }

    #[test]
    fn openai_yaml_parses_migration_hints() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers:
              openai:
                api_key: sk-test-key
                prompt: Preserve punctuation and casing.
                keywords: [Kalysto, talk-rs]
                languages: [fr, en]
                realtime_delay: high
        "})?;

        let config = Config::load(Some(file.path()))?;
        let openai = config.providers.openai.as_ref().ok_or("openai missing")?;
        assert_eq!(
            openai.prompt.as_deref(),
            Some("Preserve punctuation and casing.")
        );
        assert_eq!(
            openai.keywords.as_deref(),
            Some(&["Kalysto".into(), "talk-rs".into()][..])
        );
        assert_eq!(
            openai.languages.as_deref(),
            Some(&["fr".into(), "en".into()][..])
        );
        assert_eq!(openai.realtime_delay, Some(OpenAIRealtimeDelay::High));
        Ok(())
    }

    #[test]
    fn openai_env_hints_override_yaml_with_trimmed_nonempty_lists() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _prompt = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_PROMPT", "env prompt")?;
        let _keywords = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_KEYWORDS", " Kalysto, ,talk-rs,")?;
        let _languages = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_LANGUAGES", " fr, en ,, ")?;
        let _delay = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_REALTIME_DELAY", "xhigh")?;
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers:
              openai:
                api_key: sk-test-key
                prompt: yaml prompt
                keywords: [yaml]
                languages: [de]
                realtime_delay: low
        "})?;

        let config = Config::load(Some(file.path()))?;
        let openai = config.providers.openai.as_ref().ok_or("openai missing")?;
        assert_eq!(openai.prompt.as_deref(), Some("env prompt"));
        assert_eq!(
            openai.keywords.as_deref(),
            Some(&["Kalysto".into(), "talk-rs".into()][..])
        );
        assert_eq!(
            openai.languages.as_deref(),
            Some(&["fr".into(), "en".into()][..])
        );
        assert_eq!(openai.realtime_delay, Some(OpenAIRealtimeDelay::Xhigh));
        Ok(())
    }

    #[test]
    fn openai_env_created_section_carries_migration_hints() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _key = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_API_KEY", "sk-env")?;
        let _prompt = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_PROMPT", "env prompt")?;
        let _keywords = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_KEYWORDS", "one,two")?;
        let _languages = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_LANGUAGES", "fr,en")?;
        let _delay = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_REALTIME_DELAY", "minimal")?;
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers: {}
        "})?;

        let config = Config::load(Some(file.path()))?;
        let openai = config.providers.openai.as_ref().ok_or("openai missing")?;
        assert_eq!(openai.prompt.as_deref(), Some("env prompt"));
        assert_eq!(
            openai.keywords.as_deref(),
            Some(&["one".into(), "two".into()][..])
        );
        assert_eq!(
            openai.languages.as_deref(),
            Some(&["fr".into(), "en".into()][..])
        );
        assert_eq!(openai.realtime_delay, Some(OpenAIRealtimeDelay::Minimal));
        Ok(())
    }

    #[test]
    fn openai_empty_env_lists_clear_existing_yaml_hints() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _keywords = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_KEYWORDS", " , , ")?;
        let _languages = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_LANGUAGES", ",,")?;
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers:
              openai:
                api_key: sk-test-key
                keywords: [yaml]
                languages: [de]
        "})?;

        let config = Config::load(Some(file.path()))?;
        let openai = config.providers.openai.as_ref().ok_or("openai missing")?;
        assert!(openai.keywords.is_none());
        assert!(openai.languages.is_none());
        Ok(())
    }

    #[test]
    fn openai_empty_env_lists_are_none_in_env_created_section() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _key = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_API_KEY", "sk-env")?;
        let _keywords = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_KEYWORDS", " , ")?;
        let _languages = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_LANGUAGES", ",,")?;
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers: {}
        "})?;

        let config = Config::load(Some(file.path()))?;
        let openai = config.providers.openai.as_ref().ok_or("openai missing")?;
        assert!(openai.keywords.is_none());
        assert!(openai.languages.is_none());
        Ok(())
    }

    #[test]
    fn openai_explicit_yaml_empty_languages_remains_configured() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let file = write_config(indoc! {"
            output_dir: /tmp/test-output
            providers:
              openai:
                api_key: sk-test-key
                languages: []
        "})?;

        let config = Config::load(Some(file.path()))?;
        let openai = config.providers.openai.as_ref().ok_or("openai missing")?;
        assert_eq!(openai.languages.as_deref(), Some(&[][..]));
        Ok(())
    }

    #[test]
    fn test_config_openai_env_override() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _set_key = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_API_KEY", "sk-env-key")?;

        // No openai section in YAML — created purely from env vars.
        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let o = config.providers.openai.as_ref().expect("openai from env");
        assert_eq!(o.api_key, "sk-env-key");
        assert_eq!(o.model, "gpt-transcribe");
        assert_eq!(o.realtime_model, "gpt-live-transcribe");
        Ok(())
    }

    #[test]
    fn test_config_both_providers() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: mistral-key
  openai:
    api_key: openai-key
transcription:
  default_provider: openai
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert!(config.providers.mistral.is_some());
        assert!(config.providers.openai.is_some());
        let t = config
            .transcription
            .as_ref()
            .expect("transcription section");
        assert_eq!(t.default_provider, Provider::OpenAI);
        Ok(())
    }

    #[test]
    fn test_config_no_providers() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert!(config.providers.mistral.is_none());
        assert!(config.providers.openai.is_none());
        Ok(())
    }

    #[test]
    fn test_config_paste_section_absent_gives_none() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert!(config.paste.is_none());
        Ok(())
    }

    #[test]
    fn test_config_paste_chunk_chars_default() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
paste: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        assert_eq!(paste.chunk_chars(), 150);
        Ok(())
    }

    #[test]
    fn test_config_paste_chunk_chars_custom() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
paste:
  chunk_chars: 300
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        assert_eq!(paste.chunk_chars(), 300);
        Ok(())
    }

    #[test]
    fn test_config_paste_chunk_chars_zero_disables() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
paste:
  chunk_chars: 0
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        assert_eq!(paste.chunk_chars(), 0);
        Ok(())
    }

    #[test]
    fn test_config_paste_restore_settle_ms_default() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
paste: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        assert_eq!(paste.restore_settle_ms(), 200);
        Ok(())
    }

    #[test]
    fn test_config_paste_chunk_fetch_timeout_ms_default() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
paste: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        // Default widened to 500 (was 300 under Phase 2, 400 under the
        // legacy heuristic) to cover a transient re-focus latency that
        // the retry loop also guards against.  See
        // `crate::paste_config::DEFAULT_CHUNK_FETCH_TIMEOUT_MS`.
        assert_eq!(paste.chunk_fetch_timeout_ms(), 500);
        Ok(())
    }

    #[test]
    fn test_config_paste_target_fetch_retries_default_flat() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        // An empty flat paste section must pick up the retry default.
        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
paste: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        match paste {
            PasteConfig::Flat(f) => {
                assert_eq!(f.target_fetch_retries, 2);
                assert_eq!(f.chunk_fetch_timeout_ms, 500);
            }
            PasteConfig::Tree(_) => panic!("expected flat variant for empty paste section"),
        }
        Ok(())
    }

    /// A tree-form `paste:` section must parse in EVERY feature set,
    /// including headless (`--no-default-features`) builds that embed
    /// talk-rs only for transcription: they share the user's config
    /// file and must not reject it because of a section they never run.
    #[test]
    fn test_config_paste_tree_parses_in_every_feature_set() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = indoc! {r#"
            output_dir: /tmp/test-output
            providers: {}
            paste:
              node: match-wm-class
              patterns:
                - match: "@terminal"
                  child:
                    node: match-foreground-app
                    patterns:
                      - match: opencode-tui
                        child:
                          node: chunk
                          chunk_chars: 150
                          child:
                            node: clipboard
                            shortcut: ctrl_shift_v
                    default:
                      node: clipboard
                      shortcut: ctrl_shift_v
              default:
                node: chunk
                chunk_chars: 90
                child:
                  node: clipboard
                  shortcut: ctrl_v
                  chunk_fetch_timeout_ms: 700
        "#};
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        assert_eq!(paste.chunk_chars(), 90);
        assert_eq!(paste.shortcut(), PasteShortcut::CtrlV);
        assert_eq!(paste.chunk_fetch_timeout_ms(), 700);
        Ok(())
    }

    /// A misspelled key in the flat `paste:` form is rejected, and the
    /// error names the key — a generic "did not match any variant"
    /// message would leave the user no clue which line is wrong.
    #[test]
    fn test_config_paste_flat_unknown_key_error_names_the_key() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = indoc! {"
            output_dir: /tmp/test-output
            providers: {}
            paste:
              chunk_chars: 90
              chunk_fetch_timout_ms: 700
        "};
        let file = write_config(yaml)?;

        let err = match Config::load(Some(file.path())) {
            Ok(_) => return Err("unknown flat paste key must be rejected".into()),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("unknown field `chunk_fetch_timout_ms`"),
            "error must name the key, got: {err}"
        );
        Ok(())
    }

    #[test]
    fn test_config_paste_target_fetch_retries_custom_flat() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
paste:
  target_fetch_retries: 5
  chunk_fetch_timeout_ms: 500
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        match paste {
            PasteConfig::Flat(f) => {
                assert_eq!(f.target_fetch_retries, 5);
                assert_eq!(f.chunk_fetch_timeout_ms, 500);
            }
            PasteConfig::Tree(_) => panic!("expected flat variant"),
        }
        // flat → tree wiring threads the retry count through.
        match paste.to_tree() {
            PasteNodeConfig::Chunk { child, .. } => match *child {
                PasteNodeConfig::Clipboard {
                    target_fetch_retries,
                    chunk_fetch_timeout_ms,
                    ..
                } => {
                    assert_eq!(target_fetch_retries, 5);
                    assert_eq!(chunk_fetch_timeout_ms, 500);
                }
                other => panic!("expected Clipboard child, got {:?}", other),
            },
            other => panic!("expected Chunk root, got {:?}", other),
        }
        Ok(())
    }

    #[test]
    fn test_config_paste_target_fetch_retries_tree() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        // Tree form: explicit knob parses; omitted knob picks default.
        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
paste:
  node: chunk
  chunk_chars: 120
  child:
    node: clipboard
    chunk_fetch_timeout_ms: 500
    target_fetch_retries: 4
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        match paste.to_tree() {
            PasteNodeConfig::Chunk { child, .. } => match *child {
                PasteNodeConfig::Clipboard {
                    target_fetch_retries,
                    chunk_fetch_timeout_ms,
                    ..
                } => {
                    assert_eq!(target_fetch_retries, 4);
                    assert_eq!(chunk_fetch_timeout_ms, 500);
                }
                other => panic!("expected Clipboard child, got {:?}", other),
            },
            other => panic!("expected Chunk root, got {:?}", other),
        }
        Ok(())
    }

    #[test]
    fn test_config_paste_target_fetch_retries_tree_default_when_absent(
    ) -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
paste:
  node: clipboard
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        match paste.to_tree() {
            PasteNodeConfig::Clipboard {
                target_fetch_retries,
                chunk_fetch_timeout_ms,
                ..
            } => {
                assert_eq!(target_fetch_retries, 2);
                assert_eq!(chunk_fetch_timeout_ms, 500);
            }
            other => panic!("expected Clipboard root, got {:?}", other),
        }
        Ok(())
    }

    #[test]
    fn test_config_paste_chunk_fetch_timeout_ms_custom() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
paste:
  chunk_fetch_timeout_ms: 800
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let paste = config.paste.as_ref().expect("paste section present");
        assert_eq!(paste.chunk_fetch_timeout_ms(), 800);
        Ok(())
    }

    #[test]
    fn test_recording_defaults_when_absent() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        // No `recording:` section → fidelity-oriented defaults:
        // 48 kHz, mono, 128 kbps.  This is the human-facing profile,
        // explicitly distinct from the 16 kHz transcription profile.
        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let resolved = config.recording.clone().unwrap_or_default().resolved();
        assert_eq!(resolved.sample_rate, 48_000);
        assert_eq!(resolved.channels, 1);
        assert_eq!(resolved.bitrate, 128_000);
        Ok(())
    }

    #[test]
    fn test_recording_parsed_from_yaml() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
recording:
  sample_rate: 24000
  channels: 2
  bitrate: 96000
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let rec = config
            .recording
            .as_ref()
            .expect("recording section present");
        let resolved = rec.resolved();
        assert_eq!(resolved.sample_rate, 24_000);
        assert_eq!(resolved.channels, 2);
        assert_eq!(resolved.bitrate, 96_000);
        Ok(())
    }

    #[test]
    fn test_recording_partial_yaml_fills_defaults() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        // Only `channels` set → the other two fields fall back to
        // their defaults.
        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
recording:
  channels: 2
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let resolved = config.recording.as_ref().expect("recording").resolved();
        assert_eq!(resolved.sample_rate, 48_000); // default
        assert_eq!(resolved.channels, 2); // from yaml
        assert_eq!(resolved.bitrate, 128_000); // default
        Ok(())
    }

    #[test]
    fn test_recording_env_overrides() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _r = EnvGuard::set("TALK_RS_RECORDING_SAMPLE_RATE", "32000")?;
        let _c = EnvGuard::set("TALK_RS_RECORDING_CHANNELS", "2")?;
        let _b = EnvGuard::set("TALK_RS_RECORDING_BITRATE", "64000")?;

        // The env vars must override even when no `recording:` section
        // exists in the file (the section is created on demand).
        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let resolved = config.recording.as_ref().expect("recording").resolved();
        assert_eq!(resolved.sample_rate, 32_000);
        assert_eq!(resolved.channels, 2);
        assert_eq!(resolved.bitrate, 64_000);
        Ok(())
    }

    #[test]
    fn test_recording_env_invalid_channels_rejected() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _c = EnvGuard::set("TALK_RS_RECORDING_CHANNELS", "not-a-number")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        match Config::load(Some(file.path())) {
            Ok(_) => Err("expected invalid channels env to fail".into()),
            Err(err) => {
                assert!(
                    err.to_string().contains("TALK_RS_RECORDING_CHANNELS"),
                    "error should name the variable, got: {}",
                    err
                );
                Ok(())
            }
        }
    }

    #[test]
    fn test_config_url_defaults_to_none() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
  openai:
    api_key: sk-test-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let m = config.providers.mistral.as_ref().expect("mistral present");
        assert!(m.url.is_none());
        let o = config.providers.openai.as_ref().expect("openai present");
        assert!(o.url.is_none());
        Ok(())
    }

    #[test]
    fn test_config_url_from_yaml() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
    url: https://custom-mistral.example.com
  openai:
    api_key: sk-test-key
    url: https://custom-openai.example.com
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let m = config.providers.mistral.as_ref().expect("mistral present");
        assert_eq!(m.url.as_deref(), Some("https://custom-mistral.example.com"));
        let o = config.providers.openai.as_ref().expect("openai present");
        assert_eq!(o.url.as_deref(), Some("https://custom-openai.example.com"));
        Ok(())
    }

    #[test]
    fn test_config_url_env_override() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _set_url = EnvGuard::set(
            "TALK_RS_PROVIDERS_MISTRAL_URL",
            "https://env-mistral.example.com",
        )?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let m = config.providers.mistral.as_ref().expect("mistral present");
        assert_eq!(m.url.as_deref(), Some("https://env-mistral.example.com"));
        Ok(())
    }

    // ── Parakeet provider (Phase 1: config only) ───────────────────

    #[test]
    fn test_parakeet_section_parsed_from_yaml() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  parakeet:
    variant: fp32
    num_threads: 4
    model: my-parakeet-build
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let p = config
            .providers
            .parakeet
            .as_ref()
            .ok_or("parakeet section present")?;
        assert_eq!(p.variant, ParakeetVariant::Fp32);
        assert_eq!(p.num_threads, 4);
        assert_eq!(p.model.as_deref(), Some("my-parakeet-build"));
        assert!(p.model_dir.is_none());
        Ok(())
    }

    #[test]
    fn test_parakeet_section_absent_gives_none() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert!(config.providers.parakeet.is_none());
        Ok(())
    }

    #[test]
    fn test_parakeet_defaults_with_empty_block() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        // `parakeet: {}` materialises the section with all defaults:
        // int8 variant, 2 decode threads, no explicit model_dir/model.
        let yaml = r#"
output_dir: /tmp/test-output
providers:
  parakeet: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let p = config
            .providers
            .parakeet
            .as_ref()
            .ok_or("parakeet present")?;
        assert_eq!(p.variant, ParakeetVariant::Int8);
        assert_eq!(p.num_threads, 2);
        assert!(p.model.is_none());
        assert!(p.model_dir.is_none());
        assert_eq!(p.resolved_variant(), ParakeetVariant::Int8);
        assert_eq!(p.resolved_model_name(), "parakeet-tdt-0.6b-v3-int8");
        let dir = p.resolved_model_dir()?;
        let s = dir.to_string_lossy();
        assert!(
            s.ends_with("models/parakeet-tdt-0.6b-v3-int8"),
            "expected default model_dir to end with models/parakeet-tdt-0.6b-v3-int8, got: {}",
            s
        );
        Ok(())
    }

    #[test]
    fn test_parakeet_resolved_model_dir_user_override() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  parakeet:
    model_dir: /opt/models/my-parakeet
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let p = config
            .providers
            .parakeet
            .as_ref()
            .ok_or("parakeet present")?;
        assert_eq!(
            p.resolved_model_dir()?,
            PathBuf::from("/opt/models/my-parakeet")
        );
        // resolved_model_name still derives from the variant when the
        // explicit `model` field is unset, even when model_dir is.
        assert_eq!(p.resolved_model_name(), "parakeet-tdt-0.6b-v3-int8");
        Ok(())
    }

    #[test]
    fn test_parakeet_default_provider_resolves() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  parakeet: {}
transcription:
  default_provider: parakeet
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let t = config
            .transcription
            .as_ref()
            .ok_or("transcription section")?;
        assert_eq!(t.default_provider, Provider::Parakeet);
        Ok(())
    }

    #[test]
    fn test_parakeet_env_overrides_existing_section() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _v = EnvGuard::set("TALK_RS_PROVIDERS_PARAKEET_VARIANT", "fp32")?;
        let _t = EnvGuard::set("TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS", "4")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  parakeet:
    variant: int8
    num_threads: 2
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let p = config
            .providers
            .parakeet
            .as_ref()
            .ok_or("parakeet present")?;
        assert_eq!(p.variant, ParakeetVariant::Fp32);
        assert_eq!(p.num_threads, 4);
        assert_eq!(p.resolved_model_name(), "parakeet-tdt-0.6b-v3-fp32");
        let dir = p.resolved_model_dir()?;
        assert!(dir
            .to_string_lossy()
            .ends_with("models/parakeet-tdt-0.6b-v3-fp32"));
        Ok(())
    }

    #[test]
    fn test_parakeet_env_creates_section_from_scratch() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _d = EnvGuard::set("TALK_RS_PROVIDERS_PARAKEET_MODEL_DIR", "/tmp/from-env")?;

        // No parakeet section in YAML — created purely from env vars.
        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let p = config
            .providers
            .parakeet
            .as_ref()
            .ok_or("parakeet section from env")?;
        assert_eq!(p.variant, ParakeetVariant::Int8); // default
        assert_eq!(p.num_threads, 2); // default
        assert_eq!(p.model_dir.as_deref(), Some(Path::new("/tmp/from-env")));
        Ok(())
    }

    #[test]
    fn test_parakeet_env_invalid_variant_rejected() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _v = EnvGuard::set("TALK_RS_PROVIDERS_PARAKEET_VARIANT", "bogus")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  parakeet: {}
"#;
        let file = write_config(yaml)?;

        match Config::load(Some(file.path())) {
            Ok(_) => Err("expected invalid variant env to fail".into()),
            Err(err) => {
                assert!(
                    err.to_string().contains("bogus"),
                    "error should name the bad value, got: {}",
                    err
                );
                Ok(())
            }
        }
    }

    #[test]
    fn test_parakeet_env_invalid_num_threads_rejected() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _n = EnvGuard::set("TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS", "not-a-number")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  parakeet: {}
"#;
        let file = write_config(yaml)?;

        match Config::load(Some(file.path())) {
            Ok(_) => Err("expected invalid num_threads env to fail".into()),
            Err(err) => {
                assert!(
                    err.to_string()
                        .contains("TALK_RS_PROVIDERS_PARAKEET_NUM_THREADS"),
                    "error should name the variable, got: {}",
                    err
                );
                Ok(())
            }
        }
    }

    #[test]
    fn test_provider_from_str_includes_parakeet() {
        assert_eq!(
            "parakeet".parse::<Provider>().unwrap_or(Provider::Mistral),
            Provider::Parakeet
        );
        assert_eq!(
            "PARAKEET".parse::<Provider>().unwrap_or(Provider::Mistral),
            Provider::Parakeet
        );
        // Existing providers still parse.
        assert_eq!(
            "mistral".parse::<Provider>().unwrap_or(Provider::OpenAI),
            Provider::Mistral
        );
        assert_eq!(
            "openai".parse::<Provider>().unwrap_or(Provider::Mistral),
            Provider::OpenAI
        );
        // Unknown still errors AND mentions all three providers.
        let err = "bogus".parse::<Provider>().err().unwrap_or_default();
        assert!(err.contains("mistral"), "err missing mistral: {}", err);
        assert!(err.contains("openai"), "err missing openai: {}", err);
        assert!(err.contains("parakeet"), "err missing parakeet: {}", err);
    }

    #[test]
    fn test_provider_display_includes_parakeet() {
        assert_eq!(Provider::Mistral.to_string(), "mistral");
        assert_eq!(Provider::OpenAI.to_string(), "openai");
        assert_eq!(Provider::Parakeet.to_string(), "parakeet");
    }

    #[test]
    fn test_parakeet_variant_from_str_and_display() {
        assert_eq!(
            "int8"
                .parse::<ParakeetVariant>()
                .unwrap_or(ParakeetVariant::Fp32),
            ParakeetVariant::Int8
        );
        assert_eq!(
            "INT8"
                .parse::<ParakeetVariant>()
                .unwrap_or(ParakeetVariant::Fp32),
            ParakeetVariant::Int8
        );
        assert_eq!(
            "fp32"
                .parse::<ParakeetVariant>()
                .unwrap_or(ParakeetVariant::Int8),
            ParakeetVariant::Fp32
        );
        assert_eq!(ParakeetVariant::Int8.to_string(), "int8");
        assert_eq!(ParakeetVariant::Fp32.to_string(), "fp32");
        // Round-trip.
        for v in [ParakeetVariant::Int8, ParakeetVariant::Fp32] {
            let s = v.to_string();
            let parsed = s.parse::<ParakeetVariant>().unwrap_or_else(|_| {
                // Should never happen; choose opposite so assert fails loudly.
                if v == ParakeetVariant::Int8 {
                    ParakeetVariant::Fp32
                } else {
                    ParakeetVariant::Int8
                }
            });
            assert_eq!(parsed, v);
        }
        // Invalid value errors with a helpful message.
        let err = "bogus".parse::<ParakeetVariant>().err().unwrap_or_default();
        assert!(err.contains("bogus"), "err missing 'bogus': {}", err);
        assert!(err.contains("int8"), "err missing 'int8': {}", err);
        assert!(err.contains("fp32"), "err missing 'fp32': {}", err);
    }

    #[test]
    fn test_config_url_env_override_openai() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _set_key = EnvGuard::set("TALK_RS_PROVIDERS_OPENAI_API_KEY", "sk-env-key")?;
        let _set_url = EnvGuard::set(
            "TALK_RS_PROVIDERS_OPENAI_URL",
            "https://env-openai.example.com",
        )?;

        // No openai section in YAML — created purely from env vars.
        let yaml = r#"
output_dir: /tmp/test-output
providers: {}
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        let o = config.providers.openai.as_ref().expect("openai from env");
        assert_eq!(o.api_key, "sk-env-key");
        assert_eq!(o.url.as_deref(), Some("https://env-openai.example.com"));
        Ok(())
    }

    // ── Synthesis / TTS config ──────────────────────────────────────

    #[test]
    fn test_config_mistral_tts_defaults() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;
        let config = Config::load(Some(file.path()))?;
        let m = config.providers.mistral.as_ref().expect("mistral present");
        // TTS model defaults; TTS voice stays unset.
        assert_eq!(m.tts_model, "voxtral-mini-tts-latest");
        assert!(m.tts_voice.is_none());
        Ok(())
    }

    #[test]
    fn test_config_mistral_tts_fields_parse() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
    tts_model: voxtral-mini-tts-2026
    tts_voice: 5a271406-039d-46fe-835b-fbbb00eaf08d
"#;
        let file = write_config(yaml)?;
        let config = Config::load(Some(file.path()))?;
        let m = config.providers.mistral.as_ref().expect("mistral present");
        assert_eq!(m.tts_model, "voxtral-mini-tts-2026");
        assert_eq!(
            m.tts_voice.as_deref(),
            Some("5a271406-039d-46fe-835b-fbbb00eaf08d")
        );
        Ok(())
    }

    #[test]
    fn test_config_mistral_tts_voices_map_parses() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
    tts_voices:
      fr: 5a271406-039d-46fe-835b-fbbb00eaf08d
      en: c69964a6-ab8b-4f8a-9465-ec0925096ec8
"#;
        let file = write_config(yaml)?;
        let config = Config::load(Some(file.path()))?;
        let m = config.providers.mistral.as_ref().expect("mistral present");
        // Pin unset; the per-language map is present.
        assert!(m.tts_voice.is_none());
        let map = m.tts_voices.as_ref().expect("tts_voices present");
        assert_eq!(
            map.get("fr").map(String::as_str),
            Some("5a271406-039d-46fe-835b-fbbb00eaf08d")
        );
        assert_eq!(
            map.get("en").map(String::as_str),
            Some("c69964a6-ab8b-4f8a-9465-ec0925096ec8")
        );
        Ok(())
    }

    #[test]
    fn test_config_kokoro_lang_auto_parses() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
  kokoro:
    lang: auto
"#;
        let file = write_config(yaml)?;
        let config = Config::load(Some(file.path()))?;
        let k = config.providers.kokoro.as_ref().expect("kokoro present");
        // `auto` is stored verbatim; the speak-command resolution layer
        // interprets it as "auto-detect from text".
        assert_eq!(k.lang.as_deref(), Some("auto"));
        Ok(())
    }

    #[test]
    fn test_config_mistral_tts_env_override() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _m = EnvGuard::set("TALK_RS_PROVIDERS_MISTRAL_TTS_MODEL", "env-tts-model")?;
        let _v = EnvGuard::set("TALK_RS_PROVIDERS_MISTRAL_TTS_VOICE", "env-voice-id")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;
        let config = Config::load(Some(file.path()))?;
        let m = config.providers.mistral.as_ref().expect("mistral present");
        assert_eq!(m.tts_model, "env-tts-model");
        assert_eq!(m.tts_voice.as_deref(), Some("env-voice-id"));
        Ok(())
    }

    #[test]
    fn test_config_kokoro_section_parses() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
  kokoro:
    voice: am_michael
    num_threads: 8
    lang: fr
"#;
        let file = write_config(yaml)?;
        let config = Config::load(Some(file.path()))?;
        let k = config.providers.kokoro.as_ref().expect("kokoro present");
        assert_eq!(k.voice.as_deref(), Some("am_michael"));
        assert_eq!(k.resolved_num_threads(), 8);
        assert_eq!(k.lang.as_deref(), Some("fr"));
        // Model dir defaults to the XDG data dir with the tarball name.
        let dir = k.resolved_model_dir()?;
        assert!(
            dir.ends_with("models/kokoro-multi-lang-v1_0"),
            "unexpected default model dir: {}",
            dir.display()
        );
        Ok(())
    }

    #[test]
    fn test_config_kokoro_defaults() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
  kokoro: {}
"#;
        let file = write_config(yaml)?;
        let config = Config::load(Some(file.path()))?;
        let k = config.providers.kokoro.as_ref().expect("kokoro present");
        // Threads default to 4 when unset.
        assert_eq!(k.resolved_num_threads(), 4);
        assert!(k.voice.is_none());
        Ok(())
    }

    #[test]
    fn test_config_kokoro_env_creates_section() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;
        let _v = EnvGuard::set("TALK_RS_PROVIDERS_KOKORO_VOICE", "af_heart")?;
        let _t = EnvGuard::set("TALK_RS_PROVIDERS_KOKORO_NUM_THREADS", "2")?;

        // No kokoro section in YAML — created purely from env vars.
        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
"#;
        let file = write_config(yaml)?;
        let config = Config::load(Some(file.path()))?;
        let k = config.providers.kokoro.as_ref().expect("kokoro from env");
        assert_eq!(k.voice.as_deref(), Some("af_heart"));
        assert_eq!(k.resolved_num_threads(), 2);
        Ok(())
    }

    #[test]
    fn test_config_speak_default_provider_parses() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guards = clear_all_provider_env_vars()?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
speak:
  default_provider: kokoro
"#;
        let file = write_config(yaml)?;
        let config = Config::load(Some(file.path()))?;
        let speak = config.speak.as_ref().expect("speak section present");
        assert_eq!(speak.default_provider, Some(SynthesisProvider::Kokoro));
        Ok(())
    }

    #[test]
    fn test_synthesis_provider_from_str_and_display() {
        assert_eq!(
            "kokoro"
                .parse::<SynthesisProvider>()
                .unwrap_or(SynthesisProvider::Mistral),
            SynthesisProvider::Kokoro
        );
        assert_eq!(
            "MISTRAL"
                .parse::<SynthesisProvider>()
                .unwrap_or(SynthesisProvider::Kokoro),
            SynthesisProvider::Mistral
        );
        assert!("bogus".parse::<SynthesisProvider>().is_err());
        assert_eq!(SynthesisProvider::Kokoro.to_string(), "kokoro");
        assert_eq!(SynthesisProvider::Mistral.to_string(), "mistral");
    }
}
