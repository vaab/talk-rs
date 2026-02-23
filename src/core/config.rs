//! Configuration file handling for talk-rs.
//!
//! Config is stored in ~/.config/talk-rs/config.yaml

use crate::core::error::TalkError;
use directories::ProjectDirs;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Main configuration structure.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Output directory for recordings, screenshots, and clipboard saves.
    ///
    /// The `record` command saves files here by default when no explicit
    /// output path is given.
    pub output_dir: PathBuf,

    /// Transcription providers configuration.
    pub providers: ProvidersConfig,

    /// Optional indicator settings.
    pub indicators: Option<IndicatorsConfig>,

    /// Optional transcription defaults (provider selection, etc.).
    pub transcription: Option<TranscriptionConfig>,

    /// Optional paste behaviour settings.
    #[serde(default)]
    pub paste: Option<PasteConfig>,
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
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Mistral => write!(f, "mistral"),
            Provider::OpenAI => write!(f, "openai"),
        }
    }
}

impl std::str::FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mistral" => Ok(Provider::Mistral),
            "openai" => Ok(Provider::OpenAI),
            other => Err(format!(
                "unknown provider '{}' (expected 'mistral' or 'openai')",
                other
            )),
        }
    }
}

/// Transcription providers configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ProvidersConfig {
    /// Mistral API configuration (optional — only required when using Mistral).
    #[serde(default)]
    pub mistral: Option<MistralConfig>,

    /// OpenAI API configuration (optional — only required when using OpenAI).
    #[serde(default)]
    pub openai: Option<OpenAIConfig>,
}

/// Mistral API configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct MistralConfig {
    /// API key for Mistral transcription service.
    pub api_key: String,

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
}

fn default_mistral_model() -> String {
    "voxtral-mini-2507".to_string()
}

/// OpenAI API configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIConfig {
    /// API key for OpenAI transcription service.
    pub api_key: String,

    /// Model name for batch transcription (defaults to "whisper-1").
    #[serde(default = "default_openai_model")]
    pub model: String,

    /// Model name for realtime transcription (defaults to "gpt-4o-mini-transcribe").
    #[serde(default = "default_openai_realtime_model")]
    pub realtime_model: String,
}

fn default_openai_model() -> String {
    "whisper-1".to_string()
}

fn default_openai_realtime_model() -> String {
    "gpt-4o-mini-transcribe".to_string()
}

/// Transcription defaults.
#[derive(Debug, Deserialize, Clone)]
pub struct TranscriptionConfig {
    /// Default transcription provider when `--provider` is not specified.
    #[serde(default = "default_provider")]
    pub default_provider: Provider,
}

fn default_provider() -> Provider {
    Provider::Mistral
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

/// Indicator configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct IndicatorsConfig {
    /// Interval between boop sounds in milliseconds.
    pub boop_interval_ms: u64,

    /// Show visual indicator overlay.
    pub visual_overlay: bool,
}

/// Paste behaviour configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct PasteConfig {
    /// Maximum characters per clipboard paste chunk.
    ///
    /// Text longer than this is split on word boundaries into
    /// consecutive Ctrl+Shift+V keystrokes.  Keeping chunks small
    /// avoids terminal paste-summary behaviour that collapses large
    /// pastes.  Set to `0` to disable chunking entirely (paste in one
    /// shot).
    #[serde(default = "default_paste_chunk_chars")]
    pub chunk_chars: usize,
}

fn default_paste_chunk_chars() -> usize {
    150
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
    /// Load configuration from file.
    ///
    /// If `path` is provided, loads from that path.
    /// Otherwise, loads from the default XDG config location.
    ///
    /// Environment variables can override config values:
    /// - TALK_RS_OUTPUT_DIR
    /// - TALK_RS_PROVIDERS_MISTRAL_API_KEY
    /// - TALK_RS_PROVIDERS_MISTRAL_MODEL
    /// - TALK_RS_PROVIDERS_MISTRAL_CONTEXT_BIAS
    /// - TALK_RS_PROVIDERS_OPENAI_API_KEY
    /// - TALK_RS_PROVIDERS_OPENAI_MODEL
    /// - TALK_RS_PROVIDERS_OPENAI_REALTIME_MODEL
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

        // Mistral env var overrides (only when the section exists).
        if let Some(ref mut mistral) = config.providers.mistral {
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_API_KEY")? {
                mistral.api_key = value;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_MODEL")? {
                mistral.model = value;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_CONTEXT_BIAS")? {
                mistral.context_bias = Some(value);
            }
        } else {
            // Allow creating the Mistral section purely from env vars.
            if let Some(api_key) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_API_KEY")? {
                config.providers.mistral = Some(MistralConfig {
                    api_key,
                    model: env_var_string("TALK_RS_PROVIDERS_MISTRAL_MODEL")?
                        .unwrap_or_else(default_mistral_model),
                    context_bias: env_var_string("TALK_RS_PROVIDERS_MISTRAL_CONTEXT_BIAS")?,
                });
            }
        }

        // OpenAI env var overrides (only when the section exists).
        if let Some(ref mut openai) = config.providers.openai {
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_API_KEY")? {
                openai.api_key = value;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_MODEL")? {
                openai.model = value;
            }
            if let Some(value) = env_var_string("TALK_RS_PROVIDERS_OPENAI_REALTIME_MODEL")? {
                openai.realtime_model = value;
            }
        } else {
            // Allow creating the OpenAI section purely from env vars.
            if let Some(api_key) = env_var_string("TALK_RS_PROVIDERS_OPENAI_API_KEY")? {
                config.providers.openai = Some(OpenAIConfig {
                    api_key,
                    model: env_var_string("TALK_RS_PROVIDERS_OPENAI_MODEL")?
                        .unwrap_or_else(default_openai_model),
                    realtime_model: env_var_string("TALK_RS_PROVIDERS_OPENAI_REALTIME_MODEL")?
                        .unwrap_or_else(default_openai_realtime_model),
                });
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

fn validate_config(config: &Config) -> Result<(), TalkError> {
    if config.output_dir.as_os_str().is_empty() {
        return Err(TalkError::Config("output_dir is required".to_string()));
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
    use std::error::Error;
    use std::ffi::OsString;
    use std::io::Write;
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use tempfile::NamedTempFile;

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

    /// Clear all provider-related env vars to prevent cross-test leakage.
    fn clear_all_provider_env_vars() -> Result<Vec<EnvGuard>, Box<dyn Error>> {
        Ok(vec![
            EnvGuard::clear("TALK_RS_OUTPUT_DIR")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_API_KEY")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_MODEL")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_CONTEXT_BIAS")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_API_KEY")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_MODEL")?,
            EnvGuard::clear("TALK_RS_PROVIDERS_OPENAI_REALTIME_MODEL")?,
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
        assert_eq!(o.model, "whisper-1"); // default
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
        assert_eq!(paste.chunk_chars, 150);
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
        assert_eq!(paste.chunk_chars, 300);
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
        assert_eq!(paste.chunk_chars, 0);
        Ok(())
    }
}
