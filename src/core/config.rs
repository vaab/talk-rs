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
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Output directory for recordings, screenshots, and clipboard saves.
    pub output_dir: PathBuf,

    /// Transcription providers configuration.
    pub providers: ProvidersConfig,

    /// Audio capture and encoding settings.
    pub audio: AudioConfig,

    /// Optional dictation mode settings.
    pub dictate: Option<DictateConfig>,

    /// Optional indicator settings.
    pub indicators: Option<IndicatorsConfig>,
}

/// Transcription providers configuration.
#[derive(Debug, Deserialize)]
pub struct ProvidersConfig {
    /// Mistral API configuration.
    pub mistral: MistralConfig,
}

/// Mistral API configuration.
#[derive(Debug, Deserialize)]
pub struct MistralConfig {
    /// API key for Mistral transcription service.
    pub api_key: String,
}

/// Audio configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct AudioConfig {
    /// Sample rate in Hz (e.g., 16000).
    pub sample_rate: u32,

    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u8,

    /// Bitrate in bps for compressed formats.
    pub bitrate: u32,
}

/// Dictation mode configuration.
#[derive(Debug, Deserialize)]
pub struct DictateConfig {
    /// Chunk size in seconds for chunked mode.
    pub chunk_seconds: u64,
}

/// Indicator configuration.
#[derive(Debug, Deserialize)]
pub struct IndicatorsConfig {
    /// Interval between boop sounds in milliseconds.
    pub boop_interval_ms: u64,

    /// Show visual indicator overlay.
    pub visual_overlay: bool,
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
    /// - TALK_RS_AUDIO_SAMPLE_RATE
    /// - TALK_RS_AUDIO_CHANNELS
    /// - TALK_RS_AUDIO_BITRATE
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

        if let Some(value) = env_var_string("TALK_RS_PROVIDERS_MISTRAL_API_KEY")? {
            config.providers.mistral.api_key = value;
        }

        if let Some(value) = env_var_u32("TALK_RS_AUDIO_SAMPLE_RATE")? {
            config.audio.sample_rate = value;
        }

        if let Some(value) = env_var_u8("TALK_RS_AUDIO_CHANNELS")? {
            config.audio.channels = value;
        }

        if let Some(value) = env_var_u32("TALK_RS_AUDIO_BITRATE")? {
            config.audio.bitrate = value;
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

fn env_var_u32(key: &str) -> Result<Option<u32>, TalkError> {
    match env_var_string(key)? {
        Some(value) => {
            let parsed = value
                .parse::<u32>()
                .map_err(|err| TalkError::Config(format!("Invalid {}: {}", key, err)))?;
            Ok(Some(parsed))
        }
        None => Ok(None),
    }
}

fn env_var_u8(key: &str) -> Result<Option<u8>, TalkError> {
    match env_var_string(key)? {
        Some(value) => {
            let parsed = value
                .parse::<u8>()
                .map_err(|err| TalkError::Config(format!("Invalid {}: {}", key, err)))?;
            Ok(Some(parsed))
        }
        None => Ok(None),
    }
}

fn validate_config(config: &Config) -> Result<(), TalkError> {
    if config.output_dir.as_os_str().is_empty() {
        return Err(TalkError::Config("output_dir is required".to_string()));
    }

    if config.providers.mistral.api_key.is_empty() {
        return Err(TalkError::Config(
            "providers.mistral.api_key is required".to_string(),
        ));
    }

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

    #[test]
    fn test_config_load_valid() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guard_output_dir = EnvGuard::clear("TALK_RS_OUTPUT_DIR")?;
        let _guard_api_key = EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_API_KEY")?;
        let _guard_sample_rate = EnvGuard::clear("TALK_RS_AUDIO_SAMPLE_RATE")?;
        let _guard_channels = EnvGuard::clear("TALK_RS_AUDIO_CHANNELS")?;
        let _guard_bitrate = EnvGuard::clear("TALK_RS_AUDIO_BITRATE")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
audio:
  sample_rate: 16000
  channels: 1
  bitrate: 32000
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert_eq!(config.output_dir, PathBuf::from("/tmp/test-output"));
        assert_eq!(config.providers.mistral.api_key, "test-api-key");
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.audio.channels, 1);
        assert_eq!(config.audio.bitrate, 32000);
        Ok(())
    }

    #[test]
    fn test_config_env_override() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guard_output_dir = EnvGuard::clear("TALK_RS_OUTPUT_DIR")?;
        let _guard_sample_rate = EnvGuard::clear("TALK_RS_AUDIO_SAMPLE_RATE")?;
        let _guard_channels = EnvGuard::clear("TALK_RS_AUDIO_CHANNELS")?;
        let _guard_bitrate = EnvGuard::clear("TALK_RS_AUDIO_BITRATE")?;
        let _guard_api_key = EnvGuard::set("TALK_RS_PROVIDERS_MISTRAL_API_KEY", "override-key")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test-api-key
audio:
  sample_rate: 16000
  channels: 1
  bitrate: 32000
"#;
        let file = write_config(yaml)?;

        let config = Config::load(Some(file.path()))?;
        assert_eq!(config.providers.mistral.api_key, "override-key");
        Ok(())
    }

    #[test]
    fn test_config_missing_required_field() -> Result<(), Box<dyn Error>> {
        let _lock = env_lock()?;
        let _guard_output_dir = EnvGuard::clear("TALK_RS_OUTPUT_DIR")?;
        let _guard_api_key = EnvGuard::clear("TALK_RS_PROVIDERS_MISTRAL_API_KEY")?;
        let _guard_sample_rate = EnvGuard::clear("TALK_RS_AUDIO_SAMPLE_RATE")?;
        let _guard_channels = EnvGuard::clear("TALK_RS_AUDIO_CHANNELS")?;
        let _guard_bitrate = EnvGuard::clear("TALK_RS_AUDIO_BITRATE")?;

        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: ""
audio:
  sample_rate: 16000
  channels: 1
  bitrate: 32000
"#;
        let file = write_config(yaml)?;

        let result = Config::load(Some(file.path()));
        match result {
            Ok(_) => Err("Expected missing api_key to fail".into()),
            Err(err) => {
                assert!(err
                    .to_string()
                    .contains("providers.mistral.api_key is required"));
                Ok(())
            }
        }
    }
}
