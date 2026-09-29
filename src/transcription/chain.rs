//! Resolve eligible entries and walk providers after transient failures.

use crate::config::{ChainEntry, Config, Provider, ResolvedChain};
use crate::error::TalkError;
use crate::telemetry::TelemetrySink;
use crate::transcription::{
    catalog, transcribe_audio, transport::RetrySchedule, RequestTimeoutPolicy, TranscribeOptions,
    TranscriptionResult,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use super::outage::OutageMemory;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Attempt {
    pub provider: String,
    pub model: String,
    pub outcome: String,
}

impl Attempt {
    fn new(entry: &ChainEntry, outcome: &str) -> Self {
        Self {
            provider: entry.provider.to_string(),
            model: entry.model.clone(),
            outcome: outcome.to_string(),
        }
    }
}

#[derive(Debug)]
pub struct ChainOutcome {
    pub result: TranscriptionResult,
    pub provider: Provider,
    pub model: String,
    pub attempts: Vec<Attempt>,
}

impl ChainEntry {
    pub fn retry_schedule(&self) -> RetrySchedule {
        let mut schedule = RetrySchedule::default();
        let count = self.retries.map(|n| n as usize).unwrap_or_else(|| {
            if self.wait.is_some() {
                1000
            } else {
                0
            }
        });
        schedule
            .connection_budgets
            .truncate(count.saturating_add(1));
        if count == 0 {
            schedule.connection_budgets.truncate(1);
        }
        if count > schedule.connection_budgets.len().saturating_sub(1) {
            schedule
                .connection_budgets
                .resize(count.saturating_add(1), Duration::from_secs(120));
        }
        schedule.data_backoffs = (0..count)
            .map(|index| {
                Duration::from_secs(
                    super::transport::DATA_BACKOFF_SECS
                        .get(index)
                        .copied()
                        .unwrap_or(120),
                )
            })
            .collect();
        schedule.max_data_wait = self.wait;
        schedule.wait_started = self
            .wait
            .map(|_| std::sync::Arc::new(std::sync::OnceLock::new()));
        schedule.retry_ws_busy = count > 0;
        schedule
    }
}

impl ResolvedChain {
    /// Reject an impossible request before capture. Restrictions are hard gates,
    /// not provider preferences, and do not depend on outage state.
    pub fn eligible(
        &self,
        diarize: bool,
        realtime: bool,
        lang: Option<&str>,
    ) -> Result<Self, TalkError> {
        let mut kept = Vec::new();
        let mut skipped = Vec::new();
        for entry in &self.entries {
            let mut resolved = entry.clone();
            if realtime {
                if let Some(model) = &entry.realtime_model {
                    resolved.model = model.clone();
                }
            }
            let reason = if diarize
                && !catalog::supports(
                    resolved.provider,
                    &resolved.model,
                    "diarize",
                    &resolved.supports,
                ) {
                Some("no diarization".to_string())
            } else if realtime
                && !catalog::supports(
                    resolved.provider,
                    &resolved.model,
                    "realtime",
                    &resolved.supports,
                )
            {
                Some("no realtime".to_string())
            } else if let Some(languages) = &resolved.languages {
                if lang.is_none_or(|lang| !languages.iter().any(|value| value == lang)) {
                    Some(format!(
                        "language {} not in [{}]",
                        lang.unwrap_or("unspecified"),
                        languages.join(", ")
                    ))
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(reason) = reason {
                log::info!(
                    "chain \"{}\": skipping {}: {}",
                    self.name,
                    resolved.label(),
                    reason
                );
                skipped.push(format!("{}: {}", resolved.label(), reason));
            } else {
                kept.push(resolved);
            }
        }
        if kept.is_empty() {
            let requirement = if diarize {
                "--diarize"
            } else if realtime {
                "--realtime"
            } else if lang.is_some() {
                "--lang"
            } else {
                "the requested options"
            };
            return Err(TalkError::Config(format!(
                "chain \"{}\" has no entry supporting {} (skipped: {})",
                self.name,
                requirement,
                skipped.join(", ")
            )));
        }
        Ok(Self {
            name: self.name.clone(),
            entries: kept,
            outage_memory: self.outage_memory,
        })
    }

    fn ordered_indices(&self, memory: &OutageMemory, start: usize) -> Vec<usize> {
        let candidates: Vec<_> = (start..self.entries.len()).collect();
        let active: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|index| !memory.is_busy(self.entries[*index].provider))
            .collect();
        if active.is_empty() {
            candidates
        } else {
            for index in candidates
                .iter()
                .filter(|index| memory.is_busy(self.entries[**index].provider))
            {
                log::info!(
                    "chain \"{}\": skipping {}: provider in outage memory",
                    self.name,
                    self.entries[*index].label()
                );
            }
            active
        }
    }

    pub fn first_available(&self, outage_path: PathBuf) -> Option<&ChainEntry> {
        let memory = OutageMemory::new(outage_path);
        self.ordered_indices(&memory, 0)
            .first()
            .map(|index| &self.entries[*index])
    }

    pub fn available_entries(&self, outage_path: PathBuf) -> Vec<&ChainEntry> {
        let memory = OutageMemory::new(outage_path);
        self.ordered_indices(&memory, 0)
            .into_iter()
            .map(|index| &self.entries[index])
            .collect()
    }

    pub fn clear_outage(&self, provider: Provider, outage_path: PathBuf) {
        OutageMemory::new(outage_path).clear(provider);
    }

    pub fn record_busy(
        &self,
        provider: Provider,
        retry_after: Option<Duration>,
        outage_path: PathBuf,
    ) {
        OutageMemory::new(outage_path)
            .mark_busy(provider, retry_after.unwrap_or(self.outage_memory));
    }

    /// A live upload already tried `start - 1`; callers can supply that
    /// failed attempt and walk the remaining file-backed candidates.
    #[allow(clippy::too_many_arguments)] // file input, requirement filters, and live-upload continuation are independent
    pub async fn run_file(
        &self,
        audio: &Path,
        config: &Config,
        diarize: bool,
        lang: Option<&str>,
        sink: &Arc<dyn TelemetrySink>,
        outage_path: PathBuf,
        start: usize,
        prior: Vec<Attempt>,
        notify: Option<&dyn Fn(&str)>,
    ) -> Result<ChainOutcome, TalkError> {
        let memory = OutageMemory::new(outage_path);
        let mut attempts = prior;
        let mut last_error = None;
        let indices = self.ordered_indices(&memory, start);
        for (position, index) in indices.iter().enumerate() {
            let entry = &self.entries[*index];
            #[cfg(feature = "parakeet")]
            if entry.provider.is_local() {
                let status = super::parakeet::consent::resolve(config)?;
                if !status.present {
                    log::info!(
                        "chain \"{}\": downloading explicitly configured local Parakeet model",
                        self.name
                    );
                    super::parakeet::model::download_model(&status.model_dir, status.variant)
                        .await?;
                }
            }
            let result = transcribe_audio(
                audio,
                config,
                entry.provider,
                Some(&entry.model),
                diarize,
                TranscribeOptions {
                    allow_api: true,
                    policy: RequestTimeoutPolicy::Proportional,
                    retry_schedule: Some(entry.retry_schedule()),
                    language: lang.map(str::to_string),
                    ..TranscribeOptions::default()
                },
                sink,
            )
            .await;
            match result {
                Ok(mut result) => {
                    memory.clear(entry.provider);
                    attempts.push(Attempt::new(entry, "success"));
                    result.metadata.attempts = attempts.clone();
                    if let Err(error) = crate::recording_cache::TranscriptionCache::store(
                        audio,
                        entry.provider,
                        &entry.model,
                        false,
                        &result,
                    ) {
                        log::warn!("could not persist chain attempts: {error}");
                    }
                    return Ok(ChainOutcome {
                        result,
                        provider: entry.provider,
                        model: entry.model.clone(),
                        attempts,
                    });
                }
                Err(error) if error.is_fallback_worthy() => {
                    memory.mark_busy(
                        entry.provider,
                        error.retry_after().unwrap_or(self.outage_memory),
                    );
                    attempts.push(Attempt::new(entry, "busy"));
                    if let Some(next) = indices.get(position + 1) {
                        let message =
                            format!("{} busy → {}", entry.model, self.entries[*next].model);
                        log::info!("{message}");
                        if let Some(notify) = notify {
                            notify(&message);
                        }
                    }
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(|| {
            TalkError::Config(format!("chain \"{}\" has no remaining entries", self.name))
        }))
    }
}

pub fn outage_path() -> Result<PathBuf, TalkError> {
    OutageMemory::default_path()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct ValidationIsolation {
        _lock: std::sync::MutexGuard<'static, ()>,
        _dir: tempfile::TempDir,
        previous: Option<std::ffi::OsString>,
    }

    impl ValidationIsolation {
        fn new() -> Result<Self, Box<dyn std::error::Error>> {
            let lock = super::super::transport::validate_cache::__TEST_LOCK
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let dir = tempfile::tempdir()?;
            let previous = std::env::var_os("TALK_RS_VALIDATE_CACHE_PATH");
            std::env::set_var(
                "TALK_RS_VALIDATE_CACHE_PATH",
                dir.path().join("validate-cache.yaml"),
            );
            super::super::transport::validate_cache::__test_reset();
            Ok(Self {
                _lock: lock,
                _dir: dir,
                previous,
            })
        }
    }

    impl Drop for ValidationIsolation {
        fn drop(&mut self) {
            if let Some(value) = self.previous.take() {
                std::env::set_var("TALK_RS_VALIDATE_CACHE_PATH", value);
            } else {
                std::env::remove_var("TALK_RS_VALIDATE_CACHE_PATH");
            }
            super::super::transport::validate_cache::__test_reset();
        }
    }

    async fn mock_config(
        first_status: u16,
        preflight_status: u16,
    ) -> Result<
        (
            tempfile::TempDir,
            Config,
            ResolvedChain,
            MockServer,
            MockServer,
        ),
        Box<dyn std::error::Error>,
    > {
        let dir = tempfile::tempdir()?;
        let first = MockServer::start().await;
        let second = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(preflight_status)
                    .set_body_json(serde_json::json!({"data": [{"id": "gpt-transcribe"}]})),
            )
            .mount(&first)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data": [{"id": "voxtral-mini-2602"}]})),
            )
            .mount(&second)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(
                ResponseTemplate::new(first_status).set_body_string("busy or unauthorized"),
            )
            .mount(&first)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"text": "second provider answered"})),
            )
            .mount(&second)
            .await;
        let config_path = dir.path().join("config.yaml");
        std::fs::write(&config_path, format!("output_dir: {}\nproviders:\n  openai: {{api_key: test, url: {}}}\n  mistral: {{api_key: test, url: {}}}\ntranscription:\n  chains:\n    fallback:\n      - openai/gpt-transcribe\n      - mistral/voxtral-mini-2602\n", dir.path().display(), first.uri(), second.uri()))?;
        let config = Config::load(Some(&config_path))?;
        let chain = config
            .resolve_chain(
                crate::config::ChainCommand::Transcribe,
                Some("fallback"),
                None,
                None,
            )?
            .ok_or("chain missing")?;
        Ok((dir, config, chain, first, second))
    }

    #[tokio::test]
    async fn busy_http_response_walks_to_second_provider_and_persists_attempts(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _isolation = ValidationIsolation::new()?;
        let (dir, config, chain, first, second) = mock_config(429, 200).await?;
        let audio = dir.path().join("recording.ogg");
        std::fs::write(&audio, b"not an ogg; upload raw")?;
        let sink: Arc<dyn TelemetrySink> = Arc::new(crate::telemetry::NoOpSink);
        let output = chain
            .run_file(
                &audio,
                &config,
                false,
                None,
                &sink,
                dir.path().join("outages.yml"),
                0,
                Vec::new(),
                None,
            )
            .await?;
        assert_eq!(output.result.text, "second provider answered");
        assert_eq!(output.provider, Provider::Mistral);
        assert_eq!(
            output
                .attempts
                .iter()
                .map(|a| a.outcome.as_str())
                .collect::<Vec<_>>(),
            ["busy", "success"]
        );
        let cached = crate::recording_cache::TranscriptionCache::get(
            &audio,
            Provider::Mistral,
            "voxtral-mini-2602",
        )
        .ok_or("sidecar missing")?;
        assert_eq!(cached.metadata.attempts, output.attempts);
        assert_eq!(
            first
                .received_requests()
                .await
                .ok_or("first server requests unavailable")?
                .iter()
                .filter(|r| r.method.as_str() == "POST")
                .count(),
            1
        );
        assert_eq!(
            second
                .received_requests()
                .await
                .ok_or("second server requests unavailable")?
                .iter()
                .filter(|r| r.method.as_str() == "POST")
                .count(),
            1
        );
        Ok(())
    }

    #[tokio::test]
    async fn permanent_http_response_stops_chain_before_second_provider(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _isolation = ValidationIsolation::new()?;
        let (dir, config, chain, _first, second) = mock_config(401, 200).await?;
        let audio = dir.path().join("recording.ogg");
        std::fs::write(&audio, b"not an ogg; upload raw")?;
        let sink: Arc<dyn TelemetrySink> = Arc::new(crate::telemetry::NoOpSink);
        let error = chain
            .run_file(
                &audio,
                &config,
                false,
                None,
                &sink,
                dir.path().join("outages.yml"),
                0,
                Vec::new(),
                None,
            )
            .await
            .expect_err("401 is permanent");
        assert!(!error.is_fallback_worthy());
        assert_eq!(
            second
                .received_requests()
                .await
                .ok_or("second server requests unavailable")?
                .len(),
            0
        );
        Ok(())
    }

    #[tokio::test]
    async fn busy_model_preflight_falls_through_without_posting_audio(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _isolation = ValidationIsolation::new()?;
        let (dir, config, chain, first, second) = mock_config(200, 429).await?;
        let audio = dir.path().join("recording.ogg");
        std::fs::write(&audio, b"not an ogg; upload raw")?;
        let sink: Arc<dyn TelemetrySink> = Arc::new(crate::telemetry::NoOpSink);
        let outcome = chain
            .run_file(
                &audio,
                &config,
                false,
                None,
                &sink,
                dir.path().join("outages.yml"),
                0,
                Vec::new(),
                None,
            )
            .await?;
        assert_eq!(outcome.provider, Provider::Mistral);
        assert_eq!(
            first
                .received_requests()
                .await
                .ok_or("first requests")?
                .iter()
                .map(|r| r.method.as_str())
                .collect::<Vec<_>>(),
            ["GET"]
        );
        assert_eq!(
            second
                .received_requests()
                .await
                .ok_or("second requests")?
                .iter()
                .map(|r| r.method.as_str())
                .collect::<Vec<_>>(),
            ["GET", "POST"]
        );
        Ok(())
    }

    fn entry(provider: Provider, model: &str) -> ChainEntry {
        ChainEntry {
            provider,
            model: model.to_string(),
            realtime_model: None,
            retries: None,
            wait: None,
            languages: None,
            supports: Vec::new(),
        }
    }

    #[test]
    fn chain_schedule_limits_retries_and_wait() {
        let mut e = entry(Provider::OpenAI, "gpt-transcribe");
        assert_eq!(e.retry_schedule().data_backoffs.len(), 0);
        assert_eq!(e.retry_schedule().connection_budgets.len(), 1);
        e.retries = Some(2);
        assert_eq!(
            e.retry_schedule().data_backoffs,
            vec![Duration::from_secs(5), Duration::from_secs(15)]
        );
        e.wait = Some(Duration::from_secs(45));
        assert_eq!(
            e.retry_schedule().max_data_wait,
            Some(Duration::from_secs(45))
        );
        e.retries = None;
        assert!(e.retry_schedule().data_backoffs.len() > 6);
    }

    #[test]
    fn chain_filters_hard_requirements_and_declared_unknown_capabilities() {
        let mut french = entry(Provider::OpenAI, "gpt-transcribe");
        french.languages = Some(vec!["fr".into()]);
        let mut future = entry(Provider::Mistral, "some-future-model");
        future.supports = vec!["diarize".into(), "realtime".into()];
        let chain = ResolvedChain {
            name: "dictate".into(),
            entries: vec![
                french,
                entry(Provider::Parakeet, "parakeet"),
                future.clone(),
            ],
            outage_memory: Duration::from_secs(180),
        };
        assert_eq!(
            chain
                .eligible(true, false, None)
                .expect("declared diarize")
                .entries
                .len(),
            1
        );
        assert_eq!(
            chain
                .eligible(false, true, None)
                .expect("declared realtime")
                .entries[0]
                .model,
            future.model
        );
        assert_eq!(
            chain
                .eligible(false, false, Some("fr"))
                .expect("French")
                .entries
                .len(),
            3
        );
        assert_eq!(
            chain
                .eligible(false, false, None)
                .expect("no language hint")
                .entries
                .len(),
            2
        );
    }

    #[test]
    fn chain_refuses_when_every_entry_is_ineligible() {
        let chain = ResolvedChain {
            name: "dictate".into(),
            entries: vec![
                entry(Provider::OpenAI, "gpt-transcribe"),
                entry(Provider::Parakeet, "parakeet"),
            ],
            outage_memory: Duration::from_secs(180),
        };
        assert_eq!(chain.eligible(true, false, None).expect_err("no diarization").to_string(), "Configuration error: chain \"dictate\" has no entry supporting --diarize (skipped: openai/gpt-transcribe: no diarization, parakeet/parakeet: no diarization)");
    }

    #[test]
    fn all_in_outage_still_produces_an_attempt() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("outages.yml");
        let memory = OutageMemory::new(path.clone());
        memory.mark_busy(Provider::OpenAI, Duration::from_secs(180));
        memory.mark_busy(Provider::Mistral, Duration::from_secs(180));
        let chain = ResolvedChain {
            name: "dictate".into(),
            entries: vec![
                entry(Provider::OpenAI, "gpt-transcribe"),
                entry(Provider::Mistral, "voxtral-mini-2602"),
            ],
            outage_memory: Duration::from_secs(180),
        };
        assert_eq!(chain.ordered_indices(&memory, 0), vec![0, 1]);
        memory.clear(Provider::Mistral);
        assert_eq!(chain.ordered_indices(&memory, 0), vec![1]);
        Ok(())
    }
}
