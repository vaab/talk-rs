//! Short-lived provider-wide busy memory shared across invocations.

use crate::config::Provider;
use crate::error::TalkError;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Default, Serialize, Deserialize)]
struct Outages(BTreeMap<String, u64>);

pub(super) struct OutageMemory {
    path: PathBuf,
}

impl OutageMemory {
    pub(super) fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub(super) fn default_path() -> Result<PathBuf, TalkError> {
        Ok(crate::daemon::cache_dir()?.join("outages.yml"))
    }

    fn read(&self) -> Outages {
        match fs::read_to_string(&self.path) {
            Ok(raw) => match serde_yaml::from_str(&raw) {
                Ok(outages) => outages,
                Err(error) => {
                    log::warn!(
                        "corrupt outage memory {}: {error}; ignoring it",
                        self.path.display()
                    );
                    Outages::default()
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Outages::default(),
            Err(error) => {
                log::warn!(
                    "could not read outage memory {}: {error}",
                    self.path.display()
                );
                Outages::default()
            }
        }
    }

    pub(super) fn is_busy(&self, provider: Provider) -> bool {
        self.read()
            .0
            .get(&provider.to_string())
            .is_some_and(|until| *until > now_seconds())
    }

    pub(super) fn mark_busy(&self, provider: Provider, duration: Duration) {
        self.update(|outages| {
            outages.0.insert(
                provider.to_string(),
                now_seconds().saturating_add(duration.as_secs().max(1)),
            );
        });
    }

    pub(super) fn clear(&self, provider: Provider) {
        self.update(|outages| {
            outages.0.remove(&provider.to_string());
        });
    }

    fn update(&self, change: impl FnOnce(&mut Outages)) {
        if let Err(error) = self.update_inner(change) {
            log::warn!(
                "could not update outage memory {}: {error}",
                self.path.display()
            );
        }
    }

    fn update_inner(&self, change: impl FnOnce(&mut Outages)) -> Result<(), TalkError> {
        let parent = self.path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.path.with_extension("yml.lock"))?;
        lock.lock_exclusive()?;
        let mut outages = self.read();
        change(&mut outages);
        let yaml = serde_yaml::to_string(&outages)
            .map_err(|e| TalkError::Config(format!("cannot serialize outage memory: {e}")))?;
        let tmp = parent.join(format!(
            ".outages-{}-{}.tmp",
            std::process::id(),
            now_nanos()
        ));
        let result = (|| -> Result<(), TalkError> {
            let mut file = OpenOptions::new().create_new(true).write(true).open(&tmp)?;
            use std::io::Write;
            file.write_all(yaml.as_bytes())?;
            file.sync_all()?;
            fs::rename(&tmp, &self.path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result
    }
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outage_memory_skips_then_clears_provider_and_ignores_corruption(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let memory = OutageMemory::new(dir.path().join("outages.yml"));
        memory.mark_busy(Provider::OpenAI, Duration::from_secs(180));
        assert!(memory.is_busy(Provider::OpenAI));
        assert!(!memory.is_busy(Provider::Mistral));
        memory.clear(Provider::OpenAI);
        assert!(!memory.is_busy(Provider::OpenAI));
        fs::write(dir.path().join("outages.yml"), "invalid: [")?;
        assert!(!memory.is_busy(Provider::OpenAI));
        Ok(())
    }
}
