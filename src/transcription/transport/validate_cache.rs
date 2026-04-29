//! Disk-backed cache for `/v1/models` validation results.
//!
//! Validating a `(provider, model, api_base)` triple costs one
//! HTTP round-trip to the provider's `/v1/models` endpoint — see
//! [`super::http::validate_model`].  On a flaky VPN that round-trip
//! fails frequently, and historically every transcription paid the
//! preflight cost on every invocation.  This module memoizes
//! successful validations to a YAML file at
//! `$XDG_CACHE_HOME/talk-rs/validate-cache.yaml` with a 24-hour
//! TTL, so subsequent calls (within the TTL) skip the network
//! entirely.
//!
//! # Concurrency
//!
//! Multiple talk-rs processes (picker spawn, dictate daemon,
//! `record --ui`, …) may all attempt to write the cache
//! simultaneously.  Writes use the standard tempfile + atomic
//! rename pattern: a temporary sibling is written and fsynced,
//! then `rename(2)` swaps it onto the target path.  POSIX rename
//! is atomic, so concurrent readers always see a complete file;
//! concurrent writers race and the loser's write is lost — which
//! is acceptable because each writer is just confirming a model
//! that's already validated.
//!
//! # Cache scope
//!
//! - In-process layer: a `OnceLock<Mutex<HashMap<Key, DateTime<Utc>>>>`
//!   sits in front of the disk file.  Repeated lookups within the
//!   same process never re-read the file.  Writes propagate to
//!   both layers.
//! - Disk layer: read once on first access, then re-read whenever
//!   the file's mtime advances past our last-known mtime (so a
//!   sibling process's write becomes visible without a process
//!   restart).
//!
//! # Failure handling
//!
//! Cache failures are non-fatal.  `is_fresh` returning `false` on
//! any I/O error makes the caller fall through to the network — a
//! transient cache hiccup must not block transcription.  `record`
//! errors are logged at `warn` and swallowed for the same reason.

use crate::config::Provider;
use crate::error::TalkError;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// TTL for a successful validation: 24 hours.  Fresh entries skip
/// the network; stale entries get re-validated on next access and
/// the cache row is rewritten on success.  Stale entries are NOT
/// pre-emptively pruned — they get overwritten in place, which
/// keeps the file size bounded by `(provider, model, api_base)`
/// triples in actual use.
const TTL: Duration = Duration::hours(24);

/// File name within the talk-rs cache directory.
const CACHE_FILE_NAME: &str = "validate-cache.yaml";

/// In-memory + on-disk key for a single validation entry.
///
/// The triple `(provider, model, api_base)` is deliberately exact:
/// the same model name on different `api_base` URLs (e.g. a
/// staging proxy vs. production) is a different cache entry —
/// one's freshness says nothing about the other.
///
/// `provider` is stored as the lowercase string form returned by
/// [`Provider::to_string`] (e.g. `"mistral"`, `"openai"`) rather
/// than the enum.  Keeps the on-disk cache human-readable and
/// avoids leaking enum-derive churn (`Hash`, `Serialize`) into
/// the foundational [`crate::config::Provider`] type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Key {
    provider: String,
    model: String,
    api_base: String,
}

/// Single on-disk cache row.
///
/// `validated_at` is the wall-clock UTC timestamp at which the
/// validation last succeeded; `is_fresh` compares it against
/// `Utc::now()` and the [`TTL`] constant.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    provider: String,
    model: String,
    api_base: String,
    validated_at: DateTime<Utc>,
}

/// Top-level YAML structure: a flat list of entries.  Chosen over
/// a `HashMap<Key, …>` so the on-disk representation is
/// human-readable when debugging the cache state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CacheFile {
    entries: Vec<Entry>,
}

/// In-process overlay shared across all callers within a single
/// talk-rs process.  Initialised lazily on first access.
static IN_PROCESS: OnceLock<Mutex<HashMap<Key, DateTime<Utc>>>> = OnceLock::new();

/// Last-known mtime of the cache file on disk.  Used to detect
/// when a sibling process has written a fresh entry that we
/// should pick up.  `None` until the first read.
static LAST_DISK_MTIME: OnceLock<Mutex<Option<std::time::SystemTime>>> = OnceLock::new();

fn in_process() -> &'static Mutex<HashMap<Key, DateTime<Utc>>> {
    IN_PROCESS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn last_disk_mtime() -> &'static Mutex<Option<std::time::SystemTime>> {
    LAST_DISK_MTIME.get_or_init(|| Mutex::new(None))
}

/// Resolve the cache file path: `$XDG_CACHE_HOME/talk-rs/validate-cache.yaml`.
///
/// Reuses the project-wide cache directory established by
/// [`crate::daemon::cache_dir`] so the file sits alongside
/// `daemon.log`, `talk-rs.log`, etc.
fn cache_path() -> Result<PathBuf, TalkError> {
    Ok(crate::daemon::cache_dir()?.join(CACHE_FILE_NAME))
}

/// Read the disk file (if present and fresh-mtime-relative-to-our-last-read).
///
/// Returns the parsed entries, or `None` if the file is missing,
/// unreadable, or unparseable.  Cache corruption is non-fatal —
/// the caller falls through to the network.
fn read_disk_if_changed() -> Option<Vec<Entry>> {
    let path = cache_path().ok()?;
    let metadata = std::fs::metadata(&path).ok()?;
    let current_mtime = metadata.modified().ok()?;

    let mtime_changed = {
        let last = last_disk_mtime().lock().ok()?;
        last.map(|m| m != current_mtime).unwrap_or(true)
    };

    if !mtime_changed {
        return None; // Disk hasn't changed since last read
    }

    let bytes = std::fs::read(&path).ok()?;
    let file: CacheFile = serde_yaml::from_slice(&bytes).ok()?;

    if let Ok(mut last) = last_disk_mtime().lock() {
        *last = Some(current_mtime);
    }

    Some(file.entries)
}

/// Merge any newer-on-disk entries into the in-process map.
///
/// Called at the top of every cache lookup so a sibling process's
/// successful validation becomes visible without restarting us.
fn refresh_in_process_from_disk() {
    let Some(entries) = read_disk_if_changed() else {
        return;
    };
    let Ok(mut map) = in_process().lock() else {
        return;
    };
    for e in entries {
        let key = Key {
            provider: e.provider,
            model: e.model,
            api_base: e.api_base,
        };
        // Always overwrite: the disk version is authoritative for
        // entries we haven't validated ourselves this run.
        map.insert(key, e.validated_at);
    }
}

/// Check whether `(provider, model, api_base)` was validated
/// successfully within the last [`TTL`].
///
/// Returns `true` only when a fresh entry exists.  Errors,
/// missing entries, and stale entries all return `false` so the
/// caller falls through to the network.
pub(crate) fn is_fresh(provider: Provider, model: &str, api_base: &str) -> bool {
    refresh_in_process_from_disk();
    let key = Key {
        provider: provider.to_string(),
        model: model.to_string(),
        api_base: api_base.to_string(),
    };
    let Ok(map) = in_process().lock() else {
        return false;
    };
    let Some(&validated_at) = map.get(&key) else {
        return false;
    };
    Utc::now().signed_duration_since(validated_at) < TTL
}

/// Record a successful validation for `(provider, model, api_base)`.
///
/// Updates the in-process overlay, then atomically rewrites the
/// disk file.  Disk-write errors are logged at `warn` and
/// swallowed — a cache miss next call is strictly preferable to
/// failing a transcription that already validated on the wire.
pub(crate) fn record(provider: Provider, model: &str, api_base: &str) {
    let key = Key {
        provider: provider.to_string(),
        model: model.to_string(),
        api_base: api_base.to_string(),
    };
    let now = Utc::now();

    if let Ok(mut map) = in_process().lock() {
        map.insert(key, now);
    }

    if let Err(e) = persist_to_disk() {
        log::warn!("validate-cache: failed to persist to disk: {}", e);
    }
}

/// Serialise the current in-process map and write it atomically
/// to disk via tempfile + rename.
///
/// Errors propagate so [`record`] can log them; the in-process
/// overlay is already updated regardless of disk-write outcome.
fn persist_to_disk() -> Result<(), TalkError> {
    let path = cache_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            TalkError::Config(format!(
                "validate-cache: failed to create cache dir {}: {}",
                parent.display(),
                e
            ))
        })?;
    }

    let entries: Vec<Entry> = match in_process().lock() {
        Ok(map) => map
            .iter()
            .map(|(k, &t)| Entry {
                provider: k.provider.clone(),
                model: k.model.clone(),
                api_base: k.api_base.clone(),
                validated_at: t,
            })
            .collect(),
        Err(_) => {
            return Err(TalkError::Config(
                "validate-cache: in-process map poisoned".to_string(),
            ));
        }
    };

    let file = CacheFile { entries };
    let yaml = serde_yaml::to_string(&file).map_err(|e| {
        TalkError::Config(format!("validate-cache: failed to serialise YAML: {}", e))
    })?;

    // Atomic write: tempfile in the same directory as the target
    // (so `rename(2)` stays on the same filesystem) + rename.
    let tmp_path = path.with_extension("yaml.tmp");
    std::fs::write(&tmp_path, yaml.as_bytes()).map_err(|e| {
        TalkError::Config(format!(
            "validate-cache: failed to write tempfile {}: {}",
            tmp_path.display(),
            e
        ))
    })?;
    std::fs::rename(&tmp_path, &path).map_err(|e| {
        TalkError::Config(format!(
            "validate-cache: failed to rename {} -> {}: {}",
            tmp_path.display(),
            path.display(),
            e
        ))
    })?;

    // Update last-known mtime so the next is_fresh() doesn't
    // needlessly re-read the file we just wrote.
    if let Ok(metadata) = std::fs::metadata(&path) {
        if let Ok(mtime) = metadata.modified() {
            if let Ok(mut last) = last_disk_mtime().lock() {
                *last = Some(mtime);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// All cache tests share the same `OnceLock`-backed
    /// in-process map and the same `cache_path()` (resolved via
    /// XDG).  Serialise them so they don't trample each other.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    /// Reset the in-process overlay between tests.  Necessary
    /// because the `OnceLock` persists across `#[test]` boundaries.
    fn reset_in_process() {
        if let Some(map) = IN_PROCESS.get() {
            if let Ok(mut m) = map.lock() {
                m.clear();
            }
        }
        if let Some(mtime) = LAST_DISK_MTIME.get() {
            if let Ok(mut t) = mtime.lock() {
                *t = None;
            }
        }
    }

    /// Override the cache path for tests by overriding the home
    /// dir.  ProjectDirs honours `$HOME` on Linux.
    fn with_temp_home<F: FnOnce()>(f: F) {
        let _guard = TEST_LOCK.lock().expect("test: lock poisoned");
        reset_in_process();

        let tmp = tempfile::TempDir::new().expect("test: tempdir");
        let prev_home = std::env::var_os("HOME");
        let prev_xdg = std::env::var_os("XDG_CACHE_HOME");
        // SAFETY: tests are serialised by TEST_LOCK so concurrent
        // env mutation is impossible within this binary.
        unsafe {
            std::env::set_var("HOME", tmp.path());
            std::env::set_var("XDG_CACHE_HOME", tmp.path().join("cache"));
        }

        f();

        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            match prev_xdg {
                Some(v) => std::env::set_var("XDG_CACHE_HOME", v),
                None => std::env::remove_var("XDG_CACHE_HOME"),
            }
        }
        reset_in_process();
    }

    /// Spec: a freshly-recorded entry is reported as fresh, and
    /// missing entries report not-fresh.
    #[test]
    fn record_then_is_fresh_returns_true() {
        with_temp_home(|| {
            assert!(
                !is_fresh(Provider::Mistral, "voxtral", "https://x"),
                "missing entry must not be fresh"
            );
            record(Provider::Mistral, "voxtral", "https://x");
            assert!(
                is_fresh(Provider::Mistral, "voxtral", "https://x"),
                "just-recorded entry must be fresh"
            );
        });
    }

    /// Spec: keys are exact — same model on different api_base is
    /// a different entry.
    #[test]
    fn cache_distinguishes_api_base() {
        with_temp_home(|| {
            record(Provider::Mistral, "voxtral", "https://api.mistral.ai");
            assert!(is_fresh(
                Provider::Mistral,
                "voxtral",
                "https://api.mistral.ai"
            ));
            assert!(
                !is_fresh(Provider::Mistral, "voxtral", "https://staging.mistral.ai"),
                "different api_base must be a separate entry"
            );
        });
    }

    /// Spec: keys are exact — different models on the same api_base
    /// are independent entries.
    #[test]
    fn cache_distinguishes_model() {
        with_temp_home(|| {
            record(Provider::Mistral, "voxtral-mini-2602", "https://x");
            assert!(is_fresh(
                Provider::Mistral,
                "voxtral-mini-2602",
                "https://x"
            ));
            assert!(!is_fresh(
                Provider::Mistral,
                "voxtral-mini-2507",
                "https://x"
            ));
        });
    }

    /// Spec: a manually-constructed stale entry (older than TTL)
    /// is reported as not-fresh.  Constructed by writing directly
    /// into the in-process map with a back-dated timestamp,
    /// bypassing the public `record` (which always uses Utc::now).
    #[test]
    fn stale_entry_reports_not_fresh() {
        with_temp_home(|| {
            let stale_time = Utc::now() - Duration::hours(25);
            let key = Key {
                provider: Provider::Mistral.to_string(),
                model: "voxtral".into(),
                api_base: "https://x".into(),
            };
            in_process()
                .lock()
                .expect("test: lock")
                .insert(key, stale_time);
            assert!(
                !is_fresh(Provider::Mistral, "voxtral", "https://x"),
                "entry older than 24h must not be fresh"
            );
        });
    }

    /// Spec: the disk file roundtrips a recorded entry — a sibling
    /// process simulated by clearing the in-process map and
    /// re-reading is able to see the prior `record`.
    #[test]
    fn disk_persistence_survives_in_process_clear() {
        with_temp_home(|| {
            record(Provider::Mistral, "voxtral", "https://x");
            // Simulate a fresh process: clear the in-process map.
            // The next `is_fresh` should re-read from disk.
            if let Ok(mut map) = in_process().lock() {
                map.clear();
            }
            if let Ok(mut t) = last_disk_mtime().lock() {
                *t = None; // Force a re-read regardless of mtime.
            }
            assert!(
                is_fresh(Provider::Mistral, "voxtral", "https://x"),
                "entry must be readable from disk after in-process clear"
            );
        });
    }

    /// Spec: the on-disk file is valid YAML that round-trips
    /// through serde_yaml.  Catches any future schema drift that
    /// would break readers of older cache files.
    #[test]
    fn disk_format_is_valid_yaml() {
        with_temp_home(|| {
            record(Provider::Mistral, "voxtral", "https://x");
            record(Provider::OpenAI, "whisper-1", "https://api.openai.com");

            let path = cache_path().expect("test: path");
            let bytes = std::fs::read(&path).expect("test: read cache");
            let file: CacheFile = serde_yaml::from_slice(&bytes).expect("test: parse cache");
            assert_eq!(file.entries.len(), 2);
        });
    }

    /// Spec: a corrupt disk file does not panic — `is_fresh`
    /// silently falls through to "not fresh", so the caller goes
    /// to the network.
    #[test]
    fn corrupt_disk_file_is_not_fatal() {
        with_temp_home(|| {
            // Write garbage to the cache path, then ask is_fresh —
            // must return false, not panic.
            let path = cache_path().expect("test: path");
            std::fs::create_dir_all(path.parent().expect("test: parent"))
                .expect("test: create dir");
            std::fs::write(&path, b"this is not valid yaml: {[}}").expect("test: write garbage");

            // Force re-read.
            if let Ok(mut t) = last_disk_mtime().lock() {
                *t = None;
            }
            if let Ok(mut map) = in_process().lock() {
                map.clear();
            }

            // Lookup must not panic; missing-or-corrupt = not-fresh.
            assert!(!is_fresh(Provider::Mistral, "voxtral", "https://x"));
        });
    }

    /// Spec: re-recording the same key updates the timestamp
    /// (refreshes the entry).  Important for the case where an
    /// entry is on the verge of expiring — a successful re-validate
    /// pushes the expiry forward.
    #[test]
    fn re_recording_refreshes_timestamp() {
        with_temp_home(|| {
            // Insert a stale entry directly.
            let stale = Utc::now() - Duration::hours(23) - Duration::minutes(59);
            let key = Key {
                provider: Provider::Mistral.to_string(),
                model: "voxtral".into(),
                api_base: "https://x".into(),
            };
            in_process()
                .lock()
                .expect("test: lock")
                .insert(key.clone(), stale);

            // Re-record with the public API.
            record(Provider::Mistral, "voxtral", "https://x");

            // The entry must now be safely fresh — the timestamp
            // moved to (approximately) Utc::now().
            let new_ts = *in_process()
                .lock()
                .expect("test: lock")
                .get(&key)
                .expect("test: entry present");
            assert!(
                new_ts > stale,
                "re-record must advance the timestamp; got {} <= {}",
                new_ts,
                stale
            );
        });
    }
}
