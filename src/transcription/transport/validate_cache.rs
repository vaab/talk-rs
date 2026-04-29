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
//! simultaneously.  Writes are protected by a POSIX advisory file
//! lock ([`fs2::FileExt::lock_exclusive`]) on a sibling lock file
//! (`validate-cache.yaml.lock`).  The write path is read-modify-write:
//!
//! 1. Acquire the exclusive lock.
//! 2. Re-read the cache file from disk and merge with our
//!    in-process map (newer timestamp wins per `(provider, model,
//!    api_base)` key).
//! 3. Write the merged set to a tempfile + atomic rename onto the
//!    final path.
//! 4. Release the lock.
//!
//! This eliminates the read-modify-write race that earlier versions
//! of this module had, where a process with only its own entries in
//! the in-process map could clobber sibling-process entries on
//! `persist_to_disk()`.
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
//! # Test isolation
//!
//! The cache file path can be overridden via the
//! `TALK_RS_VALIDATE_CACHE_PATH` environment variable.  Tests that
//! exercise the cache point this at a tempfile so they neither
//! pollute the user's real cache nor depend on env-mutation
//! ordering across the test binary.  In production this variable
//! is unset and the cache lives at the standard XDG location.
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
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
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

/// Environment variable that, when set, overrides the on-disk
/// cache file path.  Used exclusively by the test suite to
/// redirect cache I/O at a tempfile so tests cannot pollute the
/// user's real cache.  Production code does not set this.
const CACHE_PATH_ENV: &str = "TALK_RS_VALIDATE_CACHE_PATH";

/// Resolve the cache file path.
///
/// Resolution order:
///
/// 1. `$TALK_RS_VALIDATE_CACHE_PATH` — explicit override (tests).
/// 2. `$XDG_CACHE_HOME/talk-rs/validate-cache.yaml` — production.
///
/// Reuses the project-wide cache directory established by
/// [`crate::daemon::cache_dir`] so the file sits alongside
/// `daemon.log`, `talk-rs.log`, etc.
fn cache_path() -> Result<PathBuf, TalkError> {
    if let Some(p) = std::env::var_os(CACHE_PATH_ENV) {
        return Ok(PathBuf::from(p));
    }
    Ok(crate::daemon::cache_dir()?.join(CACHE_FILE_NAME))
}

/// Resolve the lock file path: sibling of the cache file with the
/// extension `.lock`.  Created on demand by the locking code; the
/// file's contents are irrelevant — only its existence matters
/// for `flock(2)` to attach to.
fn lock_path(cache: &std::path::Path) -> PathBuf {
    let mut p = cache.as_os_str().to_owned();
    p.push(".lock");
    PathBuf::from(p)
}

/// Read the disk file unconditionally (no mtime gate).
///
/// Returns the parsed entries, or `None` if the file is missing,
/// unreadable, or unparseable.  Cache corruption is non-fatal —
/// the caller falls through to the network.  Used inside the
/// locked write path where we always want the freshest disk state.
fn read_disk_unconditional() -> Option<Vec<Entry>> {
    let path = cache_path().ok()?;
    let bytes = std::fs::read(&path).ok()?;
    let file: CacheFile = serde_yaml::from_slice(&bytes).ok()?;
    Some(file.entries)
}

/// Read the disk file only when its mtime has advanced past our
/// last-known mtime.
///
/// Returns the parsed entries on a fresh read, or `None` when the
/// disk hasn't changed (so callers don't redundantly merge identical
/// data into the in-process map every lookup).  Used by the read
/// path ([`is_fresh`]); the write path uses
/// [`read_disk_unconditional`] under flock for correctness.
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
/// disk file under an exclusive POSIX file lock.  Disk-write
/// errors are logged at `warn` and swallowed — a cache miss next
/// call is strictly preferable to failing a transcription that
/// already validated on the wire.
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

/// Serialise the in-process map (merged with any sibling-process
/// entries currently on disk) and atomically write it back.
///
/// The full sequence under an exclusive POSIX file lock:
///
/// 1. Acquire the exclusive lock on `validate-cache.yaml.lock`.
/// 2. Re-read the on-disk YAML.  Each entry is merged into the
///    in-process map iff it's newer than what we already have for
///    the same key (this preserves entries that other processes
///    have validated since we last refreshed and prevents the
///    "fresh-process clobbers everyone else's rows" failure mode).
/// 3. Build a [`CacheFile`] from the merged map and serialise.
/// 4. Tempfile + atomic rename onto the final path.
/// 5. Drop the lock guard (releases the flock).
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

    // ── Acquire exclusive flock on a sibling lock file ──────────
    //
    // Using a sibling rather than the cache file itself means
    // readers (which never lock) don't race against writers; the
    // atomic rename gives them either the previous-complete file
    // or the new-complete file.  The lock file is created on
    // demand and never deleted (no benefit and adds churn).
    let lock_file_path = lock_path(&path);
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_file_path)
        .map_err(|e| {
            TalkError::Config(format!(
                "validate-cache: failed to open lock file {}: {}",
                lock_file_path.display(),
                e
            ))
        })?;
    lock_file.lock_exclusive().map_err(|e| {
        TalkError::Config(format!(
            "validate-cache: failed to acquire exclusive flock on {}: {}",
            lock_file_path.display(),
            e
        ))
    })?;

    // RAII guard: any early return below releases the lock when
    // `_lock_guard` drops.
    let _lock_guard = LockGuard(&lock_file);

    // ── Re-read disk + merge into in-process map ────────────────
    //
    // Under the flock, the disk state cannot change between read
    // and write.  Newer-timestamp-wins on conflict so two
    // concurrent successful validations of the same key produce
    // the latest one; sibling-process entries we've never seen are
    // adopted into our in-process map verbatim.
    if let Some(disk_entries) = read_disk_unconditional() {
        if let Ok(mut map) = in_process().lock() {
            for e in disk_entries {
                let key = Key {
                    provider: e.provider,
                    model: e.model,
                    api_base: e.api_base,
                };
                match map.get(&key).copied() {
                    None => {
                        map.insert(key, e.validated_at);
                    }
                    Some(existing) if e.validated_at > existing => {
                        map.insert(key, e.validated_at);
                    }
                    _ => {} // we already have a newer or equal entry
                }
            }
        }
    }

    // ── Snapshot in-process map → disk ──────────────────────────
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

    // Lock released when `_lock_guard` drops at end of scope.
    Ok(())
}

/// RAII wrapper around a held [`fs2::FileExt::lock_exclusive`]
/// that releases the lock on drop.
///
/// fs2's default behaviour is to release on close, but explicit
/// unlock at the end of `persist_to_disk` is clearer and ensures
/// release happens before any subsequent disk operation in the
/// same caller frame.
struct LockGuard<'a>(&'a std::fs::File);

impl Drop for LockGuard<'_> {
    fn drop(&mut self) {
        // Best-effort unlock; the file close on subsequent drop of
        // the underlying `File` will release the lock anyway.
        let _ = fs2::FileExt::unlock(self.0);
    }
}

/// Process-wide test serialisation lock.
///
/// Every cache-touching test in this crate (whether in this
/// module's `tests` submodule, or in sibling modules like
/// `http::tests`) acquires this same lock at setup so they do
/// not race on the shared `IN_PROCESS` / `LAST_DISK_MTIME`
/// statics or on the `TALK_RS_VALIDATE_CACHE_PATH` env var.
///
/// Public so cross-module test helpers can hold the lock for
/// the duration of a test.
#[cfg(test)]
pub(crate) static __TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Test-only helper: clear the in-process map and the
/// last-known mtime so the next access reads fresh from
/// (whatever the env var points at).  Compiled out of release
/// builds via `#[cfg(test)]`.
///
/// Public so test code in sibling modules (e.g.
/// `transcription::transport::http::tests`) can reset cache
/// state at test setup/teardown without re-implementing the
/// internals.
#[cfg(test)]
pub(crate) fn __test_reset() {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Override the cache path for tests using the
    /// [`CACHE_PATH_ENV`] env var, pointing at a tempfile.
    ///
    /// Resists the env-mutation racing problem of the older
    /// `$HOME`/`$XDG_CACHE_HOME` swap: only one variable is
    /// touched, the test lock serialises access (process-wide
    /// via [`__TEST_LOCK`] so cross-module tests don't race),
    /// and any production code path bypassing the env override
    /// would be caught by a stray write to the dev's real cache
    /// file.
    fn with_temp_home<F: FnOnce()>(f: F) {
        let _guard = __TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        __test_reset();

        let tmp = tempfile::TempDir::new().expect("test: tempdir");
        let cache_file = tmp.path().join("validate-cache.yaml");
        let prev = std::env::var_os(CACHE_PATH_ENV);
        // SAFETY: tests are serialised by TEST_LOCK so concurrent
        // env mutation is impossible within this binary.
        unsafe {
            std::env::set_var(CACHE_PATH_ENV, &cache_file);
        }

        f();

        unsafe {
            match prev {
                Some(v) => std::env::set_var(CACHE_PATH_ENV, v),
                None => std::env::remove_var(CACHE_PATH_ENV),
            }
        }
        __test_reset();
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

    /// Spec: a `record()` call MUST NOT clobber sibling-process
    /// entries already on disk.  Simulates the "fresh process
    /// records its first entry" case that previously wiped the
    /// real production cache.
    ///
    /// Sequence:
    ///   1. Hand-write a YAML file containing a sibling entry
    ///      ("openai" / "whisper-1") simulating a different
    ///      process having recorded it earlier.
    ///   2. Clear our in-process map (simulating a fresh
    ///      process that has never seen the disk file).
    ///   3. Call `record()` for a different key ("mistral" /
    ///      "voxtral-mini-2602").
    ///   4. Read back the disk file; assert BOTH entries are
    ///      present.  Pre-flock this would have wiped the
    ///      sibling and left only the new entry.
    #[test]
    fn record_merges_with_sibling_disk_entries() {
        with_temp_home(|| {
            // 1. Pre-populate disk with a sibling entry.
            let path = cache_path().expect("test: path");
            std::fs::create_dir_all(path.parent().expect("test: parent"))
                .expect("test: create dir");
            let sibling_yaml = "\
entries:
- provider: openai
  model: whisper-1
  api_base: https://api.openai.com
  validated_at: 2026-04-29T10:00:00Z
";
            std::fs::write(&path, sibling_yaml).expect("test: write sibling cache");

            // 2. Clear our in-process map and our last-known
            //    mtime: simulate fresh process that has never
            //    read this file.
            if let Ok(mut map) = in_process().lock() {
                map.clear();
            }
            if let Ok(mut t) = last_disk_mtime().lock() {
                *t = None;
            }

            // 3. Record a *different* key.  Pre-flock this would
            //    have written ONLY the new entry, wiping the
            //    sibling.
            record(
                Provider::Mistral,
                "voxtral-mini-2602",
                "https://api.mistral.ai",
            );

            // 4. Read the disk file directly and assert both
            //    entries are preserved.
            let bytes = std::fs::read(&path).expect("test: read disk after record");
            let parsed: CacheFile = serde_yaml::from_slice(&bytes).expect("test: parse YAML");
            let keys: Vec<(String, String, String)> = parsed
                .entries
                .iter()
                .map(|e| (e.provider.clone(), e.model.clone(), e.api_base.clone()))
                .collect();

            assert!(
                keys.iter()
                    .any(|(p, m, _)| p == "openai" && m == "whisper-1"),
                "sibling openai entry must be preserved, got: {:?}",
                keys
            );
            assert!(
                keys.iter()
                    .any(|(p, m, _)| p == "mistral" && m == "voxtral-mini-2602"),
                "newly-recorded mistral entry must be present, got: {:?}",
                keys
            );
            assert_eq!(keys.len(), 2, "expected exactly 2 entries, got: {:?}", keys);
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
