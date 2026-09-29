//! Cross-process in-flight transcription job registry.
//!
//! Builds on the existing per-model lock files in
//! [`crate::recording_cache`] (`acquire_model_lock` /
//! `release_model_lock`).  Adds two cross-process capabilities:
//!
//! 1. **Lock payload upgrade** — historically the lock file was
//!    empty.  This module writes a small YAML payload containing
//!    `owner_pid`, `owner_started_at`, and a protocol `version`
//!    so an observer (typically a freshly-opened picker) can tell
//!    which talk-rs process owns the lock and whether the owner is
//!    still alive.
//!
//! 2. **Cancellation by signal** — the owner registers a SIGUSR1
//!    handler that triggers its [`CancellationToken`].  An observer
//!    that wants to cancel sends SIGUSR1 to `owner_pid`.  This
//!    matches the simplicity of the existing lock files and
//!    avoids inventing a daemon or a Unix-socket protocol.
//!
//! # Scope (Step 11 of transport-consolidation plan)
//!
//! Same-process and cross-process cancellation both work via the
//! same code path: in-process callers hold a [`LocalJob`] guard,
//! out-of-process callers call [`cancel_remote`] which sends
//! SIGUSR1 to the owning PID.  The in-process owner's signal
//! handler (installed by [`register_local`]) triggers the
//! [`CancellationToken`] which the transport layer's `select!`
//! arms in [`super::transport::http_request`] /
//! [`super::transport::ws_upgrade`] already observe.
//!
//! Event replay (a fresh picker attaching to an in-flight job and
//! seeing past `ConnectionEvent`s) is **not** delivered here.
//! Step 12 of the plan adds a per-job Unix socket if the picker
//! UX requires live event streaming for adopted-on-open rows.
//! For "stop a running transcription started elsewhere" the
//! SIGUSR1 path is sufficient.

use crate::config::Provider;
use crate::error::TalkError;
use crate::recording_cache;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use tokio_util::sync::CancellationToken;

/// On-disk lock payload format version.  Bump when the YAML
/// schema changes incompatibly.  Observers refuse to attach to a
/// lock with a newer-major version than they understand.
const LOCK_PAYLOAD_VERSION: u32 = 1;
static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(0);

/// YAML payload written into the per-model lock file by
/// [`register_local`].
///
/// Replaces the historical empty lock file.  The schema is
/// versioned so future additions (e.g. a Unix-socket path for
/// event replay in Step 12) can be added without breaking
/// existing readers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockPayload {
    /// On-disk format version.  Equal to `LOCK_PAYLOAD_VERSION`
    /// when written.  Observers refuse to attach to an unknown
    /// future version (graceful degradation: treat as "in flight,
    /// not observable").
    pub version: u32,
    /// PID of the talk-rs process that holds the lock.  Used by
    /// [`cancel_remote`] (`kill(pid, SIGUSR1)`) and by
    /// `is_stale` (`/proc/<pid>` existence check).
    pub owner_pid: u32,
    /// When the owner started this job (`SystemTime` epoch
    /// seconds + nanos).  Carried so a stale-PID check can
    /// disambiguate PID reuse: if `/proc/<pid>/stat` shows a
    /// process whose start time differs from this value, the
    /// lock is stale and the holder of that PID is a different
    /// process.
    pub owner_started_at_unix_secs: u64,
    /// Provider this job is targeting — informational; observers
    /// already know it from the lock filename, but having it in
    /// the payload makes diagnostic dumps self-describing.
    pub provider: String,
    /// Model identifier.
    pub model: String,
    /// Streaming flag (realtime vs one-shot).
    pub realtime: bool,
    /// Unique registration identity; absent on older lock files.
    #[serde(default)]
    pub owner_id: String,
}

impl LockPayload {
    /// Build a fresh payload for a new job by `register_local`.
    fn new(provider: Provider, model: &str, realtime: bool) -> Self {
        let started = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            version: LOCK_PAYLOAD_VERSION,
            owner_pid: process::id(),
            owner_started_at_unix_secs: started,
            provider: provider.to_string(),
            model: model.to_string(),
            realtime,
            owner_id: format!(
                "{}-{}-{}",
                process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos(),
                NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
            ),
        }
    }
}

// ── Local job registration ──────────────────────────────────────────

/// Process-local registry of cancellation tokens, keyed by lock
/// path.  The SIGUSR1 handler walks every entry and cancels their
/// tokens — coarse but matches the lock-file convention (the
/// signal cannot carry payload).
///
/// Multiple in-flight jobs in one process all share the same
/// SIGUSR1 handler; receiving the signal cancels them all.  In
/// practice talk-rs runs one transcription per process so this
/// is fine.
static LOCAL_REGISTRY: OnceLock<Mutex<Vec<RegisteredJob>>> = OnceLock::new();

struct RegisteredJob {
    id: String,
    cancel: CancellationToken,
}

fn local_registry() -> &'static Mutex<Vec<RegisteredJob>> {
    LOCAL_REGISTRY.get_or_init(|| {
        // First registration: install the SIGUSR1 handler that
        // cancels every in-flight token.
        install_sigusr1_handler();
        Mutex::new(Vec::new())
    })
}

static SIGUSR1_HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

fn install_sigusr1_handler() {
    if SIGUSR1_HANDLER_INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    // Install the SIGUSR1 listener SYNCHRONOUSLY before returning.
    // `tokio::signal::unix::signal` registers the handler at call
    // time, so by the time this function returns the signal will
    // be queued (not terminate the process) even if the spawned
    // task hasn't started polling yet.
    use tokio::signal::unix::{signal, SignalKind};
    let mut sig = match signal(SignalKind::user_defined1()) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("jobs: failed to install SIGUSR1 handler: {}", e);
            return;
        }
    };
    // Spawn the actual polling loop on the tokio runtime.
    tokio::spawn(async move {
        loop {
            sig.recv().await;
            log::info!("jobs: SIGUSR1 received, cancelling all in-flight jobs");
            if let Ok(g) = local_registry().lock() {
                for entry in g.iter() {
                    entry.cancel.cancel();
                }
            }
        }
    });
}

/// RAII guard returned by [`register_local`].
///
/// Holds the cancellation token, the lock-file path, and a
/// dedup-pointer into the process-local registry.  On `Drop`:
///
/// 1. Remove the lock file (idempotent — already-gone is fine).
/// 2. Deregister from the local registry so SIGUSR1 doesn't
///    cancel a token whose owner already shipped.
///
/// The cancellation token is exposed via [`Self::cancel_token`]
/// for the transport layer to consume.
pub struct LocalJob {
    cancel: CancellationToken,
    lock_path: PathBuf,
    id: String,
}

impl LocalJob {
    /// Cancellation token to pass to
    /// [`super::transport::http_request`] /
    /// [`super::transport::ws_upgrade`].  Fires when either:
    ///
    /// * the owner process explicitly cancels (e.g. user hits a
    ///   Stop button in the picker that called this transcription
    ///   itself), OR
    /// * another talk-rs process sends SIGUSR1 to the owner (e.g.
    ///   the user opens the picker after the transcription was
    ///   started elsewhere and clicks Stop).
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// Path of the on-disk lock file.  Exposed for tests and
    /// diagnostic dumps; production callers should not depend on
    /// this path.
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }
}

impl Drop for LocalJob {
    fn drop(&mut self) {
        // Best-effort lock-file removal.  Missing file is fine
        // (the user may have already cancelled).
        if observe_remote(&self.lock_path).is_some_and(|remote| remote.payload.owner_id == self.id)
        {
            let _ = std::fs::remove_file(&self.lock_path);
        }

        // Deregister by stable identity, independent of swap_remove ordering.
        if let Ok(mut g) = local_registry().lock() {
            if let Some(pos) = g.iter().position(|e| e.id == self.id) {
                g.swap_remove(pos);
            }
        }
    }
}

/// Register an in-flight job for the given (recording, provider,
/// model, mode) tuple.
///
/// Writes the lock file with a YAML payload describing this owner.
/// Installs the SIGUSR1 handler (process-wide, idempotent) so
/// out-of-process [`cancel_remote`] calls reach the returned
/// cancellation token.
///
/// Returns `Err(TalkError::ModelInProgress)` if the lock is already
/// held — matches the legacy [`recording_cache::acquire_model_lock`]
/// contract.
///
/// The caller MUST hold the returned [`LocalJob`] for the
/// transcription's full duration: dropping it removes the lock
/// file and deregisters the cancellation token.
pub fn register_local(
    audio_path: &Path,
    provider: Provider,
    model: &str,
    realtime: bool,
) -> Result<LocalJob, TalkError> {
    let lock_path = recording_cache::model_lock_path_public(audio_path, provider, model, realtime)?;
    let payload = LockPayload::new(provider, model, realtime);
    let yaml = serde_yaml::to_string(&payload)
        .map_err(|e| TalkError::Config(format!("failed to serialize lock payload: {}", e)))?;

    // Atomic write: write to a temp file alongside the lock, then
    // rename into place.  Avoids a partially-visible payload to a
    // racing observer.
    let tmp = {
        let mut p = lock_path.clone();
        let new_name = format!(
            "{}.{}.tmp",
            p.file_name().and_then(|n| n.to_str()).unwrap_or("lock"),
            payload.owner_id
        );
        p.set_file_name(new_name);
        p
    };
    std::fs::write(&tmp, yaml.as_bytes())
        .map_err(|e| TalkError::Config(format!("failed to write {}: {}", tmp.display(), e)))?;
    let linked = std::fs::hard_link(&tmp, &lock_path);
    let _ = std::fs::remove_file(&tmp);
    match linked {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(TalkError::ModelInProgress)
        }
        Err(e) => {
            return Err(TalkError::Config(format!(
                "failed to install lock {}: {}",
                lock_path.display(),
                e
            )))
        }
    }

    let cancel = CancellationToken::new();
    {
        let mut g = local_registry().lock().map_err(|_| {
            let _ = std::fs::remove_file(&lock_path);
            TalkError::Config("jobs registry mutex poisoned".into())
        })?;
        g.push(RegisteredJob {
            id: payload.owner_id.clone(),
            cancel: cancel.clone(),
        });
    }

    Ok(LocalJob {
        cancel,
        lock_path,
        id: payload.owner_id,
    })
}

// ── Cross-process observation ───────────────────────────────────────

/// Describes a remote in-flight job as seen by a non-owning
/// observer (typically a freshly-opened picker that lists
/// `*.lock` files for the recording).
#[derive(Debug, Clone)]
pub struct RemoteJob {
    /// Parsed YAML payload from the lock file.
    pub payload: LockPayload,
    /// Original lock-file path (the observer uses this to
    /// disambiguate between multiple per-model locks for the
    /// same recording).
    pub lock_path: PathBuf,
}

impl RemoteJob {
    /// Whether the owner process is still alive.
    ///
    /// Cheap heuristic: read `/proc/<pid>` existence.  PID reuse
    /// is not perfectly detected here; for paranoid checks
    /// callers should also compare
    /// [`LockPayload::owner_started_at_unix_secs`] against
    /// `/proc/<pid>/stat`.  In practice the lock files are
    /// short-lived enough that a same-second PID-reuse collision
    /// is statistically negligible.
    pub fn owner_alive(&self) -> bool {
        let proc_path = format!("/proc/{}", self.payload.owner_pid);
        std::path::Path::new(&proc_path).exists()
    }
}

/// List the in-flight lock files for a given recording.
///
/// Globs the recording's parent directory for files matching
/// `<stem>_*.lock`.  Each match is parsed; lock files with an
/// unreadable or version-incompatible payload are skipped (the
/// observer treats them as "in flight, but not observable").
pub fn list_in_flight_for(audio_path: &Path) -> Vec<RemoteJob> {
    let dir = match audio_path.parent() {
        Some(d) => d,
        None => return Vec::new(),
    };
    let stem = match audio_path.file_stem().and_then(|s| s.to_str()) {
        Some(s) => s,
        None => return Vec::new(),
    };
    let prefix = format!("{}_", stem);

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if !name.starts_with(&prefix) || !name.ends_with("-lock.yml") {
            continue;
        }
        if let Some(remote) = observe_remote(&path) {
            out.push(remote);
        }
    }
    out
}

/// Read a single lock file and return its parsed [`RemoteJob`].
///
/// Returns `None` if the file is unreadable, empty (legacy
/// pre-Step-11 lock — the file exists but carries no payload),
/// malformed YAML, or has an unsupported `version`.  These cases
/// all mean "lock exists, but this binary cannot observe it
/// safely" — the observer should treat the slot as in-flight but
/// uncancellable from this process.
pub fn observe_remote(lock_path: &Path) -> Option<RemoteJob> {
    let bytes = std::fs::read(lock_path).ok()?;
    if bytes.is_empty() {
        // Legacy empty lock — pre-Step-11 owner; we cannot
        // safely cancel it because we don't know the PID.
        return None;
    }
    let payload: LockPayload = serde_yaml::from_slice(&bytes).ok()?;
    if payload.version > LOCK_PAYLOAD_VERSION {
        log::warn!(
            "jobs: lock at {} has version {} > {}, refusing to attach",
            lock_path.display(),
            payload.version,
            LOCK_PAYLOAD_VERSION
        );
        return None;
    }
    Some(RemoteJob {
        payload,
        lock_path: lock_path.to_path_buf(),
    })
}

/// Send a cancellation request to the owner of a remote job.
///
/// Implemented as `kill(owner_pid, SIGUSR1)`.  The owner's
/// SIGUSR1 handler (installed by the first
/// [`register_local`] in its process) triggers the owner's
/// [`CancellationToken`] which the transport layer observes.
///
/// Returns `Ok(())` on signal send (regardless of whether the
/// owner reacts — the owner might already have completed).  An
/// `Err` indicates the signal could not be delivered (typically
/// because the PID does not exist any more — the lock is stale).
pub fn cancel_remote(remote: &RemoteJob) -> Result<(), TalkError> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    let pid = Pid::from_raw(remote.payload.owner_pid as i32);
    kill(pid, Signal::SIGUSR1).map_err(|e| {
        TalkError::Config(format!(
            "failed to signal pid {}: {}",
            remote.payload.owner_pid, e
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn tmp_audio_path(dir: &TempDir, name: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::File::create(&path).expect("create test audio file");
        path
    }

    #[tokio::test]
    async fn simultaneous_registration_has_exactly_one_owner() {
        let dir = TempDir::new().expect("tempdir");
        let audio = tmp_audio_path(&dir, "race.ogg");
        for _ in 0..30 {
            let gate = Arc::new(std::sync::Barrier::new(3));
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    let audio = audio.clone();
                    let gate = gate.clone();
                    let runtime = tokio::runtime::Handle::current();
                    std::thread::spawn(move || {
                        let _runtime = runtime.enter();
                        gate.wait();
                        let result = register_local(&audio, Provider::Mistral, "voxtral", false);
                        gate.wait();
                        result
                    })
                })
                .collect();
            gate.wait();
            gate.wait();
            let results: Vec<_> = handles
                .into_iter()
                .map(|h| h.join().expect("thread"))
                .collect();
            assert_eq!(
                results.iter().filter(|r| r.is_ok()).count(),
                1,
                "expected exactly one owner, got {:?}",
                results
                    .iter()
                    .map(|r| r
                        .as_ref()
                        .map(|job| job.id.as_str())
                        .map_err(|e| e.to_string()))
                    .collect::<Vec<_>>()
            );
            assert!(results
                .iter()
                .any(|r| matches!(r, Err(TalkError::ModelInProgress))));
            drop(results);
        }
    }

    #[tokio::test]
    async fn old_guard_does_not_remove_replacement_lock() {
        let dir = TempDir::new().expect("tempdir");
        let audio = tmp_audio_path(&dir, "replace.ogg");
        let guard = register_local(&audio, Provider::Mistral, "voxtral", false).expect("register");
        let path = guard.lock_path().to_path_buf();
        let replacement = LockPayload::new(Provider::OpenAI, "replacement", false);
        std::fs::write(&path, serde_yaml::to_string(&replacement).expect("yaml"))
            .expect("replacement");
        drop(guard);
        assert_eq!(
            observe_remote(&path)
                .expect("replacement preserved")
                .payload
                .model,
            "replacement"
        );
    }

    #[tokio::test]
    async fn dropping_middle_then_last_keeps_only_first_registered() {
        let dir = TempDir::new().expect("tempdir");
        let audio = tmp_audio_path(&dir, "registry.ogg");
        let a = register_local(&audio, Provider::Mistral, "a", false).expect("a");
        let b = register_local(&audio, Provider::Mistral, "b", false).expect("b");
        let c = register_local(&audio, Provider::Mistral, "c", false).expect("c");
        let ids = [a.id.clone(), b.id.clone(), c.id.clone()];
        drop(b);
        drop(c);
        let g = local_registry().lock().expect("registry");
        let remaining: Vec<_> = g
            .iter()
            .filter(|e| ids.contains(&e.id))
            .map(|e| e.id.as_str())
            .collect();
        assert_eq!(
            remaining,
            vec![a.id.as_str()],
            "only A should be registered"
        );
        drop(g);
        drop(a);
    }

    #[tokio::test]
    async fn register_local_writes_yaml_payload_with_pid() {
        let dir = TempDir::new().expect("tempdir");
        let audio = tmp_audio_path(&dir, "rec.ogg");

        let job = register_local(&audio, Provider::Mistral, "voxtral-mini-2602", false)
            .expect("register_local");

        let yaml = std::fs::read_to_string(job.lock_path()).expect("read lock");
        let parsed: LockPayload = serde_yaml::from_str(&yaml).expect("parse lock");
        assert_eq!(parsed.owner_pid, std::process::id());
        assert_eq!(parsed.version, LOCK_PAYLOAD_VERSION);
        assert_eq!(parsed.provider, "mistral");
        assert_eq!(parsed.model, "voxtral-mini-2602");
        assert!(!parsed.realtime);
    }

    #[tokio::test]
    async fn register_local_removes_lock_on_drop() {
        let dir = TempDir::new().expect("tempdir");
        let audio = tmp_audio_path(&dir, "rec.ogg");

        let lock_path = {
            let job = register_local(&audio, Provider::Mistral, "voxtral-mini-2602", false)
                .expect("register_local");
            assert!(job.lock_path().exists());
            job.lock_path().to_path_buf()
        };
        // After drop, the lock file is gone.
        assert!(!lock_path.exists());
    }

    #[tokio::test]
    async fn list_in_flight_for_finds_registered_job() {
        let dir = TempDir::new().expect("tempdir");
        let audio = tmp_audio_path(&dir, "rec.ogg");

        let _job = register_local(&audio, Provider::Mistral, "voxtral-mini-2602", false)
            .expect("register_local");

        let in_flight = list_in_flight_for(&audio);
        assert_eq!(in_flight.len(), 1);
        assert_eq!(in_flight[0].payload.owner_pid, std::process::id());
        assert!(in_flight[0].owner_alive());
    }

    #[tokio::test]
    async fn listing_ignores_other_recordings_and_unobservable_locks() {
        let dir = TempDir::new().expect("tempdir");
        let audio = tmp_audio_path(&dir, "target.ogg");
        let other_audio = tmp_audio_path(&dir, "other.ogg");
        let valid =
            register_local(&audio, Provider::OpenAI, "whisper-1", false).expect("valid lock");
        let _other = register_local(&other_audio, Provider::Mistral, "voxtral", false)
            .expect("other recording lock");
        std::fs::write(dir.path().join("target_bad-lock.yml"), "not: [yaml")
            .expect("malformed lock");
        std::fs::write(dir.path().join("target_legacy-lock.yml"), "").expect("legacy lock");
        let mut future = LockPayload::new(Provider::OpenAI, "future", false);
        future.version = LOCK_PAYLOAD_VERSION + 1;
        std::fs::write(
            dir.path().join("target_future-lock.yml"),
            serde_yaml::to_string(&future).expect("future yaml"),
        )
        .expect("future lock");

        let observed = list_in_flight_for(&audio);
        assert_eq!(observed.len(), 1);
        assert_eq!(observed[0].lock_path, valid.lock_path());
        assert_eq!(observed[0].payload.model, "whisper-1");
        assert_eq!(observed[0].payload.provider, "openai");
    }

    #[tokio::test]
    async fn observe_remote_returns_none_for_legacy_empty_lock() {
        let dir = TempDir::new().expect("tempdir");
        let lock_path = dir.path().join("empty.lock");
        std::fs::write(&lock_path, b"").expect("write empty lock");

        assert!(observe_remote(&lock_path).is_none());
    }

    #[tokio::test]
    async fn observe_remote_returns_none_for_future_version() {
        let dir = TempDir::new().expect("tempdir");
        let lock_path = dir.path().join("future.lock");
        let payload = "version: 999\nowner_pid: 1\nowner_started_at_unix_secs: 0\n\
             provider: mistral\nmodel: x\nrealtime: false\n";
        std::fs::write(&lock_path, payload).expect("write future lock");

        assert!(observe_remote(&lock_path).is_none());
    }

    /// NOTE: SIGUSR1 is process-wide.  When the lib test binary
    /// runs tests in parallel, an earlier test's
    /// [`register_local`] call may absorb the signal sent by
    /// this test, leaving our token uncancelled.  Marked
    /// `#[ignore]` so the default ``cargo test`` run skips it;
    /// run explicitly in isolation with
    /// ``cargo test cancel_remote_via_sigusr1 -- --ignored``.
    #[tokio::test]
    #[ignore = "SIGUSR1 is process-wide; run in isolation"]
    async fn cancel_remote_via_sigusr1_triggers_owner_token() {
        let dir = TempDir::new().expect("tempdir");
        let audio = tmp_audio_path(&dir, "rec.ogg");

        let job = register_local(&audio, Provider::Mistral, "voxtral-mini-2602", false)
            .expect("register_local");
        let token = job.cancel_token();
        assert!(!token.is_cancelled());

        // Simulate a remote observer in the same process: list the
        // lock, then cancel its owner.  Same-PID SIGUSR1 still
        // routes through the installed handler.
        let in_flight = list_in_flight_for(&audio);
        assert_eq!(in_flight.len(), 1);
        cancel_remote(&in_flight[0]).expect("cancel_remote");

        // Allow the tokio signal task to run.  In #[tokio::test]
        // the signal driver runs on the same multi-threaded
        // runtime as the test, so we need to yield enough.
        for _ in 0..100 {
            if token.is_cancelled() {
                break;
            }
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(
            token.is_cancelled(),
            "owner's cancellation token must fire on SIGUSR1"
        );
    }
}
