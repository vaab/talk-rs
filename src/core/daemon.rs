//! Daemon process management for toggle mode.
//!
//! Provides PID file management with kernel-level locking (flock)
//! and process lifecycle control (start, stop, stale detection).

use crate::core::error::TalkError;
use directories::ProjectDirs;
use nix::fcntl::{Flock, FlockArg};
use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Interval between liveness checks when waiting for daemon to exit.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Maximum time to wait for graceful shutdown before escalating to SIGTERM.
const GRACEFUL_TIMEOUT: Duration = Duration::from_secs(10);

/// Status of the daemon process.
#[derive(Debug, PartialEq, Eq)]
pub enum DaemonStatus {
    /// No daemon is running (no PID file or stale PID).
    NotRunning,
    /// A daemon is running with the given PID.
    Running { pid: u32 },
}

/// Get the cache directory for talk-rs (`$XDG_CACHE_HOME/talk-rs/`).
pub fn cache_dir() -> Result<PathBuf, TalkError> {
    ProjectDirs::from("org", "kalysto", "talk-rs")
        .map(|dirs| dirs.cache_dir().to_path_buf())
        .ok_or_else(|| TalkError::Config("could not determine cache directory".to_string()))
}

/// Get the default PID file path (`$XDG_CACHE_HOME/talk-rs/daemon.pid`).
pub fn pid_path() -> Result<PathBuf, TalkError> {
    Ok(cache_dir()?.join("daemon.pid"))
}

/// Get the daemon log file path (`$XDG_CACHE_HOME/talk-rs/daemon.log`).
pub fn log_path() -> Result<PathBuf, TalkError> {
    Ok(cache_dir()?.join("daemon.log"))
}

/// Append a debug trace line to the daemon log.
///
/// Best-effort: silently does nothing if the log path cannot be
/// resolved or the file cannot be opened.  Useful for writing
/// diagnostic messages from the toggle process whose stderr is
/// invisible when invoked via a keybinding.
pub fn trace(msg: &str) {
    if let Ok(path) = log_path() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&path) {
            let _ = writeln!(f, "{msg}");
        }
    }
}

/// Get the lock file path (`$XDG_CACHE_HOME/talk-rs/daemon.lock`).
fn lock_path() -> Result<PathBuf, TalkError> {
    Ok(cache_dir()?.join("daemon.lock"))
}

/// Ensure the cache directory exists.
fn ensure_cache_dir() -> Result<(), TalkError> {
    let dir = cache_dir()?;
    fs::create_dir_all(&dir).map_err(|e| {
        TalkError::Config(format!(
            "failed to create cache directory {}: {}",
            dir.display(),
            e
        ))
    })
}

/// Acquire an exclusive kernel-level lock on the lock file.
///
/// Returns a `Flock<File>` — the lock is released automatically when dropped.
pub fn acquire_lock() -> Result<Flock<fs::File>, TalkError> {
    ensure_cache_dir()?;
    let path = lock_path()?;
    let file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .map_err(|e| {
            TalkError::Config(format!(
                "failed to open lock file {}: {}",
                path.display(),
                e
            ))
        })?;

    Flock::lock(file, FlockArg::LockExclusive).map_err(|(_, e)| {
        TalkError::Config(format!(
            "failed to acquire lock on {}: {}",
            path.display(),
            e
        ))
    })
}

/// Check if a process is alive using `kill(pid, 0)`.
fn is_process_alive(pid: u32) -> bool {
    let nix_pid = Pid::from_raw(pid as i32);
    kill(nix_pid, None).is_ok()
}

/// Read the PID from the PID file, returning `None` if the file doesn't exist.
fn read_pid_file(path: &Path) -> Result<Option<u32>, TalkError> {
    match fs::read_to_string(path) {
        Ok(content) => {
            let pid = content.trim().parse::<u32>().map_err(|e| {
                TalkError::Config(format!("invalid PID in {}: {}", path.display(), e))
            })?;
            Ok(Some(pid))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(TalkError::Config(format!(
            "failed to read PID file {}: {}",
            path.display(),
            e
        ))),
    }
}

/// Check daemon status from the PID file.
///
/// If the PID file exists but the process is dead, removes the stale file.
/// Caller MUST hold the lock before calling this.
pub fn check_status(pid_file: &Path) -> Result<DaemonStatus, TalkError> {
    match read_pid_file(pid_file)? {
        None => Ok(DaemonStatus::NotRunning),
        Some(pid) => {
            if is_process_alive(pid) {
                Ok(DaemonStatus::Running { pid })
            } else {
                // Stale PID file — process is dead, clean up
                remove_pid_file(pid_file)?;
                Ok(DaemonStatus::NotRunning)
            }
        }
    }
}

/// Write a PID to the PID file.
///
/// Creates parent directories if needed. Caller MUST hold the lock.
pub fn write_pid_file(pid_file: &Path, pid: u32) -> Result<(), TalkError> {
    if let Some(parent) = pid_file.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            TalkError::Config(format!(
                "failed to create directory {}: {}",
                parent.display(),
                e
            ))
        })?;
    }

    fs::write(pid_file, format!("{}\n", pid)).map_err(|e| {
        TalkError::Config(format!(
            "failed to write PID file {}: {}",
            pid_file.display(),
            e
        ))
    })
}

/// Remove the PID file.
pub fn remove_pid_file(pid_file: &Path) -> Result<(), TalkError> {
    match fs::remove_file(pid_file) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()), // Already gone
        Err(e) => Err(TalkError::Config(format!(
            "failed to remove PID file {}: {}",
            pid_file.display(),
            e
        ))),
    }
}

/// Remove the PID file only if it still belongs to `expected_pid`.
///
/// Re-acquires the lock, reads the file, and removes it only when the
/// stored PID matches.  Returns `true` if the file was removed, `false`
/// if it was missing or belonged to a different process.
pub fn remove_pid_file_if_owner(expected_pid: u32, pid_file: &Path) -> Result<bool, TalkError> {
    let _lock = acquire_lock()?;

    match read_pid_file(pid_file)? {
        Some(current_pid) if current_pid == expected_pid => {
            remove_pid_file(pid_file)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Stop a daemon **only** if the PID file still contains the expected PID.
///
/// This is safe to call after releasing the lock: it re-acquires the lock,
/// verifies ownership, then delegates to [`stop_daemon`].  If the PID file
/// is missing or belongs to a different daemon the process is killed
/// directly (SIGINT) without touching the PID file.
///
/// Returns `true` if the PID file was ours and was cleaned up, `false` if
/// another daemon now owns it.
pub fn stop_if_owner(expected_pid: u32, pid_file: &Path) -> Result<bool, TalkError> {
    let _lock = acquire_lock()?;

    match read_pid_file(pid_file)? {
        Some(current_pid) if current_pid == expected_pid => {
            // PID file still ours — full graceful stop.
            stop_daemon(expected_pid, pid_file)?;
            Ok(true)
        }
        _ => {
            // PID file gone or recycled for a different daemon.
            // Just kill our process directly without touching the file.
            let _ = kill(Pid::from_raw(expected_pid as i32), Signal::SIGINT);
            Ok(false)
        }
    }
}

/// Signal a running daemon to stop (SIGINT) and remove its PID file.
///
/// Unlike [`stop_daemon`], this returns immediately without waiting for the
/// process to exit.  The daemon finishes its current work (transcription,
/// paste) and exits gracefully on its own.  Its cleanup path uses
/// [`remove_pid_file_if_owner`] so it will not clobber a PID file written
/// by a newly spawned daemon.
///
/// Caller MUST hold the lock.
pub fn signal_daemon(pid: u32, pid_file: &Path) -> Result<(), TalkError> {
    let nix_pid = Pid::from_raw(-(pid as i32));

    // Helper: append a debug trace to daemon.log (the toggle process
    // does not share the daemon's stderr, so log::warn is invisible).
    let dbg = |msg: &str| {
        if let Ok(lp) = log_path() {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&lp) {
                let _ = writeln!(f, "{}", msg);
            }
        }
    };

    // Send SIGINT to the process group.
    dbg(&format!("[DBG] signal_daemon: kill(-{}, SIGINT)", pid));
    if let Err(e) = kill(nix_pid, Signal::SIGINT) {
        if e == nix::errno::Errno::ESRCH {
            // Process group gone — try individual PID.
            dbg(&format!(
                "[DBG] signal_daemon: ESRCH on group, trying kill({}, SIGINT)",
                pid
            ));
            let individual = Pid::from_raw(pid as i32);
            if let Err(e2) = kill(individual, Signal::SIGINT) {
                if e2 == nix::errno::Errno::ESRCH {
                    // Already dead — just clean up PID file.
                    dbg(&format!(
                        "[DBG] signal_daemon: PID {} already dead (ESRCH)",
                        pid
                    ));
                    remove_pid_file(pid_file)?;
                    return Ok(());
                }
                return Err(TalkError::Config(format!(
                    "failed to send SIGINT to PID {}: {}",
                    pid, e2
                )));
            }
            dbg(&format!(
                "[DBG] signal_daemon: kill({}, SIGINT) OK (fallback)",
                pid
            ));
        } else {
            return Err(TalkError::Config(format!(
                "failed to send SIGINT to process group {}: {}",
                pid, e
            )));
        }
    } else {
        dbg(&format!("[DBG] signal_daemon: kill(-{}, SIGINT) OK", pid));
    }

    // Remove PID file immediately so the next toggle-on sees NotRunning.
    // The exiting daemon uses remove_pid_file_if_owner which is safe
    // against this pre-removal and against a new daemon's PID file.
    remove_pid_file(pid_file)?;
    Ok(())
}

/// Stop a running daemon by sending SIGINT, waiting, then escalating to SIGTERM.
///
/// Signals the entire process group (negative PID). Caller MUST hold the lock.
pub fn stop_daemon(pid: u32, pid_file: &Path) -> Result<(), TalkError> {
    let nix_pid = Pid::from_raw(-(pid as i32));

    // Phase 1: SIGINT (graceful)
    if let Err(e) = kill(nix_pid, Signal::SIGINT) {
        // If process group doesn't exist, try individual PID
        if e == nix::errno::Errno::ESRCH {
            let individual = Pid::from_raw(pid as i32);
            if let Err(e2) = kill(individual, Signal::SIGINT) {
                if e2 == nix::errno::Errno::ESRCH {
                    // Already dead
                    remove_pid_file(pid_file)?;
                    return Ok(());
                }
                return Err(TalkError::Config(format!(
                    "failed to send SIGINT to PID {}: {}",
                    pid, e2
                )));
            }
        } else {
            return Err(TalkError::Config(format!(
                "failed to send SIGINT to process group {}: {}",
                pid, e
            )));
        }
    }

    // Phase 2: Wait for exit
    let deadline = Instant::now() + GRACEFUL_TIMEOUT;
    loop {
        if !is_process_alive(pid) {
            break;
        }
        if Instant::now() >= deadline {
            // Phase 3: SIGTERM (forceful)
            log::warn!(
                "daemon did not exit within {}s, sending SIGTERM",
                GRACEFUL_TIMEOUT.as_secs()
            );
            let _ = kill(nix_pid, Signal::SIGTERM);
            // Also try individual PID
            let individual = Pid::from_raw(pid as i32);
            let _ = kill(individual, Signal::SIGTERM);

            // Brief wait after SIGTERM
            std::thread::sleep(Duration::from_secs(1));
            break;
        }
        std::thread::sleep(POLL_INTERVAL);
    }

    remove_pid_file(pid_file)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_pid_path(dir: &TempDir) -> PathBuf {
        dir.path().join("daemon.pid")
    }

    #[test]
    fn test_check_status_no_file() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);
        let status = check_status(&path).expect("check status");
        assert_eq!(status, DaemonStatus::NotRunning);
    }

    #[test]
    fn test_write_and_read_pid_file() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        write_pid_file(&path, 12345).expect("write pid");

        let content = fs::read_to_string(&path).expect("read file");
        assert_eq!(content.trim(), "12345");
    }

    #[test]
    fn test_check_status_stale_pid() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        // Write a PID that almost certainly doesn't exist
        write_pid_file(&path, 4_000_000).expect("write pid");

        let status = check_status(&path).expect("check status");
        assert_eq!(status, DaemonStatus::NotRunning);

        // Stale file should have been cleaned up
        assert!(!path.exists());
    }

    #[test]
    fn test_check_status_running_pid() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        // Use our own PID (guaranteed alive)
        let our_pid = std::process::id();
        write_pid_file(&path, our_pid).expect("write pid");

        let status = check_status(&path).expect("check status");
        assert_eq!(status, DaemonStatus::Running { pid: our_pid });
    }

    #[test]
    fn test_remove_pid_file_nonexistent() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        // Should not error on missing file
        remove_pid_file(&path).expect("remove nonexistent");
    }

    #[test]
    fn test_remove_pid_file_existing() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        write_pid_file(&path, 12345).expect("write pid");
        assert!(path.exists());

        remove_pid_file(&path).expect("remove");
        assert!(!path.exists());
    }

    #[test]
    fn test_read_pid_file_invalid_content() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        fs::write(&path, "not-a-number\n").expect("write");
        let result = read_pid_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_process_alive_self() {
        assert!(is_process_alive(std::process::id()));
    }

    #[test]
    fn test_is_process_alive_dead() {
        // PID 4000000 is almost certainly not alive
        assert!(!is_process_alive(4_000_000));
    }

    #[test]
    fn test_signal_daemon_dead_process() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        // Use a PID that is dead so SIGINT is harmless.
        let fake_pid: u32 = 4_000_000;
        write_pid_file(&path, fake_pid).expect("write pid");

        signal_daemon(fake_pid, &path).expect("signal_daemon");
        assert!(
            !path.exists(),
            "PID file should be removed after signal_daemon"
        );
    }

    #[test]
    fn test_signal_daemon_no_pid_file() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        // PID file does not exist — signal_daemon should still succeed
        // (the SIGINT to a dead PID yields ESRCH which triggers the
        // "already dead" path that calls remove_pid_file, which is a
        // no-op on a missing file).
        signal_daemon(4_000_000, &path).expect("signal_daemon");
    }

    #[test]
    fn test_stop_if_owner_matching_pid() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        // Use a PID that is dead so SIGINT is harmless
        let fake_pid: u32 = 4_000_000;
        write_pid_file(&path, fake_pid).expect("write pid");

        let owned = stop_if_owner(fake_pid, &path).expect("stop_if_owner");
        assert!(owned, "should report ownership when PID matches");
        assert!(
            !path.exists(),
            "PID file should be removed after owned stop"
        );
    }

    #[test]
    fn test_stop_if_owner_different_pid() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        // PID file contains a *different* PID than the one we claim to own.
        let file_pid: u32 = 4_000_001;
        let our_pid: u32 = 4_000_000;
        write_pid_file(&path, file_pid).expect("write pid");

        let owned = stop_if_owner(our_pid, &path).expect("stop_if_owner");
        assert!(!owned, "should report non-ownership when PIDs differ");
        // PID file must still exist — it belongs to the other daemon.
        assert!(
            path.exists(),
            "PID file should be preserved for the other daemon"
        );

        let stored = read_pid_file(&path).expect("read").expect("some pid");
        assert_eq!(stored, file_pid, "PID file content must be untouched");
    }

    #[test]
    fn test_stop_if_owner_missing_pid_file() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        // No PID file at all — should not panic, just return false.
        let owned = stop_if_owner(4_000_000, &path).expect("stop_if_owner");
        assert!(
            !owned,
            "should report non-ownership when PID file is missing"
        );
    }

    #[test]
    fn test_remove_pid_file_if_owner_matching() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        write_pid_file(&path, 12345).expect("write pid");
        let removed = remove_pid_file_if_owner(12345, &path).expect("remove_pid_file_if_owner");
        assert!(removed, "should remove when PID matches");
        assert!(!path.exists(), "PID file should be gone");
    }

    #[test]
    fn test_remove_pid_file_if_owner_different() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        write_pid_file(&path, 12345).expect("write pid");
        let removed = remove_pid_file_if_owner(99999, &path).expect("remove_pid_file_if_owner");
        assert!(!removed, "should not remove when PID differs");
        assert!(path.exists(), "PID file should remain");
    }

    #[test]
    fn test_remove_pid_file_if_owner_missing() {
        let dir = TempDir::new().expect("create temp dir");
        let path = test_pid_path(&dir);

        // No PID file — should succeed with false.
        let removed = remove_pid_file_if_owner(12345, &path).expect("remove_pid_file_if_owner");
        assert!(!removed, "should return false when PID file is missing");
    }
}
