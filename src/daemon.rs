//! Daemon process management for toggle mode.
//!
//! Provides PID file management with kernel-level locking (flock)
//! and process lifecycle control (start, stop, stale detection).

use crate::error::TalkError;
use directories::ProjectDirs;
use nix::fcntl::{Flock, FlockArg};
use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use std::fs;
use std::future::Future;
use std::marker::PhantomData;
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

const UNPUBLISHED_CHILD_GRACE: Duration = Duration::from_millis(250);
const UNPUBLISHED_CHILD_POLL: Duration = Duration::from_millis(10);

/// Status of the daemon process.
#[derive(Debug, PartialEq, Eq)]
pub enum DaemonStatus {
    /// No daemon is running (no PID file or stale PID).
    NotRunning,
    /// A daemon is running with the given PID.
    Running { pid: u32 },
}

/// When a successful toggle-off releases the namespace PID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleasePolicy {
    /// Release immediately so dictation can transcribe and paste in parallel
    /// with a subsequent recording.
    Immediate,
    /// Keep ownership until the child exits after durable finalization.
    OnChildExit,
}

mod private {
    pub trait Sealed {}
}

/// Compile-time namespace for one daemon lifecycle.
pub trait DaemonNamespace: private::Sealed {
    const STEM: &'static str;
    const DESCRIPTION: &'static str;
    const RELEASE_POLICY: ReleasePolicy;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DictateDaemon;

impl private::Sealed for DictateDaemon {}

impl DaemonNamespace for DictateDaemon {
    const STEM: &'static str = "daemon";
    const DESCRIPTION: &'static str = "dictation";
    const RELEASE_POLICY: ReleasePolicy = ReleasePolicy::Immediate;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordDaemon;

impl private::Sealed for RecordDaemon {}

impl DaemonNamespace for RecordDaemon {
    const STEM: &'static str = "record";
    const DESCRIPTION: &'static str = "recording";
    const RELEASE_POLICY: ReleasePolicy = ReleasePolicy::OnChildExit;
}

/// Namespaced daemon state. The marker type binds its PID, lock, and log.
pub struct DaemonSlot<S: DaemonNamespace> {
    directory: PathBuf,
    marker: PhantomData<S>,
}

impl<S: DaemonNamespace> Clone for DaemonSlot<S> {
    fn clone(&self) -> Self {
        Self {
            directory: self.directory.clone(),
            marker: PhantomData,
        }
    }
}

impl<S: DaemonNamespace> DaemonSlot<S> {
    fn in_dir(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            marker: PhantomData,
        }
    }

    fn state_path(&self, extension: &str) -> PathBuf {
        self.directory.join(format!("{}.{}", S::STEM, extension))
    }

    fn pid_path(&self) -> PathBuf {
        self.state_path("pid")
    }

    fn lock_path(&self) -> PathBuf {
        self.state_path("lock")
    }

    fn log_path(&self) -> PathBuf {
        self.state_path("log")
    }

    pub fn acquire_lock(&self) -> Result<DaemonLock<S>, TalkError> {
        fs::create_dir_all(&self.directory).map_err(|error| {
            TalkError::Config(format!(
                "failed to create cache directory {}: {}",
                self.directory.display(),
                error
            ))
        })?;
        let lock_path = self.lock_path();
        let file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|error| {
                TalkError::Config(format!(
                    "failed to open lock file {}: {}",
                    lock_path.display(),
                    error
                ))
            })?;
        let lock = Flock::lock(file, FlockArg::LockExclusive).map_err(|(_, error)| {
            TalkError::Config(format!(
                "failed to acquire lock on {}: {}",
                lock_path.display(),
                error
            ))
        })?;
        Ok(DaemonLock {
            slot: self.clone(),
            _lock: lock,
        })
    }

    pub fn cleanup_if_owner(&self, expected_pid: u32) -> Result<bool, TalkError> {
        let lock = self.acquire_lock()?;
        lock.cleanup_if_owner(expected_pid)
    }

    pub async fn run_as_owner<F, T>(&self, work: F) -> T
    where
        F: Future<Output = T>,
    {
        let _guard = OwnerGuard {
            slot: self.clone(),
            pid: std::process::id(),
        };
        work.await
    }

    pub fn owner_guard(&self) -> OwnerGuard<S> {
        OwnerGuard {
            slot: self.clone(),
            pid: std::process::id(),
        }
    }

    pub fn trace(&self, message: &str) {
        use std::io::Write as _;

        let path = self.log_path();
        match fs::OpenOptions::new().append(true).open(&path) {
            Ok(mut file) => {
                if let Err(error) = writeln!(file, "{message}") {
                    log::debug!(
                        "failed to append daemon trace to {}: {}",
                        path.display(),
                        error
                    );
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                log::debug!("failed to open daemon trace {}: {}", path.display(), error);
            }
        }
    }
}

pub fn dictate_slot() -> Result<DaemonSlot<DictateDaemon>, TalkError> {
    Ok(DaemonSlot::in_dir(cache_dir()?))
}

pub fn record_slot() -> Result<DaemonSlot<RecordDaemon>, TalkError> {
    Ok(DaemonSlot::in_dir(cache_dir()?))
}

#[derive(Debug, PartialEq, Eq)]
pub struct DaemonProcess<S: DaemonNamespace> {
    pid: u32,
    marker: PhantomData<S>,
}

impl<S: DaemonNamespace> Copy for DaemonProcess<S> {}

impl<S: DaemonNamespace> Clone for DaemonProcess<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: DaemonNamespace> DaemonProcess<S> {
    pub fn pid(self) -> u32 {
        self.pid
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum SlotStatus<S: DaemonNamespace> {
    NotRunning,
    Running(DaemonProcess<S>),
}

pub struct DaemonLock<S: DaemonNamespace> {
    slot: DaemonSlot<S>,
    _lock: Flock<fs::File>,
}

impl<S: DaemonNamespace> DaemonLock<S> {
    pub fn status(&self) -> Result<SlotStatus<S>, TalkError> {
        match check_status(&self.slot.pid_path())? {
            DaemonStatus::NotRunning => Ok(SlotStatus::NotRunning),
            DaemonStatus::Running { pid } => Ok(SlotStatus::Running(DaemonProcess {
                pid,
                marker: PhantomData,
            })),
        }
    }

    /// Signal a process belonging to this lock's namespace.
    ///
    /// A process from another namespace cannot be passed here:
    ///
    /// ```compile_fail
    /// # fn example() -> Result<(), talk_rs::error::TalkError> {
    /// let dictate = talk_rs::daemon::dictate_slot()?;
    /// let record = talk_rs::daemon::record_slot()?;
    /// let dictate_lock = dictate.acquire_lock()?;
    /// let record_lock = record.acquire_lock()?;
    /// if let talk_rs::daemon::SlotStatus::Running(process) = dictate_lock.status()? {
    ///     record_lock.signal(process)?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn signal(&self, process: DaemonProcess<S>) -> Result<(), TalkError> {
        send_sigint(process.pid, &self.slot.log_path())?;
        if S::RELEASE_POLICY == ReleasePolicy::Immediate {
            self.cleanup_if_owner(process.pid)?;
        }
        Ok(())
    }

    fn cleanup_if_owner(&self, expected_pid: u32) -> Result<bool, TalkError> {
        match read_pid_file(&self.slot.pid_path())? {
            Some(current_pid) if current_pid == expected_pid => {
                remove_pid_file(&self.slot.pid_path())?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn publish_spawned_child(&self, child: &mut Child) -> Result<(), TalkError> {
        if let Err(publication_error) = write_pid_file(&self.slot.pid_path(), child.id()) {
            let cleanup_result = cleanup_unpublished_child(child);
            return match cleanup_result {
                Ok(()) => Err(publication_error),
                Err(cleanup_error) => Err(TalkError::Config(format!(
                    "{}; additionally failed to clean up unpublished child {}: {}",
                    publication_error,
                    child.id(),
                    cleanup_error
                ))),
            };
        }
        Ok(())
    }

    fn spawn_current_executable(&self, args: &[String]) -> Result<u32, TalkError> {
        let executable = std::env::current_exe().map_err(|error| {
            TalkError::Config(format!("failed to determine current executable: {error}"))
        })?;
        let log_path = self.slot.log_path();
        let log_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|error| {
                TalkError::Config(format!(
                    "failed to open log file {}: {}",
                    log_path.display(),
                    error
                ))
            })?;
        let log_stderr = log_file.try_clone().map_err(|error| {
            TalkError::Config(format!("failed to clone log file handle: {error}"))
        })?;
        let mut command = Command::new(executable);
        command
            .args(args)
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_stderr))
            .stdin(Stdio::null())
            .process_group(0);
        let mut child = command.spawn().map_err(|error| {
            TalkError::Config(format!("failed to spawn daemon process: {error}"))
        })?;
        self.publish_spawned_child(&mut child)?;
        Ok(child.id())
    }
}

pub enum ToggleOutcome {
    Started { pid: u32, log_path: PathBuf },
    Signalled { pid: u32 },
}

/// Lock one namespace and either signal its child or prepare and spawn it.
pub async fn toggle_current_executable<S, Prepare, Prepared>(
    slot: &DaemonSlot<S>,
    prepare_start: Prepare,
) -> Result<ToggleOutcome, TalkError>
where
    S: DaemonNamespace,
    Prepare: FnOnce() -> Prepared,
    Prepared: Future<Output = Result<Vec<String>, TalkError>>,
{
    let lock = slot.acquire_lock()?;
    match lock.status()? {
        SlotStatus::NotRunning => {
            let args = prepare_start().await?;
            let pid = lock.spawn_current_executable(&args)?;
            Ok(ToggleOutcome::Started {
                pid,
                log_path: slot.log_path(),
            })
        }
        SlotStatus::Running(process) => {
            let pid = process.pid();
            lock.signal(process)?;
            Ok(ToggleOutcome::Signalled { pid })
        }
    }
}

pub struct OwnerGuard<S: DaemonNamespace> {
    slot: DaemonSlot<S>,
    pid: u32,
}

impl<S: DaemonNamespace> Drop for OwnerGuard<S> {
    fn drop(&mut self) {
        if let Err(error) = self.slot.cleanup_if_owner(self.pid) {
            log::warn!(
                "failed to clean up {} daemon PID {}: {}",
                S::DESCRIPTION,
                self.pid,
                error
            );
        }
    }
}

fn cleanup_unpublished_child(child: &mut Child) -> Result<(), TalkError> {
    let pid = child.id();
    let group = Pid::from_raw(-(pid as i32));
    let _ = kill(group, Signal::SIGINT);
    let deadline = std::time::Instant::now() + UNPUBLISHED_CHILD_GRACE;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(UNPUBLISHED_CHILD_POLL);
            }
            Ok(None) => break,
            Err(error) => {
                return Err(TalkError::Config(format!(
                    "failed to query unpublished child {pid}: {error}"
                )))
            }
        }
    }
    if let Err(error) = kill(group, Signal::SIGKILL) {
        if error != nix::errno::Errno::ESRCH {
            log::debug!("failed to SIGKILL unpublished process group {pid}: {error}");
        }
    }
    if child
        .try_wait()
        .map_err(|error| {
            TalkError::Config(format!(
                "failed to query unpublished child {pid} after SIGKILL: {error}"
            ))
        })?
        .is_none()
    {
        child.kill().map_err(|error| {
            TalkError::Config(format!("failed to kill unpublished child {pid}: {error}"))
        })?;
    }
    child.wait().map_err(|error| {
        TalkError::Config(format!("failed to reap unpublished child {pid}: {error}"))
    })?;
    Ok(())
}

fn send_sigint(pid: u32, log_path: &Path) -> Result<(), TalkError> {
    use std::io::Write as _;

    let trace = |message: &str| match fs::OpenOptions::new().append(true).open(log_path) {
        Ok(mut file) => {
            if let Err(error) = writeln!(file, "{message}") {
                log::debug!("failed to append daemon trace: {error}");
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => log::debug!("failed to open daemon trace: {error}"),
    };
    let group = Pid::from_raw(-(pid as i32));
    trace(&format!("[DBG] sending SIGINT to process group {pid}"));
    if let Err(group_error) = kill(group, Signal::SIGINT) {
        if group_error != nix::errno::Errno::ESRCH {
            return Err(TalkError::Config(format!(
                "failed to send SIGINT to process group {pid}: {group_error}"
            )));
        }
        let individual = Pid::from_raw(pid as i32);
        if let Err(pid_error) = kill(individual, Signal::SIGINT) {
            if pid_error != nix::errno::Errno::ESRCH {
                return Err(TalkError::Config(format!(
                    "failed to send SIGINT to PID {pid}: {pid_error}"
                )));
            }
        }
    }
    Ok(())
}

/// Get the cache directory for talk-rs (`$XDG_CACHE_HOME/talk-rs/`).
pub fn cache_dir() -> Result<PathBuf, TalkError> {
    ProjectDirs::from("org", "kalysto", "talk-rs")
        .map(|dirs| dirs.cache_dir().to_path_buf())
        .ok_or_else(|| TalkError::Config("could not determine cache directory".to_string()))
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
fn check_status(pid_file: &Path) -> Result<DaemonStatus, TalkError> {
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
fn write_pid_file(pid_file: &Path, pid: u32) -> Result<(), TalkError> {
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
fn remove_pid_file(pid_file: &Path) -> Result<(), TalkError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    struct TestChild(std::process::Child);

    impl Drop for TestChild {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

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
    fn typed_slots_keep_state_paths_namespaced() {
        let dir = TempDir::new().expect("create temp dir");
        let dictate = DaemonSlot::<DictateDaemon>::in_dir(dir.path());
        let record = DaemonSlot::<RecordDaemon>::in_dir(dir.path());

        assert_eq!(dictate.pid_path(), dir.path().join("daemon.pid"));
        assert_eq!(dictate.lock_path(), dir.path().join("daemon.lock"));
        assert_eq!(dictate.log_path(), dir.path().join("daemon.log"));
        assert_eq!(record.pid_path(), dir.path().join("record.pid"));
        assert_eq!(record.lock_path(), dir.path().join("record.lock"));
        assert_eq!(record.log_path(), dir.path().join("record.log"));
    }

    #[test]
    fn typed_slot_status_removes_only_its_stale_pid() {
        let dir = TempDir::new().expect("create temp dir");
        let dictate = DaemonSlot::<DictateDaemon>::in_dir(dir.path());
        let record = DaemonSlot::<RecordDaemon>::in_dir(dir.path());
        fs::write(dictate.pid_path(), "4000000\n").expect("write dictate pid");
        fs::write(record.pid_path(), format!("{}\n", std::process::id()))
            .expect("write record pid");

        let dictate_lock = dictate.acquire_lock().expect("lock dictate slot");
        assert_eq!(
            dictate_lock.status().expect("dictate status"),
            SlotStatus::NotRunning
        );
        assert!(!dictate.pid_path().exists());
        assert!(record.pid_path().exists());
    }

    #[test]
    fn slot_release_policies_distinguish_dictate_and_record() {
        assert_eq!(DictateDaemon::RELEASE_POLICY, ReleasePolicy::Immediate);
        assert_eq!(RecordDaemon::RELEASE_POLICY, ReleasePolicy::OnChildExit);
    }

    #[test]
    fn finalizing_child_helper() {
        let Some(ready_path) = std::env::var_os("TALK_RS_FINALIZING_CHILD_READY") else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().expect("create helper runtime");
        runtime.block_on(async {
            let mut interrupt =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                    .expect("register helper SIGINT");
            fs::write(ready_path, b"ready").expect("publish helper readiness");
            let _ = interrupt.recv().await;
            tokio::time::sleep(Duration::from_millis(300)).await;
        });
    }

    #[test]
    fn record_stop_keeps_pid_during_repeated_finalization_signals() {
        let dir = TempDir::new().expect("create temp dir");
        let ready_path = dir.path().join("ready");
        let executable = std::env::current_exe().expect("locate test executable");
        let mut command = Command::new(executable);
        command
            .args([
                "--exact",
                "daemon::tests::finalizing_child_helper",
                "--nocapture",
            ])
            .env("TALK_RS_FINALIZING_CHILD_READY", &ready_path)
            .process_group(0);
        let mut child = TestChild(command.spawn().expect("spawn finalizing helper"));
        let pid = child.0.id();
        // Ceilings only: the polls below return as soon as the
        // condition holds.  They are generous because the helper is
        // this very test binary (large, and slow to start under
        // coverage instrumentation or a loaded parallel run).
        let ready_deadline = std::time::Instant::now() + Duration::from_secs(30);
        while !ready_path.exists() && std::time::Instant::now() < ready_deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(ready_path.exists(), "helper must register SIGINT promptly");

        let slot = DaemonSlot::<RecordDaemon>::in_dir(dir.path());
        write_pid_file(&slot.pid_path(), pid).expect("publish record helper PID");
        {
            let lock = slot.acquire_lock().expect("lock record slot");
            let SlotStatus::Running(process) = lock.status().expect("record status") else {
                panic!("record helper should be running");
            };
            lock.signal(process).expect("first record stop signal");
        }
        assert!(
            slot.pid_path().exists(),
            "first stop must retain record PID"
        );
        {
            let lock = slot.acquire_lock().expect("re-lock record slot");
            let SlotStatus::Running(process) = lock.status().expect("record status") else {
                panic!("finalizing record helper should still be running");
            };
            lock.signal(process).expect("repeated record stop signal");
        }
        assert!(
            slot.pid_path().exists(),
            "repeated stop during finalization must not release the slot"
        );

        let exit_deadline = std::time::Instant::now() + Duration::from_secs(30);
        while child.0.try_wait().expect("query helper status").is_none()
            && std::time::Instant::now() < exit_deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            child
                .0
                .try_wait()
                .expect("query final helper status")
                .is_some(),
            "helper should finish finalization"
        );
        assert!(slot.cleanup_if_owner(pid).expect("cleanup helper PID"));
    }

    #[test]
    fn owner_cleanup_never_removes_another_slots_pid() {
        let dir = TempDir::new().expect("create temp dir");
        let dictate = DaemonSlot::<DictateDaemon>::in_dir(dir.path());
        let record = DaemonSlot::<RecordDaemon>::in_dir(dir.path());
        let owner = std::process::id();
        fs::write(dictate.pid_path(), format!("{owner}\n")).expect("write dictate pid");
        fs::write(record.pid_path(), format!("{owner}\n")).expect("write record pid");

        assert!(dictate.cleanup_if_owner(owner).expect("cleanup dictate"));
        assert!(!dictate.pid_path().exists());
        assert!(record.pid_path().exists());
    }

    #[tokio::test]
    async fn owned_child_keeps_pid_until_work_finishes() {
        let dir = TempDir::new().expect("create temp dir");
        let slot = DaemonSlot::<RecordDaemon>::in_dir(dir.path());
        let owner = std::process::id();
        fs::write(slot.pid_path(), format!("{owner}\n")).expect("write record pid");

        slot.run_as_owner(async {
            assert!(
                slot.pid_path().exists(),
                "PID must remain during finalization"
            );
        })
        .await;

        assert!(!slot.pid_path().exists(), "PID must clear after child exit");
    }

    #[test]
    fn failed_pid_publication_reaps_spawned_child() {
        let dir = TempDir::new().expect("create temp dir");
        let slot = DaemonSlot::<RecordDaemon>::in_dir(dir.path());
        let lock = slot.acquire_lock().expect("lock record slot");
        fs::create_dir(slot.pid_path()).expect("block PID file publication");
        let mut command = std::process::Command::new("sleep");
        command.arg("30").process_group(0);
        let mut child = TestChild(command.spawn().expect("spawn helper child"));
        let pid = child.0.id();

        let result = lock.publish_spawned_child(&mut child.0);

        assert!(result.is_err(), "PID publication should fail");
        assert!(
            child.0.try_wait().expect("query helper status").is_some(),
            "unpublished child {pid} must be reaped"
        );
    }
}
