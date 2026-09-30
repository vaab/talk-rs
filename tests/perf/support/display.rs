//! Isolated X display for GTK / X11 perf tests.
//!
//! `Xvfb` is used when installed.  Otherwise a headless Weston with
//! Xwayland provides an isolated X server *with* an EWMH window
//! manager (Weston's XWM supports `_NET_ACTIVE_WINDOW`, which bare
//! Xvfb does not) — needed by the focus/paste cut.  Nothing ever
//! touches the user's display: the chosen display is always one this
//! harness started, and only the harness's own child is signalled.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub struct IsolatedDisplay {
    child: Child,
    pub display: String,
    _runtime: tempfile::TempDir,
}

impl IsolatedDisplay {
    /// Start an isolated X server, or return `None` when neither Xvfb
    /// nor weston is available (the caller then skips).
    pub fn start() -> Option<Self> {
        let runtime = tempfile::tempdir().ok()?;
        if let Some(display) = Self::start_xvfb(&runtime) {
            return Some(display);
        }
        Self::start_weston(runtime)
    }

    fn start_xvfb(runtime: &tempfile::TempDir) -> Option<Self> {
        let number = (140..200).find(|n| {
            !std::path::Path::new(&format!("/tmp/.X{n}-lock")).exists()
                && !std::path::Path::new(&format!("/tmp/.X11-unix/X{n}")).exists()
        })?;
        let display = format!(":{number}");
        let child = Command::new("Xvfb")
            .args([&display, "-screen", "0", "1280x800x24", "-nolisten", "tcp"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let dir = tempfile::tempdir_in(runtime.path()).ok()?;
        let mut server = Self {
            child,
            display,
            _runtime: dir,
        };
        server.wait_ready().then_some(server)
    }

    fn start_weston(runtime: tempfile::TempDir) -> Option<Self> {
        let log = runtime.path().join("weston.log");
        let child = Command::new("weston")
            .args([
                "--backend=headless-backend.so",
                "--xwayland",
                "--socket=talk-rs-perf",
                "--idle-time=0",
                "--width=1280",
                "--height=800",
            ])
            .arg(format!("--log={}", log.display()))
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("XDG_RUNTIME_DIR", runtime.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let deadline = Instant::now() + Duration::from_secs(10);
        let display = loop {
            if let Some(d) = std::fs::read_to_string(&log).ok().and_then(|s| {
                s.lines()
                    .find_map(|l| l.split("xserver listening on display ").nth(1))
                    .map(|d| d.trim().to_string())
            }) {
                break d;
            }
            if Instant::now() > deadline {
                let mut child = child;
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        let mut server = Self {
            child,
            display,
            _runtime: runtime,
        };
        server.wait_ready().then_some(server)
    }

    fn wait_ready(&mut self) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return false;
            }
            if x11rb::connect(Some(&self.display)).is_ok() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        false
    }

    pub fn log_dir(&self) -> PathBuf {
        self._runtime.path().to_path_buf()
    }
}

impl Drop for IsolatedDisplay {
    fn drop(&mut self) {
        // Only our own child: SIGTERM lets weston stop its Xwayland
        // and helper clients; fall back to SIGKILL.
        let pid = nix::unistd::Pid::from_raw(self.child.id() as i32);
        let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM);
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
