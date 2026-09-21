#![cfg(feature = "ui")]

//! Opt-in real PTY/tmux test for foreground application resolution.
//!
//! The fixture owns every child and uses a private tmux socket. It never
//! touches the user's tmux server, terminal content, desktop focus, or clipboard.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use talk_rs::paste::target::{
    observed_terminal_ptys, resolve_terminal_process, ForegroundApp, TargetIdentity,
};

struct OwnedChild(Child);

impl OwnedChild {
    fn stop(&mut self) -> Result<(), String> {
        if self.0.try_wait().ok().flatten().is_none() {
            let pid = nix::unistd::Pid::from_raw(self.0.id() as i32);
            nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM)
                .map_err(|error| format!("signal owned child {pid}: {error}"))?;
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                if self
                    .0
                    .try_wait()
                    .map_err(|error| format!("wait for owned child {pid}: {error}"))?
                    .is_some()
                {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            self.0
                .kill()
                .map_err(|error| format!("kill owned child {pid}: {error}"))?;
        }
        self.0
            .wait()
            .map_err(|error| format!("reap owned child: {error}"))?;
        Ok(())
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            eprintln!("owned fixture child cleanup failed: {error}");
        }
    }
}

struct PrivateTmux {
    socket: PathBuf,
    clients: Vec<OwnedChild>,
    stopped: bool,
}

impl PrivateTmux {
    fn command(&self) -> Command {
        let mut command = Command::new("tmux");
        command.arg("-S").arg(&self.socket);
        command
    }

    fn output(&self, args: &[&str]) -> Output {
        self.command()
            .args(args)
            .output()
            .expect("run private tmux command")
    }

    fn success(&self, args: &[&str]) {
        let output = self.output(args);
        assert!(
            output.status.success(),
            "private tmux command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn attach(&mut self, session: &str) -> u32 {
        let command = format!(
            "exec tmux -S {} attach-session -t {}",
            self.socket.display(),
            session
        );
        let child = Command::new("script")
            .args(["-qfec", &command, "/dev/null"])
            .env_remove("TMUX")
            .env_remove("TMUX_PANE")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("attach private tmux client through a PTY");
        let pid = child.id();
        self.clients.push(OwnedChild(child));
        pid
    }

    fn wait_for_clients(&self, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let output = self.output(&["list-clients", "-F", "#{client_pid}"]);
            let actual = String::from_utf8_lossy(&output.stdout).lines().count();
            if output.status.success() && actual == count {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "private tmux clients did not attach"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn client_tty_for_session(&self, session: &str) -> String {
        let output = self.output(&["list-clients", "-F", "#{client_session}\t#{client_tty}"]);
        assert!(output.status.success(), "list private tmux clients");
        let matches: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                let (client_session, tty) = line.split_once('\t')?;
                (client_session == session).then(|| tty.to_string())
            })
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected one client for session {session}"
        );
        matches[0].clone()
    }

    fn client_session_for_tty(&self, tty: &Path) -> String {
        let output = self.output(&["list-clients", "-F", "#{client_tty}\t#{client_session}"]);
        assert!(output.status.success(), "list private tmux clients");
        let tty = tty.display().to_string();
        let matches: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                let (client_tty, session) = line.split_once('\t')?;
                (client_tty == tty).then(|| session.to_string())
            })
            .collect();
        assert_eq!(matches.len(), 1, "expected one client for tty {tty}");
        matches[0].clone()
    }

    fn stop(&mut self) -> Result<(), String> {
        if self.stopped {
            return Ok(());
        }
        let status = self
            .command()
            .arg("kill-server")
            .status()
            .map_err(|error| format!("stop private tmux server: {error}"))?;
        if !status.success() {
            return Err(format!("private tmux kill-server failed: {status}"));
        }
        for client in &mut self.clients {
            client.stop()?;
        }
        self.stopped = true;
        Ok(())
    }
}

impl Drop for PrivateTmux {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            eprintln!("private tmux cleanup failed: {error}");
        }
    }
}

fn compile_stub(directory: &Path) -> PathBuf {
    let executable = directory.join("opencode");
    let status = Command::new("cc")
        .args(["-O0", "-o"])
        .arg(&executable)
        .arg("tests/fixtures/foreground-app-stub.c")
        .status()
        .expect("compile foreground application fixture");
    assert!(status.success(), "C fixture compilation failed");
    executable
}

fn script_child(command: &str) -> OwnedChild {
    OwnedChild(
        Command::new("script")
            .args(["-qfec", command, "/dev/null"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start isolated PTY fixture"),
    )
}

fn wait_for_identity(
    stage: &str,
    pid: u32,
    predicate: impl Fn(&TargetIdentity) -> bool,
) -> TargetIdentity {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let identity = resolve_terminal_process(pid);
        if predicate(&identity) {
            return identity;
        }
        assert!(
            Instant::now() < deadline,
            "resolver did not reach expected state at {stage}; last={identity:?}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
#[ignore = "requires cc, script, and tmux; owns a private tmux socket"]
fn isolated_real_pty_and_tmux_follow_only_the_selected_client_pane() {
    let temp = tempfile::tempdir().expect("create fixture directory");
    let opencode = compile_stub(temp.path());
    let shell = temp.path().join("sh");
    std::fs::copy(&opencode, &shell).expect("create persistent shell fixture");

    let mut direct = script_child(&format!("exec {}", opencode.display()));
    let direct_identity = wait_for_identity("direct PTY", direct.0.id(), |identity| {
        identity.app == ForegroundApp::OpenCodeTui
    });
    assert_eq!(direct_identity.app, ForegroundApp::OpenCodeTui);
    direct.stop().expect("stop direct PTY fixture");

    let socket = temp.path().join("tmux.sock");
    let shell_session = "talk-rs-foreground-shell-test";
    let opencode_session = "talk-rs-foreground-opencode-test";
    let mut tmux = PrivateTmux {
        socket,
        clients: Vec::new(),
        stopped: false,
    };
    tmux.success(&[
        "-f",
        "/dev/null",
        "new-session",
        "-d",
        "-s",
        shell_session,
        shell.to_str().expect("shell fixture path UTF-8"),
    ]);
    tmux.success(&[
        "new-session",
        "-d",
        "-s",
        opencode_session,
        opencode.to_str().expect("fixture path UTF-8"),
    ]);
    let shell_pane = String::from_utf8(
        tmux.output(&["display-message", "-p", "-t", shell_session, "#{pane_id}"])
            .stdout,
    )
    .expect("shell pane id")
    .trim()
    .to_string();
    let opencode_pane = String::from_utf8(
        tmux.output(&[
            "display-message",
            "-p",
            "-t",
            opencode_session,
            "#{pane_id}",
        ])
        .stdout,
    )
    .expect("opencode pane id")
    .trim()
    .to_string();

    let first_client = tmux.attach(shell_session);
    let second_client = tmux.attach(opencode_session);
    tmux.wait_for_clients(2);
    let first_ptys = observed_terminal_ptys(first_client).expect("observe first client PTY");
    let second_ptys = observed_terminal_ptys(second_client).expect("observe second client PTY");
    assert_eq!(first_ptys.len(), 1, "first client must own one outer PTY");
    assert_eq!(second_ptys.len(), 1, "second client must own one outer PTY");
    assert_eq!(tmux.client_session_for_tty(&first_ptys[0]), shell_session);
    assert_eq!(
        tmux.client_session_for_tty(&second_ptys[0]),
        opencode_session
    );

    let shell_identity = wait_for_identity("selected shell pane", first_client, |identity| {
        identity.app == ForegroundApp::Shell
    });
    assert_eq!(shell_identity.app, ForegroundApp::Shell);
    assert_eq!(shell_identity.pane.as_deref(), Some(shell_pane.as_str()));

    let selected_identity =
        wait_for_identity("selected OpenCode pane", second_client, |identity| {
            identity.app == ForegroundApp::OpenCodeTui
        });
    assert_eq!(
        selected_identity.pane.as_deref(),
        Some(opencode_pane.as_str())
    );

    let first_tty = tmux.client_tty_for_session(shell_session);
    let second_tty = tmux.client_tty_for_session(opencode_session);
    tmux.success(&["switch-client", "-c", &first_tty, "-t", opencode_session]);
    let switched_first = wait_for_identity("switched first client", first_client, |identity| {
        identity.app == ForegroundApp::OpenCodeTui
    });
    assert_eq!(switched_first.pane.as_deref(), Some(opencode_pane.as_str()));
    let unchanged_second =
        wait_for_identity("unchanged second client", second_client, |identity| {
            identity.app == ForegroundApp::OpenCodeTui
        });
    assert_eq!(
        unchanged_second.pane.as_deref(),
        Some(opencode_pane.as_str())
    );

    tmux.success(&["switch-client", "-c", &second_tty, "-t", shell_session]);
    let unchanged_first =
        wait_for_identity("first client remains OpenCode", first_client, |identity| {
            identity.app == ForegroundApp::OpenCodeTui
        });
    assert_eq!(
        unchanged_first.pane.as_deref(),
        Some(opencode_pane.as_str())
    );
    let switched_second = wait_for_identity(
        "second client switched to shell",
        second_client,
        |identity| identity.app == ForegroundApp::Shell,
    );
    assert_eq!(switched_second.pane.as_deref(), Some(shell_pane.as_str()));

    tmux.success(&[
        "respawn-pane",
        "-k",
        "-t",
        &opencode_pane,
        opencode.to_str().expect("fixture path UTF-8"),
        "serve",
    ]);
    let serve_identity = wait_for_identity("OpenCode serve pane", first_client, |identity| {
        identity.app == ForegroundApp::Unknown
    });
    assert_eq!(serve_identity.app, ForegroundApp::Unknown);

    tmux.success(&["copy-mode", "-t", &opencode_pane]);
    let mode_identity = wait_for_identity("tmux copy mode", first_client, |identity| {
        identity.reason == "tmux-pane-unstable-or-in-mode"
    });
    assert_eq!(mode_identity.app, ForegroundApp::Unknown);

    tmux.stop().expect("stop private tmux fixture");
    assert!(
        tmux.clients
            .iter_mut()
            .all(|client| client.0.try_wait().ok().flatten().is_some()),
        "owned tmux clients remained after teardown"
    );
}

#[test]
#[ignore = "requires cc and script; owns isolated PTYs"]
fn ambiguous_terminal_surfaces_resolve_to_unknown() {
    let temp = tempfile::tempdir().expect("create fixture directory");
    let opencode = compile_stub(temp.path());
    let command = format!(
        "trap 'kill \"$a\" \"$b\" 2>/dev/null; wait \"$a\" \"$b\" 2>/dev/null' TERM INT EXIT; script -qfec 'exec {}' /dev/null >/dev/null 2>&1 & a=$!; script -qfec 'exec {}' /dev/null >/dev/null 2>&1 & b=$!; wait",
        opencode.display(),
        opencode.display()
    );
    let mut parent = OwnedChild(
        Command::new("sh")
            .args(["-c", &command])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start ambiguous PTY parent"),
    );
    let ptys_deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let ptys = observed_terminal_ptys(parent.0.id()).unwrap_or_default();
        if ptys.len() >= 2 {
            break;
        }
        assert!(
            Instant::now() < ptys_deadline,
            "ambiguous fixture never exposed two live PTYs; last={ptys:?}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let identity = wait_for_identity("ambiguous PTYs", parent.0.id(), |identity| {
        identity.reason == "terminal-pty-ambiguous"
    });
    assert_eq!(identity.app, ForegroundApp::Unknown);
    parent.stop().expect("stop ambiguous PTY fixture");
}
