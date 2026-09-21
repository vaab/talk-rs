//! Foreground application identity for a remembered X11 paste target.
//!
//! Resolution is deliberately conservative: only a positively identified
//! interactive OpenCode process receives the `opencode-tui` label. Every
//! ambiguous or unsupported path resolves to `unknown`.

use std::collections::{BTreeSet, HashSet, VecDeque};
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const MAX_DESCENDANT_DEPTH: usize = 32;
const MAX_DESCENDANTS: usize = 1024;
const MAX_FDS_PER_PROCESS: usize = 512;
const MAX_TMUX_DEPTH: usize = 3;
const TMUX_COMMAND_TIMEOUT: Duration = Duration::from_millis(300);
const RESOLVER_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Debug)]
struct ProbeState {
    deadline: Instant,
    visited_terminals: HashSet<(u32, PathBuf)>,
    visited_tmux_clients: HashSet<(PathBuf, u32, PathBuf)>,
}

impl ProbeState {
    fn new(deadline: Instant) -> Self {
        Self {
            deadline,
            visited_terminals: HashSet::new(),
            visited_tmux_clients: HashSet::new(),
        }
    }

    fn active(&self) -> bool {
        Instant::now() < self.deadline
    }

    fn enter(&mut self, depth: usize, process_root: u32, pty: &Path) -> bool {
        depth <= MAX_TMUX_DEPTH
            && self.active()
            && self
                .visited_terminals
                .insert((process_root, pty.to_path_buf()))
    }
}

/// Stable labels exposed to `match-foreground-app` configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForegroundApp {
    OpenCodeTui,
    PiTui,
    Emacs,
    Shell,
    Unknown,
}

impl ForegroundApp {
    pub fn label(self) -> &'static str {
        match self {
            Self::OpenCodeTui => "opencode-tui",
            Self::PiTui => "pi-tui",
            Self::Emacs => "emacs",
            Self::Shell => "shell",
            Self::Unknown => "unknown",
        }
    }
}

/// Safe diagnostic result. It intentionally contains no command line,
/// environment value, clipboard content, or terminal screen content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetIdentity {
    pub app: ForegroundApp,
    pub terminal: Option<String>,
    pub pane: Option<String>,
    pub reason: String,
}

impl TargetIdentity {
    fn identified(app: ForegroundApp, terminal: Option<String>, pane: Option<String>) -> Self {
        Self {
            app,
            terminal,
            pane,
            reason: "identified".to_string(),
        }
    }

    fn unknown(reason: impl Into<String>) -> Self {
        Self {
            app: ForegroundApp::Unknown,
            terminal: None,
            pane: None,
            reason: reason.into(),
        }
    }
}

/// Resolver boundary used by the paste router and deterministic unit tests.
pub trait ForegroundAppResolver: Send + Sync {
    fn resolve(&self, target_xid: u32) -> TargetIdentity;
}

/// Resolver backed by fresh X11, `/proc`, PTY, and tmux observations.
#[derive(Debug, Default)]
pub struct SystemForegroundAppResolver;

impl ForegroundAppResolver for SystemForegroundAppResolver {
    fn resolve(&self, target_xid: u32) -> TargetIdentity {
        resolve_foreground_app(target_xid)
    }
}

/// Resolve a remembered X11 target and revalidate it after probing.
pub fn resolve_foreground_app(target_xid: u32) -> TargetIdentity {
    let mut state = ProbeState::new(Instant::now() + RESOLVER_TIMEOUT);
    let before = match crate::x11::x11_target_snapshot(target_xid) {
        Some(snapshot) => snapshot,
        None => return TargetIdentity::unknown("target-unavailable"),
    };
    let terminal = match normalize_terminal_class(&before.wm_class) {
        Some(terminal) => terminal,
        None => return TargetIdentity::unknown("not-a-supported-terminal"),
    };
    if !state.active() {
        return TargetIdentity::unknown("resolver-deadline-exceeded");
    }
    let mut identity = resolve_terminal_process_inner(before.pid, 0, &mut state);
    if !state.active() {
        return TargetIdentity::unknown("resolver-deadline-exceeded");
    }
    let after = match crate::x11::x11_target_snapshot(target_xid) {
        Some(snapshot) => snapshot,
        None => return TargetIdentity::unknown("target-changed-during-probe"),
    };
    if before != after {
        return TargetIdentity::unknown("target-changed-during-probe");
    }
    identity.terminal = Some(terminal);
    identity
}

/// Resolve a known terminal process. This is public so the isolated PTY/tmux
/// integration test can exercise the real resolver without mutating a desktop.
#[doc(hidden)]
pub fn resolve_terminal_process(terminal_pid: u32) -> TargetIdentity {
    let mut state = ProbeState::new(Instant::now() + RESOLVER_TIMEOUT);
    resolve_terminal_process_inner(terminal_pid, 0, &mut state)
}

/// Return the complete PTY observation for an isolated integration fixture.
#[doc(hidden)]
pub fn observed_terminal_ptys(terminal_pid: u32) -> Result<Vec<PathBuf>, String> {
    let deadline = Instant::now() + RESOLVER_TIMEOUT;
    let descendants =
        descendants_including(terminal_pid, deadline).map_err(|error| error.to_string())?;
    terminal_ptys(&descendants, deadline).map_err(|error| error.to_string())
}

fn resolve_terminal_process_inner(
    terminal_pid: u32,
    tmux_depth: usize,
    state: &mut ProbeState,
) -> TargetIdentity {
    let descendants = match descendants_including(terminal_pid, state.deadline) {
        Ok(descendants) => descendants,
        Err(_) => return TargetIdentity::unknown("terminal-process-unavailable"),
    };
    let pty = match terminal_ptys(&descendants, state.deadline)
        .ok()
        .and_then(select_unique_pty)
    {
        Some(pty) => pty,
        None => return TargetIdentity::unknown("terminal-pty-ambiguous"),
    };
    resolve_pty_foreground(&pty, terminal_pid, tmux_depth, state)
}

fn resolve_pty_foreground(
    pty: &Path,
    process_root: u32,
    tmux_depth: usize,
    state: &mut ProbeState,
) -> TargetIdentity {
    if !state.enter(tmux_depth, process_root, pty) {
        return TargetIdentity::unknown("nested-tmux-cycle-depth-or-deadline");
    }
    let foreground_pid = match foreground_process_group(process_root, pty, state.deadline) {
        Ok(Some(pid)) => pid,
        Ok(None) => return TargetIdentity::unknown("foreground-pgrp-unavailable"),
        Err(_) => return TargetIdentity::unknown("foreground-observation-incomplete"),
    };
    let process = match read_process(foreground_pid) {
        Ok(process) => process,
        Err(_) => return TargetIdentity::unknown("foreground-group-leader-unavailable"),
    };
    let pty_device = match fs::metadata(pty) {
        Ok(metadata) => metadata.rdev(),
        Err(_) => return TargetIdentity::unknown("terminal-device-unavailable"),
    };
    if !valid_foreground_leader(&process, foreground_pid, pty_device) {
        return TargetIdentity::unknown("foreground-group-leader-unavailable");
    }
    let identity = if executable_name(&process.executable) == Some("tmux") {
        resolve_tmux_client(&process, pty, tmux_depth, state)
    } else {
        let app = classify_process_path(&process.executable, &process.argv);
        if app == ForegroundApp::Unknown {
            TargetIdentity::unknown("foreground-application-unsupported")
        } else {
            TargetIdentity::identified(app, Some(pty.display().to_string()), None)
        }
    };
    let foreground_after = foreground_process_group(process_root, pty, state.deadline);
    let process_after = read_process(foreground_pid);
    if !matches!(foreground_after, Ok(Some(pid)) if pid == foreground_pid)
        || process_after.as_ref().ok() != Some(&process)
        || !state.active()
    {
        return TargetIdentity::unknown("foreground-changed-during-probe");
    }
    identity
}

fn resolve_tmux_client(
    process: &ProcessInfo,
    outer_tty: &Path,
    depth: usize,
    state: &mut ProbeState,
) -> TargetIdentity {
    let mut matches = Vec::new();
    let sockets = match tmux_socket_candidates(process.pid, state.deadline) {
        Ok(sockets) => sockets,
        Err(_) => return TargetIdentity::unknown("tmux-socket-observation-incomplete"),
    };
    for socket in sockets {
        let output = match list_tmux_clients(&socket, state.deadline) {
            Ok(output) => output,
            Err(_) => return TargetIdentity::unknown("tmux-candidate-probe-failed"),
        };
        if let Some(client) = select_tmux_client(&output, process.pid, outer_tty) {
            matches.push((socket, client));
        }
    }
    if matches.is_empty() {
        return TargetIdentity::unknown("tmux-client-unavailable");
    }
    if matches.len() != 1 {
        return TargetIdentity::unknown("tmux-client-ambiguous");
    }
    let (socket, client) = match matches.pop() {
        Some(value) => value,
        None => return TargetIdentity::unknown("tmux-client-unavailable"),
    };
    if !state
        .visited_tmux_clients
        .insert((socket.clone(), client.pid, client.tty.clone()))
    {
        return TargetIdentity::unknown("nested-tmux-cycle-depth-or-deadline");
    }

    let pane_before = match query_selected_pane(&socket, &client, state.deadline) {
        Ok(Some(pane)) if !pane.in_mode => pane,
        Ok(Some(_)) => return TargetIdentity::unknown("tmux-pane-unstable-or-in-mode"),
        _ => return TargetIdentity::unknown("tmux-pane-unavailable"),
    };
    let pane_process_before = match read_process(pane_before.pid) {
        Ok(process) => process,
        Err(_) => return TargetIdentity::unknown("tmux-pane-process-unavailable"),
    };
    let mut identity = resolve_pty_foreground(&pane_before.tty, pane_before.pid, depth + 1, state);
    let pane_after = query_selected_pane(&socket, &client, state.deadline);
    let clients_after = list_tmux_clients(&socket, state.deadline);
    let client_after = clients_after
        .ok()
        .and_then(|output| select_tmux_client(&output, process.pid, outer_tty));
    let pane_process_after = read_process(pane_before.pid);
    if !matches!(
        (client_after, pane_after, pane_process_after),
        (Some(ref after_client), Ok(Some(ref after_pane)), Ok(ref after_process))
            if stable_tmux_observation(
                &client,
                after_client,
                &pane_before,
                after_pane,
                &pane_process_before,
                after_process,
            ) && state.active()
    ) {
        return TargetIdentity::unknown("tmux-changed-during-probe");
    }
    identity.pane = Some(pane_before.id);
    identity
}

pub(crate) fn normalize_terminal_class(wm_class: &(String, String)) -> Option<String> {
    let instance = wm_class.0.to_ascii_lowercase();
    let class = wm_class.1.to_ascii_lowercase();
    match (instance.as_str(), class.as_str()) {
        ("alacritty", "alacritty") => Some("alacritty".to_string()),
        ("kitty", "kitty") => Some("kitty".to_string()),
        ("gnome-terminal-server", "gnome-terminal-server")
        | ("gnome-terminal-server", "gnome-terminal")
        | ("gnome-terminal", "gnome-terminal") => Some("gnome-terminal".to_string()),
        ("org.wezfurlong.wezterm", "org.wezfurlong.wezterm")
        | ("wezterm-gui", "org.wezfurlong.wezterm") => Some("wezterm".to_string()),
        ("konsole", "konsole") => Some("konsole".to_string()),
        ("xterm", "xterm") | ("uxterm", "uxterm") => Some("xterm".to_string()),
        ("urxvt", "urxvt") | ("rxvt", "rxvt") => Some("urxvt".to_string()),
        ("tilix", "tilix") | ("com.gexperts.tilix", "com.gexperts.tilix") => {
            Some("tilix".to_string())
        }
        ("terminator", "terminator") => Some("terminator".to_string()),
        ("foot", "foot") => Some("foot".to_string()),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcStat {
    pgrp: u32,
    tty_nr: i64,
    tpgid: i32,
    start_time: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProcessInfo {
    pid: u32,
    stat: ProcStat,
    executable: PathBuf,
    argv: Vec<String>,
}

fn read_process(pid: u32) -> io::Result<ProcessInfo> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let close = stat
        .rfind(')')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed proc stat"))?;
    let fields: Vec<&str> = stat[close + 1..].split_whitespace().collect();
    let pgrp = fields
        .get(2)
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing process group"))?;
    let tty_nr = fields
        .get(4)
        .and_then(|value| value.parse::<i64>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing controlling tty"))?;
    let tpgid = fields
        .get(5)
        .and_then(|value| value.parse::<i32>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing foreground group"))?;
    let start_time = fields
        .get(19)
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing process start time"))?;
    let executable = fs::read_link(format!("/proc/{pid}/exe"))?;
    let argv = fs::read(format!("/proc/{pid}/cmdline"))?
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty())
        .map(|part| String::from_utf8_lossy(part).into_owned())
        .collect();
    Ok(ProcessInfo {
        pid,
        stat: ProcStat {
            pgrp,
            tty_nr,
            tpgid,
            start_time,
        },
        executable,
        argv,
    })
}

fn process_controls_device(stat: &ProcStat, device: u64) -> bool {
    if stat.tty_nr <= 0 {
        return false;
    }
    let raw = stat.tty_nr as u32;
    let proc_major = ((raw >> 8) & 0xfff) as u64;
    let proc_minor = ((raw & 0xff) | ((raw >> 12) & 0xfff00)) as u64;
    proc_major == nix::sys::stat::major(device) && proc_minor == nix::sys::stat::minor(device)
}

fn valid_foreground_leader(process: &ProcessInfo, pid: u32, device: u64) -> bool {
    process.pid == pid
        && process.stat.pgrp == pid
        && process.stat.tpgid == pid as i32
        && process_controls_device(&process.stat, device)
}

#[cfg(test)]
fn encode_proc_tty_nr(major: u32, minor: u32) -> i64 {
    ((minor & 0xff) | ((major & 0xfff) << 8) | ((minor & 0xfff00) << 12)) as i64
}

fn executable_name(path: &Path) -> Option<&str> {
    path.file_name().and_then(OsStr::to_str)
}

fn classify_process_path(executable: &Path, argv: &[String]) -> ForegroundApp {
    classify_process(executable.to_string_lossy().as_ref(), argv)
}

fn classify_process(executable: &str, argv: &[String]) -> ForegroundApp {
    let name = Path::new(executable)
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or_default();
    match name {
        "opencode" => match argv.get(1).map(String::as_str) {
            None => ForegroundApp::OpenCodeTui,
            Some("attach") if argv.len() >= 3 => ForegroundApp::OpenCodeTui,
            _ => ForegroundApp::Unknown,
        },
        "pi" => ForegroundApp::PiTui,
        "node" | "tsx" => {
            if argv
                .iter()
                .skip(1)
                .any(|arg| arg.split('/').any(|part| part == "pi"))
            {
                ForegroundApp::PiTui
            } else {
                ForegroundApp::Unknown
            }
        }
        "emacs" | "emacsclient" => ForegroundApp::Emacs,
        "bash" | "dash" | "fish" | "nu" | "sh" | "zsh" => ForegroundApp::Shell,
        "ssh" | "mosh-client" => ForegroundApp::Unknown,
        _ => ForegroundApp::Unknown,
    }
}

fn descendants_including(root: u32, deadline: Instant) -> io::Result<Vec<u32>> {
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([(root, 0usize)]);
    while let Some((pid, depth)) = queue.pop_front() {
        ensure_before_deadline(deadline)?;
        if !seen.insert(pid) {
            continue;
        }
        if result.len() >= MAX_DESCENDANTS || depth > MAX_DESCENDANT_DEPTH {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "terminal process tree exceeded resolver bounds",
            ));
        }
        result.push(pid);
        let children = fs::read_to_string(format!("/proc/{pid}/task/{pid}/children"))?;
        for child in children.split_whitespace() {
            if let Ok(child) = child.parse::<u32>() {
                queue.push_back((child, depth + 1));
            }
        }
    }
    Ok(result)
}

fn terminal_ptys(pids: &[u32], deadline: Instant) -> io::Result<Vec<PathBuf>> {
    let mut ptys = BTreeSet::new();
    for pid in pids {
        ensure_before_deadline(deadline)?;
        for entry in read_dir_bounded(&format!("/proc/{pid}/fd"))? {
            ensure_before_deadline(deadline)?;
            let target = fs::read_link(entry.path())?;
            if is_pts_path(&target) {
                ptys.insert(target);
            }
            let fdinfo = fs::read_to_string(format!(
                "/proc/{pid}/fdinfo/{}",
                entry.file_name().to_string_lossy()
            ))?;
            for line in fdinfo.lines() {
                if let Some(index) = line.strip_prefix("tty-index:") {
                    if let Ok(index) = index.trim().parse::<u32>() {
                        ptys.insert(PathBuf::from(format!("/dev/pts/{index}")));
                    }
                }
            }
        }
    }
    Ok(ptys.into_iter().collect())
}

fn is_pts_path(path: &Path) -> bool {
    path.parent() == Some(Path::new("/dev/pts"))
        && path
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| name.chars().all(|character| character.is_ascii_digit()))
}

fn select_unique_pty(mut ptys: Vec<PathBuf>) -> Option<PathBuf> {
    ptys.sort();
    ptys.dedup();
    (ptys.len() == 1).then(|| ptys.remove(0))
}

fn foreground_process_group(
    process_root: u32,
    pty: &Path,
    deadline: Instant,
) -> io::Result<Option<u32>> {
    let descendants = descendants_including(process_root, deadline)?;
    let device = fs::metadata(pty)?.rdev();
    let mut foreground_groups = BTreeSet::new();
    for pid in descendants {
        ensure_before_deadline(deadline)?;
        let process = read_process(pid)?;
        if process_controls_device(&process.stat, device) && process.stat.tpgid > 0 {
            foreground_groups.insert(process.stat.tpgid as u32);
        }
    }
    if foreground_groups.len() == 1 {
        Ok(foreground_groups.iter().next().copied())
    } else {
        Ok(None)
    }
}

fn collect_bounded<T>(iter: impl IntoIterator<Item = T>, limit: usize) -> Option<Vec<T>> {
    let mut result = Vec::new();
    for item in iter {
        if result.len() == limit {
            return None;
        }
        result.push(item);
    }
    Some(result)
}

fn read_dir_bounded(path: &str) -> io::Result<Vec<fs::DirEntry>> {
    let entries = collect_bounded(fs::read_dir(path)?, MAX_FDS_PER_PROCESS).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "file descriptor enumeration exceeded resolver bound",
        )
    })?;
    entries.into_iter().collect()
}

fn ensure_before_deadline(deadline: Instant) -> io::Result<()> {
    if Instant::now() < deadline {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "foreground resolver deadline exceeded",
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TmuxClient {
    pid: u32,
    tty: PathBuf,
    session: String,
}

fn select_tmux_client(output: &str, expected_pid: u32, expected_tty: &Path) -> Option<TmuxClient> {
    let matches: Vec<TmuxClient> = output
        .lines()
        .filter_map(|line| {
            let mut fields = line.split('\t');
            let pid = fields.next()?.parse::<u32>().ok()?;
            let tty = PathBuf::from(fields.next()?);
            let control_mode = fields.next()?;
            let session = fields.next()?.to_string();
            let client_active_pane = fields.next()?;
            (pid == expected_pid
                && tty == expected_tty
                && control_mode == "0"
                && client_active_pane.is_empty())
            .then_some(TmuxClient { pid, tty, session })
        })
        .collect();
    (matches.len() == 1).then(|| matches[0].clone())
}

fn stable_tmux_observation(
    client_before: &TmuxClient,
    client_after: &TmuxClient,
    pane_before: &TmuxPane,
    pane_after: &TmuxPane,
    pane_process_before: &ProcessInfo,
    pane_process_after: &ProcessInfo,
) -> bool {
    client_before == client_after
        && pane_before == pane_after
        && !pane_after.in_mode
        && pane_process_before == pane_process_after
        && pane_process_after.pid == pane_after.pid
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TmuxPane {
    id: String,
    tty: PathBuf,
    pid: u32,
    in_mode: bool,
}

fn list_tmux_clients(socket: &Path, deadline: Instant) -> io::Result<String> {
    run_bounded(
        Command::new("tmux").arg("-S").arg(socket).args([
            "list-clients",
            "-F",
            "#{client_pid}\t#{client_tty}\t#{client_control_mode}\t#{client_session}\t#{client_active_pane}",
        ]),
        deadline,
    )
}

fn query_selected_pane(
    socket: &Path,
    client: &TmuxClient,
    deadline: Instant,
) -> io::Result<Option<TmuxPane>> {
    let session_target = format!("{}:", client.session);
    let output = run_bounded(
        Command::new("tmux")
            .arg("-S")
            .arg(socket)
            .args(["display-message", "-p", "-c"])
            .arg(&client.tty)
            .arg("-t")
            .arg(session_target)
            .arg("#{pane_id}\t#{pane_tty}\t#{pane_pid}\t#{pane_in_mode}"),
        deadline,
    )?;
    let mut fields = output.trim_end().split('\t');
    let id = match fields.next() {
        Some(value) => value.to_string(),
        None => return Ok(None),
    };
    let tty = match fields.next() {
        Some(value) => PathBuf::from(value),
        None => return Ok(None),
    };
    let pid = match fields.next().and_then(|value| value.parse::<u32>().ok()) {
        Some(value) => value,
        None => return Ok(None),
    };
    let in_mode = match fields.next() {
        Some("0") => false,
        Some("1") => true,
        _ => return Ok(None),
    };
    if fields.next().is_some() || id.is_empty() || !is_pts_path(&tty) {
        return Ok(None);
    }
    Ok(Some(TmuxPane {
        id,
        tty,
        pid,
        in_mode,
    }))
}

fn tmux_socket_candidates(pid: u32, deadline: Instant) -> io::Result<Vec<PathBuf>> {
    ensure_before_deadline(deadline)?;
    let mut candidates = BTreeSet::new();
    let process = read_process(pid)?;
    {
        let mut args = process.argv.iter().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-S" => {
                    if let Some(path) = args.next() {
                        candidates.insert(PathBuf::from(path));
                    }
                }
                "-L" => {
                    if let Some(name) = args.next() {
                        if let Some(path) = default_tmux_socket(pid, name)? {
                            candidates.insert(path);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let environment = fs::read(format!("/proc/{pid}/environ"))?;
    for entry in environment.split(|byte| *byte == 0) {
        if let Some(value) = entry.strip_prefix(b"TMUX=") {
            if let Some(path) = value.split(|byte| *byte == b',').next() {
                if !path.is_empty() {
                    candidates.insert(PathBuf::from(String::from_utf8_lossy(path).into_owned()));
                }
            }
        }
    }

    if candidates.is_empty() {
        if let Some(path) = default_tmux_socket(pid, "default")? {
            candidates.insert(path);
        }
    }

    let socket_inodes = process_socket_inodes(pid, deadline)?;
    let unix_sockets = fs::read_to_string("/proc/net/unix")?;
    for line in unix_sockets.lines().skip(1) {
        ensure_before_deadline(deadline)?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if let (Some(inode), Some(path)) = (fields.get(6), fields.get(7)) {
            if socket_inodes.contains(*inode) && Path::new(path).is_absolute() {
                candidates.insert(PathBuf::from(path));
            }
        }
    }
    Ok(candidates.into_iter().collect())
}

fn default_tmux_socket(pid: u32, name: &str) -> io::Result<Option<PathBuf>> {
    if name.is_empty() || name.contains('/') {
        return Ok(None);
    }
    let uid = fs::metadata(format!("/proc/{pid}"))?.uid();
    let mut base = PathBuf::from("/tmp");
    let environment = fs::read(format!("/proc/{pid}/environ"))?;
    for entry in environment.split(|byte| *byte == 0) {
        if let Some(value) = entry.strip_prefix(b"TMUX_TMPDIR=") {
            if !value.is_empty() {
                base = PathBuf::from(String::from_utf8_lossy(value).into_owned());
            }
        }
    }
    Ok(Some(base.join(format!("tmux-{uid}")).join(name)))
}

fn process_socket_inodes(pid: u32, deadline: Instant) -> io::Result<HashSet<String>> {
    let mut inodes = HashSet::new();
    for entry in read_dir_bounded(&format!("/proc/{pid}/fd"))? {
        ensure_before_deadline(deadline)?;
        let target = fs::read_link(entry.path())?;
        let target = target.to_string_lossy();
        if let Some(inode) = target
            .strip_prefix("socket:[")
            .and_then(|value| value.strip_suffix(']'))
        {
            inodes.insert(inode.to_string());
        }
    }
    Ok(inodes)
}

fn remaining_command_budget(deadline: Instant) -> Option<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .map(|remaining| remaining.min(TMUX_COMMAND_TIMEOUT))
        .filter(|remaining| !remaining.is_zero())
}

fn run_bounded(command: &mut Command, deadline: Instant) -> io::Result<String> {
    let budget = remaining_command_budget(deadline)
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "resolver deadline exceeded"))?;
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let command_deadline = Instant::now() + budget;
    loop {
        if child.try_wait()?.is_some() {
            let output = child.wait_with_output()?;
            if !output.status.success() {
                return Err(io::Error::other("bounded command failed"));
            }
            return String::from_utf8(output.stdout)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF8 output"));
        }
        if Instant::now() >= command_deadline {
            child.kill()?;
            let _ = child.wait();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "bounded command timed out",
            ));
        }
        thread::sleep(Duration::from_millis(5));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn classifier_accepts_only_interactive_opencode_modes() {
        assert_eq!(
            classify_process("/tmp/opencode", &args(&["opencode"])),
            ForegroundApp::OpenCodeTui
        );
        assert_eq!(
            classify_process(
                "/tmp/opencode",
                &args(&["opencode", "attach", "http://127.0.0.1:4096"])
            ),
            ForegroundApp::OpenCodeTui
        );

        for argv in [
            args(&["opencode", "serve"]),
            args(&["opencode", "run", "prompt"]),
            args(&["opencode", "--wrapper-flag"]),
        ] {
            assert_eq!(
                classify_process("/tmp/opencode", &argv),
                ForegroundApp::Unknown,
                "non-interactive or ambiguous argv must stay safe"
            );
        }
    }

    #[test]
    fn classifier_labels_known_non_opencode_foregrounds_without_promoting_them() {
        assert_eq!(
            classify_process("/usr/bin/pi", &args(&["pi"])),
            ForegroundApp::PiTui
        );
        assert_eq!(
            classify_process("/usr/bin/emacs", &args(&["emacs", "-nw"])),
            ForegroundApp::Emacs
        );
        assert_eq!(
            classify_process("/usr/bin/bash", &args(&["bash"])),
            ForegroundApp::Shell
        );
        assert_eq!(
            classify_process("/usr/bin/ssh", &args(&["ssh", "host"])),
            ForegroundApp::Unknown
        );
        assert_eq!(
            classify_process("/usr/bin/node", &args(&["node", "/opt/pi/dist/cli.js"])),
            ForegroundApp::PiTui
        );
        assert_eq!(
            classify_process("/usr/bin/node", &args(&["node", "/opt/opencode/server.js"])),
            ForegroundApp::Unknown
        );
    }

    #[test]
    fn terminal_pty_must_be_unique() {
        assert_eq!(
            select_unique_pty(vec![PathBuf::from("/dev/pts/41")]),
            Some(PathBuf::from("/dev/pts/41"))
        );
        assert_eq!(
            select_unique_pty(vec![
                PathBuf::from("/dev/pts/41"),
                PathBuf::from("/dev/pts/42")
            ]),
            None
        );
        assert_eq!(select_unique_pty(Vec::new()), None);
    }

    #[test]
    fn tmux_client_selection_requires_exact_pid_tty_and_one_match() {
        let one = "410\t/dev/pts/20\t0\tone\t\n411\t/dev/pts/21\t0\ttwo\t\n";
        assert_eq!(
            select_tmux_client(one, 411, Path::new("/dev/pts/21")),
            Some(TmuxClient {
                pid: 411,
                tty: PathBuf::from("/dev/pts/21"),
                session: "two".to_string(),
            })
        );
        assert_eq!(select_tmux_client(one, 999, Path::new("/dev/pts/21")), None);

        let duplicate = "411\t/dev/pts/21\t0\ttwo\t\n411\t/dev/pts/21\t0\ttwo\t\n";
        assert_eq!(
            select_tmux_client(duplicate, 411, Path::new("/dev/pts/21")),
            None,
            "multiple matching clients must never pick an arbitrary pane"
        );

        let control = "411\t/dev/pts/21\t1\ttwo\t\n";
        assert_eq!(
            select_tmux_client(control, 411, Path::new("/dev/pts/21")),
            None,
            "control clients have no interactive selected pane"
        );

        let active_pane = "411\t/dev/pts/21\t0\ttwo\t%9\n";
        assert_eq!(
            select_tmux_client(active_pane, 411, Path::new("/dev/pts/21")),
            None,
            "unsupported client-specific active-pane flag must fail closed"
        );
    }

    #[test]
    fn terminal_class_normalization_rejects_nonterminals() {
        assert_eq!(
            normalize_terminal_class(&("kitty".to_string(), "kitty".to_string())),
            Some("kitty".to_string())
        );
        assert_eq!(
            normalize_terminal_class(&("emacs".to_string(), "Emacs".to_string())),
            None
        );
        assert_eq!(
            normalize_terminal_class(&("kitty-notes".to_string(), "Firefox".to_string())),
            None
        );
        assert_eq!(
            normalize_terminal_class(&("Alacritty".to_string(), "Alacritty".to_string())),
            Some("alacritty".to_string())
        );
        assert_eq!(
            normalize_terminal_class(&("terminator".to_string(), "Terminator".to_string())),
            Some("terminator".to_string())
        );
        assert_eq!(
            normalize_terminal_class(&(
                "gnome-terminal-server".to_string(),
                "Gnome-terminal".to_string()
            )),
            Some("gnome-terminal".to_string())
        );
    }

    #[test]
    fn bounded_collection_rejects_an_unobserved_extra_fd() {
        assert_eq!(collect_bounded(0..3, 3), Some(vec![0, 1, 2]));
        assert_eq!(collect_bounded(0..4, 3), None);
    }

    #[test]
    fn controlling_tty_wins_over_an_unrelated_open_pty_fd() {
        let controlling = nix::sys::stat::makedev(136, 41);
        let unrelated_open_fd = nix::sys::stat::makedev(136, 42);
        let stat = ProcStat {
            pgrp: 700,
            tty_nr: encode_proc_tty_nr(136, 41),
            tpgid: 700,
            start_time: 1234,
        };

        assert!(process_controls_device(&stat, controlling));
        assert!(!process_controls_device(&stat, unrelated_open_fd));

        let process = ProcessInfo {
            pid: 700,
            stat: stat.clone(),
            executable: PathBuf::from("/tmp/opencode"),
            argv: args(&["opencode"]),
        };
        assert!(valid_foreground_leader(&process, 700, controlling));
        let mut wrong_pgrp = process.clone();
        wrong_pgrp.stat.pgrp = 701;
        assert!(!valid_foreground_leader(&wrong_pgrp, 700, controlling));
        let mut wrong_tpgid = process;
        wrong_tpgid.stat.tpgid = 701;
        assert!(!valid_foreground_leader(&wrong_tpgid, 700, controlling));
    }

    #[test]
    fn recursive_probe_rejects_cycles_and_depth_cap_at_actual_entry() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut state = ProbeState::new(deadline);
        assert!(state.enter(0, 10, Path::new("/dev/pts/1")));
        assert!(!state.enter(1, 10, Path::new("/dev/pts/1")));

        let mut depth_state = ProbeState::new(deadline);
        assert!(!depth_state.enter(MAX_TMUX_DEPTH + 1, 11, Path::new("/dev/pts/2")));
    }

    #[test]
    fn expired_overall_deadline_prevents_starting_another_command() {
        let deadline = Instant::now() - Duration::from_millis(1);
        assert_eq!(remaining_command_budget(deadline), None);
    }

    #[test]
    fn running_command_cannot_outlive_overall_deadline() {
        let started = Instant::now();
        let error = run_bounded(
            Command::new("sleep").arg("1"),
            Instant::now() + Duration::from_millis(20),
        )
        .expect_err("sleep must be terminated at the resolver deadline");
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_millis(300));
    }

    #[test]
    fn pane_or_process_change_after_classification_is_rejected() {
        let client = TmuxClient {
            pid: 411,
            tty: PathBuf::from("/dev/pts/21"),
            session: "one".to_string(),
        };
        let pane = TmuxPane {
            id: "%3".to_string(),
            tty: PathBuf::from("/dev/pts/31"),
            pid: 900,
            in_mode: false,
        };
        let process = ProcessInfo {
            pid: 900,
            stat: ProcStat {
                pgrp: 900,
                tty_nr: encode_proc_tty_nr(136, 31),
                tpgid: 900,
                start_time: 55,
            },
            executable: PathBuf::from("/tmp/opencode"),
            argv: args(&["opencode"]),
        };
        assert!(stable_tmux_observation(
            &client, &client, &pane, &pane, &process, &process
        ));

        let mut switched_pane = pane.clone();
        switched_pane.id = "%4".to_string();
        assert!(!stable_tmux_observation(
            &client,
            &client,
            &pane,
            &switched_pane,
            &process,
            &process
        ));

        let mut restarted = process.clone();
        restarted.stat.start_time += 1;
        assert!(!stable_tmux_observation(
            &client, &client, &pane, &pane, &process, &restarted
        ));
    }
}
