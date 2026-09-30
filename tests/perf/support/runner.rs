//! Isolated `talk-rs` binary runner.
//!
//! Every run gets fresh `XDG_*` directories, its own validate-cache
//! path, no inherited provider credentials, no display (unless the
//! caller provides an isolated one), no real audio device (ALSA null
//! PCM, PulseAudio / PipeWire sockets pointed at nothing) and a
//! `--log-file` whose `timing:` / `perf-counter:` / `perf-mark:` lines
//! are parsed into a [`RunReport`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

/// Isolated per-run environment.
pub struct Sandbox {
    pub dir: tempfile::TempDir,
}

impl Sandbox {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().expect("sandbox dir");
        for sub in ["config/talk-rs", "cache", "data", "runtime", "output"] {
            std::fs::create_dir_all(dir.path().join(sub)).expect("sandbox subdir");
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(
                dir.path().join("runtime"),
                std::fs::Permissions::from_mode(0o700),
            );
        }
        std::fs::write(
            dir.path().join("asound.conf"),
            "pcm.!default { type null }\nctl.!default { type hw card 0 }\n",
        )
        .expect("alsa null config");
        Self { dir }
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    pub fn output_dir(&self) -> PathBuf {
        self.path().join("output")
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.path().join("cache/talk-rs")
    }

    pub fn write_config(&self, yaml: &str) {
        std::fs::write(self.path().join("config/talk-rs/config.yaml"), yaml)
            .expect("sandbox config");
    }

    pub fn log_path(&self) -> PathBuf {
        self.path().join("talk-rs.log")
    }

    /// A `talk-rs` command fully isolated from the user's session.
    pub fn command(&self, args: &[&str]) -> tokio::process::Command {
        let mut cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_talk-rs"));
        cmd.arg("-v")
            .arg("--log-file")
            .arg(self.log_path())
            .args(args)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", self.path())
            .env("XDG_CONFIG_HOME", self.path().join("config"))
            .env("XDG_CACHE_HOME", self.path().join("cache"))
            .env("XDG_DATA_HOME", self.path().join("data"))
            .env("XDG_RUNTIME_DIR", self.path().join("runtime"))
            .env(
                "TALK_RS_VALIDATE_CACHE_PATH",
                self.path().join("validate-cache.yaml"),
            )
            .env("ALSA_CONFIG_PATH", self.path().join("asound.conf"))
            .env("PULSE_SERVER", "unix:/nonexistent/talk-rs-perf-pulse")
            .env("PIPEWIRE_REMOTE", "talk-rs-perf-no-such-remote")
            .env("NO_COLOR", "1")
            // Block any network egress not aimed at the harness's own
            // loopback servers: an accidental provider or model-download
            // request fails fast instead of leaving the machine.
            .env("HTTP_PROXY", "http://127.0.0.1:9")
            .env("HTTPS_PROXY", "http://127.0.0.1:9")
            .env("ALL_PROXY", "http://127.0.0.1:9")
            .env("NO_PROXY", "127.0.0.1,localhost")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        cmd
    }
}

/// Parsed outcome of one binary run.
#[derive(Debug, Default)]
pub struct RunReport {
    pub status: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub log: String,
    pub wall_ms: f64,
    /// `timing: stop +Nms <step>` (first occurrence of each step).
    pub stop: BTreeMap<String, u64>,
    /// `timing: start +Nms <step>`.
    pub start: BTreeMap<String, u64>,
    /// The last COMPLETE framed dump (the exit dump for a finished
    /// run).  Empty when no complete frame was found.
    pub counters: BTreeMap<String, u64>,
    /// `perf-mark: name=+Nms`.
    pub marks: BTreeMap<String, u64>,
    /// Peak RSS of the child (KiB): its own final `VmHWM` dump.
    /// (`RUSAGE_CHILDREN.ru_maxrss` is a maximum over every child ever
    /// reaped, so it cannot isolate one run.)  0 without counters.
    pub child_maxrss_kb: u64,
    /// CPU (user+sys) of the child in ms, `getrusage(RUSAGE_CHILDREN)`
    /// delta around the run (serialised by [`CHILD_LOCK`]).
    pub child_cpu_ms: f64,
    /// When the child was spawned.
    pub started_at: Option<Instant>,
}

impl RunReport {
    pub fn counter(&self, name: &str) -> u64 {
        self.counters.get(name).copied().unwrap_or_else(|| {
            panic!(
                "perf-counter {name} missing (built without --features perf-counters?)\nlog:\n{}",
                self.log
            )
        })
    }

    pub fn stop_ms(&self, step: &str) -> u64 {
        *self
            .stop
            .get(step)
            .unwrap_or_else(|| panic!("timing: stop {step} missing\nlog:\n{}", self.log))
    }

    pub fn assert_success(&self) {
        assert_eq!(
            self.status,
            Some(0),
            "talk-rs failed\nstderr:\n{}\nlog:\n{}",
            self.stderr,
            self.log
        );
    }
}

/// Fields every framed dump must carry for resource-based scoring.
pub const REQUIRED_DUMP_FIELDS: [&str; 5] = [
    "vm_hwm_kb",
    "vm_rss_kb",
    "threads",
    "cpu_ms",
    "gtk_stall_max_ms",
];

/// One complete framed dump (`perf-dump-begin` … `perf-dump-end`).
#[derive(Debug, Clone, PartialEq)]
pub struct Dump {
    pub id: u64,
    pub reason: String,
    /// Milliseconds since the child's counters were initialised.
    pub t_ms: u64,
    pub values: BTreeMap<String, u64>,
}

impl Dump {
    pub fn get(&self, name: &str) -> u64 {
        *self
            .values
            .get(name)
            .unwrap_or_else(|| panic!("dump {} lacks required field {name}", self.id))
    }
}

/// Every complete, well-formed dump frame in `log`, in order.  A frame
/// counts only when its end marker was written with the same id and a
/// field count equal to the lines seen, so a read racing the writer
/// never yields a partial snapshot.
pub fn parse_dumps(log: &str) -> Vec<Dump> {
    let mut dumps = Vec::new();
    let mut open: Option<Dump> = None;
    for line in log.lines() {
        if let Some(rest) = line.split("perf-dump-begin: ").nth(1) {
            let field = |k: &str| {
                rest.split_whitespace()
                    .find_map(|kv| kv.strip_prefix(k))
                    .map(str::to_string)
            };
            open = Some(Dump {
                id: field("id=")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(u64::MAX),
                reason: field("reason=").unwrap_or_default(),
                t_ms: field("t=+")
                    .and_then(|v| v.trim_end_matches("ms").parse().ok())
                    .unwrap_or_default(),
                values: BTreeMap::new(),
            });
        } else if let Some(rest) = line.split("perf-dump-end: ").nth(1) {
            let Some(dump) = open.take() else { continue };
            let mut id = None;
            let mut fields = None;
            for kv in rest.split_whitespace() {
                if let Some(v) = kv.strip_prefix("id=") {
                    id = v.parse::<u64>().ok();
                } else if let Some(v) = kv.strip_prefix("fields=") {
                    fields = v.parse::<usize>().ok();
                }
            }
            if id == Some(dump.id) && fields == Some(dump.values.len()) {
                dumps.push(dump);
            }
        } else if let Some(rest) = line.strip_prefix("perf-counter: ") {
            if let (Some(dump), Some((name, value))) = (open.as_mut(), rest.trim().split_once('='))
            {
                match value.parse() {
                    Ok(v) => {
                        dump.values.insert(name.to_string(), v);
                    }
                    // A malformed value invalidates the frame.
                    Err(_) => open = None,
                }
            }
        }
    }
    dumps
}

pub fn parse_log(log: &str, report: &mut RunReport) {
    for line in log.lines() {
        if let Some(rest) = line.split("timing: ").nth(1) {
            let mut parts = rest.split_whitespace();
            let (Some(kind), Some(ms), Some(step)) = (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            let Some(ms) = ms
                .strip_prefix('+')
                .and_then(|v| v.strip_suffix("ms"))
                .and_then(|v| v.parse::<u64>().ok())
            else {
                continue;
            };
            let map = match kind {
                "stop" => &mut report.stop,
                "start" => &mut report.start,
                _ => continue,
            };
            map.entry(step.to_string()).or_insert(ms);
        } else if let Some(rest) = line.split("perf-mark: ").nth(1) {
            if let Some((name, value)) = rest.trim().split_once("=+") {
                if let Some(ms) = value.strip_suffix("ms").and_then(|v| v.parse().ok()) {
                    report.marks.insert(name.to_string(), ms);
                }
            }
        }
    }
    if let Some(last) = parse_dumps(log).pop() {
        report.counters = last.values;
    }
}

fn children_cpu_ms() -> f64 {
    // SAFETY: `getrusage` only writes into the zero-initialised struct
    // we own; RUSAGE_CHILDREN is a valid `who`.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_CHILDREN, &mut usage) };
    let cpu = |tv: libc::timeval| tv.tv_sec as f64 * 1000.0 + tv.tv_usec as f64 / 1000.0;
    cpu(usage.ru_utime) + cpu(usage.ru_stime)
}

/// Serialise child-resource accounting: RUSAGE_CHILDREN is
/// process-wide, so concurrent perf runs would pollute each other.
pub static CHILD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Run to completion (bounded by `timeout`) and parse everything.
pub async fn run(mut cmd: tokio::process::Command, log: &Path, timeout: Duration) -> RunReport {
    let _guard = CHILD_LOCK.lock().await;
    let cpu_before = children_cpu_ms();
    let t0 = Instant::now();
    let output = tokio::time::timeout(timeout, cmd.output())
        .await
        .expect("talk-rs run timed out")
        .expect("spawn talk-rs");
    finish(output, log, t0, cpu_before)
}

/// A running `talk-rs` child that the test drives with signals.
pub struct Live {
    child: tokio::process::Child,
    pub pid: i32,
    log: PathBuf,
    t0: Instant,
    cpu_before: f64,
    _guard: tokio::sync::MutexGuard<'static, ()>,
}

/// Spawn `cmd` (holding [`CHILD_LOCK`] until [`Live::finish`]).
pub async fn spawn(mut cmd: tokio::process::Command, log: &Path) -> Live {
    let guard = CHILD_LOCK.lock().await;
    // Nobody reads a long-lived child's pipes until it exits: a full
    // stderr pipe would block the child inside its logger (holding the
    // log lock).  Send both streams to files instead.
    let stream = |suffix: &str| {
        std::fs::File::create(log.with_extension(suffix)).expect("child output file")
    };
    cmd.stdout(stream("stdout")).stderr(stream("stderr"));
    let cpu_before = children_cpu_ms();
    let t0 = Instant::now();
    let child = cmd.spawn().expect("spawn talk-rs");
    let pid = child.id().expect("child pid") as i32;
    Live {
        child,
        pid,
        log: log.to_path_buf(),
        t0,
        cpu_before,
        _guard: guard,
    }
}

impl Live {
    pub fn signal(&self, signal: nix::sys::signal::Signal) {
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(self.pid), signal)
            .expect("signal own talk-rs child");
    }

    pub async fn wait_for_log(&self, marker: &str, timeout: Duration) {
        wait_for_log(&self.log, marker, timeout).await;
    }

    /// Ask the child for a counter dump (`SIGUSR2`) and return the
    /// first complete frame written after the request, validated to
    /// carry every [`REQUIRED_DUMP_FIELDS`].
    pub async fn dump(&self) -> Dump {
        let seen = parse_dumps(&std::fs::read_to_string(&self.log).unwrap_or_default())
            .last()
            .map(|d| d.id);
        self.signal(nix::sys::signal::Signal::SIGUSR2);
        // Generous: a saturated child (e.g. 200 waterfall threads on
        // open) can take a long time to schedule its signal task.
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            let log = std::fs::read_to_string(&self.log).unwrap_or_default();
            if let Some(dump) = parse_dumps(&log)
                .into_iter()
                .find(|d| d.reason == "signal" && seen.is_none_or(|s| d.id > s))
            {
                for field in REQUIRED_DUMP_FIELDS {
                    let _ = dump.get(field);
                }
                return dump;
            }
            let alive = std::fs::read_to_string(format!("/proc/{}/stat", self.pid))
                .is_ok_and(|stat| !stat.contains(") Z "));
            assert!(alive, "talk-rs exited while waiting for a dump:\n{log}");
            assert!(Instant::now() < deadline, "no SIGUSR2 dump:\n{log}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Wait for exit (after the caller signalled it) and parse.
    pub async fn finish(self, timeout: Duration) -> RunReport {
        let mut output = tokio::time::timeout(timeout, self.child.wait_with_output())
            .await
            .expect("talk-rs did not exit")
            .expect("wait talk-rs");
        output.stdout = std::fs::read(self.log.with_extension("stdout")).unwrap_or_default();
        output.stderr = std::fs::read(self.log.with_extension("stderr")).unwrap_or_default();
        finish(output, &self.log, self.t0, self.cpu_before)
    }
}

/// Spawn, wait for `ready_marker` then `after`, send SIGINT (the
/// dictate stop gesture) and wait for exit.  Returns the report and
/// the instant the SIGINT was sent.
pub async fn run_with_sigint(
    cmd: tokio::process::Command,
    log: &Path,
    ready_marker: &str,
    after: Duration,
    timeout: Duration,
) -> (RunReport, Instant) {
    let live = spawn(cmd, log).await;
    live.wait_for_log(ready_marker, Duration::from_secs(60))
        .await;
    tokio::time::sleep(after).await;
    let sigint_at = Instant::now();
    live.signal(nix::sys::signal::Signal::SIGINT);
    (live.finish(timeout).await, sigint_at)
}

fn finish(output: std::process::Output, log: &Path, t0: Instant, cpu_before: f64) -> RunReport {
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let cpu_after = children_cpu_ms();
    let mut report = RunReport {
        status: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        log: std::fs::read_to_string(log).unwrap_or_default(),
        wall_ms,
        child_cpu_ms: cpu_after - cpu_before,
        started_at: Some(t0),
        ..RunReport::default()
    };
    let log_text = report.log.clone();
    parse_log(&log_text, &mut report);
    report.child_maxrss_kb = report
        .counters
        .get("vm_hwm_kb")
        .copied()
        .unwrap_or_default();
    report
}

/// Poll `log` until it contains `marker`.
pub async fn wait_for_log(log: &Path, marker: &str, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if std::fs::read_to_string(log).is_ok_and(|s| s.contains(marker)) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "log never showed {marker:?}:\n{}",
            std::fs::read_to_string(log).unwrap_or_default()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    #[test]
    fn parses_timing_marks_and_the_last_complete_dump() {
        let log = indoc! {"
            2026-09-30 10:00:00.000 I talk_rs.dictate.oneshot: timing: stop +1ms capture_stopped
            2026-09-30 10:00:00.100 I talk_rs.dictate: timing: stop +433ms ogg_flushed
            2026-09-30 10:00:00.200 I talk_rs.dictate: timing: start +12ms capture_started
            2026-09-30 10:00:00.300 I talk_rs.speak: perf-mark: speak_handoff=+950ms
            2026-09-30 10:00:01.000 I talk_rs.perf_counters.enabled: perf-dump-begin: id=0 reason=signal t=+1000ms
            perf-counter: http_client_builds=3
            perf-dump-end: id=0 fields=1
            2026-09-30 10:00:02.000 I talk_rs.perf_counters.enabled: perf-dump-begin: id=1 reason=exit t=+2000ms
            perf-counter: http_client_builds=4
            perf-dump-end: id=1 fields=1
        "};
        let mut report = RunReport::default();
        parse_log(log, &mut report);
        assert_eq!(report.stop_ms("capture_stopped"), 1);
        assert_eq!(report.stop_ms("ogg_flushed"), 433);
        assert_eq!(report.start.get("capture_started"), Some(&12));
        assert_eq!(report.counter("http_client_builds"), 4);
        assert_eq!(report.marks.get("speak_handoff"), Some(&950));
        let dumps = parse_dumps(log);
        assert_eq!(dumps.len(), 2);
        assert_eq!(
            (dumps[0].id, dumps[0].reason.as_str(), dumps[0].t_ms),
            (0, "signal", 1000)
        );
    }

    /// A reader racing the writer sees a frame without its end marker,
    /// or a frame whose field count does not match: neither is a dump.
    #[test]
    fn partial_or_mismatched_frames_are_not_dumps() {
        let partial = indoc! {"
            x perf-dump-begin: id=5 reason=signal t=+10ms
            perf-counter: a=1
            perf-counter: b=2
        "};
        assert!(parse_dumps(partial).is_empty());
        let mismatched = indoc! {"
            x perf-dump-begin: id=5 reason=signal t=+10ms
            perf-counter: a=1
            perf-dump-end: id=5 fields=2
        "};
        assert!(parse_dumps(mismatched).is_empty());
        let wrong_id = indoc! {"
            x perf-dump-begin: id=5 reason=signal t=+10ms
            perf-counter: a=1
            perf-dump-end: id=6 fields=1
        "};
        assert!(parse_dumps(wrong_id).is_empty());
        let garbage = indoc! {"
            x perf-dump-begin: id=5 reason=signal t=+10ms
            perf-counter: a=12x
            perf-dump-end: id=5 fields=1
        "};
        assert!(parse_dumps(garbage).is_empty());
    }

    /// Interleaved log lines from other targets between frame lines do
    /// not break a frame; a restarted frame supersedes an unfinished one.
    #[test]
    fn interleaved_lines_and_restarted_frames() {
        let log = indoc! {"
            x perf-dump-begin: id=1 reason=signal t=+10ms
            perf-counter: a=1
            x perf-dump-begin: id=2 reason=signal t=+20ms
            2026 D other.target: unrelated line
            perf-counter: a=2
            perf-dump-end: id=2 fields=1
        "};
        let dumps = parse_dumps(log);
        assert_eq!(dumps.len(), 1);
        assert_eq!(dumps[0].id, 2);
        assert_eq!(dumps[0].get("a"), 2);
    }

    #[test]
    #[should_panic(expected = "lacks required field vm_hwm_kb")]
    fn missing_required_field_is_an_error() {
        let log = indoc! {"
            x perf-dump-begin: id=1 reason=signal t=+10ms
            perf-counter: a=1
            perf-dump-end: id=1 fields=1
        "};
        let _ = parse_dumps(log)[0].get("vm_hwm_kb");
    }
}
