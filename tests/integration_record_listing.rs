//! Real GIO event coverage on the performance harness's isolated display.
#![cfg(all(feature = "ui", feature = "perf-counters"))]

#[path = "perf/support/mod.rs"]
mod support;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use support::display::IsolatedDisplay;
use support::runner::{self, Sandbox};

struct Probe {
    dir: PathBuf,
    replies: usize,
}

impl Probe {
    async fn snapshot(&mut self) -> Vec<(PathBuf, String)> {
        std::fs::write(self.dir.join("next"), "snapshot\n").expect("command");
        std::fs::rename(self.dir.join("next"), self.dir.join("commands")).expect("publish");
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let replies = std::fs::read_to_string(self.dir.join("replies")).unwrap_or_default();
            if let Some(end) = replies.rfind('\n') {
                if let Some(reply) = replies[..end].lines().nth(self.replies) {
                    self.replies += 1;
                    let path = reply
                        .strip_prefix("snapshot => ok ")
                        .expect("snapshot reply");
                    return std::fs::read_to_string(path)
                        .expect("snapshot")
                        .lines()
                        .filter_map(|line| {
                            let fields: Vec<_> = line.split('\t').collect();
                            assert_eq!(fields.len(), 7);
                            (fields[0] == "Recordings")
                                .then(|| (PathBuf::from(fields[1]), fields[3].to_owned()))
                        })
                        .collect();
                }
            }
            assert!(Instant::now() < deadline, "probe did not reply");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    async fn expect_rows(&mut self, expected: &[(PathBuf, String)]) {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let actual = self.snapshot().await;
            if actual == expected {
                return;
            }
            if Instant::now() >= deadline {
                assert_eq!(actual, expected, "browser rows did not converge");
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "uses the harness-owned isolated X display"]
async fn pick_written_during_materialization_overrides_initial_row() {
    let display = IsolatedDisplay::start().expect("isolated X display");
    let sandbox = Sandbox::new();
    let mut expected = Vec::new();
    for i in 0..500 {
        let path = sandbox.output_dir().join(format!("voice-{i:04}.ogg"));
        write_recording(&path, "stable");
        expected.push((path, "stable".into()));
    }
    expected.reverse();
    let late = sandbox.output_dir().join("0000.ogg");
    std::fs::write(&late, b"audio").expect("late audio");
    expected.push((late.clone(), "fresh text".into()));
    let (live, mut probe) = browser(&sandbox, &display).await;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let rows = probe.snapshot().await;
        if !rows.is_empty() && rows.len() < expected.len() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "no partial materialization observed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    talk_rs::recording_cache::write_pick(&late, "mistral", "model", false, "fresh text")
        .expect("late pick");
    probe.expect_rows(&expected).await;
    live.signal(nix::sys::signal::Signal::SIGTERM);
    let _ = live.finish(Duration::from_secs(10)).await;
}

fn write_recording(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().expect("parent")).expect("directory");
    talk_rs::recording_cache::write_pick(path, "mistral", "model", false, text).expect("pick");
    std::fs::write(path, b"listing fixture").expect("audio");
}

async fn browser(sandbox: &Sandbox, display: &IsolatedDisplay) -> (runner::Live, Probe) {
    sandbox.write_config(&format!(
        "output_dir: {}\nproviders:\n  mistral:\n    api_key: fake\n",
        sandbox.output_dir().display()
    ));
    let probe_dir = sandbox.path().join("probe");
    std::fs::create_dir(&probe_dir).expect("probe directory");
    let mut cmd = sandbox.command(&["record", "--ui"]);
    cmd.env("DISPLAY", &display.display)
        .env("GDK_BACKEND", "x11")
        .env("TALK_RS_PERF_UI_PROBE", &probe_dir)
        .env("TALK_RS_PERF_AUDIO_SINK", sandbox.path().join("sink"));
    let live = runner::spawn(cmd, &sandbox.log_path()).await;
    (
        live,
        Probe {
            dir: probe_dir,
            replies: 0,
        },
    )
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "uses the harness-owned isolated X display"]
async fn nested_directory_recreation_keeps_live_transcript_updates() {
    let display = IsolatedDisplay::start().expect("isolated X display");
    let sandbox = Sandbox::new();
    let stable = sandbox.output_dir().join("stable.ogg");
    write_recording(&stable, "unchanged");
    let (live, mut probe) = browser(&sandbox, &display).await;
    let stable_row = (stable.clone(), "unchanged".into());
    probe.expect_rows(std::slice::from_ref(&stable_row)).await;
    let nested = sandbox.output_dir().join("new/month.ogg");
    let audio = nested.join("voice_with_underscores.ogg");
    for text in ["first", "recreated"] {
        write_recording(&audio, text);
        probe
            .expect_rows(&[(audio.clone(), text.into()), stable_row.clone()])
            .await;
        talk_rs::recording_cache::write_pick(&audio, "mistral", "model", false, "更新🙂")
            .expect("changed pick");
        probe
            .expect_rows(&[(audio.clone(), "更新🙂".into()), stable_row.clone()])
            .await;
        let parked = sandbox.path().join("outside-library");
        std::fs::rename(&nested, &parked).expect("move subtree out");
        probe.expect_rows(std::slice::from_ref(&stable_row)).await;
        std::fs::remove_dir_all(parked).expect("remove moved fixture");
    }
    live.signal(nix::sys::signal::Signal::SIGTERM);
    let _ = live.finish(Duration::from_secs(10)).await;
}
