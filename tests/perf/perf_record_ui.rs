//! `record --ui` on a 1500-recording library: items
//! `record-ui-row-work`, `record-ui-incremental-listing`, and the
//! playback cut of `streaming-playback-decode`.
//!
//! Runs the real recordings browser on an isolated X display with the
//! harness probe (`TALK_RS_PERF_UI_PROBE`) and the capturing audio sink
//! (`TALK_RS_PERF_AUDIO_SINK`).  Work is scored only after what the
//! user would SEE is complete and correct:
//!
//! - every library entry has exactly one row, newest first, showing
//!   its pick text, "(transcription ongoing)" or a player bar;
//! - every player bar shows its waveform (warm, stale, corrupt and
//!   missing `.wf`; OGG, M4A, MP4 and long rows);
//! - after a pick is written, that row shows the new text, nothing else
//!   changed, and the selection is preserved;
//! - Play on the long row, through its real button, produces sound from
//!   the start of that file (first meaningful sample consumed by the
//!   sink); pause, resume and handover to another row still work.
//!
//! Timer ownership is scored separately from row presence: the number
//! of player bars comes from the probe, the recurring tick sources
//! from the counters.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::require_counters;
use crate::support::display::IsolatedDisplay;
use crate::support::fixtures::{self, ExpectedRow, LibrarySpec};
use crate::support::runner::{self, Dump, Live, Sandbox};
use crate::support::Metrics;

const IDLE_WINDOW: Duration = Duration::from_secs(10);

/// One row as reported by the probe.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    section: String,
    path: PathBuf,
    kind: String,
    text: String,
    waveform: String,
    selected: bool,
    button: String,
}

fn parse_row(line: &str) -> Row {
    let f: Vec<&str> = line.split('\t').collect();
    assert_eq!(f.len(), 7, "malformed probe row: {line:?}");
    Row {
        section: f[0].to_string(),
        path: PathBuf::from(f[1]),
        kind: f[2].to_string(),
        text: f[3].to_string(),
        waveform: f[4].to_string(),
        selected: f[5] == "1",
        button: f[6].to_string(),
    }
}

/// Talks to the in-process probe through its command/reply files.
struct Probe {
    dir: PathBuf,
    replies_seen: usize,
}

impl Probe {
    async fn ask(&mut self, command: &str) -> String {
        // Atomic hand-over: the probe must never see a half-written file.
        let tmp = self.dir.join("commands.tmp");
        std::fs::write(&tmp, format!("{command}\n")).expect("probe command");
        std::fs::rename(&tmp, self.dir.join("commands")).expect("publish probe command");
        let deadline = Instant::now() + Duration::from_secs(300);
        loop {
            let replies = std::fs::read_to_string(self.dir.join("replies")).unwrap_or_default();
            // Only newline-terminated lines are complete replies.
            let complete = replies.rfind('\n').map_or("", |end| &replies[..end]);
            let lines: Vec<&str> = complete.lines().collect();
            if lines.len() > self.replies_seen {
                let reply = lines[self.replies_seen].to_string();
                self.replies_seen += 1;
                let (asked, answer) = reply.split_once(" => ").expect("probe reply format");
                assert_eq!(asked, command, "probe replied to another command");
                return answer.to_string();
            }
            assert!(
                Instant::now() < deadline,
                "probe did not answer {command:?}"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    async fn snapshot(&mut self) -> Vec<Row> {
        let answer = self.ask("snapshot").await;
        let path = answer
            .strip_prefix("ok ")
            .unwrap_or_else(|| panic!("snapshot failed: {answer}"));
        std::fs::read_to_string(path)
            .expect("snapshot file")
            .lines()
            .map(parse_row)
            .collect()
    }

    async fn row(&mut self, path: &Path) -> Option<Row> {
        let answer = self.ask(&format!("row {}", path.display())).await;
        answer.strip_prefix("ok ").map(parse_row)
    }

    /// Click the row's play/pause button; returns (handler ms, epoch ms).
    async fn click(&mut self, path: &Path) -> (f64, u128) {
        let answer = self.ask(&format!("click {}", path.display())).await;
        let field = |k: &str| {
            answer
                .split_whitespace()
                .find_map(|kv| kv.strip_prefix(k))
                .unwrap_or_else(|| panic!("click failed: {answer}"))
                .to_string()
        };
        (
            field("handler-ms=").parse().expect("handler ms"),
            field("epoch-ms=").parse().expect("epoch ms"),
        )
    }
}

/// Poll fresh snapshots until `condition` holds.
async fn wait_rows<F: Fn(&[Row]) -> bool>(
    probe: &mut Probe,
    what: &str,
    timeout: Duration,
    condition: F,
) -> Vec<Row> {
    let t0 = Instant::now();
    loop {
        let rows = probe.snapshot().await;
        if condition(&rows) {
            return rows;
        }
        if t0.elapsed() >= timeout {
            let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
            for r in &rows {
                *counts
                    .entry((r.section.clone(), r.kind.clone()))
                    .or_default() += 1;
            }
            panic!("UI never reached: {what}; rows by (section, kind): {counts:?}");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// The Recordings section must match the library exactly.
fn assert_library_rows(rows: &[Row], expected: &BTreeMap<PathBuf, ExpectedRow>, order: &[PathBuf]) {
    let library: Vec<&Row> = rows.iter().filter(|r| r.section == "Recordings").collect();
    let paths: Vec<PathBuf> = library.iter().map(|r| r.path.clone()).collect();
    assert_eq!(paths.len(), order.len(), "one row per recording");
    assert_eq!(paths, order, "newest-first order");
    for row in library {
        match &expected[&row.path] {
            ExpectedRow::Transcript(text) => assert_eq!(
                (row.kind.as_str(), &row.text),
                ("transcript", text),
                "{:?}",
                row.path
            ),
            ExpectedRow::Player => assert_eq!(row.kind, "player", "{:?}", row.path),
            ExpectedRow::InProgress => assert_eq!(row.kind, "in-progress", "{:?}", row.path),
        }
    }
}

fn library_rows(rows: &[Row]) -> usize {
    rows.iter().filter(|r| r.section == "Recordings").count()
}

fn all_waveforms_ready(rows: &[Row]) -> bool {
    rows.iter()
        .filter(|r| r.kind == "player")
        .all(|r| r.waveform == "ready")
}

fn delta(after: &Dump, before: &Dump, name: &str) -> f64 {
    after.get(name) as f64 - before.get(name) as f64
}

/// Seconds between two dumps, from the child's own clock.
fn window_s(after: &Dump, before: &Dump) -> f64 {
    after.t_ms.saturating_sub(before.t_ms) as f64 / 1000.0
}

/// Latest sink `first-audio` event: (epoch ms, sample index).
fn first_audio(sink: &Path) -> Option<(u128, u64)> {
    std::fs::read_to_string(sink)
        .ok()?
        .lines()
        .rev()
        .find_map(|l| {
            let rest = l.strip_prefix("first-audio ")?;
            let field = |k: &str| rest.split_whitespace().find_map(|kv| kv.strip_prefix(k));
            Some((
                field("epoch=")?.parse().ok()?,
                field("sample=")?.parse().ok()?,
            ))
        })
}

struct Browser {
    live: Live,
    probe: Probe,
    sink: PathBuf,
}

async fn open_browser(sandbox: &Sandbox, display: &IsolatedDisplay) -> Browser {
    let probe_dir = sandbox.path().join("probe");
    std::fs::create_dir_all(&probe_dir).expect("probe dir");
    let sink = sandbox.path().join("audio-sink.log");
    let mut cmd = sandbox.command(&["record", "--ui"]);
    cmd.env("DISPLAY", &display.display)
        .env("GDK_BACKEND", "x11")
        .env("TALK_RS_PERF_GTK_PROBE", "1")
        .env("TALK_RS_PERF_UI_PROBE", &probe_dir)
        .env("TALK_RS_PERF_AUDIO_SINK", &sink);
    Browser {
        live: runner::spawn(cmd, &sandbox.log_path()).await,
        probe: Probe {
            dir: probe_dir,
            replies_seen: 0,
        },
        sink,
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs an isolated X display (Xvfb or weston)"]
async fn perf_record_ui_library_1500() {
    require_counters!();
    let Some(display) = IsolatedDisplay::start() else {
        eprintln!("SKIP: neither Xvfb nor weston available");
        return;
    };
    let sandbox = Sandbox::new();
    let spec = LibrarySpec::user_like();
    let library = fixtures::build_library(&sandbox.output_dir(), &spec);
    sandbox.write_config(&format!(
        "output_dir: {}\nproviders:\n  mistral:\n    api_key: fake\n",
        sandbox.output_dir().display()
    ));
    let expected: BTreeMap<PathBuf, ExpectedRow> = library.rows.iter().cloned().collect();
    let order = library.expected_order();
    let players = expected
        .values()
        .filter(|e| **e == ExpectedRow::Player)
        .count();

    let mut b = open_browser(&sandbox, &display).await;
    let t_open = Instant::now();

    // ── Open: every row present and correct; waveforms delivered.
    let rows = wait_rows(
        &mut b.probe,
        "all rows listed",
        Duration::from_secs(300),
        |rows| library_rows(rows) == order.len(),
    )
    .await;
    let rows_ms = t_open.elapsed().as_secs_f64() * 1000.0;
    assert_library_rows(&rows, &expected, &order);
    wait_rows(
        &mut b.probe,
        "first waveform",
        Duration::from_secs(300),
        |rows| {
            rows.iter()
                .any(|r| r.kind == "player" && r.waveform == "ready")
        },
    )
    .await;
    let first_waveform_ms = t_open.elapsed().as_secs_f64() * 1000.0;
    let rows = wait_rows(
        &mut b.probe,
        "every waveform",
        Duration::from_secs(900),
        all_waveforms_ready,
    )
    .await;
    let all_waveforms_ms = t_open.elapsed().as_secs_f64() * 1000.0;
    assert_library_rows(&rows, &expected, &order);
    assert_eq!(rows.iter().filter(|r| r.kind == "player").count(), players);
    let selected_before: Vec<PathBuf> = rows
        .iter()
        .filter(|r| r.selected)
        .map(|r| r.path.clone())
        .collect();
    let opened = b.live.dump().await;
    assert_eq!(
        opened.get("waterfall_jobs_applied"),
        opened.get("waterfall_jobs_requested"),
        "every requested waveform was applied to its row"
    );

    // ── Idle window (no playback).
    tokio::time::sleep(IDLE_WINDOW).await;
    let idle = b.live.dump().await;
    let idle_s = window_s(&idle, &opened);

    // ── One file event: a pick appears for a no-pick OGG row.
    let target = library
        .rows
        .iter()
        .find(|(p, e)| {
            *e == ExpectedRow::Player
                && p.extension().is_some_and(|x| x == "ogg")
                && !library.long_rows.contains(p)
        })
        .map(|(p, _)| p.clone())
        .expect("an ogg player row");
    std::fs::write(
        target.with_extension("pick.yml"),
        "provider: mistral\nmodel: voxtral-mini-2602\nstreaming: false\ntext: freshly picked words\n",
    )
    .expect("new pick");
    let t_event = Instant::now();
    loop {
        let row = b.probe.row(&target).await;
        if row
            .as_ref()
            .is_some_and(|r| r.kind == "transcript" && r.text == "freshly picked words")
        {
            break;
        }
        assert!(
            t_event.elapsed() < Duration::from_secs(120),
            "pick event never shown: {row:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let event_ack_ms = t_event.elapsed().as_secs_f64() * 1000.0;
    // Let any follow-up work of the event land before sampling.
    tokio::time::sleep(Duration::from_secs(2)).await;
    let rows = b.probe.snapshot().await;
    let mut expected_after = expected.clone();
    expected_after.insert(
        target.clone(),
        ExpectedRow::Transcript("freshly picked words".into()),
    );
    assert_library_rows(&rows, &expected_after, &order);
    let selected_after: Vec<PathBuf> = rows
        .iter()
        .filter(|r| r.selected)
        .map(|r| r.path.clone())
        .collect();
    assert_eq!(
        selected_after, selected_before,
        "selection preserved across the event"
    );
    let after_event = b.live.dump().await;

    // ── Play the long row through its real play button.
    let long = library.long_rows.first().expect("long row").clone();
    let short = library
        .rows
        .iter()
        .find(|(p, e)| {
            *e == ExpectedRow::Player
                && *p != target
                && !library.long_rows.contains(p)
                && p.extension().is_some_and(|x| x == "ogg")
        })
        .map(|(p, _)| p.clone())
        .expect("short player row");
    let before_play = b.live.dump().await;
    let (handler_ms, click_epoch) = b.probe.click(&long).await;
    let t_play = Instant::now();
    let (sound_epoch, first_sample) = loop {
        if let Some(found) = first_audio(&b.sink) {
            break found;
        }
        assert!(
            t_play.elapsed() < Duration::from_secs(300),
            "no sound after Play"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    let click_to_sound_ms = sound_epoch.saturating_sub(click_epoch) as f64;
    // Sound starts at the beginning of the file (the fixture's first
    // syllable is within 0.4 s), not somewhere later.
    assert!(
        first_sample < 48_000 / 2,
        "first sound at sample {first_sample}"
    );
    tokio::time::sleep(Duration::from_secs(2)).await;
    let playing = b.live.dump().await;
    let row = b.probe.row(&long).await.expect("long row");
    assert!(
        row.button.starts_with("Pause"),
        "playing row offers Pause: {row:?}"
    );
    b.probe.click(&long).await;
    let row = b.probe.row(&long).await.expect("long row");
    assert!(
        row.button.starts_with("Resume"),
        "paused row offers Resume: {row:?}"
    );
    b.probe.click(&long).await;
    let row = b.probe.row(&long).await.expect("long row");
    assert!(
        row.button.starts_with("Pause"),
        "resumed row offers Pause: {row:?}"
    );
    b.probe.click(&short).await;
    let short_row = b.probe.row(&short).await.expect("short row");
    let long_row = b.probe.row(&long).await.expect("long row");
    assert!(
        short_row.button.starts_with("Pause"),
        "handover: {short_row:?}"
    );
    // The released row's tick reverts it on its next 16 ms callback.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let long_row_after = b.probe.row(&long).await.expect("long row");
    assert!(
        !long_row_after.button.starts_with("Pause"),
        "previous row released: {long_row:?} → {long_row_after:?}"
    );

    b.live.signal(nix::sys::signal::Signal::SIGTERM);
    let _ = b.live.finish(Duration::from_secs(30)).await;
    drop(display);

    let long_samples = spec.long_seconds * 48_000.0;
    Metrics::new("record-ui-row-work", "library-1500-idle")
        .set("player_bars", players as f64)
        .set(
            "player_tick_sources_live",
            idle.get("player_tick_sources_live") as f64,
        )
        .set(
            "player_tick_callbacks_per_s",
            delta(&idle, &opened, "player_tick_callbacks") / idle_s,
        )
        .set(
            "idle_cpu_ms_per_s",
            delta(&idle, &opened, "cpu_ms") / idle_s,
        )
        .set("idle_window_s", idle_s)
        .set("vm_hwm_kb", idle.get("vm_hwm_kb") as f64)
        .write();
    Metrics::new("record-ui-row-work", "library-1500-open")
        .set(
            "waterfall_jobs_requested",
            opened.get("waterfall_jobs_requested") as f64,
        )
        .set(
            "waterfall_jobs_applied",
            opened.get("waterfall_jobs_applied") as f64,
        )
        .set(
            "waterfall_workers_inflight_max",
            opened.get("waterfall_workers_inflight_max") as f64,
        )
        .set("waterfall_decodes", opened.get("waterfall_decodes") as f64)
        .set("threads_after_open", opened.get("threads") as f64)
        .set("first_waveform_ms", first_waveform_ms)
        .set("all_waveforms_ms", all_waveforms_ms)
        .write();
    Metrics::new("record-ui-row-work", "play-long-row")
        .set("click_handler_ms", handler_ms)
        .set("click_to_first_sound_ms", click_to_sound_ms)
        .set("gtk_stall_max_ms", playing.get("gtk_stall_max_ms") as f64)
        .set(
            "playback_decodes",
            delta(&playing, &before_play, "playback_decodes"),
        )
        .write();
    // Resident growth while the long row plays: what the process holds
    // for playback, whatever structure holds it (not gameable by a
    // gauge that is simply not updated).
    let play_rss_growth_mib = delta(&playing, &before_play, "vm_rss_kb") / 1024.0;
    Metrics::new("streaming-playback-decode", "play-long-row")
        .set("click_to_first_sound_ms", click_to_sound_ms)
        .set(
            "player_retained_mib",
            playing.get("player_retained_samples_max") as f64 * 4.0 / 1_048_576.0,
        )
        .set(
            "retained_fraction_of_file",
            playing.get("player_retained_samples_max") as f64 / long_samples,
        )
        .set("play_rss_growth_mib", play_rss_growth_mib)
        .write();
    Metrics::new("record-ui-incremental-listing", "library-1500-open")
        .set("rows_listed_ms", rows_ms)
        .set("list_calls", opened.get("list_calls") as f64)
        .set("pick_reads", opened.get("pick_reads") as f64)
        .set(
            "pick_read_attempts",
            opened.get("pick_read_attempts") as f64,
        )
        .set("gtk_stall_max_ms", opened.get("gtk_stall_max_ms") as f64)
        .write();
    Metrics::new(
        "record-ui-incremental-listing",
        "library-1500-one-pick-event",
    )
    .set("event_to_row_updated_ms", event_ack_ms)
    .set("list_calls", delta(&after_event, &idle, "list_calls"))
    .set("pick_reads", delta(&after_event, &idle, "pick_reads"))
    .set(
        "pick_read_attempts",
        delta(&after_event, &idle, "pick_read_attempts"),
    )
    .set(
        "gtk_stall_max_ms",
        after_event.get("gtk_stall_max_ms") as f64,
    )
    .write();
}
