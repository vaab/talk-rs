//! `record --ui` on a 1500-recording library (items `record-ui-row-work`
//! and `record-ui-incremental-listing`).
//!
//! Runs the real recordings browser on an isolated X display (Xvfb or
//! headless Weston + Xwayland), samples counters with `SIGUSR2` dumps:
//! after the library is populated, after a 10 s idle window, and after
//! one pick file changes on disk.  `gtk_stall_max_ms` is the largest
//! gap between 16 ms GTK main-loop ticks in each window.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::require_counters;
use crate::support::display::IsolatedDisplay;
use crate::support::fixtures::{self, LibrarySpec};
use crate::support::runner::{self, Sandbox};
use crate::support::Metrics;

const IDLE_WINDOW: Duration = Duration::from_secs(10);

/// Dump repeatedly until the work counters stop moving (two equal
/// consecutive dumps 500 ms apart) and no waterfall worker is running.
/// Returns the settled dump, with `gtk_stall_max_ms` replaced by the
/// maximum seen over the whole wait, and the wait duration.
async fn settle(live: &runner::Live, timeout: Duration) -> (BTreeMap<String, u64>, f64) {
    const WORK: [&str; 4] = [
        "list_calls",
        "pick_reads",
        "waterfall_decodes",
        "playback_decodes",
    ];
    let t0 = std::time::Instant::now();
    let mut stall_max = 0;
    let mut previous: Option<BTreeMap<String, u64>> = None;
    loop {
        let mut dump = live.dump().await;
        stall_max = stall_max.max(dump["gtk_stall_max_ms"]);
        let stable = previous
            .as_ref()
            .is_some_and(|p| WORK.iter().all(|k| p.get(*k) == dump.get(*k)));
        if stable && dump["waterfall_workers_inflight"] == 0 {
            dump.insert("gtk_stall_max_ms".into(), stall_max);
            return (dump, t0.elapsed().as_secs_f64() * 1000.0);
        }
        assert!(
            t0.elapsed() < timeout,
            "record --ui never settled: {dump:?}"
        );
        previous = Some(dump);
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

fn delta(after: &BTreeMap<String, u64>, before: &BTreeMap<String, u64>, name: &str) -> f64 {
    after.get(name).copied().unwrap_or_default() as f64
        - before.get(name).copied().unwrap_or_default() as f64
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

    let mut cmd = sandbox.command(&["record", "--ui"]);
    cmd.env("DISPLAY", &display.display)
        .env("GDK_BACKEND", "x11")
        .env("TALK_RS_PERF_GTK_PROBE", "1");
    let live = runner::spawn(cmd, &sandbox.log_path()).await;
    const ROWS_BUILT: &str = "perf-mark: record_ui_rows_built_Recordings=+";
    live.wait_for_log(ROWS_BUILT, Duration::from_secs(300))
        .await;
    let open_ms = std::fs::read_to_string(sandbox.log_path())
        .unwrap_or_default()
        .lines()
        .find_map(|l| l.split(ROWS_BUILT).nth(1))
        .and_then(|v| v.trim_end_matches("ms").parse::<f64>().ok())
        .unwrap_or(f64::NAN);
    // Let the per-row waterfall workers of the initial open finish.
    let (opened, waterfalls_ms) = settle(&live, Duration::from_secs(300)).await;

    tokio::time::sleep(IDLE_WINDOW).await;
    let idle = live.dump().await;

    // One file event: a pick appears for a row that had none (as when
    // a dictation or picker finishes while the browser is open).
    let target = &library[0];
    std::fs::write(
        target.with_extension("pick.yml"),
        "provider: mistral\nmodel: voxtral-mini-2602\nstreaming: false\ntext: new pick\n",
    )
    .expect("new pick");
    let (after_event, _) = settle(&live, Duration::from_secs(120)).await;

    live.signal(nix::sys::signal::Signal::SIGTERM);
    let _ = live.finish(Duration::from_secs(20)).await;
    drop(display);

    // Invariants: every recording listed once per listing, every
    // no-pick row got a player bar.
    let rows_with_player = spec.without_pick as u64;
    assert_eq!(
        opened["list_calls"], 2,
        "initial open lists cache + library"
    );
    assert_eq!(
        opened["player_tick_sources_live_max"], rows_with_player,
        "one player bar per no-pick row: {opened:?}"
    );

    let idle_s = IDLE_WINDOW.as_secs_f64();
    Metrics::new("record-ui-row-work", "library-1500-idle")
        .set("rows_with_player", rows_with_player as f64)
        .set(
            "player_tick_sources_live",
            idle["player_tick_sources_live"] as f64,
        )
        .set(
            "player_tick_callbacks_per_s",
            delta(&idle, &opened, "player_tick_callbacks") / idle_s,
        )
        .set(
            "waterfall_workers_inflight_max",
            opened["waterfall_workers_inflight_max"] as f64,
        )
        .set("waterfall_decodes", opened["waterfall_decodes"] as f64)
        .set("rows_built_to_waterfalls_done_ms", waterfalls_ms)
        .set(
            "idle_cpu_ms_per_s",
            delta(&idle, &opened, "cpu_ms") / idle_s,
        )
        .set("threads_after_open", opened["threads"] as f64)
        .set("vm_hwm_kb", idle["vm_hwm_kb"] as f64)
        .write();
    Metrics::new("record-ui-incremental-listing", "library-1500-open")
        .set("open_to_rows_built_ms", open_ms)
        .set("list_calls", opened["list_calls"] as f64)
        .set("pick_reads", opened["pick_reads"] as f64)
        .set(
            "recordings_with_pick",
            (spec.recordings - spec.without_pick) as f64,
        )
        .set("gtk_stall_max_ms", opened["gtk_stall_max_ms"] as f64)
        .write();
    Metrics::new(
        "record-ui-incremental-listing",
        "library-1500-one-pick-event",
    )
    .set("list_calls", delta(&after_event, &idle, "list_calls"))
    .set("pick_reads", delta(&after_event, &idle, "pick_reads"))
    .set("gtk_stall_max_ms", after_event["gtk_stall_max_ms"] as f64)
    .write();
}
