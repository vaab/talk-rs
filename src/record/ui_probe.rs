//! Performance-harness observation of the recordings browser
//! (feature `perf-counters` only; not compiled otherwise).
//!
//! The harness needs to know what the user would SEE, not only how
//! much work was done: a cut that lowers a counter by skipping rows,
//! ignoring a file event or dropping a waveform must fail.  With
//! `TALK_RS_PERF_UI_PROBE=<dir>` set, the browser:
//!
//! - polls `<dir>/commands` (one command per line, consumed once;
//!   writers publish it by atomic rename) on the GTK main loop and
//!   answers each in `<dir>/replies`;
//! - on `snapshot`, writes `<dir>/snapshot-<n>.tsv`: one line per list
//!   row, `section \t path \t kind \t text \t waveform \t selected \t
//!   button`, where `kind` is `transcript`, `player`, `in-progress` or
//!   `other`, `waveform` is `ready` / `pending` / `-`, `selected` is
//!   `0`/`1` and `button` is the play button's tooltip (or `-`);
//! - on `row <path>`, replies with that one row's description (cheap:
//!   no full walk, so polling it barely disturbs the main loop);
//! - on `click <path>`, activates the matching row's real play/pause
//!   button (the production click handler runs) and replies with the
//!   GTK-thread time the handler took (`handler-ms`) and the wall
//!   clock at the click (`epoch-ms`, comparable with the capturing
//!   audio sink's `first-audio` stamp).
//!
//! Widgets are found structurally (row name = audio path, the player
//! bar's buttons by CSS class `play-btn`, waveform readiness by the
//! `wf-ready` CSS class that the harness build adds when a waterfall
//! result is applied), so no production widget changes are needed
//! beyond that one class.

use gtk4::glib;
use gtk4::prelude::*;
use std::io::Write;
use std::path::PathBuf;

/// Install the probe on the given section lists.  No-op when the
/// environment variable is absent.
pub(super) fn install(sections: Vec<(&'static str, gtk4::ListBox)>) {
    let Some(dir) = std::env::var_os("TALK_RS_PERF_UI_PROBE").map(PathBuf::from) else {
        return;
    };
    let _ = std::fs::create_dir_all(&dir);
    let mut snapshot_seq = 0u32;
    glib::timeout_add_local(std::time::Duration::from_millis(50), move || {
        let commands = dir.join("commands");
        let Ok(text) = std::fs::read_to_string(&commands) else {
            return glib::ControlFlow::Continue;
        };
        // Writers publish commands by rename; an empty read is a writer
        // that does not, caught mid-write: leave it for the next poll.
        if text.trim().is_empty() {
            return glib::ControlFlow::Continue;
        }
        let _ = std::fs::remove_file(&commands);
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let reply = run(line, &sections, &dir, &mut snapshot_seq);
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("replies"))
            {
                // One write per reply so a reader never sees half a line.
                let _ = f.write_all(format!("{line} => {reply}\n").as_bytes());
            }
        }
        glib::ControlFlow::Continue
    });
}

fn run(
    line: &str,
    sections: &[(&'static str, gtk4::ListBox)],
    dir: &std::path::Path,
    seq: &mut u32,
) -> String {
    let (verb, arg) = line.split_once(' ').unwrap_or((line, ""));
    match verb {
        "snapshot" => {
            *seq += 1;
            let path = dir.join(format!("snapshot-{seq}.tsv"));
            let mut out = String::new();
            for (label, list) in sections {
                let mut idx = 0;
                while let Some(row) = list.row_at_index(idx) {
                    out.push_str(&describe_row(label, &row));
                    out.push('\n');
                    idx += 1;
                }
            }
            match std::fs::write(&path, out) {
                Ok(()) => format!("ok {}", path.display()),
                Err(e) => format!("error {e}"),
            }
        }
        "row" => match sections
            .iter()
            .find_map(|(label, list)| find_in(list, arg).map(|row| (label, row)))
        {
            Some((label, row)) => format!("ok {}", describe_row(label, &row)),
            None => "absent".into(),
        },
        "click" => {
            let Some(row) = find_row(sections, arg) else {
                return "error no-such-row".into();
            };
            let Some(button) = buttons(&row).into_iter().find(is_play_button) else {
                return "error no-play-button".into();
            };
            let epoch = epoch_ms();
            let t0 = std::time::Instant::now();
            button.emit_clicked();
            format!(
                "ok handler-ms={} epoch-ms={epoch}",
                t0.elapsed().as_millis()
            )
        }
        _ => "error unknown-command".into(),
    }
}

fn epoch_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default()
}

fn find_in(list: &gtk4::ListBox, path: &str) -> Option<gtk4::ListBoxRow> {
    let mut idx = 0;
    while let Some(row) = list.row_at_index(idx) {
        if row.widget_name().as_str() == path {
            return Some(row);
        }
        idx += 1;
    }
    None
}

fn find_row(sections: &[(&'static str, gtk4::ListBox)], path: &str) -> Option<gtk4::ListBoxRow> {
    sections.iter().find_map(|(_, list)| find_in(list, path))
}

fn descendants(widget: &gtk4::Widget, out: &mut Vec<gtk4::Widget>) {
    let mut child = widget.first_child();
    while let Some(w) = child {
        out.push(w.clone());
        descendants(&w, out);
        child = w.next_sibling();
    }
}

fn buttons(row: &gtk4::ListBoxRow) -> Vec<gtk4::Button> {
    let mut all = Vec::new();
    descendants(row.upcast_ref(), &mut all);
    all.into_iter()
        .filter_map(|w| w.downcast::<gtk4::Button>().ok())
        .collect()
}

/// The player bar's play/pause button (icon button with tooltip
/// starting "Play"/"Pause"/"Resume"), not rewind.
fn is_play_button(button: &gtk4::Button) -> bool {
    button.has_css_class("play-btn")
        && button
            .tooltip_text()
            .is_some_and(|t| !t.starts_with("Rewind"))
}

fn describe_row(section: &str, row: &gtk4::ListBoxRow) -> String {
    let mut all = Vec::new();
    descendants(row.upcast_ref(), &mut all);
    let waveform = all.iter().find(|w| w.has_css_class("waterfall")).map(|w| {
        if w.has_css_class("wf-ready") {
            "ready"
        } else {
            "pending"
        }
    });
    let transcript = all
        .iter()
        .filter(|w| w.has_css_class("transcript"))
        .find_map(|w| {
            w.downcast_ref::<gtk4::Label>()
                .map(|l| l.text().to_string())
        });
    let button = buttons(row)
        .into_iter()
        .find(is_play_button)
        .and_then(|b| b.tooltip_text())
        .map_or("-".to_string(), |t| t.to_string());
    let (kind, text) = match (&waveform, transcript) {
        (Some(_), _) => ("player", String::new()),
        (None, Some(t)) if t == "(transcription ongoing)" => ("in-progress", String::new()),
        (None, Some(t)) => ("transcript", t),
        (None, None) => ("other", String::new()),
    };
    format!(
        "{section}\t{}\t{kind}\t{}\t{}\t{}\t{button}",
        row.widget_name(),
        text.replace(['\t', '\n'], " "),
        waveform.unwrap_or("-"),
        u8::from(row.is_selected()),
    )
}
