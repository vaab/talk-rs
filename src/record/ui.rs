//! GTK4 recordings browser for the `record --ui` command.
//!
//! Opens a window listing all cached recordings with metadata
//! (date, duration, size, transcript preview) and controls to
//! play or delete each entry.  Recordings are split into two
//! collapsible sections: OGG recordings and dictation cache.

use super::entries::{
    compare_entries_newest_first, delete_recording, has_audio_extension, open_in_file_manager,
    RecordingCollection, RecordingEntry,
};
use super::player::WavPlayer;
use crate::config::Config;
use crate::error::TalkError;
use crate::widgets::audio_player_bar::ButtonFaces;
use crate::widgets::playback_session::{PlaybackRow, PlaybackSession, RowRef};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc;

/// The plain ▶/■ play button of a row that has a transcript (rows
/// without one use the full player bar).  Driven by the window's
/// [`PlaybackSession`] like any other row: one shared tick, decode
/// off the GTK thread, released when another row plays.
struct SimplePlayRow {
    button: gtk4::glib::WeakRef<gtk4::Button>,
    faces: ButtonFaces<gtk4::Label>,
}

impl SimplePlayRow {
    fn new(button: &gtk4::Button) -> Self {
        use gtk4::prelude::*;
        let faces = ButtonFaces::install(
            button,
            gtk4::Label::new(Some("▶")),
            gtk4::Label::new(Some("■")),
        );
        button.set_tooltip_text(Some("Play recording"));
        Self {
            button: button.downgrade(),
            faces,
        }
    }

    fn show_playing(&self) {
        use gtk4::prelude::*;
        self.faces.set_active(true);
        if let Some(button) = self.button.upgrade() {
            button.set_tooltip_text(Some("Stop playback"));
        }
    }
}

impl PlaybackRow for SimplePlayRow {
    fn set_idle(&self, _finished: bool) {
        use gtk4::prelude::*;
        self.faces.set_active(false);
        if let Some(button) = self.button.upgrade() {
            button.set_tooltip_text(Some("Play recording"));
        }
    }

    fn set_progress(&self, _fraction: f64) {}
}

/// Window title — also used for single-instance detection.
const WINDOW_TITLE: &str = "talk-rs — Recordings";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Section {
    Cache,
    Output,
}

impl Section {
    fn label(self) -> &'static str {
        match self {
            Self::Cache => "Dictation cache",
            Self::Output => "Recordings",
        }
    }
}

enum WorkerRequest {
    Start,
    WatchesReady(PathBuf),
    Refresh(PathBuf),
    Sidecar(PathBuf),
    Subtree(PathBuf),
    RemovedSubtree(PathBuf),
}

enum WorkerResult {
    WatchPaths(Section, PathBuf, Vec<PathBuf>),
    Initial(Section, Vec<RecordingEntry>),
    Updated(Section, PathBuf, Option<RecordingEntry>),
}

/// The initial snapshot remains immutable; updates to unbuilt paths override it.
/// Materialized paths are looked up in constant time, never walked per new row.
struct SectionBatch {
    pending: VecDeque<RecordingEntry>,
    known: BTreeSet<PathBuf>,
    overrides: BTreeMap<PathBuf, Option<RecordingEntry>>,
    built: HashMap<PathBuf, gtk4::ListBoxRow>,
    order: Vec<PathBuf>,
    total: usize,
}

impl SectionBatch {
    fn new(entries: Vec<RecordingEntry>) -> Self {
        let total = entries.len();
        let known = entries.iter().map(|entry| entry.path.clone()).collect();
        Self {
            pending: entries.into(),
            known,
            overrides: BTreeMap::new(),
            built: HashMap::new(),
            order: Vec::new(),
            total,
        }
    }

    fn update(&mut self, path: PathBuf, entry: Option<RecordingEntry>) {
        let existed = self.known.contains(&path);
        match (existed, entry.is_some()) {
            (false, true) => self.total += 1,
            (true, false) => self.total -= 1,
            _ => {}
        }
        if entry.is_some() {
            self.known.insert(path.clone());
        } else {
            self.known.remove(&path);
        }
        self.overrides.insert(path, entry);
    }

    fn next(&mut self) -> Option<RecordingEntry> {
        while let Some(entry) = self.pending.pop_front() {
            let path = &entry.path;
            if self.built.contains_key(path) {
                continue;
            }
            match self.overrides.remove(path) {
                Some(Some(new)) => return Some(new),
                Some(None) => continue,
                None => return Some(entry),
            }
        }
        None
    }

    fn finished(&self) -> bool {
        self.pending.is_empty()
    }
}

fn spawn_listing_worker(
    section: Section,
    collection: RecordingCollection,
    results: mpsc::Sender<WorkerResult>,
) -> mpsc::Sender<WorkerRequest> {
    let (sender, inbox) = mpsc::channel();
    std::thread::spawn(move || {
        let root = collection.root().to_path_buf();
        let mut known = BTreeSet::<PathBuf>::new();
        let mut pending = BTreeSet::<PathBuf>::new();
        let mut sidecars = BTreeSet::<PathBuf>::new();
        let mut planned = BTreeMap::<PathBuf, BTreeSet<PathBuf>>::new();
        let mut initial_done = false;
        let mut ready = false;
        while let Ok(request) = inbox.recv() {
            let mut requests = vec![request];
            for _ in 0..63 {
                match inbox.try_recv() {
                    Ok(request) => requests.push(request),
                    Err(_) => break,
                }
            }
            for request in requests {
                match request {
                    WorkerRequest::Start | WorkerRequest::Subtree(_) => {
                        let dir = match request {
                            WorkerRequest::Subtree(dir) => dir,
                            _ => root.clone(),
                        };
                        let mut dirs = Vec::new();
                        if dir.is_dir() && (!dir.is_symlink() || dir == root) {
                            collect_directories_recursive(&dir, &mut dirs);
                        }
                        planned.insert(dir.clone(), dirs.iter().cloned().collect());
                        if results
                            .send(WorkerResult::WatchPaths(section, dir, dirs))
                            .is_err()
                        {
                            return;
                        }
                    }
                    WorkerRequest::WatchesReady(dir) => {
                        let mut latest = Vec::new();
                        if dir.is_dir() && (!dir.is_symlink() || dir == root) {
                            collect_directories_recursive(&dir, &mut latest);
                        }
                        let watched = planned.entry(dir.clone()).or_default();
                        let additional: Vec<_> = latest
                            .into_iter()
                            .filter(|path| watched.insert(path.clone()))
                            .collect();
                        if !additional.is_empty() {
                            if results
                                .send(WorkerResult::WatchPaths(section, dir, additional))
                                .is_err()
                            {
                                return;
                            }
                        } else if dir == root {
                            ready = true;
                        } else if let Ok(entries) = collection.list_under(&dir) {
                            pending.extend(entries.into_iter().map(|entry| entry.path));
                        }
                    }
                    WorkerRequest::Refresh(path) => {
                        pending.insert(path);
                    }
                    WorkerRequest::Sidecar(path) => {
                        sidecars.insert(path);
                    }
                    WorkerRequest::RemovedSubtree(dir) => {
                        pending.extend(known.iter().filter(|path| path.starts_with(&dir)).cloned());
                    }
                }
            }
            if !ready {
                continue;
            }
            if !initial_done {
                let entries = match collection.list() {
                    Ok(entries) => entries,
                    Err(err) => {
                        log::warn!("record-ui: failed to list {}: {err}", root.display());
                        Vec::new()
                    }
                };
                known.extend(entries.iter().map(|entry| entry.path.clone()));
                if results
                    .send(WorkerResult::Initial(section, entries))
                    .is_err()
                {
                    return;
                }
                initial_done = true;
            }
            for sidecar in std::mem::take(&mut sidecars) {
                pending.extend(collection.affected_audio(&sidecar, known.iter()));
            }
            for path in std::mem::take(&mut pending) {
                let entry = collection.entry(&path);
                if entry.is_some() {
                    known.insert(path.clone());
                } else {
                    known.remove(&path);
                }
                if results
                    .send(WorkerResult::Updated(section, path, entry))
                    .is_err()
                {
                    return;
                }
            }
        }
    });
    sender
}

fn collect_directories_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    out.push(dir.to_path_buf());

    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            log::warn!(
                "record-ui: watch: failed to read {}: {}",
                dir.display(),
                err,
            );
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                log::warn!(
                    "record-ui: watch: failed to inspect {}: {}",
                    dir.display(),
                    err,
                );
                continue;
            }
        };

        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(err) => {
                log::warn!(
                    "record-ui: watch: failed to inspect type under {}: {}",
                    dir.display(),
                    err,
                );
                continue;
            }
        };

        if file_type.is_dir() {
            collect_directories_recursive(&entry.path(), out);
        }
    }
}

/// Open the GTK4 recordings browser.
///
/// The window appears immediately with a loading indicator; recording
/// listings are populated asynchronously via an idle callback so the
/// user never stares at a blank wait.
pub async fn record_ui() -> Result<(), TalkError> {
    tokio::task::spawn_blocking(show_recordings_window)
        .await
        .map_err(|e| TalkError::Config(format!("GTK task failed: {}", e)))?
}

/// Build and run the GTK4 recordings browser window.
fn show_recordings_window() -> Result<(), TalkError> {
    use gtk4::glib;
    use gtk4::prelude::*;
    use std::cell::RefCell;

    let t0 = std::time::Instant::now();

    gtk4::init().map_err(|e| TalkError::Config(format!("failed to initialize GTK: {}", e)))?;
    log::debug!("record-ui: gtk4::init {:.0?}", t0.elapsed());

    let theme = crate::gtk_theme::ThemeColors::resolve();
    log::debug!("record-ui: theme resolve {:.0?}", t0.elapsed());

    let window = gtk4::Window::builder()
        .title(WINDOW_TITLE)
        .default_width(800)
        .default_height(500)
        .decorated(false)
        .resizable(true)
        .build();
    window.set_size_request(400, 250);

    crate::gtk_theme::load_css(&theme.base_css(
        ".transcript { font-family: monospace; opacity: 0.7; } \
         .meta { font-family: monospace; } \
         .waterfall { background-color: black; border-radius: 0.25em; } \
         .copy-btn, .play-btn, .dictate-btn, .folder-btn, .delete-btn { min-width: 28px; min-height: 28px; max-width: 28px; max-height: 28px; padding: 0; font-size: 14px; } \
         .dictate-btn label, .folder-btn label, .delete-btn label { padding-top: 4px; } \
         .section-expander { margin: 4px 2px; } \
         .section-expander > title { font-weight: bold; opacity: 0.85; padding: 4px 0; }",
    ));

    let root = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
    root.set_margin_top(6);
    root.set_margin_bottom(6);
    root.set_margin_start(6);
    root.set_margin_end(6);

    // ── Title bar with close button ──────────────────────────
    let (title_bar, close_btn) = crate::gtk_theme::build_title_bar();
    root.append(&title_bar);

    let scrolled = gtk4::ScrolledWindow::builder()
        .vexpand(true)
        .hexpand(true)
        .build();

    // WindowHandle for dragging
    let handle = gtk4::WindowHandle::new();
    handle.set_child(Some(&root));
    window.set_child(Some(&handle));

    let main_loop = glib::MainLoop::new(None, false);

    // Playback shared by every row: the native audio player (cpal,
    // installed in background after the window is presented to avoid
    // blocking the UI on device probing), the row that owns it and the
    // one progress tick.
    let session = PlaybackSession::new();

    // Container inside the scrolled window for both sections.
    let sections_box = gtk4::Box::new(gtk4::Orientation::Vertical, 0);

    // Loading indicator — shown until the idle callback populates data.
    let loading_label = gtk4::Label::new(Some("Loading recordings…"));
    loading_label.set_vexpand(true);
    loading_label.set_valign(gtk4::Align::Center);
    loading_label.add_css_class("dim");
    sections_box.append(&loading_label);

    scrolled.set_child(Some(&sections_box));
    root.append(&scrolled);

    // Monitors must survive the lifetime of the window; the idle
    // callback fills this with gio::FileMonitor instances.
    let monitors: Rc<RefCell<BTreeMap<(Section, PathBuf), gtk4::gio::FileMonitor>>> =
        Rc::new(RefCell::new(BTreeMap::new()));

    {
        /// Build a single row (hbox) for a recording entry with all columns
        /// and buttons.
        fn build_row(
            recording: &RecordingEntry,
            session: &Rc<PlaybackSession>,
            window: &gtk4::Window,
            list: &gtk4::ListBox,
            expander: &gtk4::Expander,
            section_label: &str,
        ) -> gtk4::ListBoxRow {
            use gtk4::prelude::*;

            let hbox = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
            hbox.set_margin_top(4);
            hbox.set_margin_bottom(4);
            hbox.set_margin_start(4);
            hbox.set_margin_end(4);

            // Date (fixed width)
            let date_label = gtk4::Label::new(Some(&recording.date_label));
            date_label.set_xalign(0.0);
            date_label.set_width_chars(19);
            date_label.set_max_width_chars(19);
            date_label.set_selectable(false);
            date_label.add_css_class("meta");
            hbox.append(&date_label);

            // Duration (fixed width)
            let dur_label = gtk4::Label::new(Some(&recording.duration_label));
            dur_label.set_xalign(1.0);
            dur_label.set_width_chars(8);
            dur_label.set_max_width_chars(8);
            dur_label.set_selectable(false);
            dur_label.add_css_class("dim");
            dur_label.add_css_class("meta");
            hbox.append(&dur_label);

            // Size (fixed width)
            let size_label = gtk4::Label::new(Some(&recording.size_label));
            size_label.set_xalign(1.0);
            size_label.set_width_chars(8);
            size_label.set_max_width_chars(8);
            size_label.set_selectable(false);
            size_label.add_css_class("dim");
            size_label.add_css_class("meta");
            hbox.append(&size_label);

            // Transcript preview, in-progress indicator, or audio
            // player bar — driven by the pick-file status.
            use crate::recording_cache::TranscriptStatus;
            let mut playback_row: Option<RowRef> = None;
            match &recording.status {
                TranscriptStatus::NotAvailable => {
                    // No pick yet — show the shared audio player bar
                    // with waterfall spectrogram, cursor, drag-to-seek,
                    // and play/pause/rewind controls.
                    let (player_bar, row) =
                        crate::widgets::audio_player_bar::build_audio_player_bar(
                            &recording.path,
                            session,
                            None, // waterfall queued on the bounded loader
                            28,
                        );
                    hbox.append(&player_bar);
                    playback_row = Some(row);
                }
                TranscriptStatus::InProgress => {
                    let label = gtk4::Label::new(None);
                    label.set_markup("<i>(transcription ongoing)</i>");
                    label.set_opacity(0.35);
                    label.set_xalign(0.0);
                    label.set_hexpand(true);
                    label.set_selectable(false);
                    label.add_css_class("transcript");
                    hbox.append(&label);
                }
                TranscriptStatus::Available(_) => {
                    let transcript = gtk4::Label::new(None);
                    if recording.transcript_preview.is_empty() {
                        // Empty pick text — valid result for silent
                        // recordings.  Record-UI display (not the
                        // picker's "(no speech detected)").
                        transcript.set_markup("<i>(no text)</i>");
                        transcript.set_opacity(0.35);
                    } else {
                        transcript.set_text(&recording.transcript_preview);
                    }
                    transcript.set_xalign(0.0);
                    transcript.set_hexpand(true);
                    transcript.set_ellipsize(gtk4::pango::EllipsizeMode::End);
                    transcript.set_max_width_chars(80);
                    transcript.set_selectable(false);
                    transcript.add_css_class("transcript");
                    hbox.append(&transcript);
                }
            }

            // Dictate button — open the picker to (re-)transcribe this recording
            let dictate_btn = gtk4::Button::with_label("\u{1D413}");
            dictate_btn.set_tooltip_text(Some("Transcribe recording"));
            dictate_btn.add_css_class("dictate-btn");
            {
                let audio_path = recording.path.clone();
                dictate_btn.connect_clicked(move |_| {
                    let exe = std::env::current_exe()
                        .unwrap_or_else(|_| std::path::PathBuf::from("talk-rs"));
                    log::debug!("dictate: launching picker for {}", audio_path.display());
                    if let Err(e) = std::process::Command::new(exe)
                        .args([
                            "dictate",
                            "--pick",
                            "--input-audio-file",
                            &audio_path.to_string_lossy(),
                        ])
                        .spawn()
                    {
                        log::warn!("failed to launch picker: {}", e);
                    }
                });
            }
            hbox.append(&dictate_btn);

            // Play button (simple toggle for entries that already
            // have a transcript row; entries without a transcript
            // row use the full audio_player_bar above).
            if !matches!(
                recording.status,
                crate::recording_cache::TranscriptStatus::NotAvailable
            ) {
                let play_btn = gtk4::Button::new();
                play_btn.add_css_class("play-btn");
                let simple = Rc::new(SimplePlayRow::new(&play_btn));
                let row_ref: RowRef = simple.clone();
                if session.adopt(Rc::clone(&row_ref), &recording.path) {
                    simple.show_playing();
                }
                playback_row = Some(Rc::clone(&row_ref));
                {
                    let audio_path = recording.path.clone();
                    let session = Rc::clone(session);
                    let row: RowRef = Rc::clone(&simple) as RowRef;
                    play_btn.connect_clicked(move |_| {
                        if session.is_active(&row) {
                            session.stop();
                        } else if session.play(Rc::clone(&row), &audio_path, None) {
                            simple.show_playing();
                        }
                    });
                }
                hbox.append(&play_btn);
            }

            // Copy-to-clipboard button (only shown when transcript text exists).
            //
            // The clipboard must receive the FULL transcript, not the
            // display preview (which is truncated to ~200 chars with
            // an ellipsis for the GTK label).
            if !recording.transcript_full.is_empty() {
                let copy_btn = gtk4::Button::with_label("\u{29C9}");
                copy_btn.set_tooltip_text(Some("Copy transcript to clipboard"));
                copy_btn.add_css_class("copy-btn");
                {
                    let text = recording.transcript_full.clone();
                    copy_btn.connect_clicked(move |_| {
                        if let Some(display) = gtk4::gdk::Display::default() {
                            display.clipboard().set_text(&text);
                        }
                    });
                }
                hbox.append(&copy_btn);
            }

            // Folder button — open file manager with file highlighted
            let folder_btn = gtk4::Button::with_label("🖿\u{FE0E}");
            folder_btn.set_tooltip_text(Some("Show in file manager"));
            folder_btn.add_css_class("folder-btn");
            {
                let audio_path = recording.path.clone();
                let win_ref = window.clone();
                folder_btn.connect_clicked(move |_| {
                    open_in_file_manager(&audio_path, &win_ref);
                });
            }
            hbox.append(&folder_btn);

            // Delete button
            let delete_btn = gtk4::Button::with_label("🗑");
            delete_btn.set_tooltip_text(Some("Delete recording"));
            delete_btn.add_css_class("delete-btn");
            {
                let audio_path = recording.path.clone();
                let list_ref = list.clone();
                let expander_ref = expander.clone();
                let slabel = section_label.to_string();
                delete_btn.connect_clicked(move |btn| {
                    if let Err(e) = delete_recording(&audio_path) {
                        log::warn!("delete failed: {}", e);
                        return;
                    }

                    // Walk up widget tree to find the ListBoxRow
                    let mut widget: Option<gtk4::Widget> = btn.parent();
                    let row_to_remove: Option<gtk4::ListBoxRow> = loop {
                        match widget {
                            Some(ref w) => {
                                if let Some(row) = w.downcast_ref::<gtk4::ListBoxRow>() {
                                    break Some(row.clone());
                                }
                                widget = w.parent();
                            }
                            None => break None,
                        }
                    };
                    let Some(row) = row_to_remove else {
                        return;
                    };

                    // Select an adjacent row before removal so the
                    // scroll position stays stable.
                    let idx = row.index();
                    let next = list_ref.row_at_index(idx + 1).or_else(|| {
                        if idx > 0 {
                            list_ref.row_at_index(idx - 1)
                        } else {
                            None
                        }
                    });
                    if let Some(ref adjacent) = next {
                        list_ref.select_row(Some(adjacent));
                    }
                    list_ref.remove(&row);

                    // Count remaining rows and update expander title
                    let mut count = 0;
                    let mut child = list_ref.first_child();
                    while let Some(w) = child {
                        if w.downcast_ref::<gtk4::ListBoxRow>().is_some() {
                            count += 1;
                        }
                        child = w.next_sibling();
                    }
                    expander_ref.set_label(Some(&format!("{} ({})", slabel, count)));
                });
            }
            hbox.append(&delete_btn);

            let row = gtk4::ListBoxRow::new();
            row.set_child(Some(&hbox));
            if let Some(playback_row) = playback_row {
                let session = Rc::clone(session);
                row.connect_parent_notify(move |widget| {
                    if widget.parent().is_none() {
                        session.release_if_active(&playback_row);
                    }
                });
            }
            // Tag with audio path so FileMonitor can find rows by path.
            row.set_widget_name(&recording.path.to_string_lossy());
            row
        }

        /// Populate a ListBox with recording entries, updating the Expander
        /// title with the count.  Clears any existing rows first.
        ///
        /// Rows are built in batches of `BATCH_SIZE` via
        /// `glib::idle_add_local_once` so the GTK main loop stays
        /// responsive between batches.
        fn populate_section(
            label: &'static str,
            recordings: Vec<RecordingEntry>,
            list: &gtk4::ListBox,
            expander: &gtk4::Expander,
            session: &Rc<PlaybackSession>,
            window: &gtk4::Window,
            state: &Rc<RefCell<Option<SectionBatch>>>,
        ) {
            use gtk4::prelude::*;
            const BATCH_SIZE: usize = 20;
            const BUDGET: std::time::Duration = std::time::Duration::from_millis(5);
            *state.borrow_mut() = Some(SectionBatch::new(recordings));
            let total = state.borrow().as_ref().map_or(0, |s| s.total);
            expander.set_label(Some(&format!("{label} ({total})")));
            let state = Rc::clone(state);
            let list = list.clone();
            let expander = expander.clone();
            let session = Rc::clone(session);
            let window = window.clone();
            glib::idle_add_local(move || {
                if !window.is_visible() {
                    return glib::ControlFlow::Break;
                }
                let started = std::time::Instant::now();
                for _ in 0..BATCH_SIZE {
                    let entry = state.borrow_mut().as_mut().and_then(SectionBatch::next);
                    let Some(entry) = entry else { break };
                    let row = build_row(&entry, &session, &window, &list, &expander, label);
                    list.append(&row);
                    if let Some(batch) = state.borrow_mut().as_mut() {
                        batch.order.push(entry.path.clone());
                        batch.built.insert(entry.path, row);
                    }
                    if started.elapsed() >= BUDGET {
                        break;
                    }
                }
                if state.borrow().as_ref().is_some_and(SectionBatch::finished) {
                    let extra = state.borrow().as_ref().and_then(|batch| {
                        batch
                            .overrides
                            .iter()
                            .find(|(path, entry)| {
                                entry.is_some() && !batch.built.contains_key(*path)
                            })
                            .map(|(path, entry)| (path.clone(), entry.clone()))
                    });
                    if let Some((path, entry)) = extra {
                        apply_update(
                            (path, entry),
                            &list,
                            &expander,
                            label,
                            &session,
                            &window,
                            &state,
                        );
                    }
                }
                if list.selected_row().is_none() {
                    if let Some(row) = list.row_at_index(0) {
                        list.select_row(Some(&row));
                    }
                }
                let done = state.borrow().as_ref().is_none_or(|batch| {
                    batch.finished()
                        && !batch
                            .overrides
                            .iter()
                            .any(|(path, entry)| entry.is_some() && !batch.built.contains_key(path))
                });
                if done {
                    crate::perf_counters::mark_with(|| {
                        format!("record_ui_rows_built_{}", label.replace(' ', "_"))
                    });
                    glib::ControlFlow::Break
                } else {
                    glib::ControlFlow::Continue
                }
            });
        }

        fn apply_update(
            (path, entry): (PathBuf, Option<RecordingEntry>),
            list: &gtk4::ListBox,
            expander: &gtk4::Expander,
            label: &'static str,
            session: &Rc<PlaybackSession>,
            window: &gtk4::Window,
            state: &Rc<RefCell<Option<SectionBatch>>>,
        ) {
            use gtk4::prelude::*;
            let mut state_ref = state.borrow_mut();
            let Some(batch) = state_ref.as_mut() else {
                return;
            };
            batch.update(path.clone(), entry.clone());
            if let Some(old) = batch.built.remove(&path) {
                batch.overrides.remove(&path);
                let selected = old.is_selected();
                let index = old.index();
                if let Some(entry) = entry {
                    let row = build_row(&entry, session, window, list, expander, label);
                    list.insert(&row, index);
                    if selected {
                        list.select_row(Some(&row));
                    }
                    batch.built.insert(path.clone(), row);
                } else {
                    batch.order.retain(|item| item != &path);
                    if selected {
                        let adjacent = list
                            .row_at_index(index + 1)
                            .or_else(|| list.row_at_index(index - 1));
                        list.select_row(adjacent.as_ref());
                    }
                }
                list.remove(&old);
            } else if batch.finished() {
                if let Some(entry) = entry {
                    let pos = batch.order.partition_point(|other| {
                        compare_entries_newest_first(other, &path).is_lt()
                    });
                    let row = build_row(&entry, session, window, list, expander, label);
                    list.insert(&row, pos as i32);
                    batch.order.insert(pos, path.clone());
                    batch.built.insert(path.clone(), row);
                }
                batch.overrides.remove(&path);
            }
            expander.set_label(Some(&format!("{label} ({})", batch.total)));
        }

        /// Create an Expander + ListBox pair for a section.
        fn create_section(label: &str) -> (gtk4::Expander, gtk4::ListBox) {
            use gtk4::prelude::*;

            let expander = gtk4::Expander::new(Some(label));
            expander.set_expanded(true);
            expander.add_css_class("section-expander");

            let list = gtk4::ListBox::new();
            list.set_selection_mode(gtk4::SelectionMode::Single);
            list.set_activate_on_single_click(false);

            expander.set_child(Some(&list));
            (expander, list)
        }

        // ── Deferred data loading ───────────────────────────────
        // Populate recordings and set up file watches AFTER the
        // window is painted so the user sees the loading label.
        //
        // We hook the window's `map` signal and schedule a short
        // timeout from there — this ensures at least one frame is
        // drawn (showing "Loading recordings…") before the data
        // loading grabs the GTK thread.
        let session_idle = Rc::clone(&session);
        let win_idle = window.clone();
        let sections_idle = sections_box.clone();
        let loading_idle = loading_label.clone();
        let monitors_idle = Rc::clone(&monitors);
        let loaded = std::cell::Cell::new(false);

        fn install_monitor(
            section: Section,
            dir: &Path,
            sender: &mpsc::Sender<WorkerRequest>,
            monitors: &Rc<RefCell<BTreeMap<(Section, PathBuf), gtk4::gio::FileMonitor>>>,
        ) {
            if !dir.is_dir()
                || monitors
                    .borrow()
                    .contains_key(&(section, dir.to_path_buf()))
            {
                return;
            }
            let file = gtk4::gio::File::for_path(dir);
            let monitor = match file.monitor_directory(
                gtk4::gio::FileMonitorFlags::NONE,
                gtk4::gio::Cancellable::NONE,
            ) {
                Ok(monitor) => monitor,
                Err(err) => {
                    log::warn!("record-ui: cannot watch {}: {err}", dir.display());
                    return;
                }
            };
            let sender = sender.clone();
            let weak_monitors = Rc::downgrade(monitors);
            monitor.connect_changed(move |_, file, _, event| {
                use gtk4::gio::FileMonitorEvent;
                let Some(path) = file.path() else { return };
                if !matches!(
                    event,
                    FileMonitorEvent::Created
                        | FileMonitorEvent::Deleted
                        | FileMonitorEvent::ChangesDoneHint
                ) {
                    return;
                }
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name.ends_with(".wf") || name.ends_with(".wf.tmp") {
                    return;
                }
                let was_dir = weak_monitors.upgrade().is_some_and(|monitors| {
                    monitors.borrow().contains_key(&(section, path.clone()))
                });
                if event == FileMonitorEvent::Created && path.is_dir() && !path.is_symlink() {
                    let _ = sender.send(WorkerRequest::Subtree(path));
                } else if event == FileMonitorEvent::Deleted
                    && (was_dir || (!has_audio_extension(&path) && !name.ends_with(".yml")))
                {
                    if let Some(monitors) = weak_monitors.upgrade() {
                        monitors.borrow_mut().retain(|(owner, watched), monitor| {
                            let keep = *owner != section || !watched.starts_with(&path);
                            if !keep {
                                monitor.cancel();
                            }
                            keep
                        });
                    }
                    let _ = sender.send(WorkerRequest::RemovedSubtree(path));
                } else if name.ends_with(".yml") {
                    let _ = sender.send(WorkerRequest::Sidecar(path));
                } else if has_audio_extension(&path) {
                    let _ = sender.send(WorkerRequest::Refresh(path));
                }
            });
            monitors
                .borrow_mut()
                .insert((section, dir.to_path_buf()), monitor);
        }

        window.connect_map(move |_| {
            if loaded.replace(true) {
                return;
            }
            let session = Rc::clone(&session_idle);
            let win = win_idle.clone();
            let sections = sections_idle.clone();
            let loading = loading_idle.clone();
            let monitors = Rc::clone(&monitors_idle);
            glib::timeout_add_local_once(std::time::Duration::from_millis(16), move || {
                sections.remove(&loading);
                let (cache_expander, cache_list) = create_section("Dictation cache (0)");
                sections.append(&cache_expander);
                let (output_expander, output_list) = create_section("Recordings (0)");
                sections.append(&output_expander);
                #[cfg(feature = "perf-counters")]
                super::ui_probe::install(vec![
                    ("Dictation cache", cache_list.clone()),
                    ("Recordings", output_list.clone()),
                ]);
                let config = match Config::load(None) {
                    Ok(config) => std::sync::Arc::new(config),
                    Err(err) => {
                        log::warn!("record-ui: cannot load configuration: {err}");
                        return;
                    }
                };
                let (result_tx, result_rx) = mpsc::channel();
                let cache =
                    match RecordingCollection::dictation_cache(std::sync::Arc::clone(&config)) {
                        Ok(collection) => Some(collection),
                        Err(err) => {
                            log::warn!("record-ui: cache unavailable: {err}");
                            None
                        }
                    };
                let output = RecordingCollection::output_recordings(config);
                let cache_sender = cache.map(|collection| {
                    let root = collection.root().to_path_buf();
                    let sender =
                        spawn_listing_worker(Section::Cache, collection, result_tx.clone());
                    install_monitor(Section::Cache, &root, &sender, &monitors);
                    let _ = sender.send(WorkerRequest::Start);
                    sender
                });
                let output_root = output.root().to_path_buf();
                let output_sender =
                    spawn_listing_worker(Section::Output, output, result_tx.clone());
                install_monitor(Section::Output, &output_root, &output_sender, &monitors);
                let _ = output_sender.send(WorkerRequest::Start);
                drop(result_tx);
                let cache_state: Rc<RefCell<Option<SectionBatch>>> = Rc::new(RefCell::new(None));
                let output_state: Rc<RefCell<Option<SectionBatch>>> = Rc::new(RefCell::new(None));
                let mut pending_watches: VecDeque<(Section, PathBuf, VecDeque<PathBuf>)> =
                    VecDeque::new();
                glib::timeout_add_local(std::time::Duration::from_millis(50), move || {
                    if !win.is_visible() {
                        return glib::ControlFlow::Break;
                    }
                    let start = std::time::Instant::now();
                    for _ in 0..5 {
                        if start.elapsed() >= std::time::Duration::from_millis(5) {
                            break;
                        }
                        let result = match result_rx.try_recv() {
                            Ok(result) => result,
                            Err(mpsc::TryRecvError::Empty) => break,
                            Err(mpsc::TryRecvError::Disconnected) => {
                                return glib::ControlFlow::Break
                            }
                        };
                        match result {
                            WorkerResult::WatchPaths(section, root, dirs) => {
                                pending_watches.push_back((section, root, dirs.into()))
                            }
                            WorkerResult::Initial(section, entries) => {
                                let (list, expander, state) = match section {
                                    Section::Cache => (&cache_list, &cache_expander, &cache_state),
                                    Section::Output => {
                                        (&output_list, &output_expander, &output_state)
                                    }
                                };
                                populate_section(
                                    section.label(),
                                    entries,
                                    list,
                                    expander,
                                    &session,
                                    &win,
                                    state,
                                );
                            }
                            WorkerResult::Updated(section, path, entry) => {
                                let (list, expander, state) = match section {
                                    Section::Cache => (&cache_list, &cache_expander, &cache_state),
                                    Section::Output => {
                                        (&output_list, &output_expander, &output_state)
                                    }
                                };
                                apply_update(
                                    (path, entry),
                                    list,
                                    expander,
                                    section.label(),
                                    &session,
                                    &win,
                                    state,
                                );
                            }
                        }
                    }
                    for _ in 0..5 {
                        if start.elapsed() >= std::time::Duration::from_millis(5) {
                            break;
                        }
                        let Some((section, root, dirs)) = pending_watches.front_mut() else {
                            break;
                        };
                        let sender = match section {
                            Section::Cache => cache_sender.as_ref(),
                            Section::Output => Some(&output_sender),
                        };
                        if let Some(dir) = dirs.pop_front() {
                            if let Some(sender) = sender {
                                install_monitor(*section, &dir, sender, &monitors);
                            }
                        }
                        if dirs.is_empty() {
                            if let Some(sender) = sender {
                                let _ = sender.send(WorkerRequest::WatchesReady(root.clone()));
                            }
                            pending_watches.pop_front();
                        }
                    }
                    glib::ControlFlow::Continue
                });
            });
        });
    }

    // Escape to close
    {
        let ml = main_loop.clone();
        let win = window.clone();
        let session_ref = Rc::clone(&session);
        let key_ctl = gtk4::EventControllerKey::new();
        key_ctl.connect_key_pressed(move |_, key, _, _| {
            if key == gtk4::gdk::Key::Escape {
                session_ref.stop();
                win.set_visible(false);
                ml.quit();
                glib::Propagation::Stop
            } else {
                glib::Propagation::Proceed
            }
        });
        window.add_controller(key_ctl);
    }

    // Close button (same as Escape)
    {
        let ml = main_loop.clone();
        let win = window.clone();
        let session_ref = Rc::clone(&session);
        close_btn.connect_clicked(move |_| {
            session_ref.stop();
            win.set_visible(false);
            ml.quit();
        });
    }

    // Window close
    {
        let ml = main_loop.clone();
        let session_ref = Rc::clone(&session);
        window.connect_close_request(move |win| {
            session_ref.stop();
            win.set_visible(false);
            ml.quit();
            glib::Propagation::Proceed
        });
    }

    crate::gtk_theme::install_edge_resize(&window);

    log::debug!("record-ui: window built {:.0?}", t0.elapsed());
    crate::gtk_theme::present_centred(&window);
    log::debug!("record-ui: window presented {:.0?}", t0.elapsed());

    // Initialize the audio player after the window is presented so it
    // appears instantly instead of blocking on cpal device probing.
    {
        let session_init = Rc::clone(&session);
        glib::idle_add_local_once(move || match WavPlayer::new() {
            Ok(p) => session_init.set_player(p),
            Err(e) => log::warn!("audio output unavailable, play disabled: {}", e),
        });
    }

    crate::perf_counters::install_gtk_stall_probe();
    main_loop.run();
    window.close();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::audio::{ogg_duration_secs, wav_duration_secs};
    use super::super::entries::{format_duration, format_size};
    use super::{
        collect_directories_recursive, spawn_listing_worker, Section, SectionBatch, WorkerRequest,
        WorkerResult,
    };

    #[test]
    fn pending_snapshot_obeys_latest_pick_and_deletion_without_row_walk() {
        let a = std::path::PathBuf::from("/tmp/a.ogg");
        let b = std::path::PathBuf::from("/tmp/b.ogg");
        let entry = |path: &std::path::Path, text: &str| super::RecordingEntry {
            path: path.to_path_buf(),
            date_label: String::new(),
            duration_label: String::new(),
            size_label: String::new(),
            transcript_full: text.into(),
            transcript_preview: text.into(),
            status: crate::recording_cache::TranscriptStatus::Available(text.into()),
        };
        let mut batch = SectionBatch::new(vec![entry(&a, "old"), entry(&b, "old")]);
        batch.update(a.clone(), None);
        batch.update(b.clone(), Some(entry(&b, "latest")));
        assert_eq!(batch.total, 1);
        assert_eq!(
            batch.next().map(|e| e.transcript_full),
            Some("latest".into())
        );
        assert!(batch.finished());
        assert!(batch.built.is_empty());
    }

    #[test]
    fn worker_waits_for_watches_and_publishes_event_after_snapshot() {
        use std::sync::{mpsc, Arc};
        use std::time::Duration;
        let temp = tempfile::tempdir().expect("tempdir");
        let audio = temp.path().join("voice.ogg");
        std::fs::write(&audio, b"audio").expect("audio");
        let config = serde_yaml::from_str(&format!(
            "output_dir: {}\nproviders:\n  mistral:\n    api_key: fake\n",
            temp.path().display()
        ))
        .expect("config");
        let collection = super::RecordingCollection::output_recordings(Arc::new(config));
        let (tx, rx) = mpsc::channel();
        let worker = spawn_listing_worker(Section::Output, collection, tx);
        worker.send(WorkerRequest::Start).expect("start");
        let WorkerResult::WatchPaths(_, root, _) =
            rx.recv_timeout(Duration::from_secs(5)).expect("watch plan")
        else {
            panic!("watch plan first")
        };
        worker
            .send(WorkerRequest::Refresh(audio.clone()))
            .expect("queued refresh");
        assert!(
            rx.try_recv().is_err(),
            "no scan before watch acknowledgment"
        );
        worker
            .send(WorkerRequest::WatchesReady(root))
            .expect("ready");
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(5)).expect("initial"),
            WorkerResult::Initial(Section::Output, _)
        ));
        assert!(
            matches!(rx.recv_timeout(Duration::from_secs(5)).expect("update"), WorkerResult::Updated(Section::Output, path, Some(_)) if path == audio)
        );
    }

    #[test]
    fn directory_created_during_watch_installation_is_discovered_before_scan() {
        use std::sync::{mpsc, Arc};
        use std::time::Duration;
        let temp = tempfile::tempdir().expect("tempdir");
        let config = serde_yaml::from_str(&format!(
            "output_dir: {}\nproviders:\n  mistral:\n    api_key: fake\n",
            temp.path().display()
        ))
        .expect("config");
        let collection = super::RecordingCollection::output_recordings(Arc::new(config));
        let (tx, rx) = mpsc::channel();
        let worker = spawn_listing_worker(Section::Output, collection, tx);
        worker.send(WorkerRequest::Start).expect("start");
        let WorkerResult::WatchPaths(_, root, _) =
            rx.recv_timeout(Duration::from_secs(5)).expect("watch plan")
        else {
            panic!("watch plan first")
        };
        let new_dir = root.join("new/month");
        std::fs::create_dir_all(&new_dir).expect("new subtree");
        worker
            .send(WorkerRequest::WatchesReady(root.clone()))
            .expect("first acknowledgment");
        let WorkerResult::WatchPaths(_, recheck_root, additional) = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("new watch plan")
        else {
            panic!("new watches must precede scan")
        };
        assert_eq!(recheck_root, root);
        assert!(additional.contains(&new_dir));
        worker
            .send(WorkerRequest::WatchesReady(root))
            .expect("second acknowledgment");
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(5)).expect("initial"),
            WorkerResult::Initial(Section::Output, _)
        ));
    }

    #[test]
    fn missing_initial_directory_still_allows_later_entry_refresh() {
        use std::sync::{mpsc, Arc};
        use std::time::Duration;
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("missing");
        let config = serde_yaml::from_str(&format!(
            "output_dir: {}\nproviders:\n  mistral:\n    api_key: fake\n",
            root.display()
        ))
        .expect("config");
        let collection = super::RecordingCollection::output_recordings(Arc::new(config));
        let (tx, rx) = mpsc::channel();
        let worker = spawn_listing_worker(Section::Output, collection, tx);
        worker.send(WorkerRequest::Start).expect("start");
        let WorkerResult::WatchPaths(_, watch_root, _) =
            rx.recv_timeout(Duration::from_secs(5)).expect("watch plan")
        else {
            panic!("watch plan first")
        };
        worker
            .send(WorkerRequest::WatchesReady(watch_root))
            .expect("ready");
        assert!(
            matches!(rx.recv_timeout(Duration::from_secs(5)).expect("empty initial"), WorkerResult::Initial(Section::Output, rows) if rows.is_empty())
        );
        std::fs::create_dir_all(&root).expect("root appeared");
        let audio = root.join("voice.ogg");
        std::fs::write(&audio, b"audio").expect("recording");
        worker
            .send(WorkerRequest::Refresh(audio.clone()))
            .expect("refresh");
        assert!(
            matches!(rx.recv_timeout(Duration::from_secs(5)).expect("recovered"), WorkerResult::Updated(Section::Output, path, Some(_)) if path == audio)
        );
    }

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(5.0), "0:05");
        assert_eq!(format_duration(59.0), "0:59");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(60.0), "1:00");
        assert_eq!(format_duration(125.0), "2:05");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(3661.0), "1:01:01");
        assert_eq!(format_duration(7200.0), "2:00:00");
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(0), "0 B");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1_000), "1 KB");
        assert_eq!(format_size(999_999), "999 KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1_000_000), "1.0 MB");
        assert_eq!(format_size(15_500_000), "15.5 MB");
    }

    #[test]
    fn test_collect_directories_recursive_includes_nested_subdirs() {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        std::fs::create_dir_all(dir.path().join("2026/04")).expect("create nested dirs");
        std::fs::create_dir_all(dir.path().join("2025/12")).expect("create sibling dirs");
        std::fs::write(dir.path().join("2026/04/clip.ogg"), b"ogg").expect("write sample file");

        let mut dirs = Vec::new();
        collect_directories_recursive(dir.path(), &mut dirs);
        dirs.sort();

        let mut expected = vec![
            dir.path().to_path_buf(),
            dir.path().join("2025"),
            dir.path().join("2025/12"),
            dir.path().join("2026"),
            dir.path().join("2026/04"),
        ];
        expected.sort();

        assert_eq!(dirs, expected);
    }

    #[test]
    fn test_ogg_duration_secs_valid_file() {
        use crate::audio::writer::{AudioWriter, OggOpusWriter};
        use crate::config::AudioConfig;

        let dir = tempfile::TempDir::new().expect("create temp dir");
        let ogg_path = dir.path().join("test.ogg");

        // Build a small OGG Opus file: ~1 second of a 440 Hz sine at 16 kHz.
        let mut writer = OggOpusWriter::new(AudioConfig::new()).expect("create writer");
        let header = writer.header().expect("header");
        // 16 000 samples = 1 second at 16 kHz mono.
        let pcm: Vec<i16> = (0..16_000)
            .map(|i| {
                ((i as f32 / 16_000.0 * 440.0 * std::f32::consts::TAU).sin() * 16_000.0) as i16
            })
            .collect();
        let audio = writer.write_pcm(&pcm).expect("write pcm");
        let tail = writer.finalize().expect("finalize");
        std::fs::write(&ogg_path, [header, audio, tail].concat()).expect("write file");

        let duration = ogg_duration_secs(&ogg_path).expect("should compute duration");
        // Opus encodes at 48 kHz internally.  The 16 kHz input is
        // resampled, so the granule position reflects 48 kHz ticks.
        // Allow generous tolerance for codec frame rounding.
        assert!(
            duration > 0.5 && duration < 2.0,
            "expected ~1 s, got {:.3} s",
            duration,
        );
    }

    #[test]
    fn test_ogg_duration_secs_too_small() {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let ogg_path = dir.path().join("tiny.ogg");
        std::fs::write(&ogg_path, b"too small").expect("write");
        assert!(ogg_duration_secs(&ogg_path).is_none());
    }

    #[test]
    fn test_ogg_duration_secs_not_ogg() {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let path = dir.path().join("not.ogg");
        // Write 1 KB of zeros — no OggS magic anywhere.
        std::fs::write(&path, vec![0u8; 1024]).expect("write");
        assert!(ogg_duration_secs(&path).is_none());
    }

    #[test]
    fn test_wav_duration_secs_too_small() {
        // File smaller than header
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let wav = dir.path().join("tiny.wav");
        std::fs::write(&wav, b"small").expect("write");
        assert!(wav_duration_secs(&wav).is_none());
    }

    #[test]
    fn test_wav_duration_secs_valid() {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let wav = dir.path().join("test.wav");
        // 44 byte header + 32000 bytes of data = 1 second at 16kHz mono 16-bit
        let data = vec![0u8; 44 + 32_000];
        std::fs::write(&wav, &data).expect("write");
        let duration = wav_duration_secs(&wav).expect("should compute duration");
        assert!((duration - 1.0).abs() < 0.001);
    }
}
