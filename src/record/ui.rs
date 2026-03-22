//! GTK4 recordings browser for the `record --ui` command.
//!
//! Opens a window listing all cached recordings with metadata
//! (date, duration, size, transcript preview) and controls to
//! play or delete each entry.  Recordings are split into two
//! collapsible sections: OGG recordings and WAV dictation cache.

use super::entries::{
    delete_recording, list_ogg_recordings, list_wav_recordings, open_in_file_manager,
    RecordingEntry,
};
use super::player::WavPlayer;
use crate::config::Config;
use crate::error::TalkError;
use crate::recording_cache;

/// Window title — also used for single-instance detection.
const WINDOW_TITLE: &str = "talk-rs — Recordings";

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
    use std::rc::Rc;

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

    // Native audio player (cpal), initialized in background after the
    // window is presented to avoid blocking the UI on device probing.
    let player: Rc<RefCell<Option<WavPlayer>>> = Rc::new(RefCell::new(None));
    // Track which button is currently in "stop" mode.
    let active_play_btn: Rc<RefCell<Option<gtk4::Button>>> = Rc::new(RefCell::new(None));

    /// Stop any active playback and reset the corresponding button.
    fn stop_playback(
        player: &Rc<RefCell<Option<WavPlayer>>>,
        btn_ref: &Rc<RefCell<Option<gtk4::Button>>>,
    ) {
        if let Some(ref p) = *player.borrow() {
            p.stop();
        }
        if let Some(prev_btn) = btn_ref.borrow_mut().take() {
            prev_btn.set_label("▶");
            prev_btn.set_tooltip_text(Some("Play recording"));
        }
    }

    /// Start playback of an audio file, updating button state.
    fn start_playback(
        audio_path: &std::path::Path,
        btn: &gtk4::Button,
        player: &Rc<RefCell<Option<WavPlayer>>>,
        btn_ref: &Rc<RefCell<Option<gtk4::Button>>>,
    ) {
        stop_playback(player, btn_ref);

        let player_guard = player.borrow();
        let Some(ref p) = *player_guard else { return };
        if let Err(e) = p.play(audio_path) {
            log::warn!("failed to play {}: {}", audio_path.display(), e);
            return;
        }

        btn.set_label("■");
        btn.set_tooltip_text(Some("Stop playback"));
        *btn_ref.borrow_mut() = Some(btn.clone());

        // Poll for playback completion to reset button
        let player_poll = Rc::clone(player);
        let btn_poll = Rc::clone(btn_ref);
        let btn_widget = btn.clone();
        glib::timeout_add_local(std::time::Duration::from_millis(200), move || {
            let finished = player_poll
                .borrow()
                .as_ref()
                .is_none_or(|p| p.is_finished());
            if finished {
                btn_widget.set_label("▶");
                btn_widget.set_tooltip_text(Some("Play recording"));
                let is_active = btn_poll.borrow().as_ref().is_some_and(|b| *b == btn_widget);
                if is_active {
                    *btn_poll.borrow_mut() = None;
                }
                glib::ControlFlow::Break
            } else {
                glib::ControlFlow::Continue
            }
        });
    }

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
    let monitors: Rc<RefCell<Vec<gtk4::gio::FileMonitor>>> = Rc::new(RefCell::new(Vec::new()));

    {
        /// Build a single row (hbox) for a recording entry with all columns
        /// and buttons.
        fn build_row(
            recording: &RecordingEntry,
            player: &Rc<RefCell<Option<WavPlayer>>>,
            active_play_btn: &Rc<RefCell<Option<gtk4::Button>>>,
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

            // Transcript preview or interactive player bar
            if recording.transcript_preview.is_empty() {
                // No transcript — show the shared audio player bar
                // with waterfall spectrogram, cursor, drag-to-seek,
                // and play/pause/rewind controls.
                let player_bar = crate::widgets::audio_player_bar::build_audio_player_bar(
                    &recording.path,
                    player,
                    active_play_btn,
                    None, // waterfall computed in background by the widget
                    28,
                );
                hbox.append(&player_bar);
            } else {
                let transcript = gtk4::Label::new(Some(&recording.transcript_preview));
                transcript.set_xalign(0.0);
                transcript.set_hexpand(true);
                transcript.set_ellipsize(gtk4::pango::EllipsizeMode::End);
                transcript.set_max_width_chars(80);
                transcript.set_selectable(false);
                transcript.add_css_class("transcript");
                hbox.append(&transcript);
            }

            // Dictate button — open the picker to transcribe this recording
            if recording.transcript_preview.is_empty() {
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
            }

            // Play button (simple toggle for entries with transcripts;
            // entries without transcripts use the full audio_player_bar).
            if !recording.transcript_preview.is_empty() {
                let play_btn = gtk4::Button::with_label("▶");
                play_btn.set_tooltip_text(Some("Play recording"));
                play_btn.add_css_class("play-btn");
                {
                    let audio_path = recording.path.clone();
                    let player_ref = Rc::clone(player);
                    let active_btn_ref = Rc::clone(active_play_btn);
                    play_btn.connect_clicked(move |btn| {
                        let is_playing = active_btn_ref.borrow().as_ref().is_some_and(|b| b == btn);
                        if is_playing {
                            stop_playback(&player_ref, &active_btn_ref);
                        } else {
                            start_playback(&audio_path, btn, &player_ref, &active_btn_ref);
                        }
                    });
                }
                hbox.append(&play_btn);
            }

            // Copy-to-clipboard button (only shown when transcript text exists)
            if !recording.transcript_preview.is_empty() {
                let copy_btn = gtk4::Button::with_label("\u{29C9}");
                copy_btn.set_tooltip_text(Some("Copy transcript to clipboard"));
                copy_btn.add_css_class("copy-btn");
                {
                    let text = recording.transcript_preview.clone();
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
        #[allow(clippy::too_many_arguments)]
        fn populate_section(
            label: &str,
            recordings: Vec<RecordingEntry>,
            list: &gtk4::ListBox,
            expander: &gtk4::Expander,
            player: &Rc<RefCell<Option<WavPlayer>>>,
            active_play_btn: &Rc<RefCell<Option<gtk4::Button>>>,
            window: &gtk4::Window,
        ) {
            use gtk4::prelude::*;

            const BATCH_SIZE: usize = 20;

            // Remove existing rows.
            while let Some(child) = list.first_child() {
                list.remove(&child);
            }

            let total = recordings.len();
            expander.set_label(Some(&format!("{} ({})", label, total)));

            // Shared state for incremental batch building.
            let entries = Rc::new(RefCell::new(recordings));
            let offset = Rc::new(std::cell::Cell::new(0usize));

            let list = list.clone();
            let expander = expander.clone();
            let player = Rc::clone(player);
            let btn = Rc::clone(active_play_btn);
            let win = window.clone();
            let label = label.to_string();

            // Schedule first batch; each batch schedules the next
            // until all entries are built.
            let build_batch = Rc::new(RefCell::new(None::<Box<dyn Fn()>>));
            let build_batch_ref = Rc::clone(&build_batch);

            *build_batch.borrow_mut() = Some(Box::new(move || {
                let start = offset.get();
                let entries = entries.borrow();
                let end = (start + BATCH_SIZE).min(entries.len());

                for recording in &entries[start..end] {
                    let row = build_row(recording, &player, &btn, &win, &list, &expander, &label);
                    list.append(&row);
                }

                // Select first row once the first batch is done.
                if start == 0 {
                    if let Some(first) = list.row_at_index(0) {
                        list.select_row(Some(&first));
                    }
                }

                offset.set(end);
                if end < entries.len() {
                    let next = Rc::clone(&build_batch_ref);
                    glib::idle_add_local_once(move || {
                        if let Some(f) = next.borrow().as_ref() {
                            f();
                        }
                    });
                }
            }));

            // Kick off the first batch.
            let kick = Rc::clone(&build_batch);
            glib::idle_add_local_once(move || {
                if let Some(f) = kick.borrow().as_ref() {
                    f();
                }
            });
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
        let player_idle = Rc::clone(&player);
        let btn_idle = Rc::clone(&active_play_btn);
        let win_idle = window.clone();
        let sections_idle = sections_box.clone();
        let loading_idle = loading_label.clone();
        let monitors_idle = Rc::clone(&monitors);
        let loaded = std::cell::Cell::new(false);

        window.connect_map(move |_| {
            // Guard: only load once (map can fire on re-show).
            if loaded.replace(true) {
                return;
            }
            let player_idle = Rc::clone(&player_idle);
            let btn_idle = Rc::clone(&btn_idle);
            let win_idle = win_idle.clone();
            let sections_idle = sections_idle.clone();
            let loading_idle = loading_idle.clone();
            let monitors_idle = Rc::clone(&monitors_idle);
            glib::timeout_add_local_once(std::time::Duration::from_millis(16), move || {
                let t = std::time::Instant::now();

                // Remove loading indicator immediately so the window
                // doesn't look frozen while sections are populated.
                sections_idle.remove(&loading_idle);

                // ── WAV dictation cache section ──
                let (wav_expander, wav_list) = create_section("Dictation cache (0)");
                sections_idle.append(&wav_expander);

                // ── OGG recordings section ──
                let (ogg_expander, ogg_list) = create_section("Recordings (0)");
                sections_idle.append(&ogg_expander);

                // Populate sections via idle callbacks so the GTK
                // main loop stays responsive between each section.
                {
                    let player = Rc::clone(&player_idle);
                    let btn = Rc::clone(&btn_idle);
                    let win = win_idle.clone();
                    let wav_list_ref = wav_list.clone();
                    let wav_exp_ref = wav_expander.clone();
                    glib::idle_add_local_once(move || {
                        let wav_recordings = list_wav_recordings().unwrap_or_default();
                        log::debug!(
                            "record-ui: list_wav_recordings ({} entries) {:.0?}",
                            wav_recordings.len(),
                            t.elapsed(),
                        );
                        populate_section(
                            "Dictation cache",
                            wav_recordings,
                            &wav_list_ref,
                            &wav_exp_ref,
                            &player,
                            &btn,
                            &win,
                        );
                        if wav_list_ref.first_child().is_some() {
                            wav_list_ref.grab_focus();
                        }
                    });
                }

                {
                    let player = Rc::clone(&player_idle);
                    let btn = Rc::clone(&btn_idle);
                    let win = win_idle.clone();
                    let ogg_list_ref = ogg_list.clone();
                    let ogg_exp_ref = ogg_expander.clone();
                    glib::idle_add_local_once(move || {
                        let ogg_recordings = list_ogg_recordings().unwrap_or_default();
                        log::debug!(
                            "record-ui: list_ogg_recordings ({} entries) {:.0?}",
                            ogg_recordings.len(),
                            t.elapsed(),
                        );
                        populate_section(
                            "Recordings",
                            ogg_recordings,
                            &ogg_list_ref,
                            &ogg_exp_ref,
                            &player,
                            &btn,
                            &win,
                        );
                    });
                }

                // ── Inotify via gio::FileMonitor ──
                /// Shared context passed to [`watch_directory`] to avoid
                /// exceeding the clippy argument-count limit.
                struct WatchCtx {
                    list: gtk4::ListBox,
                    expander: gtk4::Expander,
                    player: Rc<RefCell<Option<WavPlayer>>>,
                    active_play_btn: Rc<RefCell<Option<gtk4::Button>>>,
                    window: gtk4::Window,
                }

                fn watch_directory(
                    dir: &std::path::Path,
                    label: &'static str,
                    ctx: &WatchCtx,
                    list_fn: fn() -> Result<Vec<RecordingEntry>, TalkError>,
                    monitors: &Rc<RefCell<Vec<gtk4::gio::FileMonitor>>>,
                ) {
                    use gtk4::prelude::*;

                    let gio_dir = gtk4::gio::File::for_path(dir);
                    let monitor = match gio_dir.monitor_directory(
                        gtk4::gio::FileMonitorFlags::NONE,
                        gtk4::gio::Cancellable::NONE,
                    ) {
                        Ok(m) => m,
                        Err(e) => {
                            log::warn!("failed to watch {}: {}", dir.display(), e);
                            return;
                        }
                    };

                    let list_ref = ctx.list.clone();
                    let exp_ref = ctx.expander.clone();
                    let player_ref = Rc::clone(&ctx.player);
                    let btn_ref = Rc::clone(&ctx.active_play_btn);
                    let win_ref = ctx.window.clone();

                    monitor.connect_changed(move |_monitor, file, _other, event| {
                        use gtk4::gio::FileMonitorEvent;
                        match event {
                            FileMonitorEvent::Created
                            | FileMonitorEvent::Deleted
                            | FileMonitorEvent::ChangesDoneHint => {}
                            _ => return,
                        }

                        let name = match file.basename() {
                            Some(n) => n.to_string_lossy().to_string(),
                            None => return,
                        };

                        // Ignore waterfall cache files (.wf).
                        if name.ends_with(".wf") {
                            return;
                        }

                        // YAML changed → update the affected row in place
                        // instead of rebuilding the entire list.
                        if name.ends_with(".yml") {
                            // Extract stem (everything before the first _
                            // after the timestamp, or the whole basename
                            // minus the extension for simple names).
                            // The stem matches the audio file's stem.
                            let yml_stem = name.split('_').next().unwrap_or("");
                            if yml_stem.is_empty() {
                                return;
                            }

                            // Find the matching row by widget_name (audio path).
                            let mut idx = 0;
                            while let Some(row) = list_ref.row_at_index(idx) {
                                let row_name = row.widget_name();
                                let row_path = std::path::Path::new(row_name.as_str());
                                let row_stem =
                                    row_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                                if row_stem == yml_stem {
                                    // Rebuild just this row.
                                    if let Ok(entries) = list_fn() {
                                        if let Some(entry) = entries.iter().find(|e| {
                                            e.path
                                                .file_stem()
                                                .and_then(|s| s.to_str())
                                                .unwrap_or("")
                                                == yml_stem
                                        }) {
                                            let new_row = build_row(
                                                entry,
                                                &player_ref,
                                                &btn_ref,
                                                &win_ref,
                                                &list_ref,
                                                &exp_ref,
                                                label,
                                            );
                                            // Insert new row at same position,
                                            // then remove the old one.
                                            list_ref.insert(&new_row, idx);
                                            list_ref.remove(&row);
                                        }
                                    }
                                    return;
                                }
                                idx += 1;
                            }
                            // Row not found — might be for a different section.
                            return;
                        }

                        // Audio file created/deleted → full rebuild
                        // (rare: only happens when recording or deleting).
                        if let Ok(entries) = list_fn() {
                            populate_section(
                                label,
                                entries,
                                &list_ref,
                                &exp_ref,
                                &player_ref,
                                &btn_ref,
                                &win_ref,
                            );
                        }
                    });

                    monitors.borrow_mut().push(monitor);
                }

                // Watch WAV cache directory
                if let Ok(wav_dir) = recording_cache::recordings_dir() {
                    let ctx = WatchCtx {
                        list: wav_list,
                        expander: wav_expander,
                        player: Rc::clone(&player_idle),
                        active_play_btn: Rc::clone(&btn_idle),
                        window: win_idle.clone(),
                    };
                    watch_directory(
                        &wav_dir,
                        "Dictation cache",
                        &ctx,
                        list_wav_recordings,
                        &monitors_idle,
                    );
                }

                // Watch OGG output directory
                if let Ok(config) = Config::load(None) {
                    let ctx = WatchCtx {
                        list: ogg_list,
                        expander: ogg_expander,
                        player: Rc::clone(&player_idle),
                        active_play_btn: Rc::clone(&btn_idle),
                        window: win_idle.clone(),
                    };
                    watch_directory(
                        &config.output_dir,
                        "Recordings",
                        &ctx,
                        list_ogg_recordings,
                        &monitors_idle,
                    );
                }

                log::debug!("record-ui: data loaded + watches {:.0?}", t.elapsed());
            });
        });
    }

    // Escape to close
    {
        let ml = main_loop.clone();
        let win = window.clone();
        let player_ref = Rc::clone(&player);
        let btn_ref = Rc::clone(&active_play_btn);
        let key_ctl = gtk4::EventControllerKey::new();
        key_ctl.connect_key_pressed(move |_, key, _, _| {
            if key == gtk4::gdk::Key::Escape {
                stop_playback(&player_ref, &btn_ref);
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
        let player_ref = Rc::clone(&player);
        let btn_ref = Rc::clone(&active_play_btn);
        close_btn.connect_clicked(move |_| {
            stop_playback(&player_ref, &btn_ref);
            win.set_visible(false);
            ml.quit();
        });
    }

    // Window close
    {
        let ml = main_loop.clone();
        let player_ref = Rc::clone(&player);
        let btn_ref = Rc::clone(&active_play_btn);
        window.connect_close_request(move |win| {
            stop_playback(&player_ref, &btn_ref);
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
        let player_init = Rc::clone(&player);
        glib::idle_add_local_once(move || match WavPlayer::new() {
            Ok(p) => {
                *player_init.borrow_mut() = Some(p);
            }
            Err(e) => {
                log::warn!("audio output unavailable, play disabled: {}", e);
            }
        });
    }

    main_loop.run();
    window.close();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::audio::{ogg_duration_secs, wav_duration_secs};
    use super::super::entries::{format_duration, format_size};

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
