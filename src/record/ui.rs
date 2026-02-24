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
pub async fn record_ui() -> Result<(), TalkError> {
    let ogg_recordings = list_ogg_recordings()?;
    let wav_recordings = list_wav_recordings()?;

    tokio::task::spawn_blocking(move || show_recordings_window(ogg_recordings, wav_recordings))
        .await
        .map_err(|e| TalkError::Config(format!("GTK task failed: {}", e)))?
}

/// Build and run the GTK4 recordings browser window.
fn show_recordings_window(
    ogg_recordings: Vec<RecordingEntry>,
    wav_recordings: Vec<RecordingEntry>,
) -> Result<(), TalkError> {
    use gtk4::glib;
    use gtk4::prelude::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    gtk4::init().map_err(|e| TalkError::Config(format!("failed to initialize GTK: {}", e)))?;

    let theme = crate::gtk_theme::ThemeColors::resolve();

    let window = gtk4::Window::builder()
        .title(WINDOW_TITLE)
        .default_width(800)
        .default_height(500)
        .decorated(false)
        .build();

    crate::gtk_theme::load_css(&theme.base_css(
        ".transcript { font-family: monospace; opacity: 0.7; } \
         .meta { font-family: monospace; } \
         .play-btn, .folder-btn, .delete-btn { min-width: 28px; min-height: 28px; max-width: 28px; max-height: 28px; padding: 0; font-size: 14px; } \
         .folder-btn label, .delete-btn label { padding-top: 4px; } \
         .section-expander { margin: 4px 2px; } \
         .section-expander > title { font-weight: bold; opacity: 0.85; padding: 4px 0; }",
    ));

    let root = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
    root.set_margin_top(6);
    root.set_margin_bottom(6);
    root.set_margin_start(6);
    root.set_margin_end(6);

    let scrolled = gtk4::ScrolledWindow::builder()
        .vexpand(true)
        .hexpand(true)
        .build();

    // WindowHandle for dragging
    let handle = gtk4::WindowHandle::new();
    handle.set_child(Some(&root));
    window.set_child(Some(&handle));

    let main_loop = glib::MainLoop::new(None, false);

    // Native audio player (cpal). Falls back to no-op if device unavailable.
    let player: Rc<Option<WavPlayer>> = Rc::new(match WavPlayer::new() {
        Ok(p) => Some(p),
        Err(e) => {
            log::warn!("audio output unavailable, play disabled: {}", e);
            None
        }
    });
    // Track which button is currently in "stop" mode.
    let active_play_btn: Rc<RefCell<Option<gtk4::Button>>> = Rc::new(RefCell::new(None));

    /// Stop any active playback and reset the corresponding button.
    fn stop_playback(player: &Rc<Option<WavPlayer>>, btn_ref: &Rc<RefCell<Option<gtk4::Button>>>) {
        if let Some(ref p) = **player {
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
        player: &Rc<Option<WavPlayer>>,
        btn_ref: &Rc<RefCell<Option<gtk4::Button>>>,
    ) {
        stop_playback(player, btn_ref);

        let Some(ref p) = **player else { return };
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
                .as_ref()
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

    {
        // Container inside the scrolled window for both sections
        let sections_box = gtk4::Box::new(gtk4::Orientation::Vertical, 0);

        /// Build a single row (hbox) for a recording entry with all columns
        /// and buttons.
        fn build_row(
            recording: &RecordingEntry,
            player: &Rc<Option<WavPlayer>>,
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

            // Transcript preview (expanding, ellipsized)
            let transcript = gtk4::Label::new(Some(&recording.transcript_preview));
            transcript.set_xalign(0.0);
            transcript.set_hexpand(true);
            transcript.set_ellipsize(gtk4::pango::EllipsizeMode::End);
            transcript.set_max_width_chars(80);
            transcript.set_selectable(false);
            transcript.add_css_class("transcript");
            hbox.append(&transcript);

            // Play button
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
                    loop {
                        match widget {
                            Some(ref w) => {
                                if let Some(row) = w.downcast_ref::<gtk4::ListBoxRow>() {
                                    list_ref.remove(row);
                                    break;
                                }
                                widget = w.parent();
                            }
                            None => return,
                        }
                    }

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
            row
        }

        /// Populate a ListBox with recording entries, updating the Expander
        /// title with the count.  Clears any existing rows first.
        fn populate_section(
            label: &str,
            recordings: &[RecordingEntry],
            list: &gtk4::ListBox,
            expander: &gtk4::Expander,
            player: &Rc<Option<WavPlayer>>,
            active_play_btn: &Rc<RefCell<Option<gtk4::Button>>>,
            window: &gtk4::Window,
        ) {
            use gtk4::prelude::*;

            // Remove existing rows
            while let Some(child) = list.first_child() {
                list.remove(&child);
            }

            for recording in recordings {
                let row = build_row(
                    recording,
                    player,
                    active_play_btn,
                    window,
                    list,
                    expander,
                    label,
                );
                list.append(&row);
            }

            // Select first row
            if let Some(first) = list.row_at_index(0) {
                list.select_row(Some(&first));
            }

            expander.set_label(Some(&format!("{} ({})", label, recordings.len())));
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

        // ── WAV dictation cache section (first) ──
        let (wav_expander, wav_list) = create_section("Dictation cache (0)");
        populate_section(
            "Dictation cache",
            &wav_recordings,
            &wav_list,
            &wav_expander,
            &player,
            &active_play_btn,
            &window,
        );
        sections_box.append(&wav_expander);

        // ── OGG recordings section (second) ──
        let (ogg_expander, ogg_list) = create_section("Recordings (0)");
        populate_section(
            "Recordings",
            &ogg_recordings,
            &ogg_list,
            &ogg_expander,
            &player,
            &active_play_btn,
            &window,
        );
        sections_box.append(&ogg_expander);

        scrolled.set_child(Some(&sections_box));
        root.append(&scrolled);

        // Focus the first non-empty list
        if wav_list.first_child().is_some() {
            wav_list.grab_focus();
        } else if ogg_list.first_child().is_some() {
            ogg_list.grab_focus();
        }

        // ── Inotify via gio::FileMonitor ──
        // Keep monitors alive for the lifetime of the window.
        let monitors: Rc<RefCell<Vec<gtk4::gio::FileMonitor>>> = Rc::new(RefCell::new(Vec::new()));

        /// Shared context passed to [`watch_directory`] to avoid exceeding
        /// the clippy argument-count limit.
        struct WatchCtx {
            list: gtk4::ListBox,
            expander: gtk4::Expander,
            player: Rc<Option<WavPlayer>>,
            active_play_btn: Rc<RefCell<Option<gtk4::Button>>>,
            window: gtk4::Window,
        }

        // Helper: set up a directory monitor that refreshes a section on changes.
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

            monitor.connect_changed(move |_monitor, _file, _other, event| {
                use gtk4::gio::FileMonitorEvent;
                match event {
                    FileMonitorEvent::Created
                    | FileMonitorEvent::Deleted
                    | FileMonitorEvent::ChangesDoneHint => {}
                    _ => return,
                }
                if let Ok(entries) = list_fn() {
                    populate_section(
                        label,
                        &entries,
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
                player: Rc::clone(&player),
                active_play_btn: Rc::clone(&active_play_btn),
                window: window.clone(),
            };
            watch_directory(
                &wav_dir,
                "Dictation cache",
                &ctx,
                list_wav_recordings,
                &monitors,
            );
        }

        // Watch OGG output directory
        if let Ok(config) = Config::load(None) {
            let ctx = WatchCtx {
                list: ogg_list,
                expander: ogg_expander,
                player: Rc::clone(&player),
                active_play_btn: Rc::clone(&active_play_btn),
                window: window.clone(),
            };
            watch_directory(
                &config.output_dir,
                "Recordings",
                &ctx,
                list_ogg_recordings,
                &monitors,
            );
        }

        // Prevent monitors from being dropped
        let _keep_monitors = monitors;
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

    crate::gtk_theme::present_centred(&window);
    main_loop.run();
    window.close();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::audio::wav_duration_secs;
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
