//! Picker UI for selecting among multiple transcription candidates.
//!
//! Extracted from `dictate.rs` — contains the GTK4 picker window,
//! X11 window centering/raising helpers, and WAV PCM reading used
//! by the picker's realtime transcription support.

use crate::config::{Config, Provider};
use crate::error::TalkError;
use crate::transcription::{self, BatchTranscriber, RealtimeTranscriber};
use std::path::PathBuf;

use super::backend::{read_wav_pcm_samples, run_realtime_transcription, spawn_transcription};
use super::cache;
use crate::record::player::WavPlayer;

/// Window title used for the picker — also used for single-instance
/// detection via `x11_centre_and_raise`.
pub(super) const PICKER_TITLE: &str = "talk-rs: select transcription";

/// Candidate transcription result sent from async tasks to the GTK
/// thread via a `std::sync::mpsc` channel.
pub(super) struct PickerCandidate {
    provider: Provider,
    model: String,
    text: String,
    /// When set, the candidate failed and this is the error message.
    error: Option<String>,
    /// Whether this candidate came from a realtime (streaming) transcriber.
    streaming: bool,
}

impl PickerCandidate {
    /// Build a successful transcription candidate.
    pub(super) fn success(
        provider: Provider,
        model: String,
        text: String,
        streaming: bool,
    ) -> Self {
        Self {
            provider,
            model,
            text,
            error: None,
            streaming,
        }
    }

    /// Build a failed transcription candidate.
    pub(super) fn error(
        provider: Provider,
        model: String,
        message: String,
        streaming: bool,
    ) -> Self {
        Self {
            provider,
            model,
            text: String::new(),
            error: Some(message),
            streaming,
        }
    }
}

/// Messages sent from async tasks to the GTK poll loop.
pub(super) enum PickerMessage {
    /// A transcription result (success or error).
    Candidate(PickerCandidate),
    /// All initial batch transcription tasks have completed.
    /// The poll loop should mark any remaining batch spinners as
    /// failed but keep running to receive retry results.
    InitialBatchDone,
    /// Incremental text update from a realtime transcriber.
    /// Contains the full accumulated text so GTK just replaces the label.
    StreamUpdate {
        provider: Provider,
        model: String,
        accumulated_text: String,
    },
}

/// Result of the picker: selected provider/model/text and whether it
/// was the pre-populated cached entry (in which case the caller should
/// skip pasting since the text is already in the target window).
pub(super) struct PickerSelection {
    pub(super) provider: Provider,
    pub(super) model: String,
    pub(super) text: String,
    pub(super) is_cached: bool,
    pub(super) streaming: bool,
}

/// Show a GTK4 picker window that appears immediately with a spinner,
/// then progressively fills with transcription candidates as parallel
/// API calls complete.
///
/// The window is undecorated, centered on the active monitor, and
/// always-on-top.  Layout is a 3-column table: transcript (monospace,
/// wrapping) | provider | model.
///
/// `cached_entries` contains pre-populated results that are shown
/// immediately (no spinner, no API call).  Each tuple is
/// `(provider, model, text, is_primary)` where `is_primary` marks the
/// entry that was already pasted in a previous run (selecting it again
/// skips re-pasting).
///
/// Returns `Some(PickerSelection)` for the selected candidate, or
/// `None` if the user cancelled.
pub(super) async fn pick_with_streaming_gtk(
    mut transcribers: Vec<(Provider, String, Box<dyn BatchTranscriber>)>,
    audio_path: PathBuf,
    mut cached_entries: Vec<(Provider, String, String, bool, bool)>,
    config: Config,
    mut realtime_transcribers: Vec<(Provider, String, Box<dyn RealtimeTranscriber>)>,
) -> Result<Option<PickerSelection>, TalkError> {
    if transcribers.is_empty() && cached_entries.is_empty() && realtime_transcribers.is_empty() {
        return Ok(None);
    }

    // Stable display order: sort by (provider, model, streaming) so
    // the list looks identical every time the picker opens.
    cached_entries.sort_by(|(pa, ma, _, _, sa), (pb, mb, _, _, sb)| {
        pa.to_string()
            .cmp(&pb.to_string())
            .then(ma.cmp(mb))
            .then(sa.cmp(sb))
    });
    transcribers
        .sort_by(|(pa, ma, _), (pb, mb, _)| pa.to_string().cmp(&pb.to_string()).then(ma.cmp(mb)));
    realtime_transcribers
        .sort_by(|(pa, ma, _), (pb, mb, _)| pa.to_string().cmp(&pb.to_string()).then(ma.cmp(mb)));

    // Extract (provider, model) labels before transcribers are consumed
    // so the GTK thread can pre-create rows with spinners.
    let pending_info: Vec<(Provider, String)> = transcribers
        .iter()
        .map(|(p, m, _)| (*p, m.clone()))
        .collect();
    let realtime_pending_info: Vec<(Provider, String)> = realtime_transcribers
        .iter()
        .map(|(p, m, _)| (*p, m.clone()))
        .collect();

    // Channels: transcription tasks → GTK, GTK → caller.
    let (msg_tx, msg_rx) = std::sync::mpsc::channel::<PickerMessage>();
    let (sel_tx, sel_rx) = tokio::sync::oneshot::channel::<Option<usize>>();

    // Retry channel: GTK thread → async retry listener.
    // The bool indicates whether this is a streaming (realtime) retry.
    // Uses tokio unbounded channel so the receiver can `.await` — a
    // std::sync::mpsc would block the tokio worker thread and prevent
    // the runtime from shutting down when the picker closes.
    let (retry_tx, mut retry_rx) =
        tokio::sync::mpsc::unbounded_channel::<(Provider, String, bool)>();

    // Shared storage written by the GTK thread after selection,
    // read by the caller after the GTK thread finishes.
    // Tuple: (provider, model, text, is_primary, is_error, streaming)
    let results = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(
        Provider,
        String,
        String,
        bool,
        bool,
        bool,
    )>::new()));
    let results_for_gtk = results.clone();

    // Clone audio path for the GTK play button before moving into closures.
    let audio_path_for_player = audio_path.clone();

    // ── GTK window ──────────────────────────────────────────────
    let gtk_handle = tokio::task::spawn_blocking(move || -> Result<(), TalkError> {
        use gtk4::glib;
        use gtk4::prelude::*;
        use std::cell::RefCell;
        use std::rc::Rc;

        gtk4::init().map_err(|e| TalkError::Config(format!("failed to initialize GTK: {}", e)))?;

        let theme = crate::gtk_theme::ThemeColors::resolve();
        let error_hex = &theme.error;

        let window = gtk4::Window::builder()
            .title(PICKER_TITLE)
            .default_width(900)
            .default_height(500)
            .decorated(false)
            .build();

        // CSS notes:
        //   - GtkListBox CSS node is "list", not "listbox"
        //   - Every stateful rule is duplicated with :backdrop
        //     to prevent the theme from lightening on focus loss
        //   - row:selected uses a translucent accent instead of
        //     the theme's solid #E95420
        crate::gtk_theme::load_css(&theme.base_css(&format!(
            ".transcript {{ font-family: monospace; }} \
             .error {{ font-style: italic; color: alpha({err}, 0.8); }} \
             .retry-btn {{ min-width: 0; min-height: 0; padding: 2px 6px; font-size: 14px; }} \
             .play-btn {{ min-width: 32px; min-height: 32px; padding: 0; font-size: 16px; }}",
            err = error_hex,
        )));

        let root = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
        root.set_margin_top(6);
        root.set_margin_bottom(6);
        root.set_margin_start(6);
        root.set_margin_end(6);

        // ── Audio player for playback button ─────────────────────
        let player: Rc<Option<WavPlayer>> = Rc::new(match WavPlayer::new() {
            Ok(p) => Some(p),
            Err(e) => {
                log::warn!("audio output unavailable, play button disabled: {}", e);
                None
            }
        });

        // Play-button bar above the transcription list.
        let play_bar = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
        play_bar.set_margin_start(4);
        play_bar.set_margin_end(4);
        play_bar.set_margin_bottom(4);

        let play_btn = gtk4::Button::with_label("\u{25b6}");
        play_btn.set_tooltip_text(Some("Play recorded audio"));
        play_btn.add_css_class("play-btn");
        if player.is_none() {
            play_btn.set_sensitive(false);
        }

        {
            let player_ref = Rc::clone(&player);
            let audio = audio_path_for_player;
            play_btn.connect_clicked(move |btn| {
                let is_playing = btn.label().is_some_and(|l| l == "\u{25a0}");
                if is_playing {
                    if let Some(ref p) = *player_ref {
                        p.stop();
                    }
                    btn.set_label("\u{25b6}");
                    btn.set_tooltip_text(Some("Play recorded audio"));
                } else {
                    if let Some(ref p) = *player_ref {
                        if let Err(e) = p.play(&audio) {
                            log::warn!("failed to play audio: {}", e);
                            return;
                        }
                    }
                    btn.set_label("\u{25a0}");
                    btn.set_tooltip_text(Some("Stop playback"));
                }
            });
        }

        // Poll for playback completion to auto-reset the button.
        {
            let player_poll = Rc::clone(&player);
            let btn_poll = play_btn.clone();
            glib::timeout_add_local(std::time::Duration::from_millis(200), move || {
                let is_playing = btn_poll.label().is_some_and(|l| l == "\u{25a0}");
                if is_playing {
                    let finished = player_poll
                        .as_ref()
                        .as_ref()
                        .is_none_or(|p| p.is_finished());
                    if finished {
                        btn_poll.set_label("\u{25b6}");
                        btn_poll.set_tooltip_text(Some("Play recorded audio"));
                    }
                }
                glib::ControlFlow::Continue
            });
        }

        play_bar.append(&play_btn);
        root.append(&play_bar);

        let scrolled = gtk4::ScrolledWindow::builder()
            .vexpand(true)
            .hexpand(true)
            .build();
        let list = gtk4::ListBox::new();
        list.set_selection_mode(gtk4::SelectionMode::Single);
        list.set_activate_on_single_click(false);
        scrolled.set_child(Some(&list));
        root.append(&scrolled);

        // WindowHandle enables dragging the window from any non-
        // interactive area (help label, table background, margins).
        let handle = gtk4::WindowHandle::new();
        handle.set_child(Some(&root));
        window.set_child(Some(&handle));

        let main_loop = glib::MainLoop::new(None, false);
        let sel_sender: Rc<RefCell<Option<tokio::sync::oneshot::Sender<Option<usize>>>>> =
            Rc::new(RefCell::new(Some(sel_tx)));
        type CandidateList = Vec<(Provider, String, String, bool, bool, bool)>;
        let local_candidates: Rc<RefCell<CandidateList>> = Rc::new(RefCell::new(Vec::new()));
        // Transcript cells — used to swap spinner → label when results arrive.
        let transcript_cells: Rc<RefCell<Vec<gtk4::Box>>> = Rc::new(RefCell::new(Vec::new()));

        // Helper: build a row skeleton with provider + streaming + model labels.
        // Returns (hbox, transcript_cell) — caller fills the cell
        // with either a spinner or a transcript label.
        fn make_row_skeleton(
            provider: &str,
            model: &str,
            streaming: bool,
        ) -> (gtk4::Box, gtk4::Box) {
            use gtk4::prelude::*;

            let hbox = gtk4::Box::new(gtk4::Orientation::Horizontal, 12);
            hbox.set_margin_top(4);
            hbox.set_margin_bottom(4);
            hbox.set_margin_start(4);
            hbox.set_margin_end(4);

            // Transcript cell (container swapped between spinner / label)
            let cell = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
            cell.set_hexpand(true);
            hbox.append(&cell);

            // Provider (fixed width for column alignment)
            let prov_label = gtk4::Label::new(Some(provider));
            prov_label.set_xalign(0.0);
            prov_label.set_width_chars(8);
            prov_label.set_max_width_chars(8);
            prov_label.set_selectable(false);
            prov_label.add_css_class("dim");
            hbox.append(&prov_label);

            // Streaming indicator (between provider and model)
            let stream_label = gtk4::Label::new(Some(if streaming { "\u{26a1}" } else { "" }));
            stream_label.set_xalign(0.5);
            stream_label.set_width_chars(3);
            stream_label.set_max_width_chars(3);
            stream_label.set_selectable(false);
            stream_label.add_css_class("dim");
            hbox.append(&stream_label);

            // Model (fixed width for column alignment)
            let model_label = gtk4::Label::new(Some(model));
            model_label.set_xalign(0.0);
            model_label.set_width_chars(24);
            model_label.set_max_width_chars(24);
            model_label.set_ellipsize(gtk4::pango::EllipsizeMode::End);
            model_label.set_selectable(false);
            model_label.add_css_class("dim");
            hbox.append(&model_label);

            (hbox, cell)
        }

        // Helper: create a monospace transcript label.
        fn make_transcript_label(text: &str) -> gtk4::Label {
            use gtk4::prelude::*;

            let label = gtk4::Label::new(Some(text));
            label.set_xalign(0.0);
            label.set_hexpand(true);
            label.set_wrap(true);
            label.set_wrap_mode(gtk4::pango::WrapMode::WordChar);
            label.set_selectable(false);
            label.add_css_class("transcript");
            label
        }

        // ── Pre-create all rows ─────────────────────────────────

        // Cached entries: rows with transcript already filled.
        // The `is_primary` flag marks the entry that was already
        // pasted in a previous run — it gets pre-selected.
        let mut selected_row = false;
        for (i, (provider, model, text, is_primary, streaming)) in cached_entries.iter().enumerate()
        {
            let (hbox, cell) = make_row_skeleton(&provider.to_string(), model, *streaming);
            cell.append(&make_transcript_label(text));

            let row = gtk4::ListBoxRow::new();
            row.set_child(Some(&hbox));
            list.append(&row);

            // Select the primary entry, or the very first one.
            if *is_primary || (i == 0 && !selected_row) {
                list.select_row(Some(&row));
                selected_row = true;
            }

            local_candidates.borrow_mut().push((
                *provider,
                model.clone(),
                text.clone(),
                *is_primary,
                false,
                *streaming,
            ));
            transcript_cells.borrow_mut().push(cell);
        }

        // Pending batch entries: row with spinner in transcript cell.
        for (provider, model) in &pending_info {
            let (hbox, cell) = make_row_skeleton(&provider.to_string(), model, false);
            let spinner = gtk4::Spinner::new();
            spinner.start();
            cell.append(&spinner);

            let row = gtk4::ListBoxRow::new();
            row.set_child(Some(&hbox));
            list.append(&row);

            local_candidates.borrow_mut().push((
                *provider,
                model.clone(),
                String::new(),
                false,
                false,
                false,
            ));
            transcript_cells.borrow_mut().push(cell);
        }

        // Pending realtime entries: row with empty transcript label
        // (text fills incrementally, no spinner).
        for (provider, model) in &realtime_pending_info {
            let (hbox, cell) = make_row_skeleton(&provider.to_string(), model, true);
            cell.append(&make_transcript_label(""));

            let row = gtk4::ListBoxRow::new();
            row.set_child(Some(&hbox));
            list.append(&row);

            local_candidates.borrow_mut().push((
                *provider,
                model.clone(),
                String::new(),
                false,
                false,
                true,
            ));
            transcript_cells.borrow_mut().push(cell);
        }

        // Select first row when there are no cached entries.
        if !selected_row {
            if let Some(first) = list.row_at_index(0) {
                list.select_row(Some(&first));
            }
        }

        // Enter / double-click confirms selection (only if text is loaded)
        {
            let sel = Rc::clone(&sel_sender);
            let ml = main_loop.clone();
            let cands = Rc::clone(&local_candidates);
            let win = window.clone();
            list.connect_row_activated(move |_, row| {
                let idx = row.index() as usize;
                let ready = cands
                    .borrow()
                    .get(idx)
                    .is_some_and(|(_, _, text, _, is_error, _)| !text.is_empty() && !is_error);
                if ready {
                    win.set_visible(false);
                    if let Some(tx) = sel.borrow_mut().take() {
                        let _ = tx.send(Some(idx));
                    }
                    ml.quit();
                }
            });
        }

        // Escape cancels
        {
            let sel = Rc::clone(&sel_sender);
            let ml = main_loop.clone();
            let win = window.clone();
            let player_ref = Rc::clone(&player);
            let key_ctl = gtk4::EventControllerKey::new();
            key_ctl.connect_key_pressed(move |_, key, _, _| {
                if key == gtk4::gdk::Key::Escape {
                    if let Some(ref p) = *player_ref {
                        p.stop();
                    }
                    win.set_visible(false);
                    if let Some(tx) = sel.borrow_mut().take() {
                        let _ = tx.send(None);
                    }
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
            let sel = Rc::clone(&sel_sender);
            let ml = main_loop.clone();
            let player_ref = Rc::clone(&player);
            window.connect_close_request(move |win| {
                if let Some(ref p) = *player_ref {
                    p.stop();
                }
                win.set_visible(false);
                if let Some(tx) = sel.borrow_mut().take() {
                    let _ = tx.send(None);
                }
                ml.quit();
                glib::Propagation::Proceed
            });
        }

        // Build an error cell with a retry button (↻).
        //
        // Clicking the button replaces the error with a spinner and
        // sends a retry request through `retry_tx`.
        let make_error_cell = {
            let retry_tx = retry_tx.clone();
            let cands_for_err = Rc::clone(&local_candidates);
            let cells_for_err = Rc::clone(&transcript_cells);
            move |cell: &gtk4::Box, idx: usize, msg: &str| {
                while let Some(child) = cell.first_child() {
                    cell.remove(&child);
                }
                let hbox = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);

                let err = gtk4::Label::new(Some(msg));
                err.set_xalign(0.0);
                err.set_hexpand(true);
                err.set_wrap(true);
                err.set_wrap_mode(gtk4::pango::WrapMode::WordChar);
                err.add_css_class("error");
                hbox.append(&err);

                let retry_btn = gtk4::Button::with_label("↻");
                retry_btn.set_tooltip_text(Some("Retry transcription"));
                retry_btn.add_css_class("retry-btn");
                {
                    let tx = retry_tx.clone();
                    let cands_ref = Rc::clone(&cands_for_err);
                    let cells_ref = Rc::clone(&cells_for_err);
                    retry_btn.connect_clicked(move |_| {
                        let mut cands = cands_ref.borrow_mut();
                        if let Some((provider, model, text, _, is_error, streaming)) =
                            cands.get_mut(idx)
                        {
                            let _ = tx.send((*provider, model.clone(), *streaming));
                            // Reset row state.
                            text.clear();
                            *is_error = false;
                            // Replace error with spinner (batch) or
                            // empty label (realtime).
                            let is_streaming = *streaming;
                            let cells = cells_ref.borrow();
                            if let Some(cell) = cells.get(idx) {
                                while let Some(child) = cell.first_child() {
                                    cell.remove(&child);
                                }
                                if is_streaming {
                                    cell.append(&make_transcript_label(""));
                                } else {
                                    let spinner = gtk4::Spinner::new();
                                    spinner.start();
                                    cell.append(&spinner);
                                }
                            }
                        }
                    });
                }
                hbox.append(&retry_btn);
                cell.append(&hbox);
            }
        };

        // Poll transcription results — update existing rows in-place.
        {
            let list = list.clone();
            let cands = Rc::clone(&local_candidates);
            let cells = Rc::clone(&transcript_cells);
            let make_err = make_error_cell.clone();
            glib::timeout_add_local(std::time::Duration::from_millis(50), move || {
                loop {
                    match msg_rx.try_recv() {
                        Ok(PickerMessage::Candidate(c)) => {
                            // Find the pre-created row by
                            // (provider, model, streaming).
                            let idx = {
                                let cands = cands.borrow();
                                cands.iter().position(|(p, m, _, _, _, s)| {
                                    *p == c.provider && *m == c.model && *s == c.streaming
                                })
                            };
                            if let Some(idx) = idx {
                                // Clear cell content (spinner or previous error).
                                {
                                    let cells = cells.borrow();
                                    if let Some(cell) = cells.get(idx) {
                                        while let Some(child) = cell.first_child() {
                                            cell.remove(&child);
                                        }
                                    }
                                }

                                if let Some(ref err_msg) = c.error {
                                    cands.borrow_mut()[idx].4 = true;
                                    make_err(&cells.borrow()[idx], idx, err_msg);
                                } else {
                                    let mut cands = cands.borrow_mut();
                                    cands[idx].2 = c.text.clone();
                                    cells.borrow()[idx].append(&make_transcript_label(&c.text));
                                }
                            }
                        }
                        Ok(PickerMessage::StreamUpdate {
                            provider,
                            model,
                            accumulated_text,
                        }) => {
                            // Find the realtime row by
                            // (provider, model, streaming=true).
                            let idx = {
                                let cands = cands.borrow();
                                cands.iter().position(|(p, m, _, _, _, s)| {
                                    *p == provider && *m == model && *s
                                })
                            };
                            if let Some(idx) = idx {
                                // Replace cell content with updated text.
                                {
                                    let cells = cells.borrow();
                                    if let Some(cell) = cells.get(idx) {
                                        while let Some(child) = cell.first_child() {
                                            cell.remove(&child);
                                        }
                                        cell.append(&make_transcript_label(&accumulated_text));
                                    }
                                }
                                cands.borrow_mut()[idx].2 = accumulated_text;
                            }
                        }
                        Ok(PickerMessage::InitialBatchDone) => {
                            // Mark remaining batch spinners as failed.
                            // Realtime rows (streaming=true) may still
                            // be receiving data — skip them.
                            let failed: Vec<usize> = {
                                let mut cands = cands.borrow_mut();
                                let indices: Vec<usize> = cands
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, (_, _, text, _, is_error, streaming))| {
                                        text.is_empty() && !*is_error && !*streaming
                                    })
                                    .map(|(i, _)| i)
                                    .collect();
                                for &idx in &indices {
                                    cands[idx].4 = true;
                                }
                                indices
                            };
                            for idx in failed {
                                make_err(&cells.borrow()[idx], idx, "no response");
                            }
                            list.grab_focus();
                            // Keep polling — retries and realtime
                            // results may still arrive.
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                            return glib::ControlFlow::Continue;
                        }
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            // All senders dropped (including retry
                            // listener) — stop polling.
                            return glib::ControlFlow::Break;
                        }
                    }
                }
            });
        }

        crate::gtk_theme::present_centred(&window);
        list.grab_focus();
        main_loop.run();
        window.close();

        // Publish the candidate list so the caller can look up the
        // selected index after this thread finishes.
        if let Ok(mut r) = results_for_gtk.lock() {
            *r = local_candidates.borrow().clone();
        }

        Ok(())
    });

    // ── Parallel transcriptions ─────────────────────────────────
    let audio_path_for_cache = audio_path.clone();

    // Fire initial batch.
    let done_tx = msg_tx.clone();
    let retry_msg_tx = msg_tx.clone();

    // Read WAV samples once for realtime transcribers (shared via Arc).
    let wav_samples = if !realtime_transcribers.is_empty() {
        match read_wav_pcm_samples(&audio_path) {
            Ok(samples) => Some(std::sync::Arc::new(samples)),
            Err(e) => {
                log::warn!("failed to read WAV for realtime: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Spawn realtime transcription tasks.
    for (provider, model, transcriber) in realtime_transcribers {
        let tx = msg_tx.clone();
        let samples = wav_samples.clone();
        tokio::spawn(async move {
            let Some(samples) = samples else {
                let _ = tx.send(PickerMessage::Candidate(PickerCandidate::error(
                    provider,
                    model,
                    "failed to read WAV samples".into(),
                    true,
                )));
                return;
            };
            run_realtime_transcription(transcriber, samples, tx, provider, model).await;
        });
    }

    drop(msg_tx); // only clones survive from here
    tokio::spawn({
        let audio = audio_path.clone();
        let tx = done_tx.clone();
        async move {
            let mut tasks = tokio::task::JoinSet::new();
            for (provider, model, transcriber) in transcribers {
                spawn_transcription(
                    &mut tasks,
                    tx.clone(),
                    audio.clone(),
                    provider,
                    model,
                    transcriber,
                );
            }
            drop(tx);
            while (tasks.join_next().await).is_some() {}
            // All initial batch tasks finished — notify GTK.
            let _ = done_tx.send(PickerMessage::InitialBatchDone);
        }
    });

    // ── Retry listener ──────────────────────────────────────────
    // Receives (provider, model, streaming) from the GTK retry button
    // and fires a new transcription, reusing the same result channel.
    {
        let audio = audio_path;
        let tx = retry_msg_tx;
        let wav_for_retry = wav_samples;
        tokio::spawn(async move {
            while let Some((provider, model, streaming)) = retry_rx.recv().await {
                log::info!("retrying {}:{} (streaming={})", provider, model, streaming);
                if streaming {
                    // Realtime retry: re-stream WAV samples.
                    let samples = match wav_for_retry {
                        Some(ref s) => s.clone(),
                        None => {
                            let _ = tx.send(PickerMessage::Candidate(PickerCandidate::error(
                                provider,
                                model,
                                "WAV samples unavailable".into(),
                                true,
                            )));
                            continue;
                        }
                    };
                    match transcription::create_realtime_transcriber(
                        &config,
                        provider,
                        Some(&model),
                    ) {
                        Ok(transcriber) => {
                            let tx = tx.clone();
                            let model_clone = model.clone();
                            tokio::spawn(async move {
                                run_realtime_transcription(
                                    transcriber,
                                    samples,
                                    tx,
                                    provider,
                                    model_clone,
                                )
                                .await;
                            });
                        }
                        Err(e) => {
                            log::warn!(
                                "retry {}:{} realtime transcriber creation failed: {}",
                                provider,
                                model,
                                e
                            );
                            let _ = tx.send(PickerMessage::Candidate(PickerCandidate::error(
                                provider,
                                model,
                                format!("{e}"),
                                true,
                            )));
                        }
                    }
                } else {
                    // Batch retry (existing logic).
                    match transcription::create_batch_transcriber(
                        &config,
                        provider,
                        Some(&model),
                        false,
                    ) {
                        Ok(transcriber) => {
                            let mut tasks = tokio::task::JoinSet::new();
                            spawn_transcription(
                                &mut tasks,
                                tx.clone(),
                                audio.clone(),
                                provider,
                                model,
                                transcriber,
                            );
                            while (tasks.join_next().await).is_some() {}
                        }
                        Err(e) => {
                            log::warn!(
                                "retry {}:{} transcriber creation failed: {}",
                                provider,
                                model,
                                e
                            );
                            let _ = tx.send(PickerMessage::Candidate(PickerCandidate::error(
                                provider,
                                model,
                                format!("{e}"),
                                false,
                            )));
                        }
                    }
                }
            }
            // retry_tx dropped (GTK closed) — exit.
        });
    }

    // ── Wait for result ─────────────────────────────────────────
    let selected_index = sel_rx
        .await
        .map_err(|_| TalkError::Config("GTK picker closed unexpectedly".into()))?;

    gtk_handle
        .await
        .map_err(|e| TalkError::Config(format!("GTK picker task failed: {}", e)))??;

    // Persist all successful transcription results to the picker
    // cache so that reopening the picker for the same audio file
    // returns instantly without any API calls.
    {
        let items = results
            .lock()
            .map_err(|_| TalkError::Config("results lock poisoned".into()))?;
        let to_cache: Vec<cache::CachedResult> = items
            .iter()
            .filter(|(_, _, text, _, is_error, _)| !text.is_empty() && !*is_error)
            .map(|(p, m, t, _, _, s)| cache::CachedResult {
                provider: p.to_string(),
                model: m.clone(),
                text: t.clone(),
                streaming: *s,
            })
            .collect();
        if let Err(e) = cache::write_results(&audio_path_for_cache, &to_cache) {
            log::warn!("failed to write picker cache: {}", e);
        }
    }

    match selected_index {
        Some(idx) => {
            let items = results
                .lock()
                .map_err(|_| TalkError::Config("results lock poisoned".into()))?;
            if idx < items.len() {
                let (p, m, t, cached, _is_error, streaming) = items[idx].clone();
                Ok(Some(PickerSelection {
                    provider: p,
                    model: m,
                    text: t,
                    is_cached: cached,
                    streaming,
                }))
            } else {
                Ok(None)
            }
        }
        None => Ok(None),
    }
}
