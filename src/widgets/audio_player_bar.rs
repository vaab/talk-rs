//! Shared audio player bar widget with waterfall spectrogram,
//! cursor overlay, drag-to-seek, and play/pause/rewind controls.
//!
//! Used by the picker UI and the record UI.

use crate::record::player::WavPlayer;
use gtk4::glib;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

/// Waterfall column data: (columns, peak).
pub(crate) type WfColumns = (Vec<Vec<f32>>, f32);

/// Build an interactive audio player bar widget.
///
/// Returns a horizontal `gtk4::Box` containing:
/// - Waterfall spectrogram with playback cursor overlay
/// - Drag-to-seek gesture on the waterfall
/// - Rewind and play/pause buttons
/// - 16 ms progress poller with interpolation
///
/// # Arguments
///
/// * `audio_path` — audio file for playback (WAV or OGG)
/// * `player` — shared cpal audio player
/// * `active_play_btn` — tracks which play button is currently active
///   across multiple bars; when a new bar starts playing, the previous
///   bar's button is reset
/// * `waterfall_data` — pre-computed columns, or `None` to compute in
///   a background thread
/// * `height` — content height for the waterfall DrawingArea (pixels)
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_audio_player_bar(
    audio_path: &Path,
    player: &Rc<RefCell<Option<WavPlayer>>>,
    active_play_btn: &Rc<RefCell<Option<gtk4::Button>>>,
    waterfall_data: Option<WfColumns>,
    height: i32,
) -> gtk4::Box {
    use gtk4::prelude::*;

    let play_bar = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
    play_bar.set_hexpand(true);

    // ── Waterfall spectrogram (base layer) ─────────────────────
    let waterfall_area = gtk4::DrawingArea::new();
    waterfall_area.set_hexpand(true);
    waterfall_area.set_vexpand(false);
    waterfall_area.set_content_height(height);
    waterfall_area.add_css_class("waterfall");

    let wf_data: Rc<RefCell<Option<WfColumns>>> = Rc::new(RefCell::new(waterfall_data));

    {
        let data_ref = Rc::clone(&wf_data);
        waterfall_area.set_draw_func(move |_area, cr, width, height| {
            let w = width as usize;
            let h = height as usize;
            if w == 0 || h == 0 {
                return;
            }
            let data = data_ref.borrow();
            if let Some((ref columns, peak)) = *data {
                if peak > 0.0 && !columns.is_empty() {
                    if let Ok(mut surface) = gtk4::cairo::ImageSurface::create(
                        gtk4::cairo::Format::ARgb32,
                        width,
                        height,
                    ) {
                        let stride = surface.stride() as usize;
                        if let Ok(mut surf_data) = surface.data() {
                            let num_rows = crate::x11::render_util::WATERFALL_ROWS;
                            for x in 0..w {
                                let col_idx = x * columns.len() / w;
                                let col = &columns[col_idx];
                                for y in 0..h {
                                    let data_row =
                                        (num_rows - 1) - (y * num_rows / h).min(num_rows - 1);
                                    let magnitude = if data_row < col.len() {
                                        col[data_row]
                                    } else {
                                        0.0
                                    };
                                    let norm = (magnitude / peak).clamp(0.0, 1.0);
                                    let brightness = if norm > 0.0 {
                                        (1.0 + norm * 9.0).log10()
                                    } else {
                                        0.0
                                    };
                                    let alpha = (brightness * 255.0) as u8;
                                    let off = y * stride + x * 4;
                                    if off + 3 < surf_data.len() {
                                        surf_data[off] = alpha;
                                        surf_data[off + 1] = alpha;
                                        surf_data[off + 2] = alpha;
                                        surf_data[off + 3] = alpha;
                                    }
                                }
                            }
                        }
                        cr.set_source_surface(&surface, 0.0, 0.0).unwrap_or(());
                        cr.paint().unwrap_or(());
                    }
                    return;
                }
            }
            cr.set_source_rgb(0.0, 0.0, 0.0);
            cr.paint().unwrap_or(());
        });
    }

    // ── Cursor overlay (lightweight, redraws independently) ───
    let cursor_area = gtk4::DrawingArea::new();
    cursor_area.set_hexpand(true);
    cursor_area.set_vexpand(true);

    let cursor_pos: Rc<RefCell<f64>> = Rc::new(RefCell::new(0.0));

    {
        let pos_ref = Rc::clone(&cursor_pos);
        cursor_area.set_draw_func(move |_area, cr, width, height| {
            let pos = *pos_ref.borrow();
            if pos > 0.0 {
                let cx = (pos * width as f64).clamp(0.0, width as f64 - 1.0);
                cr.set_source_rgba(1.0, 1.0, 1.0, 0.85);
                cr.set_line_width(2.0);
                cr.move_to(cx, 0.0);
                cr.line_to(cx, height as f64);
                cr.stroke().unwrap_or(());
            }
        });
    }

    // Stack waterfall + cursor using gtk4::Overlay.
    let wf_overlay = gtk4::Overlay::new();
    wf_overlay.set_child(Some(&waterfall_area));
    wf_overlay.add_overlay(&cursor_area);
    wf_overlay.set_hexpand(true);

    // If no waterfall data provided, compute in background thread.
    if wf_data.borrow().is_none() {
        let (wf_tx, wf_rx) = std::sync::mpsc::channel::<WfColumns>();
        let wf_audio_path = audio_path.to_path_buf();
        std::thread::spawn(move || {
            // Try cache first.
            match crate::record::audio::load_waterfall(&wf_audio_path) {
                Ok(result) => {
                    let _ = wf_tx.send(result);
                }
                Err(e) => {
                    log::warn!("waterfall: {}: {}", wf_audio_path.display(), e);
                }
            }
        });

        let wf_data_ref = Rc::clone(&wf_data);
        let wf_area_ref = waterfall_area.clone();
        glib::timeout_add_local(std::time::Duration::from_millis(50), move || {
            match wf_rx.try_recv() {
                Ok(result) => {
                    *wf_data_ref.borrow_mut() = Some(result);
                    wf_area_ref.queue_draw();
                    glib::ControlFlow::Break
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => glib::ControlFlow::Break,
            }
        });
    }

    play_bar.append(&wf_overlay);

    // ── Play/Pause button (created early so rewind can reference it) ──
    let play_btn = gtk4::Button::from_icon_name("media-playback-start-symbolic");
    play_btn.set_tooltip_text(Some("Play recording"));
    play_btn.add_css_class("play-btn");
    if player.borrow().is_none() {
        play_btn.set_sensitive(false);
    }

    // ── Rewind button ────────────────────────────────────────
    let rewind_btn = gtk4::Button::from_icon_name("media-skip-backward-symbolic");
    rewind_btn.set_tooltip_text(Some("Rewind to start"));
    rewind_btn.add_css_class("play-btn");
    rewind_btn.set_sensitive(false);

    {
        let player_ref = Rc::clone(player);
        let pos_ref = Rc::clone(&cursor_pos);
        let cursor_ref = cursor_area.clone();
        let active_ref = Rc::clone(active_play_btn);
        let play_btn_ref = play_btn.clone();
        rewind_btn.connect_clicked(move |btn| {
            // Only seek the shared player if this bar owns it.
            let is_active = active_ref
                .borrow()
                .as_ref()
                .is_some_and(|b| b == &play_btn_ref);
            if is_active {
                if let Some(ref p) = *player_ref.borrow() {
                    p.seek(0.0);
                }
            }
            *pos_ref.borrow_mut() = 0.0;
            btn.set_sensitive(false);
            cursor_ref.queue_draw();
        });
    }

    play_bar.append(&rewind_btn);

    let playing_flag: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));

    {
        let player_ref = Rc::clone(player);
        let audio = audio_path.to_path_buf();
        let pos_ref = Rc::clone(&cursor_pos);
        let flag = Rc::clone(&playing_flag);
        let active_ref = Rc::clone(active_play_btn);
        play_btn.connect_clicked(move |btn| {
            let player_guard = player_ref.borrow();
            let Some(ref p) = *player_guard else {
                return;
            };

            if *flag.borrow() {
                // Pause
                p.pause();
                *flag.borrow_mut() = false;
                btn.set_icon_name("media-playback-start-symbolic");
                btn.set_tooltip_text(Some("Resume playback"));
            } else if p.is_paused() || (p.has_audio() && !p.is_finished()) {
                // Check if this is the active button — if not, start fresh
                let is_active = active_ref.borrow().as_ref().is_some_and(|b| b == btn);
                if is_active {
                    // Resume
                    p.resume();
                    *flag.borrow_mut() = true;
                    btn.set_icon_name("media-playback-pause-symbolic");
                    btn.set_tooltip_text(Some("Pause playback"));
                } else {
                    // Different row — start fresh
                    reset_previous_play_btn(&active_ref);
                    let pos = *pos_ref.borrow();
                    if let Err(e) = p.play(&audio) {
                        log::warn!("failed to play audio: {}", e);
                        return;
                    }
                    if pos > 0.0 && pos < 1.0 {
                        p.seek(pos);
                    }
                    *flag.borrow_mut() = true;
                    *active_ref.borrow_mut() = Some(btn.clone());
                    btn.set_icon_name("media-playback-pause-symbolic");
                    btn.set_tooltip_text(Some("Pause playback"));
                }
            } else {
                // Start fresh
                reset_previous_play_btn(&active_ref);
                let pos = *pos_ref.borrow();
                if let Err(e) = p.play(&audio) {
                    log::warn!("failed to play audio: {}", e);
                    return;
                }
                if pos > 0.0 && pos < 1.0 {
                    p.seek(pos);
                }
                *flag.borrow_mut() = true;
                *active_ref.borrow_mut() = Some(btn.clone());
                btn.set_icon_name("media-playback-pause-symbolic");
                btn.set_tooltip_text(Some("Pause playback"));
            }
        });
    }

    // Playback progress poller (16 ms = ~60 fps cursor).
    {
        let player_poll = Rc::clone(player);
        let btn_poll = play_btn.clone();
        let rewind_poll = rewind_btn.clone();
        let pos_poll = Rc::clone(&cursor_pos);
        let cursor_poll = cursor_area.clone();
        let flag_poll = Rc::clone(&playing_flag);
        let active_poll = Rc::clone(active_play_btn);

        let interp: Rc<RefCell<(f64, std::time::Instant)>> =
            Rc::new(RefCell::new((0.0, std::time::Instant::now())));

        glib::timeout_add_local(std::time::Duration::from_millis(16), move || {
            // Only poll if this bar's play button is the active one.
            let is_active = active_poll
                .borrow()
                .as_ref()
                .is_some_and(|b| b == &btn_poll);
            if !is_active && *flag_poll.borrow() {
                // Another bar took over — reset our state.
                *flag_poll.borrow_mut() = false;
                btn_poll.set_icon_name("media-playback-start-symbolic");
                btn_poll.set_tooltip_text(Some("Play recording"));
                return glib::ControlFlow::Continue;
            }

            let player_guard = player_poll.borrow();
            let Some(ref p) = *player_guard else {
                return glib::ControlFlow::Continue;
            };
            if *flag_poll.borrow() {
                if p.is_finished() {
                    *pos_poll.borrow_mut() = 0.0;
                    *flag_poll.borrow_mut() = false;
                    *interp.borrow_mut() = (0.0, std::time::Instant::now());
                    btn_poll.set_icon_name("media-playback-start-symbolic");
                    btn_poll.set_tooltip_text(Some("Play recording"));
                    rewind_poll.set_sensitive(false);
                } else {
                    let raw = p.progress();
                    let now = std::time::Instant::now();
                    let mut st = interp.borrow_mut();
                    if (raw - st.0).abs() > 1e-9 {
                        *st = (raw, now);
                    }
                    let dur = p.duration_secs();
                    let interpolated = if dur > 0.0 {
                        let elapsed = now.duration_since(st.1).as_secs_f64();
                        (st.0 + elapsed / dur).min(1.0)
                    } else {
                        raw
                    };
                    drop(st);

                    *pos_poll.borrow_mut() = interpolated;
                    rewind_poll.set_sensitive(interpolated > 0.0);
                }
                cursor_poll.queue_draw();
            } else if is_active && p.has_audio() && !p.is_finished() {
                let progress = p.progress();
                rewind_poll.set_sensitive(progress > 0.0);
                *interp.borrow_mut() = (progress, std::time::Instant::now());
            }
            glib::ControlFlow::Continue
        });
    }

    play_bar.append(&play_btn);

    // ── Drag-to-seek on the waterfall ────────────────────────
    {
        let player_drag = Rc::clone(player);
        let pos_drag = Rc::clone(&cursor_pos);
        let cursor_drag = cursor_area.clone();
        let btn_drag = play_btn.clone();
        let rewind_drag = rewind_btn.clone();
        let flag_drag = Rc::clone(&playing_flag);
        let active_drag = Rc::clone(active_play_btn);
        let audio_drag = audio_path.to_path_buf();
        let was_playing = Rc::new(RefCell::new(false));

        let drag = gtk4::GestureDrag::new();

        {
            let player_ref = Rc::clone(&player_drag);
            let pos_ref = Rc::clone(&pos_drag);
            let cursor_ref = cursor_drag.clone();
            let btn_ref = btn_drag.clone();
            let flag_ref = Rc::clone(&flag_drag);
            let was_ref = Rc::clone(&was_playing);
            let active_ref = Rc::clone(&active_drag);
            let audio_ref = audio_drag.clone();
            drag.connect_drag_begin(move |gesture, x, _y| {
                let player_guard = player_ref.borrow();
                let Some(ref p) = *player_guard else {
                    return;
                };
                // If no audio is loaded for this file, load it now.
                if !p.has_audio() {
                    if let Err(e) = p.play(&audio_ref) {
                        log::warn!("failed to load audio for seek: {}", e);
                        return;
                    }
                    p.pause();
                    *active_ref.borrow_mut() = Some(btn_ref.clone());
                }
                let playing = *flag_ref.borrow() && !p.is_finished();
                *was_ref.borrow_mut() = playing;
                if playing {
                    p.pause();
                    *flag_ref.borrow_mut() = false;
                    btn_ref.set_icon_name("media-playback-start-symbolic");
                    btn_ref.set_tooltip_text(Some("Resume playback"));
                }
                if let Some(area) = gesture.widget().downcast_ref::<gtk4::DrawingArea>() {
                    let w = area.width() as f64;
                    if w > 0.0 {
                        let frac = (x / w).clamp(0.0, 1.0);
                        p.seek(frac);
                        *pos_ref.borrow_mut() = frac;
                        cursor_ref.queue_draw();
                    }
                }
            });
        }

        {
            let player_ref = Rc::clone(&player_drag);
            let pos_ref = Rc::clone(&pos_drag);
            let cursor_ref = cursor_drag.clone();
            let rewind_ref = rewind_drag.clone();
            drag.connect_drag_update(move |gesture, offset_x, _offset_y| {
                let player_guard = player_ref.borrow();
                let Some(ref p) = *player_guard else {
                    return;
                };
                if let Some(area) = gesture.widget().downcast_ref::<gtk4::DrawingArea>() {
                    let w = area.width() as f64;
                    if w > 0.0 {
                        let (start_x, _) = gesture.start_point().unwrap_or((0.0, 0.0));
                        let frac = ((start_x + offset_x) / w).clamp(0.0, 1.0);
                        p.seek(frac);
                        *pos_ref.borrow_mut() = frac;
                        rewind_ref.set_sensitive(frac > 0.0);
                        cursor_ref.queue_draw();
                    }
                }
            });
        }

        {
            let player_ref = Rc::clone(&player_drag);
            let btn_ref = btn_drag;
            let flag_ref = Rc::clone(&flag_drag);
            let was_ref = Rc::clone(&was_playing);
            drag.connect_drag_end(move |_gesture, _offset_x, _offset_y| {
                let player_guard = player_ref.borrow();
                let Some(ref p) = *player_guard else {
                    return;
                };
                if *was_ref.borrow() {
                    p.resume();
                    *flag_ref.borrow_mut() = true;
                    btn_ref.set_icon_name("media-playback-pause-symbolic");
                    btn_ref.set_tooltip_text(Some("Pause playback"));
                }
            });
        }

        cursor_area.add_controller(drag);
    }

    play_bar
}

/// Reset the previously active play button to its default state.
fn reset_previous_play_btn(active: &Rc<RefCell<Option<gtk4::Button>>>) {
    use gtk4::prelude::*;
    if let Some(prev) = active.borrow_mut().take() {
        prev.set_icon_name("media-playback-start-symbolic");
        prev.set_tooltip_text(Some("Play recording"));
    }
}
