//! Shared audio player bar widget with waterfall spectrogram,
//! cursor overlay, drag-to-seek, and play/pause/rewind controls.
//!
//! Used by the record UI.  A bar owns no background work of its own:
//! its waterfall comes from the bounded [`waterfall_loader`] and its
//! playback (decode, progress tick, release when another row plays)
//! from the window's [`PlaybackSession`], so a thousand bars cost no
//! more timers or threads than one.

use super::playback_session::{PlaybackRow, PlaybackSession, RowRef};
use super::waterfall_loader::{self, Interest};
use gtk4::glib;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

pub(crate) use super::waterfall_loader::WfColumns;

/// The bar's waterfall: its columns once known, and until then the
/// row's interest in the queued computation (dropped with the bar,
/// which cancels a computation nobody will see).
struct Waterfall {
    data: Option<WfColumns>,
    interest: Option<Interest>,
}

/// A button face with two looks (idle / active) that switches by
/// opacity instead of swapping the button's child.
///
/// Replacing a button's child (`set_icon_name`, `set_label`) queues a
/// relayout of the whole window; in a list of a thousand rows that
/// costs the GTK thread hundreds of milliseconds per click.  Both
/// looks are built once and stacked in an overlay; changing opacity
/// only queues a redraw of the button.
pub(crate) struct ButtonFaces<W: glib::object::IsA<gtk4::Widget>> {
    idle: W,
    active: W,
}

impl<W: glib::object::IsA<gtk4::Widget>> ButtonFaces<W> {
    /// Install `idle` (shown) and `active` (hidden) as `button`'s face.
    pub(crate) fn install(button: &gtk4::Button, idle: W, active: W) -> Self {
        use gtk4::prelude::*;
        active.set_opacity(0.0);
        let overlay = gtk4::Overlay::new();
        overlay.set_child(Some(&idle));
        overlay.add_overlay(&active);
        button.set_child(Some(&overlay));
        Self { idle, active }
    }

    /// Show the active look (`true`) or the idle one.
    pub(crate) fn set_active(&self, active: bool) {
        use gtk4::prelude::*;
        self.idle.set_opacity(if active { 0.0 } else { 1.0 });
        self.active.set_opacity(if active { 1.0 } else { 0.0 });
    }
}

/// The play/pause button of a bar: an icon button whose two looks
/// never trigger a relayout.
struct PlayButton {
    button: glib::WeakRef<gtk4::Button>,
    faces: ButtonFaces<gtk4::Image>,
}

impl PlayButton {
    fn new() -> (Self, gtk4::Button) {
        use gtk4::prelude::*;
        let button = gtk4::Button::new();
        button.add_css_class("image-button");
        button.add_css_class("play-btn");
        let faces = ButtonFaces::install(
            &button,
            gtk4::Image::from_icon_name("media-playback-start-symbolic"),
            gtk4::Image::from_icon_name("media-playback-pause-symbolic"),
        );
        let this = Self {
            button: button.downgrade(),
            faces,
        };
        this.show_idle();
        (this, button)
    }

    fn show_idle(&self) {
        use gtk4::prelude::*;
        self.faces.set_active(false);
        if let Some(button) = self.button.upgrade() {
            button.set_tooltip_text(Some("Play recording"));
        }
    }

    fn show_playing(&self) {
        use gtk4::prelude::*;
        self.faces.set_active(true);
        if let Some(button) = self.button.upgrade() {
            button.set_tooltip_text(Some("Pause playback"));
        }
    }

    fn show_paused(&self) {
        use gtk4::prelude::*;
        self.faces.set_active(false);
        if let Some(button) = self.button.upgrade() {
            button.set_tooltip_text(Some("Resume playback"));
        }
    }
}

/// The widgets the session drives while this bar owns playback.
struct BarRow {
    play: Rc<PlayButton>,
    rewind_btn: glib::WeakRef<gtk4::Button>,
    cursor_pos: Rc<RefCell<f64>>,
    cursor_area: glib::WeakRef<gtk4::DrawingArea>,
}

impl PlaybackRow for BarRow {
    fn set_idle(&self, finished: bool) {
        use gtk4::prelude::*;
        self.play.show_idle();
        if finished {
            *self.cursor_pos.borrow_mut() = 0.0;
            if let Some(button) = self.rewind_btn.upgrade() {
                button.set_sensitive(false);
            }
            if let Some(area) = self.cursor_area.upgrade() {
                area.queue_draw();
            }
        }
    }

    fn set_progress(&self, fraction: f64) {
        use gtk4::prelude::*;
        *self.cursor_pos.borrow_mut() = fraction;
        if let Some(button) = self.rewind_btn.upgrade() {
            button.set_sensitive(fraction > 0.0);
        }
        if let Some(area) = self.cursor_area.upgrade() {
            area.queue_draw();
        }
    }
}

/// Build an interactive audio player bar widget.
///
/// Returns a horizontal `gtk4::Box` containing:
/// - Waterfall spectrogram with playback cursor overlay
/// - Drag-to-seek gesture on the waterfall
/// - Rewind and play/pause buttons
///
/// # Arguments
///
/// * `audio_path` — audio file for playback (WAV or OGG)
/// * `session` — the window's shared playback (player, active row,
///   progress tick)
/// * `waterfall_data` — pre-computed columns, or `None` to queue the
///   computation on the bounded waterfall loader
/// * `height` — content height for the waterfall DrawingArea (pixels)
pub(crate) fn build_audio_player_bar(
    audio_path: &Path,
    session: &Rc<PlaybackSession>,
    waterfall_data: Option<WfColumns>,
    height: i32,
) -> (gtk4::Box, RowRef) {
    use gtk4::prelude::*;

    let play_bar = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
    play_bar.set_hexpand(true);
    crate::perf_counters::incr(crate::perf_counters::Counter::PlayerBarsBuilt);

    // ── Waterfall spectrogram (base layer) ─────────────────────
    let waterfall_area = gtk4::DrawingArea::new();
    waterfall_area.set_hexpand(true);
    waterfall_area.set_vexpand(false);
    waterfall_area.set_content_height(height);
    waterfall_area.add_css_class("waterfall");

    let ready = waterfall_data.is_some();
    let waterfall: Rc<RefCell<Waterfall>> = Rc::new(RefCell::new(Waterfall {
        data: waterfall_data,
        interest: None,
    }));

    {
        let data_ref = Rc::clone(&waterfall);
        waterfall_area.set_draw_func(move |_area, cr, width, height| {
            let w = width as usize;
            let h = height as usize;
            if w == 0 || h == 0 {
                return;
            }
            let waterfall = data_ref.borrow();
            if let Some((ref columns, peak)) = waterfall.data {
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
            if width <= 0 || height <= 0 {
                return;
            }
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

    #[cfg(feature = "perf-counters")]
    if ready {
        waterfall_area.add_css_class("wf-ready");
    }

    // No waterfall data yet: queue it.  The result is applied to this
    // bar if it is still alive; a bar removed meanwhile drops its
    // interest, cancelling the job.
    if !ready {
        let waterfall_weak = Rc::downgrade(&waterfall);
        let area_weak = waterfall_area.downgrade();
        let path = audio_path.to_path_buf();
        let interest = waterfall_loader::request(
            audio_path,
            Box::new(move |result| {
                let (Some(waterfall), Some(area)) = (waterfall_weak.upgrade(), area_weak.upgrade())
                else {
                    return;
                };
                match result {
                    Ok(columns) => {
                        let mut wf = waterfall.borrow_mut();
                        wf.data = Some(columns);
                        wf.interest = None;
                        drop(wf);
                        area.queue_draw();
                        // Harness-only marker the recordings-browser
                        // probe reads to see which rows show their
                        // waveform.
                        #[cfg(feature = "perf-counters")]
                        area.add_css_class("wf-ready");
                        crate::perf_counters::incr(
                            crate::perf_counters::Counter::WaterfallJobsApplied,
                        );
                    }
                    Err(e) => log::warn!("waterfall: {}: {}", path.display(), e),
                }
            }),
        );
        waterfall.borrow_mut().interest = Some(interest);
    }

    play_bar.append(&wf_overlay);

    // ── Play/Pause button (created early so rewind can reference it) ──
    let (play, play_btn) = PlayButton::new();
    let play = Rc::new(play);
    if !session.has_player() {
        play_btn.set_sensitive(false);
    }

    // ── Rewind button ────────────────────────────────────────
    let rewind_btn = gtk4::Button::from_icon_name("media-skip-backward-symbolic");
    rewind_btn.set_tooltip_text(Some("Rewind to start"));
    rewind_btn.add_css_class("play-btn");
    rewind_btn.set_sensitive(false);

    let row: RowRef = Rc::new(BarRow {
        play: Rc::clone(&play),
        rewind_btn: rewind_btn.downgrade(),
        cursor_pos: Rc::clone(&cursor_pos),
        cursor_area: cursor_area.downgrade(),
    });
    if session.adopt(Rc::clone(&row), audio_path) {
        if session.is_paused() {
            play.show_paused();
        } else {
            play.show_playing();
        }
    }

    {
        let session = Rc::clone(session);
        let row = Rc::clone(&row);
        let pos_ref = Rc::clone(&cursor_pos);
        let cursor_ref = cursor_area.downgrade();
        rewind_btn.connect_clicked(move |btn| {
            // Only seek the shared player if this bar owns it.
            if session.is_active(&row) {
                session.seek(0.0);
            }
            *pos_ref.borrow_mut() = 0.0;
            btn.set_sensitive(false);
            if let Some(area) = cursor_ref.upgrade() {
                area.queue_draw();
            }
        });
    }

    play_bar.append(&rewind_btn);

    {
        let session = Rc::clone(session);
        let row = Rc::clone(&row);
        let play = Rc::clone(&play);
        let audio = audio_path.to_path_buf();
        let pos_ref = Rc::clone(&cursor_pos);
        play_btn.connect_clicked(move |_| {
            if session.is_active(&row) {
                if session.is_paused() {
                    session.resume();
                    play.show_playing();
                } else {
                    session.pause();
                    play.show_paused();
                }
            } else {
                // Start fresh (from the cursor, if it was moved); the
                // decode runs on a worker and the session's tick loads
                // it when ready.
                let from = *pos_ref.borrow();
                if session.play(Rc::clone(&row), &audio, Some(from)) {
                    play.show_playing();
                }
            }
        });
    }

    play_bar.append(&play_btn);

    // ── Drag-to-seek on the waterfall ────────────────────────
    {
        let pos_drag = Rc::clone(&cursor_pos);
        let cursor_drag = cursor_area.downgrade();
        let rewind_drag = rewind_btn.downgrade();
        let audio_drag = audio_path.to_path_buf();
        let was_playing = Rc::new(RefCell::new(false));

        let drag = gtk4::GestureDrag::new();

        {
            let session = Rc::clone(session);
            let row = Rc::clone(&row);
            let pos_ref = Rc::clone(&pos_drag);
            let cursor_ref = cursor_drag.clone();
            let play = Rc::clone(&play);
            let was_ref = Rc::clone(&was_playing);
            let audio_ref = audio_drag.clone();
            drag.connect_drag_begin(move |gesture, x, _y| {
                if !session.has_player() {
                    return;
                }
                // If this bar does not own the player, load its audio
                // now (paused) so the seek has something to act on.
                if !session.is_active(&row) {
                    if !session.play(Rc::clone(&row), &audio_ref, None) {
                        return;
                    }
                    session.pause();
                }
                let playing = session.is_playing();
                *was_ref.borrow_mut() = playing;
                if playing {
                    session.pause();
                    play.show_paused();
                }
                if let Some(area) = gesture.widget().downcast_ref::<gtk4::DrawingArea>() {
                    let w = area.width() as f64;
                    if w > 0.0 {
                        let frac = (x / w).clamp(0.0, 1.0);
                        session.seek(frac);
                        *pos_ref.borrow_mut() = frac;
                        if let Some(area) = cursor_ref.upgrade() {
                            area.queue_draw();
                        }
                    }
                }
            });
        }

        {
            let session = Rc::clone(session);
            let row = Rc::clone(&row);
            let pos_ref = Rc::clone(&pos_drag);
            let cursor_ref = cursor_drag.clone();
            let rewind_ref = rewind_drag.clone();
            drag.connect_drag_update(move |gesture, offset_x, _offset_y| {
                if !session.is_active(&row) {
                    return;
                }
                if let Some(area) = gesture.widget().downcast_ref::<gtk4::DrawingArea>() {
                    let w = area.width() as f64;
                    if w > 0.0 {
                        let (start_x, _) = gesture.start_point().unwrap_or((0.0, 0.0));
                        let frac = ((start_x + offset_x) / w).clamp(0.0, 1.0);
                        session.seek(frac);
                        *pos_ref.borrow_mut() = frac;
                        if let Some(button) = rewind_ref.upgrade() {
                            button.set_sensitive(frac > 0.0);
                        }
                        if let Some(area) = cursor_ref.upgrade() {
                            area.queue_draw();
                        }
                    }
                }
            });
        }

        {
            let session = Rc::clone(session);
            let row = Rc::clone(&row);
            let play = Rc::clone(&play);
            let was_ref = Rc::clone(&was_playing);
            drag.connect_drag_end(move |_gesture, _offset_x, _offset_y| {
                if *was_ref.borrow() && session.is_active(&row) {
                    session.resume();
                    play.show_playing();
                }
            });
        }

        cursor_area.add_controller(drag);
    }

    (play_bar, row)
}

#[cfg(all(test, feature = "perf-counters"))]
#[path = "../../tests/perf/support/display.rs"]
mod isolated_display;

#[cfg(all(test, feature = "perf-counters"))]
#[test]
#[ignore = "runs only with the harness's isolated GTK display"]
fn detached_bar_releases_play_and_cursor_widgets() {
    use gtk4::prelude::*;
    let display = isolated_display::IsolatedDisplay::start().expect("isolated display");
    assert!(display.log_dir().exists());
    let prior = std::env::var_os("DISPLAY");
    std::env::set_var("DISPLAY", &display.display);
    std::env::set_var("GDK_BACKEND", "x11");
    gtk4::init().expect("GTK init");
    let session = PlaybackSession::new();
    let (bar, row) = build_audio_player_bar(
        Path::new("unused.ogg"),
        &session,
        Some((
            vec![vec![0.0; crate::x11::render_util::WATERFALL_ROWS]],
            1.0,
        )),
        28,
    );
    let play = bar
        .last_child()
        .and_downcast::<gtk4::Button>()
        .expect("play button");
    let play_weak = play.downgrade();
    let bar_weak = bar.downgrade();
    let root = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    root.append(&bar);
    root.remove(&bar);
    drop(row);
    drop(play);
    drop(bar);
    assert!(
        play_weak.upgrade().is_none(),
        "play button must be reclaimed"
    );
    assert!(bar_weak.upgrade().is_none(), "bar must be reclaimed");
    if let Some(value) = prior {
        std::env::set_var("DISPLAY", value);
    } else {
        std::env::remove_var("DISPLAY");
    }
}
