//! Live transcription text display overlay.
//!
//! Displays a text panel below the recording badge using X11
//! `put_image` for efficient rendering at 60 fps.
//!
//! - **Text** (below badge): live transcription text as words arrive,
//!   plus TTL status/error messages.
//!
//! Audio visualization (amplitude, spectrum, waterfall) has moved into
//! the recording badge itself (see `overlay.rs`).

use super::render_util::{apply_rounded_shape, lerp_color, PixelBuffer};
use crate::error::TalkError;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::*;
use x11rb::COPY_DEPTH_FROM_PARENT;

// ── Layout constants ─────────────────────────────────────────────────

/// Vertical offset from the monitor top edge (matches badge placement).
const TOP_OFFSET: i16 = 4;
/// Target frames per second for the render loop.
const FPS: u32 = 60;

/// Height of the recording badge (must match overlay.rs BADGE_H).
const BADGE_H: u16 = 52;

/// Width of the text overlay in pixels.
const TEXT_W: u16 = 1200;
/// Height of a single text line in pixels.
const TEXT_LINE_H: u16 = 44;
/// Maximum number of text lines (TTL messages + live text).
const MAX_TEXT_LINES: u16 = 5;
/// Vertical gap between badge bottom and text overlay top.
const TEXT_GAP: i16 = 4;
/// Font size for transcription text in pixels.
const TEXT_FONT_SIZE: f32 = 30.0;
/// Corner radius for the text overlay background in pixels.
const TEXT_CORNER_RADIUS: usize = 10;

// ── Colors (BGRA for little-endian ZPixmap, depth 24/32) ─────────────

const BG: [u8; 4] = [0x00, 0x00, 0x00, 0xFF]; // #000000 black
const TEXT_COLOR: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF]; // #ffffff white
/// Reddish colour for error/status messages (BGRA).
const ERROR_COLOR: [u8; 4] = [0x55, 0x55, 0xFF, 0xFF]; // #ff5555 reddish
/// Brightest dot colour (BGRA).
const DOT_HI: [u8; 4] = [0xCC, 0xCC, 0xCC, 0xFF]; // #cccccc
/// Dimmest dot colour (BGRA).
const DOT_LO: [u8; 4] = [0x33, 0x33, 0x33, 0xFF]; // #333333
/// Full wave-cycle length in frames (90 frames ≈ 1.5 s at 60 fps).
const DOT_CYCLE_FRAMES: f32 = 90.0;

// ── Font loading ─────────────────────────────────────────────────────

/// Well-known system font paths, tried in order.  CJK-capable fonts
/// come first so Chinese / Japanese / Korean text renders correctly.
const FONT_SEARCH_PATHS: &[&str] = &[
    // CJK-capable (Noto Sans CJK covers Latin + CJK + most scripts)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    // Droid fallback (broad Unicode coverage including CJK)
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    // Latin-only fallbacks
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
];

/// Load a system TrueType font for text rendering.
///
/// Searches well-known paths; returns `None` if no font is found.
fn load_system_font() -> Option<fontdue::Font> {
    for path in FONT_SEARCH_PATHS {
        if let Ok(data) = std::fs::read(path) {
            let settings = fontdue::FontSettings {
                collection_index: 0,
                scale: TEXT_FONT_SIZE,
                load_substitutions: true,
            };
            match fontdue::Font::from_bytes(data, settings) {
                Ok(font) => {
                    log::debug!("loaded font from {}", path);
                    return Some(font);
                }
                Err(e) => {
                    log::warn!("failed to parse font {}: {}", path, e);
                }
            }
        }
    }
    log::warn!("no system font found, text overlay disabled");
    None
}

/// Rasterise glyphs and return `(glyph_data, total_advance_width)`.
fn rasterise_glyphs(text: &str, font: &fontdue::Font) -> (Vec<(fontdue::Metrics, Vec<u8>)>, usize) {
    let mut glyphs: Vec<(fontdue::Metrics, Vec<u8>)> = Vec::new();
    let mut total_w: usize = 0;
    for ch in text.chars() {
        let (metrics, bitmap) = font.rasterize(ch, TEXT_FONT_SIZE);
        total_w += metrics.advance_width as usize;
        glyphs.push((metrics, bitmap));
    }
    (glyphs, total_w)
}

/// Blit pre-rasterised glyphs into the pixel buffer at `start_x`,
/// vertically centred, clipped to buffer bounds.  `opacity` (0.0–1.0)
/// scales the per-glyph coverage alpha for smooth fade-out.
fn blit_glyphs(
    pb: &mut PixelBuffer,
    glyphs: &[(fontdue::Metrics, Vec<u8>)],
    start_x: i32,
    color: [u8; 4],
    opacity: f32,
) {
    let h = pb.height;
    let w = pb.width;
    let baseline = (h as i32 * 3) / 4;
    let mut cursor_x = start_x;

    for (metrics, bitmap) in glyphs {
        blit_glyph_at(
            pb, metrics, bitmap, cursor_x, baseline, w, h, color, opacity,
        );
        cursor_x += metrics.advance_width as i32;
    }
}

/// Blit a single rasterised glyph at `cursor_x` with the given colour.
/// `opacity` (0.0–1.0) is multiplied into the glyph coverage alpha.
#[allow(clippy::too_many_arguments)]
fn blit_glyph_at(
    pb: &mut PixelBuffer,
    metrics: &fontdue::Metrics,
    bitmap: &[u8],
    cursor_x: i32,
    baseline: i32,
    buf_w: usize,
    buf_h: usize,
    color: [u8; 4],
    opacity: f32,
) {
    let gx = cursor_x + metrics.xmin;
    let gy = baseline - metrics.height as i32 - metrics.ymin;

    for row in 0..metrics.height {
        for col in 0..metrics.width {
            let alpha = bitmap[row * metrics.width + col];
            if alpha == 0 {
                continue;
            }
            let px = gx + col as i32;
            let py = gy + row as i32;
            if px >= 0 && (px as usize) < buf_w && py >= 0 && (py as usize) < buf_h {
                let off = (py as usize * buf_w + px as usize) * 4;
                let a = alpha as f32 / 255.0 * opacity;
                for (c, &fg_val) in color.iter().enumerate().take(3) {
                    let bg_val = pb.data[off + c] as f32;
                    pb.data[off + c] = (bg_val + (fg_val as f32 - bg_val) * a) as u8;
                }
            }
        }
    }
}

/// Compute per-dot brightness for a pulsing wave animation.
///
/// Returns `[f32; 3]` in the range 0.0–1.0 where each dot peaks in
/// sequence, giving a ripple / breathing effect.
fn dot_wave(frame: u64) -> [f32; 3] {
    let phase = frame as f32 * 2.0 * std::f32::consts::PI / DOT_CYCLE_FRAMES;
    let mut out = [0.0f32; 3];
    for (i, val) in out.iter_mut().enumerate() {
        let offset = i as f32 * 2.0 * std::f32::consts::PI / 3.0;
        // sin → [-1, 1] → remap to [0, 1]
        *val = ((phase - offset).sin() + 1.0) / 2.0;
    }
    out
}

/// Render centred text followed by three pulsing dots.
///
/// The text is centred horizontally.  When it overflows the buffer
/// width, leading characters are clipped so the most recent words
/// stay visible.  The three dots always occupy the same fixed width,
/// and each dot's brightness is controlled by `dot_brightnesses`
/// (0.0 = dim, 1.0 = bright).
fn render_text(pb: &mut PixelBuffer, text: &str, font: &fontdue::Font, dot_brightnesses: [f32; 3]) {
    let w = pb.width;
    let h = pb.height;
    let baseline = (h as i32 * 3) / 4;

    // ── Rasterise text glyphs ────────────────────────────────────────

    let (text_glyphs, text_w) = if text.is_empty() {
        (Vec::new(), 0)
    } else {
        rasterise_glyphs(text, font)
    };

    // ── Compute fixed dot area width ─────────────────────────────────

    let (dot_metrics, dot_bitmap) = font.rasterize('●', TEXT_FONT_SIZE);
    let space_metrics = font.metrics(' ', TEXT_FONT_SIZE);
    let space_adv = space_metrics.advance_width as usize;
    let dot_adv = dot_metrics.advance_width as usize;
    // Layout: " ● ● ●"  =  space + dot + space + dot + space + dot
    let dot_area_w = space_adv + dot_adv + space_adv + dot_adv + space_adv + dot_adv;

    let total_w = text_w + dot_area_w;

    // ── Determine start X (centred, or clipped on overflow) ──────────

    let (text_start_x, first_visible_glyph, partial_clip) = if total_w <= w {
        let x0 = (w - total_w) as i32 / 2;
        (x0, 0usize, 0i32)
    } else {
        let skip_px = total_w - w;
        let mut skipped = 0usize;
        let mut first = 0usize;
        for (i, (m, _)) in text_glyphs.iter().enumerate() {
            let adv = m.advance_width as usize;
            if skipped + adv > skip_px {
                first = i;
                break;
            }
            skipped += adv;
            first = i + 1;
        }
        let partial = skip_px.saturating_sub(skipped) as i32;
        (-(partial), first, partial)
    };
    let _ = partial_clip; // used implicitly in text_start_x

    // ── Blit text ────────────────────────────────────────────────────

    let visible_glyphs = &text_glyphs[first_visible_glyph..];
    blit_glyphs(pb, visible_glyphs, text_start_x, TEXT_COLOR, 1.0);

    let visible_text_w: usize = visible_glyphs
        .iter()
        .map(|(m, _)| m.advance_width as usize)
        .sum();
    let mut cursor_x = text_start_x + visible_text_w as i32;

    // ── Blit three dots with individual brightness ───────────────────

    cursor_x += space_adv as i32; // leading space before first dot
    for (i, &brightness) in dot_brightnesses.iter().enumerate() {
        let color = lerp_color(DOT_LO, DOT_HI, brightness);
        blit_glyph_at(
            pb,
            &dot_metrics,
            &dot_bitmap,
            cursor_x,
            baseline,
            w,
            h,
            color,
            1.0,
        );
        cursor_x += dot_adv as i32;
        if i < 2 {
            cursor_x += space_adv as i32;
        }
    }
}

/// Render centred text in a given colour **without** trailing dots.
///
/// Used for TTL status / error messages that should remain static and
/// visually distinct from the live transcription line.
fn render_text_status(
    pb: &mut PixelBuffer,
    text: &str,
    font: &fontdue::Font,
    color: [u8; 4],
    opacity: f32,
) {
    let w = pb.width;

    let (glyphs, text_w) = if text.is_empty() {
        return;
    } else {
        rasterise_glyphs(text, font)
    };

    // Centre horizontally; clip leading chars on overflow.
    let (start_x, first_visible) = if text_w <= w {
        ((w - text_w) as i32 / 2, 0usize)
    } else {
        let skip_px = text_w - w;
        let mut skipped = 0usize;
        let mut first = 0usize;
        for (i, (m, _)) in glyphs.iter().enumerate() {
            let adv = m.advance_width as usize;
            if skipped + adv > skip_px {
                first = i;
                break;
            }
            skipped += adv;
            first = i + 1;
        }
        let partial = skip_px.saturating_sub(skipped) as i32;
        (-(partial), first)
    };

    blit_glyphs(pb, &glyphs[first_visible..], start_x, color, opacity);
}

// ── Public API ───────────────────────────────────────────────────────

/// Default time-to-live for pushed status messages (seconds).
const MESSAGE_TTL_SECS: f32 = 3.0;
/// Seconds a TTL message stays fully opaque before starting to fade.
const MESSAGE_FADE_DELAY_SECS: f32 = 1.0;

enum VizCommand {
    /// Show the text panel.
    Show,
    /// Hide (no-op — text panel self-manages based on pending messages).
    Hide,
    /// Update the live transcription text displayed below the badge.
    Text { text: String },
    /// Push a status message with its own TTL.  Messages stack
    /// vertically and disappear individually when their TTL expires.
    PushMessage { text: String },
    /// Shut down the thread.
    Quit,
}

/// Handle to the visualizer background thread.
///
/// The thread owns its own X11 connection.  Sending
/// [`show`](VisualizerHandle::show) positions the text panel relative
/// to the recording badge; [`hide`](VisualizerHandle::hide) is a
/// no-op (the text panel self-manages).  Dropping the handle stops
/// the thread.
pub struct VisualizerHandle {
    tx: std::sync::mpsc::Sender<VizCommand>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl VisualizerHandle {
    /// Spawn the visualizer thread (text panel only).
    ///
    /// Audio visualization has moved into the overlay badge.  The
    /// visualizer thread now only handles the text panel (status
    /// messages and live transcription text below the badge).
    ///
    /// * `_text` — reserved for future use; text panel is always active.
    ///
    /// Returns `Err` if X11 is unreachable.
    pub fn new(_text: bool) -> Result<Self, TalkError> {
        let geom = super::monitor::primary_monitor_geometry()?;

        let (tx, rx) = std::sync::mpsc::channel();

        let thread = std::thread::Builder::new()
            .name("visualizer".into())
            .spawn(move || {
                if let Err(e) = visualizer_thread(rx, geom) {
                    log::error!("visualizer thread error: {}", e);
                }
            })
            .map_err(|e| TalkError::Audio(format!("failed to spawn visualizer thread: {}", e)))?;

        Ok(Self {
            tx,
            thread: Some(thread),
        })
    }

    /// Show the text panel.
    pub fn show(&self, _badge_width: u16) {
        let _ = self.tx.send(VizCommand::Show);
    }

    /// Hide the text panel (no-op — text panel self-manages visibility).
    pub fn hide(&self) {
        let _ = self.tx.send(VizCommand::Hide);
    }

    /// Update the live transcription text shown below the badge.
    pub fn set_text(&self, text: &str) {
        let _ = self.tx.send(VizCommand::Text {
            text: text.to_string(),
        });
    }

    /// Push a status message that stays visible for ~2 s then fades.
    /// Multiple messages stack vertically; each has its own TTL.
    pub fn push_message(&self, text: &str) {
        let _ = self.tx.send(VizCommand::PushMessage {
            text: text.to_string(),
        });
    }
}

impl Drop for VisualizerHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(VizCommand::Quit);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

// ── Thread implementation ────────────────────────────────────────────

/// Main visualizer thread (text panel only).
///
/// Owns an X11 connection for rendering the text panel below the
/// recording badge.  Blocks on commands when idle; renders at [`FPS`]
/// when there are messages to display.
fn visualizer_thread(
    rx: std::sync::mpsc::Receiver<VizCommand>,
    geom: super::monitor::MonitorGeometry,
) -> Result<(), TalkError> {
    // ── X11 connection ───────────────────────────────────────────────

    let (conn, screen_num) = x11rb::connect(None)
        .map_err(|e| TalkError::Config(format!("visualizer X11 connect: {}", e)))?;

    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;
    let depth = screen.root_depth;

    let (mon_x, mon_y, mon_w, _mon_h) = geom;

    // ── Persistent text window (lives for the entire thread) ────────
    //
    // Created once here and never destroyed until the thread exits.
    // The render loop maps/unmaps it on demand based on whether there
    // are messages to display.

    let text_x = mon_x + (mon_w as i16 / 2) - (TEXT_W as i16 / 2);
    let text_y = mon_y + TOP_OFFSET + BADGE_H as i16 + TEXT_GAP;

    let text_win = {
        let values = CreateWindowAux::new()
            .background_pixel(screen.black_pixel)
            .border_pixel(0)
            .override_redirect(1u32)
            .event_mask(EventMask::EXPOSURE);

        let win = conn
            .generate_id()
            .map_err(|e| TalkError::Config(format!("X11 id: {}", e)))?;

        conn.create_window(
            COPY_DEPTH_FROM_PARENT,
            win,
            root,
            text_x,
            text_y,
            TEXT_W,
            TEXT_LINE_H * MAX_TEXT_LINES,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &values,
        )
        .map_err(|e| TalkError::Config(format!("X11 create text win: {}", e)))?;

        apply_rounded_shape(&conn, win, TEXT_W, TEXT_LINE_H, TEXT_CORNER_RADIUS)?;

        win
    };
    let text_gc = {
        let gc = conn
            .generate_id()
            .map_err(|e| TalkError::Config(format!("X11 id: {}", e)))?;
        conn.create_gc(gc, text_win, &CreateGCAux::new())
            .map_err(|e| TalkError::Config(format!("X11 gc: {}", e)))?;
        gc
    };
    let _ = conn.flush();

    // ── Pixel buffer ─────────────────────────────────────────────────

    let mut text_pb = PixelBuffer::new(TEXT_W as usize, TEXT_LINE_H as usize);

    // ── Font (loaded in background to avoid blocking startup) ────────

    let font_rx = {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("viz-font-loader".into())
            .spawn(move || {
                let _ = tx.send(load_system_font());
            })
            .map_err(|e| TalkError::Config(format!("font loader thread: {}", e)))?;
        rx
    };
    let mut font: Option<fontdue::Font> = None;

    // ── State ────────────────────────────────────────────────────────

    let frame_dur = std::time::Duration::from_micros(1_000_000 / FPS as u64);

    let mut is_showing = false;
    let mut current_text = String::new();
    let mut text_is_mapped = false;
    // TTL status messages: each entry is (text, expiry instant).
    let mut ttl_messages: Vec<(String, std::time::Instant)> = Vec::new();
    // Last rendered line count — used to avoid redundant window resizes.
    let mut prev_text_lines: u16 = 0;
    let mut frame_counter: u64 = 0;
    let mut should_quit = false;

    // ── Event loop ───────────────────────────────────────────────────
    //
    // Two states:
    //   1. Text active (`is_showing` or TTL messages pending): render at
    //      60 fps so fade animation stays smooth.
    //   2. Idle (`!is_showing`, no TTL messages): block on `rx.recv()`.

    loop {
        // Idle: no pending messages → block.
        if !is_showing && ttl_messages.is_empty() {
            if should_quit {
                break;
            }
            match rx.recv() {
                Ok(VizCommand::Show) => {
                    is_showing = true;
                }
                Ok(VizCommand::Hide) => {}
                Ok(VizCommand::Text { text }) => {
                    current_text = text;
                }
                Ok(VizCommand::PushMessage { text }) => {
                    let expires = std::time::Instant::now()
                        + std::time::Duration::from_secs_f32(MESSAGE_TTL_SECS);
                    ttl_messages.push((text, expires));
                    // Fall through to render loop on next iteration.
                }
                Ok(VizCommand::Quit) | Err(_) => {
                    should_quit = true;
                }
            }
            continue;
        }

        // Active: drain commands without blocking.
        // Skip when quit is already requested — the channel is
        // disconnected and every try_recv would return Disconnected.
        if !should_quit {
            loop {
                match rx.try_recv() {
                    Ok(VizCommand::Hide) => {
                        is_showing = false;
                    }
                    Ok(VizCommand::Text { text }) => {
                        current_text = text;
                    }
                    Ok(VizCommand::PushMessage { text }) => {
                        let expires = std::time::Instant::now()
                            + std::time::Duration::from_secs_f32(MESSAGE_TTL_SECS);
                        ttl_messages.push((text, expires));
                    }
                    Ok(VizCommand::Show) => {
                        is_showing = true;
                    }
                    Ok(VizCommand::Quit) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        is_showing = false;
                        should_quit = true;
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                }
            }
        }

        if should_quit && ttl_messages.is_empty() {
            break;
        }

        if !is_showing && ttl_messages.is_empty() {
            // No pending messages — go idle.
            continue;
        }

        // Pick up the font once the background loader finishes.
        if font.is_none() {
            if let Ok(loaded) = font_rx.try_recv() {
                font = loaded;
            }
        }

        // ── Render frame ─────────────────────────────────────────────

        let frame_start = std::time::Instant::now();

        // Prune expired TTL messages.
        let now_ttl = std::time::Instant::now();
        ttl_messages.retain(|(_, expires)| *expires > now_ttl);

        // Collect all lines to render: TTL messages (newest first), then live text.
        // Each entry carries a fade alpha (1.0 = fully visible).
        let fade_duration = MESSAGE_TTL_SECS - MESSAGE_FADE_DELAY_SECS;
        let mut text_lines: Vec<(&str, f32)> = ttl_messages
            .iter()
            .rev()
            .map(|(t, expires)| {
                let remaining = expires.duration_since(now_ttl).as_secs_f32();
                let elapsed = MESSAGE_TTL_SECS - remaining;
                let alpha = if elapsed < MESSAGE_FADE_DELAY_SECS {
                    1.0
                } else {
                    ((MESSAGE_TTL_SECS - elapsed) / fade_duration).clamp(0.0, 1.0)
                };
                (t.as_str(), alpha)
            })
            .collect();
        let mut n_ttl = text_lines.len();
        if !current_text.is_empty() {
            text_lines.push((&current_text, 1.0));
        }
        // Cap to max lines (keep most recent).
        if text_lines.len() > MAX_TEXT_LINES as usize {
            let drop = text_lines.len() - MAX_TEXT_LINES as usize;
            text_lines = text_lines.split_off(drop);
            n_ttl = n_ttl.saturating_sub(drop);
        }
        let n_lines = text_lines.len() as u16;

        // On-demand text panel visibility + dynamic height.
        {
            let want_visible = n_lines > 0;
            if want_visible && n_lines != prev_text_lines {
                let new_h = n_lines.max(1) * TEXT_LINE_H;
                let _ = conn.configure_window(
                    text_win,
                    &x11rb::protocol::xproto::ConfigureWindowAux::new().height(u32::from(new_h)),
                );
                let _ = apply_rounded_shape(&conn, text_win, TEXT_W, new_h, TEXT_CORNER_RADIUS);
                prev_text_lines = n_lines;
            }
            if want_visible && !text_is_mapped {
                let _ = conn.map_window(text_win);
                text_is_mapped = true;
            } else if !want_visible && text_is_mapped {
                let _ = conn.unmap_window(text_win);
                text_is_mapped = false;
                prev_text_lines = 0;
            }
        }

        // Text panel — render each line into its own row.
        // TTL status lines use reddish colour without dots; the live
        // transcription line (if any) uses white with animated dots.
        if let Some(ref f) = font {
            let dot_br = dot_wave(frame_counter);
            for (i, (line, alpha)) in text_lines.iter().enumerate() {
                text_pb.clear_rounded(BG, TEXT_CORNER_RADIUS);
                if i < n_ttl {
                    render_text_status(&mut text_pb, line, f, ERROR_COLOR, *alpha);
                } else {
                    render_text(&mut text_pb, line, f, dot_br);
                }

                let _ = conn.put_image(
                    ImageFormat::Z_PIXMAP,
                    text_win,
                    text_gc,
                    TEXT_W,
                    TEXT_LINE_H,
                    0,
                    i as i16 * TEXT_LINE_H as i16,
                    0,
                    depth,
                    &text_pb.data,
                );
            }
        }
        frame_counter += 1;

        let _ = conn.flush();

        let elapsed = frame_start.elapsed();
        if elapsed < frame_dur {
            std::thread::sleep(frame_dur - elapsed);
        }
    }

    // Clean up the persistent text window.
    let _ = conn.free_gc(text_gc);
    let _ = conn.destroy_window(text_win);
    let _ = conn.flush();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Text rendering ────────────────────────────────────────────────

    #[test]
    fn render_text_no_panic_with_font() {
        if let Some(font) = load_system_font() {
            let mut pb = PixelBuffer::new(600, 36);
            pb.clear(BG);
            render_text(&mut pb, "Hello world", &font, [0.5, 0.5, 0.5]);

            let non_bg = pb.data.chunks_exact(4).filter(|p| *p != BG).count();
            assert!(non_bg > 0, "text should produce non-background pixels");
        }
        // If no font is available, the test passes trivially.
    }

    #[test]
    fn render_text_dots_always_present() {
        if let Some(font) = load_system_font() {
            let mut pb = PixelBuffer::new(600, 36);
            pb.clear(BG);
            render_text(&mut pb, "", &font, [1.0, 1.0, 1.0]);

            let non_bg = pb.data.chunks_exact(4).filter(|p| *p != BG).count();
            assert!(non_bg > 0, "dots alone should produce pixels");
        }
    }

    #[test]
    fn render_text_dots_dim_still_visible() {
        if let Some(font) = load_system_font() {
            let mut pb = PixelBuffer::new(600, 36);
            pb.clear(BG);
            render_text(&mut pb, "", &font, [0.0, 0.0, 0.0]);

            // Even at brightness 0.0, DOT_LO (#333333) is not BG (#000000)
            let non_bg = pb.data.chunks_exact(4).filter(|p| *p != BG).count();
            assert!(non_bg > 0, "dim dots should still differ from BG");
        }
    }

    #[test]
    fn dot_wave_returns_valid_range() {
        for frame in 0..200 {
            let vals = dot_wave(frame);
            for v in &vals {
                assert!(
                    (0.0..=1.0).contains(v),
                    "dot_wave({}) produced {} which is out of [0,1]",
                    frame,
                    v
                );
            }
        }
    }

    #[test]
    fn render_text_overflow_clips_left() {
        if let Some(font) = load_system_font() {
            // Narrow buffer — long text + dots should clip from the left.
            let mut pb = PixelBuffer::new(80, 36);
            pb.clear(BG);
            render_text(
                &mut pb,
                "This is a very long sentence that will not fit",
                &font,
                [0.8, 0.5, 0.2],
            );
        }
    }
}
