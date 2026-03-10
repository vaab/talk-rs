//! Visual overlay indicator for recording/transcribing status.
//!
//! Displays a small badge at the top-center of the primary monitor using X11.
//! Uses the Shape extension for binary transparency (works without a compositor).
//! The overlay runs on a dedicated background thread with a command channel.
//!
//! During **recording**, the badge is rendered dynamically at 60 fps:
//! a pulsing red dot (brightness driven by volume) plus a real-time
//! spectrogram waterfall that scrolls right-to-left, replacing the
//! former static "recording" text.
//!
//! During **transcribing**, the badge is a static PNG (unchanged).

use super::render_util::{
    apply_rounded_shape, blit_glyph_at, compute_spectrum, rasterise_glyphs, rms, PixelBuffer,
    RingBuffer, PEAK_DECAY, PEAK_FLOOR,
};
use crate::error::TalkError;
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use x11rb::connection::Connection;
use x11rb::protocol::shape;
use x11rb::protocol::xproto::*;
use x11rb::wrapper::ConnectionExt as _;
use x11rb::COPY_DEPTH_FROM_PARENT;

// ── Embedded PNG assets ──────────────────────────────────────────────

/// "transcribing" badge: 210×52, dark rounded rect with blue dot + text.
const TRANSCRIBING_PNG: &[u8] = include_bytes!("../../assets/transcribing.png");

// ── Badge layout constants ───────────────────────────────────────────

/// Badge width in pixels (~50% wider than original for full-width spectrogram).
pub(crate) const BADGE_W: u16 = 273;
/// Badge height in pixels.
const BADGE_H: u16 = 52;
/// Corner radius for the rounded rectangle background.
const CORNER_RADIUS: usize = 13;

/// X coordinate of the red dot centre (near left edge, over spectrogram).
const DOT_CX: usize = 20;
/// Y coordinate of the red dot centre (vertically centred).
const DOT_CY: usize = 26;
/// Minimum red dot radius (quiet).
const DOT_RADIUS_MIN: f32 = 3.0;
/// Maximum red dot radius (loud).
const DOT_RADIUS_MAX: f32 = 10.0;
/// Minimum dot brightness — always visible.
const DOT_MIN_BRIGHTNESS: f32 = 0.5;
/// Transparent gap (pixels) between red dot edge and spectrogram.
const DOT_GAP: f32 = 2.0;

/// Left edge of the spectrogram area (inside border margin).
const SPEC_LEFT: usize = 4;
/// Top edge of the spectrogram area (padding from badge top).
const SPEC_TOP: usize = 4;
/// Right edge of the spectrogram area (inside border margin).
const SPEC_RIGHT: usize = 269;
/// Bottom edge of the spectrogram area (padding from badge bottom).
const SPEC_BOTTOM: usize = 48;
/// Spectrogram width in pixels (time columns).
const SPEC_W: usize = SPEC_RIGHT - SPEC_LEFT;
/// Spectrogram height in pixels (frequency rows).
const SPEC_H: usize = SPEC_BOTTOM - SPEC_TOP;

/// Target frames per second for the recording render loop.
const FPS: u32 = 60;
/// Number of audio samples fed into the FFT (must be power of two).
const FFT_SIZE: usize = 2048;

/// Lowest frequency shown in the spectrogram (Hz).
const FREQ_MIN: f32 = 80.0;
/// Hard upper limit for dynamic frequency scaling (Hz).
const FREQ_MAX: f32 = 8000.0;
/// Initial effective frequency ceiling; grows as higher harmonics appear.
const FREQ_INITIAL_MAX: f32 = 320.0;
/// Minimum FFT magnitude to count a bin as "active" for frequency scaling.
const FREQ_NOISE_FLOOR: f32 = 0.01;

// ── Colors (BGRA for little-endian ZPixmap, depth 24/32) ────────────

/// Badge background: fully transparent (compositor alpha).
const BG_COLOR: [u8; 4] = [0x00, 0x00, 0x00, 0x00];

/// Border colour: medium gray, fully opaque (BGRA).
const BORDER_COLOR: [u8; 4] = [0x88, 0x88, 0x88, 0xFF];
/// Border width in pixels.
const BORDER_WIDTH: f32 = 2.0;

// ── Public types ─────────────────────────────────────────────────────

/// Which indicator to display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorKind {
    /// Red badge shown during audio recording (dynamic spectrogram).
    Recording,
    /// Blue badge shown during transcription (static PNG).
    Transcribing,
}

/// Commands sent from the main thread to the overlay thread.
enum Command {
    Show(IndicatorKind),
    Hide,
    Quit,
}

/// Handle to the overlay background thread.
///
/// Sending [`show`](OverlayHandle::show) or [`hide`](OverlayHandle::hide)
/// controls the X11 window from any thread. The overlay is destroyed
/// when this handle is dropped.
pub struct OverlayHandle {
    tx: mpsc::Sender<Command>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl OverlayHandle {
    /// Spawn the overlay thread and open an X11 connection.
    ///
    /// * `viz` — visualizer mode to render inside the badge (or `None`
    ///   for a plain badge with only the red dot).
    /// * `mono` — use monochrome colours (theme-aware).
    /// * `audio_ring` — shared ring buffer fed by the audio tee task.
    /// * `sample_rate` — capture sample rate (for FFT/RMS sizing).
    /// * `silence_tx` — optional channel to notify when silence is
    ///   detected (`true`) or audio returns (`false`).
    ///
    /// Monitor geometry is queried via GDK4 **before** spawning the
    /// thread (GDK must be called from the main thread) and passed in.
    ///
    /// Returns `Err` if the X11 display cannot be opened.
    pub fn new(
        viz: Option<crate::config::VizMode>,
        mono: bool,
        audio_ring: Arc<Mutex<RingBuffer>>,
        sample_rate: u32,
        silence_tx: Option<std::sync::mpsc::Sender<bool>>,
    ) -> Result<Self, TalkError> {
        let geom = super::monitor::primary_monitor_geometry()?;

        // Resolve monochrome palette up front (D-Bus on main thread).
        let mono_palette = if mono {
            let (fg, bg) = super::render_util::monochrome_palette();
            log::info!("monochrome overlay: fg={:?} bg={:?}", fg, bg);
            Some((fg, bg))
        } else {
            None
        };

        let (tx, rx) = mpsc::channel();

        let thread = std::thread::Builder::new()
            .name("overlay".into())
            .spawn(move || {
                if let Err(e) = overlay_thread(
                    rx,
                    geom,
                    viz,
                    mono_palette,
                    audio_ring,
                    sample_rate,
                    silence_tx,
                ) {
                    log::error!("overlay thread error: {}", e);
                }
            })
            .map_err(|e| TalkError::Audio(format!("failed to spawn overlay thread: {}", e)))?;

        Ok(Self {
            tx,
            thread: Some(thread),
        })
    }

    /// Display the indicator badge.
    pub fn show(&self, kind: IndicatorKind) {
        let _ = self.tx.send(Command::Show(kind));
    }

    /// Hide the indicator badge.
    pub fn hide(&self) {
        let _ = self.tx.send(Command::Hide);
    }
}

impl Drop for OverlayHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Quit);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

// ── PNG decoding ─────────────────────────────────────────────────────

/// Decoded RGBA image.
struct RgbaImage {
    width: u32,
    height: u32,
    /// Row-major RGBA pixels (4 bytes per pixel).
    data: Vec<u8>,
}

/// Decode an embedded PNG into RGBA pixels.
fn decode_png(bytes: &[u8]) -> Result<RgbaImage, TalkError> {
    let decoder = png::Decoder::new(bytes);
    let mut reader = decoder
        .read_info()
        .map_err(|e| TalkError::Config(format!("failed to read PNG header: {}", e)))?;

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| TalkError::Config(format!("failed to decode PNG frame: {}", e)))?;

    let data = match info.color_type {
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgb => {
            let rgb = &buf[..info.buffer_size()];
            let mut rgba = Vec::with_capacity(info.width as usize * info.height as usize * 4);
            for chunk in rgb.chunks_exact(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            rgba
        }
        other => {
            return Err(TalkError::Config(format!(
                "unsupported PNG color type: {:?}",
                other
            )));
        }
    };

    Ok(RgbaImage {
        width: info.width,
        height: info.height,
        data,
    })
}

// ── Spectrogram helpers ──────────────────────────────────────────────

/// Map FFT magnitude bins to a fixed-height spectrogram column using
/// logarithmic frequency spacing.
///
/// Returns a `Vec<f32>` of length `num_rows`, where index 0 is the
/// lowest frequency (bottom of spectrogram) and the last index is
/// the highest (top).
fn map_spectrum_to_column(
    magnitudes: &[f32],
    num_rows: usize,
    sample_rate: u32,
    freq_max: f32,
) -> Vec<f32> {
    let n_bins = magnitudes.len();
    if n_bins == 0 || num_rows == 0 {
        return vec![0.0; num_rows];
    }

    let nyquist = sample_rate as f32 / 2.0;
    let f_max = freq_max.min(nyquist);
    let log_min = FREQ_MIN.ln();
    let log_max = f_max.ln();

    // Pre-compute the fractional bin index for each row centre.
    let bin_centers: Vec<f32> = (0..num_rows)
        .map(|row| {
            let t = if num_rows > 1 {
                row as f32 / (num_rows - 1) as f32
            } else {
                0.5
            };
            let freq = (log_min + t * (log_max - log_min)).exp();
            (freq / nyquist * n_bins as f32).clamp(0.0, (n_bins - 1) as f32)
        })
        .collect();

    let mut column = Vec::with_capacity(num_rows);

    for row in 0..num_rows {
        // Each row covers the band from the midpoint to its previous
        // neighbour up to the midpoint to its next neighbour.
        let lo = if row == 0 {
            bin_centers[0]
        } else {
            (bin_centers[row - 1] + bin_centers[row]) * 0.5
        };
        let hi = if row + 1 >= num_rows {
            bin_centers[num_rows - 1]
        } else {
            (bin_centers[row] + bin_centers[row + 1]) * 0.5
        };

        let bin_start = (lo.floor() as usize).min(n_bins - 1);
        let bin_end = ((hi.ceil() as usize) + 1).min(n_bins);
        // Guarantee at least one bin.
        let bin_end = bin_end.max(bin_start + 1);

        // Use peak (max) within the bin range — matches how real-time
        // spectrum analysers work and avoids visual holes between harmonics.
        let peak_val: f32 = magnitudes[bin_start..bin_end]
            .iter()
            .copied()
            .fold(0.0f32, f32::max);

        column.push(peak_val);
    }

    column
}

/// Render the spectrogram waterfall into the pixel buffer.
///
/// `history` holds the most recent columns of spectral data (each a
/// `Vec<f32>` of length `SPEC_H`).  Newer columns are at the end.
/// The spectrogram is right-aligned: the newest column draws at the
/// right edge of the area, oldest on the left.
#[allow(clippy::too_many_arguments)]
fn render_spectrogram(
    pb: &mut PixelBuffer,
    history: &[Vec<f32>],
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    peak: f32,
    mono: Option<([u8; 4], [u8; 4])>,
) {
    if history.is_empty() || peak < PEAK_FLOOR {
        return;
    }

    let n = history.len();
    let start = n.saturating_sub(w);
    let num_cols = n - start;

    for (col_idx, column) in history[start..].iter().enumerate() {
        // Right-aligned: newest column at the right edge.
        let x = x0 + w - num_cols + col_idx;

        for (row_idx, &magnitude) in column.iter().enumerate() {
            // row 0 = low freq = bottom of area
            if row_idx >= h {
                break;
            }
            let y = y0 + h - 1 - row_idx;

            let norm = (magnitude / peak).clamp(0.0, 1.0);
            // Log scale for better visual contrast.
            let brightness = if norm > 0.0 {
                (1.0 + norm * 9.0).log10() // maps 0..1 → 0..1
            } else {
                0.0
            };

            let color = if mono.is_some() {
                // Premultiplied alpha: white × brightness.
                let alpha = (brightness * 255.0) as u8;
                [alpha, alpha, alpha, alpha]
            } else {
                super::render_util::heat_map_color(norm, brightness)
            };
            pb.set_pixel(x, y, color);
        }
    }
}

/// Render amplitude history inside a sub-region of the badge.
///
/// Draws symmetric bars around the vertical centre, scrolling left
/// (newest at right edge).  Uses premultiplied alpha white (like the
/// waterfall), or monochrome palette if provided.
#[allow(clippy::too_many_arguments)]
fn render_amplitude_badge(
    pb: &mut PixelBuffer,
    history: &[f32],
    max_rms: f32,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    mono: Option<([u8; 4], [u8; 4])>,
) {
    if history.is_empty() || max_rms < PEAK_FLOOR {
        return;
    }

    let n = history.len();
    let center_y = y0 + h / 2;
    let max_half = (h / 2).saturating_sub(1);

    for col in 0..w {
        let start = col * n / w;
        let end = ((col + 1) * n / w).max(start + 1).min(n);

        let avg_rms = if end > start {
            history[start..end].iter().sum::<f32>() / (end - start) as f32
        } else if start < n {
            history[start]
        } else {
            0.0
        };

        let norm = (avg_rms / max_rms).clamp(0.0, 1.0);
        let half_height = (norm * max_half as f32) as usize;

        let color = if let Some((fg, bg)) = mono {
            super::render_util::lerp_color(bg, fg, norm)
        } else {
            super::render_util::level_color(norm)
        };

        let top = center_y.saturating_sub(half_height);
        let bottom = center_y + half_height;
        for y in top..=bottom.min(y0 + h - 1) {
            pb.set_pixel(x0 + col, y, color);
        }
    }
}

/// Render spectrum bars inside a sub-region of the badge.
///
/// Bars grow upward from the bottom edge.  Uses premultiplied alpha
/// white (like the waterfall), or monochrome palette if provided.
#[allow(clippy::too_many_arguments)]
fn render_spectrum_badge(
    pb: &mut PixelBuffer,
    magnitudes: &[f32],
    peak: f32,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    mono: Option<([u8; 4], [u8; 4])>,
) {
    if magnitudes.is_empty() || peak < PEAK_FLOOR {
        return;
    }

    // Use lower quarter of spectrum (voice content).
    let useful = &magnitudes[..magnitudes.len() / 4];

    let num_bars = w; // one bar per pixel column — no gaps
    if num_bars == 0 || useful.is_empty() {
        return;
    }

    for bar in 0..num_bars {
        // Proportional mapping: distribute ALL bins across bars evenly.
        let start = bar * useful.len() / num_bars;
        let end = ((bar + 1) * useful.len() / num_bars).max(start + 1);

        let avg = if end > start {
            useful[start..end].iter().sum::<f32>() / (end - start) as f32
        } else {
            0.0
        };

        let norm = (avg / peak).clamp(0.0, 1.0);
        let log_norm = (1.0 + norm * 9.0).log10();

        let bar_height = (log_norm * (h.saturating_sub(4)) as f32) as usize;
        let bar_x = x0 + bar;
        let bar_y = y0 + h - 2 - bar_height;

        let color = if let Some((fg, bg)) = mono {
            super::render_util::lerp_color(bg, fg, log_norm)
        } else {
            super::render_util::level_color(log_norm)
        };

        for dy in 0..bar_height {
            pb.set_pixel(bar_x, bar_y + dy, color);
        }
    }
}

/// Draw the red dot with brightness driven by current volume level.
/// Clear a circle to `BG_COLOR` (transparent), creating a gap between
/// the red dot and the spectrogram underneath.
fn clear_dot_gap(pb: &mut PixelBuffer, cx: usize, cy: usize, outer_radius: f32) {
    let r_sq = outer_radius * outer_radius;
    let r_int = outer_radius.ceil() as usize;

    for dy in 0..=r_int {
        for dx in 0..=r_int {
            let dist_sq = (dx * dx + dy * dy) as f32;
            if dist_sq > r_sq {
                continue;
            }
            // Anti-aliased edge: fade to transparent over 1 px.
            let edge_dist = outer_radius - dist_sq.sqrt();
            let edge_alpha = edge_dist.clamp(0.0, 1.0);

            let coords: [(usize, usize); 4] = [
                (cx + dx, cy + dy),
                (cx.wrapping_sub(dx), cy + dy),
                (cx + dx, cy.wrapping_sub(dy)),
                (cx.wrapping_sub(dx), cy.wrapping_sub(dy)),
            ];
            for (px, py) in coords {
                if px < pb.width && py < pb.height {
                    if edge_alpha >= 1.0 {
                        pb.set_pixel(px, py, BG_COLOR);
                    } else {
                        // Blend existing pixel towards transparent.
                        let off = (py * pb.width + px) * 4;
                        let keep = 1.0 - edge_alpha;
                        pb.data[off] = (pb.data[off] as f32 * keep) as u8;
                        pb.data[off + 1] = (pb.data[off + 1] as f32 * keep) as u8;
                        pb.data[off + 2] = (pb.data[off + 2] as f32 * keep) as u8;
                        pb.data[off + 3] = (pb.data[off + 3] as f32 * keep) as u8;
                    }
                }
            }
        }
    }
}

fn draw_pulsing_dot(pb: &mut PixelBuffer, cx: usize, cy: usize, radius: f32, brightness: f32) {
    let r_sq = radius * radius;
    let r_int = radius.ceil() as usize;
    let brightness = brightness.clamp(DOT_MIN_BRIGHTNESS, 1.0);

    for dy in 0..=r_int {
        for dx in 0..=r_int {
            let dist_sq = (dx * dx + dy * dy) as f32;
            if dist_sq > r_sq {
                continue;
            }

            // Anti-aliasing at the edge: smooth falloff over 1px.
            let edge_dist = radius - dist_sq.sqrt();
            let edge_alpha = edge_dist.clamp(0.0, 1.0);

            // BGRA with true alpha for compositor transparency.
            let val = (255.0 * brightness * edge_alpha) as u8;
            let color = [0x00, 0x00, val, val];

            // Draw in all four quadrants.
            let coords: [(usize, usize); 4] = [
                (cx + dx, cy + dy),
                (cx.wrapping_sub(dx), cy + dy),
                (cx + dx, cy.wrapping_sub(dy)),
                (cx.wrapping_sub(dx), cy.wrapping_sub(dy)),
            ];
            for (px, py) in coords {
                if px < pb.width && py < pb.height {
                    pb.set_pixel(px, py, color);
                }
            }
        }
    }
}

/// Signed distance from a point to the border of a rounded rectangle.
///
/// Negative values are inside, positive outside.
fn rounded_rect_sdf(px: f32, py: f32, w: f32, h: f32, r: f32) -> f32 {
    let cx = px - w / 2.0;
    let cy = py - h / 2.0;
    let hw = w / 2.0 - r;
    let hh = h / 2.0 - r;
    let dx = cx.abs() - hw;
    let dy = cy.abs() - hh;
    let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
    let inside = dx.max(dy).min(0.0);
    outside + inside - r
}

/// Draw a rounded rectangle border (outline only) into the pixel buffer.
///
/// Uses an SDF for anti-aliased edges.  Only the border ring is drawn;
/// the interior is left untouched (transparent).
fn draw_rounded_border(pb: &mut PixelBuffer, color: [u8; 4], radius: f32, border_width: f32) {
    let w = pb.width as f32;
    let h = pb.height as f32;

    for y in 0..pb.height {
        for x in 0..pb.width {
            let d = rounded_rect_sdf(x as f32 + 0.5, y as f32 + 0.5, w, h, radius);

            // Outside the shape: d > 0 → skip (transparent).
            // Inside the border ring: -border_width < d <= 0.
            // Deep inside: d <= -border_width → skip (interior).

            // Outer edge anti-aliasing (smooth over 1px).
            let outer_alpha = (-d).clamp(0.0, 1.0);
            // Inner edge anti-aliasing (smooth over 1px).
            let inner_alpha = (d + border_width).clamp(0.0, 1.0);

            let alpha = outer_alpha * inner_alpha;

            if alpha > 0.0 {
                let a = (color[3] as f32 * alpha) as u8;
                // Premultiplied: color channels scaled by effective alpha.
                let b = (color[0] as f32 * alpha) as u8;
                let g = (color[1] as f32 * alpha) as u8;
                let r = (color[2] as f32 * alpha) as u8;
                let pixel = [b, g, r, a];
                pb.set_pixel(x, y, pixel);
            }
        }
    }
}

// ── Silence warning rendering ────────────────────────────────────────

/// Draw a prohibit icon (circle outline + diagonal bar) in bright red.
///
/// Replaces the pulsing red dot when silence is detected.  Anti-aliased
/// like `draw_pulsing_dot()`, placed at the same centre coordinates.
fn draw_prohibit_icon(pb: &mut PixelBuffer, cx: usize, cy: usize, radius: f32) {
    // Bright red, BGRA
    let color: [u8; 4] = [0x00, 0x00, 0xFF, 0xFF];
    let stroke = 2.0f32;
    let r_outer = radius;
    let r_inner = radius - stroke;
    let r_outer_sq = r_outer * r_outer;
    let r_inner_sq = r_inner * r_inner;
    let r_int = r_outer.ceil() as i32;

    // Draw circle outline
    for dy in -r_int..=r_int {
        for dx in -r_int..=r_int {
            let dist_sq = (dx * dx + dy * dy) as f32;
            if dist_sq > r_outer_sq {
                continue;
            }
            // Anti-aliased outer edge
            let outer_edge = r_outer - dist_sq.sqrt();
            let outer_alpha = outer_edge.clamp(0.0, 1.0);
            // Anti-aliased inner edge (hollow centre)
            let inner_edge = dist_sq.sqrt() - r_inner;
            let inner_alpha = inner_edge.clamp(0.0, 1.0);

            let alpha = outer_alpha * inner_alpha;
            if alpha <= 0.0 {
                continue;
            }

            let px = cx as i32 + dx;
            let py = cy as i32 + dy;
            if px >= 0 && (px as usize) < pb.width && py >= 0 && (py as usize) < pb.height {
                let off = (py as usize * pb.width + px as usize) * 4;
                for (c, &fg_val) in color.iter().enumerate().take(4) {
                    let bg_val = pb.data[off + c] as f32;
                    pb.data[off + c] = (bg_val + (fg_val as f32 - bg_val) * alpha) as u8;
                }
            }
        }
    }

    // Draw diagonal bar (top-left to bottom-right at 45°)
    let half_stroke = stroke / 2.0;
    for dy in -r_int..=r_int {
        for dx in -r_int..=r_int {
            let dist_sq = (dx * dx + dy * dy) as f32;
            // Only draw inside the circle
            if dist_sq > r_inner_sq {
                continue;
            }
            // Distance from the line y = x (45° diagonal)
            let line_dist = ((dx as f32) - (dy as f32)).abs() / std::f32::consts::SQRT_2;
            if line_dist > half_stroke + 1.0 {
                continue;
            }
            let alpha = (half_stroke + 1.0 - line_dist).clamp(0.0, 1.0);
            if alpha <= 0.0 {
                continue;
            }

            let px = cx as i32 + dx;
            let py = cy as i32 + dy;
            if px >= 0 && (px as usize) < pb.width && py >= 0 && (py as usize) < pb.height {
                let off = (py as usize * pb.width + px as usize) * 4;
                for (c, &fg_val) in color.iter().enumerate().take(4) {
                    let bg_val = pb.data[off + c] as f32;
                    pb.data[off + c] = (bg_val + (fg_val as f32 - bg_val) * alpha) as u8;
                }
            }
        }
    }
}

/// Render "NO SOUND" text in bright red, centred in the SPEC area.
fn render_no_sound_text(
    pb: &mut PixelBuffer,
    font: &fontdue::Font,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    let font_size = 24.0f32;
    let color: [u8; 4] = [0x00, 0x00, 0xFF, 0xFF]; // bright red BGRA
    let (glyphs, text_w) = rasterise_glyphs("NO SOUND", font, font_size);

    // Centre horizontally in the spec area
    let start_x = x0 as i32 + (w as i32 - text_w as i32) / 2;
    // Centre vertically in the spec area
    let baseline = y0 as i32 + (h as i32 * 3) / 4;

    let buf_w = pb.width;
    let buf_h = pb.height;
    let mut cursor_x = start_x;
    for (metrics, bitmap) in &glyphs {
        blit_glyph_at(
            pb, metrics, bitmap, cursor_x, baseline, buf_w, buf_h, color, 1.0,
        );
        cursor_x += metrics.advance_width as i32;
    }
}

// ── Static PNG window (transcribing) ─────────────────────────────────

/// Apply a 1-bit shape mask based on the image alpha channel.
///
/// Pixels with alpha > 128 are visible; all others are transparent.
fn apply_alpha_shape_mask(
    conn: &impl Connection,
    win: u32,
    img: &RgbaImage,
    w: u16,
    h: u16,
) -> Result<(), TalkError> {
    let mask = conn
        .generate_id()
        .map_err(|e| TalkError::Config(format!("X11 generate_id failed: {}", e)))?;

    conn.create_pixmap(1, mask, win, w, h)
        .map_err(|e| TalkError::Config(format!("X11 create_pixmap failed: {}", e)))?;

    let gc = conn
        .generate_id()
        .map_err(|e| TalkError::Config(format!("X11 generate_id failed: {}", e)))?;

    conn.create_gc(gc, mask, &CreateGCAux::new().foreground(0))
        .map_err(|e| TalkError::Config(format!("X11 create_gc failed: {}", e)))?;

    conn.poly_fill_rectangle(
        mask,
        gc,
        &[Rectangle {
            x: 0,
            y: 0,
            width: w,
            height: h,
        }],
    )
    .map_err(|e| TalkError::Config(format!("X11 poly_fill_rectangle failed: {}", e)))?;

    conn.change_gc(gc, &ChangeGCAux::new().foreground(1))
        .map_err(|e| TalkError::Config(format!("X11 change_gc failed: {}", e)))?;

    let mut opaque_points: Vec<Point> = Vec::new();
    for py in 0..img.height {
        for px in 0..img.width {
            let idx = ((py * img.width + px) * 4) as usize;
            let alpha = img.data[idx + 3];
            if alpha > 128 {
                opaque_points.push(Point {
                    x: px as i16,
                    y: py as i16,
                });
            }
        }
    }

    for chunk in opaque_points.chunks(4096) {
        conn.poly_point(CoordMode::ORIGIN, mask, gc, chunk)
            .map_err(|e| TalkError::Config(format!("X11 poly_point failed: {}", e)))?;
    }

    shape::mask(conn, shape::SO::SET, shape::SK::BOUNDING, win, 0, 0, mask)
        .map_err(|e| TalkError::Config(format!("X11 shape_mask failed: {}", e)))?;

    conn.free_gc(gc)
        .map_err(|e| TalkError::Config(format!("X11 free_gc failed: {}", e)))?;
    conn.free_pixmap(mask)
        .map_err(|e| TalkError::Config(format!("X11 free_pixmap failed: {}", e)))?;

    Ok(())
}

/// Draw visible pixels onto the window, grouped by color for efficiency.
fn draw_image(
    conn: &impl Connection,
    win: u32,
    screen: &Screen,
    img: &RgbaImage,
) -> Result<(), TalkError> {
    let cmap = screen.default_colormap;

    let mut color_groups: HashMap<(u8, u8, u8), Vec<Point>> = HashMap::new();

    for py in 0..img.height {
        for px in 0..img.width {
            let idx = ((py * img.width + px) * 4) as usize;
            let r = img.data[idx];
            let g = img.data[idx + 1];
            let b = img.data[idx + 2];
            let a = img.data[idx + 3];

            if a > 128 {
                color_groups.entry((r, g, b)).or_default().push(Point {
                    x: px as i16,
                    y: py as i16,
                });
            }
        }
    }

    let gc = conn
        .generate_id()
        .map_err(|e| TalkError::Config(format!("X11 generate_id failed: {}", e)))?;

    conn.create_gc(gc, win, &CreateGCAux::new())
        .map_err(|e| TalkError::Config(format!("X11 create_gc failed: {}", e)))?;

    for ((r, g, b), points) in &color_groups {
        let reply = conn
            .alloc_color(
                cmap,
                (*r as u16) * 257,
                (*g as u16) * 257,
                (*b as u16) * 257,
            )
            .map_err(|e| TalkError::Config(format!("X11 alloc_color failed: {}", e)))?
            .reply()
            .map_err(|e| TalkError::Config(format!("X11 alloc_color reply failed: {}", e)))?;

        conn.change_gc(gc, &ChangeGCAux::new().foreground(reply.pixel))
            .map_err(|e| TalkError::Config(format!("X11 change_gc failed: {}", e)))?;

        for chunk in points.chunks(4096) {
            conn.poly_point(CoordMode::ORIGIN, win, gc, chunk)
                .map_err(|e| TalkError::Config(format!("X11 poly_point failed: {}", e)))?;
        }
    }

    conn.free_gc(gc)
        .map_err(|e| TalkError::Config(format!("X11 free_gc failed: {}", e)))?;

    Ok(())
}

// ── X11 overlay thread ──────────────────────────────────────────────

/// Main loop for the overlay background thread.
///
/// Has two modes:
///
/// * **Transcribing** — static PNG badge; blocks on commands.
/// * **Recording** — dynamic spectrogram badge at 60 fps reading from
///   the shared `audio_ring` buffer (fed by the audio tee); drains
///   commands non-blocking each frame.
#[allow(clippy::too_many_lines)]
fn overlay_thread(
    rx: mpsc::Receiver<Command>,
    geom: super::monitor::MonitorGeometry,
    viz: Option<crate::config::VizMode>,
    mono_palette: Option<([u8; 4], [u8; 4])>,
    ring: Arc<Mutex<RingBuffer>>,
    sample_rate: u32,
    silence_tx: Option<std::sync::mpsc::Sender<bool>>,
) -> Result<(), TalkError> {
    let (conn, screen_num) = x11rb::connect(None)
        .map_err(|e| TalkError::Config(format!("failed to connect to X11: {}", e)))?;

    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;

    // Try to find a 32-bit ARGB visual for compositor transparency.
    // Falls back to the root depth if unavailable.
    let argb_ctx = if let Some(visual) = find_argb_visual(screen) {
        let colormap = conn
            .generate_id()
            .map_err(|e| TalkError::Config(format!("X11 generate_id failed: {}", e)))?;
        conn.create_colormap(ColormapAlloc::NONE, colormap, root, visual)
            .map_err(|e| TalkError::Config(format!("X11 create_colormap failed: {}", e)))?;
        log::info!("using 32-bit ARGB visual for recording badge transparency");
        Some(ArgbContext {
            visual,
            colormap,
            depth: 32,
        })
    } else {
        log::warn!("no 32-bit ARGB visual found; recording badge will have opaque background");
        None
    };

    let depth = argb_ctx.as_ref().map_or(screen.root_depth, |c| c.depth);

    // Pre-decode the transcribing indicator image.
    let transcribing_img = decode_png(TRANSCRIBING_PNG)?;

    let (mon_x, mon_y, mon_w, _mon_h) = geom;

    // Audio ring buffer and sample rate are now provided externally
    // via the audio tee task (no independent CPAL capture).
    let rms_chunk: usize = sample_rate as usize / FPS as usize;

    // Load a system font for rendering "NO SOUND" text on the badge.
    let badge_font = super::render_util::load_system_font(24.0);

    // ── Dead-signal detection state ─────────────────────────────────
    // Detect a dead/missing audio device by checking sample variance.
    // A real microphone always produces nonzero variance (thermal and
    // quantization noise), even in a silent room (~6e-10 measured).
    // A dead device produces perfectly uniform samples (variance = 0):
    // e.g. constant i16::MIN (-32768 → -1.0f32) or all zeros.
    let mut dead_signal_frames: u32 = 0;
    let mut no_sound_active: bool = false;
    let mut silence_notified: bool = false;
    const DEAD_SIGNAL_VARIANCE_CEIL: f32 = 1e-10; // ~6× below real mic noise floor
    const DEAD_SIGNAL_TRIGGER_FRAMES: u32 = 1; // trigger immediately — variance is unambiguous

    // Diagnostic logging counter (logs every ~1 second = 60 frames).
    let mut diag_frame_counter: u32 = 0;
    const DIAG_LOG_INTERVAL: u32 = 60;

    // ── State ────────────────────────────────────────────────────────

    let frame_dur = std::time::Duration::from_micros(1_000_000 / FPS as u64);

    let mut current_window: Option<Window> = None;
    let mut current_gc: Option<Gcontext> = None;
    let mut is_recording = false;

    let mut spectrogram_history: Vec<Vec<f32>> = Vec::new();
    let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
    let mut rms_peak: f32 = PEAK_FLOOR;

    // Amplitude history for amplitude viz mode: one RMS value per frame.
    let amp_window_secs: f32 = 5.0;
    let amp_max_frames = (FPS as f32 * amp_window_secs) as usize;
    let mut amp_history: Vec<f32> = vec![0.0; amp_max_frames];
    let mut amp_peak: f32 = PEAK_FLOOR;
    // Spectrum peak for spectrum viz mode.
    let mut spectrum_peak: f32 = PEAK_FLOOR;
    let mut spec_peak: f32 = PEAK_FLOOR;
    // Dynamic frequency ceiling — grows as higher harmonics appear.
    let mut effective_freq_max: f32 = FREQ_INITIAL_MAX;

    // ── Event loop ───────────────────────────────────────────────────

    loop {
        // ── Idle / transcribing: block on commands ───────────────
        if !is_recording {
            let cmd = match rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => break,
            };

            match cmd {
                Command::Show(IndicatorKind::Recording) => {
                    destroy_current(&conn, &mut current_window, &mut current_gc);

                    let badge_x = mon_x + (mon_w as i16 / 2) - (BADGE_W as i16 / 2);
                    let badge_y = mon_y + 4;

                    let win = if let Some(ref ctx) = argb_ctx {
                        // 32-bit ARGB window — compositor handles transparency.
                        create_argb_overlay_window(
                            &conn, root, ctx, badge_x, badge_y, BADGE_W, BADGE_H,
                        )?
                    } else {
                        // Fallback: opaque window with shape mask.
                        let w = create_overlay_window(
                            &conn, screen, root, badge_x, badge_y, BADGE_W, BADGE_H,
                        )?;
                        apply_rounded_shape(&conn, w, BADGE_W, BADGE_H, CORNER_RADIUS)?;
                        w
                    };

                    conn.map_window(win)
                        .map_err(|e| TalkError::Config(format!("X11 map_window failed: {}", e)))?;
                    conn.sync()
                        .map_err(|e| TalkError::Config(format!("X11 sync failed: {}", e)))?;

                    let gc = conn
                        .generate_id()
                        .map_err(|e| TalkError::Config(format!("X11 generate_id: {}", e)))?;
                    conn.create_gc(gc, win, &CreateGCAux::new())
                        .map_err(|e| TalkError::Config(format!("X11 create_gc: {}", e)))?;

                    conn.flush()
                        .map_err(|e| TalkError::Config(format!("X11 flush: {}", e)))?;

                    current_window = Some(win);
                    current_gc = Some(gc);
                    is_recording = true;
                    spectrogram_history.clear();
                    rms_peak = PEAK_FLOOR;
                    spec_peak = PEAK_FLOOR;
                    spectrum_peak = PEAK_FLOOR;
                    amp_peak = PEAK_FLOOR;
                    amp_history.clear();
                    amp_history.resize(amp_max_frames, 0.0);
                    effective_freq_max = FREQ_INITIAL_MAX;
                    dead_signal_frames = 0;
                    no_sound_active = false;
                    silence_notified = false;
                }

                Command::Show(IndicatorKind::Transcribing) => {
                    destroy_current(&conn, &mut current_window, &mut current_gc);
                    show_transcribing(
                        &conn,
                        screen,
                        root,
                        &transcribing_img,
                        mon_x,
                        mon_y,
                        mon_w,
                        &mut current_window,
                    )?;
                }

                Command::Hide => {
                    destroy_current(&conn, &mut current_window, &mut current_gc);
                }

                Command::Quit => {
                    destroy_current(&conn, &mut current_window, &mut current_gc);
                    break;
                }
            }

            continue;
        }

        // ── Recording state: 60 fps render loop ─────────────────

        let frame_start = std::time::Instant::now();

        // Non-blocking command drain.
        let mut quit = false;
        loop {
            match rx.try_recv() {
                Ok(Command::Show(IndicatorKind::Recording)) => {
                    spectrogram_history.clear();
                    rms_peak = PEAK_FLOOR;
                    spec_peak = PEAK_FLOOR;
                    spectrum_peak = PEAK_FLOOR;
                    amp_peak = PEAK_FLOOR;
                    amp_history.clear();
                    amp_history.resize(amp_max_frames, 0.0);
                    effective_freq_max = FREQ_INITIAL_MAX;
                    dead_signal_frames = 0;
                    no_sound_active = false;
                    silence_notified = false;
                }
                Ok(Command::Show(IndicatorKind::Transcribing)) => {
                    destroy_current(&conn, &mut current_window, &mut current_gc);
                    is_recording = false;
                    show_transcribing(
                        &conn,
                        screen,
                        root,
                        &transcribing_img,
                        mon_x,
                        mon_y,
                        mon_w,
                        &mut current_window,
                    )?;
                    break;
                }
                Ok(Command::Hide) => {
                    destroy_current(&conn, &mut current_window, &mut current_gc);
                    is_recording = false;
                    break;
                }
                Ok(Command::Quit) | Err(mpsc::TryRecvError::Disconnected) => {
                    destroy_current(&conn, &mut current_window, &mut current_gc);
                    is_recording = false;
                    quit = true;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
            }
        }

        if quit {
            break;
        }
        if !is_recording {
            continue;
        }

        // ── Read audio and compute visualization frame ──────

        let (frame_rms, magnitudes, frame_variance) = {
            let samples = ring
                .lock()
                .map(|g| g.read_last(FFT_SIZE.max(rms_chunk)))
                .unwrap_or_else(|_| vec![0.0; FFT_SIZE.max(rms_chunk)]);
            let rms_slice = &samples[samples.len().saturating_sub(rms_chunk.max(1))..];
            let fr = rms(rms_slice);
            // Compute sample variance to distinguish a dead device
            // (constant signal, variance ≈ 0) from a real microphone
            // (random noise, variance > 0 even when quiet).
            let n = rms_slice.len().max(1) as f32;
            let mean = rms_slice.iter().sum::<f32>() / n;
            let var = rms_slice.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
            let mags = compute_spectrum(&samples);
            (fr, mags, var)
        };

        // Update RMS peak (fast attack, slow decay).
        rms_peak *= PEAK_DECAY;
        if frame_rms > rms_peak {
            rms_peak = frame_rms;
        }
        rms_peak = rms_peak.max(PEAK_FLOOR);

        // ── Diagnostic logging (once per second) ─────────────
        diag_frame_counter += 1;
        if diag_frame_counter >= DIAG_LOG_INTERVAL {
            diag_frame_counter = 0;
            log::debug!(
                "[audio-diag] rms={:.6} variance={:.10}",
                frame_rms,
                frame_variance,
            );
        }

        // ── Dead-signal detection ────────────────────────────
        // A dead or missing audio device produces perfectly uniform
        // samples (e.g. constant -1.0 or all zeros) with variance ≈ 0.
        // A real microphone always has random noise (variance > 1e-10).
        if frame_variance < DEAD_SIGNAL_VARIANCE_CEIL {
            dead_signal_frames = dead_signal_frames.saturating_add(1);
        } else {
            if no_sound_active {
                if let Some(ref tx) = silence_tx {
                    let _ = tx.send(false);
                }
            }
            dead_signal_frames = 0;
            no_sound_active = false;
            silence_notified = false;
        }
        if dead_signal_frames >= DEAD_SIGNAL_TRIGGER_FRAMES {
            no_sound_active = true;
            if !silence_notified {
                if let Some(ref tx) = silence_tx {
                    let _ = tx.send(true);
                }
                silence_notified = true;
            }
        }

        // Per-viz-mode data updates.
        if let Some(mode) = viz {
            use crate::config::VizMode;
            match mode {
                VizMode::Waterfall => {
                    // Dynamic frequency scaling.
                    let nyquist = sample_rate as f32 / 2.0;
                    let n_mag = magnitudes.len();
                    for (i, &mag) in magnitudes.iter().enumerate().rev() {
                        if mag > FREQ_NOISE_FLOOR {
                            let freq = (i as f32 / n_mag as f32) * nyquist;
                            if freq > effective_freq_max {
                                effective_freq_max = freq.min(FREQ_MAX);
                            }
                            break;
                        }
                    }
                    let column = map_spectrum_to_column(
                        &magnitudes,
                        SPEC_H,
                        sample_rate,
                        effective_freq_max,
                    );
                    spectrogram_history.push(column);
                    if spectrogram_history.len() > SPEC_W {
                        spectrogram_history.drain(..spectrogram_history.len() - SPEC_W);
                    }
                    // All-time peak for opacity normalization.
                    let frame_spec_max = magnitudes.iter().copied().fold(0.0f32, f32::max);
                    if frame_spec_max > spec_peak {
                        spec_peak = frame_spec_max;
                    }
                }
                VizMode::Amplitude => {
                    if frame_rms > amp_peak {
                        amp_peak = frame_rms;
                    }
                    amp_history.push(frame_rms);
                    if amp_history.len() > amp_max_frames {
                        amp_history.drain(..amp_history.len() - amp_max_frames);
                    }
                }
                VizMode::Spectrum => {
                    spectrum_peak *= PEAK_DECAY;
                    let frame_peak = magnitudes.iter().copied().fold(0.0f32, f32::max);
                    if frame_peak > spectrum_peak {
                        spectrum_peak = frame_peak;
                    }
                    spectrum_peak = spectrum_peak.max(PEAK_FLOOR);
                }
            }
        }

        // ── Render badge ─────────────────────────────────────

        pb.clear(BG_COLOR);
        draw_rounded_border(&mut pb, BORDER_COLOR, CORNER_RADIUS as f32, BORDER_WIDTH);

        if no_sound_active {
            // ── NO SOUND mode: prohibit icon + text ──────────
            if let Some(ref f) = badge_font {
                render_no_sound_text(&mut pb, f, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H);
            }
            clear_dot_gap(&mut pb, DOT_CX, DOT_CY, DOT_RADIUS_MAX + DOT_GAP);
            draw_prohibit_icon(&mut pb, DOT_CX, DOT_CY, DOT_RADIUS_MAX);
        } else {
            // ── Normal mode: visualization + pulsing dot ─────
            // Render visualization inside the badge area.
            if let Some(mode) = viz {
                use crate::config::VizMode;
                match mode {
                    VizMode::Waterfall => {
                        render_spectrogram(
                            &mut pb,
                            &spectrogram_history,
                            SPEC_LEFT,
                            SPEC_TOP,
                            SPEC_W,
                            SPEC_H,
                            spec_peak,
                            mono_palette,
                        );
                    }
                    VizMode::Amplitude => {
                        render_amplitude_badge(
                            &mut pb,
                            &amp_history,
                            amp_peak,
                            SPEC_LEFT,
                            SPEC_TOP,
                            SPEC_W,
                            SPEC_H,
                            mono_palette,
                        );
                    }
                    VizMode::Spectrum => {
                        render_spectrum_badge(
                            &mut pb,
                            &magnitudes,
                            spectrum_peak,
                            SPEC_LEFT,
                            SPEC_TOP,
                            SPEC_W,
                            SPEC_H,
                            mono_palette,
                        );
                    }
                }
            }

            let vol_norm = if rms_peak > PEAK_FLOOR {
                (frame_rms / rms_peak).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let dot_radius = DOT_RADIUS_MIN + (DOT_RADIUS_MAX - DOT_RADIUS_MIN) * vol_norm;
            let dot_brightness = DOT_MIN_BRIGHTNESS + (1.0 - DOT_MIN_BRIGHTNESS) * vol_norm;
            clear_dot_gap(&mut pb, DOT_CX, DOT_CY, dot_radius + DOT_GAP);
            draw_pulsing_dot(&mut pb, DOT_CX, DOT_CY, dot_radius, dot_brightness);
        }

        // ── Blit to X11 window ───────────────────────────────

        if let (Some(win), Some(gc)) = (current_window, current_gc) {
            let _ = conn.put_image(
                ImageFormat::Z_PIXMAP,
                win,
                gc,
                BADGE_W,
                BADGE_H,
                0,
                0,
                0,
                depth,
                &pb.data,
            );
            let _ = conn.flush();
        }

        // ── Frame timing ─────────────────────────────────────

        let elapsed = frame_start.elapsed();
        if elapsed < frame_dur {
            std::thread::sleep(frame_dur - elapsed);
        }
    }

    Ok(())
}

// ── Window helpers ───────────────────────────────────────────────────

/// Find a 32-bit TrueColor visual suitable for compositor alpha transparency.
///
/// Returns `Some(visual_id)` if one is available, `None` otherwise.
fn find_argb_visual(screen: &Screen) -> Option<Visualid> {
    screen
        .allowed_depths
        .iter()
        .filter(|d| d.depth == 32)
        .flat_map(|d| &d.visuals)
        .find(|v| v.class == VisualClass::TRUE_COLOR)
        .map(|v| v.visual_id)
}

/// ARGB window context: visual, colormap, and depth for 32-bit transparency.
struct ArgbContext {
    visual: Visualid,
    colormap: Colormap,
    depth: u8,
}

/// Create an override-redirect window with 32-bit ARGB visual.
///
/// The caller must free the colormap when the window is destroyed.
fn create_argb_overlay_window(
    conn: &impl Connection,
    root: u32,
    ctx: &ArgbContext,
    x: i16,
    y: i16,
    w: u16,
    h: u16,
) -> Result<u32, TalkError> {
    let win = conn
        .generate_id()
        .map_err(|e| TalkError::Config(format!("X11 generate_id failed: {}", e)))?;

    let values = CreateWindowAux::new()
        .background_pixel(0) // transparent
        .border_pixel(0) // required for non-default visual
        .override_redirect(1u32)
        .event_mask(EventMask::EXPOSURE)
        .colormap(ctx.colormap);

    conn.create_window(
        ctx.depth,
        win,
        root,
        x,
        y,
        w,
        h,
        0,
        WindowClass::INPUT_OUTPUT,
        ctx.visual,
        &values,
    )
    .map_err(|e| TalkError::Config(format!("X11 create_window failed: {}", e)))?;

    Ok(win)
}

/// Create an override-redirect X11 window at the given position.
fn create_overlay_window(
    conn: &impl Connection,
    screen: &Screen,
    root: u32,
    x: i16,
    y: i16,
    w: u16,
    h: u16,
) -> Result<u32, TalkError> {
    let win = conn
        .generate_id()
        .map_err(|e| TalkError::Config(format!("X11 generate_id failed: {}", e)))?;

    let values = CreateWindowAux::new()
        .background_pixel(screen.black_pixel)
        .border_pixel(0)
        .override_redirect(1u32)
        .event_mask(EventMask::EXPOSURE);

    conn.create_window(
        COPY_DEPTH_FROM_PARENT,
        win,
        root,
        x,
        y,
        w,
        h,
        0,
        WindowClass::INPUT_OUTPUT,
        0,
        &values,
    )
    .map_err(|e| TalkError::Config(format!("X11 create_window failed: {}", e)))?;

    Ok(win)
}

/// Show the static transcribing badge.
#[allow(clippy::too_many_arguments)]
fn show_transcribing(
    conn: &impl Connection,
    screen: &Screen,
    root: u32,
    img: &RgbaImage,
    mon_x: i16,
    mon_y: i16,
    mon_w: u16,
    current_window: &mut Option<u32>,
) -> Result<(), TalkError> {
    let w = img.width as u16;
    let h = img.height as u16;
    let x = mon_x + (mon_w as i16 / 2) - (w as i16 / 2);
    let y = mon_y + 4;

    let win = create_overlay_window(conn, screen, root, x, y, w, h)?;
    apply_alpha_shape_mask(conn, win, img, w, h)?;

    conn.map_window(win)
        .map_err(|e| TalkError::Config(format!("X11 map_window failed: {}", e)))?;
    conn.sync()
        .map_err(|e| TalkError::Config(format!("X11 sync failed: {}", e)))?;

    draw_image(conn, win, screen, img)?;

    conn.flush()
        .map_err(|e| TalkError::Config(format!("X11 flush failed: {}", e)))?;

    *current_window = Some(win);
    Ok(())
}

/// Destroy the current overlay window and free its GC if present.
fn destroy_current(conn: &impl Connection, window: &mut Option<u32>, gc: &mut Option<u32>) {
    if let Some(g) = gc.take() {
        let _ = conn.free_gc(g);
    }
    if let Some(win) = window.take() {
        let _ = conn.destroy_window(win);
        let _ = conn.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PNG decoding ─────────────────────────────────────────────────

    #[test]
    fn test_decode_transcribing_png() {
        let img = decode_png(TRANSCRIBING_PNG).expect("decode transcribing PNG");
        assert_eq!(img.width, 210);
        assert_eq!(img.height, 52);
        assert_eq!(img.data.len(), 210 * 52 * 4);
    }

    #[test]
    fn test_transcribing_png_has_opaque_pixels() {
        let img = decode_png(TRANSCRIBING_PNG).expect("decode");
        let opaque_count = (0..img.width * img.height)
            .filter(|&i| img.data[(i * 4 + 3) as usize] > 128)
            .count();
        assert!(
            opaque_count > 1000,
            "expected >1000 opaque pixels, got {}",
            opaque_count
        );
    }

    // ── Spectrogram mapping ──────────────────────────────────────────

    #[test]
    fn map_spectrum_column_length() {
        let magnitudes = vec![1.0f32; 512];
        let column = map_spectrum_to_column(&magnitudes, SPEC_H, 48000, FREQ_MAX);
        assert_eq!(column.len(), SPEC_H);
    }

    #[test]
    fn map_spectrum_empty_magnitudes() {
        let column = map_spectrum_to_column(&[], SPEC_H, 48000, FREQ_MAX);
        assert_eq!(column.len(), SPEC_H);
        assert!(column.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn map_spectrum_low_freq_comes_first() {
        // Create magnitudes that are loud at low bins and quiet at high.
        let mut magnitudes = vec![0.0f32; 512];
        for m in magnitudes.iter_mut().take(10) {
            *m = 10.0;
        }
        let column = map_spectrum_to_column(&magnitudes, SPEC_H, 48000, FREQ_MAX);
        // First row (low freq) should be louder than last (high freq).
        assert!(
            column[0] > column[SPEC_H - 1],
            "low freq row ({}) should be louder than high freq row ({})",
            column[0],
            column[SPEC_H - 1]
        );
    }

    // ── Spectrogram rendering ────────────────────────────────────────

    #[test]
    fn render_spectrogram_no_panic() {
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);

        let history: Vec<Vec<f32>> = (0..SPEC_W)
            .map(|i| vec![(i as f32 * 0.01).sin().abs(); SPEC_H])
            .collect();
        render_spectrogram(
            &mut pb, &history, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 1.0, None,
        );

        let mut non_bg = 0;
        for y in SPEC_TOP..SPEC_BOTTOM {
            for x in SPEC_LEFT..SPEC_RIGHT {
                let off = (y * pb.width + x) * 4;
                if pb.data[off..off + 4] != BG_COLOR {
                    non_bg += 1;
                }
            }
        }
        assert!(non_bg > 0, "spectrogram should produce non-bg pixels");
    }

    #[test]
    fn render_spectrogram_empty_history_no_panic() {
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);
        render_spectrogram(&mut pb, &[], SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 1.0, None);
    }

    #[test]
    fn render_spectrogram_right_aligned() {
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);

        // Only 5 columns of history — should right-align.
        let history: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0; SPEC_H]).collect();
        render_spectrogram(
            &mut pb, &history, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 1.0, None,
        );

        // Leftmost columns of spectrogram area should still be BG (alpha=0).
        let check_col = SPEC_LEFT;
        let mid_y = SPEC_TOP + SPEC_H / 2;
        let off = (mid_y * pb.width + check_col) * 4;
        assert_eq!(
            pb.data[off + 3],
            0,
            "leftmost spectrogram column should be transparent when history is short"
        );

        // Rightmost columns should have content (non-zero alpha).
        let right_col = SPEC_LEFT + SPEC_W - 1; // last column of spectrogram area
        let off2 = (mid_y * pb.width + right_col) * 4;
        assert!(
            pb.data[off2 + 3] > 0,
            "rightmost spectrogram column should have non-zero alpha"
        );
    }

    // ── Red dot rendering ────────────────────────────────────────────

    #[test]
    fn draw_pulsing_dot_no_panic() {
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);
        draw_pulsing_dot(&mut pb, DOT_CX, DOT_CY, DOT_RADIUS_MAX, 0.8);

        let off = (DOT_CY * pb.width + DOT_CX) * 4;
        assert!(
            pb.data[off + 2] > 0,
            "dot centre red channel should be non-zero"
        );
    }

    #[test]
    fn draw_pulsing_dot_dim_vs_bright() {
        let mut pb_dim = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb_dim.clear(BG_COLOR);
        draw_pulsing_dot(
            &mut pb_dim,
            DOT_CX,
            DOT_CY,
            DOT_RADIUS_MAX,
            DOT_MIN_BRIGHTNESS,
        );
        let off = (DOT_CY * pb_dim.width + DOT_CX) * 4;
        let dim_r = pb_dim.data[off + 2];

        let mut pb_bright = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb_bright.clear(BG_COLOR);
        draw_pulsing_dot(&mut pb_bright, DOT_CX, DOT_CY, DOT_RADIUS_MAX, 1.0);
        let bright_r = pb_bright.data[off + 2];

        assert!(
            bright_r > dim_r,
            "bright dot ({}) should have higher red than dim ({})",
            bright_r,
            dim_r
        );
    }

    // ── Badge constants coherence ────────────────────────────────────

    #[test]
    fn spectrogram_area_fits_in_badge() {
        let bw = BADGE_W as usize;
        let bh = BADGE_H as usize;
        assert!(
            SPEC_RIGHT <= bw,
            "spectrogram right edge exceeds badge width"
        );
        assert!(
            SPEC_BOTTOM <= bh,
            "spectrogram bottom edge exceeds badge height"
        );
    }

    #[test]
    fn dot_fits_in_badge() {
        let r = DOT_RADIUS_MAX.ceil() as usize;
        assert!(DOT_CX >= r);
        assert!(DOT_CY >= r);
        assert!(DOT_CX + r < BADGE_W as usize);
        assert!(DOT_CY + r < BADGE_H as usize);
    }

    #[test]
    fn indicator_kind_clone_eq() {
        let a = IndicatorKind::Recording;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(IndicatorKind::Recording, IndicatorKind::Transcribing);
    }

    // ── Border rendering ───────────────────────────────────────────────

    #[test]
    fn rounded_rect_sdf_centre_is_negative() {
        let w = BADGE_W as f32;
        let h = BADGE_H as f32;
        let d = rounded_rect_sdf(w / 2.0, h / 2.0, w, h, CORNER_RADIUS as f32);
        assert!(
            d < 0.0,
            "centre of badge should be inside (negative SDF), got {}",
            d
        );
    }

    #[test]
    fn rounded_rect_sdf_outside_is_positive() {
        let w = BADGE_W as f32;
        let h = BADGE_H as f32;
        // Well outside the badge.
        let d = rounded_rect_sdf(w + 10.0, h + 10.0, w, h, CORNER_RADIUS as f32);
        assert!(
            d > 0.0,
            "point outside badge should have positive SDF, got {}",
            d
        );
    }

    #[test]
    fn draw_rounded_border_produces_border_pixels() {
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);
        draw_rounded_border(&mut pb, BORDER_COLOR, CORNER_RADIUS as f32, BORDER_WIDTH);

        // The top edge centre should have border pixels.
        let mid_x = BADGE_W as usize / 2;
        let off = mid_x * 4;
        assert!(
            pb.data[off + 3] > 0,
            "top edge centre should have non-zero alpha from border"
        );

        // Interior centre should be transparent (no border there).
        let cx = BADGE_W as usize / 2;
        let cy = BADGE_H as usize / 2;
        let off_centre = (cy * pb.width + cx) * 4;
        assert_eq!(
            pb.data[off_centre + 3],
            0,
            "badge interior should remain transparent"
        );
    }

    // ── Full badge render (integration) ──────────────────────────────

    #[test]
    fn full_badge_render_produces_content() {
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);
        draw_rounded_border(&mut pb, BORDER_COLOR, CORNER_RADIUS as f32, BORDER_WIDTH);
        draw_pulsing_dot(&mut pb, DOT_CX, DOT_CY, DOT_RADIUS_MAX, 0.7);

        let history: Vec<Vec<f32>> = (0..SPEC_W)
            .map(|i| vec![(i as f32 * 0.05).sin().abs(); SPEC_H])
            .collect();
        render_spectrogram(
            &mut pb, &history, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 1.0, None,
        );

        // Count pixels with non-zero alpha (visible content).
        let visible = pb.data.chunks_exact(4).filter(|p| p[3] > 0).count();

        assert!(
            visible > 500,
            "full badge render should produce significant visible content, got {} pixels with alpha > 0",
            visible
        );
    }
}
