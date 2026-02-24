//! Real-time audio visualizer overlays (amplitude history + FFT spectrum)
//! and live transcription text display.
//!
//! Displays small panels on either side of the recording badge using X11
//! `put_image` for efficient rendering at 60 fps.  Each panel is
//! independently toggleable:
//!
//! - **Amplitude** (left of badge): RMS volume history over time.
//! - **Spectrum** (right of badge): FFT frequency-domain bar chart.
//! - **Text** (below badge): live transcription text as words arrive.
//!
//! The visualizer opens its own CPAL capture stream so it is fully
//! decoupled from the recording pipeline.

use crate::error::TalkError;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use x11rb::connection::Connection;
use x11rb::protocol::shape::{self, ConnectionExt as _};
use x11rb::protocol::xproto::*;
use x11rb::COPY_DEPTH_FROM_PARENT;

// ── Layout constants ─────────────────────────────────────────────────

/// Width of each visualizer panel in pixels.
const VIS_W: u16 = 200;
/// Height matches the recording badge (52 px).
const VIS_H: u16 = 52;
/// Horizontal gap between badge edge and visualizer panel.
const GAP: i16 = 4;
/// Vertical offset from the monitor top edge (matches badge placement).
const TOP_OFFSET: i16 = 4;
/// Target frames per second for the render loop.
const FPS: u32 = 60;
/// Amplitude history window in seconds.  Only the most recent
/// `AMP_WINDOW_SECS` of RMS values are kept; older data scrolls off.
const AMP_WINDOW_SECS: f32 = 5.0;
/// Number of audio samples fed into the FFT (must be power of two).
const FFT_SIZE: usize = 1024;

/// Per-frame decay multiplier for the all-time peak.
///
/// Peaks jump up instantly (fast attack) but shrink by this factor each
/// frame (slow release).  At 60 fps, 0.998^60 ≈ 0.887, so a transient
/// spike halves in ~6 seconds while sustained speech keeps the peak
/// appropriately high.
const PEAK_DECAY: f32 = 0.998;

/// Minimum peak floor to avoid division by near-zero.
const PEAK_FLOOR: f32 = 0.0001;

/// Width of the text overlay in pixels.
const TEXT_W: u16 = 1200;
/// Height of the text overlay in pixels.
const TEXT_H: u16 = 44;
/// Vertical gap between badge bottom and text overlay top.
const TEXT_GAP: i16 = 4;
/// Font size for transcription text in pixels.
const TEXT_FONT_SIZE: f32 = 30.0;
/// Corner radius for the text overlay background in pixels.
const TEXT_CORNER_RADIUS: usize = 10;

// ── Colors (BGRA for little-endian ZPixmap, depth 24/32) ─────────────

const BG: [u8; 4] = [0x00, 0x00, 0x00, 0xFF]; // #000000 black
const TEXT_COLOR: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF]; // #ffffff white
/// Brightest dot colour (BGRA).
const DOT_HI: [u8; 4] = [0xCC, 0xCC, 0xCC, 0xFF]; // #cccccc
/// Dimmest dot colour (BGRA).
const DOT_LO: [u8; 4] = [0x33, 0x33, 0x33, 0xFF]; // #333333
/// Full wave-cycle length in frames (90 frames ≈ 1.5 s at 60 fps).
const DOT_CYCLE_FRAMES: f32 = 90.0;
const AMP_COLOR: [u8; 4] = [0x88, 0xFF, 0x00, 0xFF]; // #00ff88 bright green
const AMP_DIM: [u8; 4] = [0x44, 0x88, 0x00, 0xFF]; // #008844 dim green

/// Gradient stops for spectrum bars (low → high frequency).
const SPEC_COLORS: [[u8; 4]; 5] = [
    [0xFF, 0xCC, 0x00, 0xFF], // #00ccff cyan
    [0xFF, 0xFF, 0x00, 0xFF], // #00ffff aqua
    [0x44, 0xFF, 0x00, 0xFF], // #00ff44 green
    [0x00, 0xCC, 0xFF, 0xFF], // #ffcc00 amber
    [0x44, 0x44, 0xFF, 0xFF], // #ff4444 red
];

// ── Ring buffer ──────────────────────────────────────────────────────

struct RingBuffer {
    data: Vec<f32>,
    write_pos: usize,
    capacity: usize,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            write_pos: 0,
            capacity,
        }
    }

    fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            self.data[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
    }

    fn read_last(&self, n: usize) -> Vec<f32> {
        let n = n.min(self.capacity);
        let mut out = Vec::with_capacity(n);
        let start = (self.write_pos + self.capacity - n) % self.capacity;
        for i in 0..n {
            out.push(self.data[(start + i) % self.capacity]);
        }
        out
    }
}

// ── Complex type + radix-2 FFT ──────────────────────────────────────

#[derive(Clone, Copy)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    fn magnitude(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

/// In-place iterative radix-2 Cooley-Tukey FFT.
fn fft_in_place(buf: &mut [Complex]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f32::consts::PI / len as f32;
        let wn = Complex::new(angle.cos(), angle.sin());

        let mut start = 0;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[start + k];
                let v = buf[start + k + half] * w;
                buf[start + k] = u + v;
                buf[start + k + half] = u - v;
                w = w * wn;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Apply Hann window and compute FFT magnitude spectrum (positive
/// frequencies only).
fn compute_spectrum(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    let mut buf: Vec<Complex> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos());
            Complex::new(s * w, 0.0)
        })
        .collect();

    fft_in_place(&mut buf);

    buf[..n / 2].iter().map(|c| c.magnitude()).collect()
}

// ── Pixel buffer ─────────────────────────────────────────────────────

struct PixelBuffer {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

impl PixelBuffer {
    fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0u8; width * height * 4],
            width,
            height,
        }
    }

    fn clear(&mut self, color: [u8; 4]) {
        for pixel in self.data.chunks_exact_mut(4) {
            pixel.copy_from_slice(&color);
        }
    }

    fn set_pixel(&mut self, x: usize, y: usize, color: [u8; 4]) {
        if x < self.width && y < self.height {
            let off = (y * self.width + x) * 4;
            self.data[off..off + 4].copy_from_slice(&color);
        }
    }

    /// Fill the buffer with `bg` everywhere *except* within a rounded
    /// rectangle that spans the full buffer, which gets `fg`.  Pixels
    /// outside the rounded corners are left transparent (all zeroes) so
    /// the window compositor shows through.
    fn clear_rounded(&mut self, fg: [u8; 4], radius: usize) {
        let w = self.width;
        let h = self.height;
        let r = radius.min(w / 2).min(h / 2);
        let r2 = (r * r) as i64;

        // Start fully transparent.
        for b in self.data.iter_mut() {
            *b = 0;
        }

        for y in 0..h {
            for x in 0..w {
                let inside = if x < r && y < r {
                    // top-left corner
                    let dx = r as i64 - x as i64;
                    let dy = r as i64 - y as i64;
                    dx * dx + dy * dy <= r2
                } else if x >= w - r && y < r {
                    // top-right corner
                    let dx = x as i64 - (w - r - 1) as i64;
                    let dy = r as i64 - y as i64;
                    dx * dx + dy * dy <= r2
                } else if x < r && y >= h - r {
                    // bottom-left corner
                    let dx = r as i64 - x as i64;
                    let dy = y as i64 - (h - r - 1) as i64;
                    dx * dx + dy * dy <= r2
                } else if x >= w - r && y >= h - r {
                    // bottom-right corner
                    let dx = x as i64 - (w - r - 1) as i64;
                    let dy = y as i64 - (h - r - 1) as i64;
                    dx * dx + dy * dy <= r2
                } else {
                    true
                };

                if inside {
                    let off = (y * w + x) * 4;
                    self.data[off..off + 4].copy_from_slice(&fg);
                }
            }
        }
    }

    fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: [u8; 4]) {
        for dy in 0..h {
            for dx in 0..w {
                self.set_pixel(x + dx, y + dy, color);
            }
        }
    }
}

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
/// vertically centred, clipped to buffer bounds.
fn blit_glyphs(
    pb: &mut PixelBuffer,
    glyphs: &[(fontdue::Metrics, Vec<u8>)],
    start_x: i32,
    color: [u8; 4],
) {
    let h = pb.height;
    let w = pb.width;
    let baseline = (h as i32 * 3) / 4;
    let mut cursor_x = start_x;

    for (metrics, bitmap) in glyphs {
        blit_glyph_at(pb, metrics, bitmap, cursor_x, baseline, w, h, color);
        cursor_x += metrics.advance_width as i32;
    }
}

/// Blit a single rasterised glyph at `cursor_x` with the given colour.
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
                let a = alpha as f32 / 255.0;
                for (c, &fg_val) in color.iter().enumerate().take(3) {
                    let bg_val = pb.data[off + c] as f32;
                    pb.data[off + c] = (bg_val + (fg_val as f32 - bg_val) * a) as u8;
                }
            }
        }
    }
}

/// Linearly interpolate between two BGRA colours by factor `t` (0–1).
fn lerp_color(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t) as u8,
        0xFF,
    ]
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
    blit_glyphs(pb, visible_glyphs, text_start_x, TEXT_COLOR);

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
        );
        cursor_x += dot_adv as i32;
        if i < 2 {
            cursor_x += space_adv as i32;
        }
    }
}

// ── Rendering helpers ────────────────────────────────────────────────

/// Compute the root-mean-square of a slice of samples.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Render amplitude history as a symmetrical waveform.
///
/// Maps `history` (one RMS value per frame) onto the panel width.
/// Each column draws a bar centred on the vertical midline, extending
/// equally upward and downward — producing a mirror-image waveform.
/// Normalised against `max_rms` so speech fills the panel height and
/// silence stays thin.
fn render_amplitude(pb: &mut PixelBuffer, history: &[f32], max_rms: f32) {
    if history.is_empty() || max_rms < PEAK_FLOOR {
        return;
    }

    let h = pb.height;
    let w = pb.width;
    let n = history.len();
    let center = h / 2;
    // Maximum half-height leaves 1 px margin top and bottom.
    let max_half = center.saturating_sub(1);

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

        let color = if norm > 0.5 { AMP_COLOR } else { AMP_DIM };
        let top = center - half_height;
        let bottom = center + half_height;
        for y in top..=bottom {
            pb.set_pixel(col, y, color);
        }
    }
}

fn spectrum_color(t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let n = SPEC_COLORS.len() - 1;
    let idx = (t * n as f32).min(n as f32 - 0.001);
    let i = idx as usize;
    let frac = idx - i as f32;

    let a = SPEC_COLORS[i];
    let b = SPEC_COLORS[i + 1];

    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * frac) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * frac) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * frac) as u8,
        0xFF,
    ]
}

/// Render spectrum bars normalised against `peak` (the all-time maximum
/// magnitude).  When the user is silent, bars stay tiny because `peak`
/// retains the loudest value ever seen.
fn render_spectrum(pb: &mut PixelBuffer, magnitudes: &[f32], peak: f32) {
    if magnitudes.is_empty() {
        return;
    }

    let h = pb.height;
    let w = pb.width;

    // Use lower quarter of spectrum (voice content).
    let useful = &magnitudes[..magnitudes.len() / 4];

    let num_bars = w / 3; // 2px bar + 1px gap
    if num_bars == 0 {
        return;
    }
    let bins_per_bar = useful.len() / num_bars;
    if bins_per_bar == 0 {
        return;
    }

    for bar in 0..num_bars {
        let start = bar * bins_per_bar;
        let end = (start + bins_per_bar).min(useful.len());

        let avg = if end > start {
            useful[start..end].iter().sum::<f32>() / (end - start) as f32
        } else {
            0.0
        };

        // Normalise against all-time peak → noise stays small once
        // the user has spoken.
        let norm = (avg / peak).clamp(0.0, 1.0);
        let log_norm = (1.0 + norm * 9.0).log10(); // log10(1..10) → 0..1

        let bar_height = (log_norm * (h - 4) as f32) as usize;
        let bar_x = bar * 3;
        let bar_y = h - 2 - bar_height;

        let color = spectrum_color(bar as f32 / num_bars as f32);
        pb.fill_rect(bar_x, bar_y, 2, bar_height, color);
    }
}

// ── Public API ───────────────────────────────────────────────────────

/// Commands from the main thread to the visualizer thread.
enum VizCommand {
    /// Show the visualizer panels (badge width needed for positioning).
    Show { badge_width: u16 },
    /// Hide visualizer panels.
    Hide,
    /// Update the live transcription text displayed below the badge.
    Text { text: String },
    /// Shut down the thread.
    Quit,
}

/// Handle to the visualizer background thread.
///
/// The thread owns its own CPAL capture and X11 connection.  Sending
/// [`show`](VisualizerHandle::show) positions the panels relative to
/// the recording badge; [`hide`](VisualizerHandle::hide) tears them
/// down.  Dropping the handle stops the thread.
pub struct VisualizerHandle {
    tx: std::sync::mpsc::Sender<VizCommand>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl VisualizerHandle {
    /// Spawn the visualizer thread.
    ///
    /// * `amplitude` — enable the amplitude history panel (left of badge).
    /// * `spectrum` — enable the spectrum panel (right of badge).
    /// * `text` — enable the live transcription text panel (below badge).
    ///
    /// Returns `Err` if X11 or the audio device is unreachable.
    pub fn new(amplitude: bool, spectrum: bool, text: bool) -> Result<Self, TalkError> {
        let geom = super::monitor::primary_monitor_geometry()?;
        let (tx, rx) = std::sync::mpsc::channel();

        let thread = std::thread::Builder::new()
            .name("visualizer".into())
            .spawn(move || {
                if let Err(e) = visualizer_thread(rx, amplitude, spectrum, text, geom) {
                    log::error!("visualizer thread error: {}", e);
                }
            })
            .map_err(|e| TalkError::Audio(format!("failed to spawn visualizer thread: {}", e)))?;

        Ok(Self {
            tx,
            thread: Some(thread),
        })
    }

    /// Show visualizer panels positioned relative to a badge of the
    /// given pixel width.
    pub fn show(&self, badge_width: u16) {
        let _ = self.tx.send(VizCommand::Show { badge_width });
    }

    /// Hide visualizer panels.
    pub fn hide(&self) {
        let _ = self.tx.send(VizCommand::Hide);
    }

    /// Update the live transcription text shown below the badge.
    pub fn set_text(&self, text: &str) {
        let _ = self.tx.send(VizCommand::Text {
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

/// Main visualizer thread.
///
/// Owns both a CPAL input stream (for audio data) and an X11
/// connection (for rendering).  Blocks on commands when hidden;
/// renders at [`FPS`] when shown.
fn visualizer_thread(
    rx: std::sync::mpsc::Receiver<VizCommand>,
    enable_amplitude: bool,
    enable_spectrum: bool,
    enable_text: bool,
    geom: super::monitor::MonitorGeometry,
) -> Result<(), TalkError> {
    // ── Audio capture ────────────────────────────────────────────────

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| TalkError::Audio("no default input device for visualizer".into()))?;

    let supported = device
        .default_input_config()
        .map_err(|e| TalkError::Audio(format!("visualizer input config: {}", e)))?;

    let channels = supported.channels() as usize;

    let ring = Arc::new(Mutex::new(RingBuffer::new(
        supported.sample_rate().0 as usize / 2,
    )));
    let ring_w = Arc::clone(&ring);

    let stream = device
        .build_input_stream(
            &cpal::StreamConfig {
                channels: supported.channels(),
                sample_rate: supported.sample_rate(),
                buffer_size: cpal::BufferSize::Default,
            },
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mono: Vec<f32> = if channels == 1 {
                    data.to_vec()
                } else {
                    data.chunks_exact(channels)
                        .map(|f| f.iter().sum::<f32>() / channels as f32)
                        .collect()
                };
                if let Ok(mut g) = ring_w.lock() {
                    g.push(&mono);
                }
            },
            |e| log::error!("visualizer audio error: {}", e),
            None,
        )
        .map_err(|e| TalkError::Audio(format!("visualizer stream: {}", e)))?;

    stream
        .play()
        .map_err(|e| TalkError::Audio(format!("visualizer stream play: {}", e)))?;

    // ── X11 connection ───────────────────────────────────────────────

    let (conn, screen_num) = x11rb::connect(None)
        .map_err(|e| TalkError::Config(format!("visualizer X11 connect: {}", e)))?;

    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;
    let depth = screen.root_depth;

    let (mon_x, mon_y, mon_w, _mon_h) = geom;

    // ── Pixel buffers ────────────────────────────────────────────────

    let mut amp_pb = PixelBuffer::new(VIS_W as usize, VIS_H as usize);
    let mut sp_pb = PixelBuffer::new(VIS_W as usize, VIS_H as usize);
    let mut text_pb = PixelBuffer::new(TEXT_W as usize, TEXT_H as usize);

    // ── Font (loaded in background to avoid blocking panel startup) ──

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
    let rms_chunk = supported.sample_rate().0 as usize / FPS as usize;

    let mut wins = WindowState {
        amplitude_win: None,
        spectrum_win: None,
        amplitude_gc: None,
        spectrum_gc: None,
        text_win: None,
        text_gc: None,
    };
    let mut is_showing = false;
    let mut current_text = String::new();
    let mut frame_counter: u64 = 0;

    // Amplitude history: one RMS value per frame, capped to a
    // rolling window of `AMP_WINDOW_SECS` seconds.
    let amp_max_frames = (FPS as f32 * AMP_WINDOW_SECS) as usize;
    let mut amp_history: Vec<f32> = vec![0.0; amp_max_frames];
    // All-time peak RMS — only grows, never shrinks.  Gives a stable
    // Y-axis scale: once the user has spoken, silence stays small.
    let mut amp_peak: f32 = PEAK_FLOOR;

    // Spectrum peak: fast attack, slow decay.
    let mut spectrum_peak: f32 = 0.001;

    // ── Event loop ───────────────────────────────────────────────────

    loop {
        // When hidden, block on commands (no rendering needed).
        if !is_showing {
            match rx.recv() {
                Ok(VizCommand::Show { badge_width }) => {
                    amp_history.clear();
                    amp_history.resize(amp_max_frames, 0.0);
                    create_windows(
                        &conn,
                        screen,
                        root,
                        depth,
                        mon_x,
                        mon_y,
                        mon_w,
                        badge_width,
                        enable_amplitude,
                        enable_spectrum,
                        enable_text,
                        &mut wins,
                    )?;
                    is_showing = true;
                }
                Ok(VizCommand::Hide) => {}
                Ok(VizCommand::Text { text }) => {
                    current_text = text;
                }
                Ok(VizCommand::Quit) | Err(_) => break,
            }
            continue;
        }

        // When shown, drain commands without blocking.
        loop {
            match rx.try_recv() {
                Ok(VizCommand::Hide) => {
                    destroy_windows(&conn, &mut wins);
                    is_showing = false;
                }
                Ok(VizCommand::Text { text }) => {
                    current_text = text;
                }
                Ok(VizCommand::Show { badge_width }) => {
                    destroy_windows(&conn, &mut wins);
                    amp_history.clear();
                    amp_history.resize(amp_max_frames, 0.0);
                    create_windows(
                        &conn,
                        screen,
                        root,
                        depth,
                        mon_x,
                        mon_y,
                        mon_w,
                        badge_width,
                        enable_amplitude,
                        enable_spectrum,
                        enable_text,
                        &mut wins,
                    )?;
                }
                Ok(VizCommand::Quit) => {
                    destroy_windows(&conn, &mut wins);
                    return Ok(());
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    destroy_windows(&conn, &mut wins);
                    return Ok(());
                }
            }
        }

        if !is_showing {
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

        let samples = ring
            .lock()
            .map(|g| g.read_last(FFT_SIZE.max(rms_chunk)))
            .unwrap_or_else(|_| vec![0.0; FFT_SIZE.max(rms_chunk)]);

        // Compute RMS for this frame, update all-time peak, and
        // append to the rolling history window.
        let frame_rms = rms(&samples[samples.len().saturating_sub(rms_chunk)..]);
        if frame_rms > amp_peak {
            amp_peak = frame_rms;
        }
        amp_history.push(frame_rms);
        if amp_history.len() > amp_max_frames {
            amp_history.drain(..amp_history.len() - amp_max_frames);
        }

        // Amplitude panel
        if let (Some(win), Some(gc)) = (wins.amplitude_win, wins.amplitude_gc) {
            amp_pb.clear(BG);
            render_amplitude(&mut amp_pb, &amp_history, amp_peak);

            let _ = conn.put_image(
                ImageFormat::Z_PIXMAP,
                win,
                gc,
                VIS_W,
                VIS_H,
                0,
                0,
                0,
                depth,
                &amp_pb.data,
            );
        }

        // Spectrum panel
        if let (Some(win), Some(gc)) = (wins.spectrum_win, wins.spectrum_gc) {
            let magnitudes = compute_spectrum(&samples);

            // Update spectrum peak: fast attack, slow decay.
            spectrum_peak *= PEAK_DECAY;
            let frame_peak = magnitudes.iter().copied().fold(0.0f32, f32::max);
            if frame_peak > spectrum_peak {
                spectrum_peak = frame_peak;
            }
            spectrum_peak = spectrum_peak.max(PEAK_FLOOR);

            sp_pb.clear(BG);
            render_spectrum(&mut sp_pb, &magnitudes, spectrum_peak);

            let _ = conn.put_image(
                ImageFormat::Z_PIXMAP,
                win,
                gc,
                VIS_W,
                VIS_H,
                0,
                0,
                0,
                depth,
                &sp_pb.data,
            );
        }

        // Text panel — always re-render for smooth dot wave animation.
        if let (Some(win), Some(gc), Some(ref f)) = (wins.text_win, wins.text_gc, &font) {
            let dot_br = dot_wave(frame_counter);
            text_pb.clear_rounded(BG, TEXT_CORNER_RADIUS);
            render_text(&mut text_pb, &current_text, f, dot_br);

            let _ = conn.put_image(
                ImageFormat::Z_PIXMAP,
                win,
                gc,
                TEXT_W,
                TEXT_H,
                0,
                0,
                0,
                depth,
                &text_pb.data,
            );
        }
        frame_counter += 1;

        let _ = conn.flush();

        let elapsed = frame_start.elapsed();
        if elapsed < frame_dur {
            std::thread::sleep(frame_dur - elapsed);
        }
    }

    Ok(())
}

// ── Window helpers ───────────────────────────────────────────────────

struct WindowState {
    amplitude_win: Option<Window>,
    spectrum_win: Option<Window>,
    amplitude_gc: Option<Gcontext>,
    spectrum_gc: Option<Gcontext>,
    text_win: Option<Window>,
    text_gc: Option<Gcontext>,
}

#[allow(clippy::too_many_arguments)]
fn create_windows(
    conn: &impl Connection,
    screen: &Screen,
    root: Window,
    _depth: u8,
    mon_x: i16,
    mon_y: i16,
    mon_w: u16,
    badge_width: u16,
    enable_amplitude: bool,
    enable_spectrum: bool,
    enable_text: bool,
    state: &mut WindowState,
) -> Result<(), TalkError> {
    // Badge position (must match overlay.rs logic)
    let badge_x = mon_x + (mon_w as i16 / 2) - (badge_width as i16 / 2);
    let badge_y = mon_y + TOP_OFFSET;

    let values = CreateWindowAux::new()
        .background_pixel(screen.black_pixel)
        .border_pixel(0)
        .override_redirect(1u32)
        .event_mask(EventMask::EXPOSURE);

    if enable_amplitude {
        let x = badge_x - GAP - VIS_W as i16;
        let win = conn
            .generate_id()
            .map_err(|e| TalkError::Config(format!("X11 id: {}", e)))?;

        conn.create_window(
            COPY_DEPTH_FROM_PARENT,
            win,
            root,
            x,
            badge_y,
            VIS_W,
            VIS_H,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &values,
        )
        .map_err(|e| TalkError::Config(format!("X11 create amplitude win: {}", e)))?;

        conn.map_window(win)
            .map_err(|e| TalkError::Config(format!("X11 map amplitude win: {}", e)))?;

        let gc = conn
            .generate_id()
            .map_err(|e| TalkError::Config(format!("X11 id: {}", e)))?;
        conn.create_gc(gc, win, &CreateGCAux::new())
            .map_err(|e| TalkError::Config(format!("X11 gc: {}", e)))?;

        state.amplitude_win = Some(win);
        state.amplitude_gc = Some(gc);
    }

    if enable_spectrum {
        let x = badge_x + badge_width as i16 + GAP;
        let win = conn
            .generate_id()
            .map_err(|e| TalkError::Config(format!("X11 id: {}", e)))?;

        conn.create_window(
            COPY_DEPTH_FROM_PARENT,
            win,
            root,
            x,
            badge_y,
            VIS_W,
            VIS_H,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &values,
        )
        .map_err(|e| TalkError::Config(format!("X11 create spectrum win: {}", e)))?;

        conn.map_window(win)
            .map_err(|e| TalkError::Config(format!("X11 map spectrum win: {}", e)))?;

        let gc = conn
            .generate_id()
            .map_err(|e| TalkError::Config(format!("X11 id: {}", e)))?;
        conn.create_gc(gc, win, &CreateGCAux::new())
            .map_err(|e| TalkError::Config(format!("X11 gc: {}", e)))?;

        state.spectrum_win = Some(win);
        state.spectrum_gc = Some(gc);
    }

    // Text window — centred below the badge (only in realtime mode).
    if enable_text {
        let text_x = mon_x + (mon_w as i16 / 2) - (TEXT_W as i16 / 2);
        let text_y = badge_y + VIS_H as i16 + TEXT_GAP;

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
            TEXT_H,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &values,
        )
        .map_err(|e| TalkError::Config(format!("X11 create text win: {}", e)))?;

        conn.map_window(win)
            .map_err(|e| TalkError::Config(format!("X11 map text win: {}", e)))?;

        let gc = conn
            .generate_id()
            .map_err(|e| TalkError::Config(format!("X11 id: {}", e)))?;
        conn.create_gc(gc, win, &CreateGCAux::new())
            .map_err(|e| TalkError::Config(format!("X11 gc: {}", e)))?;

        // Apply rounded-corner shape mask via the XShape extension.
        apply_rounded_shape(conn, win, TEXT_W, TEXT_H, TEXT_CORNER_RADIUS)?;

        state.text_win = Some(win);
        state.text_gc = Some(gc);
    }

    conn.flush()
        .map_err(|e| TalkError::Config(format!("X11 flush: {}", e)))?;

    Ok(())
}

/// Create a 1-bit pixmap with a rounded rectangle and apply it as the
/// window's bounding shape so corners are truly transparent.
fn apply_rounded_shape(
    conn: &impl Connection,
    win: Window,
    w: u16,
    h: u16,
    radius: usize,
) -> Result<(), TalkError> {
    let pixmap: Pixmap = conn
        .generate_id()
        .map_err(|e| TalkError::Config(format!("X11 id: {}", e)))?;
    conn.create_pixmap(1, pixmap, win, w, h)
        .map_err(|e| TalkError::Config(format!("X11 create pixmap: {}", e)))?;

    let gc: Gcontext = conn
        .generate_id()
        .map_err(|e| TalkError::Config(format!("X11 id: {}", e)))?;
    conn.create_gc(gc, pixmap, &CreateGCAux::new().foreground(0))
        .map_err(|e| TalkError::Config(format!("X11 gc: {}", e)))?;

    // Clear to 0 (fully transparent).
    conn.poly_fill_rectangle(
        pixmap,
        gc,
        &[Rectangle {
            x: 0,
            y: 0,
            width: w,
            height: h,
        }],
    )
    .map_err(|e| TalkError::Config(format!("X11 fill: {}", e)))?;

    // Draw the rounded rectangle in foreground = 1 (opaque).
    conn.change_gc(gc, &ChangeGCAux::new().foreground(1))
        .map_err(|e| TalkError::Config(format!("X11 change gc: {}", e)))?;

    let r = (radius as u16).min(w / 2).min(h / 2);
    let d = r * 2;

    // Centre rectangle (full width minus corners).
    conn.poly_fill_rectangle(
        pixmap,
        gc,
        &[
            // Horizontal band spanning full width, excluding top/bottom
            // corner rows.
            Rectangle {
                x: 0,
                y: r as i16,
                width: w,
                height: h - d,
            },
            // Top band between corners.
            Rectangle {
                x: r as i16,
                y: 0,
                width: w - d,
                height: r,
            },
            // Bottom band between corners.
            Rectangle {
                x: r as i16,
                y: (h - r) as i16,
                width: w - d,
                height: r,
            },
        ],
    )
    .map_err(|e| TalkError::Config(format!("X11 fill: {}", e)))?;

    // Four corner arcs (angles in 64ths of a degree).
    // Use fully-qualified name because `xproto::Arc` is shadowed by
    // `std::sync::Arc` in scope.
    conn.poly_fill_arc(
        pixmap,
        gc,
        &[
            // top-left
            x11rb::protocol::xproto::Arc {
                x: 0,
                y: 0,
                width: d,
                height: d,
                angle1: 90 * 64,
                angle2: 90 * 64,
            },
            // top-right
            x11rb::protocol::xproto::Arc {
                x: (w - d) as i16,
                y: 0,
                width: d,
                height: d,
                angle1: 0,
                angle2: 90 * 64,
            },
            // bottom-left
            x11rb::protocol::xproto::Arc {
                x: 0,
                y: (h - d) as i16,
                width: d,
                height: d,
                angle1: 180 * 64,
                angle2: 90 * 64,
            },
            // bottom-right
            x11rb::protocol::xproto::Arc {
                x: (w - d) as i16,
                y: (h - d) as i16,
                width: d,
                height: d,
                angle1: 270 * 64,
                angle2: 90 * 64,
            },
        ],
    )
    .map_err(|e| TalkError::Config(format!("X11 fill arc: {}", e)))?;

    // Apply as bounding shape.
    conn.shape_mask(shape::SO::SET, shape::SK::BOUNDING, win, 0, 0, pixmap)
        .map_err(|e| TalkError::Config(format!("X11 shape mask: {}", e)))?;

    conn.free_gc(gc)
        .map_err(|e| TalkError::Config(format!("X11 free gc: {}", e)))?;
    conn.free_pixmap(pixmap)
        .map_err(|e| TalkError::Config(format!("X11 free pixmap: {}", e)))?;

    Ok(())
}

fn destroy_windows(conn: &impl Connection, state: &mut WindowState) {
    if let Some(gc) = state.amplitude_gc.take() {
        let _ = conn.free_gc(gc);
    }
    if let Some(win) = state.amplitude_win.take() {
        let _ = conn.destroy_window(win);
    }
    if let Some(gc) = state.spectrum_gc.take() {
        let _ = conn.free_gc(gc);
    }
    if let Some(win) = state.spectrum_win.take() {
        let _ = conn.destroy_window(win);
    }
    if let Some(gc) = state.text_gc.take() {
        let _ = conn.free_gc(gc);
    }
    if let Some(win) = state.text_win.take() {
        let _ = conn.destroy_window(win);
    }
    let _ = conn.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Ring buffer ──────────────────────────────────────────────────

    #[test]
    fn ring_buffer_push_and_read() {
        let mut rb = RingBuffer::new(8);
        rb.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = rb.read_last(5);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn ring_buffer_wraps_around() {
        let mut rb = RingBuffer::new(4);
        rb.push(&[1.0, 2.0, 3.0, 4.0]);
        rb.push(&[5.0, 6.0]);
        let out = rb.read_last(4);
        assert_eq!(out, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn ring_buffer_read_more_than_capacity() {
        let mut rb = RingBuffer::new(4);
        rb.push(&[1.0, 2.0]);
        let out = rb.read_last(10);
        // Clamped to capacity
        assert_eq!(out.len(), 4);
    }

    // ── FFT ──────────────────────────────────────────────────────────

    #[test]
    fn fft_dc_signal() {
        // A constant signal should have all energy in bin 0.
        let mut buf: Vec<Complex> = (0..16).map(|_| Complex::new(1.0, 0.0)).collect();
        fft_in_place(&mut buf);

        let dc = buf[0].magnitude();
        let rest_max = buf[1..]
            .iter()
            .map(|c| c.magnitude())
            .fold(0.0f32, f32::max);

        assert!(
            dc > rest_max * 100.0,
            "DC bin should dominate: dc={}, rest_max={}",
            dc,
            rest_max
        );
    }

    #[test]
    fn fft_sine_peak() {
        // 4 Hz sine in a 16-sample buffer at 16 Hz sample rate →
        // energy should be concentrated in bin 4.
        let n = 16usize;
        let freq = 4.0;
        let mut buf: Vec<Complex> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                Complex::new((2.0 * std::f32::consts::PI * freq * t).sin(), 0.0)
            })
            .collect();
        fft_in_place(&mut buf);

        let peak_bin = buf[..n / 2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.magnitude().partial_cmp(&b.1.magnitude()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(peak_bin, 4);
    }

    #[test]
    fn compute_spectrum_returns_half_length() {
        let samples = vec![0.0f32; 256];
        let mags = compute_spectrum(&samples);
        assert_eq!(mags.len(), 128);
    }

    // ── Spectrum peak tracking ───────────────────────────────────────

    #[test]
    fn peak_fast_attack_slow_decay() {
        let mut peak: f32 = PEAK_FLOOR;

        // Loud frame — peak jumps instantly
        peak *= PEAK_DECAY;
        let loud = 5.0;
        if loud > peak {
            peak = loud;
        }
        peak = peak.max(PEAK_FLOOR);
        assert!((peak - 5.0).abs() < f32::EPSILON);

        // Quiet frame — peak decays slightly
        peak *= PEAK_DECAY;
        let quiet = 0.001;
        if quiet > peak {
            peak = quiet;
        }
        peak = peak.max(PEAK_FLOOR);
        assert!(
            peak < 5.0 && peak > 4.9,
            "peak should have decayed slightly: {}",
            peak
        );

        // After many quiet frames, peak shrinks substantially
        for _ in 0..600 {
            peak *= PEAK_DECAY;
            peak = peak.max(PEAK_FLOOR);
        }
        assert!(
            peak < 2.0,
            "peak should have decayed after 600 frames: {}",
            peak
        );
    }

    // ── Pixel buffer ─────────────────────────────────────────────────

    #[test]
    fn pixel_buffer_clear() {
        let mut pb = PixelBuffer::new(4, 4);
        pb.clear([0xFF, 0x00, 0x00, 0xFF]);
        // Every pixel should be the clear color
        for chunk in pb.data.chunks_exact(4) {
            assert_eq!(chunk, &[0xFF, 0x00, 0x00, 0xFF]);
        }
    }

    #[test]
    fn pixel_buffer_set_pixel_bounds() {
        let mut pb = PixelBuffer::new(4, 4);
        pb.clear([0; 4]);
        pb.set_pixel(3, 3, [1, 2, 3, 4]);
        let off = (3 * 4 + 3) * 4;
        assert_eq!(&pb.data[off..off + 4], &[1, 2, 3, 4]);

        // Out of bounds — should not panic
        pb.set_pixel(10, 10, [0xFF; 4]);
    }

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

    // ── RMS helper ────────────────────────────────────────────────────

    #[test]
    fn rms_of_silence_is_zero() {
        assert!((rms(&[0.0; 100])).abs() < f32::EPSILON);
    }

    #[test]
    fn rms_of_constant_is_value() {
        let val = 0.5f32;
        let samples = vec![val; 200];
        assert!((rms(&samples) - val).abs() < 1e-6);
    }

    #[test]
    fn rms_empty_is_zero() {
        assert!((rms(&[])).abs() < f32::EPSILON);
    }

    // ── Rendering (smoke tests — no panic) ───────────────────────────

    #[test]
    fn render_amplitude_no_panic() {
        let mut pb = PixelBuffer::new(200, 52);
        pb.clear(BG);
        // Simulate 60 frames of varying amplitude
        let history: Vec<f32> = (0..60).map(|i| (i as f32 * 0.1).sin().abs()).collect();
        let max_rms = history.iter().copied().fold(0.0f32, f32::max);
        render_amplitude(&mut pb, &history, max_rms);

        // Should have non-background pixels
        let non_bg = pb.data.chunks_exact(4).filter(|p| *p != BG).count();
        assert!(non_bg > 0);
    }

    #[test]
    fn render_amplitude_silent_stays_tiny() {
        let mut pb = PixelBuffer::new(200, 52);
        pb.clear(BG);
        // Very quiet history with a high max
        let history = vec![0.0001f32; 60];
        render_amplitude(&mut pb, &history, 1.0);

        // Almost everything should still be background
        let non_bg = pb.data.chunks_exact(4).filter(|p| *p != BG).count();
        let total = pb.width * pb.height;
        assert!(
            non_bg < total / 4,
            "silent amplitude should be mostly empty, got {}/{} non-bg",
            non_bg,
            total
        );
    }

    #[test]
    fn render_amplitude_prepadded_leaves_left_empty() {
        // Simulates the fixed behavior: a full-length history where
        // only the rightmost entries have data (the rest are zero).
        // The left portion of the panel must stay empty (all BG).
        let window_frames = (FPS as f32 * AMP_WINDOW_SECS) as usize; // 300
        let active_frames = 10;
        let mut history = vec![0.0f32; window_frames];
        for val in &mut history[window_frames - active_frames..] {
            *val = 0.8;
        }

        let mut pb = PixelBuffer::new(VIS_W as usize, VIS_H as usize);
        pb.clear(BG);
        render_amplitude(&mut pb, &history, 1.0);

        // The leftmost 80% of columns should have at most a 1-pixel
        // centre line per column (render_amplitude draws a min-height
        // bar at the midline even for zero RMS).  The right side
        // should have real symmetric bars.
        let check_cols = pb.width * 80 / 100;
        let mut left_non_bg = 0usize;
        for x in 0..check_cols {
            for y in 0..pb.height {
                let off = (y * pb.width + x) * 4;
                if pb.data[off..off + 4] != BG {
                    left_non_bg += 1;
                }
            }
        }
        // At most 1 pixel per column (the baseline).
        assert!(
            left_non_bg <= check_cols,
            "left 80% should only have baseline pixels, got {} non-bg for {} cols",
            left_non_bg,
            check_cols
        );

        // The rightmost columns must have substantially taller bars
        // (more than just the 1-pixel baseline).
        let right_cols = 5;
        let mut right_non_bg = 0usize;
        for x in (pb.width - right_cols)..pb.width {
            for y in 0..pb.height {
                let off = (y * pb.width + x) * 4;
                if pb.data[off..off + 4] != BG {
                    right_non_bg += 1;
                }
            }
        }
        assert!(
            right_non_bg > right_cols * 2,
            "rightmost columns should have amplitude bars taller than baseline, got {} pixels for {} cols",
            right_non_bg,
            right_cols
        );
    }

    #[test]
    fn render_amplitude_symmetric_around_center() {
        let mut pb = PixelBuffer::new(200, 52);
        pb.clear(BG);
        let history = vec![0.8f32; 60];
        render_amplitude(&mut pb, &history, 1.0);

        let center = pb.height / 2;
        // For each column, count non-bg pixels above and below centre.
        for col in 0..pb.width {
            let mut above = 0usize;
            let mut below = 0usize;
            for y in 0..center {
                let off = (y * pb.width + col) * 4;
                if pb.data[off..off + 4] != BG {
                    above += 1;
                }
            }
            // Skip the centre pixel itself (shared), count below.
            for y in (center + 1)..pb.height {
                let off = (y * pb.width + col) * 4;
                if pb.data[off..off + 4] != BG {
                    below += 1;
                }
            }
            assert_eq!(
                above, below,
                "column {} should be symmetric: {} above vs {} below centre",
                col, above, below
            );
        }
    }

    #[test]
    fn render_spectrum_no_panic() {
        let mut pb = PixelBuffer::new(200, 52);
        pb.clear(BG);
        let mags: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        render_spectrum(&mut pb, &mags, 5.0);

        let non_bg = pb.data.chunks_exact(4).filter(|p| *p != BG).count();
        assert!(non_bg > 0);
    }

    #[test]
    fn render_spectrum_silent_stays_tiny() {
        let mut pb = PixelBuffer::new(200, 52);
        pb.clear(BG);
        // Very quiet signal with a high all-time peak
        let mags: Vec<f32> = vec![0.001; 512];
        render_spectrum(&mut pb, &mags, 100.0);

        // Almost everything should still be background
        let non_bg = pb.data.chunks_exact(4).filter(|p| *p != BG).count();
        let total = pb.width * pb.height;
        assert!(
            non_bg < total / 10,
            "silent spectrum should be mostly empty, got {}/{} non-bg",
            non_bg,
            total
        );
    }
}
