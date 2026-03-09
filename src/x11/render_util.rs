//! Shared rendering and DSP utilities for X11 audio overlays.
//!
//! Contains the pixel buffer, ring buffer, FFT, and shape-mask
//! helpers used by both the recording overlay and the visualizer
//! panels.

use crate::error::TalkError;

// ── Constants ────────────────────────────────────────────────────────

/// Per-frame decay multiplier for tracked peaks.
///
/// Peaks jump up instantly (fast attack) but shrink by this factor each
/// frame (slow release).  At 60 fps, 0.998^60 ≈ 0.887, so a transient
/// spike halves in ~6 seconds while sustained speech keeps the peak
/// appropriately high.
pub const PEAK_DECAY: f32 = 0.998;

/// Minimum peak floor to avoid division by near-zero.
pub const PEAK_FLOOR: f32 = 0.0001;

// ── Ring buffer ──────────────────────────────────────────────────────

pub struct RingBuffer {
    data: Vec<f32>,
    write_pos: usize,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            write_pos: 0,
            capacity,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            self.data[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
    }

    pub fn read_last(&self, n: usize) -> Vec<f32> {
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
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn magnitude(self) -> f32 {
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
pub fn fft_in_place(buf: &mut [Complex]) {
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
pub fn compute_spectrum(samples: &[f32]) -> Vec<f32> {
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

pub struct PixelBuffer {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl PixelBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0u8; width * height * 4],
            width,
            height,
        }
    }

    pub fn clear(&mut self, color: [u8; 4]) {
        for pixel in self.data.chunks_exact_mut(4) {
            pixel.copy_from_slice(&color);
        }
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, color: [u8; 4]) {
        if x < self.width && y < self.height {
            let off = (y * self.width + x) * 4;
            self.data[off..off + 4].copy_from_slice(&color);
        }
    }

    /// Fill the buffer with `fg` inside a rounded rectangle that spans
    /// the full buffer.  Pixels outside the rounded corners are left
    /// transparent (all zeroes) so the window shape mask clips them.
    pub fn clear_rounded(&mut self, fg: [u8; 4], radius: usize) {
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

    pub fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: [u8; 4]) {
        for dy in 0..h {
            for dx in 0..w {
                self.set_pixel(x + dx, y + dy, color);
            }
        }
    }
}

// ── DSP helpers ──────────────────────────────────────────────────────

/// Compute the root-mean-square of a slice of samples.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Map a 0–1 normalized value to a heat-map BGRA colour (premultiplied
/// alpha).
///
/// The gradient goes: transparent → blue → cyan → green → yellow → red.
/// Alpha tracks `brightness` so quiet areas fade to transparent
/// (suitable for waterfall overlays on a dark badge).
pub fn heat_map_color(norm: f32, brightness: f32) -> [u8; 4] {
    let norm = norm.clamp(0.0, 1.0);
    let alpha = (brightness * 255.0).clamp(0.0, 255.0) as u8;
    if alpha == 0 {
        return [0, 0, 0, 0];
    }

    // Piecewise-linear RGB gradient (0.0–1.0 range).
    let (r, g, b) = if norm < 0.2 {
        let t = norm / 0.2;
        (0.0, 0.0, t)
    } else if norm < 0.4 {
        let t = (norm - 0.2) / 0.2;
        (0.0, t, 1.0)
    } else if norm < 0.6 {
        let t = (norm - 0.4) / 0.2;
        (0.0, 1.0, 1.0 - t)
    } else if norm < 0.8 {
        let t = (norm - 0.6) / 0.2;
        (t, 1.0, 0.0)
    } else {
        let t = (norm - 0.8) / 0.2;
        (1.0, 1.0 - t, 0.0)
    };

    // Premultiply by alpha (= brightness).
    let af = brightness;
    [
        (b * af * 255.0) as u8,
        (g * af * 255.0) as u8,
        (r * af * 255.0) as u8,
        alpha,
    ]
}

/// Map a 0–1 normalized level to an opaque BGRA bar colour.
///
/// The gradient goes: green (low) → yellow (mid) → red (high).
/// Returns fully opaque pixels suitable for amplitude / spectrum bars.
pub fn level_color(norm: f32) -> [u8; 4] {
    let norm = norm.clamp(0.0, 1.0);
    let (r, g) = if norm < 0.5 {
        let t = norm / 0.5;
        (t, 1.0)
    } else {
        let t = (norm - 0.5) / 0.5;
        (1.0, 1.0 - t)
    };
    // BGRA, fully opaque, blue channel is always 0.
    [0, (g * 255.0) as u8, (r * 255.0) as u8, 0xFF]
}

/// Linearly interpolate between two BGRA colours by factor `t` (0–1).
pub fn lerp_color(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t) as u8,
        0xFF,
    ]
}

// ── Theme detection ──────────────────────────────────────────────────

/// Detect whether the desktop theme is dark.
///
/// Uses the `dark-light` crate (freedesktop portal via D-Bus) with a
/// fallback to the `GTK_THEME` environment variable.  Returns `true`
/// for dark themes (use white foreground), `false` for light themes
/// (use black foreground).  Defaults to dark when detection fails.
pub fn detect_is_dark_theme() -> bool {
    match dark_light::detect() {
        Ok(dark_light::Mode::Light) => {
            log::debug!("theme detection: light (via dark-light)");
            false
        }
        Ok(dark_light::Mode::Dark) => {
            log::debug!("theme detection: dark (via dark-light)");
            true
        }
        _ => {
            // Fallback: check GTK_THEME for "dark" substring.
            if let Ok(theme) = std::env::var("GTK_THEME") {
                let lower = theme.to_ascii_lowercase();
                if lower.contains("dark") {
                    log::debug!("theme detection: dark (GTK_THEME={:?})", theme);
                    return true;
                }
                if lower.contains("light") {
                    log::debug!("theme detection: light (GTK_THEME={:?})", theme);
                    return false;
                }
            }
            log::debug!("theme detection: defaulting to dark");
            true
        }
    }
}

/// Return the BGRA foreground and background colours for monochrome
/// visualiser mode based on theme detection.
///
/// - Dark theme → white foreground on black background.
/// - Light theme → black foreground on white background.
pub fn monochrome_palette() -> ([u8; 4], [u8; 4]) {
    if detect_is_dark_theme() {
        // fg = white, bg = black
        ([0xFF, 0xFF, 0xFF, 0xFF], [0x00, 0x00, 0x00, 0xFF])
    } else {
        // fg = black, bg = white
        ([0x00, 0x00, 0x00, 0xFF], [0xFF, 0xFF, 0xFF, 0xFF])
    }
}

// ── X11 shape helpers ────────────────────────────────────────────────

/// Create a 1-bit pixmap with a rounded rectangle and apply it as the
/// window's bounding shape so corners are truly transparent.
pub fn apply_rounded_shape(
    conn: &impl x11rb::connection::Connection,
    win: u32,
    w: u16,
    h: u16,
    radius: usize,
) -> Result<(), TalkError> {
    use x11rb::protocol::shape;
    use x11rb::protocol::xproto::*;

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
    shape::mask(conn, shape::SO::SET, shape::SK::BOUNDING, win, 0, 0, pixmap)
        .map_err(|e| TalkError::Config(format!("X11 shape mask: {}", e)))?;

    conn.free_gc(gc)
        .map_err(|e| TalkError::Config(format!("X11 free gc: {}", e)))?;
    conn.free_pixmap(pixmap)
        .map_err(|e| TalkError::Config(format!("X11 free pixmap: {}", e)))?;

    Ok(())
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
}
