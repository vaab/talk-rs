//! Proof-of-concept: real-time audio visualizer overlay.
//!
//! Displays a waveform (left) and FFT spectrum (right) in a small X11
//! overlay window, rendered at 60 fps via `put_image`.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example visualizer_poc
//! ```
//!
//! Press Ctrl+C to quit.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::*;
use x11rb::COPY_DEPTH_FROM_PARENT;

// ── Layout constants ─────────────────────────────────────────────────

/// Total window width in pixels.
const WIN_W: u16 = 480;
/// Total window height in pixels.
const WIN_H: u16 = 100;
/// Width of the waveform panel (left half).
const WAVE_W: usize = WIN_W as usize / 2;
/// Width of the spectrum panel (right half).
const SPEC_W: usize = WIN_W as usize / 2;
/// Height usable for drawing (same as window height).
const DRAW_H: usize = WIN_H as usize;

/// Vertical gap from the top of the primary monitor.
const TOP_MARGIN: i16 = 4;

// ── Color palette (BGRA byte order for ZPixmap on little-endian) ─────

const BG: [u8; 4] = [0x2e, 0x1a, 0x1a, 0xFF]; // #1a1a2e
const WAVE_COLOR: [u8; 4] = [0x88, 0xFF, 0x00, 0xFF]; // #00ff88

/// Gradient stops for spectrum bars (low → high frequency).
const SPEC_COLORS: [[u8; 4]; 5] = [
    [0xFF, 0xCC, 0x00, 0xFF], // #00ccff  cyan
    [0xFF, 0xFF, 0x00, 0xFF], // #00ffff  aqua
    [0x44, 0xFF, 0x00, 0xFF], // #00ff44  green
    [0x00, 0xCC, 0xFF, 0xFF], // #ffcc00  amber
    [0x44, 0x44, 0xFF, 0xFF], // #ff4444  red
];

// ── FFT size ─────────────────────────────────────────────────────────

/// Number of samples fed into the FFT (must be a power of two).
const FFT_SIZE: usize = 1024;

// ── Ring buffer for audio samples ────────────────────────────────────

/// Simple ring buffer that stores the most recent `capacity` samples.
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

    /// Append samples, overwriting the oldest ones.
    fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            self.data[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
    }

    /// Read the most recent `n` samples in chronological order.
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

// ── Minimal complex type + radix-2 FFT ──────────────────────────────

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
///
/// `buf` length must be a power of two.
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

/// Apply a Hann window to `samples` and compute the FFT magnitude
/// spectrum (first half only — positive frequencies).
fn compute_spectrum(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    let mut buf: Vec<Complex> = samples
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            // Hann window
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos());
            Complex::new(s * w, 0.0)
        })
        .collect();

    fft_in_place(&mut buf);

    // Return magnitudes of the positive-frequency half
    buf[..n / 2].iter().map(|c| c.magnitude()).collect()
}

// ── Pixel buffer rendering ──────────────────────────────────────────

/// BGRA pixel buffer (row-major, 4 bytes per pixel).
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

    /// Fill entire buffer with a solid color.
    fn clear(&mut self, color: [u8; 4]) {
        for pixel in self.data.chunks_exact_mut(4) {
            pixel.copy_from_slice(&color);
        }
    }

    /// Set a single pixel (bounds-checked).
    fn set_pixel(&mut self, x: usize, y: usize, color: [u8; 4]) {
        if x < self.width && y < self.height {
            let offset = (y * self.width + x) * 4;
            self.data[offset..offset + 4].copy_from_slice(&color);
        }
    }

    /// Draw a vertical line segment from y0 to y1 (inclusive).
    fn vline(&mut self, x: usize, y0: usize, y1: usize, color: [u8; 4]) {
        let (lo, hi) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
        for y in lo..=hi.min(self.height - 1) {
            self.set_pixel(x, y, color);
        }
    }

    /// Draw a filled rectangle.
    fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: [u8; 4]) {
        for dy in 0..h {
            for dx in 0..w {
                self.set_pixel(x + dx, y + dy, color);
            }
        }
    }
}

/// Render the waveform panel into the left half of the pixel buffer.
fn render_waveform(pb: &mut PixelBuffer, samples: &[f32]) {
    if samples.is_empty() {
        return;
    }

    let mid_y = DRAW_H / 2;

    // Draw center line (dim)
    let dim_green: [u8; 4] = [0x44, 0x66, 0x00, 0xFF];
    for x in 0..WAVE_W {
        pb.set_pixel(x, mid_y, dim_green);
    }

    // Downsample: map `samples.len()` to `WAVE_W` columns
    let step = samples.len() as f32 / WAVE_W as f32;

    let mut prev_y = mid_y;
    for col in 0..WAVE_W {
        // Average the samples that fall into this column
        let start = (col as f32 * step) as usize;
        let end = ((col + 1) as f32 * step) as usize;
        let end = end.min(samples.len());

        let avg = if end > start {
            samples[start..end].iter().sum::<f32>() / (end - start) as f32
        } else if start < samples.len() {
            samples[start]
        } else {
            0.0
        };

        // Map [-1, 1] → [0, DRAW_H-1]
        let clamped = avg.clamp(-1.0, 1.0);
        let y = ((1.0 - clamped) * 0.5 * (DRAW_H - 1) as f32) as usize;

        // Draw vertical line from previous y to current y for continuity
        pb.vline(col, prev_y, y, WAVE_COLOR);
        prev_y = y;
    }
}

/// Pick a spectrum bar color based on its position (0.0 = low freq, 1.0 = high).
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

/// Render the FFT spectrum panel into the right half of the pixel buffer.
fn render_spectrum(pb: &mut PixelBuffer, magnitudes: &[f32]) {
    if magnitudes.is_empty() {
        return;
    }

    // Use only the lower portion of the spectrum (most interesting
    // content is below ~4 kHz for voice).  Take the first quarter
    // of the magnitude array.
    let useful = &magnitudes[..magnitudes.len() / 4];

    // Number of bars to draw
    let num_bars = SPEC_W / 3; // 3px per bar (2px bar + 1px gap)
    let bins_per_bar = useful.len() / num_bars;

    // Find peak magnitude for normalization (use a running smoothed
    // peak to avoid flickering).
    let peak = useful.iter().copied().fold(0.0f32, f32::max).max(0.001);

    let x_offset = WAVE_W; // Start after the waveform panel

    for bar in 0..num_bars {
        let start = bar * bins_per_bar;
        let end = (start + bins_per_bar).min(useful.len());

        // Average magnitude for this bar
        let avg = if end > start {
            useful[start..end].iter().sum::<f32>() / (end - start) as f32
        } else {
            0.0
        };

        // Normalize to [0, 1] and apply a mild log scale for visual clarity
        let norm = (avg / peak).clamp(0.0, 1.0);
        let log_norm = (1.0 + norm * 9.0).log10(); // log10(1..10) → 0..1

        let bar_height = (log_norm * (DRAW_H - 4) as f32) as usize;
        let bar_x = x_offset + bar * 3;
        let bar_y = DRAW_H - 2 - bar_height;

        let color = spectrum_color(bar as f32 / num_bars as f32);
        pb.fill_rect(bar_x, bar_y, 2, bar_height, color);
    }
}

/// Draw a 1px vertical divider between the two panels.
fn render_divider(pb: &mut PixelBuffer) {
    let dim: [u8; 4] = [0x55, 0x44, 0x44, 0xFF];
    for y in 0..DRAW_H {
        pb.set_pixel(WAVE_W, y, dim);
    }
}

// ── Primary monitor geometry (two-pass, matching overlay.rs) ─────────

fn primary_monitor_geometry() -> (i16, i16, u16) {
    let output = std::process::Command::new("xrandr").arg("--query").output();

    match output {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout);
            parse_primary(&text).unwrap_or((0, 0, 1920))
        }
        Err(_) => (0, 0, 1920),
    }
}

/// Parse xrandr output for the primary monitor geometry.
///
/// Two-pass: first look for " primary " lines, then fall back to the
/// first " connected " line.  Returns `(x, y, width)`.
fn parse_primary(text: &str) -> Option<(i16, i16, u16)> {
    // Pass 1: explicit "primary" output
    for line in text.lines() {
        if line.contains(" primary ") {
            if let Some(geom) = parse_geometry_word(line) {
                return Some(geom);
            }
        }
    }

    // Pass 2: first connected output
    for line in text.lines() {
        if line.contains(" connected ") {
            if let Some(geom) = parse_geometry_word(line) {
                return Some(geom);
            }
        }
    }

    None
}

/// Extract `(x, y, width)` from a "WxH+X+Y" token in a line.
fn parse_geometry_word(line: &str) -> Option<(i16, i16, u16)> {
    for word in line.split_whitespace() {
        if word.contains('x') && word.contains('+') {
            let parts: Vec<&str> = word.split('+').collect();
            if parts.len() >= 3 {
                let wh: Vec<&str> = parts[0].split('x').collect();
                if wh.len() == 2 {
                    let w = wh[0].parse::<u16>().ok()?;
                    let x = parts[1].parse::<i16>().ok()?;
                    let y = parts[2].parse::<i16>().ok()?;
                    return Some((x, y, w));
                }
            }
        }
    }
    None
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    // ── 1. Audio capture ─────────────────────────────────────────────

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("no default input device");

    let supported = device
        .default_input_config()
        .expect("no default input config");

    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;

    println!(
        "Audio: {} Hz, {} ch, format {:?}",
        sample_rate,
        channels,
        supported.sample_format()
    );

    // Shared ring buffer — holds ~0.5 seconds of mono audio.
    let ring = Arc::new(Mutex::new(RingBuffer::new(sample_rate as usize / 2)));
    let ring_writer = Arc::clone(&ring);

    let stream_config = cpal::StreamConfig {
        channels: supported.channels(),
        sample_rate: supported.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Downmix to mono if needed
                let mono: Vec<f32> = if channels == 1 {
                    data.to_vec()
                } else {
                    data.chunks_exact(channels)
                        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                        .collect()
                };
                if let Ok(mut guard) = ring_writer.lock() {
                    guard.push(&mono);
                }
            },
            |err| {
                eprintln!("audio error: {}", err);
            },
            None,
        )
        .expect("failed to build input stream");

    stream.play().expect("failed to start input stream");

    // ── 2. X11 window setup ──────────────────────────────────────────

    let (conn, screen_num) = x11rb::connect(None).expect("failed to connect to X11");

    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;
    let depth = screen.root_depth;

    let (mon_x, mon_y, mon_w) = primary_monitor_geometry();
    let win_x = mon_x + (mon_w as i16 / 2) - (WIN_W as i16 / 2);
    let win_y = mon_y + TOP_MARGIN;

    let win = conn.generate_id().expect("generate_id");

    let values = CreateWindowAux::new()
        .background_pixel(screen.black_pixel)
        .border_pixel(0)
        .override_redirect(1u32)
        .event_mask(EventMask::EXPOSURE);

    conn.create_window(
        COPY_DEPTH_FROM_PARENT,
        win,
        root,
        win_x,
        win_y,
        WIN_W,
        WIN_H,
        0,
        WindowClass::INPUT_OUTPUT,
        0,
        &values,
    )
    .expect("create_window");

    conn.map_window(win).expect("map_window");
    conn.flush().expect("flush");

    // Create a GC for put_image
    let gc = conn.generate_id().expect("generate_id");
    conn.create_gc(gc, win, &CreateGCAux::new())
        .expect("create_gc");

    println!(
        "Window: {}x{} at ({}, {}), depth {}",
        WIN_W, WIN_H, win_x, win_y, depth
    );
    println!("Press Ctrl+C to quit.\n");

    // ── 3. Render loop ───────────────────────────────────────────────

    let target_fps = 60u32;
    let frame_duration = std::time::Duration::from_micros(1_000_000 / target_fps as u64);

    let mut pb = PixelBuffer::new(WIN_W as usize, WIN_H as usize);
    let mut frame_count: u64 = 0;
    let mut fps_timer = std::time::Instant::now();

    // Smoothed peak for spectrum normalization (exponential decay)
    let mut smoothed_peak: f32 = 0.001;

    loop {
        let frame_start = std::time::Instant::now();

        // Read latest audio samples
        let samples = {
            let guard = ring.lock().expect("ring lock");
            guard.read_last(FFT_SIZE)
        };

        // Render frame
        pb.clear(BG);
        render_waveform(&mut pb, &samples);

        // Compute FFT spectrum
        let mut magnitudes = compute_spectrum(&samples);

        // Smooth peak
        let raw_peak = magnitudes.iter().copied().fold(0.0f32, f32::max).max(0.001);
        smoothed_peak = smoothed_peak * 0.9 + raw_peak * 0.1;

        // Normalize magnitudes against smoothed peak
        for m in &mut magnitudes {
            *m /= smoothed_peak;
        }

        render_spectrum(&mut pb, &magnitudes);
        render_divider(&mut pb);

        // Blit pixel buffer to X11 window
        conn.put_image(
            ImageFormat::Z_PIXMAP,
            win,
            gc,
            WIN_W,
            WIN_H,
            0,
            0,
            0,
            depth,
            &pb.data,
        )
        .expect("put_image");

        conn.flush().expect("flush");

        // FPS counter
        frame_count += 1;
        let elapsed = fps_timer.elapsed();
        if elapsed >= std::time::Duration::from_secs(1) {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            print!("\rFPS: {:.1}  ", fps);
            use std::io::Write;
            let _ = std::io::stdout().flush();
            frame_count = 0;
            fps_timer = std::time::Instant::now();
        }

        // Sleep to maintain target frame rate
        let render_time = frame_start.elapsed();
        if render_time < frame_duration {
            std::thread::sleep(frame_duration - render_time);
        }
    }
}
