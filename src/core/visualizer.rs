//! Real-time audio visualizer overlays (amplitude history + FFT spectrum).
//!
//! Displays small panels on either side of the recording badge using X11
//! `put_image` for efficient rendering at 60 fps.  Each panel is
//! independently toggleable:
//!
//! - **Amplitude** (left of badge): RMS volume history over time.
//! - **Spectrum** (right of badge): FFT frequency-domain bar chart.
//!
//! The visualizer opens its own CPAL capture stream so it is fully
//! decoupled from the recording pipeline.

use crate::core::error::TalkError;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use x11rb::connection::Connection;
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

// ── Colors (BGRA for little-endian ZPixmap, depth 24/32) ─────────────

const BG: [u8; 4] = [0x00, 0x00, 0x00, 0xFF]; // #000000 black
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

    fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: [u8; 4]) {
        for dy in 0..h {
            for dx in 0..w {
                self.set_pixel(x + dx, y + dy, color);
            }
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

/// Render amplitude history as vertical bars.
///
/// Maps `history` (one RMS value per frame) onto the panel width.
/// At the start bars are wide/stretched; as recording progresses,
/// more time is compressed into the same width.  Normalised against
/// `max_rms` so speech fills the panel and silence is tiny.
fn render_amplitude(pb: &mut PixelBuffer, history: &[f32], max_rms: f32) {
    if history.is_empty() || max_rms < PEAK_FLOOR {
        return;
    }

    let h = pb.height;
    let w = pb.width;
    let n = history.len();

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
        let bar_height = (norm * (h - 2) as f32) as usize;
        let bar_y = h - 1 - bar_height;

        let color = if norm > 0.5 { AMP_COLOR } else { AMP_DIM };
        for y in bar_y..h {
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

// ── Monitor geometry ─────────────────────────────────────────────────

/// Detect primary monitor `(x, y, width)` via xrandr.
///
/// Two-pass: first look for the explicit "primary" output, then fall
/// back to the first "connected" output.
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

fn parse_primary(text: &str) -> Option<(i16, i16, u16)> {
    for line in text.lines() {
        if line.contains(" primary ") {
            if let Some(g) = parse_geom_word(line) {
                return Some(g);
            }
        }
    }
    for line in text.lines() {
        if line.contains(" connected ") {
            if let Some(g) = parse_geom_word(line) {
                return Some(g);
            }
        }
    }
    None
}

fn parse_geom_word(line: &str) -> Option<(i16, i16, u16)> {
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

// ── Public API ───────────────────────────────────────────────────────

/// Commands from the main thread to the visualizer thread.
enum VizCommand {
    /// Show the visualizer panels (badge width needed for positioning).
    Show { badge_width: u16 },
    /// Hide visualizer panels.
    Hide,
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
    ///
    /// Returns `Err` if X11 or the audio device is unreachable.
    pub fn new(amplitude: bool, spectrum: bool) -> Result<Self, TalkError> {
        let (tx, rx) = std::sync::mpsc::channel();

        let thread = std::thread::Builder::new()
            .name("visualizer".into())
            .spawn(move || {
                if let Err(e) = visualizer_thread(rx, amplitude, spectrum) {
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

    let (mon_x, mon_y, mon_w) = primary_monitor_geometry();

    // ── Pixel buffers ────────────────────────────────────────────────

    let mut amp_pb = PixelBuffer::new(VIS_W as usize, VIS_H as usize);
    let mut sp_pb = PixelBuffer::new(VIS_W as usize, VIS_H as usize);

    // ── State ────────────────────────────────────────────────────────

    let frame_dur = std::time::Duration::from_micros(1_000_000 / FPS as u64);
    let rms_chunk = supported.sample_rate().0 as usize / FPS as usize;

    let mut wins = WindowState {
        amplitude_win: None,
        spectrum_win: None,
        amplitude_gc: None,
        spectrum_gc: None,
    };
    let mut is_showing = false;

    // Amplitude history: one RMS value per frame, grows over time.
    let mut amp_history: Vec<f32> = Vec::new();

    // Spectrum peak: fast attack, slow decay.
    let mut spectrum_peak: f32 = 0.001;

    // ── Event loop ───────────────────────────────────────────────────

    loop {
        // When hidden, block on commands (no rendering needed).
        if !is_showing {
            match rx.recv() {
                Ok(VizCommand::Show { badge_width }) => {
                    amp_history.clear();
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
                        &mut wins,
                    )?;
                    is_showing = true;
                }
                Ok(VizCommand::Hide) => {}
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
                Ok(VizCommand::Show { badge_width }) => {
                    destroy_windows(&conn, &mut wins);
                    amp_history.clear();
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

        // ── Render frame ─────────────────────────────────────────────

        let frame_start = std::time::Instant::now();

        let samples = ring
            .lock()
            .map(|g| g.read_last(FFT_SIZE.max(rms_chunk)))
            .unwrap_or_else(|_| vec![0.0; FFT_SIZE.max(rms_chunk)]);

        // Compute RMS for this frame and append to history.
        let frame_rms = rms(&samples[samples.len().saturating_sub(rms_chunk)..]);
        amp_history.push(frame_rms);

        // Amplitude panel
        if let (Some(win), Some(gc)) = (wins.amplitude_win, wins.amplitude_gc) {
            let max_rms = amp_history
                .iter()
                .copied()
                .fold(0.0f32, f32::max)
                .max(PEAK_FLOOR);

            amp_pb.clear(BG);
            render_amplitude(&mut amp_pb, &amp_history, max_rms);

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

    conn.flush()
        .map_err(|e| TalkError::Config(format!("X11 flush: {}", e)))?;

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

    // ── Monitor geometry parsing ─────────────────────────────────────

    #[test]
    fn parse_primary_monitor() {
        let text = "\
Screen 0: minimum 320 x 200, current 7680 x 4320
eDP connected 3840x2560+0+1760
DisplayPort-1 connected primary 7680x4320+3840+0";

        let result = parse_primary(text);
        assert_eq!(result, Some((3840, 0, 7680)));
    }

    #[test]
    fn parse_primary_fallback_to_connected() {
        let text = "eDP connected 1920x1080+0+0 (normal)";
        let result = parse_primary(text);
        assert_eq!(result, Some((0, 0, 1920)));
    }

    #[test]
    fn parse_primary_no_match() {
        let text = "DP-1 disconnected";
        let result = parse_primary(text);
        assert_eq!(result, None);
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
    fn render_amplitude_stretches_few_values() {
        let mut pb = PixelBuffer::new(200, 52);
        pb.clear(BG);
        // Only 3 history values — should stretch across 200 columns
        let history = vec![0.5, 1.0, 0.3];
        render_amplitude(&mut pb, &history, 1.0);

        let non_bg = pb.data.chunks_exact(4).filter(|p| *p != BG).count();
        assert!(
            non_bg > 100,
            "3 values should stretch to fill most columns, got {} non-bg pixels",
            non_bg,
        );
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
