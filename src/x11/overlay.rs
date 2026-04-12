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
    apply_rounded_shape, blit_glyph_at, compute_spectrum, map_spectrum_to_column, rasterise_glyphs,
    rms, PixelBuffer, RingBuffer, FFT_SIZE, FREQ_MAX, FREQ_NOISE_FLOOR, PEAK_DECAY, PEAK_FLOOR,
};
use crate::error::TalkError;
use crate::telemetry::TranscriptionEvent;
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

/// X coordinate of the red dot centre.  Moved right from the original
/// `20` so the larger dot (radius 21 + gap 2) fits inside the border
/// without clipping the rounded corners.
const DOT_CX: usize = 26;
/// Y coordinate of the red dot centre (vertically centred in the spec
/// area: spec rows 4..48 → centre at 26, and with an odd-pixel
/// diameter the dot is perfectly symmetric top/bottom).
const DOT_CY: usize = 26;
/// Minimum red dot radius (quiet).  Scaled up proportionally from the
/// original `3.0` to remain visible at the new badge scale.
const DOT_RADIUS_MIN: f32 = 6.0;
/// Maximum red dot radius (loud).  Fills almost the entire spec area
/// height (diameter 43 inside the 44 px spec band) so the volume
/// indicator is easy to read at a glance.
const DOT_RADIUS_MAX: f32 = 16.0;
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

/// How many render frames elapse between two waterfall column pushes.
///
/// The render loop still runs at [`FPS`] — the red dot pulse, border,
/// and text all update every frame — but the spectrogram history is
/// only appended every `COLUMN_PERIOD_FRAMES` frames.  This decouples
/// the waterfall's *temporal* resolution from the render rate without
/// sacrificing animation smoothness.
///
/// At 60 fps with `COLUMN_PERIOD_FRAMES = 2` the waterfall advances
/// 30 columns/sec, giving a visible window of roughly
/// `SPEC_W / 30 ≈ 8.8` seconds across the 265-pixel spectrogram area.
const COLUMN_PERIOD_FRAMES: u32 = 2;

/// Opacity multiplier applied to the waterfall (and sibling visualizers)
/// while auto-pause is active.  The graph keeps scrolling at the same
/// rate but is rendered dimmer, leaving the `LISTENING` indicator and
/// pause icon on top at full brightness.
const DIM_FACTOR_PAUSED: f32 = 0.3;

/// Wall-clock interval between time-grid marks drawn over the
/// waterfall.  One mark per second gives ~9 visible marks across the
/// current `SPEC_W` at the slowed column rate — enough to read "how
/// long ago did that happen" at a glance without crowding.
///
/// Grid marks behave just like any other event overlaid on the
/// waterfall time axis: they appear at the right edge when emitted
/// and scroll left with the audio columns they were emitted next to.
const GRID_PERIOD_SECONDS: u64 = 1;

/// Columns between two consecutive grid marks.  At 60 fps with
/// `COLUMN_PERIOD_FRAMES = 2` the waterfall advances 30 cols/sec,
/// so `GRID_PERIOD_SECONDS = 1` ⇒ one mark every 30 columns.
const COLUMNS_PER_GRID_MARK: u64 = (FPS as u64 / COLUMN_PERIOD_FRAMES as u64) * GRID_PERIOD_SECONDS;

/// Alpha-blend factor for time-grid dots.  Each drawn grid pixel is
/// `(existing * (1 - GRID_BLEND_ALPHA) + yellow * GRID_BLEND_ALPHA)`,
/// giving a visible time reference without washing out the waterfall
/// content underneath.  Bumped to 0.6 after 0.3 proved too subtle to
/// read at a glance against the full-brightness spectrogram.
const GRID_BLEND_ALPHA: f32 = 0.6;

/// Initial effective frequency ceiling; grows as higher harmonics appear.
const FREQ_INITIAL_MAX: f32 = 320.0;

// ── Colors (BGRA for little-endian ZPixmap, depth 24/32) ────────────

/// Badge background: opaque black.
const BG_COLOR: [u8; 4] = [0x00, 0x00, 0x00, 0xFF];

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
    /// * `pause_flag` — shared atomic flag; the overlay sets it to
    ///   `true` when auto-pause triggers (silence during recording)
    ///   and clears it when speech resumes.
    ///
    /// Monitor geometry is queried via GDK4 **before** spawning the
    /// thread (GDK must be called from the main thread) and passed in.
    ///
    /// Returns `Err` if the X11 display cannot be opened.
    #[allow(clippy::too_many_arguments)] // Overlay threading inherently needs many params
    pub fn new(
        viz: Option<crate::config::VizMode>,
        mono: bool,
        audio_ring: Arc<Mutex<RingBuffer>>,
        sample_rate: u32,
        silence_tx: Option<std::sync::mpsc::Sender<bool>>,
        pause_flag: Arc<std::sync::atomic::AtomicBool>,
        auto_pause: bool,
        telemetry_rx: Option<tokio::sync::broadcast::Receiver<TranscriptionEvent>>,
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
                    pause_flag,
                    auto_pause,
                    telemetry_rx,
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

/// Render the spectrogram waterfall into the pixel buffer.
///
/// `history` holds the most recent columns of spectral data (each a
/// `Vec<f32>` of length `SPEC_H`).  Newer columns are at the end.
/// The spectrogram is right-aligned: the newest column draws at the
/// right edge of the area, oldest on the left.
///
/// `dim` scales each pixel's visible intensity.  Pass `1.0` for the
/// normal render, `DIM_FACTOR_PAUSED` during auto-pause to render a
/// dimmed (30 % brightness) waterfall behind the `LISTENING` indicator.
/// Out-of-range values are clamped to `[0.0, 1.0]`.
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
    dim: f32,
) {
    if history.is_empty() || peak < PEAK_FLOOR {
        return;
    }

    let dim = dim.clamp(0.0, 1.0);

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

            let mut color = if mono.is_some() {
                // Premultiplied alpha: white × brightness × dim.
                let alpha = (brightness * 255.0 * dim) as u8;
                [alpha, alpha, alpha, alpha]
            } else {
                let c = super::render_util::heat_map_color(norm, brightness);
                [
                    (c[0] as f32 * dim) as u8,
                    (c[1] as f32 * dim) as u8,
                    (c[2] as f32 * dim) as u8,
                    c[3],
                ]
            };
            // Keep pixel fully opaque over the black background.
            color[3] = 0xFF;
            pb.set_pixel(x, y, color);
        }
    }
}

/// Render amplitude history inside a sub-region of the badge.
///
/// Draws symmetric bars around the vertical centre, scrolling left
/// (newest at right edge).  Uses premultiplied alpha white (like the
/// waterfall), or monochrome palette if provided.
///
/// `dim` scales each pixel's visible intensity (see
/// [`render_spectrogram`] for the rationale).
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
    dim: f32,
) {
    if history.is_empty() || max_rms < PEAK_FLOOR {
        return;
    }

    let dim = dim.clamp(0.0, 1.0);

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

        let base = if let Some((fg, bg)) = mono {
            super::render_util::lerp_color(bg, fg, norm)
        } else {
            super::render_util::level_color(norm)
        };
        let color = [
            (base[0] as f32 * dim) as u8,
            (base[1] as f32 * dim) as u8,
            (base[2] as f32 * dim) as u8,
            base[3],
        ];

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
///
/// `dim` scales each pixel's visible intensity (see
/// [`render_spectrogram`] for the rationale).
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
    dim: f32,
) {
    if magnitudes.is_empty() || peak < PEAK_FLOOR {
        return;
    }

    let dim = dim.clamp(0.0, 1.0);

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

        let base = if let Some((fg, bg)) = mono {
            super::render_util::lerp_color(bg, fg, log_norm)
        } else {
            super::render_util::level_color(log_norm)
        };
        let color = [
            (base[0] as f32 * dim) as u8,
            (base[1] as f32 * dim) as u8,
            (base[2] as f32 * dim) as u8,
            base[3],
        ];

        for dy in 0..bar_height {
            pb.set_pixel(bar_x, bar_y + dy, color);
        }
    }
}

/// Draw the time grid: vertical dotted yellow lines over the
/// spectrogram area, one line per [`GRID_PERIOD_SECONDS`]
/// wall-clock second, alpha-blended with the existing pixel
/// contents so they remain visible through whatever's underneath.
///
/// # Model
///
/// Grid marks behave exactly like any other event on the waterfall
/// time axis — when a new column is pushed whose absolute index
/// (since the last history reset) is a multiple of
/// [`COLUMNS_PER_GRID_MARK`], that column becomes a grid column.
/// As the history scrolls left, the grid columns scroll with it
/// automatically because their position in the visible window is
/// determined by the absolute index rather than by a fixed screen
/// coordinate.
///
/// # Arguments
///
/// * `first_visible_abs_idx` — the absolute column index of the
///   oldest column currently in the spectrogram history
///   (`columns_pushed_total - spectrogram_history.len() as u64`).
///   Used so the caller does not need to know how to compute the
///   right-alignment offset.
/// * `num_visible_cols` — number of columns currently held in the
///   spectrogram history (`spectrogram_history.len()`).  The grid
///   function uses this to right-align the grid marks exactly the
///   same way [`render_spectrogram`] right-aligns the waterfall.
///
/// # Drawing
///
/// Each grid mark is a column of 1-pixel yellow dots spaced by
/// 1 pixel gaps (dot at y=0, empty at y=1, dot at y=2, …).  Each
/// dot alpha-blends [`GRID_BLEND_ALPHA`] of pure yellow
/// `(255, 255, 0)` with the existing pixel colour, preserving
/// whatever waterfall content was there underneath at 70 % strength.
#[allow(clippy::too_many_arguments)]
fn render_time_grid(
    pb: &mut PixelBuffer,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    first_visible_abs_idx: u64,
    num_visible_cols: usize,
    columns_per_mark: u64,
) {
    if num_visible_cols == 0 || columns_per_mark == 0 || w == 0 || h == 0 {
        return;
    }

    let num = num_visible_cols.min(w);
    let inv_alpha = 1.0 - GRID_BLEND_ALPHA;

    // Pure yellow in BGRA byte order (little-endian ZPixmap): B=0,
    // G=255, R=255, alpha doesn't participate in the blend because
    // we always leave the destination alpha untouched.
    const YELLOW_B: f32 = 0.0;
    const YELLOW_G: f32 = 255.0;
    const YELLOW_R: f32 = 255.0;

    for col_idx in 0..num {
        let abs_idx = first_visible_abs_idx + col_idx as u64;
        if !abs_idx.is_multiple_of(columns_per_mark) {
            continue;
        }

        // Right-align: the newest column of the history always sits
        // at `x0 + w - 1`, the oldest at `x0 + w - num`.
        let x = x0 + w - num + col_idx;
        if x >= pb.width {
            continue;
        }

        // 1 px dot, 1 px gap, starting at the top of the spec area.
        let mut row = 0usize;
        while row < h {
            let y = y0 + row;
            if y < pb.height {
                let off = (y * pb.width + x) * 4;
                // Read existing BGR, alpha-blend with yellow, write
                // back.  Leave the destination alpha byte untouched
                // so the pixel stays opaque relative to whatever
                // shape mask the window uses.
                let b = pb.data[off] as f32;
                let g = pb.data[off + 1] as f32;
                let r = pb.data[off + 2] as f32;
                pb.data[off] = (b * inv_alpha + YELLOW_B * GRID_BLEND_ALPHA) as u8;
                pb.data[off + 1] = (g * inv_alpha + YELLOW_G * GRID_BLEND_ALPHA) as u8;
                pb.data[off + 2] = (r * inv_alpha + YELLOW_R * GRID_BLEND_ALPHA) as u8;
            }
            // Dot (row even) + skip (row odd): advance 2 rows per
            // dot so we get the 1-on-1-off dotted pattern.
            row += 2;
        }
    }
}

// ── Phase layer rendering ───────────────────────────────────────────

/// Network transcription phase as derived from [`TranscriptionEvent`]s.
///
/// The overlay maintains a simple state machine driven by events
/// arriving from the telemetry broker.  Each variant maps to a
/// distinct colour in the phase overlay layer drawn above the
/// waterfall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// No HTTP activity in progress.
    Idle,
    /// `RequestStarted` received, waiting for first byte pull.
    Connecting,
    /// `ConnectionEstablished` received, body bytes flowing.
    Uploading,
    /// `UploadComplete` received, waiting for response headers.
    WaitingResponse,
    /// `ResponseHeaders` received, body arriving / JSON parsing.
    Receiving,
    /// `RequestCompleted { success: true }` received.
    Done,
    /// `RequestCompleted { success: false }` received.
    Error,
}

impl Phase {
    /// BGRA colour for this phase, or `None` for `Idle` (nothing
    /// drawn).  Colours are chosen to stand out against the
    /// monochrome waterfall without being garish.
    fn color(self) -> Option<[u8; 4]> {
        match self {
            Self::Idle => None,
            // dim blue  RGB(80,130,200) → BGRA
            Self::Connecting => Some([200, 130, 80, 255]),
            // bright blue  RGB(60,170,255)
            Self::Uploading => Some([255, 170, 60, 255]),
            // amber  RGB(255,180,50)
            Self::WaitingResponse => Some([50, 180, 255, 255]),
            // teal  RGB(60,200,180)
            Self::Receiving => Some([180, 200, 60, 255]),
            // green  RGB(60,255,110)
            Self::Done => Some([110, 255, 60, 255]),
            // red  RGB(200,40,40)
            Self::Error => Some([40, 40, 200, 255]),
        }
    }

    /// Advance the state machine given a telemetry event.
    fn advance(self, event: &TranscriptionEvent) -> Self {
        match event {
            TranscriptionEvent::RequestStarted { .. } => Self::Connecting,
            TranscriptionEvent::ConnectionEstablished { .. } => Self::Uploading,
            TranscriptionEvent::UploadComplete { .. } => Self::WaitingResponse,
            TranscriptionEvent::ResponseHeaders { .. } => Self::Receiving,
            TranscriptionEvent::RequestCompleted { success: true, .. } => Self::Done,
            TranscriptionEvent::RequestCompleted { success: false, .. } => Self::Error,
            TranscriptionEvent::RetryScheduled { .. } => Self::Connecting,
            TranscriptionEvent::PasteStarted { .. } => Self::Done, // green during paste
            TranscriptionEvent::Done { .. } | TranscriptionEvent::PasteCompleted { .. } => {
                Self::Idle
            }
            TranscriptionEvent::Failed { .. } => Self::Error,
            // Upload/download progress don't change the phase.
            _ => self,
        }
    }
}

/// Height of the phase colour line drawn at the top of the spec area.
const PHASE_LINE_HEIGHT: usize = 2;

/// Render the phase overlay: a thin horizontal colour band at the
/// top of the spectrogram area, one colour per column, right-aligned
/// in the same way as the waterfall.
///
/// Columns without a phase (`None`) are skipped — nothing drawn,
/// the waterfall underneath is unobscured.  Columns with a phase
/// get a solid-colour stripe of [`PHASE_LINE_HEIGHT`] pixels drawn
/// as a direct pixel overwrite (no alpha blend needed — the top
/// rows of the spec area are almost always empty in practice).
fn render_phase_line(
    pb: &mut PixelBuffer,
    x0: usize,
    y0: usize,
    w: usize,
    phase_history: &[Option<[u8; 4]>],
) {
    let n = phase_history.len();
    if n == 0 || w == 0 {
        return;
    }

    let num = n.min(w);

    for col_idx in 0..num {
        let color = match phase_history[n - num + col_idx] {
            Some(c) => c,
            None => continue,
        };

        // Right-align: newest column at `x0 + w - 1`.
        let x = x0 + w - num + col_idx;
        if x >= pb.width {
            continue;
        }

        for dy in 0..PHASE_LINE_HEIGHT {
            let y = y0 + dy;
            if y < pb.height {
                pb.set_pixel(x, y, color);
            }
        }
    }
}

/// Pixel budget for each individual throughput track.  Each of the
/// three tracks (upload, download, paste) gets its own 16 px with
/// its own independent scale — a 100 KB upload and a 1 KB download
/// both fill their 16 px when they hit their respective peaks.
const TRACK_HEIGHT_PX: usize = 16;

/// Render the three throughput tracks (Layer 3b).
///
/// Each track has its own 16 px budget and its own scale (normalised
/// against its own peak delta).  This avoids the problem where a
/// ~100 KB upload dwarfs a ~1 KB JSON response into invisibility.
///
/// Layout (all right-aligned on the same column x positions):
///   - **Upload**: grows DOWN from `y0 + PHASE_LINE_HEIGHT` (top)
///   - **Download**: grows UP from `y0 + h - 1` (bottom)
///   - **Paste**: centred at `y0 + h / 2` (middle)
#[allow(clippy::too_many_arguments)]
fn render_throughput_tracks(
    pb: &mut PixelBuffer,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    upload_history: &[u64],
    download_history: &[u64],
    paste_history: &[u64],
    _phase_history: &[Option<[u8; 4]>],
    upload_peak: u64,
    download_peak: u64,
    paste_peak: u64,
) {
    if w == 0 || h == 0 {
        return;
    }

    let num = upload_history
        .len()
        .min(download_history.len())
        .min(paste_history.len())
        .min(w);
    if num == 0 {
        return;
    }

    let track_h = TRACK_HEIGHT_PX.min(h / 3) as f32;
    let bar_zone_top = y0 + PHASE_LINE_HEIGHT;
    let bar_zone_bottom = y0 + h - 1;
    let bar_zone_center = y0 + h / 2;

    let upload_color: [u8; 4] = Phase::Uploading.color().unwrap_or([255, 170, 60, 255]);
    let download_color: [u8; 4] = Phase::Receiving.color().unwrap_or([180, 200, 60, 255]);
    let paste_color: [u8; 4] = Phase::Done.color().unwrap_or([110, 255, 60, 255]);

    for col_idx in 0..num {
        let x = x0 + w - num + col_idx;
        if x >= pb.width {
            continue;
        }

        let ui = upload_history.len() - num + col_idx;
        let di = download_history.len() - num + col_idx;
        let pi = paste_history.len() - num + col_idx;

        // ── Upload: own scale, grow DOWN from top ──
        if upload_history[ui] > 0 && upload_peak > 0 {
            let norm = (upload_history[ui] as f32 / upload_peak as f32).clamp(0.0, 1.0);
            let bar_h = ((norm * track_h) as usize).max(1);
            for dy in 0..bar_h {
                let y = bar_zone_top + dy;
                if y >= pb.height || y > bar_zone_bottom {
                    break;
                }
                pb.set_pixel(x, y, upload_color);
            }
        }

        // ── Download: own scale, grow UP from bottom ──
        if download_history[di] > 0 && download_peak > 0 {
            let norm = (download_history[di] as f32 / download_peak as f32).clamp(0.0, 1.0);
            let bar_h = ((norm * track_h) as usize).max(1);
            for dy in 0..bar_h {
                let y = bar_zone_bottom.saturating_sub(dy);
                if y < y0 || y < bar_zone_top {
                    break;
                }
                pb.set_pixel(x, y, download_color);
            }
        }

        // ── Paste: own scale, centred in middle ──
        if paste_history[pi] > 0 && paste_peak > 0 {
            let norm = (paste_history[pi] as f32 / paste_peak as f32).clamp(0.0, 1.0);
            let bar_h = ((norm * track_h) as usize).max(1);
            let half = bar_h / 2;
            for dy in 0..bar_h {
                let y = bar_zone_center.saturating_sub(half) + dy;
                if y >= pb.height || y > bar_zone_bottom || y < bar_zone_top {
                    continue;
                }
                pb.set_pixel(x, y, paste_color);
            }
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

/// Draw a pause icon (two vertical bars ⏸) in yellow.
///
/// Replaces the pulsing red dot when auto-pause is active.
fn draw_pause_icon(pb: &mut PixelBuffer, cx: usize, cy: usize, radius: f32) {
    // Warm yellow, BGRA
    let color: [u8; 4] = [0x00, 0xC0, 0xFF, 0xFF];
    let bar_h = (radius * 1.4) as i32;
    let bar_w = (radius * 0.35).max(2.0) as i32;
    let gap = (radius * 0.35).max(2.0) as i32;

    let cx = cx as i32;
    let cy = cy as i32;

    // Left bar
    let lx = cx - gap / 2 - bar_w;
    let ly = cy - bar_h / 2;
    // Right bar
    let rx = cx + gap / 2;

    for bar_x in [lx, rx] {
        for dy in 0..bar_h {
            for dx in 0..bar_w {
                let px = bar_x + dx;
                let py = ly + dy;
                if px >= 0 && (px as usize) < pb.width && py >= 0 && (py as usize) < pb.height {
                    let off = (py as usize * pb.width + px as usize) * 4;
                    for (c, &val) in color.iter().enumerate().take(4) {
                        pb.data[off + c] = val;
                    }
                }
            }
        }
    }
}

/// Render centred text in the SPEC area with the given colour.
#[allow(clippy::too_many_arguments)]
fn render_badge_text(
    pb: &mut PixelBuffer,
    font: &fontdue::Font,
    text: &str,
    color: [u8; 4],
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    let font_size = 24.0f32;
    let (glyphs, text_w) = rasterise_glyphs(text, font, font_size);

    let start_x = x0 as i32 + (w as i32 - text_w as i32) / 2;
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

/// Render "NO SOUND" text in bright red, centred in the SPEC area.
fn render_no_sound_text(
    pb: &mut PixelBuffer,
    font: &fontdue::Font,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    let color: [u8; 4] = [0x00, 0x00, 0xFF, 0xFF]; // bright red BGRA
    render_badge_text(pb, font, "NO SOUND", color, x0, y0, w, h);
}

/// Render "LISTENING" text in yellow, centred in the SPEC area.
fn render_listening_text(
    pb: &mut PixelBuffer,
    font: &fontdue::Font,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    let color: [u8; 4] = [0x00, 0xC0, 0xFF, 0xFF]; // warm yellow BGRA
    render_badge_text(pb, font, "LISTENING", color, x0, y0, w, h);
}

/// Render "TRANSCRIBING" text in light blue, centred in the SPEC area.
fn render_transcribing_text(
    pb: &mut PixelBuffer,
    font: &fontdue::Font,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    let color: [u8; 4] = [0xFF, 0xCC, 0x66, 0xFF]; // light blue BGRA
    render_badge_text(pb, font, "TRANSCRIBING", color, x0, y0, w, h);
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
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn overlay_thread(
    rx: mpsc::Receiver<Command>,
    geom: super::monitor::MonitorGeometry,
    viz: Option<crate::config::VizMode>,
    mono_palette: Option<([u8; 4], [u8; 4])>,
    ring: Arc<Mutex<RingBuffer>>,
    sample_rate: u32,
    silence_tx: Option<std::sync::mpsc::Sender<bool>>,
    pause_flag: Arc<std::sync::atomic::AtomicBool>,
    auto_pause: bool,
    mut telemetry_rx: Option<tokio::sync::broadcast::Receiver<TranscriptionEvent>>,
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
    const DEAD_SIGNAL_TRIGGER_FRAMES: u32 = 30; // 0.5s grace for PipeWire to fill the ring buffer

    // ── Auto-pause state ─────────────────────────────────────────────
    // Pause the recording pipeline when the user stops speaking.
    // Uses RMS threshold (not variance) because this distinguishes
    // "quiet room with working mic" from "speech".
    let mut quiet_frames: u32 = 0;
    let mut auto_paused: bool = false;
    const AUTOPAUSE_RMS_THRESHOLD: f32 = 0.003; // well above mic noise (~0.00004)
    const AUTOPAUSE_TRIGGER_FRAMES: u32 = 15; // 0.3 seconds at 60fps

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

    // Waterfall column advance counter.  Incremented every render
    // frame; a new column is pushed to [`spectrogram_history`] (and
    // amplitude/spectrum equivalents) only when the counter is a
    // multiple of [`COLUMN_PERIOD_FRAMES`].  This decouples the
    // waterfall's temporal resolution from the 60 fps render rate.
    let mut column_frame_counter: u32 = 0;

    // ── Phase overlay state ──────────────────────��───────────────
    //
    // Parallel to `spectrogram_history`: for each column we store
    // the colour of the HTTP phase that was active when the column
    // was pushed.  `None` = no HTTP activity (normal audio).
    // The state machine is driven by events from the telemetry
    // broker, drained non-blockingly every frame.
    let mut current_phase: Phase = Phase::Idle;
    let mut phase_history: Vec<Option<[u8; 4]>> = Vec::new();

    // True when the recording has ended and the HTTP transcription is
    // in flight.  The render loop keeps running: empty columns are
    // pushed so the waterfall continues to scroll, and the phase
    // colour layer shows the HTTP lifecycle on top.
    let mut is_transcribing: bool = false;

    // ── Byte throughput state (Layer 3b) ─────────────────────────
    //
    // Three independent tracks that can overlap in streaming mode:
    //   upload   — grows DOWN from the phase line (top of spec)
    //   download — grows UP from the bottom of the spec area
    //   paste    — grows UP from the bottom alongside download
    //
    // Each track has its own cumulative counter, previous-value
    // snapshot, peak delta, and ring-buffer history.
    let mut current_upload_bytes: u64 = 0;
    let mut prev_upload_bytes: u64 = 0;
    let mut upload_peak_delta: u64 = 1;
    let mut upload_history: Vec<u64> = Vec::new();

    let mut current_download_bytes: u64 = 0;
    let mut prev_download_bytes: u64 = 0;
    let mut download_peak_delta: u64 = 1;
    let mut download_history: Vec<u64> = Vec::new();

    let mut current_paste_chars: u64 = 0;
    let mut prev_paste_chars: u64 = 0;
    let mut paste_peak_delta: u64 = 1;
    let mut paste_history: Vec<u64> = Vec::new();

    // Running total of columns ever pushed into the spectrogram
    // history since the last reset (`spectrogram_history.clear()`).
    // Used by the time-grid overlay to compute absolute column
    // indices — grid marks land on columns where
    // `index % COLUMNS_PER_GRID_MARK == 0`, so the counter must
    // reset together with the history to keep the first grid mark
    // aligned with the first visible column of each new session.
    let mut columns_pushed_total: u64 = 0;

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
                        // 32-bit ARGB window with shape mask for rounded corners.
                        let w = create_argb_overlay_window(
                            &conn, root, ctx, badge_x, badge_y, BADGE_W, BADGE_H,
                        )?;
                        apply_rounded_shape(&conn, w, BADGE_W, BADGE_H, CORNER_RADIUS)?;
                        w
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
                    is_transcribing = false;
                    spectrogram_history.clear();
                    phase_history.clear();
                    upload_history.clear();
                    download_history.clear();
                    paste_history.clear();
                    current_upload_bytes = 0;
                    prev_upload_bytes = 0;
                    upload_peak_delta = 1;
                    current_download_bytes = 0;
                    prev_download_bytes = 0;
                    download_peak_delta = 1;
                    current_paste_chars = 0;
                    prev_paste_chars = 0;
                    paste_peak_delta = 1;
                    columns_pushed_total = 0;
                    current_phase = Phase::Idle;
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
                    quiet_frames = 0;
                    auto_paused = false;
                    pause_flag.store(false, std::sync::atomic::Ordering::Relaxed);
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
                    is_transcribing = false;
                    spectrogram_history.clear();
                    phase_history.clear();
                    upload_history.clear();
                    download_history.clear();
                    paste_history.clear();
                    current_upload_bytes = 0;
                    prev_upload_bytes = 0;
                    upload_peak_delta = 1;
                    current_download_bytes = 0;
                    prev_download_bytes = 0;
                    download_peak_delta = 1;
                    current_paste_chars = 0;
                    prev_paste_chars = 0;
                    paste_peak_delta = 1;
                    columns_pushed_total = 0;
                    current_phase = Phase::Idle;
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
                    quiet_frames = 0;
                    auto_paused = false;
                    pause_flag.store(false, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(Command::Show(IndicatorKind::Transcribing)) => {
                    // Keep the render loop running instead of
                    // switching to the static PNG.  The waterfall
                    // continues to scroll (with empty columns since
                    // recording has stopped), and the phase colour
                    // layer renders the HTTP lifecycle on top.
                    is_transcribing = true;
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

        // ── Drain telemetry events (non-blocking) ───────────
        //
        // The overlay thread is a plain OS thread (not async).
        // We use `try_recv()` to drain all pending events from
        // the broadcast channel without blocking the render loop.
        if let Some(ref mut trx) = telemetry_rx {
            loop {
                match trx.try_recv() {
                    Ok(event) => {
                        // Track counters for the three throughput tracks.
                        match &event {
                            TranscriptionEvent::UploadProgress { bytes_sent, .. } => {
                                current_upload_bytes = *bytes_sent;
                            }
                            TranscriptionEvent::DownloadProgress { bytes_received, .. } => {
                                current_download_bytes = *bytes_received;
                            }
                            TranscriptionEvent::PasteProgress { chars_pasted, .. } => {
                                current_paste_chars = *chars_pasted;
                            }
                            TranscriptionEvent::RequestStarted { .. } => {
                                // Reset upload + download for a new request.
                                current_upload_bytes = 0;
                                prev_upload_bytes = 0;
                                current_download_bytes = 0;
                                prev_download_bytes = 0;
                            }
                            TranscriptionEvent::PasteStarted { .. } => {
                                current_paste_chars = 0;
                                prev_paste_chars = 0;
                            }
                            _ => {}
                        }
                        current_phase = current_phase.advance(&event);
                    }
                    Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
                    Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                        log::debug!("overlay telemetry: skipped {} lagged events", n);
                        continue;
                    }
                    Err(tokio::sync::broadcast::error::TryRecvError::Closed) => {
                        // Sender dropped — no more events coming.
                        telemetry_rx = None;
                        break;
                    }
                }
            }
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

        // ── Auto-pause detection ─────────────────────────────
        // Only active when enabled and we have a working device (not dead signal).
        if auto_pause && !no_sound_active {
            if frame_rms < AUTOPAUSE_RMS_THRESHOLD {
                quiet_frames = quiet_frames.saturating_add(1);
            } else {
                quiet_frames = 0;
                if auto_paused {
                    auto_paused = false;
                    pause_flag.store(false, std::sync::atomic::Ordering::Relaxed);
                    log::debug!("auto-pause: resumed (speech detected)");
                }
            }
            if quiet_frames >= AUTOPAUSE_TRIGGER_FRAMES && !auto_paused {
                auto_paused = true;
                pause_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                log::debug!("auto-pause: paused (silence detected)");
            }
        } else if !auto_pause {
            // Auto-pause disabled — ensure flag stays cleared.
            quiet_frames = 0;
        } else {
            // Dead signal takes priority — reset auto-pause state.
            quiet_frames = 0;
            if auto_paused {
                auto_paused = false;
                pause_flag.store(false, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // Per-viz-mode peak tracking and frequency scaling — updated
        // every render frame (regardless of the slower column-push
        // cadence) so normalization stays smooth.  Peaks are still
        // frozen while the device is dead or auto-pause is active.
        if !auto_paused && !no_sound_active {
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
        }

        // ── Waterfall column advance ─────────────────────────
        //
        // The waterfall scrolls at a *constant* wall-clock rate
        // (one column every [`COLUMN_PERIOD_FRAMES`] render frames),
        // regardless of whether the user is currently speaking.
        //
        // During auto-pause, an *empty* column is pushed instead of
        // skipping the push entirely.  This keeps the scroll going
        // forward on the time axis so the growing "hole" in the
        // spectrogram visually represents the duration of the pause.
        // When speech resumes, the hole stops growing and real audio
        // columns fill in again.
        //
        // Peak tracking above is still frame-rate-driven so
        // normalization adapts smoothly even when the column rate
        // is slower.
        column_frame_counter = column_frame_counter.wrapping_add(1);
        // During transcription, the audio capture is stopped and the
        // dead-signal detector will fire (stale samples → variance 0
        // → no_sound_active = true).  We must keep pushing columns
        // regardless so the phase line and throughput bars continue
        // to advance on the time axis.
        if column_frame_counter.is_multiple_of(COLUMN_PERIOD_FRAMES)
            && (!no_sound_active || is_transcribing)
        {
            if let Some(mode) = viz {
                use crate::config::VizMode;
                match mode {
                    VizMode::Waterfall => {
                        let column = if auto_paused || is_transcribing {
                            // Empty column → no visible content, but the
                            // column still advances so the time axis
                            // keeps moving and the spectrogram "hole"
                            // grows with real elapsed time.
                            vec![0.0f32; SPEC_H]
                        } else {
                            map_spectrum_to_column(
                                &magnitudes,
                                SPEC_H,
                                sample_rate,
                                effective_freq_max,
                            )
                        };
                        spectrogram_history.push(column);
                        if spectrogram_history.len() > SPEC_W {
                            spectrogram_history.drain(..spectrogram_history.len() - SPEC_W);
                        }
                    }
                    VizMode::Amplitude => {
                        // 0.0 during pause / transcribing so the
                        // history also shows a visible gap.
                        let val = if auto_paused || is_transcribing {
                            0.0
                        } else {
                            frame_rms
                        };
                        amp_history.push(val);
                        if amp_history.len() > amp_max_frames {
                            amp_history.drain(..amp_history.len() - amp_max_frames);
                        }
                    }
                    VizMode::Spectrum => {
                        // Spectrum viz has no history; it renders a
                        // live snapshot of `magnitudes` each frame.
                    }
                }
            }
            // Every column push — even empty columns during pause —
            // advances the absolute-column counter the time-grid
            // overlay uses to space its vertical marks one per
            // wall-clock second.
            columns_pushed_total = columns_pushed_total.wrapping_add(1);

            // Record the current phase colour alongside the
            // spectrogram column so the phase overlay line stays
            // perfectly aligned with the waterfall time axis.
            phase_history.push(current_phase.color());
            if phase_history.len() > SPEC_W {
                phase_history.drain(..phase_history.len() - SPEC_W);
            }

            // Record per-column deltas for all three throughput tracks.
            let up_delta = current_upload_bytes.saturating_sub(prev_upload_bytes);
            prev_upload_bytes = current_upload_bytes;
            if up_delta > upload_peak_delta {
                upload_peak_delta = up_delta;
            }
            upload_history.push(up_delta);
            if upload_history.len() > SPEC_W {
                upload_history.drain(..upload_history.len() - SPEC_W);
            }

            let dl_delta = current_download_bytes.saturating_sub(prev_download_bytes);
            prev_download_bytes = current_download_bytes;
            if dl_delta > download_peak_delta {
                download_peak_delta = dl_delta;
            }
            download_history.push(dl_delta);
            if download_history.len() > SPEC_W {
                download_history.drain(..download_history.len() - SPEC_W);
            }

            let paste_delta = current_paste_chars.saturating_sub(prev_paste_chars);
            prev_paste_chars = current_paste_chars;
            if paste_delta > paste_peak_delta {
                paste_peak_delta = paste_delta;
            }
            paste_history.push(paste_delta);
            if paste_history.len() > SPEC_W {
                paste_history.drain(..paste_history.len() - SPEC_W);
            }
        }

        // ── Render badge ─────────────────────────────────────

        pb.clear(BG_COLOR);
        draw_rounded_border(&mut pb, BORDER_COLOR, CORNER_RADIUS as f32, BORDER_WIDTH);

        // Transcribing takes priority over the dead-signal (NO SOUND)
        // branch: after capture.stop() the audio ring buffer goes
        // stale and the dead-signal detector fires, but we WANT to
        // keep rendering the waterfall + phase line + throughput bars
        // rather than showing the NO SOUND prohibit icon.
        if is_transcribing {
            // ── TRANSCRIBING mode: dimmed waterfall + phase ───
            //
            // Same visual treatment as auto-pause (dimmed waterfall
            // keeps scrolling, empty columns create the "hole")
            // but the phase colour layer on top now shows the HTTP
            // lifecycle — connecting, uploading, waiting, receiving.
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
                            DIM_FACTOR_PAUSED,
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
                            DIM_FACTOR_PAUSED,
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
                            DIM_FACTOR_PAUSED,
                        );
                    }
                }
            }
            render_time_grid(
                &mut pb,
                SPEC_LEFT,
                SPEC_TOP,
                SPEC_W,
                SPEC_H,
                columns_pushed_total.saturating_sub(spectrogram_history.len() as u64),
                spectrogram_history.len(),
                COLUMNS_PER_GRID_MARK,
            );
            render_phase_line(&mut pb, SPEC_LEFT, SPEC_TOP, SPEC_W, &phase_history);
            render_throughput_tracks(
                &mut pb,
                SPEC_LEFT,
                SPEC_TOP,
                SPEC_W,
                SPEC_H,
                &upload_history,
                &download_history,
                &paste_history,
                &phase_history,
                upload_peak_delta,
                download_peak_delta,
                paste_peak_delta,
            );
            // "TRANSCRIBING" text over the dimmed waterfall (same
            // pattern as "LISTENING" during auto-pause).
            if let Some(ref f) = badge_font {
                render_transcribing_text(&mut pb, f, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H);
            }
        } else if no_sound_active {
            // ── NO SOUND mode: prohibit icon + red text ──────
            if let Some(ref f) = badge_font {
                render_no_sound_text(&mut pb, f, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H);
            }
            clear_dot_gap(&mut pb, DOT_CX, DOT_CY, DOT_RADIUS_MAX + DOT_GAP);
            draw_prohibit_icon(&mut pb, DOT_CX, DOT_CY, DOT_RADIUS_MAX);
        } else if auto_paused {
            // ── AUTO-PAUSE mode: dimmed waterfall + LISTENING ─
            //
            // The waterfall keeps scrolling at constant time and the
            // growing "hole" (empty columns pushed above) shows how
            // long we've been listening.  We render it at reduced
            // opacity so the `LISTENING` indicator and pause icon
            // remain the dominant foreground.
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
                            DIM_FACTOR_PAUSED,
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
                            DIM_FACTOR_PAUSED,
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
                            DIM_FACTOR_PAUSED,
                        );
                    }
                }
            }
            // Time grid is drawn after the waterfall so its alpha
            // blend mixes visibly with whatever sits underneath.
            // Still visible during auto-pause so the user can see
            // how long the silent window has been growing.
            render_time_grid(
                &mut pb,
                SPEC_LEFT,
                SPEC_TOP,
                SPEC_W,
                SPEC_H,
                columns_pushed_total.saturating_sub(spectrogram_history.len() as u64),
                spectrogram_history.len(),
                COLUMNS_PER_GRID_MARK,
            );
            render_phase_line(&mut pb, SPEC_LEFT, SPEC_TOP, SPEC_W, &phase_history);
            render_throughput_tracks(
                &mut pb,
                SPEC_LEFT,
                SPEC_TOP,
                SPEC_W,
                SPEC_H,
                &upload_history,
                &download_history,
                &paste_history,
                &phase_history,
                upload_peak_delta,
                download_peak_delta,
                paste_peak_delta,
            );
            if let Some(ref f) = badge_font {
                render_listening_text(&mut pb, f, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H);
            }
            clear_dot_gap(&mut pb, DOT_CX, DOT_CY, DOT_RADIUS_MAX + DOT_GAP);
            draw_pause_icon(&mut pb, DOT_CX, DOT_CY, DOT_RADIUS_MAX);
        } else {
            // ── Normal mode: visualization + pulsing dot ─────
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
                            1.0,
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
                            1.0,
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
                            1.0,
                        );
                    }
                }
            }
            // Time grid overlays the waterfall with one dotted
            // yellow vertical mark per [`GRID_PERIOD_SECONDS`].
            // Drawn after the waterfall so the alpha blend combines
            // with the spectrogram colors beneath each dot.
            render_time_grid(
                &mut pb,
                SPEC_LEFT,
                SPEC_TOP,
                SPEC_W,
                SPEC_H,
                columns_pushed_total.saturating_sub(spectrogram_history.len() as u64),
                spectrogram_history.len(),
                COLUMNS_PER_GRID_MARK,
            );
            render_phase_line(&mut pb, SPEC_LEFT, SPEC_TOP, SPEC_W, &phase_history);
            render_throughput_tracks(
                &mut pb,
                SPEC_LEFT,
                SPEC_TOP,
                SPEC_W,
                SPEC_H,
                &upload_history,
                &download_history,
                &paste_history,
                &phase_history,
                upload_peak_delta,
                download_peak_delta,
                paste_peak_delta,
            );

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
            &mut pb, &history, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 1.0, None, 1.0,
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
        render_spectrogram(
            &mut pb,
            &[],
            SPEC_LEFT,
            SPEC_TOP,
            SPEC_W,
            SPEC_H,
            1.0,
            None,
            1.0,
        );
    }

    #[test]
    fn render_time_grid_draws_dotted_yellow_at_first_mark_on_blank_buffer() {
        // On a blank (black) buffer, 65 visible columns starting at
        // absolute index 0 → grid marks at abs idx 0, 30, 60.
        // Right-aligned: columns 0..64 land at x = SPEC_LEFT + SPEC_W
        // - 65 + col_idx.  So:
        //     abs_idx=0  → col_idx=0  → x = SPEC_LEFT + SPEC_W - 65
        //     abs_idx=30 → col_idx=30 → x = SPEC_LEFT + SPEC_W - 35
        //     abs_idx=60 → col_idx=60 → x = SPEC_LEFT + SPEC_W - 5
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);

        render_time_grid(&mut pb, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 0, 65, 30);

        // Expected dot at (SPEC_LEFT + SPEC_W - 65, SPEC_TOP) — row 0 is
        // the first dot in the 1-on-1-off pattern.
        let first_x = SPEC_LEFT + SPEC_W - 65;
        let off = (SPEC_TOP * pb.width + first_x) * 4;
        // BG is opaque black → blend of 70 % black + 30 % pure yellow
        // (BGRA [0, 255, 255, _]) ⇒ B stays 0, G ≈ 76, R ≈ 76.
        assert_eq!(pb.data[off], 0, "grid blue channel unchanged by yellow");
        assert!(
            pb.data[off + 1] > 0 && pb.data[off + 1] < 200,
            "grid green should show blended yellow, got {}",
            pb.data[off + 1]
        );
        assert!(
            pb.data[off + 2] > 0 && pb.data[off + 2] < 200,
            "grid red should show blended yellow, got {}",
            pb.data[off + 2]
        );

        // Row 1 of the same column must still be background (1-off
        // in the 1-on-1-off pattern).
        let off_gap = ((SPEC_TOP + 1) * pb.width + first_x) * 4;
        assert_eq!(
            &pb.data[off_gap..off_gap + 4],
            &BG_COLOR,
            "row 1 of grid column should be untouched (dot gap)"
        );

        // Row 2 of the same column should be another dot.
        let off_dot2 = ((SPEC_TOP + 2) * pb.width + first_x) * 4;
        assert!(
            pb.data[off_dot2 + 1] > 0 && pb.data[off_dot2 + 2] > 0,
            "row 2 of grid column should be another dot"
        );

        // A non-grid column (say col_idx=5) must still be background.
        let non_grid_x = SPEC_LEFT + SPEC_W - 65 + 5;
        let off_ng = (SPEC_TOP * pb.width + non_grid_x) * 4;
        assert_eq!(
            &pb.data[off_ng..off_ng + 4],
            &BG_COLOR,
            "non-grid column should remain background"
        );
    }

    #[test]
    fn render_time_grid_blends_with_existing_waterfall_pixels() {
        // Start with a fully-white pixel buffer in the spec area.
        // The grid blend should make those pixels slightly tinted
        // toward yellow (green stays high, red stays high, blue
        // drops because yellow's blue channel is 0).
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        // Fill the whole spec area with opaque white.
        for y in SPEC_TOP..SPEC_BOTTOM {
            for x in SPEC_LEFT..SPEC_RIGHT {
                pb.set_pixel(x, y, [255, 255, 255, 255]);
            }
        }

        render_time_grid(&mut pb, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 0, 1, 30);

        // abs_idx=0 → col_idx=0 → x = SPEC_LEFT + SPEC_W - 1 (right edge).
        let x = SPEC_LEFT + SPEC_W - 1;
        let off = (SPEC_TOP * pb.width + x) * 4;
        // White (255) blended with yellow (B=0, G=255, R=255) at 30 %:
        //   B: 255*0.7 + 0*0.3   = 178
        //   G: 255*0.7 + 255*0.3 = 255
        //   R: 255*0.7 + 255*0.3 = 255
        assert!(
            pb.data[off] < 200,
            "grid dot blue should drop below white (got {})",
            pb.data[off]
        );
        assert!(pb.data[off + 1] >= 250, "green should stay high");
        assert!(pb.data[off + 2] >= 250, "red should stay high");
    }

    #[test]
    fn render_time_grid_no_marks_when_fewer_than_one_period_visible() {
        // Only 5 visible columns → no abs_idx is a multiple of 30
        // except idx 0 itself.  With first_visible_abs_idx=1, the
        // visible range is 1..=5, none of which hit idx 0, 30, 60,...
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);
        render_time_grid(&mut pb, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 1, 5, 30);

        // Nothing drawn: whole spec area still BG.
        for y in SPEC_TOP..SPEC_BOTTOM {
            for x in SPEC_LEFT..SPEC_RIGHT {
                let off = (y * pb.width + x) * 4;
                assert_eq!(
                    &pb.data[off..off + 4],
                    &BG_COLOR,
                    "no marks should have been drawn at ({},{})",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn render_time_grid_noop_for_zero_visible_columns() {
        // Sanity: should not panic or modify the buffer.
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);
        render_time_grid(&mut pb, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 0, 0, 30);
        // Buffer should still be entirely BG.
        for y in SPEC_TOP..SPEC_BOTTOM {
            for x in SPEC_LEFT..SPEC_RIGHT {
                let off = (y * pb.width + x) * 4;
                assert_eq!(&pb.data[off..off + 4], &BG_COLOR);
            }
        }
    }

    #[test]
    fn render_spectrogram_dim_produces_darker_pixels_than_full() {
        // Same history, two different dim factors: the dim render
        // should produce strictly lower channel intensities at the
        // same pixel coordinates.  This validates the Phase 0
        // auto-pause dimming layer.
        let history: Vec<Vec<f32>> = (0..SPEC_W).map(|_| vec![1.0; SPEC_H]).collect();

        let mut pb_full = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb_full.clear(BG_COLOR);
        render_spectrogram(
            &mut pb_full,
            &history,
            SPEC_LEFT,
            SPEC_TOP,
            SPEC_W,
            SPEC_H,
            1.0,
            None,
            1.0,
        );

        let mut pb_dim = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb_dim.clear(BG_COLOR);
        render_spectrogram(
            &mut pb_dim,
            &history,
            SPEC_LEFT,
            SPEC_TOP,
            SPEC_W,
            SPEC_H,
            1.0,
            None,
            DIM_FACTOR_PAUSED,
        );

        // Sample the middle column, middle row.
        let sample_x = SPEC_LEFT + SPEC_W / 2;
        let sample_y = SPEC_TOP + SPEC_H / 2;
        let off = (sample_y * pb_full.width + sample_x) * 4;

        let full_intensity =
            pb_full.data[off] as u32 + pb_full.data[off + 1] as u32 + pb_full.data[off + 2] as u32;
        let dim_intensity =
            pb_dim.data[off] as u32 + pb_dim.data[off + 1] as u32 + pb_dim.data[off + 2] as u32;

        assert!(
            full_intensity > 0,
            "full-bright spectrogram pixel should have non-zero intensity"
        );
        assert!(
            dim_intensity < full_intensity,
            "dim ({}) must be strictly less than full ({})",
            dim_intensity,
            full_intensity
        );
    }

    #[test]
    fn render_spectrogram_zero_column_produces_no_pixels() {
        // A column of all zeros (as pushed during auto-pause) should
        // render no visible content at that column's x position.
        // This validates the Phase 0 "empty column / hole" behaviour.
        let history: Vec<Vec<f32>> = vec![vec![0.0f32; SPEC_H]; SPEC_W];
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);
        render_spectrogram(
            &mut pb, &history, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 1.0, None, 1.0,
        );

        // No pixel in the spec area should have been modified from
        // the background color.
        let mut non_bg = 0;
        for y in SPEC_TOP..SPEC_BOTTOM {
            for x in SPEC_LEFT..SPEC_RIGHT {
                let off = (y * pb.width + x) * 4;
                if pb.data[off..off + 4] != BG_COLOR {
                    non_bg += 1;
                }
            }
        }
        assert_eq!(
            non_bg, 0,
            "all-zero columns should produce no non-background pixels (got {})",
            non_bg
        );
    }

    #[test]
    fn render_spectrogram_right_aligned() {
        let mut pb = PixelBuffer::new(BADGE_W as usize, BADGE_H as usize);
        pb.clear(BG_COLOR);

        // Only 5 columns of history — should right-align.
        let history: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0; SPEC_H]).collect();
        render_spectrogram(
            &mut pb, &history, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 1.0, None, 1.0,
        );

        // Leftmost columns of spectrogram area should still be BG.
        let check_col = SPEC_LEFT;
        let mid_y = SPEC_TOP + SPEC_H / 2;
        let off = (mid_y * pb.width + check_col) * 4;
        assert_eq!(
            &pb.data[off..off + 4],
            &BG_COLOR,
            "leftmost spectrogram column should remain background when history is short"
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

        // Interior centre should remain at background color (no border there).
        let cx = BADGE_W as usize / 2;
        let cy = BADGE_H as usize / 2;
        let off_centre = (cy * pb.width + cx) * 4;
        assert_eq!(
            &pb.data[off_centre..off_centre + 4],
            &BG_COLOR,
            "badge interior should remain at background color"
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
            &mut pb, &history, SPEC_LEFT, SPEC_TOP, SPEC_W, SPEC_H, 1.0, None, 1.0,
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
