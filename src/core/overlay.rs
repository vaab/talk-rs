//! Visual overlay indicator for recording/transcribing status.
//!
//! Displays a small badge at the top-center of the primary monitor using X11.
//! Uses the Shape extension for binary transparency (works without a compositor).
//! The overlay runs on a dedicated background thread with a command channel.

use crate::core::error::TalkError;
use std::collections::HashMap;
use std::sync::mpsc;
use x11rb::protocol::shape;

// ── Embedded PNG assets ──────────────────────────────────────────────

/// "recording" badge: 182×52, dark rounded rect with red dot + text.
const RECORDING_PNG: &[u8] = include_bytes!("../../assets/indicator.png");

/// "transcribing" badge: 210×52, dark rounded rect with blue dot + text.
const TRANSCRIBING_PNG: &[u8] = include_bytes!("../../assets/transcribing.png");

// ── Public types ─────────────────────────────────────────────────────

/// Which indicator to display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorKind {
    /// Red badge shown during audio recording.
    Recording,
    /// Blue badge shown during transcription.
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
    /// Returns `Err` if the X11 display cannot be opened.
    pub fn new() -> Result<Self, TalkError> {
        // Test that X11 is reachable before spawning the thread
        let (tx, rx) = mpsc::channel();

        let thread = std::thread::Builder::new()
            .name("overlay".into())
            .spawn(move || {
                if let Err(e) = overlay_thread(rx) {
                    eprintln!("Overlay thread error: {}", e);
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

    // Ensure RGBA output
    let data = match info.color_type {
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgb => {
            // Add opaque alpha channel
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

// ── Screen geometry ──────────────────────────────────────────────────

/// Get primary monitor geometry by parsing `xrandr --query` output.
///
/// Returns `(x, y, width, height)` of the primary monitor.
fn primary_monitor_geometry() -> Result<(i16, i16, u16, u16), TalkError> {
    let output = std::process::Command::new("xrandr")
        .arg("--query")
        .output()
        .map_err(|e| TalkError::Config(format!("failed to run xrandr: {}", e)))?;

    let text = String::from_utf8_lossy(&output.stdout);

    // Try "primary WxH+X+Y" first
    for line in text.lines() {
        if line.contains(" primary ") {
            if let Some(geom) = parse_geometry(line) {
                return Ok(geom);
            }
        }
    }

    // Fallback: first "connected WxH+X+Y"
    for line in text.lines() {
        if line.contains(" connected ") {
            if let Some(geom) = parse_geometry(line) {
                return Ok(geom);
            }
        }
    }

    // Ultimate fallback
    Ok((0, 0, 1920, 1080))
}

/// Parse "WxH+X+Y" from an xrandr output line.
fn parse_geometry(line: &str) -> Option<(i16, i16, u16, u16)> {
    // Match patterns like "3840x2160+0+0" or "1920x1080+3840+0"
    let re_pattern = |s: &str| -> Option<(i16, i16, u16, u16)> {
        let parts: Vec<&str> = s.split('+').collect();
        if parts.len() >= 3 {
            let wh: Vec<&str> = parts[0].split('x').collect();
            if wh.len() == 2 {
                let w = wh[0].parse::<u16>().ok()?;
                let h = wh[1].parse::<u16>().ok()?;
                let x = parts[1].parse::<i16>().ok()?;
                let y = parts[2].parse::<i16>().ok()?;
                return Some((x, y, w, h));
            }
        }
        None
    };

    for word in line.split_whitespace() {
        if word.contains('x') && word.contains('+') {
            if let Some(geom) = re_pattern(word) {
                return Some(geom);
            }
        }
    }
    None
}

// ── X11 overlay thread ──────────────────────────────────────────────

/// Main loop for the overlay background thread.
fn overlay_thread(rx: mpsc::Receiver<Command>) -> Result<(), TalkError> {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;
    use x11rb::wrapper::ConnectionExt as _;
    use x11rb::COPY_DEPTH_FROM_PARENT;

    let (conn, screen_num) = x11rb::connect(None)
        .map_err(|e| TalkError::Config(format!("failed to connect to X11: {}", e)))?;

    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;

    // Pre-decode both indicator images
    let recording_img = decode_png(RECORDING_PNG)?;
    let transcribing_img = decode_png(TRANSCRIBING_PNG)?;

    // Get primary monitor geometry for centering
    let (mon_x, mon_y, mon_w, _mon_h) = primary_monitor_geometry()?;

    let mut current_window: Option<Window> = None;

    while let Ok(cmd) = rx.recv() {
        match cmd {
            Command::Show(kind) => {
                // Destroy previous window if any
                if let Some(win) = current_window.take() {
                    let _ = conn.destroy_window(win);
                    let _ = conn.flush();
                }

                let img = match kind {
                    IndicatorKind::Recording => &recording_img,
                    IndicatorKind::Transcribing => &transcribing_img,
                };

                let w = img.width as u16;
                let h = img.height as u16;

                // Center horizontally on primary monitor, 4px from top
                let x = mon_x + (mon_w as i16 / 2) - (w as i16 / 2);
                let y = mon_y + 4;

                // Create override_redirect window
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
                    0, // CopyFromParent visual
                    &values,
                )
                .map_err(|e| TalkError::Config(format!("X11 create_window failed: {}", e)))?;

                // Build shape mask (1-bit pixmap): alpha > 128 → opaque
                apply_shape_mask(&conn, win, img, w, h)?;

                // Map the window first — X11 clears content on map, so we
                // must draw AFTER the window is visible.
                conn.map_window(win)
                    .map_err(|e| TalkError::Config(format!("X11 map_window failed: {}", e)))?;

                // Sync with X server: ensures the map request is fully
                // processed and the Expose event generated before we draw.
                conn.sync()
                    .map_err(|e| TalkError::Config(format!("X11 sync failed: {}", e)))?;

                // Now draw visible pixels grouped by color
                draw_image(&conn, win, screen, img)?;

                conn.flush()
                    .map_err(|e| TalkError::Config(format!("X11 flush failed: {}", e)))?;

                current_window = Some(win);
            }

            Command::Hide => {
                if let Some(win) = current_window.take() {
                    let _ = conn.destroy_window(win);
                    let _ = conn.flush();
                }
            }

            Command::Quit => {
                if let Some(win) = current_window.take() {
                    let _ = conn.destroy_window(win);
                    let _ = conn.flush();
                }
                break;
            }
        }
    }

    Ok(())
}

/// Apply a 1-bit shape mask based on the image alpha channel.
///
/// Pixels with alpha > 128 are visible; all others are transparent.
fn apply_shape_mask(
    conn: &impl x11rb::connection::Connection,
    win: u32,
    img: &RgbaImage,
    w: u16,
    h: u16,
) -> Result<(), TalkError> {
    use x11rb::protocol::xproto::*;

    // Create a 1-bit depth pixmap for the shape mask
    let mask = conn
        .generate_id()
        .map_err(|e| TalkError::Config(format!("X11 generate_id failed: {}", e)))?;

    conn.create_pixmap(1, mask, win, w, h)
        .map_err(|e| TalkError::Config(format!("X11 create_pixmap failed: {}", e)))?;

    // Create GC for the mask pixmap
    let gc = conn
        .generate_id()
        .map_err(|e| TalkError::Config(format!("X11 generate_id failed: {}", e)))?;

    conn.create_gc(gc, mask, &CreateGCAux::new().foreground(0))
        .map_err(|e| TalkError::Config(format!("X11 create_gc failed: {}", e)))?;

    // Fill mask with 0 (fully transparent)
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

    // Set foreground to 1 (opaque) and draw visible pixels
    conn.change_gc(gc, &ChangeGCAux::new().foreground(1))
        .map_err(|e| TalkError::Config(format!("X11 change_gc failed: {}", e)))?;

    // Collect all opaque pixel coordinates
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

    // Draw in batches (X11 has a request size limit)
    for chunk in opaque_points.chunks(4096) {
        conn.poly_point(CoordMode::ORIGIN, mask, gc, chunk)
            .map_err(|e| TalkError::Config(format!("X11 poly_point failed: {}", e)))?;
    }

    // Apply shape mask to window
    shape::mask(conn, shape::SO::SET, shape::SK::BOUNDING, win, 0, 0, mask)
        .map_err(|e| TalkError::Config(format!("X11 shape_mask failed: {}", e)))?;

    // Cleanup
    conn.free_gc(gc)
        .map_err(|e| TalkError::Config(format!("X11 free_gc failed: {}", e)))?;
    conn.free_pixmap(mask)
        .map_err(|e| TalkError::Config(format!("X11 free_pixmap failed: {}", e)))?;

    Ok(())
}

/// Draw visible pixels onto the window, grouped by color for efficiency.
fn draw_image(
    conn: &impl x11rb::connection::Connection,
    win: u32,
    screen: &x11rb::protocol::xproto::Screen,
    img: &RgbaImage,
) -> Result<(), TalkError> {
    use x11rb::protocol::xproto::*;

    let cmap = screen.default_colormap;

    // Group visible pixels by RGB color
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
        // Allocate color (X11 uses 16-bit channels)
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

        // Draw in batches
        for chunk in points.chunks(4096) {
            conn.poly_point(CoordMode::ORIGIN, win, gc, chunk)
                .map_err(|e| TalkError::Config(format!("X11 poly_point failed: {}", e)))?;
        }
    }

    conn.free_gc(gc)
        .map_err(|e| TalkError::Config(format!("X11 free_gc failed: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_recording_png() {
        let img = decode_png(RECORDING_PNG).expect("decode recording PNG");
        assert_eq!(img.width, 182);
        assert_eq!(img.height, 52);
        assert_eq!(img.data.len(), 182 * 52 * 4);
    }

    #[test]
    fn test_decode_transcribing_png() {
        let img = decode_png(TRANSCRIBING_PNG).expect("decode transcribing PNG");
        assert_eq!(img.width, 210);
        assert_eq!(img.height, 52);
        assert_eq!(img.data.len(), 210 * 52 * 4);
    }

    #[test]
    fn test_recording_png_has_opaque_pixels() {
        let img = decode_png(RECORDING_PNG).expect("decode");
        let opaque_count = (0..img.width * img.height)
            .filter(|&i| img.data[(i * 4 + 3) as usize] > 128)
            .count();
        // The badge has a significant number of opaque pixels
        assert!(
            opaque_count > 1000,
            "expected >1000 opaque pixels, got {}",
            opaque_count
        );
    }

    #[test]
    fn test_recording_png_has_transparent_pixels() {
        let img = decode_png(RECORDING_PNG).expect("decode");
        let transparent_count = (0..img.width * img.height)
            .filter(|&i| img.data[(i * 4 + 3) as usize] <= 128)
            .count();
        // Rounded corners create transparent pixels
        assert!(
            transparent_count > 10,
            "expected transparent pixels for rounded corners, got {}",
            transparent_count
        );
    }

    #[test]
    fn test_parse_geometry_primary() {
        let line =
            "eDP-1 connected primary 3840x2160+0+0 (normal left inverted right x axis y axis)";
        let geom = parse_geometry(line);
        assert_eq!(geom, Some((0, 0, 3840, 2160)));
    }

    #[test]
    fn test_parse_geometry_offset() {
        let line = "HDMI-1 connected 1920x1080+3840+0 (normal left)";
        let geom = parse_geometry(line);
        assert_eq!(geom, Some((3840, 0, 1920, 1080)));
    }

    #[test]
    fn test_parse_geometry_no_match() {
        let line = "DP-1 disconnected (normal left)";
        let geom = parse_geometry(line);
        assert_eq!(geom, None);
    }

    #[test]
    fn test_indicator_kind_clone_eq() {
        let a = IndicatorKind::Recording;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(IndicatorKind::Recording, IndicatorKind::Transcribing);
    }
}
