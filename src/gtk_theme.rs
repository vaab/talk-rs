//! GTK4 theme colour resolution.
//!
//! Probes the current GTK theme for concrete hex colour values so
//! that hand-written CSS is state-independent (no `:backdrop`
//! surprises from the theme engine).

use gtk4::prelude::*;

/// Concrete hex colour strings resolved from the active GTK4 theme.
pub struct ThemeColors {
    /// Background: `@theme_base_color` darkened (dark themes) or barely
    /// tinted (light themes) to approximate `@view_bg_color`.
    pub view_bg: String,
    /// Selection highlight: `@theme_selected_bg_color` at 32 % opacity.
    pub selection: String,
    /// Foreground: `@theme_fg_color` pinned so selected rows stay readable.
    pub foreground: String,
    /// Error colour: `@error_color` (falls back to CSS name if unavailable).
    pub error: String,
}

impl ThemeColors {
    /// Probe the running GTK theme and resolve colour values.
    ///
    /// Must be called after `gtk4::init()`.
    #[allow(deprecated)]
    pub fn resolve() -> Self {
        let probe = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
        let ctx = probe.style_context();

        let rgba_to_hex = |c: &gtk4::gdk::RGBA| -> String {
            format!(
                "#{:02x}{:02x}{:02x}",
                (c.red() * 255.0).round() as u8,
                (c.green() * 255.0).round() as u8,
                (c.blue() * 255.0).round() as u8,
            )
        };

        let view_bg = if let Some(base) = ctx.lookup_color("theme_base_color") {
            let luma = 0.299 * base.red() + 0.587 * base.green() + 0.114 * base.blue();
            let factor: f32 = if luma < 0.5 { 0.77 } else { 0.98 };
            let to_u8 =
                |v: f32| -> u8 { ((v * factor * 255.0).round() as i32).clamp(0, 255) as u8 };
            format!(
                "#{:02x}{:02x}{:02x}",
                to_u8(base.red()),
                to_u8(base.green()),
                to_u8(base.blue()),
            )
        } else {
            "@theme_base_color".to_string()
        };

        let error = ctx
            .lookup_color("error_color")
            .map(|c| rgba_to_hex(&c))
            .unwrap_or_else(|| "@error_color".to_string());

        let selection = ctx
            .lookup_color("theme_selected_bg_color")
            .map(|c| {
                format!(
                    "rgba({},{},{},0.32)",
                    (c.red() * 255.0).round() as u8,
                    (c.green() * 255.0).round() as u8,
                    (c.blue() * 255.0).round() as u8,
                )
            })
            .unwrap_or_else(|| "alpha(@theme_selected_bg_color, 0.32)".to_string());

        let foreground = ctx
            .lookup_color("theme_fg_color")
            .map(|c| rgba_to_hex(&c))
            .unwrap_or_else(|| "@theme_fg_color".to_string());

        Self {
            view_bg,
            selection,
            foreground,
            error,
        }
    }

    /// Generate the shared base CSS rules, with optional extra rules appended.
    ///
    /// The base rules style the window chrome, scrollable list, row
    /// selection highlight, and the `.dim` utility class.  Callers
    /// append domain-specific rules via `extra_rules`.
    pub fn base_css(&self, extra_rules: &str) -> String {
        format!(
            "* {{ font-size: 13px; }} \
             window, window:backdrop {{ border: 1px solid alpha(white, 0.15); border-radius: 12px; background-color: {bg}; }} \
             scrolledwindow, viewport, list, \
             scrolledwindow:backdrop, viewport:backdrop, list:backdrop {{ background-color: transparent; background-image: none; border-radius: 10px; }} \
             row:not(:selected), row:not(:selected):backdrop {{ background-color: transparent; background-image: none; }} \
             row:selected, row:selected:backdrop {{ background-color: {sel}; color: {fg}; }} \
             .dim {{ opacity: 0.55; }} \
             row {{ border-radius: 8px; }} \
             .close-btn {{ min-width: 24px; min-height: 24px; padding: 0; opacity: 0.35; }} \
             .close-btn:hover {{ opacity: 0.9; }} \
             {extra}",
            bg = self.view_bg,
            fg = self.foreground,
            sel = self.selection,
            extra = extra_rules,
        )
    }
}

/// Load a CSS string as the application-level style provider.
pub fn load_css(css_text: &str) {
    let css = gtk4::CssProvider::new();
    css.load_from_data(css_text);
    if let Some(display) = gtk4::gdk::Display::default() {
        gtk4::style_context_add_provider_for_display(
            &display,
            &css,
            gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}

/// Build a minimal title bar with a right-aligned close button.
///
/// Returns `(title_bar_box, close_button)` so callers can connect
/// the button's `clicked` signal to their own close logic.
pub fn build_title_bar() -> (gtk4::Box, gtk4::Button) {
    let title_bar = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
    let spacer = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
    spacer.set_hexpand(true);
    title_bar.append(&spacer);

    let close_btn = gtk4::Button::from_icon_name("window-close-symbolic");
    close_btn.set_tooltip_text(Some("Close"));
    close_btn.add_css_class("close-btn");
    title_bar.append(&close_btn);

    (title_bar, close_btn)
}

/// Detect which window edge (if any) a pointer position is near.
///
/// Returns `Some(edge)` when the pointer is within `threshold` pixels
/// of a window border; corners take priority over sides.
fn detect_edge(x: f64, y: f64, w: f64, h: f64, threshold: f64) -> Option<gtk4::gdk::SurfaceEdge> {
    let top = y < threshold;
    let bottom = y > h - threshold;
    let left = x < threshold;
    let right = x > w - threshold;

    if top && left {
        Some(gtk4::gdk::SurfaceEdge::NorthWest)
    } else if top && right {
        Some(gtk4::gdk::SurfaceEdge::NorthEast)
    } else if bottom && left {
        Some(gtk4::gdk::SurfaceEdge::SouthWest)
    } else if bottom && right {
        Some(gtk4::gdk::SurfaceEdge::SouthEast)
    } else if top {
        Some(gtk4::gdk::SurfaceEdge::North)
    } else if bottom {
        Some(gtk4::gdk::SurfaceEdge::South)
    } else if left {
        Some(gtk4::gdk::SurfaceEdge::West)
    } else if right {
        Some(gtk4::gdk::SurfaceEdge::East)
    } else {
        None
    }
}

/// Map a window edge to the appropriate CSS resize cursor name.
fn edge_cursor(edge: Option<gtk4::gdk::SurfaceEdge>) -> Option<&'static str> {
    match edge {
        Some(gtk4::gdk::SurfaceEdge::North) | Some(gtk4::gdk::SurfaceEdge::South) => {
            Some("ns-resize")
        }
        Some(gtk4::gdk::SurfaceEdge::East) | Some(gtk4::gdk::SurfaceEdge::West) => {
            Some("ew-resize")
        }
        Some(gtk4::gdk::SurfaceEdge::NorthEast) | Some(gtk4::gdk::SurfaceEdge::SouthWest) => {
            Some("nesw-resize")
        }
        Some(gtk4::gdk::SurfaceEdge::NorthWest) | Some(gtk4::gdk::SurfaceEdge::SouthEast) => {
            Some("nwse-resize")
        }
        _ => None,
    }
}

/// Install edge/corner resize handles on an undecorated GTK4 window.
///
/// Undecorated windows lack window-manager resize grips.  This adds
/// motion and click controllers that detect pointer proximity to the
/// window border and initiate a GDK resize drag.
///
/// The click controller recomputes the edge from its own pointer
/// coordinates (rather than relying on cached motion state) to avoid
/// a race where hand jitter between the last motion event and the
/// press event causes the cached edge to be stale.
pub fn install_edge_resize(window: &gtk4::Window) {
    use std::cell::Cell;
    use std::rc::Rc;

    /// Pixel threshold from window border to show a resize cursor.
    const CURSOR_THRESHOLD: f64 = 6.0;

    /// Pixel threshold for clicks to initiate a resize — slightly
    /// larger than the cursor zone to forgive hand jitter between
    /// hover and press.
    const CLICK_THRESHOLD: f64 = 10.0;

    // Motion controller: detect edge and update cursor.
    {
        // Track previous edge to avoid redundant cursor changes.
        let prev_edge: Rc<Cell<Option<gtk4::gdk::SurfaceEdge>>> = Rc::new(Cell::new(None));
        let motion = gtk4::EventControllerMotion::new();
        let win = window.clone();
        let prev = Rc::clone(&prev_edge);
        motion.connect_motion(move |_ctrl, x, y| {
            let edge = detect_edge(
                x,
                y,
                win.width() as f64,
                win.height() as f64,
                CURSOR_THRESHOLD,
            );
            if edge != prev.get() {
                prev.set(edge);
                win.set_cursor_from_name(edge_cursor(edge));
            }
        });

        // Clear cursor when pointer leaves window.
        let prev_leave = Rc::clone(&prev_edge);
        let win_leave = window.clone();
        motion.connect_leave(move |_ctrl| {
            prev_leave.set(None);
            win_leave.set_cursor_from_name(None);
        });

        window.add_controller(motion);
    }

    // Click controller: initiate resize when on an edge.
    // Capture phase so it runs before WindowHandle's drag.
    //
    // Recomputes the edge from (x, y) at press time — does NOT read
    // the motion controller's cached edge.
    {
        let click = gtk4::GestureClick::new();
        click.set_propagation_phase(gtk4::PropagationPhase::Capture);
        let win = window.clone();
        click.connect_pressed(move |gesture, _n_press, x, y| {
            let edge = detect_edge(
                x,
                y,
                win.width() as f64,
                win.height() as f64,
                CLICK_THRESHOLD,
            );
            if let Some(edge) = edge {
                gesture.set_state(gtk4::EventSequenceState::Claimed);
                // Compute surface-relative coordinates for GDK.
                let (sx, sy) = if let Some(native) = win.native() {
                    let (nx, ny) = native.surface_transform();
                    (x + nx, y + ny)
                } else {
                    (x, y)
                };
                if let Some(native) = win.native() {
                    if let Some(surface) = native.surface() {
                        use gtk4::gdk::prelude::ToplevelExt;
                        if let Ok(toplevel) = surface.dynamic_cast::<gtk4::gdk::Toplevel>() {
                            let device = gesture.current_event().and_then(|e| e.device());
                            let timestamp = gesture.current_event().map_or(0, |e| e.time());
                            toplevel.begin_resize(edge, device.as_ref(), 1, sx, sy, timestamp);
                        }
                    }
                }
            } else {
                gesture.set_state(gtk4::EventSequenceState::Denied);
            }
        });
        window.add_controller(click);
    }
}

/// Present a window centred on the monitor containing the mouse pointer.
///
/// The window is initially invisible (`opacity = 0`), then centred and
/// revealed in the `map` signal so the user never sees a position jump.
/// Requires an X11 backend (`GDK_BACKEND=x11`).
pub fn present_centred(window: &gtk4::Window) {
    window.set_opacity(0.0);
    let window_reveal = window.clone();
    window.connect_map(move |w| {
        let xid = w
            .surface()
            .and_then(|s| s.downcast_ref::<gdk4_x11::X11Surface>().map(|xs| xs.xid()));
        if let Some(xid) = xid {
            #[allow(clippy::cast_possible_truncation)]
            let xid32 = xid as u32;
            crate::x11::x11_centre_and_raise_xid(xid32);
        }
        window_reveal.set_opacity(1.0);
    });
    window.present();
}
