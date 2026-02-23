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
