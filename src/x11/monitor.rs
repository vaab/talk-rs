//! Primary monitor geometry via GDK4.
//!
//! Queries the GDK display for the monitor list and returns the
//! geometry of the largest monitor (heuristic for "primary").
//! Coordinates are returned in **physical pixels** (scaled by the
//! GDK scale factor) so they can be used directly for X11 window
//! placement.
//!
//! **Must** be called from a thread where GTK4 has been initialised
//! (or after `gtk4::init()`).

use crate::config::MonitorSelector;
use crate::error::TalkError;
use gtk4::prelude::*;

/// Primary monitor rectangle: `(x, y, width, height)` in physical pixels.
pub type MonitorGeometry = (i16, i16, u16, u16);

/// Query the geometry of the largest available monitor via GDK4.
///
/// GDK4 dropped the "primary" flag, so we pick the monitor with the
/// largest pixel area as a heuristic.  The returned coordinates are
/// multiplied by the monitor's scale factor to yield physical pixels
/// suitable for raw X11 positioning.
pub fn primary_monitor_geometry() -> Result<MonitorGeometry, TalkError> {
    monitor_geometry(MonitorSelector::Largest)
}

/// Query GDK monitors in physical pixels, then apply the shared selector.
pub fn monitor_geometry(selector: MonitorSelector) -> Result<MonitorGeometry, TalkError> {
    let display = gtk4::gdk::Display::default()
        .ok_or_else(|| TalkError::Config("no GDK display available".to_string()))?;

    let monitors = display.monitors();
    let count = monitors.n_items();

    if count == 0 {
        return Err(TalkError::Config("no monitors detected by GDK".to_string()));
    }

    let mut geometries = Vec::with_capacity(count as usize);
    for i in 0..count {
        let Some(obj) = monitors.item(i) else {
            continue;
        };
        let Ok(mon) = obj.downcast::<gtk4::gdk::Monitor>() else {
            continue;
        };
        let geo = mon.geometry();
        let scale = mon.scale_factor() as i64;
        geometries.push((
            (i64::from(geo.x()) * scale) as i16,
            (i64::from(geo.y()) * scale) as i16,
            (i64::from(geo.width()) * scale) as u16,
            (i64::from(geo.height()) * scale) as u16,
        ));
    }

    super::placement::select_monitor(&geometries, selector)
        .ok_or_else(|| TalkError::Config("no usable monitor found".to_string()))
}
