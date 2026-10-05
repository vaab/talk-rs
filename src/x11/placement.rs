//! Shared physical-pixel geometry for X11 indicator windows.

use super::monitor::MonitorGeometry;
use crate::config::{MonitorSelector, OverlayPosition};

/// Choose a physical monitor rectangle; equal ranks keep the first GDK item.
pub fn select_monitor(
    monitors: &[MonitorGeometry],
    selector: MonitorSelector,
) -> Option<MonitorGeometry> {
    let mut best = *monitors.first()?;
    for &candidate in &monitors[1..] {
        let better = match selector {
            MonitorSelector::Largest => area(candidate) > area(best),
            MonitorSelector::Smallest => area(candidate) < area(best),
            MonitorSelector::Leftmost => (candidate.0, candidate.1) < (best.0, best.1),
            MonitorSelector::Rightmost => {
                candidate.0 > best.0 || (candidate.0 == best.0 && candidate.1 < best.1)
            }
        };
        if better {
            best = candidate;
        }
    }
    Some(best)
}

fn area(rect: MonitorGeometry) -> u64 {
    u64::from(rect.2) * u64::from(rect.3)
}

/// Position within the free monitor span, leaving a scaled inset at each edge.
/// Oversized windows anchor at the monitor origin, never beyond its far edge.
pub fn resolve_rect(
    monitor: MonitorGeometry,
    win_w: u16,
    win_h: u16,
    position: OverlayPosition,
    margin: u16,
) -> (i16, i16) {
    let (x, y) = position.percentages();
    (
        resolve_axis(
            i32::from(monitor.0),
            i32::from(monitor.2),
            i32::from(win_w),
            x,
            i32::from(margin),
        ),
        resolve_axis(
            i32::from(monitor.1),
            i32::from(monitor.3),
            i32::from(win_h),
            y,
            i32::from(margin),
        ),
    )
}

fn resolve_axis(origin: i32, extent: i32, size: i32, pct: f32, margin: i32) -> i16 {
    if size >= extent {
        return origin.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    let inset = margin.min((extent - size) / 2);
    let free = extent - 2 * inset - size;
    let offset = if pct == 50.0 {
        // Preserve the legacy integer midpoint for odd monitor/window sizes.
        extent / 2 - size / 2
    } else {
        inset + (pct as f64 / 100.0 * f64::from(free)).round() as i32
    };
    (origin + offset).clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Position the text panel next to its badge, flipping above in the lower half.
pub fn adjacent_panel_rect(
    monitor: MonitorGeometry,
    badge: (i16, i16, u16, u16),
    panel_w: u16,
    panel_h: u16,
    gap: u16,
) -> (i16, i16) {
    let (mx, my, mw, mh) = monitor;
    let (bx, by, bw, bh) = badge;
    let centered_x = i32::from(bx) + i32::from(bw) / 2 - i32::from(panel_w) / 2;
    let min_x = i32::from(mx);
    let max_x = min_x + (i32::from(mw) - i32::from(panel_w)).max(0);
    let x = centered_x.clamp(min_x, max_x);
    let bottom_half = i32::from(by) + i32::from(bh) / 2 >= i32::from(my) + i32::from(mh) / 2;
    let adjacent_y = if bottom_half {
        i32::from(by) - i32::from(gap) - i32::from(panel_h)
    } else {
        i32::from(by) + i32::from(bh) + i32::from(gap)
    };
    let min_y = i32::from(my);
    let max_y = min_y + (i32::from(mh) - i32::from(panel_h)).max(0);
    (
        x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        adjacent_y
            .clamp(min_y, max_y)
            .clamp(i16::MIN as i32, i16::MAX as i32) as i16,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AxisPosition;

    #[test]
    fn resolve_rect_centers_and_edges_inside_free_span() {
        let monitor = (100, 200, 1000, 800);
        assert_eq!(
            resolve_rect(monitor, 200, 100, OverlayPosition::Top, 4),
            (500, 204)
        );
        assert_eq!(
            resolve_rect(monitor, 200, 100, OverlayPosition::BottomRight, 4),
            (896, 896)
        );
        assert_eq!(
            resolve_rect(monitor, 200, 100, OverlayPosition::Center, 4),
            (500, 550)
        );
        assert_eq!(
            resolve_rect(monitor, 200, 100, OverlayPosition::BottomLeft, 4),
            (104, 896)
        );
    }

    #[test]
    fn resolve_rect_percentages_and_oversized_windows() {
        let monitor = (0, 0, 1000, 800);
        let custom = OverlayPosition::Axes {
            x: AxisPosition::Percent(30.0),
            y: AxisPosition::Percent(70.0),
        };
        assert_eq!(resolve_rect(monitor, 200, 100, custom, 4), (242, 488));
        assert_eq!(resolve_rect(monitor, 1200, 900, custom, 4), (0, 0));
        assert_eq!(
            resolve_rect(monitor, 200, 100, OverlayPosition::Top, 20),
            (400, 20)
        );
    }

    #[test]
    fn monitor_selection_is_relative_and_ties_are_stable() {
        let monitors = [
            (-600, 100, 400, 400),
            (100, 0, 800, 300),
            (100, -40, 800, 300),
        ];
        assert_eq!(
            select_monitor(&monitors, MonitorSelector::Largest),
            Some(monitors[1])
        );
        assert_eq!(
            select_monitor(&monitors, MonitorSelector::Smallest),
            Some(monitors[0])
        );
        assert_eq!(
            select_monitor(&monitors, MonitorSelector::Leftmost),
            Some(monitors[0])
        );
        assert_eq!(
            select_monitor(&monitors, MonitorSelector::Rightmost),
            Some(monitors[2])
        );
        assert_eq!(select_monitor(&[], MonitorSelector::Largest), None);
    }

    #[test]
    fn text_panel_follows_badge_and_flips_above_near_bottom() {
        let monitor = (0, 0, 1920, 1080);
        assert_eq!(
            adjacent_panel_rect(monitor, (824, 4, 273, 52), 1200, 220, 4),
            (360, 60)
        );
        assert_eq!(
            adjacent_panel_rect(monitor, (1600, 900, 273, 52), 1200, 220, 4),
            (720, 676)
        );
        assert_eq!(
            adjacent_panel_rect((0, 0, 800, 600), (0, 4, 273, 52), 1200, 220, 4),
            (0, 60)
        );
    }
}
