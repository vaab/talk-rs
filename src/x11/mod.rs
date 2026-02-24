//! X11 window management helpers.
//!
//! Centre, raise, and activate windows using the `x11rb` crate.
//! Shared by the dictate picker and recordings browser.

pub mod monitor;
pub mod overlay;
pub mod visualizer;

/// Centre a known X11 window on the monitor containing the mouse
/// pointer, set it always-on-top, and activate it.
///
/// This is the core positioning helper — callers supply the XID
/// directly (from GDK or from a title search).
pub fn x11_centre_and_raise_xid(wid: u32) -> bool {
    use x11rb::connection::Connection;
    use x11rb::protocol::randr::ConnectionExt as _;
    use x11rb::protocol::xproto::*;

    let (conn, screen_num) = match x11rb::connect(None) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;

    // ── Intern atoms ────────────────────────────────────────────
    let atom_names: &[&[u8]] = &[
        b"_NET_WM_STATE",
        b"_NET_WM_STATE_ABOVE",
        b"_NET_ACTIVE_WINDOW",
    ];
    let cookies: Vec<_> = atom_names
        .iter()
        .map(|n| conn.intern_atom(false, n))
        .collect::<Vec<_>>();
    let mut atoms = Vec::new();
    for cookie in cookies {
        let cookie = match cookie {
            Ok(c) => c,
            Err(_) => return false,
        };
        let atom = match cookie.reply() {
            Ok(r) => r.atom,
            Err(_) => return false,
        };
        atoms.push(atom);
    }
    let a_wm_state = atoms[0];
    let a_above = atoms[1];
    let a_active = atoms[2];

    // ── Query pointer position ──────────────────────────────────
    let pointer = match conn.query_pointer(root) {
        Ok(cookie) => match cookie.reply() {
            Ok(p) => p,
            Err(_) => return false,
        },
        Err(_) => return false,
    };
    let px = pointer.root_x as i32;
    let py = pointer.root_y as i32;

    // ── Find monitor at pointer via RandR ───────────────────────
    let (mon_x, mon_y, mon_w, mon_h) = {
        let default = (0i32, 0i32, 1920i32, 1080i32);
        let resources = match conn.randr_get_screen_resources(root) {
            Ok(cookie) => match cookie.reply() {
                Ok(r) => r,
                Err(_) => return false,
            },
            Err(_) => return false,
        };

        let mut found = default;
        let mut any_monitor = false;
        for &crtc in &resources.crtcs {
            let info = match conn.randr_get_crtc_info(crtc, 0) {
                Ok(cookie) => match cookie.reply() {
                    Ok(i) => i,
                    Err(_) => continue,
                },
                Err(_) => continue,
            };
            if info.width == 0 || info.height == 0 {
                continue;
            }
            let cx = info.x as i32;
            let cy = info.y as i32;
            let cw = info.width as i32;
            let ch = info.height as i32;

            if !any_monitor {
                found = (cx, cy, cw, ch);
                any_monitor = true;
            }

            if px >= cx && px < cx + cw && py >= cy && py < cy + ch {
                found = (cx, cy, cw, ch);
                break;
            }
        }
        found
    };

    // ── Get physical window geometry ────────────────────────────
    let geom = match conn.get_geometry(wid) {
        Ok(cookie) => match cookie.reply() {
            Ok(g) => g,
            Err(_) => return false,
        },
        Err(_) => return false,
    };

    let win_w = geom.width as i32;
    let win_h = geom.height as i32;
    if win_w == 0 || win_h == 0 {
        return false;
    }

    // ── Centre window on monitor ────────────────────────────────
    let x = mon_x + (mon_w - win_w) / 2;
    let y = mon_y + (mon_h - win_h) / 2;

    let _ = conn.configure_window(wid, &ConfigureWindowAux::new().x(x).y(y));

    // ── Set always-on-top (_NET_WM_STATE_ADD ABOVE) ─────────────
    let above_event = ClientMessageEvent::new(32, wid, a_wm_state, [1u32, a_above, 0, 0, 0]);
    let _ = conn.send_event(
        false,
        root,
        EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
        above_event,
    );

    // ── Activate window (_NET_ACTIVE_WINDOW) ────────────────────
    let activate_event = ClientMessageEvent::new(32, wid, a_active, [1u32, 0, 0, 0, 0]);
    let _ = conn.send_event(
        false,
        root,
        EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
        activate_event,
    );

    let _ = conn.flush();
    true
}

/// Find the X11 window whose `_NET_WM_NAME` matches `title`, centre
/// it on the monitor containing the mouse pointer, set it
/// always-on-top, and activate it.
///
/// Used for single-instance detection (raising an already-open
/// picker).  For newly created windows, prefer
/// [`x11_centre_and_raise_xid`] with the XID obtained from
/// [`gdk4_x11`] — it avoids the `_NET_CLIENT_LIST` race entirely.
pub fn x11_centre_and_raise(title: &str) -> bool {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;

    let (conn, screen_num) = match x11rb::connect(None) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;

    // ── Intern atoms for title search ───────────────────────────
    let atom_names: &[&[u8]] = &[b"_NET_CLIENT_LIST", b"_NET_WM_NAME", b"UTF8_STRING"];
    let cookies: Vec<_> = atom_names
        .iter()
        .map(|n| conn.intern_atom(false, n))
        .collect::<Vec<_>>();
    let mut atoms = Vec::new();
    for cookie in cookies {
        let cookie = match cookie {
            Ok(c) => c,
            Err(_) => return false,
        };
        let atom = match cookie.reply() {
            Ok(r) => r.atom,
            Err(_) => return false,
        };
        atoms.push(atom);
    }
    let a_client_list = atoms[0];
    let a_wm_name = atoms[1];
    let a_utf8 = atoms[2];

    // ── Find window by _NET_WM_NAME ─────────────────────────────
    let client_list = match conn.get_property(false, root, a_client_list, AtomEnum::WINDOW, 0, 1024)
    {
        Ok(cookie) => match cookie.reply() {
            Ok(prop) => prop,
            Err(_) => return false,
        },
        Err(_) => return false,
    };

    let wid_vec: Vec<u32> = match client_list.value32() {
        Some(iter) => iter.collect(),
        None => return false,
    };

    for &wid in &wid_vec {
        // Try _NET_WM_NAME (UTF-8) first.
        if let Ok(cookie) = conn.get_property(false, wid, a_wm_name, a_utf8, 0, 256) {
            if let Ok(prop) = cookie.reply() {
                if String::from_utf8_lossy(&prop.value) == title {
                    return x11_centre_and_raise_xid(wid);
                }
            }
        }
        // Fallback: WM_NAME (Latin-1).
        if let Ok(cookie) =
            conn.get_property(false, wid, AtomEnum::WM_NAME, AtomEnum::STRING, 0, 256)
        {
            if let Ok(prop) = cookie.reply() {
                if String::from_utf8_lossy(&prop.value) == title {
                    return x11_centre_and_raise_xid(wid);
                }
            }
        }
    }

    false
}

/// Activate (focus) a window by XID via `_NET_ACTIVE_WINDOW`.
///
/// Sends a ClientMessage to the root window requesting the window
/// manager to bring `wid` to the foreground.  Returns `true` if the
/// request was sent successfully, `false` on connection or protocol
/// error.
///
/// Equivalent to `xdotool windowactivate <wid>` (without `--sync`).
pub fn x11_activate_window(wid: u32) -> bool {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;

    let (conn, screen_num) = match x11rb::connect(None) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let root = conn.setup().roots[screen_num].root;

    let atom = match conn.intern_atom(false, b"_NET_ACTIVE_WINDOW") {
        Ok(cookie) => match cookie.reply() {
            Ok(r) => r.atom,
            Err(_) => return false,
        },
        Err(_) => return false,
    };

    // data[0] = 2 → "message from a pager" (same as xdotool)
    // data[1] = 0 → CurrentTime
    let event = ClientMessageEvent::new(32, wid, atom, [2u32, 0, 0, 0, 0]);
    let _ = conn.send_event(
        false,
        root,
        EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
        event,
    );

    let _ = conn.flush();
    true
}

/// Simulate a key combination (e.g. Ctrl+Shift+V) using the XTest
/// extension.
///
/// `keysyms` is a slice of X11 keysyms to press simultaneously.
/// All keys are pressed in order, then released in reverse order,
/// matching the behaviour of `xdotool key`.
///
/// Returns `true` if the key events were sent, `false` on error.
pub fn x11_send_key_combo(keysyms: &[u32]) -> bool {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;
    use x11rb::protocol::xtest;

    if keysyms.is_empty() {
        return true;
    }

    let (conn, _screen_num) = match x11rb::connect(None) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let setup = conn.setup();
    let min_keycode = setup.min_keycode;
    let max_keycode = setup.max_keycode;

    // Fetch the full keyboard mapping so we can resolve keysym → keycode.
    let count = max_keycode - min_keycode + 1;
    let mapping = match conn.get_keyboard_mapping(min_keycode, count) {
        Ok(cookie) => match cookie.reply() {
            Ok(m) => m,
            Err(_) => return false,
        },
        Err(_) => return false,
    };
    let syms_per_code = mapping.keysyms_per_keycode as usize;

    let keysym_to_keycode = |sym: u32| -> Option<u8> {
        for i in 0..count as usize {
            for col in 0..syms_per_code {
                if mapping.keysyms[i * syms_per_code + col] == sym {
                    return Some((i as u8) + min_keycode);
                }
            }
        }
        None
    };

    // Resolve all keysyms to keycodes up-front.
    let keycodes: Vec<u8> = match keysyms.iter().map(|s| keysym_to_keycode(*s)).collect() {
        Some(v) => v,
        None => return false,
    };

    const KEY_PRESS: u8 = 2;
    const KEY_RELEASE: u8 = 3;
    // xdotool default inter-key delay: 12 000 µs.
    const DELAY: std::time::Duration = std::time::Duration::from_millis(12);

    // root=0 for key events (XTest spec: root is ignored for
    // KeyPress/KeyRelease; xdotool also passes None).
    let send = |type_: u8, kc: u8| -> bool {
        if xtest::fake_input(&conn, type_, kc, 0, 0u32, 0, 0, 0).is_err() {
            return false;
        }
        // Flush after every event, matching xdotool's XFlush-per-key.
        if conn.flush().is_err() {
            return false;
        }
        std::thread::sleep(DELAY);
        true
    };

    // Press all keys in order.
    for &kc in &keycodes {
        if !send(KEY_PRESS, kc) {
            return false;
        }
    }
    // Release all keys in reverse order.
    for &kc in keycodes.iter().rev() {
        if !send(KEY_RELEASE, kc) {
            return false;
        }
    }

    true
}

/// Send a single key press+release `count` times with no inter-key
/// delay, matching `xdotool key --delay 0 --repeat N <key>`.
///
/// Returns `true` if all events were sent, `false` on error.
pub fn x11_send_key_repeat(keysym: u32, count: usize) -> bool {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;
    use x11rb::protocol::xtest;

    if count == 0 {
        return true;
    }

    let (conn, _screen_num) = match x11rb::connect(None) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let setup = conn.setup();
    let min_keycode = setup.min_keycode;
    let max_keycode = setup.max_keycode;

    let km_count = max_keycode - min_keycode + 1;
    let mapping = match conn.get_keyboard_mapping(min_keycode, km_count) {
        Ok(cookie) => match cookie.reply() {
            Ok(m) => m,
            Err(_) => return false,
        },
        Err(_) => return false,
    };
    let syms_per_code = mapping.keysyms_per_keycode as usize;

    let keycode = (|| -> Option<u8> {
        for i in 0..km_count as usize {
            for col in 0..syms_per_code {
                if mapping.keysyms[i * syms_per_code + col] == keysym {
                    return Some((i as u8) + min_keycode);
                }
            }
        }
        None
    })();

    let kc = match keycode {
        Some(v) => v,
        None => return false,
    };

    const KEY_PRESS: u8 = 2;
    const KEY_RELEASE: u8 = 3;

    // Flush after every event, matching xdotool's XFlush-per-key.
    // Yield between events: xdotool calls usleep(0) even with
    // --delay 0, giving the X server time to consume each event.
    for _ in 0..count {
        if xtest::fake_input(&conn, KEY_PRESS, kc, 0, 0u32, 0, 0, 0).is_err() {
            return false;
        }
        if conn.flush().is_err() {
            return false;
        }
        std::thread::yield_now();
        if xtest::fake_input(&conn, KEY_RELEASE, kc, 0, 0u32, 0, 0, 0).is_err() {
            return false;
        }
        if conn.flush().is_err() {
            return false;
        }
        std::thread::yield_now();
    }

    true
}

/// Return the XID of the currently active (focused) window via
/// `_NET_ACTIVE_WINDOW` on the root window, or `None` if the query
/// fails.
///
/// Equivalent to `xdotool getactivewindow`.
pub fn x11_get_active_window() -> Option<u32> {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;

    let (conn, screen_num) = x11rb::connect(None).ok()?;
    let root = conn.setup().roots[screen_num].root;

    let atom = conn
        .intern_atom(false, b"_NET_ACTIVE_WINDOW")
        .ok()?
        .reply()
        .ok()?
        .atom;

    let prop = conn
        .get_property(false, root, atom, AtomEnum::WINDOW, 0, 1)
        .ok()?
        .reply()
        .ok()?;

    let wid = prop.value32()?.next()?;
    if wid == 0 {
        return None;
    }
    Some(wid)
}
