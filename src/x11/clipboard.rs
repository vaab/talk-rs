//! Native X11 clipboard via `x11rb`.
//!
//! Provides get/set operations on the `CLIPBOARD` selection without
//! shelling out to `xclip`.  The setter spawns a short-lived
//! background thread that serves `SelectionRequest` events so the
//! paste target can retrieve the data.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use x11rb::connection::Connection as _;
use x11rb::protocol::xproto::*;
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;

/// How long the serve thread waits for paste consumers before
/// self-terminating.
const DEFAULT_SERVE_DURATION: Duration = Duration::from_secs(5);

/// Grace period after stop is requested — gives clipboard managers
/// time to grab restored content before the window is destroyed.
const GRACE_PERIOD: Duration = Duration::from_millis(200);

/// Handle to a background thread serving clipboard content.
///
/// Dropping the handle signals a graceful stop (with a short grace
/// period) and joins the thread.
pub struct ClipboardServeHandle {
    stop: Arc<AtomicBool>,
    skip_grace: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ClipboardServeHandle {
    /// Stop serving immediately — no grace period.
    ///
    /// Use when replacing the clipboard content (the old content is
    /// no longer needed).
    pub fn stop_now(&self) {
        self.skip_grace.store(true, Ordering::Relaxed);
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl Drop for ClipboardServeHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Read the current `CLIPBOARD` content as UTF-8 text, or `None` if
/// the clipboard is empty or the request fails.
///
/// This is a one-shot blocking call: connect → `ConvertSelection` →
/// wait for `SelectionNotify` → read property → disconnect.
pub fn x11_clipboard_get() -> Option<String> {
    let (conn, screen_num) = x11rb::connect(None).ok()?;
    let setup = conn.setup();
    let root = setup.roots[screen_num].root;

    let window = conn.generate_id().ok()?;
    conn.create_window(
        0,
        window,
        root,
        0,
        0,
        1,
        1,
        0,
        WindowClass::INPUT_ONLY,
        0,
        &CreateWindowAux::new(),
    )
    .ok()?;

    let clipboard = intern(&conn, b"CLIPBOARD")?;
    let utf8_string = intern(&conn, b"UTF8_STRING")?;
    let property = intern(&conn, b"TALK_RS_SEL")?;

    conn.convert_selection(window, clipboard, utf8_string, property, 0u32)
        .ok()?;
    conn.flush().ok()?;

    let deadline = Instant::now() + Duration::from_secs(2);
    let result = loop {
        if Instant::now() > deadline {
            break None;
        }
        let event = match conn.poll_for_event() {
            Ok(Some(e)) => e,
            Ok(None) => {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            Err(_) => break None,
        };
        if let Event::SelectionNotify(notify) = event {
            if notify.requestor == window && notify.selection == clipboard {
                if notify.property == 0u32 {
                    // Conversion failed (empty clipboard or no owner).
                    break None;
                }
                let prop = conn
                    .get_property(true, window, property, utf8_string, 0, u32::MAX)
                    .ok()?
                    .reply()
                    .ok()?;
                break String::from_utf8(prop.value).ok();
            }
        }
    };

    let _ = conn.destroy_window(window);
    let _ = conn.flush();
    result
}

/// Claim `CLIPBOARD` ownership and serve the content from a
/// background thread.
///
/// Returns a [`ClipboardServeHandle`] that keeps the thread alive.
/// The thread terminates when:
/// - Another application takes ownership (`SelectionClear`).
/// - The handle is dropped (with a short grace period for clipboard
///   managers).
/// - `stop_now()` is called (immediate, no grace).
/// - The default serve duration (5 s) expires.
///
/// Returns `None` if the X11 connection or ownership claim fails.
pub fn x11_clipboard_set(text: &str) -> Option<ClipboardServeHandle> {
    let (conn, screen_num) = x11rb::connect(None).ok()?;
    let setup = conn.setup();
    let root = setup.roots[screen_num].root;

    let window = conn.generate_id().ok()?;
    conn.create_window(
        0,
        window,
        root,
        0,
        0,
        1,
        1,
        0,
        WindowClass::INPUT_ONLY,
        0,
        &CreateWindowAux::new(),
    )
    .ok()?;

    let clipboard = intern(&conn, b"CLIPBOARD")?;
    let utf8_string = intern(&conn, b"UTF8_STRING")?;
    let targets_atom = intern(&conn, b"TARGETS")?;

    conn.set_selection_owner(window, clipboard, 0u32).ok()?;
    conn.flush().ok()?;

    // Verify we actually got ownership.
    let owner = conn
        .get_selection_owner(clipboard)
        .ok()?
        .reply()
        .ok()?
        .owner;
    if owner != window {
        let _ = conn.destroy_window(window);
        let _ = conn.flush();
        return None;
    }

    let stop = Arc::new(AtomicBool::new(false));
    let skip_grace = Arc::new(AtomicBool::new(false));
    let stop2 = Arc::clone(&stop);
    let skip2 = Arc::clone(&skip_grace);
    let text = text.to_string();

    let atoms = Atoms {
        clipboard,
        utf8_string,
        targets: targets_atom,
    };

    let handle = std::thread::spawn(move || {
        serve_loop(&conn, &atoms, &text, &stop2, &skip2);
        let _ = conn.destroy_window(window);
        let _ = conn.flush();
    });

    Some(ClipboardServeHandle {
        stop,
        skip_grace,
        handle: Some(handle),
    })
}

// ── helpers ──────────────────────────────────────────────────────

/// Pre-interned atoms used by the clipboard helpers.
struct Atoms {
    clipboard: Atom,
    utf8_string: Atom,
    targets: Atom,
}

fn intern(conn: &RustConnection, name: &[u8]) -> Option<Atom> {
    conn.intern_atom(false, name)
        .ok()?
        .reply()
        .ok()
        .map(|r| r.atom)
}

/// Main event loop for the clipboard serve thread.
fn serve_loop(
    conn: &RustConnection,
    atoms: &Atoms,
    text: &str,
    stop: &AtomicBool,
    skip_grace: &AtomicBool,
) {
    let deadline = Instant::now() + DEFAULT_SERVE_DURATION;

    loop {
        if stop.load(Ordering::Relaxed) || Instant::now() > deadline {
            // Grace period: keep serving briefly so clipboard managers
            // can grab restored content — unless skip_grace is set.
            if !skip_grace.load(Ordering::Relaxed) {
                let grace_end = Instant::now() + GRACE_PERIOD;
                while Instant::now() < grace_end {
                    poll_and_serve(conn, atoms, text);
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            break;
        }

        match conn.poll_for_event() {
            Ok(Some(Event::SelectionRequest(req))) if req.selection == atoms.clipboard => {
                serve_request(conn, &req, text, atoms);
            }
            Ok(Some(Event::SelectionClear(ev))) if ev.selection == atoms.clipboard => {
                // Another app took ownership — exit immediately.
                break;
            }
            Ok(_) => {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(_) => break,
        }
    }
}

/// One non-blocking poll-and-serve cycle (used during grace period).
fn poll_and_serve(conn: &RustConnection, atoms: &Atoms, text: &str) {
    if let Ok(Some(Event::SelectionRequest(req))) = conn.poll_for_event() {
        if req.selection == atoms.clipboard {
            serve_request(conn, &req, text, atoms);
        }
    }
}

/// Respond to a single `SelectionRequest`.
fn serve_request(conn: &RustConnection, req: &SelectionRequestEvent, text: &str, atoms: &Atoms) {
    // Per ICCCM, if property is None use target as fallback.
    let prop = if req.property == 0u32 {
        req.target
    } else {
        req.property
    };

    let ok = if req.target == atoms.targets {
        // Advertise supported targets.
        let supported = [atoms.targets, atoms.utf8_string];
        conn.change_property32(
            PropMode::REPLACE,
            req.requestor,
            prop,
            AtomEnum::ATOM,
            &supported,
        )
        .is_ok()
    } else if req.target == atoms.utf8_string {
        // Provide the text.
        conn.change_property8(
            PropMode::REPLACE,
            req.requestor,
            prop,
            atoms.utf8_string,
            text.as_bytes(),
        )
        .is_ok()
    } else {
        false
    };

    let reply_prop = if ok { prop } else { 0u32 };

    let notify = SelectionNotifyEvent {
        response_type: SELECTION_NOTIFY_EVENT,
        sequence: 0,
        time: req.time,
        requestor: req.requestor,
        selection: req.selection,
        target: req.target,
        property: reply_prop,
    };

    let _ = conn.send_event(false, req.requestor, EventMask::NO_EVENT, notify);
    let _ = conn.flush();
}
