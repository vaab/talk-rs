//! Native X11 clipboard via `x11rb`.
//!
//! Provides get/set operations on the `CLIPBOARD` selection without
//! shelling out to `xclip`.  The setter spawns a short-lived
//! background thread that serves `SelectionRequest` events so the
//! paste target can retrieve the data.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::clipboard::{ClipboardSnapshot, ClipboardTarget};
use x11rb::connection::Connection as _;
use x11rb::protocol::xproto::*;
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;

const SNAPSHOT_LIMIT: usize = 32 * 1024 * 1024;
const META_TARGETS: &[&[u8]] = &[
    b"TARGETS",
    b"MULTIPLE",
    b"TIMESTAMP",
    b"SAVE_TARGETS",
    b"DELETE",
];

fn atom_name(conn: &RustConnection, atom: Atom) -> Option<String> {
    let reply = conn.get_atom_name(atom).ok()?.reply().ok()?;
    String::from_utf8(reply.name).ok()
}

fn request_target(
    conn: &RustConnection,
    window: Window,
    clipboard: Atom,
    target: Atom,
    property: Atom,
    limit: usize,
) -> Option<GetPropertyReply> {
    conn.delete_property(window, property).ok()?;
    conn.convert_selection(window, clipboard, target, property, 0u32)
        .ok()?;
    conn.flush().ok()?;
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        match conn.poll_for_event().ok()? {
            Some(Event::SelectionNotify(n))
                if n.requestor == window && n.selection == clipboard && n.target == target =>
            {
                if n.property == 0 {
                    return None;
                }
                return conn
                    .get_property(
                        true,
                        window,
                        property,
                        AtomEnum::ANY,
                        0,
                        (limit / 4 + 1) as u32,
                    )
                    .ok()?
                    .reply()
                    .ok();
            }
            Some(_) => {}
            None => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    None
}

/// Snapshot the current owner's content targets without altering ownership.
/// Oversized, incremental, or failed conversions are omitted with a warning.
pub fn x11_clipboard_snapshot() -> Result<Option<ClipboardSnapshot>, String> {
    let (conn, screen) = x11rb::connect(None).map_err(|e| e.to_string())?;
    let clipboard = intern(&conn, b"CLIPBOARD").ok_or("could not intern CLIPBOARD")?;
    let owner = conn
        .get_selection_owner(clipboard)
        .map_err(|e| e.to_string())?
        .reply()
        .map_err(|e| e.to_string())?
        .owner;
    if owner == 0 {
        return Ok(None);
    }
    let root = conn.setup().roots[screen].root;
    let window = conn.generate_id().map_err(|e| e.to_string())?;
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
    .map_err(|e| e.to_string())?;
    let targets_atom = intern(&conn, b"TARGETS").ok_or("could not intern TARGETS")?;
    let property = intern(&conn, b"TALK_RS_SEL").ok_or("could not intern property")?;
    let incr = intern(&conn, b"INCR").ok_or("could not intern INCR")?;
    let list = request_target(
        &conn,
        window,
        clipboard,
        targets_atom,
        property,
        SNAPSHOT_LIMIT,
    )
    .ok_or("could not fetch clipboard TARGETS")?;
    if list.format != 32 || list.bytes_after != 0 {
        return Err("invalid clipboard TARGETS".into());
    }
    let mut targets = Vec::new();
    let mut total = 0usize;
    for atom in list.value32().into_iter().flatten() {
        let Some(name) = atom_name(&conn, atom) else {
            continue;
        };
        if META_TARGETS.contains(&name.as_bytes()) {
            continue;
        }
        let remaining = SNAPSHOT_LIMIT.saturating_sub(total);
        if remaining == 0 {
            log::warn!("clipboard snapshot reached 32 MiB cap");
            break;
        }
        let Some(value) = request_target(&conn, window, clipboard, atom, property, remaining)
        else {
            log::warn!("clipboard snapshot: could not fetch target {name}");
            continue;
        };
        if value.type_ == incr || value.bytes_after != 0 || value.value.len() > remaining {
            log::warn!("clipboard snapshot: skipping incremental/oversized target {name}");
            continue;
        }
        if !matches!(value.format, 8 | 16 | 32) {
            continue;
        }
        let Some(property_type) = atom_name(&conn, value.type_) else {
            continue;
        };
        total += value.value.len();
        targets.push(ClipboardTarget {
            name,
            property_type,
            format: value.format,
            bytes: value.value,
        });
    }
    let _ = conn.destroy_window(window);
    let _ = conn.flush();
    Ok(Some(ClipboardSnapshot { targets }))
}

/// Compute the X11 client-base of a window XID for a given server
/// `resource_id_mask`.
///
/// X11 packs each client's resources (windows, atoms, pixmaps, …)
/// into a contiguous block: every XID minted by a given client
/// shares the same high-bits prefix.  The prefix is exactly
/// `xid & !resource_id_mask`.  Two windows from the SAME client
/// (e.g. a top-level toplevel and the focus-grabbing child widget
/// it created) therefore yield the same client-base — which is the
/// stable identity we need to gate paste chunks on, since the
/// paste-fetcher widget id is ephemeral but its parent client is
/// long-lived.
///
/// Pure function — no X11 connection required.  Exposed so the
/// paste pipeline and tests can both compute bases from raw XIDs.
pub fn client_base(xid: u32, resource_id_mask: u32) -> u32 {
    xid & !resource_id_mask
}

/// How long the serve thread waits for paste consumers before
/// self-terminating.
const DEFAULT_SERVE_DURATION: Duration = Duration::from_secs(5);

/// Grace period after stop is requested — gives clipboard managers
/// time to grab restored content before the window is destroyed.
const GRACE_PERIOD: Duration = Duration::from_millis(200);

/// Per-client-base UTF8_STRING fetch counts.
///
/// Keyed by `requestor & !resource_id_mask` (the X11 client-base),
/// each value is the number of `UTF8_STRING` SelectionRequests this
/// handle has answered for requestors belonging to that client.
///
/// The map deliberately distinguishes clients rather than tracking a
/// monolithic counter: clipboard managers and the actual paste
/// target each have their OWN client-base, and the dropped-chunk bug
/// proved that gating on "anyone fetched" advances the chunk before
/// the target has consumed it.  Tracking per-client fetches lets the
/// paste pipeline wait on the specific target's confirmation.
type FetchMap = Arc<Mutex<HashMap<u32, u32>>>;

/// Handle to a background thread serving clipboard content.
///
/// Dropping the handle signals a graceful stop (with a short grace
/// period) and joins the thread.
pub struct ClipboardServeHandle {
    stop: Arc<AtomicBool>,
    skip_grace: Arc<AtomicBool>,
    /// Per-client-base UTF8_STRING fetch counts.  See [`FetchMap`].
    fetches: FetchMap,
    /// X11 server's `resource_id_mask` — the mask used by the serve
    /// thread to derive each requestor's client-base.  Exposed so
    /// callers can mask target XIDs with the SAME value the serve
    /// loop is using (the mask is server-wide, so any connection
    /// would give the same answer — exposing the actual one removes
    /// the round-trip and prevents drift if the server ever changes
    /// it on reconnect).
    resource_id_mask: u32,
    /// Our own client-base (= the serve thread's `resource_id_base`
    /// masked with `resource_id_mask`).  Fetches with a requestor
    /// matching this client-base are NOT recorded — defensive
    /// exclusion of any self-directed requests, and a safe place to
    /// drop trace-only read-backs should they ever share our base.
    ///
    /// Captured here so the field survives in the handle for
    /// `#[cfg(test)]` introspection (see [`Self::own_client_base`]);
    /// the actual exclusion is performed by `serve_request` with a
    /// stack copy of the same value, which is why the field would
    /// otherwise be "never read" in non-test builds.
    #[allow(dead_code)] // Read only by the test-only accessor below.
    own_client_base: u32,
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

    /// Total `UTF8_STRING` `SelectionRequest`s answered across all
    /// non-own client-bases — i.e. how many times any external paste
    /// target pulled the offered text.  Backward-compat surface for
    /// the legacy blind-paste gate (`wait_until_served`) used when
    /// no target client-base could be resolved.
    pub fn served_count(&self) -> u32 {
        match self.fetches.lock() {
            Ok(map) => map.values().sum(),
            Err(_) => 0,
        }
    }

    /// Number of `UTF8_STRING` fetches answered for the given X11
    /// client-base.  Used by the deterministic per-chunk paste gate
    /// to confirm the actual target consumed each chunk.  Returns
    /// `0` when the client-base has not (yet) requested anything.
    pub fn fetches_by_client(&self, base: u32) -> u32 {
        match self.fetches.lock() {
            Ok(map) => map.get(&base).copied().unwrap_or(0),
            Err(_) => 0,
        }
    }

    /// X11 server `resource_id_mask` snapshotted at serve-thread
    /// creation.  Callers wanting to compute a client-base from a
    /// raw window XID with the SAME mask the serve loop uses should
    /// read it from here — see [`client_base`].
    pub fn resource_id_mask(&self) -> u32 {
        self.resource_id_mask
    }

    /// Our own client-base — exposed for test introspection.  Used
    /// internally by `serve_request` to drop self-directed fetches
    /// from the count.
    #[cfg(test)]
    pub fn own_client_base(&self) -> u32 {
        self.own_client_base
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
    x11_clipboard_set_snapshot(ClipboardSnapshot {
        targets: vec![ClipboardTarget {
            name: "UTF8_STRING".into(),
            property_type: "UTF8_STRING".into(),
            format: 8,
            bytes: text.as_bytes().to_vec(),
        }],
    })
}

pub fn x11_clipboard_set_snapshot(snapshot: ClipboardSnapshot) -> Option<ClipboardServeHandle> {
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

    let resource_id_mask = setup.resource_id_mask;
    let own_client_base = client_base(setup.resource_id_base, resource_id_mask);

    let stop = Arc::new(AtomicBool::new(false));
    let skip_grace = Arc::new(AtomicBool::new(false));
    let fetches: FetchMap = Arc::new(Mutex::new(HashMap::new()));
    let stop2 = Arc::clone(&stop);
    let skip2 = Arc::clone(&skip_grace);
    let fetches2 = Arc::clone(&fetches);

    let atoms = Atoms {
        clipboard,
        utf8_string,
        targets: targets_atom,
    };

    let handle = std::thread::spawn(move || {
        serve_loop(
            &conn,
            &atoms,
            &snapshot,
            &stop2,
            &skip2,
            &fetches2,
            resource_id_mask,
            own_client_base,
        );
        let _ = conn.destroy_window(window);
        let _ = conn.flush();
    });

    Some(ClipboardServeHandle {
        stop,
        skip_grace,
        fetches,
        resource_id_mask,
        own_client_base,
        handle: Some(handle),
    })
}

/// Hold the restored selection until another client takes ownership.
/// Runs only in the hidden, detached clipboard-hold subprocess.
pub fn x11_clipboard_hold(text: &str) -> Result<(), String> {
    x11_clipboard_hold_snapshot(&ClipboardSnapshot {
        targets: vec![ClipboardTarget {
            name: "UTF8_STRING".into(),
            property_type: "UTF8_STRING".into(),
            format: 8,
            bytes: text.as_bytes().to_vec(),
        }],
    })
}

pub fn x11_clipboard_hold_snapshot(snapshot: &ClipboardSnapshot) -> Result<(), String> {
    let (conn, screen_num) = x11rb::connect(None).map_err(|e| e.to_string())?;
    let root = conn.setup().roots[screen_num].root;
    let window = conn.generate_id().map_err(|e| e.to_string())?;
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
    .map_err(|e| e.to_string())?;
    let atoms = Atoms {
        clipboard: intern(&conn, b"CLIPBOARD").ok_or("could not intern CLIPBOARD")?,
        utf8_string: intern(&conn, b"UTF8_STRING").ok_or("could not intern UTF8_STRING")?,
        targets: intern(&conn, b"TARGETS").ok_or("could not intern TARGETS")?,
    };
    conn.set_selection_owner(window, atoms.clipboard, 0u32)
        .map_err(|e| e.to_string())?;
    conn.flush().map_err(|e| e.to_string())?;
    let owner = conn
        .get_selection_owner(atoms.clipboard)
        .map_err(|e| e.to_string())?
        .reply()
        .map_err(|e| e.to_string())?
        .owner;
    if owner != window {
        return Err("clipboard holder did not acquire ownership".to_string());
    }

    std::io::stdout()
        .write_all(&[1])
        .map_err(|e| e.to_string())?;
    std::io::stdout().flush().map_err(|e| e.to_string())?;

    let mask = conn.setup().resource_id_mask;
    let own_base = client_base(conn.setup().resource_id_base, mask);
    let fetches = Arc::new(Mutex::new(HashMap::new()));
    // Block on the X connection: the holder may live for hours until
    // another client takes the selection, so it must not busy-poll.
    loop {
        match conn.wait_for_event() {
            Ok(Event::SelectionRequest(req)) if req.selection == atoms.clipboard => {
                serve_request(&conn, &req, snapshot, &atoms, &fetches, mask, own_base);
            }
            Ok(Event::SelectionClear(ev)) if ev.selection == atoms.clipboard => break,
            Ok(_) => {}
            Err(e) => return Err(format!("clipboard holder X11 connection failed: {e}")),
        }
    }
    Ok(())
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
#[allow(clippy::too_many_arguments)] // Thin wrapper; splitting would
                                     // pessimise hot-path readability for no benefit.
fn serve_loop(
    conn: &RustConnection,
    atoms: &Atoms,
    snapshot: &ClipboardSnapshot,
    stop: &AtomicBool,
    skip_grace: &AtomicBool,
    fetches: &FetchMap,
    resource_id_mask: u32,
    own_client_base: u32,
) {
    let deadline = Instant::now() + DEFAULT_SERVE_DURATION;

    loop {
        if stop.load(Ordering::Relaxed) || Instant::now() > deadline {
            // Grace period: keep serving briefly so clipboard managers
            // can grab restored content — unless skip_grace is set.
            if !skip_grace.load(Ordering::Relaxed) {
                let grace_end = Instant::now() + GRACE_PERIOD;
                while Instant::now() < grace_end {
                    poll_and_serve(
                        conn,
                        atoms,
                        snapshot,
                        fetches,
                        resource_id_mask,
                        own_client_base,
                    );
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            break;
        }

        match conn.poll_for_event() {
            Ok(Some(Event::SelectionRequest(req))) if req.selection == atoms.clipboard => {
                serve_request(
                    conn,
                    &req,
                    snapshot,
                    atoms,
                    fetches,
                    resource_id_mask,
                    own_client_base,
                );
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
fn poll_and_serve(
    conn: &RustConnection,
    atoms: &Atoms,
    snapshot: &ClipboardSnapshot,
    fetches: &FetchMap,
    resource_id_mask: u32,
    own_client_base: u32,
) {
    if let Ok(Some(Event::SelectionRequest(req))) = conn.poll_for_event() {
        if req.selection == atoms.clipboard {
            serve_request(
                conn,
                &req,
                snapshot,
                atoms,
                fetches,
                resource_id_mask,
                own_client_base,
            );
        }
    }
}

/// Respond to a single `SelectionRequest`.
#[allow(clippy::too_many_arguments)] // Splitting the per-client-base
                                     // bookkeeping into a separate struct would just
                                     // shuffle the arguments around without removing
                                     // any.
fn serve_request(
    conn: &RustConnection,
    req: &SelectionRequestEvent,
    snapshot: &ClipboardSnapshot,
    atoms: &Atoms,
    fetches: &FetchMap,
    resource_id_mask: u32,
    own_client_base: u32,
) {
    // Per ICCCM, if property is None use target as fallback.
    let prop = if req.property == 0u32 {
        req.target
    } else {
        req.property
    };

    let ok = if req.target == atoms.targets {
        // Advertise supported targets.
        let supported: Vec<_> = std::iter::once(atoms.targets)
            .chain(
                snapshot
                    .targets
                    .iter()
                    .filter_map(|t| intern(conn, t.name.as_bytes())),
            )
            .collect();
        conn.change_property32(
            PropMode::REPLACE,
            req.requestor,
            prop,
            AtomEnum::ATOM,
            &supported,
        )
        .is_ok()
    } else if let Some(target) = snapshot
        .targets
        .iter()
        .find(|t| intern(conn, t.name.as_bytes()) == Some(req.target))
    {
        let kind = intern(conn, target.property_type.as_bytes()).unwrap_or(req.target);
        let wrote = match target.format {
            8 => conn
                .change_property8(PropMode::REPLACE, req.requestor, prop, kind, &target.bytes)
                .is_ok(),
            16 => conn
                .change_property16(
                    PropMode::REPLACE,
                    req.requestor,
                    prop,
                    kind,
                    &target
                        .bytes
                        .chunks_exact(2)
                        .map(|v| u16::from_ne_bytes([v[0], v[1]]))
                        .collect::<Vec<_>>(),
                )
                .is_ok(),
            32 => conn
                .change_property32(
                    PropMode::REPLACE,
                    req.requestor,
                    prop,
                    kind,
                    &target
                        .bytes
                        .chunks_exact(4)
                        .map(|v| u32::from_ne_bytes([v[0], v[1], v[2], v[3]]))
                        .collect::<Vec<_>>(),
                )
                .is_ok(),
            _ => false,
        };
        if wrote && req.target == atoms.utf8_string {
            // Record that a paste target actually fetched our text,
            // bucketed by the requestor's X11 client-base.  This is
            // the key paste-diagnostic signal: see
            // `ClipboardServeHandle::fetches_by_client` (used by the
            // deterministic per-chunk paste gate) and
            // `ClipboardServeHandle::served_count` (legacy blind-paste
            // fallback).  Self-directed requests (matching our own
            // client-base) are excluded defensively.
            let requestor_base = client_base(req.requestor, resource_id_mask);
            if requestor_base != own_client_base {
                if let Ok(mut map) = fetches.lock() {
                    let this_count = {
                        let entry = map.entry(requestor_base).or_insert(0);
                        *entry += 1;
                        *entry
                    };
                    let total: u32 = map.values().sum();
                    log::trace!(
                        "clipboard serve: answered UTF8_STRING request from requestor {} \
                         (client-base {:#x}, {} bytes, this-client count={}, total={})",
                        req.requestor,
                        requestor_base,
                        target.bytes.len(),
                        this_count,
                        total,
                    );
                }
            } else {
                log::trace!(
                    "clipboard serve: skipping count for self-directed UTF8_STRING request \
                     from requestor {} (own client-base {:#x})",
                    req.requestor,
                    own_client_base,
                );
            }
        }
        wrote
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
