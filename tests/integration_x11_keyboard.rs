#![cfg(feature = "ui")]

//! Opt-in real-X11 keyboard integration test.
//!
//! Requires `Xvfb` or `Xephyr` and is intentionally ignored by plain
//! `cargo test`. Run explicitly with:
//! `cargo test --test integration_x11_keyboard -- --ignored --nocapture`

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    ConnectionExt as _, CreateWindowAux, EventMask, InputFocus, WindowClass,
};
use x11rb::protocol::Event;

const CTRL_L: u32 = 0xffe3;
const CTRL_R: u32 = 0xffe4;
const SHIFT_L: u32 = 0xffe1;
const SHIFT_R: u32 = 0xffe2;
const ALT_L: u32 = 0xffe9;
const ALT_R: u32 = 0xffea;
const SUPER_L: u32 = 0xffeb;
const SUPER_R: u32 = 0xffec;
const CAPS_LOCK: u32 = 0xffe5;
const NUM_LOCK: u32 = 0xff7f;
const V: u32 = 0x0076;
const WORKER_ENV: &str = "TALK_RS_ISOLATED_X11_KEYBOARD_WORKER";

struct IsolatedXServer {
    child: Child,
    display: String,
    lock_path: std::path::PathBuf,
    stopped: bool,
}

impl IsolatedXServer {
    fn start() -> Self {
        let display_number = (120..180)
            .find(|number| {
                !std::path::Path::new(&format!("/tmp/.X{number}-lock")).exists()
                    && !std::path::Path::new(&format!("/tmp/.X11-unix/X{number}")).exists()
            })
            .expect("no free isolated X11 display number");
        let display = format!(":{display_number}");
        let lock_path = std::path::PathBuf::from(format!("/tmp/.X{display_number}-lock"));

        let child = Command::new("Xvfb")
            .args([
                display.as_str(),
                "-screen",
                "0",
                "800x600x24",
                "-nolisten",
                "tcp",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .or_else(|xvfb_error| {
                eprintln!("Xvfb unavailable ({xvfb_error}); using isolated Xephyr fallback");
                Command::new("Xephyr")
                    .args([
                        "-screen",
                        "800x600",
                        "-ac",
                        "-nolisten",
                        "tcp",
                        display.as_str(),
                    ])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
            })
            .expect("neither Xvfb nor Xephyr could start");

        let mut server = Self {
            child,
            display,
            lock_path,
            stopped: false,
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = server.child.try_wait().expect("query isolated X server") {
                panic!("isolated X server exited during startup: {status}");
            }
            let lock_owner = std::fs::read_to_string(&server.lock_path)
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok());
            if let Some(owner) = lock_owner {
                assert_eq!(
                    owner,
                    server.child.id(),
                    "selected display lock belongs to another X server"
                );
            }
            if lock_owner == Some(server.child.id())
                && x11rb::connect(Some(&server.display)).is_ok()
            {
                assert!(
                    server
                        .child
                        .try_wait()
                        .expect("confirm isolated X server ownership")
                        .is_none(),
                    "selected display belonged to another server"
                );
                return server;
            }
            assert!(
                Instant::now() < deadline,
                "isolated X server startup timed out"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn stop(&mut self) -> Result<(), String> {
        if self.stopped {
            return Ok(());
        }
        if self.child.try_wait().ok().flatten().is_some() {
            self.stopped = true;
            return Ok(());
        }
        let pid = nix::unistd::Pid::from_raw(self.child.id() as i32);
        nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM)
            .map_err(|error| format!("signal isolated X server {pid}: {error}"))?;
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if self
                .child
                .try_wait()
                .map_err(|error| format!("wait for isolated X server {pid}: {error}"))?
                .is_some()
            {
                self.stopped = true;
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        self.child
            .kill()
            .map_err(|error| format!("kill isolated X server {pid}: {error}"))?;
        self.child
            .wait()
            .map_err(|error| format!("reap isolated X server {pid}: {error}"))?;
        self.stopped = true;
        Ok(())
    }
}

impl Drop for IsolatedXServer {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            eprintln!("isolated X server teardown failed: {error}");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObservedKeyEvent {
    Press { keycode: u8, state: u16 },
    Release { keycode: u8, state: u16 },
}

struct Observer {
    conn: x11rb::rust_connection::RustConnection,
    root: u32,
}

impl Observer {
    fn new() -> Self {
        let (conn, screen_number) = x11rb::connect(None).expect("connect isolated display");
        let screen = &conn.setup().roots[screen_number];
        let root = screen.root;
        let window = conn.generate_id().expect("allocate observer window id");
        conn.create_window(
            x11rb::COPY_DEPTH_FROM_PARENT,
            window,
            root,
            0,
            0,
            100,
            100,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new().event_mask(EventMask::KEY_PRESS | EventMask::KEY_RELEASE),
        )
        .expect("create observer window")
        .check()
        .expect("server accepted observer window");
        conn.map_window(window)
            .expect("map observer window")
            .check()
            .expect("server mapped observer window");
        conn.set_input_focus(InputFocus::PARENT, window, x11rb::CURRENT_TIME)
            .expect("focus observer window")
            .check()
            .expect("server focused observer window");
        conn.get_input_focus()
            .expect("focus synchronization request")
            .reply()
            .expect("focus synchronization reply");
        let mut observer = Self { conn, root };
        observer.drain_events();
        observer
    }

    fn keycode(&self, keysym: u32) -> u8 {
        let setup = self.conn.setup();
        let count = setup.max_keycode - setup.min_keycode + 1;
        let mapping = self
            .conn
            .get_keyboard_mapping(setup.min_keycode, count)
            .expect("request keyboard mapping")
            .reply()
            .expect("read keyboard mapping");
        let width = mapping.keysyms_per_keycode as usize;
        mapping
            .keysyms
            .chunks_exact(width)
            .position(|symbols| symbols.contains(&keysym))
            .map(|index| setup.min_keycode + index as u8)
            .unwrap_or_else(|| panic!("keysym {keysym:#x} missing from isolated keymap"))
    }

    fn inject_checked(&self, event_type: u8, keycode: u8) {
        x11rb::protocol::xtest::fake_input(
            &self.conn,
            event_type,
            keycode,
            x11rb::CURRENT_TIME,
            0u32,
            0,
            0,
            0,
        )
        .expect("serialize fixture XTest event")
        .check()
        .expect("server accepted fixture XTest event");
        self.round_trip();
    }

    fn round_trip(&self) {
        self.conn
            .get_input_focus()
            .expect("synchronization request")
            .reply()
            .expect("synchronization reply");
    }

    fn is_pressed(&self, keycode: u8) -> bool {
        let keys = self
            .conn
            .query_keymap()
            .expect("query keymap")
            .reply()
            .expect("query keymap reply")
            .keys;
        keys[(keycode / 8) as usize] & (1 << (keycode % 8)) != 0
    }

    fn modifier_mask(&self) -> u16 {
        self.conn
            .query_pointer(self.root)
            .expect("query pointer modifier state")
            .reply()
            .expect("query pointer modifier state reply")
            .mask
            .into()
    }

    fn drain_events(&mut self) {
        while self
            .conn
            .poll_for_event()
            .expect("poll X11 event")
            .is_some()
        {}
    }

    fn events(&mut self, expected: usize) -> Vec<ObservedKeyEvent> {
        self.round_trip();
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut events = Vec::new();
        while events.len() < expected && Instant::now() < deadline {
            match self.conn.poll_for_event().expect("poll X11 event") {
                Some(Event::KeyPress(event)) => {
                    events.push(ObservedKeyEvent::Press {
                        keycode: event.detail,
                        state: event.state.into(),
                    });
                }
                Some(Event::KeyRelease(event)) => {
                    events.push(ObservedKeyEvent::Release {
                        keycode: event.detail,
                        state: event.state.into(),
                    });
                }
                Some(_) => {}
                None => std::thread::sleep(Duration::from_millis(1)),
            }
        }
        assert_eq!(events.len(), expected, "missing XTest key events");
        events
    }

    fn available_events(&mut self) -> Vec<ObservedKeyEvent> {
        self.round_trip();
        let mut events = Vec::new();
        loop {
            match self.conn.poll_for_event().expect("poll X11 event") {
                Some(Event::KeyPress(event)) => {
                    events.push(ObservedKeyEvent::Press {
                        keycode: event.detail,
                        state: event.state.into(),
                    });
                }
                Some(Event::KeyRelease(event)) => {
                    events.push(ObservedKeyEvent::Release {
                        keycode: event.detail,
                        state: event.state.into(),
                    });
                }
                Some(_) => {}
                None => return events,
            }
        }
    }
}

fn key_sequence(events: &[ObservedKeyEvent]) -> Vec<(bool, u8)> {
    events
        .iter()
        .map(|event| match event {
            ObservedKeyEvent::Press { keycode, .. } => (true, *keycode),
            ObservedKeyEvent::Release { keycode, .. } => (false, *keycode),
        })
        .collect()
}

fn expected_sequence(keycodes: &[u8]) -> Vec<(bool, u8)> {
    keycodes
        .iter()
        .copied()
        .map(|keycode| (true, keycode))
        .chain(
            keycodes
                .iter()
                .rev()
                .copied()
                .map(|keycode| (false, keycode)),
        )
        .collect()
}

fn press_state(events: &[ObservedKeyEvent], keycode: u8) -> u16 {
    events
        .iter()
        .find_map(|event| match event {
            ObservedKeyEvent::Press {
                keycode: observed,
                state,
            } if *observed == keycode => Some(*state),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing KeyPress for keycode {keycode}"))
}

fn run_isolated_contract() {
    let mut observer = Observer::new();

    for (keysyms, expected_mask) in [
        (&[CTRL_L, V][..], 1 << 2),
        (&[CTRL_L, SHIFT_L, V][..], (1 << 2) | (1 << 0)),
    ] {
        let keycodes: Vec<u8> = keysyms
            .iter()
            .map(|keysym| observer.keycode(*keysym))
            .collect();
        talk_rs::x11::x11_send_key_combo_checked(keysyms).expect("send isolated shortcut");
        let events = observer.events(keycodes.len() * 2);
        assert_eq!(key_sequence(&events), expected_sequence(&keycodes));
        let v_state = press_state(&events, observer.keycode(V));
        assert_eq!(v_state & ((1 << 2) | (1 << 0)), expected_mask);
        for keycode in keycodes {
            assert!(
                !observer.is_pressed(keycode),
                "shortcut key remained pressed"
            );
        }
    }

    for keysym in [
        CTRL_L, CTRL_R, SHIFT_L, SHIFT_R, ALT_L, ALT_R, SUPER_L, SUPER_R, V,
    ] {
        let held = observer.keycode(keysym);
        let v = observer.keycode(V);
        observer.inject_checked(2, held);
        observer.drain_events();
        let started = Instant::now();
        let error = talk_rs::x11::x11_send_key_combo_checked(&[CTRL_L, V])
            .expect_err("pre-held modifier must abort shortcut injection");
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(error.to_string().contains("pre-held"));
        assert!(
            observer.is_pressed(held),
            "talk-rs released a user-held modifier"
        );
        let events = observer.available_events();
        assert!(
            !events.iter().any(|event| matches!(
                event,
                ObservedKeyEvent::Press { keycode, .. } if *keycode == v
            )),
            "shortcut V was injected while a modifier was held"
        );
        observer.inject_checked(3, held);
        observer.drain_events();
    }

    for lock_keysym in [CAPS_LOCK, NUM_LOCK] {
        let held = observer.keycode(lock_keysym);
        let v = observer.keycode(V);
        observer.inject_checked(2, held);
        observer.drain_events();
        talk_rs::x11::x11_send_key_combo_checked(&[CTRL_L, V])
            .expect_err("physically held lock key must block injection");
        assert!(
            observer.is_pressed(held),
            "talk-rs released a held lock key"
        );
        assert!(
            !observer.available_events().iter().any(|event| matches!(
                event,
                ObservedKeyEvent::Press { keycode, .. } if *keycode == v
            )),
            "shortcut V was injected while a lock key was physically held"
        );
        observer.inject_checked(3, held);
        observer.inject_checked(2, held);
        observer.inject_checked(3, held);
        observer.drain_events();
    }

    for lock_keysym in [CAPS_LOCK, NUM_LOCK] {
        let lock = observer.keycode(lock_keysym);
        observer.inject_checked(2, lock);
        observer.inject_checked(3, lock);
        let locked_mask = observer.modifier_mask();
        observer.drain_events();
        talk_rs::x11::x11_send_key_combo_checked(&[CTRL_L, V]).expect("shortcut with lock active");
        let events = observer.events(4);
        assert_eq!(
            press_state(&events, observer.keycode(V)) & locked_mask,
            locked_mask
        );
        assert_eq!(observer.modifier_mask(), locked_mask, "lock state changed");
        observer.inject_checked(2, lock);
        observer.inject_checked(3, lock);
        observer.drain_events();
    }

    let repeated_keysyms = [CTRL_L, V];
    let repeated_keycodes = [observer.keycode(CTRL_L), observer.keycode(V)];
    let repeats = 12;
    for _ in 0..repeats {
        talk_rs::x11::x11_send_key_combo_checked(&repeated_keysyms).expect("repeat shortcut");
    }
    let expected: Vec<_> = (0..repeats)
        .flat_map(|_| expected_sequence(&repeated_keycodes))
        .collect();
    assert_eq!(key_sequence(&observer.events(expected.len())), expected);
}

#[test]
#[ignore = "requires Xvfb or Xephyr; run with --ignored --nocapture"]
fn isolated_x11_shortcuts_preserve_keyboard_state() {
    if std::env::var_os(WORKER_ENV).is_some() {
        run_isolated_contract();
        return;
    }

    let mut server = IsolatedXServer::start();
    eprintln!(
        "isolated X server pid={} display={}",
        server.child.id(),
        server.display
    );
    let status = Command::new(std::env::current_exe().expect("current test executable"))
        .args([
            "--ignored",
            "--exact",
            "isolated_x11_shortcuts_preserve_keyboard_state",
            "--nocapture",
        ])
        .env(WORKER_ENV, "1")
        .env("DISPLAY", &server.display)
        .env_remove("WAYLAND_DISPLAY")
        .status()
        .expect("launch isolated X11 test worker");
    assert!(status.success(), "isolated X11 worker failed: {status}");
    server.stop().expect("stop and reap isolated X server");
    assert!(
        server
            .child
            .try_wait()
            .expect("query isolated X server after teardown")
            .is_some(),
        "isolated X server child remained running after fixture teardown"
    );
}
