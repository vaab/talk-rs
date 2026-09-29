#![cfg(feature = "ui")]

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    ConnectionExt as _, CreateWindowAux, EventMask, InputFocus, PropMode, SelectionNotifyEvent,
    WindowClass, SELECTION_NOTIFY_EVENT,
};
use x11rb::protocol::Event;
use x11rb::wrapper::ConnectionExt as _;

struct IsolatedXServer {
    _test_lock: std::sync::MutexGuard<'static, ()>,
    child: Child,
    display: String,
    lock_path: std::path::PathBuf,
}

impl IsolatedXServer {
    fn start() -> Self {
        static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let test_lock = TEST_LOCK.lock().expect("isolated display test lock");
        let number = (180..240)
            .find(|n| {
                !std::path::Path::new(&format!("/tmp/.X{n}-lock")).exists()
                    && !std::path::Path::new(&format!("/tmp/.X11-unix/X{n}")).exists()
            })
            .expect("free isolated display");
        let display = format!(":{number}");
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
            .or_else(|_| {
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
            .expect("Xvfb or Xephyr");
        let server = Self {
            _test_lock: test_lock,
            child,
            display,
            lock_path: format!("/tmp/.X{number}-lock").into(),
        };
        eprintln!(
            "isolated X server pid={} display={}",
            server.child.id(),
            server.display
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let owner = std::fs::read_to_string(&server.lock_path)
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok());
            if owner == Some(server.child.id()) && x11rb::connect(Some(&server.display)).is_ok() {
                return server;
            }
            assert!(
                Instant::now() < deadline,
                "isolated X server startup timed out"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for IsolatedXServer {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let pid = nix::unistd::Pid::from_raw(self.child.id() as i32);
            let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM);
        }
        let _ = self.child.wait();
    }
}

fn atom(conn: &x11rb::rust_connection::RustConnection, name: &[u8]) -> u32 {
    conn.intern_atom(false, name)
        .expect("intern request")
        .reply()
        .expect("intern reply")
        .atom
}

struct Owner {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Owner {
    fn start(display: &str, text: &'static str) -> Self {
        Self::start_targets(display, text.as_bytes(), b"UTF8_STRING")
    }

    fn start_targets(display: &str, bytes: &'static [u8], target_name: &'static [u8]) -> Self {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stopped = stop.clone();
        let display = display.to_owned();
        let thread = std::thread::spawn(move || {
            let (conn, screen) = x11rb::connect(Some(&display)).expect("owner connect");
            let window = conn.generate_id().expect("owner window id");
            conn.create_window(
                0,
                window,
                conn.setup().roots[screen].root,
                0,
                0,
                1,
                1,
                0,
                WindowClass::INPUT_ONLY,
                0,
                &CreateWindowAux::new(),
            )
            .expect("owner window");
            let clipboard = atom(&conn, b"CLIPBOARD");
            let targets = atom(&conn, b"TARGETS");
            let offered = atom(&conn, target_name);
            let html = atom(&conn, b"text/html");
            let offer_html = target_name == b"image/png";
            conn.set_selection_owner(window, clipboard, x11rb::CURRENT_TIME)
                .expect("claim clipboard");
            conn.flush().expect("flush claim");
            assert_eq!(
                conn.get_selection_owner(clipboard)
                    .expect("owner query")
                    .reply()
                    .expect("owner reply")
                    .owner,
                window
            );
            ready_tx.send(()).expect("report ownership");
            while !stopped.load(std::sync::atomic::Ordering::Relaxed) {
                match conn.poll_for_event().expect("owner event") {
                    Some(Event::SelectionRequest(req)) if req.selection == clipboard => {
                        let property = if req.property == 0 {
                            req.target
                        } else {
                            req.property
                        };
                        let ok = if req.target == html && offer_html {
                            conn.change_property8(
                                PropMode::REPLACE,
                                req.requestor,
                                property,
                                html,
                                b"<b>original</b>",
                            )
                            .is_ok()
                        } else if req.target == offered {
                            conn.change_property8(
                                PropMode::REPLACE,
                                req.requestor,
                                property,
                                offered,
                                bytes,
                            )
                            .is_ok()
                        } else if req.target == targets {
                            let supported = if offer_html {
                                vec![targets, offered, html]
                            } else {
                                vec![targets, offered]
                            };
                            conn.change_property32(
                                PropMode::REPLACE,
                                req.requestor,
                                property,
                                x11rb::protocol::xproto::AtomEnum::ATOM,
                                &supported,
                            )
                            .is_ok()
                        } else {
                            false
                        };
                        let notify = SelectionNotifyEvent {
                            response_type: SELECTION_NOTIFY_EVENT,
                            sequence: 0,
                            time: req.time,
                            requestor: req.requestor,
                            selection: clipboard,
                            target: req.target,
                            property: if ok { property } else { 0 },
                        };
                        conn.send_event(false, req.requestor, EventMask::NO_EVENT, notify)
                            .expect("notify requestor");
                        conn.flush().expect("flush reply");
                    }
                    Some(Event::SelectionClear(ev)) if ev.selection == clipboard => break,
                    _ => std::thread::sleep(Duration::from_millis(2)),
                }
            }
        });
        ready_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("owner ready");
        Self {
            stop,
            thread: Some(thread),
        }
    }
}

fn read_target(display: &str, target: &str) -> Vec<u8> {
    let output = Command::new("xclip")
        .args(["-o", "-selection", "clipboard", "-t", target])
        .env("DISPLAY", display)
        .env_remove("WAYLAND_DISPLAY")
        .output()
        .expect("read isolated clipboard target");
    assert!(
        output.status.success(),
        "target {target}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("owner thread");
        }
    }
}

fn read_clipboard(display: &str) -> String {
    let output = Command::new("xclip")
        .args(["-o", "-selection", "clipboard", "-t", "UTF8_STRING"])
        .env("DISPLAY", display)
        .env_remove("WAYLAND_DISPLAY")
        .output()
        .expect("read clipboard with xclip");
    assert!(
        output.status.success(),
        "clipboard read failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("clipboard UTF-8")
}

fn holder_pids(display: &str) -> Vec<i32> {
    let binary = env!("CARGO_BIN_EXE_talk-rs");
    std::fs::read_dir("/proc")
        .expect("proc directory")
        .flatten()
        .filter_map(|entry| {
            let pid = entry.file_name().to_string_lossy().parse::<i32>().ok()?;
            let cmd = std::fs::read(entry.path().join("cmdline")).ok()?;
            let argv: Vec<_> = cmd.split(|b| *b == 0).collect();
            if argv.first().copied() != Some(binary.as_bytes())
                || argv.get(1).copied() != Some(b"clipboard-hold".as_slice())
            {
                return None;
            }
            let env = std::fs::read(entry.path().join("environ")).ok()?;
            let expected = format!("DISPLAY={display}");
            env.split(|b| *b == 0)
                .any(|v| v == expected.as_bytes())
                .then_some(pid)
        })
        .collect()
}

#[test]
#[ignore = "requires Xvfb or Xephyr; run with --ignored"]
fn original_clipboard_survives_dictate_exit_then_holder_exits_on_new_owner() {
    let server = IsolatedXServer::start();
    let temp = tempfile::tempdir().expect("temporary home");
    let config_dir = temp.path().join("config/talk-rs");
    std::fs::create_dir_all(&config_dir).expect("config directory");
    std::fs::write(
        config_dir.join("config.yaml"),
        format!(
            "output_dir: {}\nproviders:\n  mistral:\n    api_key: dummy\n",
            temp.path().display()
        ),
    )
    .expect("config");
    let audio = temp.path().join("x.ogg");
    std::fs::write(&audio, b"test input").expect("audio path");
    talk_rs::recording_cache::write_pick(&audio, "mistral", "voxtral-mini-2602", false, "DICTATED")
        .expect("pick");
    let original = Owner::start(&server.display, "USER-ORIGINAL");
    assert_eq!(read_clipboard(&server.display), "USER-ORIGINAL");

    // A focused X11 client consumes the offered chunk when it receives the paste key.
    let (target, screen) = x11rb::connect(Some(&server.display)).expect("target connection");
    let window = target.generate_id().expect("target window id");
    target
        .create_window(
            0,
            window,
            target.setup().roots[screen].root,
            0,
            0,
            100,
            100,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new().event_mask(EventMask::KEY_PRESS),
        )
        .expect("create target");
    target.map_window(window).expect("map target");
    target
        .set_input_focus(InputFocus::PARENT, window, x11rb::CURRENT_TIME)
        .expect("focus target");
    target.flush().expect("flush target");
    let display = server.display.clone();
    let fetcher = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(8);
        while Instant::now() < deadline {
            if let Some(Event::KeyPress(_)) = target.poll_for_event().expect("target events") {
                return read_clipboard(&display);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("paste key did not reach target");
    });

    let mut dictate = Command::new(env!("CARGO_BIN_EXE_talk-rs"))
        .args([
            "dictate",
            "--no-sounds",
            "--no-overlay",
            "--no-bt-auto-switch",
            "--input-audio-file",
        ])
        .arg(&audio)
        .env("DISPLAY", &server.display)
        .env_remove("WAYLAND_DISPLAY")
        .env("HOME", temp.path())
        .env("XDG_CONFIG_HOME", temp.path().join("config"))
        .env("XDG_CACHE_HOME", temp.path().join("cache"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start dictate");
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = dictate.try_wait().expect("poll dictate") {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = dictate.kill();
            let _ = dictate.wait();
            panic!("dictate timed out");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(status.success(), "dictate failed: {status}");
    assert_eq!(fetcher.join().expect("fetcher"), "DICTATED");
    drop(original);
    assert_eq!(
        read_clipboard(&server.display),
        "USER-ORIGINAL",
        "original clipboard lost after dictate exited"
    );
    std::thread::sleep(Duration::from_millis(1100));
    assert_eq!(
        read_clipboard(&server.display),
        "USER-ORIGINAL",
        "clipboard did not survive one second"
    );

    let replacement = Owner::start(&server.display, "NEW");
    assert_eq!(read_clipboard(&server.display), "NEW");
    let deadline = Instant::now() + Duration::from_secs(2);
    while !holder_pids(&server.display).is_empty() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        holder_pids(&server.display).is_empty(),
        "clipboard holder remained after SelectionClear"
    );
    drop(replacement);
}

#[test]
#[ignore = "requires Xvfb or Xephyr; run with --ignored"]
fn image_target_survives_production_paste_and_restore() {
    let server = IsolatedXServer::start();
    let temp = tempfile::tempdir().expect("temporary home");
    let config_dir = temp.path().join("config/talk-rs");
    std::fs::create_dir_all(&config_dir).expect("config directory");
    std::fs::write(
        config_dir.join("config.yaml"),
        format!(
            "output_dir: {}\nproviders:\n  mistral:\n    api_key: dummy\n",
            temp.path().display()
        ),
    )
    .expect("config");
    let audio = temp.path().join("x.ogg");
    std::fs::write(&audio, b"test input").expect("audio path");
    talk_rs::recording_cache::write_pick(&audio, "mistral", "voxtral-mini-2602", false, "DICTATED")
        .expect("pick");
    let original = Owner::start_targets(&server.display, b"\x89PNG\0binary", b"image/png");
    assert_eq!(
        read_target(&server.display, "image/png"),
        b"\x89PNG\0binary"
    );
    assert_eq!(
        read_target(&server.display, "text/html"),
        b"<b>original</b>"
    );

    let (target, screen) = x11rb::connect(Some(&server.display)).expect("target connection");
    let window = target.generate_id().expect("target window id");
    target
        .create_window(
            0,
            window,
            target.setup().roots[screen].root,
            0,
            0,
            100,
            100,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new().event_mask(EventMask::KEY_PRESS),
        )
        .expect("create target");
    target.map_window(window).expect("map target");
    target
        .set_input_focus(InputFocus::PARENT, window, x11rb::CURRENT_TIME)
        .expect("focus target");
    target.flush().expect("flush target");
    let display = server.display.clone();
    let fetcher = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(8);
        while Instant::now() < deadline {
            if let Some(Event::KeyPress(_)) = target.poll_for_event().expect("target events") {
                return read_target(&display, "UTF8_STRING");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("paste key did not reach target");
    });
    let status = Command::new(env!("CARGO_BIN_EXE_talk-rs"))
        .args([
            "dictate",
            "--no-sounds",
            "--no-overlay",
            "--no-bt-auto-switch",
            "--input-audio-file",
        ])
        .arg(&audio)
        .env("DISPLAY", &server.display)
        .env_remove("WAYLAND_DISPLAY")
        .env("HOME", temp.path())
        .env("XDG_CONFIG_HOME", temp.path().join("config"))
        .env("XDG_CACHE_HOME", temp.path().join("cache"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("dictate result");
    assert!(status.success(), "dictate failed: {status}");
    assert_eq!(fetcher.join().expect("fetcher"), b"DICTATED");
    drop(original);
    let targets = read_target(&server.display, "TARGETS");
    assert!(
        String::from_utf8_lossy(&targets).contains("image/png"),
        "TARGETS missing image/png"
    );
    assert!(String::from_utf8_lossy(&targets).contains("text/html"));
    assert!(!String::from_utf8_lossy(&targets).contains("UTF8_STRING"));
    assert_eq!(
        read_target(&server.display, "image/png"),
        b"\x89PNG\0binary"
    );
    assert_eq!(
        read_target(&server.display, "text/html"),
        b"<b>original</b>"
    );
    let replacement = Owner::start(&server.display, "NEW");
    let deadline = Instant::now() + Duration::from_secs(2);
    while !holder_pids(&server.display).is_empty() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        holder_pids(&server.display).is_empty(),
        "clipboard holder remained after SelectionClear"
    );
    drop(replacement);
}
