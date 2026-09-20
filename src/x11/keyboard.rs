use std::time::Duration;

const KEY_PRESS: u8 = 2;
const KEY_RELEASE: u8 = 3;
const CONFLICT_TIMEOUT: Duration = Duration::from_millis(250);
const CONFLICT_POLL_INTERVAL: Duration = Duration::from_millis(5);
const LOCK_KEYSYMS: &[u32] = &[
    0xffe5, // Caps_Lock
    0xff7f, // Num_Lock
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InjectedKeyEvent {
    Press,
    Release,
}

pub(super) trait KeyboardBackend {
    fn resolve_keysyms(&self, keysyms: &[u32]) -> Result<Vec<u8>, String>;
    fn conflict_keycodes(&self, requested: &[u8]) -> Vec<u8>;
    fn pressed_keycodes(&mut self) -> Result<Vec<u8>, String>;
    fn conflicting_modifier_mask(&self) -> u16;
    fn active_modifier_mask(&mut self) -> Result<u16, String>;
    fn inject(&mut self, event: InjectedKeyEvent, keycode: u8) -> Result<(), String>;
    fn synchronize(&mut self) -> Result<(), String>;
}

#[derive(Debug, PartialEq, Eq)]
pub struct KeyComboError {
    message: String,
}

#[derive(Debug, PartialEq, Eq)]
struct ModifierPolicy {
    physical_keycodes: Vec<u8>,
    conflicting_mask: u16,
}

impl std::fmt::Display for KeyComboError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for KeyComboError {}

pub(super) fn send_key_combo_with_backend<B: KeyboardBackend>(
    backend: &mut B,
    keysyms: &[u32],
    conflict_timeout: Duration,
    poll_interval: Duration,
) -> Result<(), KeyComboError> {
    if keysyms.is_empty() {
        return Ok(());
    }

    let keycodes = backend
        .resolve_keysyms(keysyms)
        .map_err(|error| KeyComboError::new(format!("keyboard mapping phase failed: {error}")))?;
    wait_for_conflicts_to_clear(backend, &keycodes, conflict_timeout, poll_interval)?;

    inject_keycodes(backend, &keycodes)
}

pub(super) fn send_key_repeat_with_backend<B: KeyboardBackend>(
    backend: &mut B,
    keysym: u32,
    count: usize,
    conflict_timeout: Duration,
    poll_interval: Duration,
) -> Result<(), KeyComboError> {
    if count == 0 {
        return Ok(());
    }

    let keycodes = backend
        .resolve_keysyms(&[keysym])
        .map_err(|error| KeyComboError::new(format!("keyboard mapping phase failed: {error}")))?;
    wait_for_conflicts_to_clear(backend, &keycodes, conflict_timeout, poll_interval)?;
    for repeat_index in 0..count {
        inject_keycodes(backend, &keycodes).map_err(|error| {
            KeyComboError::new(format!(
                "repeat {}/{} failed: {}",
                repeat_index + 1,
                count,
                error
            ))
        })?;
    }
    Ok(())
}

fn inject_keycodes<B: KeyboardBackend>(
    backend: &mut B,
    keycodes: &[u8],
) -> Result<(), KeyComboError> {
    let mut possibly_pressed = Vec::with_capacity(keycodes.len());
    for &keycode in keycodes {
        possibly_pressed.push(keycode);
        if let Err(error) = backend.inject(InjectedKeyEvent::Press, keycode) {
            let cleanup = release_all(backend, &possibly_pressed);
            return Err(KeyComboError::new(format_injection_error(
                "press", keycode, &error, &cleanup,
            )));
        }
    }

    let release_errors = release_all(backend, &possibly_pressed);
    if !release_errors.is_empty() {
        let cleanup_errors = release_all(backend, &possibly_pressed);
        let cleanup_detail = if cleanup_errors.is_empty() {
            "cleanup release requests succeeded".to_string()
        } else {
            format!("cleanup errors: {}", cleanup_errors.join("; "))
        };
        return Err(KeyComboError::new(format!(
            "XTest release phase failed after attempting every injected key: {}; {}",
            release_errors.join("; "),
            cleanup_detail,
        )));
    }

    Ok(())
}

pub(super) fn send_key_combo(keysyms: &[u32]) -> Result<(), KeyComboError> {
    if keysyms.is_empty() {
        return Ok(());
    }
    let mut backend = X11KeyboardBackend::connect()?;
    send_key_combo_with_backend(
        &mut backend,
        keysyms,
        CONFLICT_TIMEOUT,
        CONFLICT_POLL_INTERVAL,
    )
}

pub(super) fn send_key_repeat(keysym: u32, count: usize) -> Result<(), KeyComboError> {
    if count == 0 {
        return Ok(());
    }
    let mut backend = X11KeyboardBackend::connect()?;
    send_key_repeat_with_backend(
        &mut backend,
        keysym,
        count,
        CONFLICT_TIMEOUT,
        CONFLICT_POLL_INTERVAL,
    )
}

impl KeyComboError {
    fn new(message: String) -> Self {
        Self { message }
    }
}

fn wait_for_conflicts_to_clear<B: KeyboardBackend>(
    backend: &mut B,
    requested: &[u8],
    timeout: Duration,
    poll_interval: Duration,
) -> Result<(), KeyComboError> {
    let conflict_keycodes = backend.conflict_keycodes(requested);
    let conflicting_modifier_mask = backend.conflicting_modifier_mask();
    let started = std::time::Instant::now();

    loop {
        let pressed = backend.pressed_keycodes().map_err(|error| {
            KeyComboError::new(format!("keyboard-state query phase failed: {error}"))
        })?;
        let held: Vec<u8> = pressed
            .into_iter()
            .filter(|keycode| conflict_keycodes.contains(keycode))
            .collect();
        let active_modifier_mask = backend.active_modifier_mask().map_err(|error| {
            KeyComboError::new(format!("modifier-state query phase failed: {error}"))
        })? & conflicting_modifier_mask;

        if held.is_empty() && active_modifier_mask == 0 {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            return Err(KeyComboError::new(format!(
                "shortcut injection aborted after {} ms: pre-held keycodes {:?}, effective modifier mask {:#04x}",
                timeout.as_millis(),
                held,
                active_modifier_mask,
            )));
        }
        std::thread::sleep(poll_interval);
    }
}

fn release_all<B: KeyboardBackend>(backend: &mut B, pressed: &[u8]) -> Vec<String> {
    let mut errors = Vec::new();
    for &keycode in pressed.iter().rev() {
        if let Err(error) = backend.inject(InjectedKeyEvent::Release, keycode) {
            errors.push(format!("keycode {keycode}: {error}"));
        }
    }
    if let Err(error) = backend.synchronize() {
        errors.push(format!("cleanup synchronization: {error}"));
    }
    errors
}

fn format_injection_error(
    phase: &str,
    keycode: u8,
    source: &str,
    cleanup_errors: &[String],
) -> String {
    if cleanup_errors.is_empty() {
        format!(
            "XTest {phase} phase failed for keycode {keycode}: {source}; \
             cleanup release requests were accepted"
        )
    } else {
        format!(
            "XTest {phase} phase failed for keycode {keycode}: {source}; cleanup errors: {}",
            cleanup_errors.join("; ")
        )
    }
}

struct X11KeyboardBackend {
    conn: x11rb::rust_connection::RustConnection,
    root: u32,
    min_keycode: u8,
    keysyms_per_keycode: usize,
    keysyms: Vec<u32>,
    modifier_policy: ModifierPolicy,
}

impl X11KeyboardBackend {
    fn connect() -> Result<Self, KeyComboError> {
        use x11rb::connection::Connection;
        use x11rb::protocol::xproto::ConnectionExt as _;

        let (conn, screen_number) = x11rb::connect(None)
            .map_err(|error| KeyComboError::new(format!("X11 connection phase failed: {error}")))?;
        let setup = conn.setup();
        let min_keycode = setup.min_keycode;
        let count = setup.max_keycode - min_keycode + 1;
        let root = setup.roots[screen_number].root;
        let mapping = conn
            .get_keyboard_mapping(min_keycode, count)
            .map_err(|error| {
                KeyComboError::new(format!("keyboard mapping request failed: {error}"))
            })?
            .reply()
            .map_err(|error| {
                KeyComboError::new(format!("keyboard mapping reply failed: {error}"))
            })?;
        let modifier_mapping = conn
            .get_modifier_mapping()
            .map_err(|error| {
                KeyComboError::new(format!("modifier mapping request failed: {error}"))
            })?
            .reply()
            .map_err(|error| {
                KeyComboError::new(format!("modifier mapping reply failed: {error}"))
            })?;
        let keysyms_per_keycode = mapping.keysyms_per_keycode as usize;
        if keysyms_per_keycode == 0 {
            return Err(KeyComboError::new(
                "keyboard mapping reply contained zero keysyms per keycode".to_string(),
            ));
        }
        let modifier_policy = derive_modifier_policy(
            min_keycode,
            keysyms_per_keycode,
            &mapping.keysyms,
            modifier_mapping.keycodes_per_modifier() as usize,
            &modifier_mapping.keycodes,
        );

        Ok(Self {
            conn,
            root,
            min_keycode,
            keysyms_per_keycode,
            keysyms: mapping.keysyms,
            modifier_policy,
        })
    }
}

impl KeyboardBackend for X11KeyboardBackend {
    fn resolve_keysyms(&self, keysyms: &[u32]) -> Result<Vec<u8>, String> {
        keysyms
            .iter()
            .map(|keysym| {
                self.keysyms
                    .chunks_exact(self.keysyms_per_keycode)
                    .position(|symbols| symbols.contains(keysym))
                    .map(|index| self.min_keycode + index as u8)
                    .ok_or_else(|| {
                        format!("keysym {keysym:#x} is not present in the server keymap")
                    })
            })
            .collect()
    }

    fn conflict_keycodes(&self, requested: &[u8]) -> Vec<u8> {
        let mut conflicts = requested.to_vec();
        conflicts.extend_from_slice(&self.modifier_policy.physical_keycodes);
        conflicts.sort_unstable();
        conflicts.dedup();
        conflicts
    }

    fn pressed_keycodes(&mut self) -> Result<Vec<u8>, String> {
        use x11rb::protocol::xproto::ConnectionExt as _;

        let keys = self
            .conn
            .query_keymap()
            .map_err(|error| error.to_string())?
            .reply()
            .map_err(|error| error.to_string())?
            .keys;
        Ok((0u16..=255)
            .filter_map(|keycode| {
                let keycode = keycode as u8;
                (keys[(keycode / 8) as usize] & (1 << (keycode % 8)) != 0).then_some(keycode)
            })
            .collect())
    }

    fn conflicting_modifier_mask(&self) -> u16 {
        self.modifier_policy.conflicting_mask
    }

    fn active_modifier_mask(&mut self) -> Result<u16, String> {
        use x11rb::protocol::xproto::ConnectionExt as _;

        self.conn
            .query_pointer(self.root)
            .map_err(|error| error.to_string())?
            .reply()
            .map(|reply| reply.mask.into())
            .map_err(|error| error.to_string())
    }

    fn inject(&mut self, event: InjectedKeyEvent, keycode: u8) -> Result<(), String> {
        use x11rb::protocol::xtest;

        let event_type = match event {
            InjectedKeyEvent::Press => KEY_PRESS,
            InjectedKeyEvent::Release => KEY_RELEASE,
        };
        xtest::fake_input(
            &self.conn,
            event_type,
            keycode,
            x11rb::CURRENT_TIME,
            0u32,
            0,
            0,
            0,
        )
        .map_err(|error| error.to_string())?
        .check()
        .map_err(|error| error.to_string())
    }

    fn synchronize(&mut self) -> Result<(), String> {
        use x11rb::wrapper::ConnectionExt as _;

        self.conn.sync().map_err(|error| error.to_string())
    }
}

fn derive_modifier_policy(
    min_keycode: u8,
    keysyms_per_keycode: usize,
    keysyms: &[u32],
    keycodes_per_modifier: usize,
    modifier_keycodes: &[u8],
) -> ModifierPolicy {
    if keycodes_per_modifier == 0 {
        return ModifierPolicy {
            physical_keycodes: Vec::new(),
            conflicting_mask: 0,
        };
    }
    let mut physical_keycodes = Vec::new();
    let mut conflicting_mask = 0u16;
    for (modifier_index, keycodes) in modifier_keycodes
        .chunks_exact(keycodes_per_modifier)
        .enumerate()
    {
        let valid_keycodes: Vec<u8> = keycodes
            .iter()
            .copied()
            .filter(|keycode| *keycode != 0)
            .filter(|keycode| {
                keycode_symbols(min_keycode, keysyms_per_keycode, keysyms, *keycode).is_some()
            })
            .collect();
        physical_keycodes.extend_from_slice(&valid_keycodes);

        if valid_keycodes.is_empty() {
            continue;
        }
        let lock_only = valid_keycodes.iter().all(|keycode| {
            keycode_symbols(min_keycode, keysyms_per_keycode, keysyms, *keycode)
                .is_some_and(is_lock_only_mapping)
        });
        if !lock_only {
            conflicting_mask |= 1 << modifier_index;
        }
    }
    physical_keycodes.sort_unstable();
    physical_keycodes.dedup();
    ModifierPolicy {
        physical_keycodes,
        conflicting_mask,
    }
}

fn keycode_symbols(
    min_keycode: u8,
    keysyms_per_keycode: usize,
    keysyms: &[u32],
    keycode: u8,
) -> Option<&[u32]> {
    let index = keycode.checked_sub(min_keycode)? as usize;
    let start = index.checked_mul(keysyms_per_keycode)?;
    let end = start.checked_add(keysyms_per_keycode)?;
    keysyms.get(start..end)
}

fn is_lock_only_mapping(symbols: &[u32]) -> bool {
    let mut mapped = symbols.iter().copied().filter(|symbol| *symbol != 0);
    let Some(first) = mapped.next() else {
        return false;
    };
    LOCK_KEYSYMS.contains(&first) && mapped.all(|symbol| LOCK_KEYSYMS.contains(&symbol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet, VecDeque};

    const CTRL: u32 = 0xffe3;
    const SHIFT: u32 = 0xffe1;
    const V: u32 = 0x0076;

    struct FakeBackend {
        mapping: HashMap<u32, u8>,
        conflicts: Vec<u8>,
        pressed: VecDeque<Vec<u8>>,
        events: Vec<(InjectedKeyEvent, u8)>,
        fail_calls: HashSet<usize>,
        inject_calls: usize,
        conflicting_modifier_mask: u16,
        active_modifier_mask: u16,
    }

    impl FakeBackend {
        fn idle() -> Self {
            Self {
                mapping: HashMap::from([(CTRL, 37), (SHIFT, 50), (V, 55)]),
                conflicts: vec![37, 50, 55, 105, 62, 64, 108, 133, 134],
                pressed: VecDeque::from([Vec::new()]),
                events: Vec::new(),
                fail_calls: HashSet::new(),
                inject_calls: 0,
                conflicting_modifier_mask: 0,
                active_modifier_mask: 0,
            }
        }
    }

    impl KeyboardBackend for FakeBackend {
        fn resolve_keysyms(&self, keysyms: &[u32]) -> Result<Vec<u8>, String> {
            keysyms
                .iter()
                .map(|keysym| {
                    self.mapping
                        .get(keysym)
                        .copied()
                        .ok_or_else(|| format!("unmapped keysym {keysym:#x}"))
                })
                .collect()
        }

        fn conflict_keycodes(&self, _requested: &[u8]) -> Vec<u8> {
            self.conflicts.clone()
        }

        fn pressed_keycodes(&mut self) -> Result<Vec<u8>, String> {
            Ok(match self.pressed.len() {
                0 => Vec::new(),
                1 => self.pressed.front().cloned().unwrap_or_default(),
                _ => self.pressed.pop_front().unwrap_or_default(),
            })
        }

        fn conflicting_modifier_mask(&self) -> u16 {
            self.conflicting_modifier_mask
        }

        fn active_modifier_mask(&mut self) -> Result<u16, String> {
            Ok(self.active_modifier_mask)
        }

        fn inject(&mut self, event: InjectedKeyEvent, keycode: u8) -> Result<(), String> {
            let call = self.inject_calls;
            self.inject_calls += 1;
            self.events.push((event, keycode));
            if self.fail_calls.contains(&call) {
                Err(format!("server rejected injection call {call}"))
            } else {
                Ok(())
            }
        }

        fn synchronize(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn combo_orders_press_and_reverse_release() {
        let mut backend = FakeBackend::idle();

        send_key_combo_with_backend(
            &mut backend,
            &[CTRL, SHIFT, V],
            Duration::ZERO,
            Duration::ZERO,
        )
        .expect("idle keyboard should accept combo");

        assert_eq!(
            backend.events,
            vec![
                (InjectedKeyEvent::Press, 37),
                (InjectedKeyEvent::Press, 50),
                (InjectedKeyEvent::Press, 55),
                (InjectedKeyEvent::Release, 55),
                (InjectedKeyEvent::Release, 50),
                (InjectedKeyEvent::Release, 37),
            ]
        );
    }

    #[test]
    fn preheld_conflicts_timeout_without_injection_or_release() {
        for held in [37, 105, 50, 62, 64, 108, 133, 134] {
            let mut backend = FakeBackend::idle();
            backend.pressed = VecDeque::from([vec![held]]);

            let error = send_key_combo_with_backend(
                &mut backend,
                &[CTRL, V],
                Duration::ZERO,
                Duration::ZERO,
            )
            .expect_err("pre-held modifier must abort at the bounded deadline");

            assert!(error.to_string().contains("pre-held"));
            assert!(error.to_string().contains(&held.to_string()));
            assert!(backend.events.is_empty(), "held key {held} was modified");
        }
    }

    #[test]
    fn combo_defers_until_conflicting_key_is_released() {
        let mut backend = FakeBackend::idle();
        backend.pressed = VecDeque::from([vec![105], Vec::new()]);

        send_key_combo_with_backend(
            &mut backend,
            &[CTRL, V],
            Duration::from_millis(10),
            Duration::ZERO,
        )
        .expect("shortcut should continue after the held key is released");

        assert_eq!(backend.events, expected_ctrl_v_events());
    }

    #[test]
    fn effective_modifier_state_blocks_without_resetting_keys() {
        let mut backend = FakeBackend::idle();
        backend.conflicting_modifier_mask = 0x04;
        backend.active_modifier_mask = 0x04;

        let error =
            send_key_combo_with_backend(&mut backend, &[CTRL, V], Duration::ZERO, Duration::ZERO)
                .expect_err("effective Control state must block injection");

        assert!(error.to_string().contains("effective modifier mask 0x04"));
        assert!(backend.events.is_empty());
    }

    #[test]
    fn press_failures_are_precise_and_cleanup_every_possible_press() {
        for failing_press in 0..3 {
            let mut backend = FakeBackend::idle();
            backend.fail_calls = HashSet::from([failing_press, failing_press + 1]);

            let error = send_key_combo_with_backend(
                &mut backend,
                &[CTRL, SHIFT, V],
                Duration::ZERO,
                Duration::ZERO,
            )
            .expect_err("configured press failure must propagate");

            assert!(error.to_string().contains("press"));
            let attempted_releases: Vec<u8> = backend.events[failing_press + 1..]
                .iter()
                .filter_map(|(event, keycode)| {
                    (*event == InjectedKeyEvent::Release).then_some(*keycode)
                })
                .collect();
            let mut expected = vec![37, 50, 55][..=failing_press].to_vec();
            expected.reverse();
            assert_eq!(attempted_releases, expected);
        }
    }

    #[test]
    fn release_failures_try_every_remaining_release() {
        for failing_release in 0..3 {
            let mut backend = FakeBackend::idle();
            backend.fail_calls = HashSet::from([3 + failing_release]);

            let error = send_key_combo_with_backend(
                &mut backend,
                &[CTRL, SHIFT, V],
                Duration::ZERO,
                Duration::ZERO,
            )
            .expect_err("configured release failure must propagate");

            assert!(error.to_string().contains("release"));
            assert_eq!(
                &backend.events[3..],
                &[
                    (InjectedKeyEvent::Release, 55),
                    (InjectedKeyEvent::Release, 50),
                    (InjectedKeyEvent::Release, 37),
                    (InjectedKeyEvent::Release, 55),
                    (InjectedKeyEvent::Release, 50),
                    (InjectedKeyEvent::Release, 37),
                ]
            );
        }
    }

    #[test]
    fn modifier_policy_derives_altgr_and_custom_slots() {
        let min_keycode = 8;
        let keysyms_per_keycode = 2;
        let mut keysyms = vec![0; 8 * keysyms_per_keycode];
        keysyms[(12 - min_keycode) as usize * keysyms_per_keycode] = 0xfe03;
        keysyms[(13 - min_keycode) as usize * keysyms_per_keycode] = 0x1008_ff12;
        let modifier_keycodes = [
            0, 0, // Shift
            0, 0, // Lock
            0, 0, // Control
            0, 0, // Mod1
            0, 0, // Mod2
            13, 0, // Mod3: custom modifier
            0, 0, // Mod4
            12, 0, // Mod5: ISO_Level3_Shift / AltGr
        ];

        let policy = derive_modifier_policy(
            min_keycode,
            keysyms_per_keycode,
            &keysyms,
            2,
            &modifier_keycodes,
        );

        assert_eq!(policy.physical_keycodes, vec![12, 13]);
        assert_eq!(policy.conflicting_mask, (1 << 5) | (1 << 7));
    }

    #[test]
    fn modifier_policy_exempts_only_lock_only_slots() {
        let min_keycode = 8;
        let keysyms_per_keycode = 2;
        let mut keysyms = vec![0; 8 * keysyms_per_keycode];
        keysyms[(9 - min_keycode) as usize * keysyms_per_keycode] = 0xffe5;
        keysyms[(10 - min_keycode) as usize * keysyms_per_keycode] = 0xff7f;
        keysyms[(11 - min_keycode) as usize * keysyms_per_keycode] = 0xffe5;
        keysyms[(12 - min_keycode) as usize * keysyms_per_keycode] = 0x1008_ff12;
        let modifier_keycodes = [
            0, 0, // Shift
            9, 0, // Lock: CapsLock only
            0, 0, // Control
            0, 0, // Mod1
            10, 0, // Mod2: NumLock only
            11, 12, // Mod3: mixed lock + custom, therefore conflicting
            0, 0, // Mod4
            0, 0, // Mod5
        ];

        let policy = derive_modifier_policy(
            min_keycode,
            keysyms_per_keycode,
            &keysyms,
            2,
            &modifier_keycodes,
        );

        assert_eq!(policy.physical_keycodes, vec![9, 10, 11, 12]);
        assert_eq!(policy.conflicting_mask, 1 << 5);
    }

    #[test]
    fn modifier_policy_ignores_invalid_keycodes() {
        let policy = derive_modifier_policy(8, 1, &[0; 4], 2, &[1, 0, 0, 0]);

        assert!(policy.physical_keycodes.is_empty());
        assert_eq!(policy.conflicting_mask, 0);
    }

    #[test]
    fn repeat_preheld_conflict_performs_no_injection() {
        let mut backend = FakeBackend::idle();
        backend.pressed = VecDeque::from([vec![55]]);

        let error =
            send_key_repeat_with_backend(&mut backend, V, 3, Duration::ZERO, Duration::ZERO)
                .expect_err("pre-held repeated key must block injection");

        assert!(error.to_string().contains("pre-held"));
        assert!(backend.events.is_empty());
    }

    #[test]
    fn repeat_press_failures_attempt_cleanup() {
        for failing_repeat in 0..3 {
            let mut backend = FakeBackend::idle();
            backend.fail_calls = HashSet::from([failing_repeat * 2]);

            let error =
                send_key_repeat_with_backend(&mut backend, V, 3, Duration::ZERO, Duration::ZERO)
                    .expect_err("configured repeat press failure must propagate");

            assert!(error.to_string().contains("press"));
            assert_eq!(
                backend.events.last(),
                Some(&(InjectedKeyEvent::Release, 55))
            );
        }
    }

    #[test]
    fn repeat_release_failures_continue_and_cleanup() {
        for failing_repeat in 0..3 {
            let mut backend = FakeBackend::idle();
            backend.fail_calls = HashSet::from([failing_repeat * 2 + 1]);

            let error =
                send_key_repeat_with_backend(&mut backend, V, 3, Duration::ZERO, Duration::ZERO)
                    .expect_err("configured repeat release failure must propagate");

            assert!(error.to_string().contains("release"));
            assert!(backend.events.ends_with(&[
                (InjectedKeyEvent::Release, 55),
                (InjectedKeyEvent::Release, 55),
            ]));
        }
    }

    fn expected_ctrl_v_events() -> Vec<(InjectedKeyEvent, u8)> {
        vec![
            (InjectedKeyEvent::Press, 37),
            (InjectedKeyEvent::Press, 55),
            (InjectedKeyEvent::Release, 55),
            (InjectedKeyEvent::Release, 37),
        ]
    }
}
