//! Bluetooth headset profile auto-switching.
//!
//! When a Bluetooth headset is connected, it typically advertises two
//! mutually exclusive PulseAudio/PipeWire profiles:
//!
//! * `a2dp_sink` — high-quality stereo *output* only, no microphone.
//! * `headset-head-unit-msbc` / `*-cvsd` / `headset-head-unit` —
//!   Hands-Free Profile (HFP) which enables the headset microphone at
//!   the cost of lower audio quality.
//!
//! For voice dictation we need the microphone, so on capture start we
//! save the card's current profile and switch it to the best available
//! HFP profile.  On capture stop we restore the saved profile so the
//! user gets high-quality stereo output back immediately.
//!
//! ## Crash resilience
//!
//! The saved profile is persisted to `$XDG_RUNTIME_DIR/talk-rs/card-profile.json`
//! *before* the switch is performed, so an unclean termination
//! (SIGKILL, power loss) leaves a recoverable state file behind.  The
//! next invocation calls [`recover_stale_profile`] at startup to detect
//! and restore the previous profile *before* activating HFP for the new
//! recording.  This is the key improvement over the legacy `memo` shell
//! script which could permanently lose the original profile on crash.
//!
//! ## Implementation
//!
//! All PulseAudio interaction is via [`libpulse_binding`] — no
//! subprocess (`pactl`) or D-Bus is used.  On PipeWire systems,
//! `pipewire-pulse` provides the PulseAudio client API as a
//! compatibility shim, so `libpulse_binding` works equally well there.

use crate::error::TalkError;
use libpulse_binding as pulse;
use pulse::callbacks::ListResult;
use pulse::context::introspect::CardInfo;
use pulse::context::{Context, FlagSet as ContextFlagSet, State as ContextState};
use pulse::mainloop::standard::{IterateResult, Mainloop};
use pulse::operation::State as OperationState;
use pulse::proplist::{properties as pa_props, Proplist};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

/// PulseAudio card profiles we will switch to for microphone capture,
/// in order of preference (best quality first).
///
/// These are PulseAudio-standard profile names — not vendor-specific —
/// so detection works for any HFP-capable Bluetooth headset.
const PREFERRED_HFP_PROFILES: &[&str] = &[
    // HFP with mSBC codec, 16 kHz wideband — best voice quality.
    "headset-head-unit-msbc",
    // HFP with CVSD codec, 8 kHz narrowband — wider compatibility.
    "headset-head-unit-cvsd",
    // Generic HFP/HSP — last-resort fallback for older stacks.
    "headset-head-unit",
];

/// Application name reported to the PulseAudio server when we connect.
const PA_APP_NAME: &str = "talk-rs";

/// State file name relative to `$XDG_RUNTIME_DIR/talk-rs/`.
const STATE_FILE_NAME: &str = "card-profile.json";

/// Saved profile state, persisted to disk for crash recovery.
///
/// Contains everything needed to undo a profile switch: the card name
/// (PulseAudio identifier, e.g. `bluez_card.AA_BB_CC_DD_EE_FF`) and
/// the profile name that was active before we switched (e.g.
/// `a2dp_sink`).  `switched_at` is informational only.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SavedProfile {
    /// PulseAudio card name (e.g. `bluez_card.AA_BB_CC_DD_EE_FF`).
    pub card_name: String,
    /// Profile that was active before we switched to HFP.
    pub original_profile: String,
    /// UTC timestamp of the switch; informational only.
    pub switched_at: chrono::DateTime<chrono::Utc>,
}

/// Snapshot of a PulseAudio sound card, used internally by detection.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CardSnapshot {
    /// PulseAudio card name (stable identifier).
    name: String,
    /// `device.form_factor` property, if set ("headset", "speaker", …).
    form_factor: Option<String>,
    /// Currently active profile name (e.g. `a2dp_sink`).
    active_profile: String,
    /// All profile names available on this card.
    profiles: Vec<String>,
}

// ---------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------

/// Detect a connected Bluetooth headset, save its current profile to
/// the on-disk state file, then switch the card to the best available
/// HFP profile.
///
/// Returns:
/// - `Ok(Some(SavedProfile))` — a headset was found and switched; the
///   caller should keep this value alive (typically inside a
///   [`HeadsetGuard`]) and pass it to [`restore_profile`] on shutdown.
/// - `Ok(None)` — no Bluetooth headset card present, or the card is
///   already on an HFP profile (no action needed).
/// - `Err(_)` — PulseAudio connection or profile-switch failed.  The
///   caller should treat this as non-fatal: log a warning and proceed
///   with the recording on whatever input device is currently active.
pub fn activate_headset() -> Result<Option<SavedProfile>, TalkError> {
    let cards = list_cards()?;
    let Some(card) = find_headset_card(&cards) else {
        log::debug!("bt_profile: no Bluetooth headset card detected");
        return Ok(None);
    };

    // Pick the best HFP profile available on this card.
    let Some(hfp_profile) = pick_hfp_profile(&card.profiles) else {
        log::warn!(
            "bt_profile: headset card '{}' has no HFP profile available (profiles: {:?})",
            card.name,
            card.profiles
        );
        return Ok(None);
    };

    // No-op when already on an HFP profile (e.g. another tool already
    // switched it, or the user manually selected HFP).
    if card.active_profile == hfp_profile
        || PREFERRED_HFP_PROFILES.contains(&card.active_profile.as_str())
    {
        log::debug!(
            "bt_profile: card '{}' already on HFP profile '{}', skipping switch",
            card.name,
            card.active_profile
        );
        return Ok(None);
    }

    let saved = SavedProfile {
        card_name: card.name.clone(),
        original_profile: card.active_profile.clone(),
        switched_at: chrono::Utc::now(),
    };

    // Write the state file BEFORE switching so that a crash between
    // steps still leaves recovery information on disk.
    write_state_file(&saved)?;

    log::info!(
        "bt_profile: switching card '{}' from '{}' to '{}' for microphone capture",
        card.name,
        card.active_profile,
        hfp_profile
    );

    if let Err(err) = set_card_profile(&card.name, hfp_profile) {
        // Switch failed: state file is on disk but the card is still
        // on its original profile.  Delete the state file so we don't
        // wrongly "restore" on next launch.
        let _ = remove_state_file();
        return Err(err);
    }

    Ok(Some(saved))
}

/// Restore a previously saved profile and delete the state file.
///
/// Idempotent: repeated calls with the same `SavedProfile` are safe;
/// missing state file is not an error.  Logs (does not return) profile
/// switch errors so that drop-time restoration in [`HeadsetGuard`]
/// never panics.
pub fn restore_profile(saved: &SavedProfile) -> Result<(), TalkError> {
    log::info!(
        "bt_profile: restoring card '{}' to profile '{}'",
        saved.card_name,
        saved.original_profile
    );

    let switch_result = set_card_profile(&saved.card_name, &saved.original_profile);

    // Always remove the state file, even if the switch failed: if the
    // switch can't be done now (card disconnected, daemon dead, …),
    // recover_stale_profile would just hit the same error on next
    // launch.  Leaving the file would create a permanent error loop.
    let _ = remove_state_file();

    switch_result
}

/// Detect a stale state file from a previous unclean termination and
/// restore the saved profile.
///
/// Should be called at the very beginning of every command that may
/// switch profiles, BEFORE [`activate_headset`].  Returns `Ok(true)`
/// if a stale profile was recovered, `Ok(false)` if there was nothing
/// to recover.  Errors are logged (not returned) for malformed state
/// files; the malformed file is deleted to avoid an error loop.
pub fn recover_stale_profile() -> Result<bool, TalkError> {
    let path = state_file_path()?;
    if !path.exists() {
        return Ok(false);
    }

    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(err) => {
            log::warn!(
                "bt_profile: failed to read stale state file {}: {}",
                path.display(),
                err
            );
            let _ = fs::remove_file(&path);
            return Ok(false);
        }
    };

    let saved: SavedProfile = match serde_json::from_str(&content) {
        Ok(s) => s,
        Err(err) => {
            log::warn!(
                "bt_profile: malformed state file {} (deleting): {}",
                path.display(),
                err
            );
            let _ = fs::remove_file(&path);
            return Ok(false);
        }
    };

    log::info!(
        "bt_profile: recovering stale profile from previous run: card='{}' profile='{}' (saved at {})",
        saved.card_name,
        saved.original_profile,
        saved.switched_at
    );

    match restore_profile(&saved) {
        Ok(()) => Ok(true),
        Err(err) => {
            // restore_profile already removed the state file; just
            // log the error and proceed (the new run will set up its
            // own state).
            log::warn!("bt_profile: stale profile recovery failed: {}", err);
            Ok(false)
        }
    }
}

/// RAII guard that restores a saved Bluetooth headset profile when
/// dropped.
///
/// Wrapping the saved profile in this guard ensures the original
/// profile is restored on:
/// - normal function return,
/// - `?`-propagated errors,
/// - panics,
/// - SIGINT-driven `tokio::select!` exits.
///
/// Drop is best-effort: errors during restoration are logged (not
/// propagated) so the guard's destructor cannot itself panic.
///
/// Construct from the result of [`activate_headset`]:
///
/// ```ignore
/// let _bt_guard = HeadsetGuard::new(activate_headset().ok().flatten());
/// // ... recording runs ...
/// // _bt_guard is dropped here, profile is restored automatically.
/// ```
pub struct HeadsetGuard {
    saved: Option<SavedProfile>,
}

impl HeadsetGuard {
    /// Wrap an optional saved profile in a guard.  `None` produces a
    /// no-op guard (used when no headset was found or auto-switch is
    /// disabled).
    pub fn new(saved: Option<SavedProfile>) -> Self {
        Self { saved }
    }
}

impl Drop for HeadsetGuard {
    fn drop(&mut self) {
        if let Some(saved) = self.saved.take() {
            if let Err(err) = restore_profile(&saved) {
                log::warn!(
                    "bt_profile: failed to restore profile on guard drop ({}): {}",
                    saved.card_name,
                    err
                );
            }
        }
    }
}

// ---------------------------------------------------------------------
// Detection logic (pure functions, unit-testable)
// ---------------------------------------------------------------------

/// Pick the best HFP profile from a list of available profile names.
///
/// Priority order: msbc (16 kHz wideband) > cvsd (8 kHz narrowband) >
/// generic head-unit.  Returns `None` if no preferred HFP profile is
/// present in `available`.
fn pick_hfp_profile<'a>(available: &[String]) -> Option<&'a str> {
    PREFERRED_HFP_PROFILES
        .iter()
        .find(|preferred| available.iter().any(|a| a == *preferred))
        .copied()
}

/// Find a Bluetooth headset card from a list of card snapshots.
///
/// A card qualifies as a headset if either:
/// 1. Its `device.form_factor` property is `"headset"` (the
///    PulseAudio-standard way), OR
/// 2. Its name starts with `bluez_card.` AND it has at least one
///    `headset-head-unit*` profile available (fallback for stacks that
///    don't set `form_factor`).
///
/// When multiple cards qualify, the first one is returned.
fn find_headset_card(cards: &[CardSnapshot]) -> Option<&CardSnapshot> {
    cards.iter().find(|c| is_headset_card(c))
}

fn is_headset_card(card: &CardSnapshot) -> bool {
    if card.form_factor.as_deref() == Some("headset") {
        return true;
    }
    if card.name.starts_with("bluez_card.")
        && card
            .profiles
            .iter()
            .any(|p| p.starts_with("headset-head-unit"))
    {
        return true;
    }
    false
}

// ---------------------------------------------------------------------
// State file (atomic write + parse)
// ---------------------------------------------------------------------

/// Resolve the state file path: `$XDG_RUNTIME_DIR/talk-rs/card-profile.json`.
///
/// Falls back to a per-user `/tmp/talk-rs-<USER>/` directory when
/// `XDG_RUNTIME_DIR` is not set (rare on modern systemd systems but
/// possible in minimal containers).  As a last resort, falls back to
/// `/tmp/talk-rs/`; this is acceptable for crash-recovery state since
/// the file is single-user and removed on successful restore.
fn state_file_path() -> Result<PathBuf, TalkError> {
    let dir = if let Ok(rt) = std::env::var("XDG_RUNTIME_DIR") {
        PathBuf::from(rt).join("talk-rs")
    } else if let Ok(user) = std::env::var("USER") {
        PathBuf::from(format!("/tmp/talk-rs-{}", user))
    } else {
        PathBuf::from("/tmp/talk-rs")
    };
    Ok(dir.join(STATE_FILE_NAME))
}

/// Atomically write the saved-profile state file.
///
/// Writes to `<path>.tmp` then renames into place so a crash mid-write
/// cannot leave a half-written file.
fn write_state_file(saved: &SavedProfile) -> Result<(), TalkError> {
    let path = state_file_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(TalkError::Io)?;
    }

    let json = serde_json::to_string_pretty(saved).map_err(|err| {
        TalkError::Audio(format!(
            "bt_profile: failed to serialize saved profile: {}",
            err
        ))
    })?;

    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json).map_err(TalkError::Io)?;
    fs::rename(&tmp, &path).map_err(TalkError::Io)?;
    Ok(())
}

/// Delete the state file.  Missing file is not an error.
fn remove_state_file() -> Result<(), TalkError> {
    let path = state_file_path()?;
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(TalkError::Io(err)),
    }
}

// ---------------------------------------------------------------------
// PulseAudio interaction
// ---------------------------------------------------------------------

/// Connect to PulseAudio, run the given closure with a ready Context,
/// then disconnect.
///
/// Drives the standard mainloop manually since we don't need async
/// (these calls are infrequent and short-lived).
fn with_pulse_context<F, R>(f: F) -> Result<R, TalkError>
where
    F: FnOnce(Rc<RefCell<Mainloop>>, Rc<RefCell<Context>>) -> Result<R, TalkError>,
{
    let mut proplist = Proplist::new()
        .ok_or_else(|| TalkError::Audio("bt_profile: failed to create PA proplist".to_string()))?;
    proplist
        .set_str(pa_props::APPLICATION_NAME, PA_APP_NAME)
        .map_err(|_| {
            TalkError::Audio("bt_profile: failed to set PA application name".to_string())
        })?;

    let mainloop = Mainloop::new()
        .ok_or_else(|| TalkError::Audio("bt_profile: failed to create PA mainloop".to_string()))?;
    let mainloop = Rc::new(RefCell::new(mainloop));

    let context = {
        let ml = mainloop.borrow();
        Context::new_with_proplist(&*ml, PA_APP_NAME, &proplist).ok_or_else(|| {
            TalkError::Audio("bt_profile: failed to create PA context".to_string())
        })?
    };
    let context = Rc::new(RefCell::new(context));

    context
        .borrow_mut()
        .connect(None, ContextFlagSet::NOFLAGS, None)
        .map_err(|err| {
            TalkError::Audio(format!("bt_profile: PA context connect failed: {}", err))
        })?;

    // Drive the mainloop until the context is Ready or fails.
    loop {
        match mainloop.borrow_mut().iterate(false) {
            IterateResult::Quit(_) | IterateResult::Err(_) => {
                return Err(TalkError::Audio(
                    "bt_profile: PA mainloop terminated during connect".to_string(),
                ));
            }
            IterateResult::Success(_) => {}
        }
        match context.borrow().get_state() {
            ContextState::Ready => break,
            ContextState::Failed | ContextState::Terminated => {
                return Err(TalkError::Audio(
                    "bt_profile: PA context failed to reach Ready state".to_string(),
                ));
            }
            _ => {}
        }
    }

    let result = f(mainloop.clone(), context.clone());

    // Disconnect and let RAII close the mainloop.
    context.borrow_mut().disconnect();

    result
}

/// List all PulseAudio sound cards.
fn list_cards() -> Result<Vec<CardSnapshot>, TalkError> {
    with_pulse_context(|mainloop, context| {
        let cards: Rc<RefCell<Vec<CardSnapshot>>> = Rc::new(RefCell::new(Vec::new()));
        let cards_inner = cards.clone();

        let done = Rc::new(RefCell::new(false));
        let done_inner = done.clone();

        let op = context.borrow().introspect().get_card_info_list(
            move |result: ListResult<&CardInfo>| match result {
                ListResult::Item(card) => {
                    if let Some(snap) = card_info_to_snapshot(card) {
                        cards_inner.borrow_mut().push(snap);
                    }
                }
                ListResult::End | ListResult::Error => {
                    *done_inner.borrow_mut() = true;
                }
            },
        );

        // Drive the mainloop until the operation completes OR the
        // End/Error callback fires.
        loop {
            match mainloop.borrow_mut().iterate(false) {
                IterateResult::Quit(_) | IterateResult::Err(_) => {
                    return Err(TalkError::Audio(
                        "bt_profile: PA mainloop terminated during list_cards".to_string(),
                    ));
                }
                IterateResult::Success(_) => {}
            }
            if *done.borrow() || op.get_state() == OperationState::Done {
                break;
            }
            if op.get_state() == OperationState::Cancelled {
                return Err(TalkError::Audio(
                    "bt_profile: list_cards operation cancelled".to_string(),
                ));
            }
        }

        let snapshots = cards.borrow().clone();
        Ok(snapshots)
    })
}

/// Convert a libpulse `CardInfo` into our internal `CardSnapshot`.
///
/// Returns `None` for cards that have no usable name (should never
/// happen in practice but keeps the type system honest).
fn card_info_to_snapshot(card: &CardInfo) -> Option<CardSnapshot> {
    let name = card.name.as_deref()?.to_string();

    let form_factor = card.proplist.get_str(pa_props::DEVICE_FORM_FACTOR);

    let active_profile = card
        .active_profile
        .as_ref()
        .and_then(|p| p.name.as_deref())
        .unwrap_or("")
        .to_string();

    let profiles = card
        .profiles
        .iter()
        .filter_map(|p| p.name.as_deref().map(str::to_string))
        .collect();

    Some(CardSnapshot {
        name,
        form_factor,
        active_profile,
        profiles,
    })
}

/// Switch a card to a named profile.  Blocks until PulseAudio confirms
/// success or failure.
fn set_card_profile(card_name: &str, profile: &str) -> Result<(), TalkError> {
    with_pulse_context(|mainloop, context| {
        let done = Rc::new(RefCell::new(false));
        let success = Rc::new(RefCell::new(false));
        let done_cb = done.clone();
        let success_cb = success.clone();

        let op = context.borrow_mut().introspect().set_card_profile_by_name(
            card_name,
            profile,
            Some(Box::new(move |ok| {
                *success_cb.borrow_mut() = ok;
                *done_cb.borrow_mut() = true;
            })),
        );

        loop {
            match mainloop.borrow_mut().iterate(false) {
                IterateResult::Quit(_) | IterateResult::Err(_) => {
                    return Err(TalkError::Audio(
                        "bt_profile: PA mainloop terminated during set_card_profile".to_string(),
                    ));
                }
                IterateResult::Success(_) => {}
            }
            if *done.borrow() {
                break;
            }
            if op.get_state() == OperationState::Cancelled {
                return Err(TalkError::Audio(
                    "bt_profile: set_card_profile operation cancelled".to_string(),
                ));
            }
        }

        if *success.borrow() {
            Ok(())
        } else {
            Err(TalkError::Audio(format!(
                "bt_profile: PulseAudio refused to switch card '{}' to profile '{}'",
                card_name, profile
            )))
        }
    })
}

// ---------------------------------------------------------------------
// Test-only public shims
// ---------------------------------------------------------------------

/// Test helper: expose `list_cards()` for the integration smoke test.
/// Returns the raw card-snapshot debug output to avoid leaking the
/// internal `CardSnapshot` type into the public API.
#[doc(hidden)]
pub fn list_cards_for_test() -> Result<Vec<String>, TalkError> {
    let cards = list_cards()?;
    Ok(cards
        .into_iter()
        .map(|c| {
            format!(
                "name={} form_factor={:?} active_profile={} profiles={:?}",
                c.name, c.form_factor, c.active_profile, c.profiles
            )
        })
        .collect())
}

/// Test helper: directly set a card profile (used by integration
/// tests to put a headset back to A2DP before exercising the
/// activate_headset round-trip).
#[doc(hidden)]
pub fn set_card_profile_for_test(card: &str, profile: &str) -> Result<(), TalkError> {
    set_card_profile(card, profile)
}

/// Test helper: query the active profile of a single card.  Returns
/// `Ok(None)` when no card with that name is found.
#[doc(hidden)]
pub fn get_active_profile_for_test(card_name: &str) -> Result<Option<String>, TalkError> {
    let cards = list_cards()?;
    Ok(cards
        .into_iter()
        .find(|c| c.name == card_name)
        .map(|c| c.active_profile))
}

/// Test helper: find the first headset card and return its name +
/// active profile + available profiles, for integration tests.
#[doc(hidden)]
pub fn find_headset_for_test() -> Result<Option<(String, String, Vec<String>)>, TalkError> {
    let cards = list_cards()?;
    Ok(find_headset_card(&cards)
        .map(|c| (c.name.clone(), c.active_profile.clone(), c.profiles.clone())))
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(
        name: &str,
        form_factor: Option<&str>,
        active: &str,
        profiles: &[&str],
    ) -> CardSnapshot {
        CardSnapshot {
            name: name.to_string(),
            form_factor: form_factor.map(str::to_string),
            active_profile: active.to_string(),
            profiles: profiles.iter().map(|s| s.to_string()).collect(),
        }
    }

    // -- pick_hfp_profile ---------------------------------------------

    #[test]
    fn pick_hfp_prefers_msbc_when_available() {
        let avail: Vec<String> = [
            "a2dp_sink",
            "headset-head-unit-cvsd",
            "headset-head-unit-msbc",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(pick_hfp_profile(&avail), Some("headset-head-unit-msbc"));
    }

    #[test]
    fn pick_hfp_falls_back_to_cvsd_when_no_msbc() {
        let avail: Vec<String> = ["a2dp_sink", "headset-head-unit-cvsd"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(pick_hfp_profile(&avail), Some("headset-head-unit-cvsd"));
    }

    #[test]
    fn pick_hfp_falls_back_to_generic_when_no_codec_variant() {
        let avail: Vec<String> = ["a2dp_sink", "headset-head-unit"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(pick_hfp_profile(&avail), Some("headset-head-unit"));
    }

    #[test]
    fn pick_hfp_returns_none_when_no_hfp_profile() {
        let avail: Vec<String> = ["a2dp_sink", "off"].iter().map(|s| s.to_string()).collect();
        assert_eq!(pick_hfp_profile(&avail), None);
    }

    #[test]
    fn pick_hfp_returns_none_for_empty() {
        assert_eq!(pick_hfp_profile(&[]), None);
    }

    // -- find_headset_card --------------------------------------------

    #[test]
    fn find_headset_picks_form_factor_headset() {
        let cards = vec![
            snap(
                "alsa_card.pci",
                Some("internal"),
                "output:hdmi",
                &["output:hdmi"],
            ),
            snap(
                "bluez_card.AA_BB_CC_DD_EE_FF",
                Some("headset"),
                "a2dp_sink",
                &["a2dp_sink", "headset-head-unit-msbc"],
            ),
        ];
        let found = find_headset_card(&cards).expect("headset should be found");
        assert_eq!(found.name, "bluez_card.AA_BB_CC_DD_EE_FF");
    }

    #[test]
    fn find_headset_falls_back_to_bluez_name_when_no_form_factor() {
        // Some BT stacks omit device.form_factor; the bluez_card.* +
        // headset-head-unit* fallback should still detect the card.
        let cards = vec![snap(
            "bluez_card.AA_BB",
            None,
            "a2dp_sink",
            &["a2dp_sink", "headset-head-unit-cvsd"],
        )];
        assert!(find_headset_card(&cards).is_some());
    }

    #[test]
    fn find_headset_ignores_bluez_without_hfp_profile() {
        // A BT speaker has bluez_card.* prefix but no head-unit profile.
        let cards = vec![snap(
            "bluez_card.SPEAKER",
            None,
            "a2dp_sink",
            &["a2dp_sink"],
        )];
        assert!(find_headset_card(&cards).is_none());
    }

    #[test]
    fn find_headset_ignores_non_headset_form_factor() {
        let cards = vec![snap(
            "alsa_card.pci",
            Some("internal"),
            "output:analog",
            &["output:analog", "input:analog"],
        )];
        assert!(find_headset_card(&cards).is_none());
    }

    #[test]
    fn find_headset_returns_first_match_when_multiple() {
        let cards = vec![
            snap(
                "bluez_card.FIRST",
                Some("headset"),
                "a2dp_sink",
                &["a2dp_sink", "headset-head-unit-msbc"],
            ),
            snap(
                "bluez_card.SECOND",
                Some("headset"),
                "a2dp_sink",
                &["a2dp_sink", "headset-head-unit-msbc"],
            ),
        ];
        let found = find_headset_card(&cards).expect("headset should be found");
        assert_eq!(found.name, "bluez_card.FIRST");
    }

    #[test]
    fn find_headset_returns_none_for_empty() {
        assert!(find_headset_card(&[]).is_none());
    }

    // -- SavedProfile JSON round-trip ---------------------------------

    #[test]
    fn saved_profile_json_round_trip() {
        let saved = SavedProfile {
            card_name: "bluez_card.AA_BB".to_string(),
            original_profile: "a2dp_sink".to_string(),
            switched_at: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&saved).expect("serialize");
        let back: SavedProfile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(saved, back);
    }

    // -- State file path resolution -----------------------------------

    #[test]
    fn state_file_path_uses_xdg_runtime_dir() {
        // Cannot mutate process env safely from parallel tests, but we
        // can at least verify the file name and parent.
        let path = state_file_path().expect("path resolves");
        assert!(path.ends_with(STATE_FILE_NAME));
        assert!(path.parent().expect("has parent").ends_with("talk-rs"));
    }

    // -- HeadsetGuard -------------------------------------------------

    #[test]
    fn headset_guard_with_none_is_noop_on_drop() {
        // No saved profile — drop must not panic and must not touch
        // PulseAudio (we have no PA running in unit tests).
        let g = HeadsetGuard::new(None);
        drop(g);
    }
}
