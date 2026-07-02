//! Real-PulseAudio integration smoke tests for the `bt_profile` module.
//!
//! Exercises the libpulse-binding mainloop / context lifecycle against
//! the PulseAudio / pipewire-pulse server running on the host.  These
//! are integration tests (not unit tests) because they talk to the
//! real audio daemon and can only be meaningfully run in an
//! interactive desktop session.
//!
//! Marked `#[ignore]` so `cargo test` skips them by default; run with
//! `cargo test --test bt_profile_smoke -- --ignored` (or all tests via
//! `cargo test -- --include-ignored`) when you want to verify the
//! native audio stack actually answers.
//!
//! `round_trip_switch` is additionally gated behind the
//! `TALK_RS_BT_INTEGRATION=1` env var so it does not interfere with a
//! casual `--ignored` run on a machine where the user is currently
//! listening to music — the round-trip briefly takes the headset out
//! of A2DP and would interrupt playback.
//!
//! Requires the `capture` feature (the `bt_profile` module lives
//! behind it — libpulse-binding is a capture-gated dependency).
#![cfg(feature = "capture")]

use talk_rs::audio::bt_profile;

#[test]
#[ignore = "requires running PulseAudio/PipeWire server"]
fn list_cards_smoke() {
    let cards = match bt_profile::list_cards_for_test() {
        Ok(c) => c,
        Err(err) => panic!("list_cards failed: {}", err),
    };

    eprintln!("Found {} card(s):", cards.len());
    for card in &cards {
        eprintln!("  - {}", card);
    }
    // We don't assert a non-empty list because a CI box may have no
    // audio cards.  The success path of completing without error is
    // what proves the libpulse mainloop wiring works.
}

#[test]
#[ignore = "requires connected BT headset; gated by TALK_RS_BT_INTEGRATION=1"]
fn round_trip_switch() {
    if std::env::var("TALK_RS_BT_INTEGRATION").ok().as_deref() != Some("1") {
        eprintln!(
            "skipped: set TALK_RS_BT_INTEGRATION=1 to run this test \
             (it briefly takes the BT headset out of A2DP)"
        );
        return;
    }

    // Locate a headset.
    let (card_name, original_profile, profiles) = match bt_profile::find_headset_for_test() {
        Ok(Some(t)) => t,
        Ok(None) => {
            eprintln!("skipped: no Bluetooth headset card detected");
            return;
        }
        Err(err) => panic!("find_headset failed: {}", err),
    };

    eprintln!(
        "headset: {} active={} profiles={:?}",
        card_name, original_profile, profiles
    );

    // Pick a non-HFP profile to set as our "starting point" so we can
    // observe the switch.  Prefer a2dp-sink (modern) → a2dp_sink
    // (older naming) → off.
    let baseline_profile = profiles
        .iter()
        .find(|p| p.as_str() == "a2dp-sink")
        .or_else(|| profiles.iter().find(|p| p.as_str() == "a2dp_sink"))
        .or_else(|| profiles.iter().find(|p| p.as_str() == "off"))
        .cloned()
        .expect("headset must have at least 'off' profile");

    eprintln!("setting baseline profile: {}", baseline_profile);
    bt_profile::set_card_profile_for_test(&card_name, &baseline_profile)
        .expect("set baseline profile");

    // Verify the baseline took effect.
    let active = bt_profile::get_active_profile_for_test(&card_name)
        .expect("get active profile")
        .expect("card present");
    assert_eq!(active, baseline_profile, "baseline profile didn't stick");

    // Now run the real activate_headset path.
    let saved = bt_profile::activate_headset()
        .expect("activate_headset")
        .expect("activate_headset returned Some — headset present");

    assert_eq!(saved.card_name, card_name);
    assert_eq!(saved.original_profile, baseline_profile);

    // Verify the card is now on an HFP profile.
    let active = bt_profile::get_active_profile_for_test(&card_name)
        .expect("get active profile after switch")
        .expect("card present after switch");
    assert!(
        active.starts_with("headset-head-unit"),
        "expected HFP profile after activate_headset, got '{}'",
        active
    );
    eprintln!("activate_headset → switched to '{}'", active);

    // Restore via the public API.
    bt_profile::restore_profile(&saved).expect("restore_profile");

    let active = bt_profile::get_active_profile_for_test(&card_name)
        .expect("get active profile after restore")
        .expect("card present after restore");
    assert_eq!(
        active, baseline_profile,
        "restore_profile did not bring the card back to baseline"
    );
    eprintln!("restore_profile → back to '{}'", active);

    // Finally restore the user's original profile too (in case our
    // baseline was different from what was active when the test
    // started).
    if original_profile != baseline_profile {
        bt_profile::set_card_profile_for_test(&card_name, &original_profile)
            .expect("restore user's pre-test profile");
        eprintln!("restored user profile: {}", original_profile);
    }
}
