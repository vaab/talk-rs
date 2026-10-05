//! Real-PipeWire check of capture readiness proof and explicit targeting.
//!
//! Opt-in: it creates (and always removes) an inaudible virtual
//! microphone fed with noise on the running PipeWire/PulseAudio server.  It never changes the default
//! source and never touches Bluetooth cards.
//!
//! ```sh
//! cargo test --test integration_capture_readiness -- --ignored --nocapture
//! ```
#![cfg(feature = "capture")]

use std::process::Command;
use std::time::Duration;
use talk_rs::audio::pipewire_capture::PipeWireCapture;
use talk_rs::audio::{AudioCapture, CaptureReadiness};
use talk_rs::config::AudioConfig;

/// Virtual source loaded with `pactl`, unloaded on drop (also on panic).
struct VirtualSource {
    modules: Vec<String>,
    name: String,
    generator: Option<std::process::Child>,
}

fn load_module(args: &[&str]) -> String {
    let output = Command::new("pactl")
        .arg("load-module")
        .args(args)
        .output()
        .expect("run pactl load-module");
    assert!(
        output.status.success(),
        "pactl load-module {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

impl VirtualSource {
    /// Build an inaudible microphone carrying real signal: noise is
    /// played into a null sink (never a real device), and a remapped
    /// source exposes that sink's monitor as an ordinary source node.
    fn load(name: &str) -> Self {
        let sink = format!("{name}_sink");
        let mut source = Self {
            modules: Vec::new(),
            name: name.to_string(),
            generator: None,
        };
        source.modules.push(load_module(&[
            "module-null-sink",
            &format!("sink_name={sink}"),
            "channel_map=mono",
        ]));
        source.modules.push(load_module(&[
            "module-remap-source",
            &format!("master={sink}.monitor"),
            &format!("source_name={name}"),
            "channel_map=mono",
        ]));
        source.generator = Command::new("pacat")
            .args([
                "--playback",
                &format!("--device={sink}"),
                "--format=s16le",
                "--rate=48000",
                "--channels=1",
                "--raw",
                "/dev/urandom",
            ])
            .spawn()
            .ok();
        source
    }
}

impl Drop for VirtualSource {
    fn drop(&mut self) {
        if let Some(mut generator) = self.generator.take() {
            let _ = generator.kill();
            let _ = generator.wait();
        }
        for module in self.modules.iter().rev() {
            let _ = Command::new("pactl")
                .args(["unload-module", module])
                .status();
        }
    }
}

/// Rows of `pactl list <kind> short`, split on tabs.
fn pactl_short(kind: &str) -> Vec<Vec<String>> {
    let output = Command::new("pactl")
        .args(["list", kind, "short"])
        .output()
        .expect("run pactl list");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| line.split('\t').map(str::to_string).collect())
        .collect()
}

fn capture_config() -> AudioConfig {
    AudioConfig {
        sample_rate: 48_000,
        channels: 1,
        bitrate: 32_000,
    }
}

#[tokio::test]
#[ignore = "needs a running PipeWire server; creates a temporary virtual source"]
async fn readiness_is_proven_on_the_explicitly_targeted_source() {
    let source = VirtualSource::load(&format!("talk_rs_readiness_{}", std::process::id()));
    let mut capture = PipeWireCapture::new(capture_config(), source.name.clone());
    let mut audio = capture.start().expect("start capture");
    let readiness = capture.readiness();
    assert!(
        matches!(readiness, CaptureReadiness::Pending(_)),
        "live capture must require a proof"
    );
    let ready = readiness
        .wait(talk_rs::audio::CAPTURE_READY_LIMIT)
        .await
        .expect("readiness proven")
        .expect("live capture yields a proof");
    assert_eq!(ready.source, source.name);
    // The proof names the target; also prove PipeWire actually linked
    // the stream to it (and not to the default microphone).
    let source_index = pactl_short("sources")
        .into_iter()
        .find(|cols| cols.get(1) == Some(&source.name))
        .and_then(|cols| cols.first().cloned())
        .expect("virtual source listed");
    let outputs = pactl_short("source-outputs");
    assert!(
        outputs
            .iter()
            .any(|cols| cols.get(1) == Some(&source_index)),
        "a capture stream must be linked to source #{source_index}: {outputs:?}"
    );
    let chunk = tokio::time::timeout(Duration::from_secs(2), audio.recv())
        .await
        .expect("audio flows after readiness");
    assert!(chunk.is_some_and(|c| !c.is_empty()));
    capture.stop().expect("stop capture");
}

#[tokio::test]
#[ignore = "needs a running PipeWire server"]
async fn missing_target_fails_instead_of_falling_back_to_another_microphone() {
    let mut capture = PipeWireCapture::new(
        capture_config(),
        format!("talk_rs_no_such_source_{}", std::process::id()),
    );
    let _audio = capture.start().expect("start capture");
    let result = capture.readiness().wait(Duration::from_secs(3)).await;
    capture.stop().expect("stop capture");
    eprintln!("missing target outcome: {result:?}");
    assert!(
        result.is_err(),
        "a missing target must never be proven ready: {result:?}"
    );
}
