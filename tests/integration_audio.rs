//! Integration tests for audio capture with real hardware.
//!
//! Non-ignored tests verify basic device enumeration via cpal.
//! Ignored tests require actual audio hardware and should be run with:
//!   cargo test -- --ignored

use cpal::traits::{DeviceTrait, HostTrait};
use talk_rs::audio::cpal_capture::CpalCapture;
use talk_rs::audio::AudioCapture;
use talk_rs::config::AudioConfig;

fn test_audio_config() -> AudioConfig {
    AudioConfig {
        sample_rate: 16000,
        channels: 1,
        bitrate: 32000,
    }
}

#[test]
fn test_default_input_device_exists() {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No default input device found");

    let name = device.name().expect("Device has no name");
    assert!(!name.is_empty(), "Device name should not be empty");

    let configs = device
        .supported_input_configs()
        .expect("Failed to query supported input configs");
    let count = configs.count();
    assert!(count > 0, "Device should support at least one input config");
}

#[test]
#[ignore] // Requires audio hardware
fn test_cpal_capture_start_stop() {
    let config = test_audio_config();
    let mut capture = CpalCapture::new(config);

    let _receiver = capture.start().expect("Failed to start capture");

    std::thread::sleep(std::time::Duration::from_millis(200));

    let error = capture.last_error().expect("Failed to check last_error");
    assert!(error.is_none(), "Unexpected capture error: {:?}", error);

    capture.stop().expect("Failed to stop capture");
}

#[test]
#[ignore] // Requires audio hardware
fn test_cpal_capture_receives_audio_data() {
    let config = test_audio_config();
    let mut capture = CpalCapture::new(config);

    let mut receiver = capture.start().expect("Failed to start capture");

    // Wait for a few chunks (20ms each, wait up to 1s)
    let mut chunks_received = 0;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);

    while std::time::Instant::now() < deadline && chunks_received < 3 {
        match receiver.try_recv() {
            Ok(chunk) => {
                assert!(!chunk.is_empty(), "Received empty audio chunk");
                chunks_received += 1;
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                panic!("Audio channel disconnected unexpectedly");
            }
        }
    }

    assert!(
        chunks_received >= 1,
        "Expected at least 1 audio chunk, got {}",
        chunks_received
    );

    let error = capture.last_error().expect("Failed to check last_error");
    assert!(error.is_none(), "Unexpected capture error: {:?}", error);

    capture.stop().expect("Failed to stop capture");
}
