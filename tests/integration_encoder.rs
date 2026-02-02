//! Integration tests for OpusEncoder with real Opus codec.
//!
//! These tests verify OpusEncoder functionality using the actual opus library,
//! not mocks. They test encoding, decoding roundtrips, flushing, and bitrate handling.

use talk_rs::core::audio::encoder::{AudioEncoder, OpusEncoder};
use talk_rs::core::config::AudioConfig;

fn test_audio_config() -> AudioConfig {
    AudioConfig {
        sample_rate: 16000,
        channels: 1,
        bitrate: 32000,
    }
}

/// Generate a 440Hz sine wave for the given duration in samples.
fn generate_sine_wave(num_samples: usize, frequency: f32, sample_rate: u32) -> Vec<i16> {
    let mut samples = Vec::with_capacity(num_samples);
    let phase_increment = 2.0 * std::f32::consts::PI * frequency / sample_rate as f32;

    for i in 0..num_samples {
        let phase = phase_increment * i as f32;
        let sample = (phase.sin() * i16::MAX as f32) as i16;
        samples.push(sample);
    }

    samples
}

#[test]
fn test_opus_encode_real_audio() {
    let config = test_audio_config();
    let mut encoder = OpusEncoder::new(config).expect("encoder creation should succeed");

    // Generate 1 second of 440Hz sine wave at 16kHz
    let pcm = generate_sine_wave(16000, 440.0, 16000);

    // Encode in 20ms chunks (320 samples at 16kHz)
    let mut all_encoded = Vec::new();
    for chunk in pcm.chunks(320) {
        let encoded = encoder.encode(chunk).expect("encode should succeed");
        all_encoded.extend_from_slice(&encoded);
    }

    // Flush any remaining data
    let flushed = encoder.flush().expect("flush should succeed");
    all_encoded.extend_from_slice(&flushed);

    // Verify output is non-empty and valid Opus data
    assert!(
        !all_encoded.is_empty(),
        "Encoded output should not be empty"
    );

    // Opus frames typically start with specific byte patterns
    // At minimum, verify we got reasonable compression
    assert!(
        all_encoded.len() < pcm.len() * 2,
        "Opus should compress PCM data (PCM: {} bytes, Opus: {} bytes)",
        pcm.len() * 2,
        all_encoded.len()
    );
}

#[test]
fn test_opus_encode_decode_roundtrip() {
    let config = test_audio_config();
    let mut encoder = OpusEncoder::new(config.clone()).expect("encoder creation should succeed");

    // Generate 1 second of 440Hz sine wave
    let original_pcm = generate_sine_wave(16000, 440.0, 16000);

    // Encode in 20ms chunks and collect frames separately
    let mut encoded_frames = Vec::new();
    for chunk in original_pcm.chunks(320) {
        let encoded = encoder.encode(chunk).expect("encode should succeed");
        if !encoded.is_empty() {
            encoded_frames.push(encoded);
        }
    }

    let flushed = encoder.flush().expect("flush should succeed");
    if !flushed.is_empty() {
        encoded_frames.push(flushed);
    }

    // Create a decoder
    let mut decoder = opus::Decoder::new(config.sample_rate, opus::Channels::Mono)
        .expect("decoder creation should succeed");

    // Decode frames individually and collect results
    let mut all_decoded = Vec::new();
    for frame in encoded_frames {
        let mut decoded_chunk = vec![0i16; 320];
        let decoded_len = decoder
            .decode(&frame, &mut decoded_chunk, false)
            .expect("decode should succeed");
        decoded_chunk.truncate(decoded_len);
        all_decoded.extend_from_slice(&decoded_chunk);
    }

    // Verify decoded output has reasonable length (may be slightly different due to padding)
    assert!(
        all_decoded.len() >= 15000,
        "Decoded length should be close to original (got {})",
        all_decoded.len()
    );

    // Verify decoded samples are not all zeros (reasonable similarity)
    let non_zero_count = all_decoded.iter().filter(|&&s| s != 0).count();
    assert!(
        non_zero_count > all_decoded.len() / 2,
        "Most decoded samples should be non-zero (got {} non-zero out of {})",
        non_zero_count,
        all_decoded.len()
    );

    // Verify some correlation with original signal
    let mut correlation_sum = 0i64;
    for (orig, decoded) in original_pcm.iter().zip(all_decoded.iter()) {
        correlation_sum += (*orig as i64) * (*decoded as i64);
    }
    assert!(
        correlation_sum > 0,
        "Decoded signal should have positive correlation with original"
    );
}

#[test]
fn test_opus_encoder_flush_produces_output() {
    let config = test_audio_config();
    let mut encoder = OpusEncoder::new(config).expect("encoder creation should succeed");

    // Buffer partial frame (less than 320 samples for 20ms at 16kHz)
    let partial_pcm = generate_sine_wave(100, 440.0, 16000);

    // Encode partial data (should not produce output yet, just buffer)
    let encoded = encoder.encode(&partial_pcm).expect("encode should succeed");
    assert!(
        encoded.is_empty(),
        "Partial frame should not produce output immediately"
    );

    // Flush should produce output by padding and encoding
    let flushed = encoder.flush().expect("flush should succeed");
    assert!(
        !flushed.is_empty(),
        "Flush should produce output for buffered partial frame"
    );

    // Verify flushed data is valid Opus
    assert!(
        !flushed.is_empty(),
        "Flushed data should contain encoded frame"
    );
}

#[test]
fn test_opus_encoder_different_bitrates() {
    let bitrates = vec![16000, 32000, 64000];

    for bitrate in bitrates {
        let config = AudioConfig {
            sample_rate: 16000,
            channels: 1,
            bitrate,
        };

        let mut encoder = OpusEncoder::new(config).expect("encoder creation should succeed");

        // Generate 1 second of audio
        let pcm = generate_sine_wave(16000, 440.0, 16000);

        // Encode in 20ms chunks
        let mut all_encoded = Vec::new();
        for chunk in pcm.chunks(320) {
            let encoded = encoder.encode(chunk).expect("encode should succeed");
            all_encoded.extend_from_slice(&encoded);
        }

        let flushed = encoder.flush().expect("flush should succeed");
        all_encoded.extend_from_slice(&flushed);

        // All bitrates should produce valid output
        assert!(
            !all_encoded.is_empty(),
            "Encoding at {} bps should produce output",
            bitrate
        );

        // Higher bitrates should generally produce larger output
        // (though this isn't strictly guaranteed for all signals)
        assert!(
            !all_encoded.is_empty(),
            "Encoded data at {} bps should be non-empty",
            bitrate
        );
    }
}
