//! Audio encoding interfaces and implementations.

use crate::core::config::AudioConfig;
use crate::core::error::TalkError;

/// Trait for audio encoding from PCM samples to compressed formats.
pub trait AudioEncoder: Send {
    /// Encode a chunk of PCM samples.
    ///
    /// Takes a slice of i16 PCM samples and returns encoded bytes.
    /// The encoder may buffer samples internally for optimal compression.
    fn encode(&mut self, pcm: &[i16]) -> Result<Vec<u8>, TalkError>;

    /// Flush any remaining buffered samples and finalize encoding.
    ///
    /// Must be called after the last encode() to ensure all data is output.
    fn flush(&mut self) -> Result<Vec<u8>, TalkError>;
}

/// Mock encoder for testing that passes through PCM data.
pub struct MockEncoder {
    /// Configuration kept for potential future use (e.g., validation, logging).
    #[allow(dead_code)]
    config: AudioConfig,
    buffer: Vec<u8>,
}

impl MockEncoder {
    /// Create a new mock encoder with the given configuration.
    pub fn new(config: AudioConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
        }
    }
}

impl AudioEncoder for MockEncoder {
    fn encode(&mut self, pcm: &[i16]) -> Result<Vec<u8>, TalkError> {
        // Convert i16 samples to bytes (little-endian)
        let mut encoded = Vec::with_capacity(pcm.len() * 2);
        for &sample in pcm {
            encoded.extend_from_slice(&sample.to_le_bytes());
        }
        self.buffer.extend_from_slice(&encoded);
        Ok(encoded)
    }

    fn flush(&mut self) -> Result<Vec<u8>, TalkError> {
        let result = self.buffer.clone();
        self.buffer.clear();
        Ok(result)
    }
}

/// Opus encoder implementation using the opus crate.
#[derive(Debug)]
pub struct OpusEncoder {
    config: AudioConfig,
    encoder: opus::Encoder,
    buffer: Vec<i16>,
}

impl OpusEncoder {
    /// Create a new Opus encoder with the given configuration.
    pub fn new(config: AudioConfig) -> Result<Self, TalkError> {
        let mut encoder = opus::Encoder::new(
            config.sample_rate,
            match config.channels {
                1 => opus::Channels::Mono,
                2 => opus::Channels::Stereo,
                _ => {
                    return Err(TalkError::Audio(format!(
                        "Unsupported channel count: {}",
                        config.channels
                    )))
                }
            },
            opus::Application::Voip,
        )
        .map_err(|err| TalkError::Audio(format!("Failed to create Opus encoder: {}", err)))?;

        // Set bitrate from config
        encoder
            .set_bitrate(opus::Bitrate::Bits(config.bitrate as i32))
            .map_err(|err| TalkError::Audio(format!("Failed to set Opus bitrate: {}", err)))?;

        Ok(Self {
            config,
            encoder,
            buffer: Vec::new(),
        })
    }
}

// OpusEncoder is Send because opus::Encoder is Send
unsafe impl Send for OpusEncoder {}

impl AudioEncoder for OpusEncoder {
    fn encode(&mut self, pcm: &[i16]) -> Result<Vec<u8>, TalkError> {
        // Opus expects frames of specific sizes. For Voip application,
        // valid frame sizes are 10, 20, 40, or 60ms.
        // We'll encode in 20ms chunks as per CHUNK_DURATION_MS.
        let frame_size = (self.config.sample_rate as usize * 20) / 1000;
        let samples_per_frame = frame_size * self.config.channels as usize;

        let mut encoded = Vec::new();

        // Add new samples to buffer
        self.buffer.extend_from_slice(pcm);

        // Encode complete frames
        while self.buffer.len() >= samples_per_frame {
            let frame: Vec<i16> = self.buffer.drain(..samples_per_frame).collect();

            let mut output = vec![0u8; 4000]; // Max Opus frame size
            let len = self
                .encoder
                .encode(&frame, &mut output)
                .map_err(|err| TalkError::Audio(format!("Opus encoding failed: {}", err)))?;

            output.truncate(len);
            encoded.extend_from_slice(&output);
        }

        Ok(encoded)
    }

    fn flush(&mut self) -> Result<Vec<u8>, TalkError> {
        let mut encoded = Vec::new();

        // Encode any remaining samples in buffer by padding with zeros
        if !self.buffer.is_empty() {
            let frame_size = (self.config.sample_rate as usize * 20) / 1000;
            let samples_per_frame = frame_size * self.config.channels as usize;

            // Pad with zeros to complete the frame
            while self.buffer.len() < samples_per_frame {
                self.buffer.push(0i16);
            }

            let frame = self.buffer.drain(..).collect::<Vec<i16>>();

            let mut output = vec![0u8; 4000];
            let len = self
                .encoder
                .encode(&frame, &mut output)
                .map_err(|err| TalkError::Audio(format!("Opus flush failed: {}", err)))?;

            output.truncate(len);
            encoded.extend_from_slice(&output);
        }

        Ok(encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AudioConfig {
        AudioConfig::new()
    }

    #[test]
    fn test_mock_encoder_encode() {
        let config = test_config();
        let mut encoder = MockEncoder::new(config);

        let pcm = vec![100i16, 200i16, 300i16];
        let encoded = encoder.encode(&pcm).expect("encode should succeed");

        // Should return 6 bytes (3 samples * 2 bytes each)
        assert_eq!(encoded.len(), 6);
        assert_eq!(encoded[0..2], 100i16.to_le_bytes());
        assert_eq!(encoded[2..4], 200i16.to_le_bytes());
        assert_eq!(encoded[4..6], 300i16.to_le_bytes());
    }

    #[test]
    fn test_mock_encoder_flush() {
        let config = test_config();
        let mut encoder = MockEncoder::new(config);

        let pcm = vec![100i16, 200i16];
        encoder.encode(&pcm).expect("encode should succeed");

        let flushed = encoder.flush().expect("flush should succeed");
        assert_eq!(flushed.len(), 4);

        // After flush, buffer should be empty
        let flushed_again = encoder.flush().expect("flush should succeed");
        assert_eq!(flushed_again.len(), 0);
    }

    #[test]
    fn test_opus_encoder_creation() {
        let config = test_config();
        let encoder = OpusEncoder::new(config).expect("creation should succeed");
        assert_eq!(encoder.config.sample_rate, 16_000);
        assert_eq!(encoder.config.channels, 1);
    }

    #[test]
    fn test_opus_encoder_mono_encode() {
        let config = test_config();
        let mut encoder = OpusEncoder::new(config).expect("creation should succeed");

        // Create a 20ms chunk of samples (320 samples at 16kHz)
        let pcm: Vec<i16> = (0..320).map(|i| (i as i16).wrapping_mul(100)).collect();
        let encoded = encoder.encode(&pcm).expect("encode should succeed");

        // Opus should produce some output
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_opus_encoder_stereo_encode() {
        let config = AudioConfig {
            sample_rate: 16_000,
            channels: 2,
            bitrate: 64_000,
        };
        let mut encoder = OpusEncoder::new(config).expect("creation should succeed");

        // Create a 20ms chunk of stereo samples (640 samples at 16kHz, 2 channels)
        let pcm: Vec<i16> = (0..640).map(|i| (i as i16).wrapping_mul(100)).collect();
        let encoded = encoder.encode(&pcm).expect("encode should succeed");

        // Opus should produce some output
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_opus_encoder_flush() {
        let config = test_config();
        let mut encoder = OpusEncoder::new(config).expect("creation should succeed");

        // Encode partial data
        let pcm: Vec<i16> = (0..100).map(|i| (i as i16).wrapping_mul(50)).collect();
        encoder.encode(&pcm).expect("encode should succeed");

        // Flush should encode remaining data
        let flushed = encoder.flush().expect("flush should succeed");
        assert!(!flushed.is_empty());
    }

    #[test]
    fn test_opus_encoder_invalid_channels() {
        let config = AudioConfig {
            sample_rate: 16_000,
            channels: 5,
            bitrate: 32_000,
        };
        let result = OpusEncoder::new(config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported channel count"));
    }

    #[test]
    fn test_opus_encode_decode_roundtrip() {
        let config = test_config();
        let mut encoder =
            OpusEncoder::new(config.clone()).expect("encoder creation should succeed");

        // Create a simple test signal (sine wave approximation)
        let mut pcm = Vec::new();
        for i in 0..320 {
            let sample = ((i as f32 * 0.1).sin() * i16::MAX as f32) as i16;
            pcm.push(sample);
        }

        let encoded = encoder.encode(&pcm).expect("encode should succeed");
        assert!(!encoded.is_empty());

        // Create a decoder to verify roundtrip
        let mut decoder = opus::Decoder::new(config.sample_rate, opus::Channels::Mono)
            .expect("decoder creation should succeed");

        let mut decoded = vec![0i16; 320];
        let decoded_len = decoder
            .decode(&encoded, &mut decoded, false)
            .expect("decode should succeed");

        assert_eq!(decoded_len, 320);

        // Verify decoded samples are reasonable (not all zeros)
        assert!(decoded.iter().any(|&s| s != 0));
    }
}
