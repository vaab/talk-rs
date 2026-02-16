//! Audio container writers for producing valid OGG/Opus and WAV output.
//!
//! Provides the [`AudioWriter`] trait and two implementations:
//! - [`OggOpusWriter`]: encodes PCM to Opus and wraps in OGG container
//! - [`WavWriter`]: wraps raw PCM in a WAV container

use byteorder::{LittleEndian, WriteBytesExt};
use ogg::{PacketWriteEndInfo, PacketWriter};

use crate::core::config::AudioConfig;
use crate::core::error::TalkError;

/// Trait for audio container writers.
///
/// Takes raw PCM i16 samples as input and produces containerized audio bytes.
/// Each implementation handles its own encoding (if any) and container format.
pub trait AudioWriter: Send {
    /// Return the header/preamble bytes for this container format.
    /// Must be called once before any `write_pcm()` calls.
    fn header(&mut self) -> Result<Vec<u8>, TalkError>;

    /// Write PCM samples and return containerized output bytes.
    /// May return empty Vec if buffering internally (e.g., collecting a full Opus frame).
    fn write_pcm(&mut self, pcm: &[i16]) -> Result<Vec<u8>, TalkError>;

    /// Finalize the container. Returns any trailing bytes (flush, EOS page, etc.).
    /// For WAV: returns updated header with correct data size.
    fn finalize(&mut self) -> Result<Vec<u8>, TalkError>;

    /// MIME type for HTTP uploads.
    fn mime_type(&self) -> &str;

    /// File extension for this format.
    fn extension(&self) -> &str;
}

/// Generate a pseudo-random serial number for OGG streams.
fn rand_serial() -> u32 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(42)
}

/// OGG/Opus container writer.
///
/// Encodes PCM input with Opus and wraps the output in OGG pages
/// conforming to RFC 7845 (OpusHead + OpusTags headers).
pub struct OggOpusWriter {
    /// Ogg page writer, wrapping a Vec<u8> buffer.
    packet_writer: PacketWriter<'static, Vec<u8>>,
    /// Opus encoder.
    encoder: opus::Encoder,
    /// PCM sample buffer (accumulates until a full Opus frame).
    pcm_buffer: Vec<i16>,
    /// Frame size in samples (per channel) for 20ms at the configured sample rate.
    frame_size: usize,
    /// Number of channels.
    channels: u8,
    /// Stream serial number.
    serial: u32,
    /// Total granule position (in 48kHz samples, per RFC 7845).
    granule_position: u64,
    /// Sample rate ratio for granule calculation (48000 / sample_rate).
    granule_rate_ratio: f64,
    /// Whether header has been written.
    header_written: bool,
}

// OggOpusWriter is Send because opus::Encoder is Send and PacketWriter<Vec<u8>> is Send
unsafe impl Send for OggOpusWriter {}

impl OggOpusWriter {
    /// Create a new OGG/Opus writer with the given audio configuration.
    pub fn new(config: AudioConfig) -> Result<Self, TalkError> {
        let opus_channels = match config.channels {
            1 => opus::Channels::Mono,
            2 => opus::Channels::Stereo,
            _ => {
                return Err(TalkError::Audio(format!(
                    "Unsupported channel count: {}",
                    config.channels
                )))
            }
        };

        let mut encoder =
            opus::Encoder::new(config.sample_rate, opus_channels, opus::Application::Voip)
                .map_err(|e| TalkError::Audio(format!("Failed to create Opus encoder: {e}")))?;

        encoder
            .set_bitrate(opus::Bitrate::Bits(config.bitrate as i32))
            .map_err(|e| TalkError::Audio(format!("Failed to set Opus bitrate: {e}")))?;

        let frame_size = (config.sample_rate as usize * 20) / 1000; // 20ms frames
        let serial = rand_serial();

        Ok(Self {
            packet_writer: PacketWriter::new(Vec::new()),
            encoder,
            pcm_buffer: Vec::new(),
            frame_size,
            channels: config.channels,
            serial,
            granule_position: 0,
            granule_rate_ratio: 48000.0 / config.sample_rate as f64,
            header_written: false,
        })
    }
}

impl AudioWriter for OggOpusWriter {
    fn header(&mut self) -> Result<Vec<u8>, TalkError> {
        // Build OpusHead packet (19 bytes for mono/stereo, mapping family 0)
        let mut head = Vec::with_capacity(19);
        head.extend_from_slice(b"OpusHead");
        head.push(1); // version
        head.push(self.channels); // channel count
                                  // Pre-skip: standard value for Opus
        head.write_u16::<LittleEndian>(3840).unwrap(); // pre-skip (infallible on Vec)
        head.write_u32::<LittleEndian>(48000).unwrap(); // input sample rate (always 48kHz for Opus)
        head.write_i16::<LittleEndian>(0).unwrap(); // output gain
        head.push(0); // mapping family 0

        self.packet_writer
            .write_packet(head, self.serial, PacketWriteEndInfo::EndPage, 0)
            .map_err(|e| TalkError::Audio(format!("failed to write OpusHead: {e}")))?;

        // Build OpusTags packet
        let mut tags = Vec::new();
        tags.extend_from_slice(b"OpusTags");
        let vendor = b"talk-rs";
        tags.write_u32::<LittleEndian>(vendor.len() as u32).unwrap(); // infallible on Vec
        tags.extend_from_slice(vendor);
        tags.write_u32::<LittleEndian>(0).unwrap(); // 0 user comments (infallible on Vec)

        self.packet_writer
            .write_packet(tags, self.serial, PacketWriteEndInfo::EndPage, 0)
            .map_err(|e| TalkError::Audio(format!("failed to write OpusTags: {e}")))?;

        self.header_written = true;

        // Extract the bytes written so far
        let bytes = self.packet_writer.inner_mut().clone();
        self.packet_writer.inner_mut().clear();
        Ok(bytes)
    }

    fn write_pcm(&mut self, pcm: &[i16]) -> Result<Vec<u8>, TalkError> {
        self.pcm_buffer.extend_from_slice(pcm);

        let samples_per_frame = self.frame_size * self.channels as usize;
        let mut output = Vec::new();

        while self.pcm_buffer.len() >= samples_per_frame {
            let frame: Vec<i16> = self.pcm_buffer.drain(..samples_per_frame).collect();

            // Encode with Opus
            let mut opus_output = vec![0u8; 4000];
            let len = self
                .encoder
                .encode(&frame, &mut opus_output)
                .map_err(|e| TalkError::Audio(format!("Opus encoding failed: {e}")))?;
            opus_output.truncate(len);

            // Update granule position (in 48kHz samples)
            self.granule_position += (self.frame_size as f64 * self.granule_rate_ratio) as u64;

            // Write as OGG packet (each Opus frame gets its own page for streaming)
            self.packet_writer
                .write_packet(
                    opus_output,
                    self.serial,
                    PacketWriteEndInfo::EndPage,
                    self.granule_position,
                )
                .map_err(|e| TalkError::Audio(format!("failed to write OGG page: {e}")))?;

            // Extract bytes
            output.extend_from_slice(self.packet_writer.inner_mut());
            self.packet_writer.inner_mut().clear();
        }

        Ok(output)
    }

    fn finalize(&mut self) -> Result<Vec<u8>, TalkError> {
        let mut output = Vec::new();

        // Pad remaining buffer and encode final frame
        if !self.pcm_buffer.is_empty() {
            let samples_per_frame = self.frame_size * self.channels as usize;
            while self.pcm_buffer.len() < samples_per_frame {
                self.pcm_buffer.push(0i16);
            }

            let frame = self.pcm_buffer.drain(..).collect::<Vec<i16>>();
            let mut opus_output = vec![0u8; 4000];
            let len = self
                .encoder
                .encode(&frame, &mut opus_output)
                .map_err(|e| TalkError::Audio(format!("Opus flush failed: {e}")))?;
            opus_output.truncate(len);

            self.granule_position += (self.frame_size as f64 * self.granule_rate_ratio) as u64;

            // Write as final OGG packet with EndStream flag
            self.packet_writer
                .write_packet(
                    opus_output,
                    self.serial,
                    PacketWriteEndInfo::EndStream,
                    self.granule_position,
                )
                .map_err(|e| TalkError::Audio(format!("failed to write final OGG page: {e}")))?;

            output.extend_from_slice(self.packet_writer.inner_mut());
            self.packet_writer.inner_mut().clear();
        }

        Ok(output)
    }

    fn mime_type(&self) -> &str {
        "audio/ogg"
    }

    fn extension(&self) -> &str {
        "ogg"
    }
}

/// WAV container writer.
///
/// Wraps raw PCM i16 samples in a standard 44-byte WAV header.
/// Uses a placeholder size on initial header; `finalize()` returns a
/// corrected header with actual data size.
pub struct WavWriter {
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    data_size: u32,
    header_written: bool,
}

impl WavWriter {
    /// Create a new WAV writer with the given audio configuration.
    pub fn new(config: AudioConfig) -> Self {
        Self {
            sample_rate: config.sample_rate,
            channels: config.channels as u16,
            bits_per_sample: 16,
            data_size: 0,
            header_written: false,
        }
    }

    /// Build a 44-byte WAV header with the given data size.
    fn build_header(&self, data_size: u32) -> Result<Vec<u8>, TalkError> {
        let byte_rate = self.sample_rate * self.channels as u32 * (self.bits_per_sample as u32 / 8);
        let block_align = self.channels * (self.bits_per_sample / 8);

        let mut hdr = Vec::with_capacity(44);
        hdr.extend_from_slice(b"RIFF");
        hdr.write_u32::<LittleEndian>(if data_size == 0xFFFF_FFFF {
            0xFFFF_FFFF
        } else {
            36 + data_size
        })?;
        hdr.extend_from_slice(b"WAVE");
        hdr.extend_from_slice(b"fmt ");
        hdr.write_u32::<LittleEndian>(16)?; // fmt chunk size
        hdr.write_u16::<LittleEndian>(1)?; // PCM format
        hdr.write_u16::<LittleEndian>(self.channels)?;
        hdr.write_u32::<LittleEndian>(self.sample_rate)?;
        hdr.write_u32::<LittleEndian>(byte_rate)?;
        hdr.write_u16::<LittleEndian>(block_align)?;
        hdr.write_u16::<LittleEndian>(self.bits_per_sample)?;
        hdr.extend_from_slice(b"data");
        hdr.write_u32::<LittleEndian>(data_size)?;

        Ok(hdr)
    }
}

impl AudioWriter for WavWriter {
    fn header(&mut self) -> Result<Vec<u8>, TalkError> {
        self.header_written = true;
        // Use placeholder sizes for streaming
        self.build_header(0xFFFF_FFFF)
    }

    fn write_pcm(&mut self, pcm: &[i16]) -> Result<Vec<u8>, TalkError> {
        let mut bytes = Vec::with_capacity(pcm.len() * 2);
        for &sample in pcm {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        self.data_size += bytes.len() as u32;
        Ok(bytes)
    }

    fn finalize(&mut self) -> Result<Vec<u8>, TalkError> {
        // Return a corrected 44-byte header with the actual sizes.
        // The caller is responsible for seeking to offset 0 and writing this.
        self.build_header(self.data_size)
    }

    fn mime_type(&self) -> &str {
        "audio/wav"
    }

    fn extension(&self) -> &str {
        "wav"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::AudioConfig;

    fn test_config() -> AudioConfig {
        AudioConfig {
            sample_rate: 16_000,
            channels: 1,
            bitrate: 32_000,
        }
    }

    #[test]
    fn test_ogg_opus_writer_produces_valid_ogg() {
        let mut writer = OggOpusWriter::new(test_config()).unwrap();
        let header = writer.header().unwrap();

        // Must start with OggS magic
        assert_eq!(&header[0..4], b"OggS", "output must start with OggS");
        // Must contain OpusHead somewhere in first pages
        assert!(
            header.windows(8).any(|w| w == b"OpusHead"),
            "output must contain OpusHead"
        );
        assert!(
            header.windows(8).any(|w| w == b"OpusTags"),
            "output must contain OpusTags"
        );

        // Write some PCM data
        let pcm: Vec<i16> = (0..320)
            .map(|i| ((i as f32 * 0.1).sin() * 10000.0) as i16)
            .collect();
        let audio_bytes = writer.write_pcm(&pcm).unwrap();

        // Finalize
        let final_bytes = writer.finalize().unwrap();

        // Combined output should be parseable
        let all_bytes = [header, audio_bytes, final_bytes].concat();
        assert!(
            all_bytes.len() > 100,
            "output should have substantial content"
        );
        assert_eq!(&all_bytes[0..4], b"OggS");
    }

    #[test]
    fn test_wav_writer_produces_valid_wav() {
        let mut writer = WavWriter::new(test_config());
        let header = writer.header().unwrap();

        assert_eq!(&header[0..4], b"RIFF", "must start with RIFF");
        assert_eq!(&header[8..12], b"WAVE", "must contain WAVE");
        assert_eq!(&header[12..16], b"fmt ", "must contain fmt chunk");
        assert_eq!(header.len(), 44, "WAV header must be 44 bytes");

        // Write some PCM
        let pcm = vec![100i16, -200, 300];
        let audio_bytes = writer.write_pcm(&pcm).unwrap();
        assert_eq!(audio_bytes.len(), 6); // 3 samples * 2 bytes

        // Finalize returns corrected header
        let final_header = writer.finalize().unwrap();
        assert_eq!(final_header.len(), 44);
        assert_eq!(&final_header[0..4], b"RIFF");
    }

    #[test]
    fn test_ogg_opus_writer_mime_type() {
        let writer = OggOpusWriter::new(test_config()).unwrap();
        assert_eq!(writer.mime_type(), "audio/ogg");
        assert_eq!(writer.extension(), "ogg");
    }

    #[test]
    fn test_wav_writer_mime_type() {
        let writer = WavWriter::new(test_config());
        assert_eq!(writer.mime_type(), "audio/wav");
        assert_eq!(writer.extension(), "wav");
    }
}
