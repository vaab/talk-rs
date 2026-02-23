//! Audio decoding and resampling utilities for the recordings browser.

use crate::core::error::TalkError;

/// Opus always uses 48 kHz internally (RFC 7845).
pub(super) const OPUS_SAMPLE_RATE: u32 = 48_000;

/// Compute WAV duration in seconds from file size.
///
/// Assumes 16-bit mono 16 kHz PCM with a 44-byte header:
/// `(file_size - 44) / (16000 * 2)` = seconds.
pub(super) fn wav_duration_secs(path: &std::path::Path) -> Option<f64> {
    let size = std::fs::metadata(path).ok()?.len();
    if size <= 44 {
        return None;
    }
    Some((size - 44) as f64 / 32_000.0)
}

/// Compute OGG Opus duration in seconds from the last page's granule position.
///
/// Iterates all packets to find the last absolute granule position,
/// then divides by 48 000 (Opus always uses 48 kHz per RFC 7845).
pub(super) fn ogg_duration_secs(path: &std::path::Path) -> Option<f64> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = ogg::reading::PacketReader::new(std::io::BufReader::new(file));
    let mut last_absgp: u64 = 0;
    loop {
        match reader.read_packet() {
            Ok(Some(pkt)) => {
                last_absgp = pkt.absgp_page();
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    if last_absgp == 0 {
        return None;
    }
    Some(last_absgp as f64 / OPUS_SAMPLE_RATE as f64)
}

/// Read a WAV file and return mono `f32` samples resampled to `target_rate`.
///
/// Supports 16-bit PCM WAV files (mono or stereo). Stereo is averaged
/// to mono. If the WAV sample rate differs from `target_rate`, linear
/// interpolation is used to resample.
pub(super) fn read_wav_as_f32(
    path: &std::path::Path,
    target_rate: u32,
) -> Result<Vec<f32>, TalkError> {
    let data = std::fs::read(path)
        .map_err(|e| TalkError::Config(format!("failed to read WAV {}: {}", path.display(), e)))?;

    if data.len() < 44 {
        return Err(TalkError::Config(format!(
            "WAV file too small ({}B): {}",
            data.len(),
            path.display()
        )));
    }

    let channels = u16::from_le_bytes([data[22], data[23]]) as usize;
    let wav_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);

    if bits_per_sample != 16 {
        return Err(TalkError::Config(format!(
            "unsupported WAV format: {}-bit (only 16-bit PCM supported)",
            bits_per_sample
        )));
    }

    // Read i16 frames → mono f32
    let sample_data = &data[44..];
    let bytes_per_frame = 2 * channels; // 2 bytes per i16 × channels
    let num_frames = sample_data.len() / bytes_per_frame;
    let mut mono = Vec::with_capacity(num_frames);

    for frame in 0..num_frames {
        let offset = frame * bytes_per_frame;
        let mut sum: f32 = 0.0;
        for ch in 0..channels {
            let i = offset + ch * 2;
            if i + 1 < sample_data.len() {
                let s = i16::from_le_bytes([sample_data[i], sample_data[i + 1]]);
                sum += s as f32 / 32_768.0;
            }
        }
        mono.push(sum / channels as f32);
    }

    // Resample if rates differ
    if wav_rate == target_rate || wav_rate == 0 {
        return Ok(mono);
    }
    resample_linear(&mono, wav_rate, target_rate)
}

/// Read an OGG Opus file and return mono `f32` samples resampled to `target_rate`.
///
/// Decodes using the `ogg` crate for demuxing and the `opus` crate for
/// Opus decoding.  Stereo is averaged to mono.  If the device sample rate
/// differs from 48 kHz, linear interpolation resamples the output.
pub(super) fn read_ogg_as_f32(
    path: &std::path::Path,
    target_rate: u32,
) -> Result<Vec<f32>, TalkError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TalkError::Config(format!("failed to open OGG {}: {}", path.display(), e)))?;
    let mut reader = ogg::reading::PacketReader::new(std::io::BufReader::new(file));

    // Read the first packet (OpusHead header) to determine channel count
    let head_pkt = reader
        .read_packet()
        .map_err(|e| TalkError::Config(format!("failed to read OGG header: {}", e)))?
        .ok_or_else(|| TalkError::Config("OGG file has no packets".to_string()))?;

    // OpusHead: bytes 0..7 = "OpusHead", byte 9 = channel count
    let head_data = &head_pkt.data;
    if head_data.len() < 19 || &head_data[..8] != b"OpusHead" {
        return Err(TalkError::Config(format!(
            "invalid OpusHead in {}",
            path.display()
        )));
    }
    let channel_count = head_data[9] as usize;
    let opus_channels = if channel_count >= 2 {
        opus::Channels::Stereo
    } else {
        opus::Channels::Mono
    };

    let mut decoder = opus::Decoder::new(OPUS_SAMPLE_RATE, opus_channels)
        .map_err(|e| TalkError::Config(format!("failed to create Opus decoder: {}", e)))?;

    // Skip the OpusTags packet (second packet)
    let _ = reader.read_packet();

    // Decode all remaining audio packets
    // Max Opus frame: 120ms at 48kHz = 5760 samples/channel
    let max_frame_samples = 5760 * channel_count;
    let mut decode_buf = vec![0.0f32; max_frame_samples];
    let mut all_mono = Vec::new();

    loop {
        match reader.read_packet() {
            Ok(Some(pkt)) => {
                let samples_per_channel =
                    decoder
                        .decode_float(&pkt.data, &mut decode_buf, false)
                        .map_err(|e| TalkError::Config(format!("Opus decode error: {}", e)))?;

                // Convert to mono
                for i in 0..samples_per_channel {
                    if channel_count >= 2 {
                        let mut sum: f32 = 0.0;
                        for ch in 0..channel_count {
                            sum += decode_buf[i * channel_count + ch];
                        }
                        all_mono.push(sum / channel_count as f32);
                    } else {
                        all_mono.push(decode_buf[i]);
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                log::warn!("OGG read error (continuing): {}", e);
                break;
            }
        }
    }

    // Resample from 48kHz to target_rate if needed
    if OPUS_SAMPLE_RATE == target_rate {
        return Ok(all_mono);
    }
    resample_linear(&all_mono, OPUS_SAMPLE_RATE, target_rate)
}

/// Resample mono f32 samples using linear interpolation.
pub(super) fn resample_linear(
    mono: &[f32],
    src_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>, TalkError> {
    if src_rate == 0 {
        return Ok(mono.to_vec());
    }
    let ratio = target_rate as f64 / src_rate as f64;
    let new_len = (mono.len() as f64 * ratio).ceil() as usize;
    let mut resampled = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let s0 = mono.get(idx).copied().unwrap_or(0.0);
        let s1 = mono.get(idx + 1).copied().unwrap_or(s0);
        resampled.push(s0 + (s1 - s0) * frac);
    }
    Ok(resampled)
}
