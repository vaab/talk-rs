//! Audio decoding and resampling utilities for the recordings browser.

use crate::error::TalkError;

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

/// Compute OGG Opus duration in seconds from the last page's granule
/// position.
///
/// Seeks to near the end of the file and scans backward for the last
/// OGG page header (`OggS`), reading the absolute granule position
/// from it.  This is O(1) regardless of file size — only the last
/// ~64 KB are read, instead of iterating every packet sequentially.
pub(super) fn ogg_duration_secs(path: &std::path::Path) -> Option<f64> {
    use std::io::{Read, Seek, SeekFrom};

    /// Size of the tail buffer.  OGG pages are at most ~65 535 bytes
    /// (255 segments × 255 bytes each), so this guarantees at least
    /// one complete page in the buffer.
    const TAIL_SIZE: u64 = 65_536;

    /// Minimum OGG page header size (capture pattern through the
    /// segment-count byte, inclusive).
    const MIN_HEADER: usize = 27;

    /// Offset of the 8-byte absolute granule position within an OGG
    /// page header (immediately after the 4-byte magic and 2 flag
    /// bytes).
    const GRANULE_OFFSET: usize = 6;

    let mut file = std::fs::File::open(path).ok()?;
    let file_size = file.metadata().ok()?.len();
    if file_size < MIN_HEADER as u64 {
        return None;
    }

    // Read the tail of the file.
    let start = file_size.saturating_sub(TAIL_SIZE);
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut buf = Vec::with_capacity((file_size - start) as usize);
    file.read_to_end(&mut buf).ok()?;

    // Scan backward for the last OGG page capture pattern.
    // Verify the version byte (must be 0) to reduce false positives.
    let last_page = (0..buf.len().saturating_sub(MIN_HEADER))
        .rev()
        .find(|&i| buf[i..i + 4] == *b"OggS" && buf[i + 4] == 0)?;

    let granule = u64::from_le_bytes(
        buf[last_page + GRANULE_OFFSET..last_page + GRANULE_OFFSET + 8]
            .try_into()
            .ok()?,
    );
    if granule == 0 {
        return None;
    }
    Some(granule as f64 / OPUS_SAMPLE_RATE as f64)
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

/// Return the `.wf` cache path for a given audio file.
fn waterfall_cache_path(audio_path: &std::path::Path) -> std::path::PathBuf {
    audio_path.with_extension("wf")
}

/// Write waterfall data to a binary `.wf` cache file.
///
/// Format (all little-endian):
///   u32  num_columns
///   u32  num_rows
///   f32  peak
///   f32 × (num_columns × num_rows)  column-major data
pub(super) fn write_waterfall_cache(
    audio_path: &std::path::Path,
    columns: &[Vec<f32>],
    peak: f32,
) -> Result<(), TalkError> {
    use std::io::Write;

    let cache = waterfall_cache_path(audio_path);
    let num_cols = columns.len() as u32;
    let num_rows = columns.first().map_or(0, |c| c.len()) as u32;

    let mut buf = Vec::with_capacity(12 + (num_cols * num_rows * 4) as usize);
    buf.write_all(&num_cols.to_le_bytes())
        .map_err(|e| TalkError::Config(format!("wf cache write: {e}")))?;
    buf.write_all(&num_rows.to_le_bytes())
        .map_err(|e| TalkError::Config(format!("wf cache write: {e}")))?;
    buf.write_all(&peak.to_le_bytes())
        .map_err(|e| TalkError::Config(format!("wf cache write: {e}")))?;
    for col in columns {
        for &val in col {
            buf.write_all(&val.to_le_bytes())
                .map_err(|e| TalkError::Config(format!("wf cache write: {e}")))?;
        }
    }
    std::fs::write(&cache, &buf)
        .map_err(|e| TalkError::Config(format!("wf cache write {}: {e}", cache.display())))?;
    Ok(())
}

/// Read waterfall data from a `.wf` cache file, if it exists and is
/// newer than the audio file.
///
/// Returns `None` if the cache is missing, stale, or corrupt.
pub(super) fn read_waterfall_cache(audio_path: &std::path::Path) -> Option<(Vec<Vec<f32>>, f32)> {
    let cache = waterfall_cache_path(audio_path);
    let cache_meta = std::fs::metadata(&cache).ok()?;
    let audio_meta = std::fs::metadata(audio_path).ok()?;

    // Stale check: cache must be newer than the audio file.
    if cache_meta.modified().ok()? < audio_meta.modified().ok()? {
        return None;
    }

    let data = std::fs::read(&cache).ok()?;
    if data.len() < 12 {
        return None;
    }

    let num_cols = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    let num_rows = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
    let peak = f32::from_le_bytes(data[8..12].try_into().ok()?);

    let expected = 12 + num_cols * num_rows * 4;
    if data.len() < expected || num_cols == 0 || num_rows == 0 {
        return None;
    }

    let mut columns = Vec::with_capacity(num_cols);
    let mut offset = 12;
    for _ in 0..num_cols {
        let mut col = Vec::with_capacity(num_rows);
        for _ in 0..num_rows {
            col.push(f32::from_le_bytes(
                data[offset..offset + 4].try_into().ok()?,
            ));
            offset += 4;
        }
        columns.push(col);
    }

    Some((columns, peak))
}

/// Read any supported audio file (WAV or OGG Opus) as mono 16-bit PCM
/// at 16 kHz, suitable for [`crate::x11::render_util::generate_waterfall_columns`].
pub(super) fn read_audio_as_i16(path: &std::path::Path) -> Result<Vec<i16>, TalkError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let samples_f32 = match ext.as_str() {
        "wav" => read_wav_as_f32(path, 16_000)?,
        "ogg" | "opus" => read_ogg_as_f32(path, 16_000)?,
        _ => {
            return Err(TalkError::Config(format!(
                "unsupported audio format: {}",
                path.display()
            )))
        }
    };
    Ok(samples_f32
        .iter()
        .map(|&s| (s * 32_767.0).clamp(-32_768.0, 32_767.0) as i16)
        .collect())
}
