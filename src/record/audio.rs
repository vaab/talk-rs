//! Audio decoding and resampling utilities for the recordings browser.

use crate::error::TalkError;

/// Opus always uses 48 kHz internally (RFC 7845).
pub(super) const OPUS_SAMPLE_RATE: u32 = 48_000;

/// Compute WAV duration in seconds from file size.
///
/// Assumes 16-bit mono 16 kHz PCM with a 44-byte header:
/// `(file_size - 44) / (16000 * 2)` = seconds.
#[cfg(all(test, feature = "ui"))]
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
#[cfg(feature = "ui")]
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

/// Compute MP4/M4A duration in seconds by parsing the `mvhd` (movie
/// header) box.
///
/// MP4 is a box-structured format: each box has a 4-byte big-endian
/// size and a 4-byte ASCII type, followed by payload.  The `mvhd` box
/// lives inside `moov` and carries the file's overall duration as
/// `duration / timescale` seconds.
///
/// This walks the top-level boxes to find `moov`, then walks `moov`'s
/// children to find `mvhd` — O(box-count), no full decode required.
/// The function is tolerant of unknown / extended-size boxes and
/// returns `None` rather than erroring when the structure is
/// unexpected (matching `ogg_duration_secs`' contract).
#[cfg(feature = "ui")]
pub(super) fn m4a_duration_secs(path: &std::path::Path) -> Option<f64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let file_size = file.metadata().ok()?.len();

    // Walk top-level boxes looking for `moov`.
    let mut cursor: u64 = 0;
    while cursor + 8 <= file_size {
        file.seek(SeekFrom::Start(cursor)).ok()?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header).ok()?;
        let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as u64;
        let btype = &header[4..8];
        // Resolve actual box size: 0 = "to end of file", 1 = 64-bit
        // size in the next 8 bytes.
        let (box_size, payload_offset) = match size {
            0 => (file_size - cursor, 8u64),
            1 => {
                let mut ext = [0u8; 8];
                file.read_exact(&mut ext).ok()?;
                (u64::from_be_bytes(ext), 16u64)
            }
            n => (n, 8u64),
        };
        if box_size < payload_offset {
            return None;
        }

        if btype == b"moov" {
            // Search children of moov for `mvhd`.
            let moov_end = cursor + box_size;
            let mut inner = cursor + payload_offset;
            while inner + 8 <= moov_end {
                file.seek(SeekFrom::Start(inner)).ok()?;
                let mut ih = [0u8; 8];
                file.read_exact(&mut ih).ok()?;
                let isize = u32::from_be_bytes([ih[0], ih[1], ih[2], ih[3]]) as u64;
                let itype = &ih[4..8];
                let (ibox_size, ipayload_off) = match isize {
                    0 => (moov_end - inner, 8u64),
                    1 => {
                        let mut ext = [0u8; 8];
                        file.read_exact(&mut ext).ok()?;
                        (u64::from_be_bytes(ext), 16u64)
                    }
                    n => (n, 8u64),
                };
                if ibox_size < ipayload_off {
                    return None;
                }

                if itype == b"mvhd" {
                    // mvhd starts with a 1-byte version + 3-byte flags.
                    // Layout for v0 / v1 differs in field width.
                    file.seek(SeekFrom::Start(inner + ipayload_off)).ok()?;
                    let mut vf = [0u8; 4];
                    file.read_exact(&mut vf).ok()?;
                    let version = vf[0];
                    if version == 0 {
                        // v0: u32 creation, u32 modification, u32 timescale, u32 duration.
                        let mut rest = [0u8; 16];
                        file.read_exact(&mut rest).ok()?;
                        let timescale = u32::from_be_bytes([rest[8], rest[9], rest[10], rest[11]]);
                        let duration = u32::from_be_bytes([rest[12], rest[13], rest[14], rest[15]]);
                        if timescale == 0 {
                            return None;
                        }
                        return Some(duration as f64 / timescale as f64);
                    } else if version == 1 {
                        // v1: u64 creation, u64 modification, u32 timescale, u64 duration.
                        let mut rest = [0u8; 28];
                        file.read_exact(&mut rest).ok()?;
                        let timescale =
                            u32::from_be_bytes([rest[16], rest[17], rest[18], rest[19]]);
                        let duration = u64::from_be_bytes([
                            rest[20], rest[21], rest[22], rest[23], rest[24], rest[25], rest[26],
                            rest[27],
                        ]);
                        if timescale == 0 {
                            return None;
                        }
                        return Some(duration as f64 / timescale as f64);
                    } else {
                        return None;
                    }
                }

                if ibox_size == 0 {
                    return None;
                }
                inner = inner.checked_add(ibox_size)?;
            }
            return None;
        }

        if box_size == 0 {
            return None;
        }
        cursor = cursor.checked_add(box_size)?;
    }
    None
}

/// Read an M4A / MP4 / raw AAC file and return mono `f32` samples
/// resampled to `target_rate`.
///
/// Uses [`symphonia`] for demuxing (ISO/MP4 container) and AAC
/// decoding.  Multichannel input is downmixed to mono by averaging
/// the channels.  If the source sample rate differs from
/// `target_rate`, linear interpolation resamples the output (matching
/// `read_ogg_as_f32` / `read_wav_as_f32`).
pub(super) fn read_m4a_as_f32(
    path: &std::path::Path,
    target_rate: u32,
) -> Result<Vec<f32>, TalkError> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)
        .map_err(|e| TalkError::Config(format!("failed to open M4A {}: {}", path.display(), e)))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| TalkError::Config(format!("M4A probe {}: {}", path.display(), e)))?;
    let mut format = probed.format;

    let track = format.default_track().ok_or_else(|| {
        TalkError::Config(format!("M4A has no default track: {}", path.display()))
    })?;
    let codec_params = track.codec_params.clone();
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| TalkError::Config(format!("M4A decoder {}: {}", path.display(), e)))?;

    let mut mono: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    // The authoritative rate and channel count come from the FIRST
    // successfully decoded buffer's `SignalSpec`, never from
    // `codec_params`.  Symphonia's ISO/MP4 demuxer is known to
    // underreport channels for AAC tracks (e.g. a stereo AAC-LC track
    // surfaces as `codec_params.channels = 1` even though the AAC
    // decoder produces a 2-channel buffer).  Trusting `codec_params`
    // here causes the stereo-to-mono mixdown below to walk an
    // interleaved buffer with the wrong stride, doubling the output
    // sample count and halving playback speed / pitch.  See
    // `read_m4a_as_f32_stereo_decodes_to_correct_duration` for the
    // regression test.
    let mut src_rate: u32 = 0;
    let mut channel_count: usize = 0;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymError::ResetRequired) => break,
            Err(e) => {
                log::warn!("M4A read error (continuing): {}", e);
                break;
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                if sample_buf.is_none() {
                    let spec = *audio_buf.spec();
                    src_rate = spec.rate;
                    channel_count = spec.channels.count().max(1);
                    let duration = audio_buf.capacity() as u64;
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }
                if let Some(buf) = &mut sample_buf {
                    buf.copy_interleaved_ref(audio_buf);
                    let interleaved = buf.samples();
                    if channel_count == 1 {
                        mono.extend_from_slice(interleaved);
                    } else {
                        let frames = interleaved.len() / channel_count;
                        for i in 0..frames {
                            let mut sum = 0.0f32;
                            for ch in 0..channel_count {
                                sum += interleaved[i * channel_count + ch];
                            }
                            mono.push(sum / channel_count as f32);
                        }
                    }
                }
            }
            Err(SymError::DecodeError(e)) => {
                log::warn!("M4A decode error (skipping packet): {}", e);
                continue;
            }
            Err(SymError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => {
                return Err(TalkError::Config(format!(
                    "M4A decode {}: {}",
                    path.display(),
                    e
                )));
            }
        }
    }

    if src_rate == 0 || src_rate == target_rate {
        return Ok(mono);
    }
    resample_linear(&mono, src_rate, target_rate)
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
#[cfg(feature = "ui")]
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
#[cfg(feature = "ui")]
fn write_waterfall_cache(
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
#[cfg(feature = "ui")]
fn read_waterfall_cache(audio_path: &std::path::Path) -> Option<(Vec<Vec<f32>>, f32)> {
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

/// Load waterfall columns for an audio file, using the `.wf` binary
/// cache when available.  On cache miss the columns are computed from
/// the audio and persisted for next time.
#[cfg(feature = "ui")]
pub(crate) fn load_waterfall(
    audio_path: &std::path::Path,
) -> Result<(Vec<Vec<f32>>, f32), TalkError> {
    if let Some(cached) = read_waterfall_cache(audio_path) {
        return Ok(cached);
    }
    let samples = read_audio_as_i16(audio_path)?;
    let result = crate::x11::render_util::generate_waterfall_columns(&samples, 16_000);
    if let Err(e) = write_waterfall_cache(audio_path, &result.0, result.1) {
        log::warn!("waterfall cache write: {}", e);
    }
    Ok(result)
}

/// Read any supported audio file (WAV or OGG Opus) as mono 16-bit PCM
/// at 16 kHz, suitable for [`crate::x11::render_util::generate_waterfall_columns`].
pub(crate) fn read_audio_as_i16(path: &std::path::Path) -> Result<Vec<i16>, TalkError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let samples_f32 = match ext.as_str() {
        "wav" => read_wav_as_f32(path, 16_000)?,
        "ogg" | "opus" => read_ogg_as_f32(path, 16_000)?,
        "m4a" | "mp4" | "aac" => read_m4a_as_f32(path, 16_000)?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Absolute path to the project's `tests/fixtures/` directory.
    ///
    /// Anchored on `CARGO_MANIFEST_DIR` so the tests work regardless of
    /// the current working directory at run time (matches the pattern
    /// used elsewhere in the workspace).
    fn fixture(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests");
        p.push("fixtures");
        p.push(name);
        p
    }

    /// The shipped m4a fixture: 440 Hz sine, 0.5 s, mono, 44.1 kHz,
    /// AAC-LC encoded by ffmpeg.  See `tests/fixtures/README.md` for
    /// the regeneration command if it ever needs to be rebuilt.
    const FIXTURE_M4A: &str = "sine_440_0.5s_mono.m4a";
    const FIXTURE_M4A_DURATION_SECS: f64 = 0.5;

    /// Stereo AAC-LC fixture: 440 Hz sine, 0.5 s, stereo, 44.1 kHz.
    /// Mirrors the format produced by the AudioRecorder app the user
    /// imports voice memos from.  Used to lock in the fix for a bug
    /// where symphonia's ISO/MP4 demuxer underreports stereo AAC
    /// tracks as `codec_params.channels = 1`, causing the mixdown to
    /// produce a buffer twice as long as it should be (playback ran
    /// at half speed / an octave low).
    const FIXTURE_M4A_STEREO: &str = "sine_440_0.5s_stereo.m4a";

    #[cfg(feature = "ui")]
    #[test]
    fn m4a_duration_secs_matches_fixture() {
        let path = fixture(FIXTURE_M4A);
        assert!(path.exists(), "fixture missing: {}", path.display());
        let dur = m4a_duration_secs(&path).expect("mvhd duration should parse");
        // ffmpeg's AAC encoder rounds up to a frame boundary, so the
        // actual encoded duration can be slightly longer than the
        // requested 0.5 s.  Tolerate ±0.1 s to keep the test robust
        // against future ffmpeg-version drift if the fixture is ever
        // regenerated.
        assert!(
            (dur - FIXTURE_M4A_DURATION_SECS).abs() < 0.1,
            "duration {} not within 0.1s of {}",
            dur,
            FIXTURE_M4A_DURATION_SECS
        );
    }

    #[cfg(feature = "ui")]
    #[test]
    fn m4a_duration_secs_missing_file_returns_none() {
        let path = fixture("does-not-exist.m4a");
        assert!(m4a_duration_secs(&path).is_none());
    }

    #[test]
    fn read_m4a_as_f32_decodes_known_fixture() {
        let path = fixture(FIXTURE_M4A);
        assert!(path.exists(), "fixture missing: {}", path.display());

        // Read at 16 kHz to match the waterfall pipeline.
        let samples = read_m4a_as_f32(&path, 16_000).expect("decode");

        // Expected sample count: ~0.5s × 16 000 Hz = 8 000 samples.
        // Allow a generous ±20% window: AAC encoders typically pad
        // with priming/leading silence and the encoder may round the
        // duration up to a frame boundary, so an exact count would be
        // brittle across ffmpeg versions.
        let expected = (FIXTURE_M4A_DURATION_SECS * 16_000.0) as usize;
        let lo = expected * 8 / 10;
        let hi = expected * 12 / 10;
        assert!(
            samples.len() >= lo && samples.len() <= hi,
            "expected {}..{} samples, got {}",
            lo,
            hi,
            samples.len()
        );

        // The signal is a 440 Hz sine — peak amplitude must be
        // meaningfully non-zero.  A silent or broken decode would
        // produce a peak near 0.
        let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(
            peak > 0.1,
            "expected a non-silent sine wave, got peak {}",
            peak
        );
    }

    /// Regression test for the "everything shifted to the grave" bug:
    /// stereo AAC-LC m4a files (the common AudioRecorder output, also
    /// the iPhone Voice Memos format on recent iOS) were decoded at
    /// twice the real duration, causing playback to run at half speed
    /// and an octave below pitch.
    ///
    /// Root cause: symphonia 0.5.x's ISO/MP4 demuxer surfaces a stereo
    /// AAC track with `codec_params.channels = 1`, but the AAC decoder
    /// itself produces a 2-channel `AudioBuffer`.  The decode loop
    /// must read the channel count from the decoded buffer's
    /// `SignalSpec`, not from `codec_params`, otherwise the mono
    /// mixdown walks the interleaved buffer with the wrong stride and
    /// concatenates L,R,L,R,... as mono samples — doubling the
    /// returned vector's length.
    ///
    /// This test asserts the returned mono buffer length matches the
    /// real audio duration (within the ±20 % envelope already used by
    /// the mono variant).  Without the fix the length is roughly 2×
    /// the upper bound and the assertion fails decisively.
    #[test]
    fn read_m4a_as_f32_stereo_decodes_to_correct_duration() {
        let path = fixture(FIXTURE_M4A_STEREO);
        assert!(path.exists(), "fixture missing: {}", path.display());

        // Read at 16 kHz to match the waterfall pipeline, and again
        // at 48 kHz to match the typical playback path.
        for &target_rate in &[16_000u32, 48_000u32] {
            let samples = read_m4a_as_f32(&path, target_rate).expect("decode");

            let expected = (FIXTURE_M4A_DURATION_SECS * target_rate as f64) as usize;
            let lo = expected * 8 / 10;
            let hi = expected * 12 / 10;
            assert!(
                samples.len() >= lo && samples.len() <= hi,
                "stereo m4a at {} Hz: expected {}..{} samples, got {} \
                 (a result near {} indicates the stereo-mono mixdown \
                 bug has regressed)",
                target_rate,
                lo,
                hi,
                samples.len(),
                expected * 2,
            );

            // The signal is a 440 Hz sine — peak must be non-trivial.
            // The threshold is intentionally well below the encoded
            // sine's expected amplitude (~0.1–0.3 after AAC encode and
            // linear-interpolation resampling).  A silent or broken
            // decode would produce a peak near 0.
            let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
            assert!(
                peak > 0.05,
                "stereo m4a at {} Hz: expected non-silent sine, got peak {}",
                target_rate,
                peak
            );
        }
    }

    #[test]
    fn read_audio_as_i16_dispatches_m4a() {
        let path = fixture(FIXTURE_M4A);
        assert!(path.exists(), "fixture missing: {}", path.display());

        let samples = read_audio_as_i16(&path).expect("dispatch + decode");
        assert!(!samples.is_empty(), "dispatcher returned no samples");

        // i16 conversion: peak should be far from zero for a real sine.
        let peak = samples
            .iter()
            .fold(0i32, |m, &s| m.max(s.unsigned_abs() as i32));
        assert!(peak > 1_000, "i16 peak too low: {}", peak);
    }

    #[test]
    fn read_audio_as_i16_rejects_unknown_extension() {
        // Use a path with a deliberately unsupported extension.
        // The function must error rather than guess a decoder.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("clip.flac");
        std::fs::write(&path, b"not really a flac").expect("write");
        let err = read_audio_as_i16(&path).expect_err("should reject .flac");
        assert!(
            err.to_string().contains("unsupported audio format"),
            "unexpected error: {}",
            err
        );
    }
}
