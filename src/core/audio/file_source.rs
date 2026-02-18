//! WAV file audio source.
//!
//! Reads a WAV file and produces PCM i16 chunks through the same
//! channel interface as the live microphone capture, enabling
//! reproducible benchmarks across transcription providers.

use super::{AudioCapture, CHANNEL_CAPACITY, CHUNK_DURATION_MS};
use crate::core::config::AudioConfig;
use crate::core::error::TalkError;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Audio capture that reads PCM samples from a WAV file.
///
/// Requires the file to be 16-bit PCM, mono, 16 kHz — matching the
/// hardcoded [`AudioConfig`] used by all transcription paths.
pub struct WavFileSource {
    path: PathBuf,
    running: Arc<AtomicBool>,
}

impl WavFileSource {
    /// Create a new WAV file source.
    ///
    /// The file is validated (header parsed) eagerly so that errors
    /// surface before audio capture is "started".
    pub fn new(path: &Path, audio_config: &AudioConfig) -> Result<Self, TalkError> {
        validate_wav_header(path, audio_config)?;
        Ok(Self {
            path: path.to_path_buf(),
            running: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl AudioCapture for WavFileSource {
    fn start(&mut self) -> Result<mpsc::Receiver<Vec<i16>>, TalkError> {
        if self.running.load(Ordering::Acquire) {
            return Err(TalkError::Audio("File source already running".to_string()));
        }

        let (sender, receiver) = mpsc::channel(CHANNEL_CAPACITY);
        let running = Arc::clone(&self.running);
        let path = self.path.clone();

        running.store(true, Ordering::Release);

        tokio::spawn(async move {
            if let Err(e) = read_wav_chunks(&path, &sender, &running).await {
                log::error!("WAV file reader error: {}", e);
            }
            // Sender is dropped here, closing the channel.
            running.store(false, Ordering::Release);
        });

        Ok(receiver)
    }

    fn stop(&mut self) -> Result<(), TalkError> {
        self.running.store(false, Ordering::Release);
        Ok(())
    }
}

/// Validate that a WAV file matches the expected audio configuration.
fn validate_wav_header(path: &Path, audio_config: &AudioConfig) -> Result<(), TalkError> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        TalkError::Audio(format!(
            "failed to open audio file {}: {}",
            path.display(),
            e
        ))
    })?;

    let header = parse_wav_header(&mut file, path)?;

    if header.audio_format != 1 {
        return Err(TalkError::Audio(format!(
            "{}: expected PCM format (1), got {}. Convert with: \
             ffmpeg -i {} -ar 16000 -ac 1 -sample_fmt s16 output.wav",
            path.display(),
            header.audio_format,
            path.display(),
        )));
    }

    if header.bits_per_sample != 16 {
        return Err(TalkError::Audio(format!(
            "{}: expected 16-bit samples, got {}-bit. Convert with: \
             ffmpeg -i {} -ar 16000 -ac 1 -sample_fmt s16 output.wav",
            path.display(),
            header.bits_per_sample,
            path.display(),
        )));
    }

    if header.num_channels != audio_config.channels as u16 {
        return Err(TalkError::Audio(format!(
            "{}: expected {} channel(s), got {}. Convert with: \
             ffmpeg -i {} -ar 16000 -ac 1 -sample_fmt s16 output.wav",
            path.display(),
            audio_config.channels,
            header.num_channels,
            path.display(),
        )));
    }

    if header.sample_rate != audio_config.sample_rate {
        return Err(TalkError::Audio(format!(
            "{}: expected {} Hz sample rate, got {} Hz. Convert with: \
             ffmpeg -i {} -ar 16000 -ac 1 -sample_fmt s16 output.wav",
            path.display(),
            audio_config.sample_rate,
            header.sample_rate,
            path.display(),
        )));
    }

    Ok(())
}

/// Parsed WAV header fields.
struct WavHeader {
    audio_format: u16,
    num_channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    /// Byte offset where the PCM data starts.
    data_offset: u64,
    /// Number of bytes of PCM data.
    data_size: u32,
}

/// Parse a WAV file header and locate the data chunk.
fn parse_wav_header<R: Read + Seek>(reader: &mut R, path: &Path) -> Result<WavHeader, TalkError> {
    let err = |msg: &str| TalkError::Audio(format!("{}: {}", path.display(), msg));

    // RIFF header
    let mut riff = [0u8; 4];
    reader
        .read_exact(&mut riff)
        .map_err(|_| err("too short to be a WAV file"))?;
    if &riff != b"RIFF" {
        return Err(err("not a WAV file (missing RIFF header)"));
    }

    // Skip file size
    let _file_size = reader
        .read_u32::<LittleEndian>()
        .map_err(|_| err("truncated RIFF header"))?;

    let mut wave = [0u8; 4];
    reader
        .read_exact(&mut wave)
        .map_err(|_| err("truncated RIFF header"))?;
    if &wave != b"WAVE" {
        return Err(err("not a WAV file (missing WAVE identifier)"));
    }

    // Find fmt and data chunks
    let mut audio_format = 0u16;
    let mut num_channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data_offset = 0u64;
    let mut data_size = 0u32;
    let mut found_fmt = false;
    let mut found_data = false;

    loop {
        let mut chunk_id = [0u8; 4];
        if reader.read_exact(&mut chunk_id).is_err() {
            break;
        }
        let chunk_size = reader
            .read_u32::<LittleEndian>()
            .map_err(|_| err("truncated chunk header"))?;

        match &chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    return Err(err("fmt chunk too small"));
                }
                audio_format = reader
                    .read_u16::<LittleEndian>()
                    .map_err(|_| err("truncated fmt chunk"))?;
                num_channels = reader
                    .read_u16::<LittleEndian>()
                    .map_err(|_| err("truncated fmt chunk"))?;
                sample_rate = reader
                    .read_u32::<LittleEndian>()
                    .map_err(|_| err("truncated fmt chunk"))?;
                let _byte_rate = reader
                    .read_u32::<LittleEndian>()
                    .map_err(|_| err("truncated fmt chunk"))?;
                let _block_align = reader
                    .read_u16::<LittleEndian>()
                    .map_err(|_| err("truncated fmt chunk"))?;
                bits_per_sample = reader
                    .read_u16::<LittleEndian>()
                    .map_err(|_| err("truncated fmt chunk"))?;
                found_fmt = true;

                // Skip any extra fmt bytes
                let read_so_far = 16u32;
                if chunk_size > read_so_far {
                    reader
                        .seek(SeekFrom::Current((chunk_size - read_so_far) as i64))
                        .map_err(|_| err("failed to skip extra fmt bytes"))?;
                }
            }
            b"data" => {
                data_offset = reader
                    .stream_position()
                    .map_err(|_| err("failed to get data offset"))?;
                data_size = chunk_size;
                found_data = true;
                break;
            }
            _ => {
                // Skip unknown chunk
                reader
                    .seek(SeekFrom::Current(chunk_size as i64))
                    .map_err(|_| err("failed to skip unknown chunk"))?;
            }
        }
    }

    if !found_fmt {
        return Err(err("missing fmt chunk"));
    }
    if !found_data {
        return Err(err("missing data chunk"));
    }

    Ok(WavHeader {
        audio_format,
        num_channels,
        sample_rate,
        bits_per_sample,
        data_offset,
        data_size,
    })
}

/// Read PCM chunks from a WAV file and send through the channel.
///
/// Chunks are sized identically to live capture (20ms at the file's
/// sample rate).  Chunks are sent as fast as the channel will accept
/// them — backpressure from the consumer provides natural throttling.
async fn read_wav_chunks(
    path: &Path,
    sender: &mpsc::Sender<Vec<i16>>,
    running: &AtomicBool,
) -> Result<(), TalkError> {
    let audio_config = AudioConfig::new();
    let mut file = std::fs::File::open(path).map_err(|e| {
        TalkError::Audio(format!(
            "failed to open audio file {}: {}",
            path.display(),
            e
        ))
    })?;

    let header = parse_wav_header(&mut file, path)?;
    file.seek(SeekFrom::Start(header.data_offset))
        .map_err(|e| TalkError::Audio(format!("failed to seek to data: {}", e)))?;

    let frames_per_chunk = (audio_config.sample_rate as usize * CHUNK_DURATION_MS as usize) / 1000;
    let samples_per_chunk = frames_per_chunk * audio_config.channels as usize;
    let bytes_per_sample = (header.bits_per_sample / 8) as usize;
    let total_samples = header.data_size as usize / bytes_per_sample;

    let mut samples_read = 0usize;
    let mut chunk = Vec::with_capacity(samples_per_chunk);

    while samples_read < total_samples && running.load(Ordering::Acquire) {
        let sample = match file.read_i16::<LittleEndian>() {
            Ok(s) => s,
            Err(_) => break,
        };
        chunk.push(sample);
        samples_read += 1;

        if chunk.len() >= samples_per_chunk {
            let batch = std::mem::replace(&mut chunk, Vec::with_capacity(samples_per_chunk));
            if sender.send(batch).await.is_err() {
                log::debug!("audio channel closed, stopping file reader");
                return Ok(());
            }
            // Yield to let the consumer process
            tokio::task::yield_now().await;
        }
    }

    // Send any remaining partial chunk
    if !chunk.is_empty() && running.load(Ordering::Acquire) {
        let _ = sender.send(chunk).await;
    }

    let duration = samples_read as f64 / audio_config.sample_rate as f64;
    log::info!(
        "file source: read {} samples ({:.1}s) from {}",
        samples_read,
        duration,
        path.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::audio::AudioWriter;
    use crate::core::audio::WavWriter;
    use tempfile::NamedTempFile;

    /// Create a valid 16kHz mono 16-bit WAV file with synthetic PCM data.
    fn create_test_wav(num_samples: usize) -> NamedTempFile {
        let audio_config = AudioConfig::new();
        let mut writer = WavWriter::new(audio_config);

        let mut file = NamedTempFile::new().expect("create temp file");
        let header = writer.header().expect("write header");
        std::io::Write::write_all(&mut file, &header).expect("write header bytes");

        // Generate sine wave samples
        let samples: Vec<i16> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (f32::sin(2.0 * std::f32::consts::PI * 440.0 * t) * 10000.0) as i16
            })
            .collect();
        let pcm_bytes = writer.write_pcm(&samples).expect("write pcm");
        std::io::Write::write_all(&mut file, &pcm_bytes).expect("write pcm bytes");

        // Finalize: write corrected header
        let final_header = writer.finalize().expect("finalize");
        std::io::Seek::seek(&mut file, SeekFrom::Start(0)).expect("seek");
        std::io::Write::write_all(&mut file, &final_header).expect("write final header");

        file
    }

    #[test]
    fn test_validate_wav_header_valid() {
        let file = create_test_wav(16000); // 1 second
        let config = AudioConfig::new();
        assert!(validate_wav_header(file.path(), &config).is_ok());
    }

    #[test]
    fn test_validate_wav_header_not_wav() {
        let mut file = NamedTempFile::new().expect("create temp");
        std::io::Write::write_all(&mut file, b"not a wav file at all").expect("write");
        let config = AudioConfig::new();
        let result = validate_wav_header(file.path(), &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("RIFF"));
    }

    #[test]
    fn test_validate_wav_header_wrong_sample_rate() {
        // Create a WAV with wrong sample rate by manually building header
        let mut file = NamedTempFile::new().expect("create temp");
        let mut hdr = Vec::new();
        hdr.extend_from_slice(b"RIFF");
        hdr.extend_from_slice(&100u32.to_le_bytes()); // file size
        hdr.extend_from_slice(b"WAVE");
        hdr.extend_from_slice(b"fmt ");
        hdr.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        hdr.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        hdr.extend_from_slice(&1u16.to_le_bytes()); // mono
        hdr.extend_from_slice(&44100u32.to_le_bytes()); // 44.1kHz (wrong!)
        hdr.extend_from_slice(&88200u32.to_le_bytes()); // byte rate
        hdr.extend_from_slice(&2u16.to_le_bytes()); // block align
        hdr.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        hdr.extend_from_slice(b"data");
        hdr.extend_from_slice(&0u32.to_le_bytes()); // data size
        std::io::Write::write_all(&mut file, &hdr).expect("write");

        let config = AudioConfig::new();
        let result = validate_wav_header(file.path(), &config);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("16000 Hz"));
        assert!(msg.contains("44100 Hz"));
        assert!(msg.contains("ffmpeg"));
    }

    #[test]
    fn test_wav_file_source_new_valid() {
        let file = create_test_wav(16000);
        let config = AudioConfig::new();
        assert!(WavFileSource::new(file.path(), &config).is_ok());
    }

    #[test]
    fn test_wav_file_source_new_nonexistent() {
        let config = AudioConfig::new();
        let result = WavFileSource::new(Path::new("/nonexistent/file.wav"), &config);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_wav_file_source_reads_all_samples() {
        let num_samples = 3200; // 200ms = 10 chunks of 320 samples
        let file = create_test_wav(num_samples);
        let config = AudioConfig::new();
        let mut source = WavFileSource::new(file.path(), &config).expect("create source");

        let mut rx = source.start().expect("start");

        let mut total = 0usize;
        while let Some(chunk) = rx.recv().await {
            total += chunk.len();
        }

        assert_eq!(total, num_samples);
    }

    #[tokio::test]
    async fn test_wav_file_source_chunk_size() {
        let file = create_test_wav(16000); // 1 second
        let config = AudioConfig::new();
        let mut source = WavFileSource::new(file.path(), &config).expect("create source");

        let mut rx = source.start().expect("start");

        // First chunk should be 320 samples (20ms at 16kHz)
        let chunk = rx.recv().await.expect("first chunk");
        assert_eq!(chunk.len(), 320);
    }

    #[tokio::test]
    async fn test_wav_file_source_stop_aborts() {
        let file = create_test_wav(160_000); // 10 seconds
        let config = AudioConfig::new();
        let mut source = WavFileSource::new(file.path(), &config).expect("create source");

        let mut rx = source.start().expect("start");

        // Read a few chunks then stop
        let _ = rx.recv().await;
        let _ = rx.recv().await;
        source.stop().expect("stop");

        // Channel should close shortly after
        let mut remaining = 0usize;
        while rx.recv().await.is_some() {
            remaining += 1;
        }

        // Should have stopped well before reading all 500 chunks
        assert!(remaining < 450);
    }

    #[tokio::test]
    async fn test_wav_file_source_samples_are_nonzero() {
        let file = create_test_wav(3200);
        let config = AudioConfig::new();
        let mut source = WavFileSource::new(file.path(), &config).expect("create source");

        let mut rx = source.start().expect("start");

        let chunk = rx.recv().await.expect("chunk");
        // Sine wave should have non-zero samples
        assert!(chunk.iter().any(|&s| s != 0));
    }
}
