//! Audio resampling using rubato's sinc interpolation.
//!
//! Records at the device's native rate (typically 48 kHz) and downsamples
//! to 16 kHz with a proper anti-aliasing filter before encoding and
//! transcription.

use crate::core::error::TalkError;
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use tokio::sync::mpsc;

/// Sinc-interpolated audio resampler with built-in anti-aliasing.
pub struct AudioResampler {
    inner: Async<f32>,
    /// Pre-allocated input buffer (one `Vec<f32>` per channel; mono = 1).
    input_buf: Vec<Vec<f32>>,
    /// Pre-allocated output buffer.
    output_buf: Vec<Vec<f32>>,
    /// Frames to skip at the start (resampler group delay compensation).
    delay_frames_remaining: usize,
    /// Expected input frames per `process_chunk` call.
    chunk_frames: usize,
}

impl AudioResampler {
    /// Create a resampler for the given rate conversion.
    ///
    /// `chunk_frames` must match the capture chunk size (e.g. 960 for
    /// 48 kHz at 20 ms).
    pub fn new(from_rate: u32, to_rate: u32, chunk_frames: usize) -> Result<Self, TalkError> {
        let sinc_len = 128;
        let window = WindowFunction::Blackman2;
        let f_cutoff = rubato::calculate_cutoff(sinc_len, window);

        let params = SincInterpolationParameters {
            sinc_len,
            f_cutoff,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 256,
            window,
        };

        let ratio = to_rate as f64 / from_rate as f64;
        let channels = 1usize; // mono

        let resampler = Async::<f32>::new_sinc(
            ratio,
            1.0, // fixed ratio, no dynamic adjustment
            &params,
            chunk_frames,
            channels,
            FixedAsync::Input,
        )
        .map_err(|e| TalkError::Audio(format!("failed to create resampler: {}", e)))?;

        let delay = resampler.output_delay();
        let max_out = resampler.output_frames_max();
        let input_buf = vec![vec![0.0f32; chunk_frames]; channels];
        let output_buf = vec![vec![0.0f32; max_out]; channels];

        log::info!(
            "resampler: {}Hz -> {}Hz, chunk={} frames, max_out={}, delay={} frames",
            from_rate,
            to_rate,
            chunk_frames,
            max_out,
            delay
        );

        Ok(Self {
            inner: resampler,
            input_buf,
            output_buf,
            delay_frames_remaining: delay,
            chunk_frames,
        })
    }

    /// Process a full chunk of mono i16 PCM samples.
    ///
    /// Input must be exactly `chunk_frames` samples.  Returns resampled
    /// samples (may be empty during initial delay compensation).
    pub fn process_chunk(&mut self, input: &[i16]) -> Result<Vec<i16>, TalkError> {
        self.resample_inner(input, None)
    }

    /// Process a final partial chunk (fewer than `chunk_frames` samples).
    ///
    /// Uses rubato's `partial_len` indexing to avoid zero-pad artefacts.
    pub fn flush(&mut self, input: &[i16]) -> Result<Vec<i16>, TalkError> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        self.resample_inner(input, Some(input.len()))
    }

    /// Expected number of input frames per `process_chunk` call.
    pub fn chunk_frames(&self) -> usize {
        self.chunk_frames
    }

    /// Common resampling logic for both full and partial chunks.
    fn resample_inner(
        &mut self,
        input: &[i16],
        partial_len: Option<usize>,
    ) -> Result<Vec<i16>, TalkError> {
        // Convert i16 -> f32 normalised to [-1.0, 1.0]
        let buf = &mut self.input_buf[0];
        let copy_len = input.len().min(buf.len());
        for (dst, &src) in buf[..copy_len].iter_mut().zip(input.iter()) {
            *dst = src as f32 / f32::from(i16::MAX);
        }
        // Zero-pad remainder (for partial chunks)
        for dst in buf[copy_len..].iter_mut() {
            *dst = 0.0;
        }

        // Reset output buffer length to max capacity
        let max_out = self.output_buf[0].capacity();
        self.output_buf[0].resize(max_out, 0.0);

        let input_adapter =
            SequentialSliceOfVecs::new(&self.input_buf as &[Vec<f32>], 1, self.chunk_frames)
                .map_err(|e| TalkError::Audio(format!("resample input adapter error: {}", e)))?;
        let mut output_adapter =
            SequentialSliceOfVecs::new_mut(&mut self.output_buf, 1, max_out)
                .map_err(|e| TalkError::Audio(format!("resample output adapter error: {}", e)))?;

        let indexing = Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len,
            active_channels_mask: None,
        };

        let (_, out_frames) = self
            .inner
            .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
            .map_err(|e| TalkError::Audio(format!("resample error: {}", e)))?;

        // Trim startup delay
        let skip = self.delay_frames_remaining.min(out_frames);
        self.delay_frames_remaining -= skip;

        // Convert f32 -> i16
        Ok(self.output_buf[0][skip..out_frames]
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)
            .collect())
    }
}

/// Spawn an async resampling task between two channels.
///
/// Reads `Vec<i16>` chunks at `from_rate`, resamples to `to_rate`, and
/// forwards resampled chunks.  Returns the output receiver.
///
/// If `from_rate == to_rate`, returns `input_rx` unchanged (no-op).
pub fn spawn_resample_task(
    from_rate: u32,
    to_rate: u32,
    input_rx: mpsc::Receiver<Vec<i16>>,
    chunk_frames: usize,
) -> Result<mpsc::Receiver<Vec<i16>>, TalkError> {
    if from_rate == to_rate {
        log::info!(
            "capture rate matches target rate ({}Hz), skipping resample",
            from_rate
        );
        return Ok(input_rx);
    }

    let mut resampler = AudioResampler::new(from_rate, to_rate, chunk_frames)?;
    let expected = resampler.chunk_frames();
    let (tx, rx) = mpsc::channel(super::CHANNEL_CAPACITY);

    tokio::spawn(async move {
        let mut input_rx = input_rx;
        while let Some(chunk) = input_rx.recv().await {
            let result = if chunk.len() == expected {
                resampler.process_chunk(&chunk)
            } else {
                // Final partial chunk from capture stop
                log::debug!(
                    "resample: partial chunk ({}/{} frames), flushing",
                    chunk.len(),
                    expected
                );
                resampler.flush(&chunk)
            };

            match result {
                Ok(resampled) => {
                    if !resampled.is_empty() && tx.send(resampled).await.is_err() {
                        log::debug!("resample output channel closed");
                        break;
                    }
                }
                Err(e) => {
                    log::error!("resample error: {}", e);
                    break;
                }
            }
        }
        log::debug!("resample task finished");
    });

    Ok(rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_creation() {
        let r = AudioResampler::new(48_000, 16_000, 960);
        assert!(r.is_ok());
        assert_eq!(r.unwrap().chunk_frames(), 960);
    }

    #[test]
    fn test_resampler_same_rate() {
        let r = AudioResampler::new(16_000, 16_000, 320);
        assert!(r.is_ok());
    }

    #[test]
    fn test_process_chunk_output_length() {
        let mut r = AudioResampler::new(48_000, 16_000, 960).unwrap();

        // Feed several chunks to get past the startup delay
        let input = vec![0i16; 960];
        let mut total_out = 0;
        for _ in 0..10 {
            let out = r.process_chunk(&input).unwrap();
            total_out += out.len();
        }

        // After 10 chunks (10 * 960 = 9600 frames at 48kHz = 200ms),
        // we should have roughly 3200 output frames (200ms at 16kHz),
        // minus the startup delay.
        let expected_approx = 10 * 320; // 960 / 3 = 320 per chunk
        assert!(
            total_out > expected_approx - 200 && total_out < expected_approx + 200,
            "total output {} not near expected {}",
            total_out,
            expected_approx
        );
    }

    #[test]
    fn test_process_chunk_preserves_signal() {
        let mut r = AudioResampler::new(48_000, 16_000, 960).unwrap();

        // Generate a 1kHz sine wave at 48kHz (well below Nyquist for both rates)
        let input: Vec<i16> = (0..960)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                (f32::sin(2.0 * std::f32::consts::PI * 1000.0 * t) * 16000.0) as i16
            })
            .collect();

        // Process several chunks to get past delay
        for _ in 0..5 {
            let _ = r.process_chunk(&input);
        }

        // After the delay, output should have non-trivial energy
        let out = r.process_chunk(&input).unwrap();
        let energy: f64 = out.iter().map(|&s| (s as f64) * (s as f64)).sum();
        assert!(energy > 0.0, "resampled signal should have energy");
    }

    #[test]
    fn test_flush_empty() {
        let mut r = AudioResampler::new(48_000, 16_000, 960).unwrap();
        let out = r.flush(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_flush_partial() {
        let mut r = AudioResampler::new(48_000, 16_000, 960).unwrap();
        // Feed a few full chunks first
        let full = vec![0i16; 960];
        for _ in 0..5 {
            let _ = r.process_chunk(&full);
        }
        // Then flush with a partial chunk
        let partial = vec![0i16; 480];
        let out = r.flush(&partial);
        assert!(out.is_ok());
    }

    #[tokio::test]
    async fn test_spawn_resample_task_passthrough() {
        let (tx, rx) = mpsc::channel(10);
        // Same rate: should return the same receiver (no resampling)
        let out_rx = spawn_resample_task(16_000, 16_000, rx, 320).unwrap();

        tx.send(vec![1, 2, 3]).await.unwrap();
        drop(tx);

        // The receiver should yield the original data
        let mut out_rx = out_rx;
        let chunk = out_rx.recv().await.unwrap();
        assert_eq!(chunk, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_spawn_resample_task_downsamples() {
        let (tx, rx) = mpsc::channel(10);
        let mut out_rx = spawn_resample_task(48_000, 16_000, rx, 960).unwrap();

        // Send several chunks of silence
        for _ in 0..10 {
            tx.send(vec![0i16; 960]).await.unwrap();
        }
        drop(tx);

        // Should receive some output chunks
        let mut count = 0;
        while out_rx.recv().await.is_some() {
            count += 1;
        }
        assert!(count > 0, "should produce output chunks");
    }
}
