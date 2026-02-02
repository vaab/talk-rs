//! Mock audio capture for testing.

use super::{AudioCapture, CHANNEL_CAPACITY, CHUNK_DURATION_MS};
use crate::core::error::TalkError;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{self, Duration, MissedTickBehavior};

/// Mock audio capture that generates synthetic PCM samples.
pub struct MockAudioCapture {
    sample_rate: u32,
    channels: u8,
    frequency_hz: f32,
    running: Arc<AtomicBool>,
}

impl MockAudioCapture {
    /// Create a new mock capture with explicit configuration.
    pub fn new(sample_rate: u32, channels: u8, frequency_hz: f32) -> Self {
        Self {
            sample_rate,
            channels,
            frequency_hz,
            running: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl AudioCapture for MockAudioCapture {
    fn start(&mut self) -> Result<mpsc::Receiver<Vec<i16>>, TalkError> {
        if self.running.load(Ordering::Acquire) {
            return Err(TalkError::Audio("Capture already running".to_string()));
        }

        let (sender, receiver) = mpsc::channel(CHANNEL_CAPACITY);
        let sample_rate = self.sample_rate;
        let channels = self.channels;
        let frequency_hz = self.frequency_hz;
        let running = Arc::clone(&self.running);

        running.store(true, Ordering::Release);

        tokio::spawn(async move {
            let frames_per_chunk = (sample_rate as usize * CHUNK_DURATION_MS as usize) / 1000;
            let samples_per_chunk = frames_per_chunk * channels as usize;
            let mut sample_index = 0u64;
            let mut ticker = time::interval(Duration::from_millis(CHUNK_DURATION_MS));
            ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

            loop {
                ticker.tick().await;
                if !running.load(Ordering::Acquire) {
                    break;
                }

                let mut chunk = Vec::with_capacity(samples_per_chunk);
                for _ in 0..frames_per_chunk {
                    let t = sample_index as f32 / sample_rate as f32;
                    let value = (2.0 * PI * frequency_hz * t).sin();
                    let sample = (value * i16::MAX as f32) as i16;

                    for _ in 0..channels {
                        chunk.push(sample);
                    }

                    sample_index += 1;
                }

                if sender.send(chunk).await.is_err() {
                    break;
                }
            }
        });

        Ok(receiver)
    }

    fn stop(&mut self) -> Result<(), TalkError> {
        self.running.store(false, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_chunk_samples(sample_rate: u32, channels: u8) -> usize {
        let frames = (sample_rate as usize * CHUNK_DURATION_MS as usize) / 1000;
        frames * channels as usize
    }

    #[tokio::test]
    async fn test_mock_capture_generates_samples() {
        let mut capture = MockAudioCapture::new(16_000, 1, 440.0);
        let mut receiver = capture.start().expect("start should succeed");

        let chunk = receiver.recv().await.expect("chunk should arrive");

        assert_eq!(chunk.len(), expected_chunk_samples(16_000, 1));
        assert!(chunk.iter().any(|&sample| sample != 0));

        capture.stop().expect("stop should succeed");
    }

    #[tokio::test]
    async fn test_mock_capture_stereo_chunk_size() {
        let mut capture = MockAudioCapture::new(16_000, 2, 220.0);
        let mut receiver = capture.start().expect("start should succeed");

        let chunk = receiver.recv().await.expect("chunk should arrive");

        assert_eq!(chunk.len(), expected_chunk_samples(16_000, 2));

        capture.stop().expect("stop should succeed");
    }
}
