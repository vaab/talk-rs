//! Audio tee: splits a PCM stream so the overlay visualizer reads
//! from the same PipeWire capture as the recording pipeline.

use crate::x11::render_util::RingBuffer;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Spawn an async task that reads i16 PCM chunks from `input`,
/// converts to f32, pushes into the shared `ring` buffer, and
/// forwards the original i16 chunks unchanged to the returned receiver.
pub fn spawn_audio_tee(
    mut input: mpsc::Receiver<Vec<i16>>,
    ring: Arc<Mutex<RingBuffer>>,
) -> mpsc::Receiver<Vec<i16>> {
    let (tx, rx) = mpsc::channel(super::CHANNEL_CAPACITY);
    tokio::spawn(async move {
        while let Some(chunk) = input.recv().await {
            // Convert i16 → f32 and push to ring buffer for overlay
            let f32_samples: Vec<f32> = chunk.iter().map(|&s| s as f32 / 32768.0).collect();
            if let Ok(mut guard) = ring.lock() {
                guard.push(&f32_samples);
            }
            // Forward original chunk unchanged
            if tx.send(chunk).await.is_err() {
                break; // downstream closed
            }
        }
    });
    rx
}
