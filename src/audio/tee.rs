//! Audio tee: splits a PCM stream so the overlay visualizer reads
//! from the same PipeWire capture as the recording pipeline.

use crate::x11::render_util::RingBuffer;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// How many milliseconds of audio to keep in the lookback buffer while
/// paused.  When the pause clears (speech detected), this buffer is
/// flushed downstream first so the onset of speech is not lost.
const LOOKBACK_MS: u64 = 300;

/// Spawn an async task that reads i16 PCM chunks from `input`,
/// converts to f32, pushes into the shared `ring` buffer, and
/// forwards the original i16 chunks unchanged to the returned receiver.
///
/// When `pause` is `true`, the ring buffer is still fed (so the overlay
/// can detect when sound returns) but audio is buffered in a rolling
/// lookback window instead of being forwarded immediately.  When
/// `pause` transitions back to `false`, the lookback is flushed
/// downstream before normal forwarding resumes so the speech onset
/// is preserved.
pub fn spawn_audio_tee(
    mut input: mpsc::Receiver<Vec<i16>>,
    ring: Arc<Mutex<RingBuffer>>,
    pause: Arc<AtomicBool>,
) -> mpsc::Receiver<Vec<i16>> {
    let (tx, rx) = mpsc::channel(super::CHANNEL_CAPACITY);

    // Maximum number of chunks to keep in the lookback buffer.
    let chunk_duration_ms = super::CHUNK_DURATION_MS.max(1);
    let max_lookback_chunks = (LOOKBACK_MS / chunk_duration_ms) as usize;

    tokio::spawn(async move {
        let mut lookback: VecDeque<Vec<i16>> = VecDeque::new();
        let mut was_paused = false;

        while let Some(chunk) = input.recv().await {
            // Convert i16 → f32 and push to ring buffer for overlay
            let f32_samples: Vec<f32> = chunk.iter().map(|&s| s as f32 / 32768.0).collect();
            if let Ok(mut guard) = ring.lock() {
                guard.push(&f32_samples);
            }

            let paused = pause.load(Ordering::Relaxed);

            if paused {
                // While paused, accumulate chunks in a rolling window.
                lookback.push_back(chunk);
                while lookback.len() > max_lookback_chunks {
                    lookback.pop_front();
                }
                was_paused = true;
            } else {
                // Flush lookback on pause→resume so the speech onset
                // is not lost.
                if was_paused {
                    for buffered in lookback.drain(..) {
                        if tx.send(buffered).await.is_err() {
                            return;
                        }
                    }
                    was_paused = false;
                }
                // Normal forwarding
                if tx.send(chunk).await.is_err() {
                    break; // downstream closed
                }
            }
        }
    });
    rx
}
