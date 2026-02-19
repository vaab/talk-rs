//! CPAL-based audio capture implementation.

use super::{AudioCapture, CHANNEL_CAPACITY, CHUNK_DURATION_MS};
use crate::core::config::AudioConfig;
use crate::core::error::TalkError;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Safety timeout for the flush signal.  This is NOT a heuristic sleep —
/// the normal path is a deterministic signal from the callback.  This
/// timeout only fires if the audio hardware is completely unresponsive
/// (driver crash, device unplugged, etc.).
const FLUSH_SAFETY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Audio capture using the CPAL backend.
pub struct CpalCapture {
    config: AudioConfig,
    stream: Mutex<Option<cpal::Stream>>,
    running: Arc<AtomicBool>,
    last_error: Arc<Mutex<Option<String>>>,

    /// Receives a signal when the CPAL callback has stored its final
    /// samples in `final_buffer`.  Created by `start()`, consumed by
    /// `stop()`.
    flush_done_rx: Mutex<Option<std_mpsc::Receiver<()>>>,

    /// Final partial buffer written by the callback on shutdown.
    /// The callback stores any remaining samples here instead of
    /// using `try_send` (which could silently drop them).
    final_buffer: Arc<Mutex<Option<Vec<i16>>>>,

    /// Clone of the channel sender kept alive so that `stop()` can
    /// inject the final samples after the stream (and its sender) has
    /// been dropped.  Dropping this clone closes the channel.
    sender_clone: Mutex<Option<mpsc::Sender<Vec<i16>>>>,
}

// CPAL streams are not marked as Send, but capture is managed behind a Mutex
// and only accessed through AudioCapture methods.
unsafe impl Send for CpalCapture {}

impl CpalCapture {
    /// Create a new CPAL capture from audio configuration.
    pub fn new(config: AudioConfig) -> Self {
        Self {
            config,
            stream: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
            last_error: Arc::new(Mutex::new(None)),
            flush_done_rx: Mutex::new(None),
            final_buffer: Arc::new(Mutex::new(None)),
            sender_clone: Mutex::new(None),
        }
    }

    /// Return the last error captured by the audio backend.
    pub fn last_error(&self) -> Result<Option<String>, TalkError> {
        self.last_error
            .lock()
            .map(|value| value.clone())
            .map_err(|_| TalkError::Audio("last_error lock poisoned".to_string()))
    }

    /// Query the default input device for its best mono capture rate.
    ///
    /// Returns the highest supported rate (capped at 48 kHz) that the
    /// default input device can deliver for the given channel count.
    /// Falls back to `fallback` if no device or no matching config is
    /// found (e.g. headless CI).
    pub fn preferred_capture_rate(channels: u8, fallback: u32) -> u32 {
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => {
                log::warn!("no input device found, using fallback rate {}Hz", fallback);
                return fallback;
            }
        };

        let configs = match device.supported_input_configs() {
            Ok(c) => c,
            Err(err) => {
                log::warn!(
                    "failed to query input configs: {}, using fallback rate {}Hz",
                    err,
                    fallback
                );
                return fallback;
            }
        };

        let mut best_rate = 0u32;
        for config in configs {
            if config.channels() == channels as u16 {
                best_rate = best_rate.max(config.max_sample_rate().0);
            }
        }

        if best_rate == 0 {
            log::warn!(
                "no supported input config for {} channel(s), using fallback rate {}Hz",
                channels,
                fallback
            );
            return fallback;
        }

        // Cap at 48 kHz — higher rates waste CPU for speech with no
        // quality benefit (Voxtral accepts at most 16 kHz anyway).
        let rate = best_rate.min(48_000);
        log::info!(
            "preferred capture rate: {}Hz (device max: {}Hz)",
            rate,
            best_rate
        );
        rate
    }
}

impl AudioCapture for CpalCapture {
    fn start(&mut self) -> Result<mpsc::Receiver<Vec<i16>>, TalkError> {
        if self.running.load(Ordering::Acquire) {
            return Err(TalkError::Audio("Capture already running".to_string()));
        }

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| TalkError::Audio("No input device available".to_string()))?;

        let mut supported_configs = device
            .supported_input_configs()
            .map_err(|err| TalkError::Audio(format!("Failed to query input configs: {}", err)))?;

        let config_range = supported_configs
            .find(|config| {
                config.channels() == self.config.channels as u16
                    && config.min_sample_rate().0 <= self.config.sample_rate
                    && config.max_sample_rate().0 >= self.config.sample_rate
            })
            .ok_or_else(|| {
                TalkError::Audio(format!(
                    "No supported config for {}Hz, {} channels",
                    self.config.sample_rate, self.config.channels
                ))
            })?;

        let supported_config =
            config_range.with_sample_rate(cpal::SampleRate(self.config.sample_rate));
        let sample_format = supported_config.sample_format();
        let stream_config: cpal::StreamConfig = supported_config.into();

        let (sender, receiver) = mpsc::channel(CHANNEL_CAPACITY);
        let sender_for_stop = sender.clone();
        let frames_per_chunk =
            (self.config.sample_rate as usize * CHUNK_DURATION_MS as usize) / 1000;
        let samples_per_chunk = frames_per_chunk * self.config.channels as usize;

        {
            let mut guard = self
                .last_error
                .lock()
                .map_err(|_| TalkError::Audio("last_error lock poisoned".to_string()))?;
            *guard = None;
        }

        let running = Arc::clone(&self.running);
        let last_error = Arc::clone(&self.last_error);

        // Flush synchronisation: the callback signals `flush_done_tx`
        // after storing its final samples in `final_buffer`.
        let (flush_done_tx, flush_done_rx) = std_mpsc::channel::<()>();
        let final_buffer: Arc<Mutex<Option<Vec<i16>>>> = Arc::new(Mutex::new(None));
        let final_buffer_for_stream = Arc::clone(&final_buffer);

        // Macro to reduce repetition across the 10 sample-format arms.
        // Each arm moves the same variables; this is safe because match
        // arms are mutually exclusive.
        macro_rules! build {
            ($T:ty) => {
                build_stream::<$T>(
                    &device,
                    &stream_config,
                    sender,
                    running,
                    last_error,
                    samples_per_chunk,
                    FlushSync {
                        buffer: final_buffer_for_stream,
                        done_tx: flush_done_tx,
                    },
                )
            };
        }

        let stream = match sample_format {
            SampleFormat::I8 => build!(i8),
            SampleFormat::U8 => build!(u8),
            SampleFormat::I16 => build!(i16),
            SampleFormat::U16 => build!(u16),
            SampleFormat::I32 => build!(i32),
            SampleFormat::U32 => build!(u32),
            SampleFormat::I64 => build!(i64),
            SampleFormat::U64 => build!(u64),
            SampleFormat::F32 => build!(f32),
            SampleFormat::F64 => build!(f64),
            _ => Err(TalkError::Audio("Unsupported sample format".to_string())),
        }?;

        let mut guard = self
            .stream
            .lock()
            .map_err(|_| TalkError::Audio("stream lock poisoned".to_string()))?;
        *guard = Some(stream);

        self.running.store(true, Ordering::Release);

        if let Some(stream) = guard.as_ref() {
            stream
                .play()
                .map_err(|err| TalkError::Audio(format!("Failed to start stream: {}", err)))?;
        }

        // Store synchronisation handles for stop().
        {
            let mut rx_guard = self
                .flush_done_rx
                .lock()
                .map_err(|_| TalkError::Audio("flush_done_rx lock poisoned".to_string()))?;
            *rx_guard = Some(flush_done_rx);
        }
        self.final_buffer = final_buffer;
        {
            let mut sc_guard = self
                .sender_clone
                .lock()
                .map_err(|_| TalkError::Audio("sender_clone lock poisoned".to_string()))?;
            *sc_guard = Some(sender_for_stop);
        }

        Ok(receiver)
    }

    fn stop(&mut self) -> Result<(), TalkError> {
        self.running.store(false, Ordering::Release);

        // ── 1. Wait for the callback to signal that it has stored its
        //       final samples.  This is deterministic: the callback
        //       sends on `flush_done_tx` as soon as it sees `running =
        //       false`.  The safety timeout only fires if the audio
        //       hardware is completely unresponsive.
        if let Some(rx) = self
            .flush_done_rx
            .lock()
            .map_err(|_| TalkError::Audio("flush_done_rx lock poisoned".to_string()))?
            .take()
        {
            match rx.recv_timeout(FLUSH_SAFETY_TIMEOUT) {
                Ok(()) => log::debug!("audio callback flush completed"),
                Err(std_mpsc::RecvTimeoutError::Timeout) => {
                    log::warn!(
                        "audio flush signal timed out after {}s — \
                         possible hardware issue, some audio may be lost",
                        FLUSH_SAFETY_TIMEOUT.as_secs()
                    );
                }
                Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                    // The stream (and its callback) was already dropped.
                    log::debug!("audio flush signal sender already dropped");
                }
            }
        }

        // ── 2. Drop the CPAL stream.  The callback can no longer fire
        //       and its sender clone is dropped, but the channel stays
        //       open because `sender_clone` is still alive.
        {
            let mut guard = self
                .stream
                .lock()
                .map_err(|_| TalkError::Audio("stream lock poisoned".to_string()))?;
            let _ = guard.take();
        }

        // ── 3. Inject final samples through the clone sender, then
        //       drop it to close the channel.  `block_in_place` is
        //       safe here because the tokio runtime is multi-threaded
        //       (`#[tokio::main]`).
        let final_samples = self
            .final_buffer
            .lock()
            .map_err(|_| TalkError::Audio("final_buffer lock poisoned".to_string()))?
            .take();

        let sender = self
            .sender_clone
            .lock()
            .map_err(|_| TalkError::Audio("sender_clone lock poisoned".to_string()))?
            .take();

        match (final_samples, sender) {
            (Some(samples), Some(tx)) => {
                let n = samples.len();
                tokio::task::block_in_place(|| match tx.blocking_send(samples) {
                    Ok(()) => log::debug!("injected {} final audio samples", n),
                    Err(_) => {
                        log::warn!(
                            "audio channel closed before {} final samples could be sent",
                            n
                        );
                    }
                });
                // `tx` dropped here → channel closes
            }
            (None, Some(_tx)) => {
                // No final samples to inject.
                // `_tx` dropped here → channel closes
            }
            (Some(samples), None) => {
                log::warn!(
                    "{} final audio samples lost — no sender available",
                    samples.len()
                );
            }
            (None, None) => {}
        }

        Ok(())
    }
}

/// Synchronisation handles for deterministic shutdown flushing.
///
/// Passed into `build_stream` so the CPAL callback can store its final
/// samples and signal completion instead of using `try_send`.
struct FlushSync {
    /// Shared buffer where the callback places its final samples.
    buffer: Arc<Mutex<Option<Vec<i16>>>>,
    /// One-shot signal: callback sends `()` once flush is complete.
    done_tx: std_mpsc::Sender<()>,
}

/// Build a CPAL input stream that converts samples to `i16` and sends
/// them through a tokio mpsc channel.
///
/// On shutdown (`running = false`), the callback stores any remaining
/// samples in `flush.buffer` and signals `flush.done_tx` instead of
/// using `try_send` (which could silently drop the tail).
fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sender: mpsc::Sender<Vec<i16>>,
    running: Arc<AtomicBool>,
    last_error: Arc<Mutex<Option<String>>>,
    samples_per_chunk: usize,
    flush: FlushSync,
) -> Result<cpal::Stream, TalkError>
where
    T: Sample + SizedSample + Copy,
    i16: FromSample<T>,
{
    let mut buffer: Vec<i16> = Vec::with_capacity(samples_per_chunk);

    let running_for_data = Arc::clone(&running);
    let running_for_error = Arc::clone(&running);

    // Track whether we have already flushed the final buffer so the
    // callback does it exactly once (the callback may fire several
    // more times between `running = false` and the stream being dropped).
    let mut flushed = false;
    let final_buffer = flush.buffer;
    let mut flush_done_tx = Some(flush.done_tx);

    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let is_running = running_for_data.load(Ordering::Acquire);

                if !is_running && flushed {
                    return;
                }

                // Always convert incoming samples — even if we are
                // stopping — so the final CPAL hardware buffer is not
                // lost.
                buffer.extend(data.iter().map(|sample| i16::from_sample(*sample)));

                if is_running {
                    // Normal operation: emit full-sized chunks.
                    while buffer.len() >= samples_per_chunk {
                        let chunk: Vec<i16> = buffer.drain(..samples_per_chunk).collect();
                        if sender.try_send(chunk).is_err() {
                            log::warn!("audio channel full, dropped {} samples", samples_per_chunk);
                        }
                    }
                } else {
                    // Stopping: store ALL remaining samples (full
                    // chunks + partial) in the shared buffer for
                    // deterministic delivery by `stop()`.  Do NOT use
                    // `try_send` — the channel may be full and we
                    // must not lose the tail of the recording.
                    let all_remaining = std::mem::take(&mut buffer);
                    if !all_remaining.is_empty() {
                        if let Ok(mut guard) = final_buffer.lock() {
                            *guard = Some(all_remaining);
                        }
                    }
                    flushed = true;
                    if let Some(tx) = flush_done_tx.take() {
                        let _ = tx.send(());
                    }
                }
            },
            move |err| {
                if let Ok(mut guard) = last_error.lock() {
                    *guard = Some(err.to_string());
                }
                running_for_error.store(false, Ordering::Release);
            },
            None,
        )
        .map_err(|err| TalkError::Audio(format!("Failed to build stream: {}", err)))
}
