//! CPAL-based audio capture implementation.

use super::{AudioCapture, CHANNEL_CAPACITY, CHUNK_DURATION_MS};
use crate::core::config::AudioConfig;
use crate::core::error::TalkError;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Audio capture using the CPAL backend.
pub struct CpalCapture {
    config: AudioConfig,
    stream: Mutex<Option<cpal::Stream>>,
    running: Arc<AtomicBool>,
    last_error: Arc<Mutex<Option<String>>>,
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
        }
    }

    /// Return the last error captured by the audio backend.
    pub fn last_error(&self) -> Result<Option<String>, TalkError> {
        self.last_error
            .lock()
            .map(|value| value.clone())
            .map_err(|_| TalkError::Audio("last_error lock poisoned".to_string()))
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

        let stream = match sample_format {
            SampleFormat::I8 => build_stream::<i8>(
                &device,
                &stream_config,
                sender,
                running,
                last_error,
                samples_per_chunk,
            ),
            SampleFormat::U8 => build_stream::<u8>(
                &device,
                &stream_config,
                sender,
                running,
                last_error,
                samples_per_chunk,
            ),
            SampleFormat::I16 => build_stream::<i16>(
                &device,
                &stream_config,
                sender,
                running,
                last_error,
                samples_per_chunk,
            ),
            SampleFormat::U16 => build_stream::<u16>(
                &device,
                &stream_config,
                sender,
                running,
                last_error,
                samples_per_chunk,
            ),
            SampleFormat::I32 => build_stream::<i32>(
                &device,
                &stream_config,
                sender,
                running,
                last_error,
                samples_per_chunk,
            ),
            SampleFormat::U32 => build_stream::<u32>(
                &device,
                &stream_config,
                sender,
                running,
                last_error,
                samples_per_chunk,
            ),
            SampleFormat::I64 => build_stream::<i64>(
                &device,
                &stream_config,
                sender,
                running,
                last_error,
                samples_per_chunk,
            ),
            SampleFormat::U64 => build_stream::<u64>(
                &device,
                &stream_config,
                sender,
                running,
                last_error,
                samples_per_chunk,
            ),
            SampleFormat::F32 => build_stream::<f32>(
                &device,
                &stream_config,
                sender,
                running,
                last_error,
                samples_per_chunk,
            ),
            SampleFormat::F64 => build_stream::<f64>(
                &device,
                &stream_config,
                sender,
                running,
                last_error,
                samples_per_chunk,
            ),
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

        Ok(receiver)
    }

    fn stop(&mut self) -> Result<(), TalkError> {
        self.running.store(false, Ordering::Release);

        let mut guard = self
            .stream
            .lock()
            .map_err(|_| TalkError::Audio("stream lock poisoned".to_string()))?;
        let _ = guard.take();

        Ok(())
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sender: mpsc::Sender<Vec<i16>>,
    running: Arc<AtomicBool>,
    last_error: Arc<Mutex<Option<String>>>,
    samples_per_chunk: usize,
) -> Result<cpal::Stream, TalkError>
where
    T: Sample + SizedSample + Copy,
    i16: FromSample<T>,
{
    let mut buffer: Vec<i16> = Vec::with_capacity(samples_per_chunk);

    let running_for_data = Arc::clone(&running);
    let running_for_error = Arc::clone(&running);

    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                if !running_for_data.load(Ordering::Acquire) {
                    return;
                }

                buffer.extend(data.iter().map(|sample| i16::from_sample(*sample)));

                while buffer.len() >= samples_per_chunk {
                    let chunk: Vec<i16> = buffer.drain(..samples_per_chunk).collect();
                    let _ = sender.try_send(chunk);
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
