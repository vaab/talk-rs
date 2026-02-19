//! Native PipeWire audio capture.
//!
//! Captures audio directly from PipeWire using the `pipewire` Rust
//! bindings, matching the audio path of `pw-cat --record`.  This
//! ensures talk-rs captures from the same PipeWire default source
//! (including Bluetooth devices) with identical routing.

use super::{AudioCapture, CHANNEL_CAPACITY, CHUNK_DURATION_MS};
use crate::core::config::AudioConfig;
use crate::core::error::TalkError;
use tokio::sync::mpsc;

use pipewire as pw;
use pw::spa;
use spa::param::audio::{AudioFormat, AudioInfoRaw};
use spa::pod::{self, Pod, Value};
use spa::utils::Direction;

/// Signal sent to the PipeWire main loop to request shutdown.
struct Terminate;

/// Audio capture using native PipeWire bindings.
pub struct PipeWireCapture {
    config: AudioConfig,
    /// Handle to the dedicated PipeWire thread.
    thread_handle: Option<std::thread::JoinHandle<()>>,
    /// Channel to signal the PipeWire main loop to quit.
    quit_sender: Option<pw::channel::Sender<Terminate>>,
}

impl PipeWireCapture {
    /// Create a new PipeWire capture with the given audio configuration.
    pub fn new(config: AudioConfig) -> Self {
        Self {
            config,
            thread_handle: None,
            quit_sender: None,
        }
    }
}

impl AudioCapture for PipeWireCapture {
    fn start(&mut self) -> Result<mpsc::Receiver<Vec<i16>>, TalkError> {
        if self.thread_handle.is_some() {
            return Err(TalkError::Audio("PipeWire capture already running".into()));
        }

        let (audio_tx, audio_rx) = mpsc::channel(CHANNEL_CAPACITY);
        let (quit_tx, quit_rx) = pw::channel::channel::<Terminate>();

        let rate = self.config.sample_rate;
        let channels = self.config.channels;
        let frames_per_chunk = (rate as usize * CHUNK_DURATION_MS as usize) / 1000;
        let samples_per_chunk = frames_per_chunk * channels as usize;

        let handle = std::thread::Builder::new()
            .name("pipewire-capture".into())
            .spawn(move || {
                if let Err(e) =
                    run_capture_loop(quit_rx, audio_tx, rate, channels, samples_per_chunk)
                {
                    log::error!("PipeWire capture error: {}", e);
                }
            })
            .map_err(|e| TalkError::Audio(format!("failed to spawn PipeWire thread: {}", e)))?;

        self.thread_handle = Some(handle);
        self.quit_sender = Some(quit_tx);

        Ok(audio_rx)
    }

    fn stop(&mut self) -> Result<(), TalkError> {
        // Signal the PipeWire main loop to quit.
        if let Some(tx) = self.quit_sender.take() {
            let _ = tx.send(Terminate);
        }

        // Wait for the PipeWire thread to finish.
        if let Some(handle) = self.thread_handle.take() {
            if handle.join().is_err() {
                log::warn!("PipeWire capture thread panicked");
            }
        }

        Ok(())
    }
}

// `pw::channel::Sender` is designed for cross-thread use but is not
// marked `Send` because it holds a raw pointer to the loop source.
// We only keep the `Sender` side, which is safe to hold on any thread.
unsafe impl Send for PipeWireCapture {}

/// Run the PipeWire capture loop on a dedicated thread.
///
/// Blocks until `quit_rx` receives a `Terminate` signal.
fn run_capture_loop(
    quit_rx: pw::channel::Receiver<Terminate>,
    audio_tx: mpsc::Sender<Vec<i16>>,
    rate: u32,
    channels: u8,
    samples_per_chunk: usize,
) -> Result<(), TalkError> {
    let mainloop = pw::main_loop::MainLoopRc::new(None)
        .map_err(|e| TalkError::Audio(format!("PipeWire MainLoop: {}", e)))?;

    // Attach quit signal: when `stop()` sends Terminate, quit the loop.
    let _quit = quit_rx.attach(mainloop.loop_(), {
        let ml = mainloop.clone();
        move |_| ml.quit()
    });

    let context = pw::context::ContextRc::new(&mainloop, None)
        .map_err(|e| TalkError::Audio(format!("PipeWire Context: {}", e)))?;

    let core = context
        .connect_rc(None)
        .map_err(|e| TalkError::Audio(format!("PipeWire connect: {}", e)))?;

    let props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE     => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE     => "Speech",
    };

    let stream = pw::stream::StreamRc::new(core.clone(), "talk-rs", props)
        .map_err(|e| TalkError::Audio(format!("PipeWire Stream: {}", e)))?;

    // Buffer to accumulate samples and emit fixed-size chunks.
    // PipeWire delivers buffers at the graph quantum (typically ~1024
    // frames ≈ 21.3 ms at 48 kHz), which may not match our 20 ms
    // chunk size exactly.
    let mut buffer: Vec<i16> = Vec::with_capacity(samples_per_chunk * 2);

    let _listener = stream
        .add_local_listener()
        .process(move |stream_ref, _: &mut ()| {
            let Some(mut pw_buf) = stream_ref.dequeue_buffer() else {
                return;
            };
            let datas = pw_buf.datas_mut();
            if datas.is_empty() {
                return;
            }

            let size = datas[0].chunk().size() as usize;
            let Some(raw) = datas[0].data() else {
                return;
            };
            let pcm = &raw[..size];

            // Convert s16le bytes to i16 samples.
            buffer.extend(
                pcm.chunks_exact(2)
                    .map(|pair| i16::from_le_bytes([pair[0], pair[1]])),
            );

            // Emit fixed-size chunks.
            while buffer.len() >= samples_per_chunk {
                let chunk: Vec<i16> = buffer.drain(..samples_per_chunk).collect();
                if audio_tx.try_send(chunk).is_err() {
                    log::warn!("audio channel full, dropped {} samples", samples_per_chunk);
                }
            }
        })
        .register()
        .map_err(|e| TalkError::Audio(format!("PipeWire listener: {}", e)))?;

    // Build the format pod: s16le, requested rate, mono/stereo.
    let mut audio_info = AudioInfoRaw::new();
    audio_info.set_format(AudioFormat::S16LE);
    audio_info.set_rate(rate);
    audio_info.set_channels(channels as u32);

    let format_bytes = serialize_audio_info(audio_info)?;
    let format_pod = Pod::from_bytes(&format_bytes)
        .ok_or_else(|| TalkError::Audio("invalid audio format pod".into()))?;

    stream
        .connect(
            Direction::Input,
            None, // default source
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut [format_pod],
        )
        .map_err(|e| TalkError::Audio(format!("PipeWire stream connect: {}", e)))?;

    log::info!(
        "PipeWire capture started: {}Hz, {} ch, s16le",
        rate,
        channels
    );

    // Run the main loop — blocks until quit signal.
    mainloop.run();

    log::debug!("PipeWire capture mainloop exited");
    Ok(())
}

/// Serialize `AudioInfoRaw` into a SPA Pod byte vector.
fn serialize_audio_info(info: AudioInfoRaw) -> Result<Vec<u8>, TalkError> {
    use spa::pod::serialize::PodSerializer;
    use std::io;

    PodSerializer::serialize(
        io::Cursor::new(Vec::new()),
        &Value::Object(pod::Object {
            type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
            id: spa::param::ParamType::EnumFormat.as_raw(),
            properties: info.into(),
        }),
    )
    .map(|(cursor, _)| cursor.into_inner())
    .map_err(|e| TalkError::Audio(format!("audio format serialization: {:?}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_audio_info() {
        let mut info = AudioInfoRaw::new();
        info.set_format(AudioFormat::S16LE);
        info.set_rate(48_000);
        info.set_channels(1);

        let bytes = serialize_audio_info(info);
        assert!(bytes.is_ok());
        let bytes = bytes.unwrap();
        assert!(!bytes.is_empty());

        // Should be a valid Pod
        let pod = Pod::from_bytes(&bytes);
        assert!(pod.is_some());
    }
}
