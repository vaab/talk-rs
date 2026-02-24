//! PipeWire monitor+mic mixed capture.
//!
//! Opens two PipeWire streams on a single main loop:
//!
//! - **Mic** – default audio source (microphone).
//! - **Monitor** – monitor port of the default audio sink (system
//!   playback, e.g. conference call audio).
//!
//! Both streams deliver s16le PCM at the same rate and channel count.
//! A shared [`MixerState`] accumulates samples from each stream and
//! emits mixed, fixed-size chunks through a tokio mpsc channel.
//!
//! Because both streams live on the same PipeWire main loop their
//! callbacks never run concurrently, so the shared state uses
//! `Rc<RefCell<>>` instead of a mutex.

use super::pipewire_capture::serialize_audio_info;
use super::{AudioCapture, CHANNEL_CAPACITY, CHUNK_DURATION_MS};
use crate::config::AudioConfig;
use crate::error::TalkError;
use tokio::sync::mpsc;

use pipewire as pw;
use pw::spa;
use spa::param::audio::{AudioFormat, AudioInfoRaw};
use spa::pod::Pod;
use spa::utils::Direction;

use std::cell::RefCell;
use std::rc::Rc;

/// Signal sent to the PipeWire main loop to request shutdown.
struct Terminate;

/// Shared mixing state for the mic and monitor callbacks.
///
/// Both callbacks run on the PipeWire main-loop thread and push
/// samples into their respective buffer.  After every push the state
/// checks whether both buffers have accumulated at least one full
/// chunk; if so it mixes them with saturating addition and sends the
/// result through `audio_tx`.
struct MixerState {
    mic_buf: Vec<i16>,
    mon_buf: Vec<i16>,
    audio_tx: mpsc::Sender<Vec<i16>>,
    samples_per_chunk: usize,
}

impl MixerState {
    fn new(audio_tx: mpsc::Sender<Vec<i16>>, samples_per_chunk: usize) -> Self {
        Self {
            mic_buf: Vec::with_capacity(samples_per_chunk * 2),
            mon_buf: Vec::with_capacity(samples_per_chunk * 2),
            audio_tx,
            samples_per_chunk,
        }
    }

    /// Append microphone samples and try to emit mixed chunks.
    fn push_mic(&mut self, samples: &[i16]) {
        self.mic_buf.extend_from_slice(samples);
        self.try_emit();
    }

    /// Append monitor samples and try to emit mixed chunks.
    fn push_mon(&mut self, samples: &[i16]) {
        self.mon_buf.extend_from_slice(samples);
        self.try_emit();
    }

    /// Mix and emit chunks while both buffers have enough data.
    fn try_emit(&mut self) {
        let n = self.samples_per_chunk;
        while self.mic_buf.len() >= n && self.mon_buf.len() >= n {
            let mixed: Vec<i16> = self
                .mic_buf
                .drain(..n)
                .zip(self.mon_buf.drain(..n))
                .map(|(m, s)| m.saturating_add(s))
                .collect();
            if self.audio_tx.try_send(mixed).is_err() {
                log::warn!("audio channel full, dropped {} samples", n);
            }
        }
    }

    /// Flush any remaining samples from the mic-only or monitor-only
    /// buffer when one side has closed.  This ensures the tail of the
    /// recording is not lost.
    fn flush_remaining(&mut self) {
        let n = self.samples_per_chunk;

        // If one buffer still has data, send it as-is (no mixing partner).
        for buf in [&mut self.mic_buf, &mut self.mon_buf] {
            while buf.len() >= n {
                let chunk: Vec<i16> = buf.drain(..n).collect();
                if self.audio_tx.try_send(chunk).is_err() {
                    log::warn!("audio channel full during flush");
                }
            }
            // Send any remaining partial chunk.
            if !buf.is_empty() {
                let partial: Vec<i16> = std::mem::take(buf);
                if self.audio_tx.try_send(partial).is_err() {
                    log::warn!("audio channel full during partial flush");
                }
            }
        }
    }
}

/// Audio capture that mixes microphone and system-audio monitor.
pub struct MonitorCapture {
    config: AudioConfig,
    thread_handle: Option<std::thread::JoinHandle<()>>,
    quit_sender: Option<pw::channel::Sender<Terminate>>,
}

impl MonitorCapture {
    /// Create a new monitor capture with the given audio configuration.
    pub fn new(config: AudioConfig) -> Self {
        Self {
            config,
            thread_handle: None,
            quit_sender: None,
        }
    }
}

impl AudioCapture for MonitorCapture {
    fn start(&mut self) -> Result<mpsc::Receiver<Vec<i16>>, TalkError> {
        if self.thread_handle.is_some() {
            return Err(TalkError::Audio("MonitorCapture already running".into()));
        }

        let (audio_tx, audio_rx) = mpsc::channel(CHANNEL_CAPACITY);
        let (quit_tx, quit_rx) = pw::channel::channel::<Terminate>();

        let rate = self.config.sample_rate;
        let channels = self.config.channels;
        let frames_per_chunk = (rate as usize * CHUNK_DURATION_MS as usize) / 1000;
        let samples_per_chunk = frames_per_chunk * channels as usize;

        let handle = std::thread::Builder::new()
            .name("pipewire-monitor".into())
            .spawn(move || {
                if let Err(e) =
                    run_monitor_loop(quit_rx, audio_tx, rate, channels, samples_per_chunk)
                {
                    log::error!("PipeWire monitor capture error: {}", e);
                }
            })
            .map_err(|e| {
                TalkError::Audio(format!("failed to spawn PipeWire monitor thread: {}", e))
            })?;

        self.thread_handle = Some(handle);
        self.quit_sender = Some(quit_tx);

        Ok(audio_rx)
    }

    fn stop(&mut self) -> Result<(), TalkError> {
        if let Some(tx) = self.quit_sender.take() {
            let _ = tx.send(Terminate);
        }
        if let Some(handle) = self.thread_handle.take() {
            if handle.join().is_err() {
                log::warn!("PipeWire monitor thread panicked");
            }
        }
        Ok(())
    }
}

// See `pipewire_capture.rs` for the safety rationale.
unsafe impl Send for MonitorCapture {}

// ── PipeWire main-loop with two streams ─────────────────────────────

/// Run a PipeWire main loop that hosts a mic stream and a monitor
/// stream, mixing their output into `audio_tx`.
fn run_monitor_loop(
    quit_rx: pw::channel::Receiver<Terminate>,
    audio_tx: mpsc::Sender<Vec<i16>>,
    rate: u32,
    channels: u8,
    samples_per_chunk: usize,
) -> Result<(), TalkError> {
    let mainloop = pw::main_loop::MainLoopRc::new(None)
        .map_err(|e| TalkError::Audio(format!("PipeWire MainLoop: {}", e)))?;

    let _quit = quit_rx.attach(mainloop.loop_(), {
        let ml = mainloop.clone();
        move |_| ml.quit()
    });

    let context = pw::context::ContextRc::new(&mainloop, None)
        .map_err(|e| TalkError::Audio(format!("PipeWire Context: {}", e)))?;

    let core = context
        .connect_rc(None)
        .map_err(|e| TalkError::Audio(format!("PipeWire connect: {}", e)))?;

    // ── Shared mixer state ──────────────────────────────────────────
    let state = Rc::new(RefCell::new(MixerState::new(audio_tx, samples_per_chunk)));

    // ── Build format pod (shared by both streams) ───────────────────
    let mut audio_info = AudioInfoRaw::new();
    audio_info.set_format(AudioFormat::S16LE);
    audio_info.set_rate(rate);
    audio_info.set_channels(channels as u32);

    let format_bytes = serialize_audio_info(audio_info)?;

    // We need two copies because each connect() call consumes the slice.
    let mic_pod_bytes = format_bytes.clone();
    let mon_pod_bytes = format_bytes;

    // ── Stream 1: Microphone (default source) ───────────────────────
    let mic_props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE     => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE     => "Speech",
    };

    let mic_stream = pw::stream::StreamRc::new(core.clone(), "talk-rs-mic", mic_props)
        .map_err(|e| TalkError::Audio(format!("PipeWire mic stream: {}", e)))?;

    let mic_state = Rc::clone(&state);
    let _mic_listener = mic_stream
        .add_local_listener()
        .process(move |stream_ref, _: &mut ()| {
            decode_and_push(stream_ref, &mic_state, true);
        })
        .register()
        .map_err(|e| TalkError::Audio(format!("PipeWire mic listener: {}", e)))?;

    let mic_pod = Pod::from_bytes(&mic_pod_bytes)
        .ok_or_else(|| TalkError::Audio("invalid mic format pod".into()))?;

    mic_stream
        .connect(
            Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut [mic_pod],
        )
        .map_err(|e| TalkError::Audio(format!("PipeWire mic connect: {}", e)))?;

    log::info!(
        "PipeWire mic capture started: {}Hz, {} ch, s16le",
        rate,
        channels
    );

    // ── Stream 2: Monitor (default sink monitor) ────────────────────
    let mon_props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE         => "Audio",
        *pw::keys::MEDIA_CATEGORY     => "Capture",
        *pw::keys::MEDIA_ROLE         => "Speech",
        *pw::keys::STREAM_CAPTURE_SINK => "true",
    };

    let mon_stream = pw::stream::StreamRc::new(core.clone(), "talk-rs-monitor", mon_props)
        .map_err(|e| TalkError::Audio(format!("PipeWire monitor stream: {}", e)))?;

    let mon_state = Rc::clone(&state);
    let _mon_listener = mon_stream
        .add_local_listener()
        .process(move |stream_ref, _: &mut ()| {
            decode_and_push(stream_ref, &mon_state, false);
        })
        .register()
        .map_err(|e| TalkError::Audio(format!("PipeWire monitor listener: {}", e)))?;

    let mon_pod = Pod::from_bytes(&mon_pod_bytes)
        .ok_or_else(|| TalkError::Audio("invalid monitor format pod".into()))?;

    mon_stream
        .connect(
            Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut [mon_pod],
        )
        .map_err(|e| TalkError::Audio(format!("PipeWire monitor connect: {}", e)))?;

    log::info!(
        "PipeWire monitor capture started: {}Hz, {} ch, s16le",
        rate,
        channels
    );

    // ── Run ─────────────────────────────────────────────────────────
    mainloop.run();

    // Flush any buffered samples after the loop exits.
    if let Ok(mut s) = state.try_borrow_mut() {
        s.flush_remaining();
    }

    log::debug!("PipeWire monitor capture mainloop exited");
    Ok(())
}

/// Decode s16le bytes from a PipeWire buffer and push them into the
/// shared [`MixerState`].
fn decode_and_push(stream_ref: &pw::stream::Stream, state: &Rc<RefCell<MixerState>>, is_mic: bool) {
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

    let samples: Vec<i16> = pcm
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect();

    let mut s = state.borrow_mut();
    if is_mic {
        s.push_mic(&samples);
    } else {
        s.push_mon(&samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_emits_when_both_buffers_full() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut mixer = MixerState::new(tx, 4);

        // Push 4 mic samples.
        mixer.push_mic(&[100, 200, 300, 400]);
        // Not enough monitor samples yet — nothing emitted.
        assert!(rx.try_recv().is_err());

        // Push 4 monitor samples — should emit one mixed chunk.
        mixer.push_mon(&[10, 20, 30, 40]);
        let chunk = rx.try_recv().expect("should have emitted a chunk");
        assert_eq!(chunk, vec![110, 220, 330, 440]);
    }

    #[test]
    fn test_mixer_saturating_add() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut mixer = MixerState::new(tx, 2);

        mixer.push_mic(&[i16::MAX, i16::MIN]);
        mixer.push_mon(&[1, -1]);

        let chunk = rx.try_recv().expect("should have emitted a chunk");
        assert_eq!(chunk, vec![i16::MAX, i16::MIN]);
    }

    #[test]
    fn test_mixer_multiple_chunks() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut mixer = MixerState::new(tx, 2);

        mixer.push_mic(&[1, 2, 3, 4]);
        mixer.push_mon(&[10, 20, 30, 40]);

        let c1 = rx.try_recv().expect("first chunk");
        let c2 = rx.try_recv().expect("second chunk");
        assert_eq!(c1, vec![11, 22]);
        assert_eq!(c2, vec![33, 44]);
    }

    #[test]
    fn test_mixer_flush_remaining() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut mixer = MixerState::new(tx, 4);

        // Only mic has data.
        mixer.push_mic(&[1, 2, 3, 4, 5]);
        assert!(rx.try_recv().is_err());

        mixer.flush_remaining();
        let full = rx.try_recv().expect("full chunk from flush");
        assert_eq!(full, vec![1, 2, 3, 4]);
        let partial = rx.try_recv().expect("partial chunk from flush");
        assert_eq!(partial, vec![5]);
    }
}
