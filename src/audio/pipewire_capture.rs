//! Native PipeWire audio capture.
//!
//! Captures audio directly from PipeWire using the `pipewire` Rust
//! bindings, matching the audio path of `pw-cat --record`.  This
//! ensures talk-rs captures from the same PipeWire default source
//! (including Bluetooth devices) with identical routing.

use super::{
    AudioCapture, CaptureReadiness, CaptureReady, CaptureReadyReceiver, CHANNEL_CAPACITY,
    CHUNK_DURATION_MS,
};
use crate::config::AudioConfig;
use crate::error::TalkError;
use std::collections::VecDeque;
use tokio::sync::{mpsc, oneshot};

use pipewire as pw;
use pw::spa;
use spa::param::audio::{AudioFormat, AudioInfoRaw};
use spa::pod::{self, Pod, Value};
use spa::utils::Direction;

/// Signal sent to the PipeWire main loop to request shutdown.
pub(crate) struct Terminate;

/// Turns PipeWire stream events into the capture readiness proof.
///
/// Readiness is proven when the stream reached
/// [`pw::stream::StreamState::Streaming`] AND then delivered a buffer
/// with samples AND, for a Bluetooth HFP source, the kernel reports the
/// voice link (SCO/eSCO) to that headset as connected.
///
/// Sample VALUES are never used: a Bluetooth headset microphone streams
/// exact zeros both before its voice link is up and whenever its noise
/// gate is closed (silent user), so zeros can mean either.  The voice
/// link state tells them apart; it is checked at each delivered buffer
/// until proven.  A stream error or disconnection before readiness is
/// reported as a failure naming the source.
pub(crate) struct ReadinessProbe {
    source: String,
    streaming: bool,
    link_check: Option<crate::audio::bt_link::LinkCheck>,
    waiting_logged: bool,
    tx: Option<oneshot::Sender<Result<CaptureReady, TalkError>>>,
}

impl ReadinessProbe {
    /// Probe for `source`, with the voice-link check Bluetooth HFP
    /// sources need (`None` for other microphones).
    pub(crate) fn new(
        source: String,
        link_check: Option<crate::audio::bt_link::LinkCheck>,
    ) -> (Self, CaptureReadyReceiver) {
        let (tx, rx) = oneshot::channel();
        (
            Self {
                source,
                streaming: false,
                link_check,
                waiting_logged: false,
                tx: Some(tx),
            },
            rx,
        )
    }

    /// Probe for `source`, checking the Bluetooth voice link when
    /// `source` is a Bluetooth HFP microphone.
    pub(crate) fn for_source(source: String) -> (Self, CaptureReadyReceiver) {
        let link_check = crate::audio::bt_link::voice_link_check(&source);
        Self::new(source, link_check)
    }

    /// Record a stream state transition.
    pub(crate) fn on_state(
        &mut self,
        old: &pw::stream::StreamState,
        new: &pw::stream::StreamState,
    ) {
        use pw::stream::StreamState;
        log::debug!(
            "capture stream on '{}': {:?} -> {:?}",
            self.source,
            old,
            new
        );
        match new {
            StreamState::Streaming => self.streaming = true,
            StreamState::Error(message) => self.fail(format!(
                "capture stream on '{}' failed: {}",
                self.source, message
            )),
            StreamState::Unconnected if !matches!(old, StreamState::Unconnected) => {
                self.fail(format!(
                    "capture stream on '{}' was disconnected before delivering audio",
                    self.source
                ))
            }
            _ => self.streaming = false,
        }
    }

    /// Record a delivered buffer.
    pub(crate) fn on_buffer(&mut self, samples: &[i16]) {
        if self.tx.is_none() || !self.streaming || samples.is_empty() {
            return;
        }
        if let Some(check) = self.link_check.as_mut() {
            match check() {
                Ok(true) => {}
                Ok(false) => {
                    if !self.waiting_logged {
                        self.waiting_logged = true;
                        log::debug!(
                            "capture on '{}': waiting for the Bluetooth voice link",
                            self.source
                        );
                    }
                    return;
                }
                Err(error) => {
                    self.fail(format!(
                        "cannot check the Bluetooth voice link of '{}': {}",
                        self.source, error
                    ));
                    return;
                }
            }
        }
        if let Some(tx) = self.tx.take() {
            let ready = CaptureReady {
                source: self.source.clone(),
                first_buffer_at: std::time::Instant::now(),
            };
            log::info!(
                "capture ready: '{}' streaming{}",
                ready.source,
                if self.link_check.is_some() {
                    " with its Bluetooth voice link up"
                } else {
                    ""
                }
            );
            let _ = tx.send(Ok(ready));
        }
    }

    fn fail(&mut self, message: String) {
        if let Some(tx) = self.tx.take() {
            log::error!("{}", message);
            let _ = tx.send(Err(TalkError::Audio(message)));
        }
    }
}

/// Forwards fixed-size chunks to the pipeline without ever dropping one.
///
/// The PipeWire callback must not block, so a chunk that does not fit in
/// the bounded channel is kept in an in-order overflow queue and sent
/// first on the next delivery.  Chunks are only discarded once the
/// receiver is gone (nobody left to record them).
pub(crate) struct ChunkForwarder {
    tx: mpsc::Sender<Vec<i16>>,
    overflow: VecDeque<Vec<i16>>,
}

impl ChunkForwarder {
    pub(crate) fn new(tx: mpsc::Sender<Vec<i16>>) -> Self {
        Self {
            tx,
            overflow: VecDeque::new(),
        }
    }

    /// Queue `chunk` after every chunk not yet delivered, then deliver as
    /// many as the channel accepts.
    pub(crate) fn push(&mut self, chunk: Vec<i16>) {
        self.overflow.push_back(chunk);
        self.drain();
    }

    /// Deliver queued chunks in order until the channel is full.
    pub(crate) fn drain(&mut self) {
        while let Some(chunk) = self.overflow.pop_front() {
            match self.tx.try_send(chunk) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(chunk)) => {
                    self.overflow.push_front(chunk);
                    break;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    self.overflow.clear();
                    break;
                }
            }
        }
    }

    /// Chunks still waiting for room in the channel.
    pub(crate) fn pending(&self) -> usize {
        self.overflow.len()
    }
}

/// Stream properties targeting exactly `source`, never another device.
///
/// `target.object` names the PipeWire node (equal to the PulseAudio
/// source name under pipewire-pulse); `node.dont-reconnect` makes the
/// session manager neither fall back to another node when the target is
/// missing nor move the stream later: the stream then fails instead.
pub(crate) fn capture_properties(role: &str, source: &str) -> pw::properties::PropertiesBox {
    pw::properties::properties! {
        *pw::keys::MEDIA_TYPE          => "Audio",
        *pw::keys::MEDIA_CATEGORY      => "Capture",
        *pw::keys::MEDIA_ROLE          => role,
        *pw::keys::TARGET_OBJECT       => source,
        *pw::keys::NODE_DONT_RECONNECT => "true",
    }
}

/// Audio capture using native PipeWire bindings.
pub struct PipeWireCapture {
    config: AudioConfig,
    /// Source (node name) the stream is explicitly linked to.
    source: String,
    /// Readiness proof channel, taken once by [`AudioCapture::readiness`].
    ready_rx: Option<CaptureReadyReceiver>,
    /// Handle to the dedicated PipeWire thread.
    thread_handle: Option<std::thread::JoinHandle<()>>,
    /// Channel to signal the PipeWire main loop to quit.
    quit_sender: Option<pw::channel::Sender<Terminate>>,
}

impl PipeWireCapture {
    /// Create a PipeWire capture explicitly targeted at `source` (a
    /// PipeWire node name / PulseAudio source name).
    pub fn new(config: AudioConfig, source: String) -> Self {
        Self {
            config,
            source,
            ready_rx: None,
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
        let (probe, ready_rx) = ReadinessProbe::for_source(self.source.clone());
        let source = self.source.clone();

        let handle = std::thread::Builder::new()
            .name("pipewire-capture".into())
            .spawn(move || {
                // On setup failure the probe is dropped inside, closing
                // the readiness channel: the waiter reports it.
                if let Err(e) = run_capture_loop(
                    quit_rx,
                    audio_tx,
                    rate,
                    channels,
                    samples_per_chunk,
                    &source,
                    probe,
                ) {
                    log::error!("PipeWire capture error: {}", e);
                }
            })
            .map_err(|e| TalkError::Audio(format!("failed to spawn PipeWire thread: {}", e)))?;

        self.thread_handle = Some(handle);
        self.quit_sender = Some(quit_tx);
        self.ready_rx = Some(ready_rx);

        Ok(audio_rx)
    }

    fn readiness(&mut self) -> CaptureReadiness {
        match self.ready_rx.take() {
            Some(rx) => CaptureReadiness::Pending(rx),
            None => {
                // Not started (or already taken): a closed channel makes
                // the waiter fail loudly rather than assume readiness.
                let (_, rx) = oneshot::channel();
                CaptureReadiness::Pending(rx)
            }
        }
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
    source: &str,
    probe: ReadinessProbe,
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

    let props = capture_properties("Speech", source);

    let stream = pw::stream::StreamRc::new(core.clone(), "talk-rs", props)
        .map_err(|e| TalkError::Audio(format!("PipeWire Stream: {}", e)))?;

    // Buffer to accumulate samples and emit fixed-size chunks.
    // PipeWire delivers buffers at the graph quantum (typically ~1024
    // frames ≈ 21.3 ms at 48 kHz), which may not match our 20 ms
    // chunk size exactly.
    //
    // Shared with the main scope so we can flush remaining samples
    // after `mainloop.run()` returns.
    use std::cell::RefCell;
    use std::rc::Rc;
    let buffer: Rc<RefCell<Vec<i16>>> =
        Rc::new(RefCell::new(Vec::with_capacity(samples_per_chunk * 2)));
    let forwarder = Rc::new(RefCell::new(ChunkForwarder::new(audio_tx)));
    let probe = Rc::new(RefCell::new(probe));

    let _listener = stream
        .add_local_listener()
        .state_changed({
            let probe = Rc::clone(&probe);
            move |_, _: &mut (), old, new| probe.borrow_mut().on_state(&old, &new)
        })
        .process({
            let buf = Rc::clone(&buffer);
            let forwarder = Rc::clone(&forwarder);
            let probe = Rc::clone(&probe);
            move |stream_ref, _: &mut ()| {
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
                let samples: Vec<i16> = pcm
                    .chunks_exact(2)
                    .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                    .collect();
                probe.borrow_mut().on_buffer(&samples);

                let mut buffer = buf.borrow_mut();
                buffer.extend_from_slice(&samples);

                // Emit fixed-size chunks, never dropping one.
                let mut forwarder = forwarder.borrow_mut();
                forwarder.drain();
                while buffer.len() >= samples_per_chunk {
                    let chunk: Vec<i16> = buffer.drain(..samples_per_chunk).collect();
                    forwarder.push(chunk);
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
            None, // target given by `target.object`; no default-source fallback
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut [format_pod],
        )
        .map_err(|e| TalkError::Audio(format!("PipeWire stream connect: {}", e)))?;

    log::info!(
        "PipeWire capture started on '{}': {}Hz, {} ch, s16le",
        source,
        rate,
        channels
    );

    // Run the main loop — blocks until quit signal.
    mainloop.run();

    // Flush any remaining samples in the accumulation buffer.
    // Full chunks are sent as-is; a trailing partial chunk (< one
    // full chunk) is also emitted so the last fraction of audio is
    // never silently lost.
    let remaining = buffer.borrow_mut().split_off(0);
    if !remaining.is_empty() {
        log::debug!(
            "flushing {} residual samples ({} full chunks + {} trailing)",
            remaining.len(),
            remaining.len() / samples_per_chunk,
            remaining.len() % samples_per_chunk,
        );
        let mut forwarder = forwarder.borrow_mut();
        for chunk in remaining.chunks(samples_per_chunk) {
            forwarder.push(chunk.to_vec());
        }
    }
    let pending = forwarder.borrow().pending();
    if pending > 0 {
        log::warn!(
            "audio channel full at capture stop, dropped {} trailing chunks",
            pending
        );
    }

    log::debug!("PipeWire capture mainloop exited");
    Ok(())
}

/// Serialize `AudioInfoRaw` into a SPA Pod byte vector.
pub(crate) fn serialize_audio_info(info: AudioInfoRaw) -> Result<Vec<u8>, TalkError> {
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
    use pw::stream::StreamState;

    fn proof(rx: &mut CaptureReadyReceiver) -> Option<Result<CaptureReady, TalkError>> {
        rx.try_recv().ok()
    }

    /// Shared scripted voice-link state for tests.
    fn scripted_link(
        state: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> crate::audio::bt_link::LinkCheck {
        Box::new(move || Ok(state.load(std::sync::atomic::Ordering::SeqCst)))
    }

    #[test]
    fn readiness_requires_streaming_state_before_a_buffer_counts() {
        let (mut probe, mut rx) = ReadinessProbe::new("alsa_input.mic".into(), None);
        // A buffer delivered before the stream is streaming proves nothing.
        probe.on_buffer(&[100; 960]);
        probe.on_state(&StreamState::Unconnected, &StreamState::Connecting);
        probe.on_state(&StreamState::Connecting, &StreamState::Paused);
        probe.on_buffer(&[100; 960]);
        assert!(proof(&mut rx).is_none());

        probe.on_state(&StreamState::Paused, &StreamState::Streaming);
        assert!(proof(&mut rx).is_none(), "streaming alone is not proof");
        let before = std::time::Instant::now();
        probe.on_buffer(&[100; 960]);
        match proof(&mut rx) {
            Some(Ok(ready)) => {
                assert_eq!(ready.source, "alsa_input.mic");
                assert!(ready.first_buffer_at >= before);
            }
            other => panic!("expected readiness proof, got {other:?}"),
        }
    }

    #[test]
    fn empty_buffer_while_streaming_is_not_proof() {
        let (mut probe, mut rx) = ReadinessProbe::new("mic".into(), None);
        probe.on_state(&StreamState::Paused, &StreamState::Streaming);
        probe.on_buffer(&[]);
        assert!(proof(&mut rx).is_none());
    }

    /// A silent wired/built-in microphone is ready: sample values are
    /// never part of the proof.
    #[test]
    fn silent_non_bluetooth_microphone_is_ready_on_first_buffer() {
        let (mut probe, mut rx) = ReadinessProbe::new("alsa_input.mic".into(), None);
        probe.on_state(&StreamState::Paused, &StreamState::Streaming);
        probe.on_buffer(&[0; 960]);
        assert!(matches!(proof(&mut rx), Some(Ok(_))));
    }

    /// Bluetooth HFP: buffers stream (as zeros) before the voice link is
    /// up; readiness waits for the link, then holds even if the user is
    /// silent (noise-gated headset still delivering zeros).
    #[test]
    fn bluetooth_readiness_follows_the_voice_link_not_the_samples() {
        let link = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (mut probe, mut rx) = ReadinessProbe::new(
            "bluez_input.7C_E9_13_93_6C_E6.0".into(),
            Some(scripted_link(std::sync::Arc::clone(&link))),
        );
        probe.on_state(&StreamState::Paused, &StreamState::Streaming);
        for _ in 0..37 {
            probe.on_buffer(&[0; 960]);
        }
        // Even real signal is no proof while the link is reported down.
        probe.on_buffer(&[500; 960]);
        assert!(proof(&mut rx).is_none(), "voice link not up yet");
        link.store(true, std::sync::atomic::Ordering::SeqCst);
        probe.on_buffer(&[0; 960]);
        assert!(
            matches!(proof(&mut rx), Some(Ok(_))),
            "silent user, link up"
        );
    }

    #[test]
    fn failing_voice_link_check_fails_readiness() {
        let (mut probe, mut rx) = ReadinessProbe::new(
            "bluez_input.7C_E9_13_93_6C_E6.0".into(),
            Some(Box::new(|| Err(TalkError::Audio("no HCI".into())))),
        );
        probe.on_state(&StreamState::Paused, &StreamState::Streaming);
        probe.on_buffer(&[0; 960]);
        match proof(&mut rx) {
            Some(Err(error)) => assert!(error.to_string().contains("no HCI")),
            other => panic!("expected failure, got {other:?}"),
        }
    }

    #[test]
    fn stream_error_before_first_buffer_fails_naming_the_source() {
        let (mut probe, mut rx) = ReadinessProbe::new("bluez_input.headset".into(), None);
        probe.on_state(
            &StreamState::Connecting,
            &StreamState::Error("no target".into()),
        );
        match proof(&mut rx) {
            Some(Err(error)) => {
                let text = error.to_string();
                assert!(text.contains("bluez_input.headset"), "{text}");
                assert!(text.contains("no target"), "{text}");
            }
            other => panic!("expected failure, got {other:?}"),
        }
    }

    #[test]
    fn disconnection_before_first_buffer_fails() {
        let (mut probe, mut rx) = ReadinessProbe::new("mic".into(), None);
        probe.on_state(&StreamState::Unconnected, &StreamState::Connecting);
        probe.on_state(&StreamState::Connecting, &StreamState::Unconnected);
        assert!(matches!(proof(&mut rx), Some(Err(_))));
    }

    #[test]
    fn forwarder_keeps_every_chunk_in_order_when_channel_is_full() {
        let (tx, mut rx) = mpsc::channel(2);
        let mut forwarder = ChunkForwarder::new(tx);
        let chunks: Vec<Vec<i16>> = (0..10).map(|i| vec![i; 4]).collect();
        let mut received = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            forwarder.push(chunk.clone());
            // The consumer only drains every third delivery.
            if index % 3 == 2 {
                while let Ok(chunk) = rx.try_recv() {
                    received.push(chunk);
                }
            }
        }
        while forwarder.pending() > 0 {
            while let Ok(chunk) = rx.try_recv() {
                received.push(chunk);
            }
            forwarder.drain();
        }
        while let Ok(chunk) = rx.try_recv() {
            received.push(chunk);
        }
        assert_eq!(received, chunks);
    }

    #[test]
    fn capture_properties_pin_the_target_without_fallback() {
        let props = capture_properties("Speech", "bluez_input.AA_BB.0");
        assert_eq!(props.get("target.object"), Some("bluez_input.AA_BB.0"));
        assert_eq!(props.get("node.dont-reconnect"), Some("true"));
        assert_eq!(props.get("media.role"), Some("Speech"));
    }

    #[test]
    fn test_serialize_audio_info() {
        let mut info = AudioInfoRaw::new();
        info.set_format(AudioFormat::S16LE);
        info.set_rate(48_000);
        info.set_channels(1);

        let expected = Value::Object(pod::Object {
            type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
            id: spa::param::ParamType::EnumFormat.as_raw(),
            properties: info.into(),
        });
        let mut input = AudioInfoRaw::new();
        input.set_format(AudioFormat::S16LE);
        input.set_rate(48_000);
        input.set_channels(1);
        let bytes = serialize_audio_info(input).expect("serialize audio format");
        let (remaining, actual) =
            spa::pod::deserialize::PodDeserializer::deserialize_any_from(&bytes)
                .expect("decode SPA Pod");
        assert_eq!(remaining, &[] as &[u8]);
        assert_eq!(actual, expected);
    }
}
