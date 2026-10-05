//! Shared recording-time feedback for live capture workflows.
//!
//! This module owns the recording phase only: start/stop tones, the PCM
//! visualization tee, the recording badge, and the silence-gated boop.  It
//! deliberately does not own transcription states, text panels, telemetry
//! producers, paste behavior, or dead-audio policy.

use crate::audio::indicator::SoundPlayer;
use crate::audio::ring_buffer::RingBuffer;
use crate::config::VizMode;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[cfg(feature = "ui")]
use crate::telemetry::TranscriptionEvent;
#[cfg(feature = "ui")]
use crate::x11::overlay::OverlayHandle;

/// Recording-overlay inputs supplied by the workflow that owns higher-level
/// policies such as auto-pause, dead-audio notification, and telemetry.
#[cfg(feature = "ui")]
pub struct RecordingOverlayOptions {
    pub badge: crate::config::OverlaySettings,
    pub no_sound: crate::config::OverlaySettings,
    pub silence_tx: Option<std::sync::mpsc::Sender<bool>>,
    pub auto_pause: bool,
    pub telemetry_rx: Option<tokio::sync::broadcast::Receiver<TranscriptionEvent>>,
}

/// Runtime options shared by `dictate` and `record`.
pub struct RecordingFeedbackOptions {
    pub no_sounds: bool,
    pub no_boop: bool,
    pub no_overlay: bool,
    pub viz: Option<VizMode>,
    pub mono: bool,
    pub boop_interval_ms: u64,
    pub capture_rate: u32,
    /// Whether silence should pause downstream PCM forwarding.  Dictation
    /// enables this for auto-pause; standalone recording always disables it.
    pub pause_audio: bool,
    /// Independent negative gate owned by dictate's dead-audio alert policy.
    pub suppress_boop: Option<Arc<AtomicBool>>,
    #[cfg(feature = "ui")]
    pub overlay: RecordingOverlayOptions,
}

/// Whether the recording badge should disappear when recording ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordingBadgeTeardown {
    Hide,
    /// Leave the badge visible so dictate can immediately transition the same
    /// handle to its separately-owned transcribing state without flicker.
    KeepVisible,
}

pub struct RecordingFeedback {
    player: Option<SoundPlayer>,
    ring: Arc<Mutex<RingBuffer>>,
    pause_flag: Arc<AtomicBool>,
    boop_play_when: Arc<AtomicBool>,
    boop_token: Option<CancellationToken>,
    no_boop: bool,
    boop_interval_ms: u64,
    pause_audio: bool,
    suppress_boop: Option<Arc<AtomicBool>>,
    #[cfg(feature = "ui")]
    overlay: Option<OverlayHandle>,
    #[cfg(feature = "ui")]
    pending_overlay: Option<PendingOverlay>,
    #[cfg(test)]
    test_events: Option<Arc<Mutex<Vec<&'static str>>>>,
    #[cfg(test)]
    test_badge_available: bool,
}

#[cfg(feature = "ui")]
struct PendingOverlay {
    badge: crate::config::OverlaySettings,
    no_sound: crate::config::OverlaySettings,
    enabled: bool,
    viz: Option<VizMode>,
    mono: bool,
    capture_rate: u32,
    silence_tx: Option<std::sync::mpsc::Sender<bool>>,
    auto_pause: bool,
    telemetry_rx: Option<tokio::sync::broadcast::Receiver<TranscriptionEvent>>,
}

impl RecordingFeedback {
    pub fn new(options: RecordingFeedbackOptions) -> Self {
        let player = if options.no_sounds {
            log::debug!("sound indicators disabled");
            None
        } else {
            match SoundPlayer::new() {
                Ok(player) => {
                    log::debug!("sound player initialized");
                    Some(player)
                }
                Err(error) => {
                    log::warn!("sound indicators unavailable: {}", error);
                    None
                }
            }
        };

        let pause_flag = Arc::new(AtomicBool::new(false));
        let boop_play_when = Arc::clone(&pause_flag);
        let ring = Arc::new(Mutex::new(RingBuffer::new(
            options.capture_rate as usize / 2,
        )));

        Self {
            player,
            ring,
            pause_flag,
            boop_play_when,
            boop_token: None,
            no_boop: options.no_boop,
            boop_interval_ms: options.boop_interval_ms,
            pause_audio: options.pause_audio,
            suppress_boop: options.suppress_boop,
            #[cfg(feature = "ui")]
            overlay: None,
            #[cfg(feature = "ui")]
            pending_overlay: Some(PendingOverlay {
                badge: options.overlay.badge,
                no_sound: options.overlay.no_sound,
                enabled: !options.no_overlay,
                viz: options.viz,
                mono: options.mono,
                capture_rate: options.capture_rate,
                silence_tx: options.overlay.silence_tx,
                auto_pause: options.overlay.auto_pause,
                telemetry_rx: options.overlay.telemetry_rx,
            }),
            #[cfg(test)]
            test_events: None,
            #[cfg(test)]
            test_badge_available: false,
        }
    }

    pub async fn play_start(&self) {
        self.record_test_event("start-tone");
        if let Some(player) = &self.player {
            log::debug!("playing start sound");
            player.play_start().await;
        }
    }

    /// Show the neutral "starting" badge as soon as the shortcut is handled.
    ///
    /// This is the user's immediate visual acknowledgement; it is NOT the
    /// recording cue.  The red recording badge is only shown by
    /// [`announce_recording`](Self::announce_recording) once capture has
    /// proven it is delivering audio and the start tone was emitted.
    pub fn prepare_recording(&mut self) {
        self.initialize_display();
        #[cfg(test)]
        if self.test_badge_available {
            self.record_test_event("badge-arming");
        }
        #[cfg(feature = "ui")]
        if let Some(overlay) = &self.overlay {
            log::debug!("showing arming overlay");
            overlay.show(crate::x11::overlay::IndicatorKind::Arming);
        }
    }

    /// Give the "speak now" cue, but only on proven capture readiness.
    ///
    /// `proof` is the outcome of waiting for the capture's readiness
    /// (`Ok(None)` for sources that produce their own samples, such as
    /// file replay).  On success: play the start tone and wait until the
    /// output device reports it emitted, then show the red recording badge
    /// and start the boop loop.  On failure nothing is cued and the error
    /// is returned; the caller tears the feedback down.
    ///
    /// Capture is already running while the tone plays, so every sample
    /// from the first buffer on is kept (the tone may be faintly captured).
    pub async fn announce_recording(
        &mut self,
        proof: Result<Option<crate::audio::CaptureReady>, crate::error::TalkError>,
    ) -> Result<Option<crate::audio::CaptureReady>, crate::error::TalkError> {
        let proof = proof?;
        self.record_test_event("capture-ready");
        if let Some(ready) = &proof {
            log::info!(
                "capture proven on '{}' ({:.0}ms ago)",
                ready.source,
                ready.first_buffer_at.elapsed().as_secs_f64() * 1000.0
            );
        }
        self.play_start().await;
        self.record_test_event("tone-emitted");
        self.show_recording_badge();
        self.start_boop();
        Ok(proof)
    }

    /// Initialize display resources (overlay thread).  Idempotent.
    pub fn initialize_display(&mut self) {
        #[cfg(feature = "ui")]
        self.initialize_overlay();
    }

    pub fn show_recording_badge(&self) {
        #[cfg(test)]
        if self.test_badge_available {
            self.record_test_event("badge-recording");
        }

        #[cfg(feature = "ui")]
        if let Some(overlay) = &self.overlay {
            log::debug!("showing recording overlay");
            overlay.show(crate::x11::overlay::IndicatorKind::Recording);
        }
    }

    pub fn start_boop(&mut self) {
        if self.no_boop || self.boop_interval_ms == 0 {
            log::debug!("boop sounds disabled");
            return;
        }

        self.record_test_event("boop-start");
        if let Some(player) = &self.player {
            self.boop_token = Some(player.start_boop_loop(
                std::time::Duration::from_millis(self.boop_interval_ms),
                Some(Arc::clone(&self.boop_play_when)),
                self.suppress_boop.clone(),
            ));
        }
    }

    /// Feed the visualizer ring while preserving the caller's PCM stream.
    /// Standalone recording sets `pause_audio=false`, so silence never removes
    /// samples from the saved recording even though it still gates the boop.
    pub fn route_audio(&self, input: mpsc::Receiver<Vec<i16>>) -> mpsc::Receiver<Vec<i16>> {
        if self.has_badge() {
            crate::audio::tee::spawn_audio_tee(
                input,
                Arc::clone(&self.ring),
                Arc::clone(&self.pause_flag),
                self.pause_audio,
            )
        } else {
            input
        }
    }

    pub fn teardown_recording(&mut self, badge: RecordingBadgeTeardown) {
        if let Some(token) = self.boop_token.take() {
            log::debug!("stopping boop loop");
            token.cancel();
        }
        self.record_test_event("boop-cancel");

        if badge == RecordingBadgeTeardown::Hide {
            #[cfg(test)]
            if self.test_badge_available {
                self.record_test_event("badge-hide");
            }
            #[cfg(feature = "ui")]
            if let Some(overlay) = &self.overlay {
                log::debug!("hiding recording overlay");
                overlay.hide();
            }
        }
    }

    pub async fn play_stop(&self) {
        self.record_test_event("stop-tone");
        if let Some(player) = &self.player {
            log::debug!("playing stop sound");
            player.play_stop().await;
        }
    }

    pub fn pause_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.pause_flag)
    }

    pub fn player(&self) -> Option<&SoundPlayer> {
        self.player.as_ref()
    }

    #[cfg(feature = "ui")]
    pub fn overlay(&self) -> Option<&OverlayHandle> {
        self.overlay.as_ref()
    }

    fn has_badge(&self) -> bool {
        #[cfg(feature = "ui")]
        if self.overlay.is_some() {
            return true;
        }
        #[cfg(test)]
        if self.test_badge_available {
            return true;
        }
        false
    }

    #[cfg(feature = "ui")]
    fn initialize_overlay(&mut self) {
        let Some(options) = self.pending_overlay.take() else {
            return;
        };
        if !options.enabled {
            log::debug!("visual overlay disabled");
            return;
        }
        if let Err(error) = gtk4::init() {
            log::warn!("GTK4 init failed (overlay unavailable): {}", error);
            return;
        }
        match OverlayHandle::new(
            options.viz,
            options.mono,
            Arc::clone(&self.ring),
            options.capture_rate,
            options.silence_tx,
            Arc::clone(&self.pause_flag),
            options.auto_pause,
            options.telemetry_rx,
            options.badge,
            options.no_sound,
        ) {
            Ok(overlay) => {
                log::debug!(
                    "overlay initialized (viz={:?}, mono={})",
                    options.viz,
                    options.mono
                );
                self.overlay = Some(overlay);
            }
            Err(error) => log::warn!("visual overlay unavailable: {}", error),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(
        events: Arc<Mutex<Vec<&'static str>>>,
        badge_available: bool,
    ) -> Self {
        let pause_flag = Arc::new(AtomicBool::new(false));
        Self {
            player: None,
            ring: Arc::new(Mutex::new(RingBuffer::new(8))),
            boop_play_when: Arc::clone(&pause_flag),
            pause_flag,
            boop_token: None,
            no_boop: false,
            boop_interval_ms: 5_000,
            pause_audio: false,
            suppress_boop: None,
            #[cfg(feature = "ui")]
            overlay: None,
            #[cfg(feature = "ui")]
            pending_overlay: None,
            test_events: Some(events),
            test_badge_available: badge_available,
        }
    }

    #[cfg(test)]
    pub(crate) fn boop_play_when_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.boop_play_when)
    }

    fn record_test_event(&self, event: &'static str) {
        #[cfg(test)]
        if let Some(events) = &self.test_events {
            if let Ok(mut events) = events.lock() {
                events.push(event);
            }
        }
        #[cfg(not(test))]
        let _ = event;
    }
}

impl Drop for RecordingFeedback {
    fn drop(&mut self) {
        if let Some(token) = self.boop_token.take() {
            token.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn event_log() -> Arc<Mutex<Vec<&'static str>>> {
        Arc::new(Mutex::new(Vec::new()))
    }

    fn events(log: &Arc<Mutex<Vec<&'static str>>>) -> Vec<&'static str> {
        log.lock().map(|events| events.clone()).unwrap_or_default()
    }

    #[tokio::test]
    async fn shared_feedback_teardown_cancels_boop_before_hiding_badge() {
        let log = event_log();
        let mut feedback = RecordingFeedback::new_for_test(Arc::clone(&log), true);

        feedback.prepare_recording();
        feedback
            .announce_recording(Ok(None))
            .await
            .expect("file source needs no proof");
        feedback.teardown_recording(RecordingBadgeTeardown::Hide);

        assert_eq!(
            events(&log),
            vec![
                "badge-arming",
                "capture-ready",
                "start-tone",
                "tone-emitted",
                "badge-recording",
                "boop-start",
                "boop-cancel",
                "badge-hide"
            ]
        );
    }

    #[tokio::test]
    async fn recording_cue_follows_capture_proof_and_tone_emission() {
        let log = event_log();
        let mut feedback = RecordingFeedback::new_for_test(Arc::clone(&log), true);
        feedback.prepare_recording();
        feedback
            .announce_recording(Ok(Some(crate::audio::CaptureReady {
                source: "test.microphone".into(),
                first_buffer_at: std::time::Instant::now(),
            })))
            .await
            .expect("proof permits the recording cue");

        assert_eq!(
            events(&log),
            vec![
                "badge-arming",
                "capture-ready",
                "start-tone",
                "tone-emitted",
                "badge-recording",
                "boop-start"
            ]
        );
    }

    #[tokio::test]
    async fn failed_capture_proof_never_shows_recording_cue() {
        let log = event_log();
        let mut feedback = RecordingFeedback::new_for_test(Arc::clone(&log), true);
        feedback.prepare_recording();
        let failure = feedback
            .announce_recording(Err(crate::error::TalkError::Audio(
                "test.microphone: first buffer not delivered".into(),
            )))
            .await;
        feedback.teardown_recording(RecordingBadgeTeardown::Hide);

        assert!(failure.is_err());
        assert_eq!(
            events(&log),
            vec!["badge-arming", "boop-cancel", "badge-hide"]
        );
    }

    #[test]
    fn shared_feedback_boop_uses_overlay_pause_flag_as_positive_gate() {
        let log = event_log();
        let feedback = RecordingFeedback::new_for_test(log, true);

        assert!(Arc::ptr_eq(
            &feedback.pause_flag(),
            &feedback.boop_play_when_flag()
        ));
    }

    #[tokio::test]
    async fn unavailable_display_does_not_disable_audio_feedback() {
        let log = event_log();
        let mut feedback = RecordingFeedback::new_for_test(Arc::clone(&log), false);

        feedback.prepare_recording();
        feedback
            .announce_recording(Ok(None))
            .await
            .expect("file source needs no proof");
        feedback.teardown_recording(RecordingBadgeTeardown::Hide);
        feedback.play_stop().await;

        assert_eq!(
            events(&log),
            vec![
                "capture-ready",
                "start-tone",
                "tone-emitted",
                "boop-start",
                "boop-cancel",
                "stop-tone"
            ]
        );
    }

    #[tokio::test]
    async fn dictate_teardown_can_preserve_badge_for_transcribing_handoff() {
        let log = event_log();
        let mut feedback = RecordingFeedback::new_for_test(Arc::clone(&log), true);

        feedback.prepare_recording();
        feedback
            .announce_recording(Ok(None))
            .await
            .expect("file source needs no proof");
        feedback.teardown_recording(RecordingBadgeTeardown::KeepVisible);
        feedback.play_stop().await;

        assert_eq!(
            events(&log),
            vec![
                "badge-arming",
                "capture-ready",
                "start-tone",
                "tone-emitted",
                "badge-recording",
                "boop-start",
                "boop-cancel",
                "stop-tone"
            ]
        );
    }

    #[tokio::test]
    async fn standalone_recording_forwards_pcm_while_silence_gate_is_set() {
        let log = event_log();
        let feedback = RecordingFeedback::new_for_test(log, true);
        feedback
            .pause_flag()
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.send(vec![1, 2, 3])
            .await
            .expect("test input should remain open");
        drop(tx);

        let mut routed = feedback.route_audio(rx);

        assert_eq!(routed.recv().await, Some(vec![1, 2, 3]));
    }
}
