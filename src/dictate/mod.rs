//! Dictate command implementation.
//!
//! Records audio, streams it to the transcription API, and pastes the result
//! into the focused application via clipboard.

mod models;
mod picker;
mod realtime;
mod streaming;
mod text;
mod toggle;

use crate::audio::bt_profile;
use crate::audio::file_source::{OggFileSource, WavFileSource};
use crate::audio::indicator::SoundPlayer;
use crate::audio::monitor_capture::MonitorCapture;
use crate::audio::pipewire_capture::PipeWireCapture;
use crate::audio::resample;
use crate::audio::{AudioCapture, CHUNK_DURATION_MS};
use crate::clipboard::{Clipboard, X11Clipboard};
use crate::config::{AudioConfig, Config, Provider};
use crate::daemon;
use crate::error::TalkError;
use crate::paste::{
    focus_window, get_active_window, paste_text_to_target, simulate_paste, PASTE_CHUNK_CHARS,
};
use crate::recording_cache;
use crate::telemetry::{BroadcastSink, TelemetrySink, TranscriptionEvent};
use crate::transcription;
use crate::x11::overlay::{IndicatorKind, OverlayHandle};
use crate::x11::render_util::RingBuffer;
use crate::x11::visualizer::VisualizerHandle;
use models::{resolve_model, resolve_provider};
use picker::{run_pick, PickParams};
use realtime::dictate_realtime;
use std::path::PathBuf;
use streaming::dictate_streaming;
use toggle::toggle_dispatch;
use tokio_util::sync::CancellationToken;

/// Options for the dictate command.
pub struct DictateOpts {
    pub save: Option<PathBuf>,
    pub output_yaml: Option<PathBuf>,
    pub input_audio_file: Option<PathBuf>,
    pub retry_last: bool,
    pub pick: bool,
    pub replace_last_paste: bool,
    pub provider: Option<Provider>,
    pub model: Option<String>,
    pub diarize: bool,
    pub timestamp: bool,
    pub realtime: bool,
    pub toggle: bool,
    pub no_sounds: bool,
    pub no_boop: bool,
    pub no_chunk_paste: bool,
    pub no_paste: bool,
    pub monitor: bool,
    pub no_overlay: bool,
    pub no_auto_pause: bool,
    pub viz: Option<crate::config::VizMode>,
    pub mono: bool,
    pub upload_format: crate::transcription::UploadFormat,
    pub no_bt_auto_switch: bool,
    pub daemon: bool,
    pub target_window: Option<String>,
    pub verbose: u8,
}

/// Dictate: record audio, transcribe, and paste into focused application.
pub async fn dictate(opts: DictateOpts) -> Result<(), TalkError> {
    // Toggle mode: start or stop a daemon
    if opts.toggle {
        return toggle_dispatch(&opts).await;
    }

    let save_path = opts.save;

    // Load configuration
    let config = Config::load(None)?;

    // Effective paste chunk size: --no-chunk-paste disables chunking (0),
    // otherwise use config value or the built-in default (150).
    let paste_chunk_chars = if opts.no_chunk_paste {
        0
    } else {
        config
            .paste
            .as_ref()
            .map(|p| p.chunk_chars)
            .unwrap_or(PASTE_CHUNK_CHARS)
    };

    // Resolve paste shortcut from config (default: Ctrl+Shift+V).
    let paste_shortcut = config
        .paste
        .as_ref()
        .map(|p| p.shortcut.clone())
        .unwrap_or_default();

    // Determine target window: use --target-window arg (from daemon mode)
    // or capture the currently active window.
    let target_window = if let Some(wid) = opts.target_window {
        log::debug!("using target window from argument: {}", wid);
        Some(wid)
    } else if !opts.daemon {
        let wid = get_active_window().await;
        if let Some(ref w) = wid {
            log::debug!("captured active window: {}", w);
        }
        wid
    } else {
        None
    };

    let mut input_audio_file = opts.input_audio_file.clone();
    let mut replace_char_count: Option<usize> = None;
    let mut cached_brief: Option<recording_cache::RecordingMetadataBrief> = None;
    if opts.retry_last {
        let last_audio = recording_cache::last_recording_path()
            .or_else(|_| recording_cache::latest_recording_path())?;
        input_audio_file = Some(last_audio);
    }

    // Load existing pick so the picker can seed its initial state.
    // Source of truth: the pick file (Layer 1).  Sidecars are
    // Layer 3 internals and never consulted here.
    if let Some(ref audio) = input_audio_file {
        if let Some((provider, model, _streaming, text)) = recording_cache::read_pick(audio) {
            cached_brief = Some(recording_cache::RecordingMetadataBrief {
                transcript: text,
                provider: Some(provider.to_string()),
                model: Some(model),
            });
        }
    }

    // Character count for --replace-last-paste comes from the paste
    // state file, not from any transcript source.
    if opts.replace_last_paste {
        if let Ok(Some(state)) = recording_cache::read_last_paste_state() {
            replace_char_count = Some(state.char_count);
        }
    }

    if opts.pick {
        return run_pick(
            config,
            PickParams {
                input_audio_file,
                cached_brief,
                replace_char_count,
                replace_last_paste: opts.replace_last_paste,
                provider: opts.provider,
                model: opts.model,
                target_window,
                paste_chunk_chars,
                paste_shortcut: paste_shortcut.clone(),
            },
        )
        .await;
    }

    // Mode C: file input + default options -> consult the pick file
    // directly.  If it exists, paste its text without running any
    // capture/transcription pipeline.  Matches the `transcribe`
    // command's default branch.
    let specific_options = opts.provider.is_some() || opts.model.is_some() || opts.diarize;
    if let Some(ref audio) = input_audio_file {
        if !specific_options {
            if let crate::recording_cache::TranscriptStatus::Available(text) =
                crate::recording_cache::get_transcript(audio)
            {
                log::info!(
                    "pick file present for {} — pasting without new transcription",
                    audio.display()
                );
                paste_text_to_target(
                    target_window.as_ref(),
                    &text,
                    replace_char_count.unwrap_or(0),
                    paste_chunk_chars,
                    None,
                    &crate::telemetry::NoOpSink,
                    paste_shortcut.clone(),
                )
                .await?;
                let _ = recording_cache::write_last_paste_state(target_window.as_deref(), &text);
                println!("{}", text);
                return Ok(());
            }
        }
    }

    // Initialize sound player (single-channel with preemption)
    let player = if opts.no_sounds {
        log::debug!("sound indicators disabled");
        None
    } else {
        match SoundPlayer::new() {
            Ok(p) => {
                log::debug!("sound player initialized");
                Some(p)
            }
            Err(e) => {
                log::warn!("sound indicators unavailable: {}", e);
                None
            }
        }
    };

    // Ensure GTK4/GDK4 is initialised so the overlay and visualizer can
    // query monitor geometry via GDK.  `gtk4::init()` is idempotent —
    // safe to call even if the picker path already initialised it.
    if let Err(e) = gtk4::init() {
        log::warn!(
            "GTK4 init failed (overlay/visualizer may be unavailable): {}",
            e
        );
    }

    // Resolve visualizer mode: CLI flag overrides config.
    let viz_mode = opts
        .viz
        .or_else(|| config.indicators.as_ref().and_then(|ind| ind.viz));
    if let Some(mode) = viz_mode {
        log::info!("visualizer mode: {}", mode);
    }

    // Overlay is created AFTER capture_rate is determined (see below),
    // because it needs the sample rate and a shared ring buffer.

    // Initialize visualizer (text panel for status messages / live
    // transcription text).  Audio visualization has moved into the
    // overlay badge itself — the visualizer thread only handles text.
    let visualizer = match VisualizerHandle::new(opts.realtime) {
        Ok(h) => {
            log::debug!("visualizer text panel initialized");
            Some(h)
        }
        Err(e) => {
            log::warn!("visualizer unavailable: {}", e);
            None
        }
    };

    // Generate cache recording path (always, even without --save)
    let (cache_path, _cache_timestamp) = recording_cache::generate_recording_path()?;
    log::info!("cache recording: {}", cache_path.display());

    let provider = resolve_provider(opts.provider, &config);
    let effective_model = resolve_model(opts.model.as_deref(), &config, provider, opts.realtime);

    // Create audio source: live microphone or audio file input.
    //
    // For live capture, record at the device's native rate (typically
    // 48 kHz) and downsample to 16 kHz with a proper anti-aliasing
    // filter.  PCM WAV input is assumed to be 16 kHz already; OGG/Opus
    // input is decoded by `OggFileSource`.
    let encode_config = AudioConfig::new(); // 16 kHz target for encoder
    let from_file = input_audio_file.is_some();
    let (mut capture, capture_rate): (Box<dyn AudioCapture>, u32) =
        if let Some(ref path) = input_audio_file {
            log::info!("using audio file input: {}", path.display());
            let is_ogg = path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("ogg"));
            let source: Box<dyn AudioCapture> = if is_ogg {
                Box::new(OggFileSource::new(path)?)
            } else {
                Box::new(WavFileSource::new(path, &encode_config)?)
            };
            (source, encode_config.sample_rate)
        } else {
            // Prefer PipeWire native capture — matches pw-cat's audio
            // routing (including Bluetooth devices) exactly.  Fall back
            // to cpal/ALSA if PipeWire is unavailable.
            let rate = 48_000u32; // PipeWire native rate; resampled to 16 kHz downstream
            let capture_config = AudioConfig {
                sample_rate: rate,
                channels: encode_config.channels,
                bitrate: encode_config.bitrate,
            };
            if opts.monitor {
                log::info!(
                    "capture at {}Hz (PipeWire, mic+monitor), target {}Hz",
                    rate,
                    encode_config.sample_rate
                );
                (
                    Box::new(MonitorCapture::new(capture_config)) as Box<dyn AudioCapture>,
                    rate,
                )
            } else {
                log::info!(
                    "capture at {}Hz (PipeWire), target {}Hz",
                    rate,
                    encode_config.sample_rate
                );
                (
                    Box::new(PipeWireCapture::new(capture_config)) as Box<dyn AudioCapture>,
                    rate,
                )
            }
        };

    // Register SIGINT handler early — before any long-running resources
    // (PipeWire capture, sound playback) — so that a quick toggle-off
    // is never missed.  Without this, there is a ~1 s race window
    // between capture.start() and the ctrl_c().await inside
    // dictate_streaming/dictate_realtime where SIGINT has no handler
    // and the daemon becomes an unkillable orphan.
    let shutdown = CancellationToken::new();
    let shutdown_clone = shutdown.clone();
    let daemon_pid = std::process::id();
    tokio::spawn(async move {
        log::warn!(
            "[DBG] daemon {}: ctrl_c handler task polled, registering handler",
            daemon_pid
        );
        let _ = tokio::signal::ctrl_c().await;
        log::warn!(
            "[DBG] daemon {}: SIGINT received! cancelling shutdown token",
            daemon_pid
        );
        shutdown_clone.cancel();
    });

    log::info!(
        "starting {} transcription{}",
        if opts.realtime { "realtime" } else { "batch" },
        if from_file { " (from file)" } else { "" }
    );

    // Bluetooth headset profile auto-switching.
    //
    // Only meaningful when we are about to capture live audio — file
    // inputs read from disk and never touch the microphone.  Resolution
    // order: CLI flag `--no-bt-auto-switch` wins; otherwise config
    // `audio.bt_auto_switch` (default `true`).  When enabled we:
    //
    // 1. Recover any stale profile from a prior unclean termination
    //    (the state file at $XDG_RUNTIME_DIR/talk-rs/card-profile.json
    //    is left behind by SIGKILL/crash).  This must run BEFORE
    //    activate_headset so we are restoring to the user's TRUE
    //    original profile, not to whatever HFP profile was active when
    //    the previous run died.
    // 2. Switch any connected BT headset to its best HFP profile so
    //    the headset microphone is enabled.  The returned guard is
    //    moved into dictate_streaming / dictate_realtime, where it is
    //    triggered explicitly the moment `capture.stop()` returns —
    //    so the user gets A2DP audio back IMMEDIATELY on toggle-off,
    //    not only after the transcription + paste pipeline finishes.
    //    The guard's Drop is also the safety net for early ?-returns
    //    and panics on the path between activation and dispatch.
    //
    // Failures are logged but never fatal; the dictation proceeds on
    // whatever input device is currently active.
    let bt_auto_switch_enabled = !opts.no_bt_auto_switch
        && config
            .audio
            .as_ref()
            .map(|a| a.bt_auto_switch_enabled())
            .unwrap_or(true);
    let bt_guard = if from_file || !bt_auto_switch_enabled {
        if !bt_auto_switch_enabled {
            log::debug!("bt_profile: auto-switching disabled by config/flag");
        }
        bt_profile::HeadsetGuard::new(None)
    } else {
        if let Err(err) = bt_profile::recover_stale_profile() {
            log::warn!("bt_profile: stale-recovery failed (non-fatal): {}", err);
        }
        let saved = match bt_profile::activate_headset() {
            Ok(s) => s,
            Err(err) => {
                log::warn!("bt_profile: activate_headset failed (non-fatal): {}", err);
                None
            }
        };
        bt_profile::HeadsetGuard::new(saved)
    };

    // Play start sound BEFORE starting capture so that the tone is
    // never captured in the recording.
    if let Some(ref p) = player {
        log::debug!("playing start sound");
        p.play_start().await;
    }

    // Start capture AFTER the start sound so the tone is not recorded.
    let raw_audio_rx = capture.start()?;

    // Create shared ring buffer for overlay visualization (reads from
    // the same PipeWire stream as the recording pipeline).
    let ring = std::sync::Arc::new(std::sync::Mutex::new(RingBuffer::new(
        capture_rate as usize / 2,
    )));

    // Silence notification channel (overlay → text panel).
    let (silence_tx, silence_rx) = std::sync::mpsc::channel::<bool>();

    // Auto-pause flag: overlay sets this when it detects silence during
    // recording; the audio tee stops forwarding samples downstream.
    let pause_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Create the telemetry broadcast sink.  Transcription producers
    // emit events into this channel; the overlay thread subscribes
    // to receive them for the phase-colour layer.  The broker is
    // created before the overlay so the receiver can be passed in.
    let broker = std::sync::Arc::new(BroadcastSink::new(256));
    let sink: std::sync::Arc<dyn TelemetrySink> = broker.clone();

    // Initialize overlay (visual indicator on X11).
    // Created here (after capture_rate is known) so we can pass
    // the shared ring buffer and sample rate.
    let overlay = if opts.no_overlay {
        log::debug!("visual overlay disabled");
        None
    } else {
        let telemetry_rx = broker.subscribe();
        match OverlayHandle::new(
            viz_mode,
            opts.mono,
            ring.clone(),
            capture_rate,
            Some(silence_tx),
            pause_flag.clone(),
            !opts.no_auto_pause,
            Some(telemetry_rx),
        ) {
            Ok(h) => {
                log::debug!(
                    "overlay initialized (viz={:?}, mono={})",
                    viz_mode,
                    opts.mono
                );
                Some(h)
            }
            Err(e) => {
                log::warn!("visual overlay unavailable: {}", e);
                None
            }
        }
    };

    // Show recording indicator
    if let Some(ref o) = overlay {
        log::debug!("showing recording overlay");
        o.show(IndicatorKind::Recording);
    }

    // Show visualizer text panel (positioned relative to recording badge)
    if let Some(ref viz) = visualizer {
        log::debug!("showing visualizer text panel");
        viz.show(crate::x11::overlay::BADGE_W);
    }

    // Shared flag: suppresses the boop heartbeat while a no-sound
    // alert is active so the two sounds don't collide.
    let suppress_boop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Start boop loop (configurable interval, disabled by --no-boop or interval=0)
    let boop_interval_ms = config
        .indicators
        .as_ref()
        .map(|i| i.boop_interval_ms)
        .unwrap_or(5000);
    let boop_token = if opts.no_boop || boop_interval_ms == 0 {
        log::debug!("boop sounds disabled");
        None
    } else {
        player.as_ref().map(|p| {
            // Gate the heartbeat on the LISTENING (auto-pause) state:
            // boops only fire while the overlay's `pause_flag` is set,
            // so the user hears the chirp during silence but not while
            // actively speaking.  When `--no-auto-pause` is in effect,
            // `pause_flag` never flips on and the boop stays silent —
            // matching the user's mental model that boops belong to
            // the "LISTENING" badge.
            //
            // Independently, `suppress_boop` silences the heartbeat
            // while a no-sound alert is active so the two tones don't
            // collide.
            let play_when = Some(pause_flag.clone());
            let suppress = Some(suppress_boop.clone());
            p.start_boop_loop(
                std::time::Duration::from_millis(boop_interval_ms),
                play_when,
                suppress,
            )
        })
    };

    // Tee audio: split the raw capture stream so the overlay
    // visualizer reads the same PipeWire data as the recording
    // pipeline.  When the overlay is disabled, pass through directly.
    let raw_for_resample = if overlay.is_some() {
        let teed =
            crate::audio::tee::spawn_audio_tee(raw_audio_rx, ring.clone(), pause_flag.clone());
        // Spawn silence notification thread: forwards silence events
        // from the overlay to the visualizer text panel and plays
        // periodic alert sounds when no audio is detected.
        let vis_push = visualizer.as_ref().map(|v| v.message_pusher());
        let alert_player = player.as_ref().map(|p| p.alert_player());
        if vis_push.is_some() || alert_player.is_some() {
            let suppress = suppress_boop.clone();
            let _ = std::thread::Builder::new()
                .name("silence-notifier".into())
                .spawn(move || {
                    let mut alerting = false;
                    loop {
                        let event = if alerting {
                            match silence_rx.recv_timeout(std::time::Duration::from_secs(2)) {
                                Ok(val) => Some(val),
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                                    // Still silent — replay alert sound.
                                    if let Some(ref ap) = alert_player {
                                        ap.play();
                                    }
                                    None
                                }
                                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                            }
                        } else {
                            match silence_rx.recv() {
                                Ok(val) => Some(val),
                                Err(_) => break,
                            }
                        };

                        if let Some(is_silent) = event {
                            if is_silent {
                                alerting = true;
                                suppress.store(true, std::sync::atomic::Ordering::Relaxed);
                                if let Some(ref push) = vis_push {
                                    push(
                                        "No audio detected \u{2014} check your microphone"
                                            .to_string(),
                                    );
                                }
                                // Play alert immediately on first detection.
                                if let Some(ref ap) = alert_player {
                                    ap.play();
                                }
                            } else {
                                alerting = false;
                                suppress.store(false, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                });
        }
        teed
    } else {
        raw_audio_rx
    };

    // Set up the resample pipeline (common to both modes).  Audio has
    // been buffering in the capture channel since capture.start() above.
    let capture_chunk = (capture_rate as usize * CHUNK_DURATION_MS as usize) / 1000;
    let audio_rx = resample::spawn_resample_task(
        capture_rate,
        encode_config.sample_rate,
        raw_for_resample,
        capture_chunk,
    )?;

    if opts.diarize && opts.realtime {
        return Err(TalkError::Config(
            "--diarize is not supported with --realtime: \
             the Mistral realtime WebSocket endpoint does not support speaker diarization"
                .to_string(),
        ));
    }

    let mut t_stop: Option<std::time::Instant> = None;
    let result = if opts.realtime {
        // Realtime mode (--realtime): stream audio over WebSocket.
        // Each segment is pasted into the focused application as it
        // arrives, providing real-time feedback while dictating.

        // Save clipboard and focus target window before recording starts
        let rt_clipboard = X11Clipboard::new();
        let saved_clipboard = rt_clipboard.get_text().await.ok();
        if let Some(ref wid) = target_window {
            log::debug!("pre-focusing target window: {}", wid);
            if !focus_window(wid).await {
                log::warn!("could not pre-focus target window {}", wid);
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // Create segment channel for per-segment pasting
        let (seg_tx, mut seg_rx) = tokio::sync::mpsc::channel::<String>(32);

        // Spawn paste consumer: each segment is pasted immediately
        let paste_task = tokio::spawn(async move {
            let paste_clip = X11Clipboard::new();
            let mut is_first = true;
            while let Some(segment) = seg_rx.recv().await {
                let paste_text = if is_first {
                    is_first = false;
                    segment
                } else {
                    format!(" {}", segment)
                };
                if let Err(e) = paste_clip.set_text(&paste_text).await {
                    log::warn!("per-segment clipboard set failed: {}", e);
                    continue;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                if let Err(e) = simulate_paste(paste_shortcut.clone()).await {
                    log::warn!("per-segment paste failed: {}", e);
                }
                tokio::time::sleep(std::time::Duration::from_millis(15)).await;
            }
        });

        let result = match dictate_realtime(
            config.clone(),
            provider,
            opts.model.as_deref(),
            &cache_path,
            audio_rx,
            &mut *capture,
            from_file,
            player.as_ref(),
            boop_token.as_ref(),
            Some(seg_tx),
            visualizer.as_ref(),
            &shutdown,
            bt_guard,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                // Enrich model-not-found errors with available models.
                let enriched =
                    transcription::enrich_model_error(&config, provider, opts.model.as_deref(), e)
                        .await;
                return Err(enriched);
            }
        };

        // Wait for all pending pastes to complete
        if let Err(e) = paste_task.await {
            log::warn!("paste task error: {}", e);
        }

        // Restore original clipboard
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        if let Some(saved) = saved_clipboard {
            log::debug!("restoring original clipboard");
            let _ = rt_clipboard.set_text(&saved).await;
        }

        result
    } else {
        // Batch mode (default): capture audio, encode, then transcribe.
        // Dictate is an autonomous pipeline (no human watching),
        // so use `Proportional` so a hung server cannot wedge the
        // pipeline forever.
        let mut transcriber = transcription::create_batch_transcriber(
            &config,
            provider,
            opts.model.as_deref(),
            opts.diarize,
            transcription::RequestTimeoutPolicy::Proportional,
        )?;
        transcriber.set_sink(sink.clone());

        let (stream_result, t_stop_val) = dictate_streaming(
            &mut *capture,
            from_file,
            encode_config.clone(),
            audio_rx,
            &cache_path,
            transcriber,
            &shutdown,
            player.as_ref(),
            boop_token.as_ref(),
            overlay.as_ref(),
            visualizer.as_ref(),
            &config,
            provider,
            opts.model.as_deref(),
            opts.diarize,
            bt_guard,
        )
        .await;
        t_stop = t_stop_val;

        // If the streaming transcription failed, fall back to a
        // single call to `transcribe_audio` using the saved OGG
        // file.  Retry lives inside `transcribe_audio` (see
        // `transcription::transport::retry`) — no loop here.
        match stream_result {
            Ok(r) => r,
            Err(first_err) => {
                log::warn!("streaming transcription failed: {}", first_err);

                // Fall back to file-based transcription.  Retry is
                // already handled by Layer 3.  Same `Proportional`
                // policy as the streaming path above — autonomous
                // dictate must not hang.
                let result = transcription::transcribe_audio(
                    &cache_path,
                    &config,
                    provider,
                    opts.model.as_deref(),
                    opts.diarize,
                    transcription::TranscribeOptions {
                        allow_api: true,
                        policy: transcription::RequestTimeoutPolicy::Proportional,
                        cancel_token: None,
                        skip_legacy_lock: false,
                    },
                    &sink,
                )
                .await;

                match result {
                    Ok(r) => r,
                    Err(final_err) => {
                        // Model errors get enriched with available
                        // models (display concern).  Other errors are
                        // already after-retry permanent failures.
                        let final_err = if transcription::is_model_error(provider, &final_err) {
                            transcription::enrich_model_error(
                                &config,
                                provider,
                                opts.model.as_deref(),
                                final_err,
                            )
                            .await
                        } else {
                            final_err
                        };
                        let final_msg = format!("{}", final_err);
                        log::error!("{}", final_msg);
                        if let Some(ref viz) = visualizer {
                            viz.push_message(&format!("Error: {}", final_msg));
                        }
                        if let Some(ref o) = overlay {
                            o.hide();
                        }
                        // Clean up PID so a new daemon can start
                        // while the visualizer drains its messages.
                        if opts.daemon {
                            if let Ok(pid_file) = daemon::pid_path() {
                                let _ =
                                    daemon::remove_pid_file_if_owner(std::process::id(), &pid_file);
                            }
                        }
                        return Ok(());
                    }
                }
            }
        }
    };

    if let Some(t) = t_stop {
        log::info!(
            "timing: stop +{}ms transcription_done",
            t.elapsed().as_millis()
        );
    }

    // Stop boop loop (idempotent — may already be cancelled by
    // dictate_streaming or dictate_realtime).
    if let Some(token) = boop_token {
        log::debug!("stopping boop loop");
        token.cancel();
    }

    // For realtime mode, play stop sound and hide visualizer here
    // (batch mode already did this inside dictate_streaming on SIGINT).
    if opts.realtime {
        if let Some(ref p) = player {
            log::debug!("playing stop sound");
            p.play_stop().await;
        }
        if let Some(ref viz) = visualizer {
            log::debug!("hiding visualizer");
            viz.hide();
        }
        if let Some(ref o) = overlay {
            log::debug!("hiding overlay");
            o.hide();
        }
    }
    // Batch mode: overlay is still showing "Transcribing" badge —
    // it will be hidden after paste (below).

    let text = crate::transcription::format_transcription_output(&result, opts.timestamp)
        .trim()
        .to_string();
    let segments = result.segments;
    let metadata = result.metadata;

    // Write the authoritative pick file UNLESS the user asked for a
    // specific model/provider/diarize — those are "give me a
    // one-off transcription", not "produce the authoritative
    // transcript for this recording".  Never overwrites an
    // existing pick (user may have edited it).
    let specific_options = opts.provider.is_some() || opts.model.is_some() || opts.diarize;
    if !specific_options {
        if let Err(e) = recording_cache::write_pick_if_absent(
            &cache_path,
            &provider.to_string(),
            &effective_model,
            opts.realtime,
            &text,
        ) {
            log::warn!("failed to write pick file: {}", e);
        }
    }

    // Write recording cache metadata and rotate old entries.
    let result_for_cache = transcription::TranscriptionResult {
        text: text.clone(),
        metadata: metadata.clone(),
        diarization: None,
        segments: segments.clone(),
    };
    let cache_meta_path = recording_cache::TranscriptionCache::store(
        &cache_path,
        provider,
        &effective_model,
        opts.realtime,
        &result_for_cache,
    );
    if let Err(ref e) = cache_meta_path {
        log::warn!("failed to write recording metadata: {}", e);
    }
    if let Ok(ref meta_path) = cache_meta_path {
        if let Err(e) = recording_cache::write_last_pointers(&cache_path, meta_path) {
            log::warn!("failed to update last recording pointers: {}", e);
        }
    }
    if let Err(e) = recording_cache::rotate_cache() {
        log::warn!("failed to rotate recording cache: {}", e);
    }

    // Copy cache OGG to --save path if specified
    if let Some(ref path) = save_path {
        if let Err(e) = std::fs::copy(&cache_path, path) {
            log::warn!("failed to copy recording to {}: {}", path.display(), e);
        } else {
            log::info!("audio saved to: {}", path.display());
        }
    }

    // Copy cache metadata YAML to --output-yaml path if specified
    if let Some(ref yaml_path) = opts.output_yaml {
        match cache_meta_path {
            Ok(ref src) => {
                if let Err(e) = std::fs::copy(src, yaml_path) {
                    log::warn!("failed to copy metadata to {}: {}", yaml_path.display(), e);
                } else {
                    log::info!("metadata YAML saved to: {}", yaml_path.display());
                }
            }
            Err(_) => {
                log::warn!("skipping --output-yaml: cache metadata was not written");
            }
        }
    }

    if text.is_empty() {
        log::warn!(
            "[DBG] daemon {}: empty transcription, exiting normally",
            std::process::id()
        );
        log::warn!("empty transcription — nothing to paste");
        // Hide transcribing overlay and visualizer (batch mode keeps
        // them visible until now).
        if !opts.realtime {
            if let Some(ref o) = overlay {
                o.hide();
            }
        }
        return Ok(());
    }

    log::info!("transcription: {}", text);

    // Paste into focused application (batch mode only;
    // realtime mode pastes per-segment during recording)
    if !opts.realtime && !opts.no_paste {
        // Keep the overlay visible during paste so the phase layer
        // continues to show.  Emit telemetry events so consumers
        // can track the paste duration on the time axis.
        sink.emit(TranscriptionEvent::PasteStarted {
            t: std::time::Instant::now(),
        });
        if let Some(t) = t_stop {
            log::info!("timing: stop +{}ms paste_start", t.elapsed().as_millis());
        }
        paste_text_to_target(
            target_window.as_ref(),
            &text,
            0,
            paste_chunk_chars,
            t_stop,
            &*sink,
            paste_shortcut,
        )
        .await?;
        let _ = recording_cache::write_last_paste_state(target_window.as_deref(), &text);
        sink.emit(TranscriptionEvent::PasteCompleted {
            t: std::time::Instant::now(),
        });
        // Hide AFTER paste so the overlay stays visible throughout.
        if let Some(ref o) = overlay {
            o.hide();
        }
    }

    // Print transcription to stdout (batch mode only;
    // realtime mode already prints segments as they arrive)
    if !opts.realtime {
        println!("{}", text);
    }

    // If running as daemon, clean up PID file on normal exit — but only
    // if we still own it.  Between our SIGINT and this cleanup a new daemon
    // may have spawned and written its own PID, so blindly removing the
    // file would orphan it.
    if opts.daemon {
        if let Ok(pid_file) = daemon::pid_path() {
            let _ = daemon::remove_pid_file_if_owner(std::process::id(), &pid_file);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dictate_pipeline_with_mocks() {
        use crate::audio::mock::MockAudioCapture;
        use crate::audio::{AudioCapture, AudioWriter, OggOpusWriter};
        use crate::clipboard::MockClipboard;
        use crate::config::AudioConfig;
        use crate::transcription::{BatchTranscriber, MockBatchTranscriber, TranscriptionBody};

        let audio_config = AudioConfig::new();

        // Initialize mock capture
        let mut capture =
            MockAudioCapture::new(audio_config.sample_rate, audio_config.channels, 440.0);
        let audio_rx = capture.start().expect("start capture");

        // Initialize writer
        let mut writer = OggOpusWriter::new(audio_config).expect("create writer");

        // Create stream channel
        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(25);

        // Spawn encode task (process a few chunks then stop)
        let encode_task = tokio::spawn(async move {
            let mut rx = audio_rx;
            let mut count = 0;
            let header = writer.header().expect("header");
            if stream_tx.send(header).await.is_err() {
                return;
            }
            while let Some(pcm_chunk) = rx.recv().await {
                let encoded = writer.write_pcm(&pcm_chunk).expect("encode");
                if !encoded.is_empty() && stream_tx.send(encoded).await.is_err() {
                    break;
                }
                count += 1;
                if count >= 3 {
                    break;
                }
            }
            // Finalize
            let remaining = writer.finalize().expect("finalize");
            if !remaining.is_empty() {
                let _ = stream_tx.send(remaining).await;
            }
            // stream_tx dropped here
        });

        // Spawn transcription with mock
        let transcriber = MockBatchTranscriber::new("Hello world from dictation");
        let transcribe_task = tokio::spawn(async move {
            transcriber
                .fetch_transcription(TranscriptionBody::Stream {
                    chunks: stream_rx,
                    file_name: "audio.ogg".to_string(),
                })
                .await
        });

        // Wait for encode to finish
        encode_task.await.expect("encode task");

        // Stop capture
        capture.stop().expect("stop capture");

        // Get transcription
        let result = transcribe_task
            .await
            .expect("transcribe task")
            .expect("transcription");
        let text = result.text;
        assert_eq!(text, "Hello world from dictation");

        // Test clipboard operations with mock
        let clipboard = MockClipboard::with_content("original");
        let saved = clipboard.get_text().await.expect("get clipboard");
        assert_eq!(saved, "original");

        clipboard.set_text(&text).await.expect("set clipboard");
        let current = clipboard.get_text().await.expect("get clipboard");
        assert_eq!(current, "Hello world from dictation");

        // Restore
        clipboard.set_text(&saved).await.expect("restore clipboard");
        let restored = clipboard.get_text().await.expect("get clipboard");
        assert_eq!(restored, "original");
    }
}
