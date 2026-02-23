//! Dictate command implementation.
//!
//! Records audio, streams it to the transcription API, and pastes the result
//! into the focused application via clipboard.

mod models;
mod pick;
mod picker;
mod picker_backend;
mod realtime;
mod streaming;
mod text;
mod toggle;

use crate::core::audio::file_source::WavFileSource;
use crate::core::audio::indicator::SoundPlayer;
use crate::core::audio::monitor_capture::MonitorCapture;
use crate::core::audio::pipewire_capture::PipeWireCapture;
use crate::core::audio::resample;
use crate::core::audio::{AudioCapture, CHUNK_DURATION_MS};
use crate::core::clipboard::{Clipboard, X11Clipboard};
use crate::core::config::{AudioConfig, Config, Provider};
use crate::core::daemon;
use crate::core::error::TalkError;
use crate::core::overlay::{IndicatorKind, OverlayHandle};
use crate::core::recording_cache;
use crate::core::transcription;
use crate::core::visualizer::VisualizerHandle;
use crate::paste::{
    focus_window, get_active_window, paste_text_to_target, simulate_paste, PASTE_CHUNK_CHARS,
};
use models::{resolve_model, resolve_provider};
use pick::{run_pick, PickParams};
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
    pub realtime: bool,
    pub toggle: bool,
    pub no_sounds: bool,
    pub no_boop: bool,
    pub no_chunk_paste: bool,
    pub monitor: bool,
    pub no_overlay: bool,
    pub amplitude: bool,
    pub spectrum: bool,
    pub daemon: bool,
    pub target_window: Option<String>,
    pub verbose: u8,
}

/// Dictate: record audio, transcribe, and paste into focused application.
pub async fn dictate(opts: DictateOpts) -> Result<(), TalkError> {
    // Toggle mode: start or stop a daemon
    if opts.toggle {
        return toggle_dispatch(
            opts.provider,
            opts.model,
            opts.diarize,
            opts.realtime,
            opts.no_sounds,
            opts.no_boop,
            opts.no_chunk_paste,
            opts.monitor,
            opts.no_overlay,
            opts.amplitude,
            opts.spectrum,
            opts.save.as_deref(),
            opts.verbose,
        )
        .await;
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
        let last_meta = recording_cache::last_metadata_path()
            .ok()
            .or(recording_cache::metadata_path_for_recording(&last_audio)?);
        if let Some(meta) = last_meta {
            if let Ok(previous) = recording_cache::read_metadata_brief(&meta) {
                replace_char_count = Some(previous.transcript.chars().count());
                cached_brief = Some(previous);
            }
        }
        input_audio_file = Some(last_audio);
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
            },
        )
        .await;
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

    // Initialize overlay (visual indicator on X11)
    let overlay = if opts.no_overlay {
        log::debug!("visual overlay disabled");
        None
    } else {
        match OverlayHandle::new() {
            Ok(h) => {
                log::debug!("overlay initialized");
                Some(h)
            }
            Err(e) => {
                log::warn!("visual overlay unavailable: {}", e);
                None
            }
        }
    };

    // Initialize visualizer (amplitude / spectrum panels)
    let visualizer = if opts.amplitude || opts.spectrum {
        match VisualizerHandle::new(opts.amplitude, opts.spectrum, opts.realtime) {
            Ok(h) => {
                log::debug!(
                    "visualizer initialized (amplitude={}, spectrum={}, text={})",
                    opts.amplitude,
                    opts.spectrum,
                    opts.realtime,
                );
                Some(h)
            }
            Err(e) => {
                log::warn!("visualizer unavailable: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Generate cache recording path (always, even without --save)
    let (cache_wav_path, cache_timestamp) = recording_cache::generate_recording_path()?;
    log::info!("cache recording: {}", cache_wav_path.display());

    let provider = resolve_provider(opts.provider, &config);
    let effective_model = resolve_model(opts.model.as_deref(), &config, provider, opts.realtime);

    // Create audio source: live microphone or WAV file input.
    //
    // For live capture, record at the device's native rate (typically
    // 48 kHz) and downsample to 16 kHz with a proper anti-aliasing
    // filter.  WAV file input is assumed to be 16 kHz already.
    let encode_config = AudioConfig::new(); // 16 kHz target for encoder
    let from_file = input_audio_file.is_some();
    let (mut capture, capture_rate): (Box<dyn AudioCapture>, u32) =
        if let Some(ref path) = input_audio_file {
            log::info!("using audio file input: {}", path.display());
            (
                Box::new(WavFileSource::new(path, &encode_config)?),
                encode_config.sample_rate,
            )
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

    // Start capture BEFORE the start sound so that audio is already
    // being buffered while the sound plays.  The channel holds ~500 ms
    // of data (25 × 20 ms chunks), which is more than enough for the
    // setup that follows before the resample task drains it.
    let raw_audio_rx = capture.start()?;

    log::info!(
        "starting {} transcription{}",
        if opts.realtime { "realtime" } else { "batch" },
        if from_file { " (from file)" } else { "" }
    );

    // Play start sound — recording is already active so the user can
    // start speaking as soon as they hear the tone.
    if let Some(ref p) = player {
        log::debug!("playing start sound");
        p.play_start().await;
    }

    // Show recording indicator
    if let Some(ref o) = overlay {
        log::debug!("showing recording overlay");
        o.show(IndicatorKind::Recording);
    }

    // Show visualizer panels (positioned relative to 182px recording badge)
    if let Some(ref viz) = visualizer {
        log::debug!("showing visualizer");
        viz.show(182);
    }

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
        player
            .as_ref()
            .map(|p| p.start_boop_loop(std::time::Duration::from_millis(boop_interval_ms)))
    };

    // Set up the resample pipeline (common to both modes).  Audio has
    // been buffering in the capture channel since capture.start() above.
    let capture_chunk = (capture_rate as usize * CHUNK_DURATION_MS as usize) / 1000;
    let audio_rx = resample::spawn_resample_task(
        capture_rate,
        encode_config.sample_rate,
        raw_audio_rx,
        capture_chunk,
    )?;

    if opts.diarize && opts.realtime {
        return Err(TalkError::Config(
            "--diarize is not supported with --realtime: \
             the Mistral realtime WebSocket endpoint does not support speaker diarization"
                .to_string(),
        ));
    }

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
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                if let Err(e) = simulate_paste().await {
                    log::warn!("per-segment paste failed: {}", e);
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        });

        let result = dictate_realtime(
            config,
            provider,
            opts.model.as_deref(),
            &cache_wav_path,
            audio_rx,
            &mut *capture,
            from_file,
            player.as_ref(),
            boop_token.as_ref(),
            Some(seg_tx),
            visualizer.as_ref(),
            &shutdown,
        )
        .await?;

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
        // Provider validation is handled by toggle_validate() in the
        // caller — no network round-trip here.
        let transcriber = transcription::create_batch_transcriber(
            &config,
            provider,
            opts.model.as_deref(),
            opts.diarize,
        )?;

        let stream_result = dictate_streaming(
            &mut *capture,
            from_file,
            encode_config.clone(),
            audio_rx,
            &cache_wav_path,
            transcriber,
            &shutdown,
            player.as_ref(),
            boop_token.as_ref(),
            overlay.as_ref(),
            visualizer.as_ref(),
        )
        .await;

        // If the initial streaming transcription failed (timeout,
        // API error, network issue), retry up to 5 times using the
        // saved WAV file.  Each retry creates a fresh transcriber
        // and applies the same 5-second timeout.
        const MAX_RETRIES: u32 = 5;
        const RETRY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

        match stream_result {
            Ok(r) => r,
            Err(first_err) => {
                log::warn!("initial transcription failed: {}", first_err);
                let mut last_err = first_err;
                let mut succeeded = None;

                for attempt in 1..=MAX_RETRIES {
                    let reason = last_err.to_string();
                    let msg = format!(
                        "Transcription failed: {}. Retrying ({}/{})...",
                        reason, attempt, MAX_RETRIES,
                    );
                    log::warn!("{}", msg);
                    if let Some(ref viz) = visualizer {
                        viz.set_text(&msg);
                    }

                    let retry_transcriber = match transcription::create_batch_transcriber(
                        &config,
                        provider,
                        opts.model.as_deref(),
                        opts.diarize,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            last_err = e;
                            continue;
                        }
                    };

                    let wav = cache_wav_path.to_path_buf();
                    let mut task =
                        tokio::spawn(async move { retry_transcriber.transcribe_file(&wav).await });

                    let outcome = tokio::select! {
                        res = &mut task => {
                            match res {
                                Ok(Ok(r)) => Ok(r),
                                Ok(Err(e)) => Err(e),
                                Err(e) => Err(TalkError::Transcription(
                                    format!("retry task panicked: {}", e),
                                )),
                            }
                        }
                        _ = tokio::time::sleep(RETRY_TIMEOUT) => {
                            task.abort();
                            Err(TalkError::Transcription(
                                "transcription timed out after 5 s".to_string(),
                            ))
                        }
                    };

                    match outcome {
                        Ok(r) => {
                            log::info!("transcription succeeded on retry {}", attempt);
                            if let Some(ref viz) = visualizer {
                                viz.set_text("");
                            }
                            succeeded = Some(r);
                            break;
                        }
                        Err(e) => {
                            last_err = e;
                        }
                    }
                }

                match succeeded {
                    Some(r) => r,
                    None => {
                        // All retries exhausted — show error, skip YAML/paste,
                        // preserve WAV for the picker.
                        let final_msg = format!(
                            "Transcription failed after {} attempts: {}",
                            MAX_RETRIES, last_err,
                        );
                        log::error!("{}", final_msg);
                        if let Some(ref viz) = visualizer {
                            viz.set_text(&format!("Error: {}", final_msg));
                        }
                        // Brief pause so the user can read the error.
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        if let Some(ref viz) = visualizer {
                            viz.hide();
                        }
                        if let Some(ref o) = overlay {
                            o.hide();
                        }
                        return Ok(());
                    }
                }
            }
        }
    };

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

    let text = crate::core::transcription::format_transcription_output(&result)
        .trim()
        .to_string();
    let metadata = result.metadata;

    // Write recording cache metadata and rotate old entries.
    let cache_wav_filename = cache_wav_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("recording.wav");
    let cache_meta_path = recording_cache::write_metadata(
        &cache_timestamp,
        provider,
        &effective_model,
        opts.realtime,
        &text,
        cache_wav_filename,
        &metadata,
    );
    if let Err(ref e) = cache_meta_path {
        log::warn!("failed to write recording metadata: {}", e);
    }
    if let Err(e) = recording_cache::rotate_cache() {
        log::warn!("failed to rotate recording cache: {}", e);
    }

    // Copy cache WAV to --save path if specified
    if let Some(ref path) = save_path {
        if let Err(e) = std::fs::copy(&cache_wav_path, path) {
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
        // Hide transcribing overlay (batch mode keeps it visible until now).
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
    if !opts.realtime {
        // Hide transcribing overlay just before pasting.
        if let Some(ref o) = overlay {
            o.hide();
        }
        paste_text_to_target(target_window.as_ref(), &text, 0, paste_chunk_chars).await?;
        let _ = recording_cache::write_last_paste_state(target_window.as_deref(), &text);
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
        use crate::core::audio::mock::MockAudioCapture;
        use crate::core::audio::{AudioCapture, AudioWriter, OggOpusWriter};
        use crate::core::clipboard::MockClipboard;
        use crate::core::config::AudioConfig;
        use crate::core::transcription::{BatchTranscriber, MockBatchTranscriber};

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
        let transcribe_task =
            tokio::spawn(
                async move { transcriber.transcribe_stream(stream_rx, "audio.ogg").await },
            );

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
