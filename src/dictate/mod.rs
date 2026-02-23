//! Dictate command implementation.
//!
//! Records audio, streams it to the transcription API, and pastes the result
//! into the focused application via clipboard.

mod models;
mod picker;
mod text;
mod toggle;

use crate::core::audio::file_source::WavFileSource;
use crate::core::audio::indicator::SoundPlayer;
use crate::core::audio::monitor_capture::MonitorCapture;
use crate::core::audio::pipewire_capture::PipeWireCapture;
use crate::core::audio::resample;
use crate::core::audio::{AudioCapture, AudioWriter, OggOpusWriter, WavWriter, CHUNK_DURATION_MS};
use crate::core::clipboard::{Clipboard, X11Clipboard};
use crate::core::config::{AudioConfig, Config, Provider};
use crate::core::daemon;
use crate::core::error::TalkError;
use crate::core::overlay::{IndicatorKind, OverlayHandle};
use crate::core::picker_cache;
use crate::core::recording_cache;
use crate::core::transcription::{
    self, BatchTranscriber, MistralProviderMetadata, OpenAIProviderMetadata,
    OpenAIRealtimeMetadata, ProviderSpecificMetadata, RealtimeTranscriber, TranscriptionEvent,
    TranscriptionMetadata, TranscriptionResult,
};
use crate::core::visualizer::VisualizerHandle;
use crate::paste::{
    focus_window, get_active_window, paste_text_to_target, simulate_paste, PASTE_CHUNK_CHARS,
};
use crate::x11::x11_centre_and_raise;
use models::{build_retry_candidates, resolve_model, resolve_provider};
use picker::{pick_with_streaming_gtk, PICKER_TITLE};
use std::path::PathBuf;
use text::flush_sentences;
use toggle::toggle_dispatch;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
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
        // Single-instance: if a picker window is already open, just
        // raise and focus it instead of opening a second one.
        if x11_centre_and_raise(PICKER_TITLE) {
            log::info!("picker already open — raised existing window");
            return Ok(());
        }

        let audio_path = input_audio_file.clone().ok_or_else(|| {
            TalkError::Config("--pick requires --input-audio-file or --retry-last".to_string())
        })?;
        if !audio_path.exists() {
            return Err(TalkError::Config(format!(
                "input audio file not found: {}",
                audio_path.display()
            )));
        }

        // Load previously cached picker results for this audio file
        // so that reopening the picker skips API calls entirely.
        let picker_cache_data = picker_cache::read(&audio_path);

        // Determine the "already pasted" (provider, model, streaming) triple.
        // The picker cache's `selected` field takes precedence
        // (the user may have picked a different model last time).
        // Fall back to the recording metadata otherwise.
        let selected_key: Option<(Provider, String, bool)> = picker_cache_data
            .selected
            .as_ref()
            .and_then(|s| {
                Some((
                    s.provider.parse::<Provider>().ok()?,
                    s.model.clone(),
                    s.streaming,
                ))
            })
            .or_else(|| {
                cached_brief.as_ref().and_then(|b| {
                    Some((
                        b.provider.as_deref()?.parse().ok()?,
                        b.model.as_deref()?.to_string(),
                        false,
                    ))
                })
            });

        // Collect all available cached transcriptions.
        // Tuple: (provider, model, text, streaming)
        let mut all_entries: Vec<(Provider, String, String, bool)> = Vec::new();

        // From recording metadata (always batch / streaming=false).
        if let Some(ref brief) = cached_brief {
            if let (Some(ps), Some(m)) = (brief.provider.as_deref(), brief.model.as_deref()) {
                if let Ok(p) = ps.parse::<Provider>() {
                    all_entries.push((p, m.to_string(), brief.transcript.clone(), false));
                }
            }
        }

        // From picker cache results (skip duplicates).
        for cr in &picker_cache_data.results {
            if let Ok(p) = cr.provider.parse::<Provider>() {
                if !all_entries
                    .iter()
                    .any(|(ep, em, _, es)| *ep == p && *em == cr.model && *es == cr.streaming)
                {
                    all_entries.push((p, cr.model.clone(), cr.text.clone(), cr.streaming));
                }
            }
        }

        // Build cached_entries with is_primary flag.  The selected
        // (already-pasted) entry goes first so it is pre-selected
        // in the GTK list.
        // Tuple: (provider, model, text, is_primary, streaming)
        let mut cached_entries: Vec<(Provider, String, String, bool, bool)> = Vec::new();
        if let Some((ref sp, ref sm, ss)) = selected_key {
            if let Some(idx) = all_entries
                .iter()
                .position(|(p, m, _, s)| p == sp && m == sm && *s == ss)
            {
                let (p, m, t, s) = all_entries.remove(idx);
                cached_entries.push((p, m, t, true, s));
            }
        }
        for (p, m, t, s) in all_entries {
            cached_entries.push((p, m, t, false, s));
        }

        log::debug!(
            "picker cache: {} cached entries (primary={}, from_cache={})",
            cached_entries.len(),
            cached_entries.iter().filter(|(_, _, _, p, _)| *p).count(),
            picker_cache_data.results.len(),
        );
        for (p, m, _, is_primary, streaming) in &cached_entries {
            log::debug!(
                "  cached: {}:{} (primary={}, streaming={})",
                p,
                m,
                is_primary,
                streaming,
            );
        }

        let candidates = build_retry_candidates(&config, opts.provider, opts.model.as_deref());
        log::debug!("picker candidates: {} total", candidates.len());
        for (p, m, s) in &candidates {
            log::debug!("  candidate: {}:{} (streaming={})", p, m, s);
        }

        // Filter out every (provider, model, streaming) triple that
        // already has a cached result — no need to re-transcribe.
        let filtered: Vec<(Provider, String, bool)> = candidates
            .into_iter()
            .filter(|(p, m, s)| {
                let dominated = cached_entries
                    .iter()
                    .any(|(cp, cm, _, _, cs)| cp == p && cm == m && cs == s);
                if dominated {
                    log::debug!("  filtered out (cached): {}:{} (streaming={})", p, m, s);
                }
                !dominated
            })
            .collect();
        log::debug!(
            "picker: {} transcribers needed (after filtering)",
            filtered.len(),
        );
        for (p, m, s) in &filtered {
            log::debug!("  needs API call: {}:{} (streaming={})", p, m, s);
        }

        // Split filtered candidates into batch and realtime groups.
        let batch_filtered: Vec<(Provider, String)> = filtered
            .iter()
            .filter(|(_, _, s)| !s)
            .map(|(p, m, _)| (*p, m.clone()))
            .collect();
        let realtime_filtered: Vec<(Provider, String)> = filtered
            .iter()
            .filter(|(_, _, s)| *s)
            .map(|(p, m, _)| (*p, m.clone()))
            .collect();

        // Create batch transcribers before entering GTK (needs &Config).
        let mut transcribers: Vec<(Provider, String, Box<dyn BatchTranscriber>)> = Vec::new();
        for (provider, model) in batch_filtered {
            match transcription::create_batch_transcriber(&config, provider, Some(&model), false) {
                Ok(t) => transcribers.push((provider, model, t)),
                Err(e) => log::warn!("skipping batch {}:{}: {}", provider, model, e),
            }
        }

        // Create realtime transcribers.
        let mut rt_transcribers: Vec<(Provider, String, Box<dyn RealtimeTranscriber>)> = Vec::new();
        for (provider, model) in realtime_filtered {
            match transcription::create_realtime_transcriber(&config, provider, Some(&model)) {
                Ok(t) => rt_transcribers.push((provider, model, t)),
                Err(e) => log::warn!("skipping realtime {}:{}: {}", provider, model, e),
            }
        }

        if transcribers.is_empty() && cached_entries.is_empty() && rt_transcribers.is_empty() {
            return Err(TalkError::Transcription(
                "no transcription providers available".to_string(),
            ));
        }

        let audio_path_for_selection = audio_path.clone();
        let selected = pick_with_streaming_gtk(
            transcribers,
            audio_path,
            cached_entries,
            config,
            rt_transcribers,
        )
        .await?;
        let selection = match selected {
            Some(s) => s,
            None => return Ok(()),
        };

        // Record which entry the user selected so it appears
        // pre-selected the next time the picker opens.
        if let Err(e) = picker_cache::write_selected(
            &audio_path_for_selection,
            &selection.provider.to_string(),
            &selection.model,
            selection.streaming,
        ) {
            log::warn!("failed to update picker selection: {}", e);
        }

        // If the user selected the cached entry, nothing to do — the
        // text is already in the target window.
        if selection.is_cached {
            log::info!("cached entry selected — no paste needed");
            return Ok(());
        }

        let delete_chars = if opts.replace_last_paste {
            // Prefer the paste-state file (written after every paste,
            // including picker selections) over recording metadata so
            // that successive picker replacements delete the correct
            // number of characters.
            recording_cache::read_last_paste_state()?
                .map(|s| s.char_count)
                .or(replace_char_count)
                .unwrap_or(0)
        } else {
            0
        };

        paste_text_to_target(
            target_window.as_ref(),
            &selection.text,
            delete_chars,
            paste_chunk_chars,
        )
        .await?;
        let _ = recording_cache::write_last_paste_state(target_window.as_deref(), &selection.text);
        println!("{}", selection.text);
        return Ok(());
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

/// Realtime dictation mode via WebSocket.
///
/// Streams raw PCM audio to the transcription API and receives
/// incremental transcription events. Returns the accumulated text.
///
/// Audio is always tee'd to `cache_wav_path` so the recording is
/// cached for later review.
///
/// `player` and `boop_token` are passed so that when recording stops
/// (SIGINT), the stop sound fires immediately — the user hears it the
/// instant they toggle, not after the WebSocket finishes.
///
/// When `visualizer` is provided, the live transcription text is pushed
/// to the text overlay as words arrive.
#[allow(clippy::too_many_arguments)]
async fn dictate_realtime(
    config: Config,
    provider: Provider,
    model: Option<&str>,
    cache_wav_path: &std::path::Path,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    capture: &mut dyn AudioCapture,
    from_file: bool,
    player: Option<&SoundPlayer>,
    boop_token: Option<&CancellationToken>,
    segment_tx: Option<tokio::sync::mpsc::Sender<String>>,
    visualizer: Option<&VisualizerHandle>,
    shutdown: &CancellationToken,
) -> Result<TranscriptionResult, TalkError> {
    // Create and validate the transcriber before starting audio capture
    // so the user gets immediate feedback on misconfiguration.
    let transcriber = transcription::create_realtime_transcriber(&config, provider, model)?;
    log::info!("validating {} provider configuration", provider);
    transcriber.validate().await?;

    // Always tee audio to the cache WAV for recording cache.
    log::info!("caching audio to: {}", cache_wav_path.display());
    let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
    let wav_task = tokio::spawn(audio_tee_to_wav(
        audio_rx,
        fwd_tx,
        cache_wav_path.to_path_buf(),
        AudioConfig::new(),
    ));
    let audio_rx = fwd_rx;

    let mut event_rx = transcriber.transcribe_realtime(audio_rx).await?;
    let started = std::time::Instant::now();

    if from_file {
        log::info!("transcribing audio file (realtime)...");
    } else {
        log::info!("recording (realtime)... press Ctrl+C to stop");
    }

    let capture_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let capture_stop_clone = capture_stop.clone();

    // Wait for the shared shutdown token (registered early in dictate())
    // instead of a local ctrl_c() handler.  This avoids a race window
    // where SIGINT arrives before this task is spawned.
    let shutdown_clone = shutdown.clone();
    let ctrlc_task = tokio::spawn(async move {
        log::warn!("[DBG] dictate_realtime: waiting on shutdown token");
        shutdown_clone.cancelled().await;
        log::warn!("[DBG] dictate_realtime: shutdown token fired, setting capture_stop");
        capture_stop_clone.store(true, std::sync::atomic::Ordering::Release);
    });

    // Completed sentences/phrases emitted so far.
    let mut segments: Vec<String> = Vec::new();
    // Buffer for the current in-progress phrase (live TextDelta).
    let mut current_line = String::new();
    let mut detected_language: Option<String> = None;
    let mut unknown_event_types: Vec<String> = Vec::new();
    let mut event_counts: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    let mut api_segment_count: usize = 0;
    let mut session_id: Option<String> = None;
    let mut conversation_id: Option<String> = None;
    let mut last_rate_limits: Option<serde_json::Value> = None;
    let mut ws_upgrade_headers: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();

    let bump = |key: &str, counts: &mut std::collections::BTreeMap<String, u64>| {
        let entry = counts.entry(key.to_string()).or_insert(0);
        *entry += 1;
    };

    loop {
        // Check if Ctrl+C was pressed — stop capture to trigger end-of-audio
        if capture_stop.load(std::sync::atomic::Ordering::Acquire) {
            log::info!("stopping recording");

            // Immediate audible + visual feedback: the user hears the
            // stop sound the instant they toggle, not after the
            // transcription WebSocket finishes.
            if let Some(token) = boop_token {
                token.cancel();
            }
            if let Some(p) = player {
                let stop = p.sounds.stop.clone();
                p.play(&stop);
            }

            capture.stop()?;
            // Reset so we don't stop again
            capture_stop.store(false, std::sync::atomic::Ordering::Release);
        }

        tokio::select! {
            event = event_rx.recv() => {
                match event {
                    Some(TranscriptionEvent::TextDelta { text }) => {
                        bump("text_delta", &mut event_counts);
                        current_line.push_str(&text);
                        eprint!("\r{}", current_line);

                        // Push live text to the overlay.
                        if let Some(viz) = visualizer {
                            let mut live = segments.join(" ");
                            if !live.is_empty() && !current_line.is_empty() {
                                live.push(' ');
                            }
                            live.push_str(&current_line);
                            viz.set_text(&live);
                        }

                        // Flush completed sentences from the buffer.
                        // Split on sentence-ending punctuation followed by
                        // whitespace or end-of-string.
                        let prev_count = segments.len();
                        flush_sentences(&mut current_line, &mut segments);
                        if let Some(ref tx) = segment_tx {
                            for seg in &segments[prev_count..] {
                                let _ = tx.send(seg.clone()).await;
                            }
                        }
                    }
                    Some(TranscriptionEvent::SegmentDelta { text, .. }) => {
                        bump("segment_delta", &mut event_counts);
                        api_segment_count += 1;
                        // If the API sends segment events, use them as
                        // authoritative sentence boundaries.
                        let segment_text = text.trim().to_string();
                        if !segment_text.is_empty() {
                            println!("{}", segment_text);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(segment_text.clone()).await;
                            }
                            segments.push(segment_text);
                        }
                        let blank = " ".repeat(current_line.len());
                        eprint!("\r{}\r", blank);
                        current_line.clear();

                        // Update overlay with completed segments.
                        if let Some(viz) = visualizer {
                            viz.set_text(&segments.join(" "));
                        }
                    }
                    Some(TranscriptionEvent::Done) => {
                        bump("done", &mut event_counts);
                        // Flush any trailing text that didn't end with punctuation
                        let trailing = current_line.trim().to_string();
                        if !trailing.is_empty() {
                            println!("{}", trailing);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(trailing.clone()).await;
                            }
                            segments.push(trailing);
                        }
                        eprintln!();
                        break;
                    }
                    Some(TranscriptionEvent::Error { message }) => {
                        bump("error", &mut event_counts);
                        return Err(TalkError::Transcription(format!(
                            "Realtime transcription error: {}",
                            message
                        )));
                    }
                    Some(TranscriptionEvent::SessionCreated) => {
                        bump("session_created", &mut event_counts);
                        log::debug!("session created event received");
                    }
                    Some(TranscriptionEvent::SessionInfo { session_id: sid, conversation_id: cid }) => {
                        bump("session_info", &mut event_counts);
                        if sid.is_some() {
                            session_id = sid;
                        }
                        if cid.is_some() {
                            conversation_id = cid;
                        }
                    }
                    Some(TranscriptionEvent::RateLimitsUpdated { raw }) => {
                        bump("rate_limits_updated", &mut event_counts);
                        last_rate_limits = Some(raw);
                    }
                    Some(TranscriptionEvent::TransportMetadata { headers }) => {
                        bump("transport_metadata", &mut event_counts);
                        ws_upgrade_headers.extend(headers);
                    }
                    Some(TranscriptionEvent::Language { language }) => {
                        bump("language", &mut event_counts);
                        log::info!("detected language: {}", language);
                        detected_language = Some(language);
                    }
                    Some(TranscriptionEvent::Unknown { event_type, .. }) => {
                        bump("unknown", &mut event_counts);
                        if let Some(kind) = event_type {
                            bump(&format!("event:{kind}"), &mut event_counts);
                            if !unknown_event_types.contains(&kind) {
                                unknown_event_types.push(kind);
                            }
                        }
                    }
                    None => {
                        // Channel closed without Done event
                        let trailing = current_line.trim().to_string();
                        if !trailing.is_empty() {
                            println!("{}", trailing);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(trailing.clone()).await;
                            }
                            segments.push(trailing);
                        }
                        eprintln!();
                        break;
                    }
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                // Periodic check for Ctrl+C flag
            }
        }
    }

    ctrlc_task.abort();

    // Wait for WAV tee task to finish writing
    match wav_task.await {
        Ok(Ok(())) => log::debug!("cache WAV saved"),
        Ok(Err(e)) => log::warn!("cache WAV write error: {}", e),
        Err(e) => log::warn!("cache WAV task panicked: {}", e),
    }

    let provider_specific = match provider {
        Provider::OpenAI => Some(ProviderSpecificMetadata::OpenAI(OpenAIProviderMetadata {
            model: model.map(str::to_string),
            usage_raw: None,
            rate_limit_headers: std::collections::BTreeMap::new(),
            unknown_event_types,
            realtime: Some(OpenAIRealtimeMetadata {
                session_id,
                conversation_id,
                event_counts,
                last_rate_limits,
                ws_upgrade_headers: ws_upgrade_headers.clone(),
            }),
        })),
        Provider::Mistral => Some(ProviderSpecificMetadata::Mistral(MistralProviderMetadata {
            model: model.map(str::to_string),
            usage_raw: None,
            unknown_event_types,
        })),
    };

    Ok(TranscriptionResult {
        text: segments.join(" "),
        metadata: TranscriptionMetadata {
            request_latency_ms: None,
            session_elapsed_ms: Some(started.elapsed().as_millis() as u64),
            request_id: ws_upgrade_headers.get("x-request-id").cloned(),
            provider_processing_ms: ws_upgrade_headers
                .get("openai-processing-ms")
                .and_then(|s| s.parse::<u64>().ok()),
            detected_language,
            audio_seconds: None,
            segment_count: Some(if api_segment_count > 0 {
                api_segment_count
            } else {
                segments.len()
            }),
            word_count: None,
            token_usage: None,
            provider_specific,
        },
        diarization: None,
    })
}

/// Tee audio from `source` into both a WAV file and a forwarding channel.
///
/// Each `Vec<i16>` chunk is forwarded to `fwd_tx` for the transcriber
/// and then written as raw PCM s16le to the WAV file.  The WAV is an
/// exact mirror of what the API received — not more, not less.  When the
/// source channel closes, the WAV header is patched with the final size.
async fn audio_tee_to_wav(
    mut source: tokio::sync::mpsc::Receiver<Vec<i16>>,
    fwd_tx: tokio::sync::mpsc::Sender<Vec<i16>>,
    wav_path: PathBuf,
    audio_config: AudioConfig,
) -> Result<(), TalkError> {
    let mut writer = WavWriter::new(audio_config);
    let header = writer.header()?;

    let mut file = tokio::fs::File::create(&wav_path)
        .await
        .map_err(TalkError::Io)?;
    file.write_all(&header).await.map_err(TalkError::Io)?;

    let mut total_samples: u64 = 0;

    while let Some(pcm_chunk) = source.recv().await {
        // Forward to transcriber first.  Only write to the debug WAV
        // what was successfully forwarded, so the capture is an exact
        // mirror of what the API received.
        let wav_chunk = pcm_chunk.clone();
        if fwd_tx.send(pcm_chunk).await.is_err() {
            log::debug!("transcriber channel closed, stopping debug WAV");
            break;
        }

        total_samples += wav_chunk.len() as u64;
        let pcm_bytes = writer.write_pcm(&wav_chunk)?;
        file.write_all(&pcm_bytes).await.map_err(TalkError::Io)?;
    }

    // No drain loop — the WAV contains exactly what was forwarded.

    // Signal end-of-audio to transcriber before WAV finalisation.
    drop(fwd_tx);

    // Patch WAV header with actual data size
    let final_header = writer.finalize()?;
    file.seek(std::io::SeekFrom::Start(0))
        .await
        .map_err(TalkError::Io)?;
    file.write_all(&final_header).await.map_err(TalkError::Io)?;
    file.sync_all().await.map_err(TalkError::Io)?;

    log::info!(
        "debug WAV: {} samples ({:.1}s) saved to {}",
        total_samples,
        total_samples as f64 / 16000.0,
        wav_path.display()
    );

    Ok(())
}

/// Batch dictation mode.
///
/// Encodes audio and streams it directly to a single transcription request.
/// Audio is also tee'd to `cache_wav_path` for the recording cache.
///
/// When `from_file` is true, the function waits for the audio source to
/// exhaust naturally (in addition to allowing Ctrl+C for early abort).
#[allow(clippy::too_many_arguments)]
async fn dictate_streaming(
    capture: &mut dyn AudioCapture,
    from_file: bool,
    audio_config: AudioConfig,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    cache_wav_path: &std::path::Path,
    transcriber: Box<dyn BatchTranscriber>,
    shutdown: &CancellationToken,
    player: Option<&SoundPlayer>,
    boop_token: Option<&CancellationToken>,
    overlay: Option<&OverlayHandle>,
    visualizer: Option<&VisualizerHandle>,
) -> Result<TranscriptionResult, TalkError> {
    // Tee raw PCM to cache WAV before encoding
    log::info!("caching audio to: {}", cache_wav_path.display());
    let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
    let cache_wav_task = tokio::spawn(audio_tee_to_wav(
        audio_rx,
        fwd_tx,
        cache_wav_path.to_path_buf(),
        audio_config.clone(),
    ));

    // Create channel for streaming encoded audio to transcriber
    let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(25);

    // Oneshot to signal when encoding is done (file source exhausted
    // or capture stopped).  Enables the stop logic to race Ctrl+C
    // against natural completion for file input.
    let (encode_done_tx, encode_done_rx) = tokio::sync::oneshot::channel::<()>();

    // Spawn encode task: PCM → OGG Opus → stream_tx
    let encode_task = tokio::spawn(async move {
        let mut rx = fwd_rx;
        let mut writer = match OggOpusWriter::new(audio_config) {
            Ok(writer) => writer,
            Err(err) => {
                log::error!("error creating audio writer: {}", err);
                return Err(err);
            }
        };

        let header = match writer.header() {
            Ok(bytes) => bytes,
            Err(err) => {
                log::error!("error creating audio header: {}", err);
                return Err(err);
            }
        };
        if stream_tx.send(header).await.is_err() {
            log::warn!("transcription stream closed during header send");
            let _ = encode_done_tx.send(());
            return Ok::<(), TalkError>(());
        }

        while let Some(pcm_chunk) = rx.recv().await {
            let encoded_data = match writer.write_pcm(&pcm_chunk) {
                Ok(data) => data,
                Err(err) => {
                    log::error!("error encoding audio: {}", err);
                    let _ = encode_done_tx.send(());
                    return Err(err);
                }
            };
            if !encoded_data.is_empty() && stream_tx.send(encoded_data).await.is_err() {
                log::warn!("transcription stream closed during audio send");
                break;
            }
        }

        // Finalize writer
        match writer.finalize() {
            Ok(remaining) => {
                if !remaining.is_empty() {
                    let _ = stream_tx.send(remaining).await;
                }
            }
            Err(err) => {
                log::error!("error finalizing audio writer: {}", err);
                let _ = encode_done_tx.send(());
                return Err(err);
            }
        }

        // stream_tx is dropped here, closing the channel
        let _ = encode_done_tx.send(());
        Ok::<(), TalkError>(())
    });

    // Spawn transcription task
    let mut transcribe_task =
        tokio::spawn(async move { transcriber.transcribe_stream(stream_rx, "audio.ogg").await });

    // Wait for recording to end: SIGINT (via shared shutdown token)
    // for live mic, natural completion for file input, or SIGINT to
    // abort file early.  The shutdown token is registered early in
    // dictate() so there is no race window.
    let rec_start = std::time::Instant::now();
    if from_file {
        log::info!("transcribing audio file (batch)...");
        tokio::select! {
            _ = shutdown.cancelled() => {
                log::info!("aborting after {:.2}s", rec_start.elapsed().as_secs_f64());
                capture.stop()?;
            }
            _ = encode_done_rx => {
                log::info!("audio file playback complete after {:.2}s", rec_start.elapsed().as_secs_f64());
            }
        }
    } else {
        log::info!("recording — waiting for shutdown signal");
        shutdown.cancelled().await;
        log::info!(
            "shutdown signal received after {:.2}s recording — stopping capture",
            rec_start.elapsed().as_secs_f64()
        );
        capture.stop()?;
        log::info!("capture stopped");

        // Immediate audible + visual feedback: the user hears the
        // stop sound and sees the "transcribing" badge the instant
        // they toggle, not after the API call finishes.
        if let Some(token) = boop_token {
            token.cancel();
        }
        if let Some(p) = player {
            p.play_stop().await;
        }
        if let Some(viz) = visualizer {
            viz.hide();
        }
        if let Some(o) = overlay {
            o.show(IndicatorKind::Transcribing);
        }
    }

    // Race the transcription result against a 5-second timeout FIRST.
    //
    // We must NOT await encode_task before this: when the API is hung
    // the transcriber stops reading from stream_rx, so the bounded
    // channel (capacity 25) fills up and stream_tx.send() inside
    // encode_task blocks forever.  Awaiting encode_task would therefore
    // deadlock.  Instead we race the transcription against the timeout,
    // then clean up the pipeline afterwards.
    const TRANSCRIPTION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    let t0 = std::time::Instant::now();
    log::info!(
        "waiting for transcription (timeout: {:?})",
        TRANSCRIPTION_TIMEOUT
    );

    let result = tokio::select! {
        res = &mut transcribe_task => {
            log::info!("transcription completed after {:.2}s", t0.elapsed().as_secs_f64());
            match res {
                Ok(Ok(result)) => Ok(result),
                Ok(Err(err)) => Err(err),
                Err(err) => Err(TalkError::Transcription(format!(
                    "transcription task panicked: {}", err
                ))),
            }
        }
        _ = tokio::time::sleep(TRANSCRIPTION_TIMEOUT) => {
            log::warn!("transcription timed out after {:.2}s — aborting", t0.elapsed().as_secs_f64());
            transcribe_task.abort();
            Err(TalkError::Transcription(
                "transcription timed out after 5 s".to_string(),
            ))
        }
    };

    // Clean up the pipeline.  Aborting the transcribe_task (on timeout)
    // drops stream_rx, which unblocks encode_task's stream_tx.send().
    // Give encode_task a short grace period to finish, then abort it.
    const CLEANUP_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

    match tokio::time::timeout(CLEANUP_GRACE, encode_task).await {
        Ok(Ok(Ok(()))) => log::debug!("encode task completed"),
        Ok(Ok(Err(err))) => log::warn!("encode error: {}", err),
        Ok(Err(err)) => log::warn!("encode task panicked: {}", err),
        Err(_) => log::warn!(
            "encode task did not finish within {:?} — abandoned",
            CLEANUP_GRACE
        ),
    }

    match tokio::time::timeout(CLEANUP_GRACE, cache_wav_task).await {
        Ok(Ok(Ok(()))) => log::debug!("cache WAV task completed"),
        Ok(Ok(Err(err))) => log::warn!("cache WAV write error: {}", err),
        Ok(Err(err)) => log::warn!("cache WAV task panicked: {}", err),
        Err(_) => log::warn!(
            "cache WAV task did not finish within {:?} — abandoned",
            CLEANUP_GRACE
        ),
    }

    log::info!(
        "dictate_streaming total elapsed: {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    result
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
