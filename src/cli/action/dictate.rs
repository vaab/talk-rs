//! Dictate command implementation.
//!
//! Records audio, streams it to the transcription API, and pastes the result
//! into the focused application via clipboard.

use crate::core::audio::cpal_capture::CpalCapture;
use crate::core::audio::indicator::SoundPlayer;
use crate::core::audio::{AudioCapture, AudioWriter, OggOpusWriter, WavWriter};
use crate::core::clipboard::{Clipboard, X11Clipboard};
use crate::core::config::{AudioConfig, Config, Provider};
use crate::core::daemon::{self, DaemonStatus};
use crate::core::error::TalkError;
use crate::core::overlay::{IndicatorKind, OverlayHandle};
use crate::core::transcription::{self, BatchTranscriber, TranscriptionEvent};
use crate::core::visualizer::VisualizerHandle;
use std::os::unix::process::CommandExt as _;
use std::path::PathBuf;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

/// Options for the dictate command.
pub struct DictateOpts {
    pub args: Vec<String>,
    pub provider: Option<Provider>,
    pub model: Option<String>,
    pub realtime: bool,
    pub toggle: bool,
    pub no_sounds: bool,
    pub no_overlay: bool,
    pub amplitude: bool,
    pub spectrum: bool,
    pub daemon: bool,
    pub target_window: Option<String>,
    pub verbose: u8,
}

/// Resolve the effective provider from CLI override or config default.
fn resolve_provider(cli_provider: Option<Provider>, config: &Config) -> Provider {
    if let Some(p) = cli_provider {
        return p;
    }
    config
        .transcription
        .as_ref()
        .map(|t| t.default_provider)
        .unwrap_or(Provider::Mistral)
}

/// Parse command-line arguments for the dictate command.
///
/// Returns an optional output file path for saving the audio recording.
pub fn parse_args(args: &[String]) -> Result<Option<PathBuf>, TalkError> {
    match args.len() {
        0 => Ok(None),
        1 => Ok(Some(PathBuf::from(&args[0]))),
        _ => Err(TalkError::Audio(
            "dictate command takes at most one argument (output file path)".to_string(),
        )),
    }
}

/// Get the currently focused window ID using xdotool.
async fn get_active_window() -> Option<String> {
    let output = tokio::process::Command::new("xdotool")
        .arg("getactivewindow")
        .output()
        .await
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

/// Focus a window by ID using xdotool.
async fn focus_window(window_id: &str) -> bool {
    let result = tokio::process::Command::new("xdotool")
        .args(["windowactivate", "--sync", window_id])
        .output()
        .await;

    matches!(result, Ok(output) if output.status.success())
}

/// Simulate a key combination using xdotool.
async fn simulate_paste() -> Result<(), TalkError> {
    let output = tokio::process::Command::new("xdotool")
        .args(["key", "ctrl+shift+v"])
        .output()
        .await
        .map_err(|e| TalkError::Clipboard(format!("failed to run xdotool: {e}")))?;

    if !output.status.success() {
        return Err(TalkError::Clipboard(
            "xdotool key simulation failed".to_string(),
        ));
    }
    Ok(())
}

/// Dictate: record audio, transcribe, and paste into focused application.
pub async fn dictate(opts: DictateOpts) -> Result<(), TalkError> {
    // Toggle mode: start or stop a daemon
    if opts.toggle {
        return toggle_dispatch(
            opts.provider,
            opts.model,
            opts.realtime,
            opts.no_sounds,
            opts.no_overlay,
            opts.amplitude,
            opts.spectrum,
            opts.verbose,
        )
        .await;
    }

    let save_path = parse_args(&opts.args)?;

    // Load configuration
    let config = Config::load(None)?;

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
        match VisualizerHandle::new(opts.amplitude, opts.spectrum) {
            Ok(h) => {
                log::debug!(
                    "visualizer initialized (amplitude={}, spectrum={})",
                    opts.amplitude,
                    opts.spectrum,
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

    // Play start sound
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

    // Start boop loop (every 5 seconds)
    let boop_token = player
        .as_ref()
        .map(|p| p.start_boop_loop(std::time::Duration::from_secs(5)));

    log::info!(
        "starting {} transcription",
        if opts.realtime { "realtime" } else { "batch" }
    );

    let text = if opts.realtime {
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

        // Player and boop_token are passed so the stop sound fires
        // immediately when the user toggles — not after the WebSocket
        // finishes collecting transcription results.
        let provider = resolve_provider(opts.provider, &config);
        let text = dictate_realtime(
            config,
            provider,
            opts.model.as_deref(),
            save_path.as_deref(),
            player.as_ref(),
            boop_token.as_ref(),
            Some(seg_tx),
            visualizer.as_ref(),
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

        text
    } else {
        // Batch mode (default): capture audio, encode, then transcribe
        let provider = resolve_provider(opts.provider, &config);
        let transcriber =
            transcription::create_batch_transcriber(&config, provider, opts.model.as_deref())?;

        // Pre-flight: verify provider connectivity and model validity
        // before starting audio capture.
        log::info!("validating {} provider configuration", provider);
        transcriber.validate().await?;

        let mut capture = CpalCapture::new(config.audio.clone());
        let audio_rx = capture.start()?;

        // Optionally create output file for saving audio
        let save_file = if let Some(ref path) = save_path {
            Some(tokio::fs::File::create(path).await.map_err(TalkError::Io)?)
        } else {
            None
        };

        dictate_streaming(
            &mut capture,
            config.audio.clone(),
            audio_rx,
            save_file,
            transcriber,
        )
        .await?
    };

    // Stop boop loop (idempotent — may already be cancelled by
    // dictate_realtime for realtime mode).
    if let Some(token) = boop_token {
        log::debug!("stopping boop loop");
        token.cancel();
    }

    // Hide recording indicator and visualizer
    if let Some(ref o) = overlay {
        log::debug!("hiding overlay");
        o.hide();
    }
    if let Some(ref viz) = visualizer {
        log::debug!("hiding visualizer");
        viz.hide();
    }

    // For batch mode (default), play stop sound here (realtime mode
    // already played it inside dictate_realtime on SIGINT).
    if !opts.realtime {
        if let Some(ref p) = player {
            log::debug!("playing stop sound");
            p.play_stop().await;
        }
    }

    let text = text.trim().to_string();

    if text.is_empty() {
        log::warn!("empty transcription — nothing to paste");
        return Ok(());
    }

    log::info!("transcription: {}", text);

    // Paste into focused application (batch mode only;
    // realtime mode pastes per-segment during recording)
    if !opts.realtime {
        let clipboard = X11Clipboard::new();

        // Refocus target window
        if let Some(ref wid) = target_window {
            log::debug!("refocusing target window: {}", wid);
            if !focus_window(wid).await {
                log::warn!("could not refocus target window {}", wid);
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // Save current clipboard
        log::debug!("saving current clipboard");
        let saved_clipboard = clipboard.get_text().await.ok();

        // Set clipboard to transcription
        log::debug!("setting clipboard to transcription text");
        clipboard.set_text(&text).await?;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Simulate paste
        log::debug!("simulating paste (ctrl+shift+v)");
        simulate_paste().await?;

        // Restore clipboard after paste
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        if let Some(saved) = saved_clipboard {
            log::debug!("restoring original clipboard");
            let _ = clipboard.set_text(&saved).await;
        }
    }

    if let Some(ref path) = save_path {
        log::info!("audio saved to: {}", path.display());
    }

    // Print transcription to stdout (batch mode only;
    // realtime mode already prints segments as they arrive)
    if !opts.realtime {
        println!("{}", text);
    }

    // If running as daemon, clean up PID file on normal exit
    if opts.daemon {
        if let Ok(pid_file) = daemon::pid_path() {
            let _ = daemon::remove_pid_file(&pid_file);
        }
    }

    Ok(())
}

/// Toggle dispatch: start a new daemon or stop a running one.
#[allow(clippy::too_many_arguments)]
async fn toggle_dispatch(
    provider: Option<Provider>,
    model: Option<String>,
    realtime: bool,
    no_sounds: bool,
    no_overlay: bool,
    amplitude: bool,
    spectrum: bool,
    verbose: u8,
) -> Result<(), TalkError> {
    let pid_file = daemon::pid_path()?;

    // Acquire exclusive lock to prevent race between concurrent toggle calls
    let _lock = daemon::acquire_lock()?;

    match daemon::check_status(&pid_file)? {
        DaemonStatus::NotRunning => {
            toggle_start(
                &pid_file, provider, model, realtime, no_sounds, no_overlay, amplitude, spectrum,
                verbose,
            )
            .await
        }
        DaemonStatus::Running { pid } => toggle_stop(pid, &pid_file),
    }
}

/// Start a new daemon: capture window, spawn detached dictate process, write PID.
#[allow(clippy::too_many_arguments)]
async fn toggle_start(
    pid_file: &std::path::Path,
    provider: Option<Provider>,
    model: Option<String>,
    realtime: bool,
    no_sounds: bool,
    no_overlay: bool,
    amplitude: bool,
    spectrum: bool,
    verbose: u8,
) -> Result<(), TalkError> {
    // Pre-flight: validate provider/model before spawning the daemon
    // so the user gets immediate feedback on misconfiguration.
    let config = Config::load(None)?;
    let effective_provider = resolve_provider(provider, &config);
    log::info!("validating {} provider configuration", effective_provider);
    if realtime {
        let t = transcription::create_realtime_transcriber(
            &config,
            effective_provider,
            model.as_deref(),
        )?;
        t.validate().await?;
    } else {
        let t =
            transcription::create_batch_transcriber(&config, effective_provider, model.as_deref())?;
        t.validate().await?;
    }

    // Capture active window before spawning daemon
    let target_window = get_active_window().await;

    // Find our own executable
    let exe = std::env::current_exe()
        .map_err(|e| TalkError::Config(format!("failed to determine current executable: {}", e)))?;

    // Build daemon command: talk-rs [-v...] dictate --daemon [--realtime] [--no-sounds] [--target-window=WID]
    let mut cmd = std::process::Command::new(&exe);

    // Forward verbosity level (before subcommand)
    if verbose > 0 {
        cmd.arg(format!("-{}", "v".repeat(verbose as usize)));
    }

    cmd.arg("dictate").arg("--daemon");

    if let Some(p) = provider {
        cmd.arg("--provider").arg(p.to_string());
    }

    if let Some(ref m) = model {
        cmd.arg("--model").arg(m);
    }

    if realtime {
        cmd.arg("--realtime");
    }

    if no_sounds {
        cmd.arg("--no-sounds");
    }

    if no_overlay {
        cmd.arg("--no-overlay");
    }

    if amplitude {
        cmd.arg("--amplitude");
    }

    if spectrum {
        cmd.arg("--spectrum");
    }

    if let Some(ref wid) = target_window {
        cmd.arg("--target-window").arg(wid);
    }

    // Redirect stdout/stderr to log file
    let log_file_path = daemon::log_path()?;
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_path)
        .map_err(|e| {
            TalkError::Config(format!(
                "failed to open log file {}: {}",
                log_file_path.display(),
                e
            ))
        })?;
    let log_stderr = log_file
        .try_clone()
        .map_err(|e| TalkError::Config(format!("failed to clone log file handle: {}", e)))?;

    cmd.stdout(std::process::Stdio::from(log_file));
    cmd.stderr(std::process::Stdio::from(log_stderr));
    cmd.stdin(std::process::Stdio::null());

    // Create new process group (equivalent to setsid for signal isolation)
    cmd.process_group(0);

    let child = cmd
        .spawn()
        .map_err(|e| TalkError::Config(format!("failed to spawn daemon process: {}", e)))?;

    let child_pid = child.id();
    daemon::write_pid_file(pid_file, child_pid)?;

    log::info!(
        "dictation started (PID {}, logs: {})",
        child_pid,
        log_file_path.display()
    );

    Ok(())
}

/// Stop a running daemon: SIGINT, wait, SIGTERM fallback, clean up PID file.
fn toggle_stop(pid: u32, pid_file: &std::path::Path) -> Result<(), TalkError> {
    log::info!("stopping dictation (PID {})", pid);
    daemon::stop_daemon(pid, pid_file)?;
    log::info!("dictation stopped");
    Ok(())
}

/// Flush completed sentences from the live buffer to stdout.
///
/// Scans the buffer for sentence-ending punctuation (`.` `!` `?` `。` `！` `？`)
/// followed by whitespace. Everything up to and including the punctuation is
/// emitted as a line on stdout and appended to `segments`. The remainder stays
/// in the buffer for further accumulation.
fn flush_sentences(buffer: &mut String, segments: &mut Vec<String>) {
    loop {
        // Find the earliest sentence-ending punctuation followed by whitespace.
        let boundary = buffer.char_indices().position(|(i, ch)| {
            if matches!(ch, '。' | '！' | '？') {
                // CJK sentence-ending punctuation: always a boundary
                // (no space expected between CJK sentences)
                true
            } else if matches!(ch, '.' | '!' | '?') {
                // Latin sentence-ending punctuation: require whitespace
                // or end-of-string after it to avoid splitting "3.14"
                let after = i + ch.len_utf8();
                after >= buffer.len() || buffer[after..].starts_with(|c: char| c.is_whitespace())
            } else {
                false
            }
        });

        let Some(pos) = boundary else {
            break;
        };

        // Convert char position back to byte offset (including the punctuation char)
        let (byte_offset, punct_char) = buffer.char_indices().nth(pos).unwrap_or((0, '.'));
        let split_at = byte_offset + punct_char.len_utf8();

        let sentence = buffer[..split_at].trim().to_string();
        if !sentence.is_empty() {
            println!("{}", sentence);
            segments.push(sentence);
        }

        // Remove the emitted sentence + any leading whitespace from the remainder
        let remainder = buffer[split_at..].trim_start().to_string();
        // Clear stderr live preview
        let blank = " ".repeat(buffer.len());
        eprint!("\r{}\r", blank);

        *buffer = remainder;
        if !buffer.is_empty() {
            eprint!("\r{}", buffer);
        }
    }
}

/// Realtime dictation mode via WebSocket.
///
/// Streams raw PCM audio to the Voxtral Realtime API and receives
/// incremental transcription events. Returns the accumulated text.
///
/// When `save_path` is provided, a debug WAV copy of the captured PCM
/// is written alongside the transcription so the user can verify that
/// recording start/stop timing and content are correct.
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
    save_path: Option<&std::path::Path>,
    player: Option<&SoundPlayer>,
    boop_token: Option<&CancellationToken>,
    segment_tx: Option<tokio::sync::mpsc::Sender<String>>,
    visualizer: Option<&VisualizerHandle>,
) -> Result<String, TalkError> {
    // Create and validate the transcriber before starting audio capture
    // so the user gets immediate feedback on misconfiguration.
    let transcriber = transcription::create_realtime_transcriber(&config, provider, model)?;
    log::info!("validating {} provider configuration", provider);
    transcriber.validate().await?;

    let mut capture = CpalCapture::new(config.audio.clone());
    let audio_rx = capture.start()?;

    // If a save path is given, tee the audio stream: one copy goes to
    // the transcriber, the other is written to a WAV file for debugging.
    let (audio_rx, wav_task) = if let Some(path) = save_path {
        let wav_path = if path.extension().is_some() {
            path.to_path_buf()
        } else {
            path.with_extension("wav")
        };
        log::info!("saving debug audio to: {}", wav_path.display());

        let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
        let audio_cfg = config.audio.clone();
        let task = tokio::spawn(audio_tee_to_wav(audio_rx, fwd_tx, wav_path, audio_cfg));
        (fwd_rx, Some(task))
    } else {
        // Also save a debug capture automatically in the cache directory
        let auto_path = crate::core::daemon::cache_dir()
            .ok()
            .map(|d| d.join("debug-capture.wav"));
        if let Some(ref wav_path) = auto_path {
            if let Some(parent) = wav_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            log::info!("saving debug audio to: {}", wav_path.display());
            let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
            let audio_cfg = config.audio.clone();
            let task = tokio::spawn(audio_tee_to_wav(
                audio_rx,
                fwd_tx,
                wav_path.clone(),
                audio_cfg,
            ));
            (fwd_rx, Some(task))
        } else {
            (audio_rx, None)
        }
    };

    let mut event_rx = transcriber.transcribe_realtime(audio_rx).await?;

    log::info!("recording (realtime)... press Ctrl+C to stop");

    let capture_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let capture_stop_clone = capture_stop.clone();

    // Spawn Ctrl+C handler that stops capture
    let ctrlc_task = tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        capture_stop_clone.store(true, std::sync::atomic::Ordering::Release);
    });

    // Completed sentences/phrases emitted so far.
    let mut segments: Vec<String> = Vec::new();
    // Buffer for the current in-progress phrase (live TextDelta).
    let mut current_line = String::new();

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
                        return Err(TalkError::Transcription(format!(
                            "Realtime transcription error: {}",
                            message
                        )));
                    }
                    Some(TranscriptionEvent::SessionCreated) => {
                        log::debug!("session created event received");
                    }
                    Some(TranscriptionEvent::Language { language }) => {
                        log::info!("detected language: {}", language);
                    }
                    Some(TranscriptionEvent::Unknown { .. }) => {
                        // Ignore unknown/future event types
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
    if let Some(task) = wav_task {
        match task.await {
            Ok(Ok(())) => log::debug!("debug WAV saved"),
            Ok(Err(e)) => log::warn!("debug WAV write error: {}", e),
            Err(e) => log::warn!("debug WAV task panicked: {}", e),
        }
    }

    Ok(segments.join(" "))
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
async fn dictate_streaming(
    capture: &mut CpalCapture,
    audio_config: AudioConfig,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    save_file: Option<tokio::fs::File>,
    transcriber: Box<dyn BatchTranscriber>,
) -> Result<String, TalkError> {
    // Create channel for streaming encoded audio to transcriber
    let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(25);

    // Spawn encode task: PCM → OGG Opus → stream_tx (+ optional file)
    let encode_task = tokio::spawn(async move {
        let mut rx = audio_rx;
        let mut file = save_file;
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
        if stream_tx.send(header.clone()).await.is_err() {
            log::warn!("transcription stream closed during header send");
            return Ok::<(), TalkError>(());
        }
        if let Some(ref mut f) = file {
            if let Err(err) = f.write_all(&header).await {
                log::error!("error writing header to file: {}", err);
            }
        }

        while let Some(pcm_chunk) = rx.recv().await {
            let encoded_data = match writer.write_pcm(&pcm_chunk) {
                Ok(data) => data,
                Err(err) => {
                    log::error!("error encoding audio: {}", err);
                    return Err(err);
                }
            };
            if !encoded_data.is_empty() {
                // Send to transcription stream
                if stream_tx.send(encoded_data.clone()).await.is_err() {
                    log::warn!("transcription stream closed during audio send");
                    break;
                }
                // Optionally write to file
                if let Some(ref mut f) = file {
                    if let Err(err) = f.write_all(&encoded_data).await {
                        log::error!("error writing audio to file: {}", err);
                    }
                }
            }
        }

        // Finalize writer
        match writer.finalize() {
            Ok(remaining) => {
                if !remaining.is_empty() {
                    let _ = stream_tx.send(remaining.clone()).await;
                    if let Some(ref mut f) = file {
                        let _ = f.write_all(&remaining).await;
                    }
                }
            }
            Err(err) => {
                log::error!("error finalizing audio writer: {}", err);
                return Err(err);
            }
        }

        // Sync file if saving
        if let Some(ref mut f) = file {
            let _ = f.sync_all().await;
        }

        // stream_tx is dropped here, closing the channel
        Ok::<(), TalkError>(())
    });

    // Spawn transcription task
    let transcribe_task =
        tokio::spawn(async move { transcriber.transcribe_stream(stream_rx, "audio.ogg").await });

    log::info!("recording (batch)... press Ctrl+C to stop and transcribe");

    // Wait for SIGINT
    tokio::signal::ctrl_c()
        .await
        .map_err(|err| TalkError::Audio(format!("Failed to listen for Ctrl+C: {}", err)))?;

    log::info!("stopping recording");

    // Stop capture (closes audio channel → encode task finishes → stream_tx drops → transcription completes)
    capture.stop()?;

    // Wait for encode task
    match encode_task.await {
        Ok(Ok(())) => log::debug!("encode task completed"),
        Ok(Err(err)) => log::error!("encode error: {}", err),
        Err(err) => log::error!("encode task panicked: {}", err),
    }

    // Wait for transcription result
    let text = match transcribe_task.await {
        Ok(Ok(text)) => text,
        Ok(Err(err)) => {
            return Err(err);
        }
        Err(err) => {
            return Err(TalkError::Transcription(format!(
                "Transcription task panicked: {}",
                err
            )));
        }
    };

    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_no_args() {
        let result = parse_args(&[]).expect("should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_args_with_file() {
        let args = vec!["output.ogg".to_string()];
        let result = parse_args(&args).expect("should succeed");
        assert_eq!(result, Some(PathBuf::from("output.ogg")));
    }

    #[test]
    fn test_parse_args_too_many() {
        let args = vec!["a.ogg".to_string(), "b.ogg".to_string()];
        let result = parse_args(&args);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dictate_pipeline_with_mocks() {
        use crate::core::audio::mock::MockAudioCapture;
        use crate::core::audio::{AudioCapture, AudioWriter, OggOpusWriter};
        use crate::core::clipboard::MockClipboard;
        use crate::core::config::AudioConfig;
        use crate::core::transcription::{BatchTranscriber, MockBatchTranscriber};

        let audio_config = AudioConfig {
            sample_rate: 16_000,
            channels: 1,
            bitrate: 32_000,
        };

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
        let text = transcribe_task
            .await
            .expect("transcribe task")
            .expect("transcription");
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

    #[test]
    fn test_flush_sentences_single_sentence() {
        let mut buf = "Hello world.".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["Hello world."]);
        assert_eq!(buf, "");
    }

    #[test]
    fn test_flush_sentences_trailing_partial() {
        let mut buf = "First sentence. And then".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["First sentence."]);
        assert_eq!(buf, "And then");
    }

    #[test]
    fn test_flush_sentences_multiple() {
        let mut buf = "One. Two! Three? Rest".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["One.", "Two!", "Three?"]);
        assert_eq!(buf, "Rest");
    }

    #[test]
    fn test_flush_sentences_no_punctuation() {
        let mut buf = "no punctuation here".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert!(segs.is_empty());
        assert_eq!(buf, "no punctuation here");
    }

    #[test]
    fn test_flush_sentences_chinese_punctuation() {
        let mut buf = "你好。世界".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["你好。"]);
        assert_eq!(buf, "世界");
    }

    #[test]
    fn test_flush_sentences_period_at_end() {
        let mut buf = "End of text.".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["End of text."]);
        assert_eq!(buf, "");
    }
}
