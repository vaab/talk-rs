//! Dictate command implementation.
//!
//! Records audio, streams it to the transcription API, and pastes the result
//! into the focused application via clipboard.

use crate::core::audio::cpal_capture::CpalCapture;
use crate::core::audio::indicator::SoundPlayer;
use crate::core::audio::{AudioCapture, AudioWriter, OggOpusWriter};
use crate::core::clipboard::{Clipboard, X11Clipboard};
use crate::core::config::{AudioConfig, Config};
use crate::core::daemon::{self, DaemonStatus};
use crate::core::error::TalkError;
use crate::core::overlay::{IndicatorKind, OverlayHandle};
use crate::core::transcription::{
    MistralRealtimeTranscriber, MistralTranscriber, Transcriber, TranscriptionEvent,
};
use std::os::unix::process::CommandExt as _;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

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
pub async fn dictate(
    args: Vec<String>,
    batch: bool,
    toggle: bool,
    no_sounds: bool,
    no_overlay: bool,
    daemon: bool,
    target_window_arg: Option<String>,
) -> Result<(), TalkError> {
    // Toggle mode: start or stop a daemon
    if toggle {
        return toggle_dispatch(batch, no_sounds, no_overlay).await;
    }

    let save_path = parse_args(&args)?;

    // Load configuration
    let config = Config::load(None)?;

    // Determine target window: use --target-window arg (from daemon mode)
    // or capture the currently active window.
    let target_window = if let Some(wid) = target_window_arg {
        Some(wid)
    } else if !daemon {
        let wid = get_active_window().await;
        if let Some(ref w) = wid {
            eprintln!("Captured active window: {}", w);
        }
        wid
    } else {
        None
    };

    // Initialize sound player (single-channel with preemption)
    let player = if no_sounds {
        None
    } else {
        match SoundPlayer::new() {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("Warning: sound indicators unavailable: {}", e);
                None
            }
        }
    };

    // Initialize overlay (visual indicator on X11)
    let overlay = if no_overlay {
        None
    } else {
        match OverlayHandle::new() {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("Warning: visual overlay unavailable: {}", e);
                None
            }
        }
    };

    // Play start sound
    if let Some(ref p) = player {
        p.play_start().await;
    }

    // Show recording indicator
    if let Some(ref o) = overlay {
        o.show(IndicatorKind::Recording);
    }

    // Start boop loop (every 5 seconds)
    let boop_token = player
        .as_ref()
        .map(|p| p.start_boop_loop(std::time::Duration::from_secs(5)));

    let text = if batch {
        // Batch mode: capture audio, encode, then transcribe
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
            config.providers.mistral,
        )
        .await?
    } else {
        // Realtime mode (default): stream audio over WebSocket
        if save_path.is_some() {
            eprintln!(
                "Warning: audio file saving is not supported in realtime mode, ignoring file path"
            );
        }
        dictate_realtime(config).await?
    };

    // Stop boop loop and play stop sound (preempts any in-progress boop)
    if let Some(token) = boop_token {
        token.cancel();
    }

    // Hide recording indicator
    if let Some(ref o) = overlay {
        o.hide();
    }

    if let Some(ref p) = player {
        p.play_stop().await;
    }

    let text = text.trim().to_string();

    if text.is_empty() {
        eprintln!("Empty transcription — nothing to paste");
        return Ok(());
    }

    eprintln!("Transcription: {}", text);

    // Paste into focused application
    let clipboard = X11Clipboard::new();

    // Refocus target window
    if let Some(ref wid) = target_window {
        if !focus_window(wid).await {
            eprintln!("Warning: could not refocus target window");
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // Save current clipboard
    let saved_clipboard = clipboard.get_text().await.ok();

    // Set clipboard to transcription
    clipboard.set_text(&text).await?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Simulate paste
    simulate_paste().await?;

    // Restore clipboard after paste
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    if let Some(saved) = saved_clipboard {
        let _ = clipboard.set_text(&saved).await;
    }

    if let Some(ref path) = save_path {
        eprintln!("Audio saved to: {}", path.display());
    }

    // Print transcription to stdout
    println!("{}", text);

    // If running as daemon, clean up PID file on normal exit
    if daemon {
        if let Ok(pid_file) = daemon::pid_path() {
            let _ = daemon::remove_pid_file(&pid_file);
        }
    }

    Ok(())
}

/// Toggle dispatch: start a new daemon or stop a running one.
async fn toggle_dispatch(batch: bool, no_sounds: bool, no_overlay: bool) -> Result<(), TalkError> {
    let pid_file = daemon::pid_path()?;

    // Acquire exclusive lock to prevent race between concurrent toggle calls
    let _lock = daemon::acquire_lock()?;

    match daemon::check_status(&pid_file)? {
        DaemonStatus::NotRunning => toggle_start(&pid_file, batch, no_sounds, no_overlay).await,
        DaemonStatus::Running { pid } => toggle_stop(pid, &pid_file),
    }
}

/// Start a new daemon: capture window, spawn detached dictate process, write PID.
async fn toggle_start(
    pid_file: &std::path::Path,
    batch: bool,
    no_sounds: bool,
    no_overlay: bool,
) -> Result<(), TalkError> {
    // Capture active window before spawning daemon
    let target_window = get_active_window().await;

    // Find our own executable
    let exe = std::env::current_exe()
        .map_err(|e| TalkError::Config(format!("failed to determine current executable: {}", e)))?;

    // Build daemon command: talk-rs dictate --daemon [--batch] [--no-sounds] [--target-window=WID]
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("dictate").arg("--daemon");

    if batch {
        cmd.arg("--batch");
    }

    if no_sounds {
        cmd.arg("--no-sounds");
    }

    if no_overlay {
        cmd.arg("--no-overlay");
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

    eprintln!("Dictation started (PID {})", child_pid);

    Ok(())
}

/// Stop a running daemon: SIGINT, wait, SIGTERM fallback, clean up PID file.
fn toggle_stop(pid: u32, pid_file: &std::path::Path) -> Result<(), TalkError> {
    eprintln!("Stopping dictation (PID {})...", pid);
    daemon::stop_daemon(pid, pid_file)?;
    eprintln!("Dictation stopped");
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
async fn dictate_realtime(config: Config) -> Result<String, TalkError> {
    let mut capture = CpalCapture::new(config.audio.clone());
    let audio_rx = capture.start()?;

    let transcriber = MistralRealtimeTranscriber::new(config.providers.mistral);
    let mut event_rx = transcriber.transcribe_realtime(audio_rx).await?;

    eprintln!("Recording (realtime)... Press Ctrl+C to stop.");

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
            eprintln!("\nStopping recording...");
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

                        // Flush completed sentences from the buffer.
                        // Split on sentence-ending punctuation followed by
                        // whitespace or end-of-string.
                        flush_sentences(&mut current_line, &mut segments);
                    }
                    Some(TranscriptionEvent::SegmentDelta { text, .. }) => {
                        // If the API sends segment events, use them as
                        // authoritative sentence boundaries.
                        let segment_text = text.trim().to_string();
                        if !segment_text.is_empty() {
                            println!("{}", segment_text);
                            segments.push(segment_text);
                        }
                        let blank = " ".repeat(current_line.len());
                        eprint!("\r{}\r", blank);
                        current_line.clear();
                    }
                    Some(TranscriptionEvent::Done) => {
                        // Flush any trailing text that didn't end with punctuation
                        let trailing = current_line.trim().to_string();
                        if !trailing.is_empty() {
                            println!("{}", trailing);
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
                        eprintln!("Session created");
                    }
                    Some(TranscriptionEvent::Language { language }) => {
                        eprintln!("Detected language: {}", language);
                    }
                    Some(TranscriptionEvent::Unknown { .. }) => {
                        // Ignore unknown/future event types
                    }
                    None => {
                        // Channel closed without Done event
                        let trailing = current_line.trim().to_string();
                        if !trailing.is_empty() {
                            println!("{}", trailing);
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

    Ok(segments.join(" "))
}

/// Batch dictation mode.
///
/// Encodes audio and streams it directly to a single transcription request.
async fn dictate_streaming(
    capture: &mut CpalCapture,
    audio_config: AudioConfig,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    save_file: Option<tokio::fs::File>,
    mistral_config: crate::core::config::MistralConfig,
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
                eprintln!("Error creating writer: {}", err);
                return Err(err);
            }
        };

        let header = match writer.header() {
            Ok(bytes) => bytes,
            Err(err) => {
                eprintln!("Error creating header: {}", err);
                return Err(err);
            }
        };
        if stream_tx.send(header.clone()).await.is_err() {
            eprintln!("Transcription stream closed");
            return Ok::<(), TalkError>(());
        }
        if let Some(ref mut f) = file {
            if let Err(err) = f.write_all(&header).await {
                eprintln!("Error writing to file: {}", err);
            }
        }

        while let Some(pcm_chunk) = rx.recv().await {
            let encoded_data = match writer.write_pcm(&pcm_chunk) {
                Ok(data) => data,
                Err(err) => {
                    eprintln!("Error encoding audio: {}", err);
                    return Err(err);
                }
            };
            if !encoded_data.is_empty() {
                // Send to transcription stream
                if stream_tx.send(encoded_data.clone()).await.is_err() {
                    eprintln!("Transcription stream closed");
                    break;
                }
                // Optionally write to file
                if let Some(ref mut f) = file {
                    if let Err(err) = f.write_all(&encoded_data).await {
                        eprintln!("Error writing to file: {}", err);
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
                eprintln!("Error finalizing writer: {}", err);
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
    let transcriber = MistralTranscriber::new(mistral_config);
    let transcribe_task =
        tokio::spawn(async move { transcriber.transcribe_stream(stream_rx, "audio.ogg").await });

    eprintln!("Recording... Press Ctrl+C to stop and transcribe.");

    // Wait for SIGINT
    tokio::signal::ctrl_c()
        .await
        .map_err(|err| TalkError::Audio(format!("Failed to listen for Ctrl+C: {}", err)))?;

    eprintln!("Stopping recording...");

    // Stop capture (closes audio channel → encode task finishes → stream_tx drops → transcription completes)
    capture.stop()?;

    // Wait for encode task
    match encode_task.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => eprintln!("Encode error: {}", err),
        Err(err) => eprintln!("Encode task panicked: {}", err),
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
        use crate::core::transcription::{MockTranscriber, Transcriber};

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
        let transcriber = MockTranscriber::new("Hello world from dictation");
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
