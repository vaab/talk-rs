//! Dictate command implementation.
//!
//! Records audio, streams it to the transcription API, and pastes the result
//! into the focused application via clipboard.

use crate::core::audio::cpal_capture::CpalCapture;
use crate::core::audio::{AudioCapture, AudioEncoder, OpusEncoder};
use crate::core::clipboard::{Clipboard, X11Clipboard};
use crate::core::config::Config;
use crate::core::error::TalkError;
use crate::core::transcription::{MistralTranscriber, Transcriber};
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
pub async fn dictate(args: Vec<String>) -> Result<(), TalkError> {
    let save_path = parse_args(&args)?;

    // Load configuration
    let config = Config::load(None)?;

    // Capture active window before recording starts
    let target_window = get_active_window().await;
    if let Some(ref wid) = target_window {
        eprintln!("Captured active window: {}", wid);
    }

    // Initialize audio capture
    let mut capture = CpalCapture::new(config.audio.clone());
    let audio_rx = capture.start()?;

    // Initialize audio encoder
    let mut encoder = OpusEncoder::new(config.audio.clone())?;

    // Create channel for streaming encoded audio to transcriber
    let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(25);

    // Optionally create output file for saving audio
    let save_file = if let Some(ref path) = save_path {
        Some(tokio::fs::File::create(path).await.map_err(TalkError::Io)?)
    } else {
        None
    };

    // Spawn encode task: PCM → Opus → stream_tx (+ optional file)
    let encode_task = tokio::spawn(async move {
        let mut rx = audio_rx;
        let mut file = save_file;

        while let Some(pcm_chunk) = rx.recv().await {
            match encoder.encode(&pcm_chunk) {
                Ok(encoded_data) => {
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
                Err(err) => {
                    eprintln!("Error encoding audio: {}", err);
                    return Err(err);
                }
            }
        }

        // Flush encoder
        match encoder.flush() {
            Ok(remaining) => {
                if !remaining.is_empty() {
                    let _ = stream_tx.send(remaining.clone()).await;
                    if let Some(ref mut f) = file {
                        let _ = f.write_all(&remaining).await;
                    }
                }
            }
            Err(err) => {
                eprintln!("Error flushing encoder: {}", err);
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
    let transcriber = MistralTranscriber::new(config.providers.mistral);
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

    Ok(())
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
        use crate::core::audio::{AudioCapture, AudioEncoder, OpusEncoder};
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

        // Initialize encoder
        let mut encoder = OpusEncoder::new(audio_config).expect("create encoder");

        // Create stream channel
        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(25);

        // Spawn encode task (process a few chunks then stop)
        let encode_task = tokio::spawn(async move {
            let mut rx = audio_rx;
            let mut count = 0;
            while let Some(pcm_chunk) = rx.recv().await {
                if let Ok(encoded) = encoder.encode(&pcm_chunk) {
                    if !encoded.is_empty() && stream_tx.send(encoded).await.is_err() {
                        break;
                    }
                }
                count += 1;
                if count >= 3 {
                    break;
                }
            }
            // Flush
            if let Ok(remaining) = encoder.flush() {
                if !remaining.is_empty() {
                    let _ = stream_tx.send(remaining).await;
                }
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
}
