//! Dictate command implementation.
//!
//! Records audio, streams it to the transcription API, and pastes the result
//! into the focused application via clipboard.

use crate::core::audio::cpal_capture::CpalCapture;
use crate::core::audio::{AudioCapture, AudioWriter, OggOpusWriter};
use crate::core::clipboard::{Clipboard, X11Clipboard};
use crate::core::config::{AudioConfig, Config};
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
pub async fn dictate(
    args: Vec<String>,
    chunked: bool,
    chunk_seconds_override: Option<u64>,
) -> Result<(), TalkError> {
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

    // Optionally create output file for saving audio
    let save_file = if let Some(ref path) = save_path {
        Some(tokio::fs::File::create(path).await.map_err(TalkError::Io)?)
    } else {
        None
    };

    let text = if chunked {
        let chunk_seconds = chunk_seconds_override
            .or_else(|| config.dictate.as_ref().map(|d| d.chunk_seconds))
            .ok_or_else(|| {
                TalkError::Config(
                    "chunk_seconds not specified: use -n flag or set dictate.chunk_seconds in config"
                        .to_string(),
                )
            })?;

        dictate_chunked(
            &mut capture,
            config.audio.clone(),
            audio_rx,
            save_file,
            chunk_seconds,
            config.providers.mistral,
        )
        .await?
    } else {
        dictate_streaming(
            &mut capture,
            config.audio.clone(),
            audio_rx,
            save_file,
            config.providers.mistral,
        )
        .await?
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

/// Streaming (non-chunked) dictation mode.
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

/// Chunked dictation mode.
///
/// Encodes audio into a shared buffer, periodically drains it into chunks,
/// and transcribes each chunk separately. Results are accumulated and joined.
async fn dictate_chunked(
    capture: &mut CpalCapture,
    audio_config: AudioConfig,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    save_file: Option<tokio::fs::File>,
    chunk_seconds: u64,
    mistral_config: crate::core::config::MistralConfig,
) -> Result<String, TalkError> {
    let chunk_duration = std::time::Duration::from_secs(chunk_seconds);
    let accumulated_text = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
    let accumulated_text_clone = accumulated_text.clone();

    // Shared buffer for raw PCM samples
    let audio_buffer = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<i16>::new()));
    let audio_buffer_encode = audio_buffer.clone();
    let audio_buffer_chunk = audio_buffer.clone();

    // Encode task: PCM → buffer (+ optional file writer)
    let audio_config_encode = audio_config.clone();
    let encode_task = tokio::spawn(async move {
        let mut rx = audio_rx;
        let mut file_writer = if let Some(file) = save_file {
            let mut writer = match OggOpusWriter::new(audio_config_encode.clone()) {
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
            let mut file = file;
            if let Err(err) = file.write_all(&header).await {
                eprintln!("Error writing to file: {}", err);
            }
            Some((file, writer))
        } else {
            None
        };

        while let Some(pcm_chunk) = rx.recv().await {
            audio_buffer_encode
                .lock()
                .await
                .extend_from_slice(&pcm_chunk);
            if let Some((ref mut f, ref mut writer)) = file_writer {
                let encoded_data = match writer.write_pcm(&pcm_chunk) {
                    Ok(data) => data,
                    Err(err) => {
                        eprintln!("Error encoding audio: {}", err);
                        return Err(err);
                    }
                };
                if !encoded_data.is_empty() {
                    if let Err(err) = f.write_all(&encoded_data).await {
                        eprintln!("Error writing to file: {}", err);
                    }
                }
            }
        }

        // Finalize writer for file if saving
        if let Some((ref mut f, ref mut writer)) = file_writer {
            match writer.finalize() {
                Ok(remaining) => {
                    if !remaining.is_empty() {
                        let _ = f.write_all(&remaining).await;
                    }
                }
                Err(err) => {
                    eprintln!("Error finalizing writer: {}", err);
                    return Err(err);
                }
            }
            let _ = f.sync_all().await;
        }

        Ok::<(), TalkError>(())
    });

    // Chunk timer task: periodically drain buffer and transcribe
    let config_mistral = mistral_config.clone();
    let audio_config_chunk = audio_config.clone();
    let chunk_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(chunk_duration);
        interval.tick().await; // First tick is immediate, skip it

        loop {
            interval.tick().await;

            // Drain the buffer
            let pcm_samples: Vec<i16> = {
                let mut buf = audio_buffer_chunk.lock().await;
                std::mem::take(&mut *buf)
            };

            if pcm_samples.is_empty() {
                continue;
            }

            let mut writer = match OggOpusWriter::new(audio_config_chunk.clone()) {
                Ok(writer) => writer,
                Err(err) => {
                    eprintln!("Error creating writer: {}", err);
                    continue;
                }
            };
            let mut ogg_bytes = Vec::new();
            match writer.header() {
                Ok(bytes) => ogg_bytes.extend(bytes),
                Err(err) => {
                    eprintln!("Error creating header: {}", err);
                    continue;
                }
            }
            match writer.write_pcm(&pcm_samples) {
                Ok(bytes) => ogg_bytes.extend(bytes),
                Err(err) => {
                    eprintln!("Error encoding audio: {}", err);
                    continue;
                }
            }
            match writer.finalize() {
                Ok(bytes) => ogg_bytes.extend(bytes),
                Err(err) => {
                    eprintln!("Error finalizing writer: {}", err);
                    continue;
                }
            }

            // Create channel and send full OGG payload
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            let _ = tx.send(ogg_bytes).await;
            drop(tx);

            // Transcribe this chunk
            let transcriber = MistralTranscriber::new(config_mistral.clone());
            match transcriber.transcribe_stream(rx, "audio.ogg").await {
                Ok(text) => {
                    let text = text.trim().to_string();
                    if !text.is_empty() {
                        eprintln!("Chunk transcription: {}", text);
                        accumulated_text_clone.lock().await.push(text);
                    }
                }
                Err(err) => {
                    eprintln!("Chunk transcription error: {}", err);
                }
            }
        }
    });

    eprintln!(
        "Recording (chunked, {}s intervals)... Press Ctrl+C to stop.",
        chunk_seconds
    );

    // Wait for SIGINT
    tokio::signal::ctrl_c()
        .await
        .map_err(|err| TalkError::Audio(format!("Failed to listen for Ctrl+C: {}", err)))?;

    eprintln!("Stopping recording...");

    // Stop capture
    capture.stop()?;

    // Wait for encode task
    match encode_task.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => eprintln!("Encode error: {}", err),
        Err(err) => eprintln!("Encode task panicked: {}", err),
    }

    // Cancel chunk timer
    chunk_task.abort();

    // Process final partial chunk (remaining in buffer)
    let final_pcm: Vec<i16> = {
        let mut buf = audio_buffer.lock().await;
        std::mem::take(&mut *buf)
    };

    if !final_pcm.is_empty() {
        let mut writer = OggOpusWriter::new(audio_config)?;
        let mut ogg_bytes = Vec::new();
        ogg_bytes.extend(writer.header()?);
        ogg_bytes.extend(writer.write_pcm(&final_pcm)?);
        ogg_bytes.extend(writer.finalize()?);

        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let _ = tx.send(ogg_bytes).await;
        drop(tx);

        let transcriber = MistralTranscriber::new(mistral_config);
        match transcriber.transcribe_stream(rx, "audio.ogg").await {
            Ok(text) => {
                let text = text.trim().to_string();
                if !text.is_empty() {
                    eprintln!("Final chunk transcription: {}", text);
                    accumulated_text.lock().await.push(text);
                }
            }
            Err(err) => {
                eprintln!("Final chunk transcription error: {}", err);
            }
        }
    }

    // Combine all chunk results
    let all_text = accumulated_text.lock().await.join(" ");
    Ok(all_text)
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

    #[tokio::test]
    async fn test_chunked_dictate_pipeline_with_mocks() {
        use crate::core::audio::mock::MockAudioCapture;
        use crate::core::audio::{AudioCapture, AudioWriter, OggOpusWriter};
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

        // Shared buffer for raw PCM (like chunked mode)
        let audio_buffer = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<i16>::new()));
        let audio_buffer_encode = audio_buffer.clone();

        // Encode task: PCM → buffer (process a few chunks then stop)
        let encode_task = tokio::spawn(async move {
            let mut rx = audio_rx;
            let mut count = 0;
            while let Some(pcm_chunk) = rx.recv().await {
                audio_buffer_encode
                    .lock()
                    .await
                    .extend_from_slice(&pcm_chunk);
                count += 1;
                if count >= 6 {
                    break;
                }
            }
        });

        // Wait for encode to finish
        encode_task.await.expect("encode task");
        capture.stop().expect("stop capture");

        // Simulate chunked transcription: split buffer into two chunks
        let all_samples: Vec<i16> = {
            let mut buf = audio_buffer.lock().await;
            std::mem::take(&mut *buf)
        };
        assert!(!all_samples.is_empty(), "should have audio data");

        let mid = all_samples.len() / 2;
        let chunk1 = &all_samples[..mid];
        let chunk2 = &all_samples[mid..];

        let mut accumulated_text = Vec::<String>::new();

        // Transcribe chunk 1
        {
            let mut writer = OggOpusWriter::new(audio_config.clone()).expect("writer");
            let mut ogg_bytes = Vec::new();
            ogg_bytes.extend(writer.header().expect("header"));
            ogg_bytes.extend(writer.write_pcm(chunk1).expect("encode"));
            ogg_bytes.extend(writer.finalize().expect("finalize"));
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            tx.send(ogg_bytes).await.expect("send chunk1");
            drop(tx);

            let transcriber = MockTranscriber::new("First chunk");
            let text = transcriber
                .transcribe_stream(rx, "audio.ogg")
                .await
                .expect("transcribe chunk 1");
            let text = text.trim().to_string();
            if !text.is_empty() {
                accumulated_text.push(text);
            }
        }

        // Transcribe chunk 2
        {
            let mut writer = OggOpusWriter::new(audio_config).expect("writer");
            let mut ogg_bytes = Vec::new();
            ogg_bytes.extend(writer.header().expect("header"));
            ogg_bytes.extend(writer.write_pcm(chunk2).expect("encode"));
            ogg_bytes.extend(writer.finalize().expect("finalize"));
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            tx.send(ogg_bytes).await.expect("send chunk2");
            drop(tx);

            let transcriber = MockTranscriber::new("Second chunk");
            let text = transcriber
                .transcribe_stream(rx, "audio.ogg")
                .await
                .expect("transcribe chunk 2");
            let text = text.trim().to_string();
            if !text.is_empty() {
                accumulated_text.push(text);
            }
        }

        // Verify accumulated results
        let combined = accumulated_text.join(" ");
        assert_eq!(combined, "First chunk Second chunk");
    }

    #[tokio::test]
    async fn test_chunked_buffer_drain_and_accumulate() {
        // Test that the buffer drain + accumulate pattern works correctly
        let buffer = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<i16>::new()));

        // Simulate encoding: push some data
        {
            let mut buf = buffer.lock().await;
            buf.extend_from_slice(&[1, 2, 3]);
            buf.extend_from_slice(&[4, 5, 6]);
        }

        // Drain (simulating chunk timer)
        let drained: Vec<i16> = {
            let mut buf = buffer.lock().await;
            std::mem::take(&mut *buf)
        };
        assert_eq!(drained.len(), 6);
        assert_eq!(drained, vec![1, 2, 3, 4, 5, 6]);

        // Buffer should be empty after drain
        assert!(buffer.lock().await.is_empty());

        // Simulate more encoding after drain
        {
            let mut buf = buffer.lock().await;
            buf.extend_from_slice(&[7, 8, 9]);
        }

        // Second drain
        let drained2: Vec<i16> = {
            let mut buf = buffer.lock().await;
            std::mem::take(&mut *buf)
        };
        assert_eq!(drained2.len(), 3);
        assert_eq!(drained2, vec![7, 8, 9]);
    }
}
