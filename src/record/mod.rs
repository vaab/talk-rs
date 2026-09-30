//! Record command implementation.
//!
//! Captures audio from the system, encodes it with Opus, and writes to a file.
//! Supports graceful shutdown via SIGINT (Ctrl+C).

pub(crate) mod audio;
#[cfg(feature = "ui")]
mod entries;
#[cfg(feature = "ui")]
pub(crate) use entries::recording_navigation;
#[cfg(feature = "ui")]
pub(crate) mod player;
#[cfg(all(feature = "capture", feature = "ui"))]
pub(crate) mod toggle;
#[cfg(feature = "ui")]
pub(crate) mod ui;
#[cfg(all(feature = "ui", feature = "perf-counters"))]
mod ui_probe;

#[cfg(feature = "capture")]
use crate::audio::bt_profile;
#[cfg(feature = "capture")]
use crate::audio::monitor_capture::MonitorCapture;
#[cfg(feature = "capture")]
use crate::audio::pipewire_capture::PipeWireCapture;
#[cfg(all(feature = "capture", feature = "ui"))]
use crate::audio::recording_feedback::RecordingOverlayOptions;
#[cfg(feature = "capture")]
use crate::audio::recording_feedback::{
    RecordingBadgeTeardown, RecordingFeedback, RecordingFeedbackOptions,
};
#[cfg(feature = "capture")]
use crate::audio::{AudioCapture, AudioWriter, OggOpusWriter, WavWriter};
#[cfg(feature = "capture")]
use crate::config::{AudioConfig, Config};
#[cfg(feature = "capture")]
use crate::error::TalkError;
#[cfg(feature = "capture")]
use chrono::Local;
#[cfg(feature = "capture")]
use std::io::SeekFrom;
#[cfg(feature = "capture")]
use std::path::{Path, PathBuf};
#[cfg(feature = "capture")]
use tokio::io::{AsyncSeekExt, AsyncWriteExt};

/// Generate a default timestamped filename for a recording.
///
/// Uses an ISO 8601 local timestamp with numeric timezone offset, e.g.
/// `2026-04-11T13-15-52+0200.ogg`.  Colons in the time portion are
/// replaced by dashes so the filename is safe on every filesystem.  The
/// format matches the ``memo`` tool so recordings from both tools can
/// coexist in the same directory.
#[cfg(feature = "capture")]
pub fn default_filename() -> String {
    let now = Local::now();
    now.format("%Y-%m-%dT%H-%M-%S%z.ogg").to_string()
}

/// Resolve the output file path from CLI arguments and the configured
/// `output_dir`.
///
/// - No arguments → `<output_dir>/YYYY/MM/YYYY-MM-DDTHH-MM-SS±ZZZZ.ogg`
///   (auto-namespaced by year and month to keep the flat directory from
///   growing unbounded).  The filename is an ISO 8601 local timestamp
///   with numeric timezone offset, matching the `memo` tool's scheme.
/// - One argument → used as-is
/// - More than one → error
///
/// This is a pure computation: it does not create the directory.  The
/// caller is responsible for calling [`std::fs::create_dir_all`] on the
/// parent before opening the file.
#[cfg(feature = "capture")]
fn resolve_output_path(args: &[String], output_dir: &Path) -> Result<PathBuf, TalkError> {
    match args.len() {
        0 => {
            let now = Local::now();
            let year = now.format("%Y").to_string();
            let month = now.format("%m").to_string();
            Ok(output_dir.join(year).join(month).join(default_filename()))
        }
        1 => Ok(PathBuf::from(&args[0])),
        _ => Err(TalkError::Audio(
            "record command takes at most one argument (output file path)".to_string(),
        )),
    }
}

/// Parse command-line arguments for the record command.
///
/// Returns the output file path. If not provided, generates a
/// timestamp-based filename inside the configured `output_dir`.
#[cfg(feature = "capture")]
pub fn parse_args(args: &[String]) -> Result<PathBuf, TalkError> {
    let config = Config::load(None)?;
    resolve_output_path(args, &config.output_dir)
}

#[cfg(feature = "capture")]
fn create_writer(path: &Path, config: AudioConfig) -> Result<Box<dyn AudioWriter>, TalkError> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("wav") => Ok(Box::new(WavWriter::new(config))),
        // Human-facing recordings use Opus `Application::Audio` for
        // fuller fidelity (vs the speech-optimised Voip mode used by
        // the transcription path).
        _ => Ok(Box::new(OggOpusWriter::new_for_recording(config)?)),
    }
}

/// Record audio from the system and write to a file.
///
/// # Arguments
/// * `args` - Command-line arguments (optional output file path)
///
/// # Flow
/// 1. Parse arguments to get output file path
/// 2. Load configuration (audio settings)
/// 3. Initialize CpalCapture and OpusEncoder
/// 4. Spawn async task to read from capture channel, encode, and write to file
/// 5. Wait for SIGINT (Ctrl+C) to gracefully shutdown
/// 6. Flush encoder and close file
#[cfg(feature = "capture")]
pub struct RecordOpts {
    pub args: Vec<String>,
    pub monitor: bool,
    pub no_sounds: bool,
    pub no_boop: bool,
    pub no_overlay: bool,
    pub viz: Option<crate::config::VizMode>,
    pub mono: bool,
    pub no_bt_auto_switch: bool,
}

#[cfg(feature = "capture")]
pub async fn record(opts: RecordOpts) -> Result<(), TalkError> {
    // Register SIGINT before configuration, Bluetooth, capture, writer, or
    // file resources so a quick toggle-off cannot be lost during startup.
    // Foreground Ctrl-C uses this same signal stream.
    let mut interrupt =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .map_err(|error| TalkError::Audio(format!("Failed to listen for Ctrl+C: {error}")))?;

    // Parse arguments
    let output_path = parse_args(&opts.args)?;

    // Bluetooth headset profile auto-switching.  Resolution order:
    // CLI flag `--no-bt-auto-switch` wins; otherwise config
    // `audio.bt_auto_switch` (default `true`).  When enabled we
    // recover any stale profile from a prior unclean termination,
    // then switch the headset to HFP for microphone capture.  The
    // returned guard restores the original profile on Drop (normal
    // return, panic, SIGINT).  Failures are logged but non-fatal.
    let config_for_bt = Config::load(None).ok();
    let bt_auto_switch_enabled = !opts.no_bt_auto_switch
        && config_for_bt
            .as_ref()
            .and_then(|c| c.audio.as_ref())
            .map(|a| a.bt_auto_switch_enabled())
            .unwrap_or(true);
    let mut bt_guard = if !bt_auto_switch_enabled {
        log::debug!("bt_profile: auto-switching disabled by config/flag");
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

    // Resolve the human-facing recording quality (sample rate /
    // channels / bitrate).  This is intentionally NOT `AudioConfig::new()`
    // (the 16 kHz transcription profile): the `record` command produces
    // a file for a human to listen to, so it uses the configurable
    // `recording:` section, defaulting to 48 kHz mono 128 kbps.
    let recording_config = config_for_bt
        .as_ref()
        .and_then(|c| c.recording.clone())
        .unwrap_or_default();
    let audio_config = recording_config.resolved();
    log::info!(
        "recording quality: {} Hz, {} channel(s), {} bps",
        audio_config.sample_rate,
        audio_config.channels,
        audio_config.bitrate
    );

    // Initialize audio capture.  Use PipeWire for both mic-only and
    // mic+monitor: it negotiates the requested rate/channels with the
    // device (duplicating a mono source to stereo when asked), which
    // makes a 48 kHz stereo request robust even on mono-only mics —
    // unlike the strict exact-match CpalCapture used previously.
    let mut capture: Box<dyn AudioCapture> = if opts.monitor {
        log::info!("recording with mic+monitor (PipeWire)");
        Box::new(MonitorCapture::new(audio_config.clone()))
    } else {
        log::info!("recording with mic (PipeWire)");
        Box::new(PipeWireCapture::new(audio_config.clone()))
    };

    let viz_mode = opts.viz.or_else(|| {
        config_for_bt
            .as_ref()
            .and_then(|config| config.indicators.as_ref())
            .and_then(|indicators| indicators.viz)
    });
    let boop_interval_ms = config_for_bt
        .as_ref()
        .and_then(|config| config.indicators.as_ref())
        .map(|indicators| indicators.boop_interval_ms)
        .unwrap_or(5_000);
    let mut feedback = RecordingFeedback::new(RecordingFeedbackOptions {
        no_sounds: opts.no_sounds,
        no_boop: opts.no_boop,
        no_overlay: opts.no_overlay,
        viz: viz_mode,
        mono: config_for_bt
            .as_ref()
            .map_or(opts.mono, |config| config.resolved_mono(opts.mono)),
        boop_interval_ms,
        capture_rate: audio_config.sample_rate,
        pause_audio: false,
        suppress_boop: None,
        #[cfg(feature = "ui")]
        overlay: RecordingOverlayOptions {
            silence_tx: None,
            auto_pause: true,
            telemetry_rx: None,
        },
    });
    record_audio_to_file(
        &output_path,
        audio_config,
        &mut *capture,
        &mut feedback,
        async { interrupt.recv().await },
        move || bt_guard.restore_now_async(),
    )
    .await?;

    println!("Recording saved to: {}", output_path.display());
    Ok(())
}

#[cfg(feature = "capture")]
async fn record_audio_to_file<Stop, OnStop>(
    output_path: &Path,
    audio_config: AudioConfig,
    capture: &mut dyn AudioCapture,
    feedback: &mut RecordingFeedback,
    stop: Stop,
    on_stop: OnStop,
) -> Result<(), TalkError>
where
    Stop: std::future::Future<Output = Option<()>>,
    OnStop: FnOnce(),
{
    let mut rx = start_capture_with_feedback(feedback, capture).await?;

    let setup = async {
        let writer = create_writer(output_path, audio_config)?;
        let is_wav = matches!(
            output_path.extension().and_then(|e| e.to_str()),
            Some("wav")
        );

        // Auto-generated paths create YYYY/MM; explicit paths may be nested.
        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await.map_err(|err| {
                    TalkError::Io(std::io::Error::new(
                        err.kind(),
                        format!(
                            "failed to create recording directory {}: {}",
                            parent.display(),
                            err
                        ),
                    ))
                })?;
            }
        }
        let file = tokio::fs::File::create(output_path)
            .await
            .map_err(TalkError::Io)?;
        Ok::<_, TalkError>((writer, is_wav, file))
    }
    .await;
    let (mut writer, is_wav, mut file) = match setup {
        Ok(resources) => resources,
        Err(error) => {
            feedback.teardown_recording(RecordingBadgeTeardown::Hide);
            if let Err(stop_error) = capture.stop() {
                log::warn!("failed to stop capture after recording setup error: {stop_error}");
            }
            return Err(error);
        }
    };

    // Spawn async task to read from capture channel, encode, and write to file
    let encode_task = tokio::spawn(async move {
        let header = match writer.header() {
            Ok(bytes) => bytes,
            Err(err) => {
                log::error!("error creating header: {}", err);
                return Err(err);
            }
        };
        if let Err(err) = file.write_all(&header).await {
            log::error!("error writing header: {}", err);
            return Err(TalkError::Io(err));
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
                if let Err(err) = file.write_all(&encoded_data).await {
                    log::error!("error writing to file: {}", err);
                    return Err(TalkError::Io(err));
                }
            }
        }

        // Finalize writer to write remaining data
        match writer.finalize() {
            Ok(remaining_data) => {
                if !remaining_data.is_empty() {
                    if is_wav {
                        if let Err(err) = file.seek(SeekFrom::Start(0)).await {
                            log::error!("error seeking to start: {}", err);
                            return Err(TalkError::Io(err));
                        }
                    }
                    if let Err(err) = file.write_all(&remaining_data).await {
                        log::error!("error writing flushed data: {}", err);
                        return Err(TalkError::Io(err));
                    }
                }
            }
            Err(err) => {
                log::error!("error finalizing writer: {}", err);
                return Err(err);
            }
        }

        // Close file
        if let Err(err) = file.sync_all().await {
            log::error!("error syncing file: {}", err);
            return Err(TalkError::Io(err));
        }

        Ok::<(), TalkError>(())
    });

    // Wait for SIGINT (Ctrl+C or toggle-off).
    if stop.await.is_none() {
        feedback.teardown_recording(RecordingBadgeTeardown::Hide);
        capture.stop()?;
        on_stop();
        match encode_task.await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => log::warn!("recording finalization after closed listener: {error}"),
            Err(error) => log::warn!("recording task after closed listener: {error}"),
        }
        return Err(TalkError::Audio(
            "SIGINT listener closed before recording stopped".to_string(),
        ));
    }

    println!("Stopping recording...");

    stop_capture_with_feedback(feedback, capture, || async move {
        on_stop();
        match encode_task.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(err),
            Err(err) => Err(TalkError::Audio(format!("Encode task panicked: {}", err))),
        }
    })
    .await?;

    Ok(())
}

#[cfg(feature = "capture")]
async fn start_capture_with_feedback(
    feedback: &mut RecordingFeedback,
    capture: &mut dyn AudioCapture,
) -> Result<tokio::sync::mpsc::Receiver<Vec<i16>>, TalkError> {
    // Badge first (instant visual acknowledgement), then the awaited tone,
    // then capture so the tone never enters the recording.
    feedback.prepare_recording();
    feedback.play_start().await;
    let raw_audio = capture.start()?;
    feedback.start_boop();
    Ok(feedback.route_audio(raw_audio))
}

#[cfg(feature = "capture")]
async fn stop_capture_with_feedback<Finalize, FinalizeFuture>(
    feedback: &mut RecordingFeedback,
    capture: &mut dyn AudioCapture,
    finalize: Finalize,
) -> Result<(), TalkError>
where
    Finalize: FnOnce() -> FinalizeFuture,
    FinalizeFuture: std::future::Future<Output = Result<(), TalkError>>,
{
    feedback.teardown_recording(RecordingBadgeTeardown::Hide);
    capture.stop()?;
    finalize().await?;
    feedback.play_stop().await;
    Ok(())
}

#[cfg(feature = "capture")]
pub async fn record_daemon(opts: RecordOpts) -> Result<(), TalkError> {
    let slot = crate::daemon::record_slot()?;
    let _owner = slot.owner_guard();
    record(opts).await
}

#[cfg(all(test, feature = "capture"))]
mod tests {
    use super::*;
    use chrono::Datelike;
    use std::sync::{Arc, Mutex};

    struct FiniteCapture {
        samples: Vec<i16>,
        stopped: Arc<std::sync::atomic::AtomicBool>,
    }

    impl AudioCapture for FiniteCapture {
        fn start(&mut self) -> Result<tokio::sync::mpsc::Receiver<Vec<i16>>, TalkError> {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            tx.try_send(self.samples.clone())
                .map_err(|error| TalkError::Audio(error.to_string()))?;
            Ok(rx)
        }

        fn stop(&mut self) -> Result<(), TalkError> {
            self.stopped
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn recording_runner_writes_decodable_audio_and_stops_capture_before_returning() {
        for extension in ["wav", "ogg"] {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let path = dir.path().join("nested").join(format!("voice.{extension}"));
            let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let mut capture = FiniteCapture {
                samples: (0..640)
                    .map(|index| if index % 2 == 0 { 2000 } else { -2000 })
                    .collect(),
                stopped: Arc::clone(&stopped),
            };
            let events = Arc::new(Mutex::new(Vec::new()));
            let mut feedback = RecordingFeedback::new_for_test(events, true);

            record_audio_to_file(
                &path,
                AudioConfig::new(),
                &mut capture,
                &mut feedback,
                async { Some(()) },
                || {},
            )
            .await
            .expect("recording should be finalized");

            let bytes = std::fs::read(&path).expect("read recording");
            assert!(stopped.load(std::sync::atomic::Ordering::SeqCst));
            let decoded = audio::read_audio_as_i16(&path).expect("recording must decode");
            assert!(
                decoded.len() >= 640,
                "recording must retain the audio frames"
            );
            assert!(
                decoded.iter().any(|sample| *sample != 0),
                "recording must not be silent"
            );
            match extension {
                "wav" => {
                    assert_eq!(&bytes[..4], b"RIFF");
                    assert_eq!(&bytes[8..12], b"WAVE");
                    assert_eq!(bytes.len(), 44 + 640 * 2);
                }
                "ogg" => {
                    assert!(bytes.starts_with(b"OggS"));
                    assert!(bytes.windows(8).any(|window| window == b"OpusHead"));
                }
                _ => unreachable!(),
            }
        }
    }

    #[tokio::test]
    async fn recording_directory_failure_stops_started_capture() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let occupied_parent = dir.path().join("not-a-directory");
        std::fs::write(&occupied_parent, b"user file").expect("create conflicting file");
        let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut capture = FiniteCapture {
            samples: vec![2000; 320],
            stopped: Arc::clone(&stopped),
        };
        let mut feedback = RecordingFeedback::new_for_test(Arc::new(Mutex::new(Vec::new())), true);
        let destination = occupied_parent.join("recording.wav");

        let error = record_audio_to_file(
            &destination,
            AudioConfig::new(),
            &mut capture,
            &mut feedback,
            async { Some(()) },
            || {},
        )
        .await
        .expect_err("parent file prevents recording");

        assert!(error
            .to_string()
            .contains("failed to create recording directory"));
        assert!(stopped.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(
            std::fs::read(&occupied_parent).expect("parent file preserved"),
            b"user file"
        );
    }

    #[tokio::test]
    async fn closed_stop_listener_stops_capture_and_reports_error() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut capture = FiniteCapture {
            samples: vec![2000; 320],
            stopped: Arc::clone(&stopped),
        };
        let mut feedback = RecordingFeedback::new_for_test(Arc::new(Mutex::new(Vec::new())), true);
        let error = record_audio_to_file(
            &dir.path().join("recording.wav"),
            AudioConfig::new(),
            &mut capture,
            &mut feedback,
            async { None },
            || {},
        )
        .await
        .expect_err("closed listener cannot report a successful recording");
        assert_eq!(
            error.to_string(),
            "Audio error: SIGINT listener closed before recording stopped"
        );
        assert!(stopped.load(std::sync::atomic::Ordering::SeqCst));
    }

    struct OrderedCapture {
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl AudioCapture for OrderedCapture {
        fn start(&mut self) -> Result<tokio::sync::mpsc::Receiver<Vec<i16>>, TalkError> {
            if let Ok(mut events) = self.events.lock() {
                events.push("capture-start");
            }
            let (_tx, rx) = tokio::sync::mpsc::channel(1);
            Ok(rx)
        }

        fn stop(&mut self) -> Result<(), TalkError> {
            if let Ok(mut events) = self.events.lock() {
                events.push("capture-stop");
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn record_feedback_wraps_capture_and_durable_finalization_in_order() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut feedback = crate::audio::recording_feedback::RecordingFeedback::new_for_test(
            Arc::clone(&events),
            true,
        );
        let mut capture = OrderedCapture {
            events: Arc::clone(&events),
        };

        let _rx = start_capture_with_feedback(&mut feedback, &mut capture)
            .await
            .expect("start should succeed");
        stop_capture_with_feedback(&mut feedback, &mut capture, || async {
            if let Ok(mut events) = events.lock() {
                events.push("encoder-finalize");
                events.push("file-sync-all");
            }
            Ok(())
        })
        .await
        .expect("stop should succeed");

        assert_eq!(
            events
                .lock()
                .map(|events| events.clone())
                .unwrap_or_default(),
            vec![
                "badge-show",
                "start-tone",
                "capture-start",
                "boop-start",
                "boop-cancel",
                "badge-hide",
                "capture-stop",
                "encoder-finalize",
                "file-sync-all",
                "stop-tone",
            ]
        );
    }

    #[test]
    fn test_resolve_output_path_no_args_nests_by_year_and_month() {
        let output_dir = PathBuf::from("/tmp/test-output");
        let args: Vec<String> = vec![];
        let result = resolve_output_path(&args, &output_dir).expect("resolve should succeed");

        // Path must live inside output_dir/YYYY/MM/.
        let month_dir = result.parent().expect("should have month parent");
        let year_dir = month_dir.parent().expect("should have year parent");
        let root = year_dir.parent().expect("should have root parent");

        assert_eq!(
            root, output_dir,
            "root above the YYYY/MM subdirs should be output_dir"
        );

        let year_name = year_dir
            .file_name()
            .expect("year dir name")
            .to_string_lossy();
        let month_name = month_dir
            .file_name()
            .expect("month dir name")
            .to_string_lossy();

        assert_eq!(year_name.len(), 4, "year segment should be 4 digits");
        assert!(
            year_name.chars().all(|c| c.is_ascii_digit()),
            "year segment should be all digits, got: {}",
            year_name
        );
        assert_eq!(month_name.len(), 2, "month segment should be 2 digits");
        assert!(
            month_name.chars().all(|c| c.is_ascii_digit()),
            "month segment should be all digits, got: {}",
            month_name
        );

        // Filename must follow the ISO 8601 local timestamp pattern
        // with numeric timezone, e.g. ``2026-04-11T13-15-52+0200.ogg``,
        // matching the ``memo`` tool's scheme.
        let filename = result
            .file_name()
            .expect("should have filename")
            .to_string_lossy();
        assert!(filename.ends_with(".ogg"), "filename should end with .ogg");

        // The stem must parse back as a ``chrono`` local timestamp with
        // a numeric timezone offset.  This is the real specification:
        // whatever we produce must round-trip through the same format.
        let stem = filename
            .strip_suffix(".ogg")
            .expect("filename should end with .ogg");
        let parsed = chrono::DateTime::parse_from_str(stem, "%Y-%m-%dT%H-%M-%S%z")
            .unwrap_or_else(|e| panic!("filename stem {} should parse as timestamp: {}", stem, e));

        // Sanity: the year/month in the directory path must match the
        // year/month embedded in the parsed timestamp.
        assert_eq!(
            format!("{:04}", parsed.year()),
            year_name,
            "parsed year from filename {} should equal parent year dir {}",
            filename,
            year_name
        );
        assert_eq!(
            format!("{:02}", parsed.month()),
            month_name,
            "parsed month from filename {} should equal parent month dir {}",
            filename,
            month_name
        );
    }

    #[test]
    fn test_resolve_output_path_with_filename() {
        let output_dir = PathBuf::from("/tmp/test-output");
        let args = vec!["my-recording.ogg".to_string()];
        let result = resolve_output_path(&args, &output_dir).expect("resolve should succeed");

        assert_eq!(result, PathBuf::from("my-recording.ogg"));
    }

    #[test]
    fn test_resolve_output_path_with_absolute_path() {
        let output_dir = PathBuf::from("/tmp/test-output");
        let args = vec!["/tmp/my-recording.ogg".to_string()];
        let result = resolve_output_path(&args, &output_dir).expect("resolve should succeed");

        assert_eq!(result, PathBuf::from("/tmp/my-recording.ogg"));
    }

    #[test]
    fn test_resolve_output_path_too_many_args() {
        let output_dir = PathBuf::from("/tmp/test-output");
        let args = vec!["file1.ogg".to_string(), "file2.ogg".to_string()];
        let result = resolve_output_path(&args, &output_dir);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at most one argument"));
    }

    #[tokio::test]
    async fn test_record_pipeline_with_mock_capture() {
        use crate::audio::{mock::MockAudioCapture, AudioCapture, AudioWriter, OggOpusWriter};
        use crate::config::AudioConfig;
        use std::fs;
        use tempfile::TempDir;

        // Create temporary directory for test output
        let temp_dir = TempDir::new().expect("create temp dir");
        let output_path = temp_dir.path().join("test-recording.ogg");

        // Use test audio config instead of loading from file
        let audio_config = AudioConfig::new();

        // Initialize mock capture
        let mut capture =
            MockAudioCapture::new(audio_config.sample_rate, audio_config.channels, 440.0);
        let mut rx = capture.start().expect("start capture");

        // Initialize writer
        let mut writer = OggOpusWriter::new(audio_config).expect("create writer");

        // Create output file
        let mut file = tokio::fs::File::create(&output_path)
            .await
            .expect("create file");

        // Write header
        let header = writer.header().expect("header");
        file.write_all(&header).await.expect("write header");

        // Simulate encoding a few chunks
        for _ in 0..3 {
            if let Some(pcm_chunk) = rx.recv().await {
                let encoded_data = writer.write_pcm(&pcm_chunk).expect("encode");
                if !encoded_data.is_empty() {
                    file.write_all(&encoded_data).await.expect("write to file");
                }
            }
        }

        // Finalize writer
        let remaining_data = writer.finalize().expect("finalize");
        if !remaining_data.is_empty() {
            file.write_all(&remaining_data)
                .await
                .expect("write flushed data");
        }

        file.sync_all().await.expect("sync file");

        // Stop capture
        capture.stop().expect("stop capture");

        // Verify file was created and has content
        let metadata = fs::metadata(&output_path).expect("get file metadata");
        assert!(metadata.len() > 0, "output file should have content");

        let bytes = fs::read(&output_path).expect("read output file");
        assert!(bytes.starts_with(b"OggS"), "output should start with OggS");
        let has_opus_head = bytes
            .windows(b"OpusHead".len())
            .any(|window| window == b"OpusHead");
        assert!(has_opus_head, "output should contain OpusHead");
    }
}
