//! GTK4 recordings browser for the `record --ui` command.
//!
//! Opens a window listing all cached recordings with metadata
//! (date, duration, size, transcript preview) and controls to
//! play or delete each entry.  Recordings are split into two
//! collapsible sections: OGG recordings and WAV dictation cache.

use crate::core::config::Config;
use crate::core::error::TalkError;
use crate::core::recording_cache;
use std::path::PathBuf;

/// Window title — also used for single-instance detection.
const WINDOW_TITLE: &str = "talk-rs — Recordings";

/// Opus always uses 48 kHz internally (RFC 7845).
const OPUS_SAMPLE_RATE: u32 = 48_000;

/// Compute WAV duration in seconds from file size.
///
/// Assumes 16-bit mono 16 kHz PCM with a 44-byte header:
/// `(file_size - 44) / (16000 * 2)` = seconds.
fn wav_duration_secs(path: &std::path::Path) -> Option<f64> {
    let size = std::fs::metadata(path).ok()?.len();
    if size <= 44 {
        return None;
    }
    Some((size - 44) as f64 / 32_000.0)
}

/// Compute OGG Opus duration in seconds from the last page's granule position.
///
/// Iterates all packets to find the last absolute granule position,
/// then divides by 48 000 (Opus always uses 48 kHz per RFC 7845).
fn ogg_duration_secs(path: &std::path::Path) -> Option<f64> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = ogg::reading::PacketReader::new(std::io::BufReader::new(file));
    let mut last_absgp: u64 = 0;
    loop {
        match reader.read_packet() {
            Ok(Some(pkt)) => {
                last_absgp = pkt.absgp_page();
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    if last_absgp == 0 {
        return None;
    }
    Some(last_absgp as f64 / OPUS_SAMPLE_RATE as f64)
}

/// Format seconds into `M:SS` or `H:MM:SS`.
fn format_duration(secs: f64) -> String {
    let total = secs.round() as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{}:{:02}:{:02}", h, m, s)
    } else {
        format!("{}:{:02}", m, s)
    }
}

/// Format byte count into human-readable size.
fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{} KB", bytes / 1_000)
    } else {
        format!("{} B", bytes)
    }
}

/// Entry for one cached recording (WAV or OGG).
struct RecordingEntry {
    path: PathBuf,
    date_label: String,
    duration_label: String,
    size_label: String,
    transcript_preview: String,
}

/// Parse a date label from a timestamp-based filename stem.
///
/// Expected format: `2026-02-18T12-33-45` → `"2026-02-18 12:33:45"`.
fn date_label_from_stem(stem: &str) -> String {
    if stem.len() >= 19 {
        let date_part = &stem[..10];
        let time_part = stem[11..19].replace('-', ":");
        format!("{} {}", date_part, time_part)
    } else {
        stem.to_string()
    }
}

/// Gather OGG recordings (actual `talk-rs record` output), sorted newest-first.
///
/// Reads `output_dir` from the user configuration file.
fn list_ogg_recordings() -> Result<Vec<RecordingEntry>, TalkError> {
    let config = Config::load(None)?;
    let dir = config.output_dir;
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&dir).map_err(|e| {
        TalkError::Config(format!(
            "failed to read recordings directory {}: {}",
            dir.display(),
            e
        ))
    })?;

    let mut oggs: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| TalkError::Config(format!("failed to read directory entry: {}", e)))?;
        let path = entry.path();

        // Skip symlinks
        if path.is_symlink() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("ogg") {
            oggs.push(path);
        }
    }

    oggs.sort();
    oggs.reverse();

    let mut result = Vec::with_capacity(oggs.len());
    for ogg_path in oggs {
        let stem = ogg_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let date_label = date_label_from_stem(stem);

        let duration_label = ogg_duration_secs(&ogg_path)
            .map(format_duration)
            .unwrap_or_else(|| "?:??".to_string());

        let size_label = std::fs::metadata(&ogg_path)
            .map(|m| format_size(m.len()))
            .unwrap_or_else(|_| "?".to_string());

        result.push(RecordingEntry {
            path: ogg_path,
            date_label,
            duration_label,
            size_label,
            transcript_preview: String::new(),
        });
    }

    Ok(result)
}

/// Gather WAV dictation cache entries (with companion YML), sorted newest-first.
fn list_wav_recordings() -> Result<Vec<RecordingEntry>, TalkError> {
    let dir = recording_cache::recordings_dir()?;
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&dir).map_err(|e| {
        TalkError::Config(format!(
            "failed to read recordings directory {}: {}",
            dir.display(),
            e
        ))
    })?;

    let mut wavs: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| TalkError::Config(format!("failed to read directory entry: {}", e)))?;
        let path = entry.path();

        // Skip symlinks (last_recording.wav, last_metadata.yml)
        if path.is_symlink() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("wav") {
            wavs.push(path);
        }
    }

    // Sort lexicographically (timestamp-based names → chronological)
    wavs.sort();
    // Reverse for newest-first
    wavs.reverse();

    let mut result = Vec::with_capacity(wavs.len());
    for wav_path in wavs {
        let stem = wav_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let date_label = date_label_from_stem(stem);

        let duration_label = wav_duration_secs(&wav_path)
            .map(format_duration)
            .unwrap_or_else(|| "?:??".to_string());

        let size_label = std::fs::metadata(&wav_path)
            .map(|m| format_size(m.len()))
            .unwrap_or_else(|_| "?".to_string());

        // Try to read transcript from companion metadata YAML
        let transcript_preview = match recording_cache::metadata_path_for_recording(&wav_path) {
            Ok(Some(meta_path)) => recording_cache::read_metadata_brief(&meta_path)
                .map(|b| {
                    // Single line, truncated preview (char-safe)
                    let line = b.transcript.replace('\n', " ");
                    if line.chars().count() > 200 {
                        let truncated: String = line.chars().take(200).collect();
                        format!("{truncated}…")
                    } else {
                        line
                    }
                })
                .unwrap_or_default(),
            _ => String::new(),
        };

        result.push(RecordingEntry {
            path: wav_path,
            date_label,
            duration_label,
            size_label,
            transcript_preview,
        });
    }

    Ok(result)
}

/// Delete a recording and its companion metadata YAML files.
///
/// For WAV files, also removes matching `*_<model>.yml` companion files.
/// For OGG files, only the single file is removed (no companions).
fn delete_recording(file_path: &std::path::Path) -> Result<(), TalkError> {
    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let stem = file_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

    // Delete the file itself
    if let Err(e) = std::fs::remove_file(file_path) {
        log::warn!("failed to remove {}: {}", file_path.display(), e);
    }

    // For WAV files, also delete matching YAML metadata files
    if ext == "wav" && !stem.is_empty() {
        if let Ok(dir) = recording_cache::recordings_dir() {
            let yml_prefix = format!("{}_", stem);
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if name.starts_with(&yml_prefix)
                        && path.extension().and_then(|e| e.to_str()) == Some("yml")
                    {
                        if let Err(e) = std::fs::remove_file(&path) {
                            log::warn!("failed to remove metadata {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Open the system file manager with `file_path` highlighted.
///
/// Uses GTK's [`FileLauncher`](gtk4::FileLauncher) which passes the
/// activation token so the file manager window is raised to the front.
fn open_in_file_manager(file_path: &std::path::Path, parent_window: &gtk4::Window) {
    let gio_file = gtk4::gio::File::for_path(file_path);
    let launcher = gtk4::FileLauncher::new(Some(&gio_file));
    launcher.open_containing_folder(
        Some(parent_window),
        gtk4::gio::Cancellable::NONE,
        |result| {
            if let Err(e) = result {
                log::warn!("failed to open file manager: {}", e);
            }
        },
    );
}

/// Read a WAV file and return mono `f32` samples resampled to `target_rate`.
///
/// Supports 16-bit PCM WAV files (mono or stereo). Stereo is averaged
/// to mono. If the WAV sample rate differs from `target_rate`, linear
/// interpolation is used to resample.
fn read_wav_as_f32(path: &std::path::Path, target_rate: u32) -> Result<Vec<f32>, TalkError> {
    let data = std::fs::read(path)
        .map_err(|e| TalkError::Config(format!("failed to read WAV {}: {}", path.display(), e)))?;

    if data.len() < 44 {
        return Err(TalkError::Config(format!(
            "WAV file too small ({}B): {}",
            data.len(),
            path.display()
        )));
    }

    let channels = u16::from_le_bytes([data[22], data[23]]) as usize;
    let wav_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);

    if bits_per_sample != 16 {
        return Err(TalkError::Config(format!(
            "unsupported WAV format: {}-bit (only 16-bit PCM supported)",
            bits_per_sample
        )));
    }

    // Read i16 frames → mono f32
    let sample_data = &data[44..];
    let bytes_per_frame = 2 * channels; // 2 bytes per i16 × channels
    let num_frames = sample_data.len() / bytes_per_frame;
    let mut mono = Vec::with_capacity(num_frames);

    for frame in 0..num_frames {
        let offset = frame * bytes_per_frame;
        let mut sum: f32 = 0.0;
        for ch in 0..channels {
            let i = offset + ch * 2;
            if i + 1 < sample_data.len() {
                let s = i16::from_le_bytes([sample_data[i], sample_data[i + 1]]);
                sum += s as f32 / 32_768.0;
            }
        }
        mono.push(sum / channels as f32);
    }

    // Resample if rates differ
    if wav_rate == target_rate || wav_rate == 0 {
        return Ok(mono);
    }
    resample_linear(&mono, wav_rate, target_rate)
}

/// Read an OGG Opus file and return mono `f32` samples resampled to `target_rate`.
///
/// Decodes using the `ogg` crate for demuxing and the `opus` crate for
/// Opus decoding.  Stereo is averaged to mono.  If the device sample rate
/// differs from 48 kHz, linear interpolation resamples the output.
fn read_ogg_as_f32(path: &std::path::Path, target_rate: u32) -> Result<Vec<f32>, TalkError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TalkError::Config(format!("failed to open OGG {}: {}", path.display(), e)))?;
    let mut reader = ogg::reading::PacketReader::new(std::io::BufReader::new(file));

    // Read the first packet (OpusHead header) to determine channel count
    let head_pkt = reader
        .read_packet()
        .map_err(|e| TalkError::Config(format!("failed to read OGG header: {}", e)))?
        .ok_or_else(|| TalkError::Config("OGG file has no packets".to_string()))?;

    // OpusHead: bytes 0..7 = "OpusHead", byte 9 = channel count
    let head_data = &head_pkt.data;
    if head_data.len() < 19 || &head_data[..8] != b"OpusHead" {
        return Err(TalkError::Config(format!(
            "invalid OpusHead in {}",
            path.display()
        )));
    }
    let channel_count = head_data[9] as usize;
    let opus_channels = if channel_count >= 2 {
        opus::Channels::Stereo
    } else {
        opus::Channels::Mono
    };

    let mut decoder = opus::Decoder::new(OPUS_SAMPLE_RATE, opus_channels)
        .map_err(|e| TalkError::Config(format!("failed to create Opus decoder: {}", e)))?;

    // Skip the OpusTags packet (second packet)
    let _ = reader.read_packet();

    // Decode all remaining audio packets
    // Max Opus frame: 120ms at 48kHz = 5760 samples/channel
    let max_frame_samples = 5760 * channel_count;
    let mut decode_buf = vec![0.0f32; max_frame_samples];
    let mut all_mono = Vec::new();

    loop {
        match reader.read_packet() {
            Ok(Some(pkt)) => {
                let samples_per_channel =
                    decoder
                        .decode_float(&pkt.data, &mut decode_buf, false)
                        .map_err(|e| TalkError::Config(format!("Opus decode error: {}", e)))?;

                // Convert to mono
                for i in 0..samples_per_channel {
                    if channel_count >= 2 {
                        let mut sum: f32 = 0.0;
                        for ch in 0..channel_count {
                            sum += decode_buf[i * channel_count + ch];
                        }
                        all_mono.push(sum / channel_count as f32);
                    } else {
                        all_mono.push(decode_buf[i]);
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                log::warn!("OGG read error (continuing): {}", e);
                break;
            }
        }
    }

    // Resample from 48kHz to target_rate if needed
    if OPUS_SAMPLE_RATE == target_rate {
        return Ok(all_mono);
    }
    resample_linear(&all_mono, OPUS_SAMPLE_RATE, target_rate)
}

/// Resample mono f32 samples using linear interpolation.
fn resample_linear(mono: &[f32], src_rate: u32, target_rate: u32) -> Result<Vec<f32>, TalkError> {
    if src_rate == 0 {
        return Ok(mono.to_vec());
    }
    let ratio = target_rate as f64 / src_rate as f64;
    let new_len = (mono.len() as f64 * ratio).ceil() as usize;
    let mut resampled = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let s0 = mono.get(idx).copied().unwrap_or(0.0);
        let s1 = mono.get(idx + 1).copied().unwrap_or(s0);
        resampled.push(s0 + (s1 - s0) * frac);
    }
    Ok(resampled)
}

// ── Native WAV playback via cpal ─────────────────────────────────────

/// Shared state between the GUI thread and the `cpal` output callback.
struct WavPlaybackState {
    samples: Vec<f32>,
    position: usize,
}

/// Plays audio files (WAV or OGG) through `cpal`'s default output device.
///
/// Created once when the recordings window opens. The `cpal` output
/// stream runs continuously (outputting silence when idle). Calling
/// [`play`](WavPlayer::play) loads an audio file and starts from the
/// beginning; [`stop`](WavPlayer::stop) clears the buffer.
// Named WavPlayer for backwards compatibility; handles both WAV and OGG.
struct WavPlayer {
    state: std::sync::Arc<std::sync::Mutex<WavPlaybackState>>,
    device_sample_rate: u32,
    // Dropping this stops the stream.
    _stream: cpal::Stream,
}

impl WavPlayer {
    /// Open the default output device and start a silent stream.
    fn new() -> Result<Self, TalkError> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| TalkError::Audio("no default audio output device".to_string()))?;
        let config = device
            .default_output_config()
            .map_err(|e| TalkError::Audio(format!("output config: {}", e)))?;

        let device_sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        let state = std::sync::Arc::new(std::sync::Mutex::new(WavPlaybackState {
            samples: Vec::new(),
            position: 0,
        }));
        let state_cb = std::sync::Arc::clone(&state);

        let stream = device
            .build_output_stream(
                &cpal::StreamConfig {
                    channels: config.channels(),
                    sample_rate: config.sample_rate(),
                    buffer_size: cpal::BufferSize::Default,
                },
                move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if let Ok(mut guard) = state_cb.try_lock() {
                        let frames = output.len() / channels;
                        for frame_idx in 0..frames {
                            let sample = if guard.position < guard.samples.len() {
                                let s = guard.samples[guard.position];
                                guard.position += 1;
                                s
                            } else {
                                0.0
                            };
                            for ch in 0..channels {
                                output[frame_idx * channels + ch] = sample;
                            }
                        }
                    } else {
                        for s in output.iter_mut() {
                            *s = 0.0;
                        }
                    }
                },
                |err| log::error!("audio output error: {}", err),
                None,
            )
            .map_err(|e| TalkError::Audio(format!("output stream: {}", e)))?;

        stream
            .play()
            .map_err(|e| TalkError::Audio(format!("start output stream: {}", e)))?;

        Ok(Self {
            state,
            device_sample_rate,
            _stream: stream,
        })
    }

    /// Load an audio file (WAV or OGG) and start playing it from the beginning.
    fn play(&self, audio_path: &std::path::Path) -> Result<(), TalkError> {
        let ext = audio_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let samples = match ext {
            "ogg" => read_ogg_as_f32(audio_path, self.device_sample_rate)?,
            _ => read_wav_as_f32(audio_path, self.device_sample_rate)?,
        };
        if let Ok(mut guard) = self.state.lock() {
            guard.samples = samples;
            guard.position = 0;
        }
        Ok(())
    }

    /// Stop playback immediately.
    fn stop(&self) {
        if let Ok(mut guard) = self.state.lock() {
            guard.samples.clear();
            guard.position = 0;
        }
    }

    /// `true` when all samples have been consumed (or nothing loaded).
    fn is_finished(&self) -> bool {
        self.state
            .lock()
            .map(|g| g.samples.is_empty() || g.position >= g.samples.len())
            .unwrap_or(true)
    }
}

/// Open the GTK4 recordings browser.
pub async fn record_ui() -> Result<(), TalkError> {
    let ogg_recordings = list_ogg_recordings()?;
    let wav_recordings = list_wav_recordings()?;

    tokio::task::spawn_blocking(move || show_recordings_window(ogg_recordings, wav_recordings))
        .await
        .map_err(|e| TalkError::Config(format!("GTK task failed: {}", e)))?
}

/// Build and run the GTK4 recordings browser window.
fn show_recordings_window(
    ogg_recordings: Vec<RecordingEntry>,
    wav_recordings: Vec<RecordingEntry>,
) -> Result<(), TalkError> {
    use gtk4::glib;
    use gtk4::prelude::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    gtk4::init().map_err(|e| TalkError::Config(format!("failed to initialize GTK: {}", e)))?;

    let theme = crate::gtk_theme::ThemeColors::resolve();

    let window = gtk4::Window::builder()
        .title(WINDOW_TITLE)
        .default_width(800)
        .default_height(500)
        .decorated(false)
        .build();

    crate::gtk_theme::load_css(&theme.base_css(
        ".transcript { font-family: monospace; opacity: 0.7; } \
         .meta { font-family: monospace; } \
         .play-btn, .folder-btn, .delete-btn { min-width: 28px; min-height: 28px; max-width: 28px; max-height: 28px; padding: 0; font-size: 14px; } \
         .folder-btn label, .delete-btn label { padding-top: 4px; } \
         .section-expander { margin: 4px 2px; } \
         .section-expander > title { font-weight: bold; opacity: 0.85; padding: 4px 0; }",
    ));

    let root = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
    root.set_margin_top(6);
    root.set_margin_bottom(6);
    root.set_margin_start(6);
    root.set_margin_end(6);

    let scrolled = gtk4::ScrolledWindow::builder()
        .vexpand(true)
        .hexpand(true)
        .build();

    // WindowHandle for dragging
    let handle = gtk4::WindowHandle::new();
    handle.set_child(Some(&root));
    window.set_child(Some(&handle));

    let main_loop = glib::MainLoop::new(None, false);

    // Native audio player (cpal). Falls back to no-op if device unavailable.
    let player: Rc<Option<WavPlayer>> = Rc::new(match WavPlayer::new() {
        Ok(p) => Some(p),
        Err(e) => {
            log::warn!("audio output unavailable, play disabled: {}", e);
            None
        }
    });
    // Track which button is currently in "stop" mode.
    let active_play_btn: Rc<RefCell<Option<gtk4::Button>>> = Rc::new(RefCell::new(None));

    /// Stop any active playback and reset the corresponding button.
    fn stop_playback(player: &Rc<Option<WavPlayer>>, btn_ref: &Rc<RefCell<Option<gtk4::Button>>>) {
        if let Some(ref p) = **player {
            p.stop();
        }
        if let Some(prev_btn) = btn_ref.borrow_mut().take() {
            prev_btn.set_label("▶");
            prev_btn.set_tooltip_text(Some("Play recording"));
        }
    }

    /// Start playback of an audio file, updating button state.
    fn start_playback(
        audio_path: &std::path::Path,
        btn: &gtk4::Button,
        player: &Rc<Option<WavPlayer>>,
        btn_ref: &Rc<RefCell<Option<gtk4::Button>>>,
    ) {
        stop_playback(player, btn_ref);

        let Some(ref p) = **player else { return };
        if let Err(e) = p.play(audio_path) {
            log::warn!("failed to play {}: {}", audio_path.display(), e);
            return;
        }

        btn.set_label("■");
        btn.set_tooltip_text(Some("Stop playback"));
        *btn_ref.borrow_mut() = Some(btn.clone());

        // Poll for playback completion to reset button
        let player_poll = Rc::clone(player);
        let btn_poll = Rc::clone(btn_ref);
        let btn_widget = btn.clone();
        glib::timeout_add_local(std::time::Duration::from_millis(200), move || {
            let finished = player_poll
                .as_ref()
                .as_ref()
                .is_none_or(|p| p.is_finished());
            if finished {
                btn_widget.set_label("▶");
                btn_widget.set_tooltip_text(Some("Play recording"));
                let is_active = btn_poll.borrow().as_ref().is_some_and(|b| *b == btn_widget);
                if is_active {
                    *btn_poll.borrow_mut() = None;
                }
                glib::ControlFlow::Break
            } else {
                glib::ControlFlow::Continue
            }
        });
    }

    {
        // Container inside the scrolled window for both sections
        let sections_box = gtk4::Box::new(gtk4::Orientation::Vertical, 0);

        /// Build a single row (hbox) for a recording entry with all columns
        /// and buttons.
        fn build_row(
            recording: &RecordingEntry,
            player: &Rc<Option<WavPlayer>>,
            active_play_btn: &Rc<RefCell<Option<gtk4::Button>>>,
            window: &gtk4::Window,
            list: &gtk4::ListBox,
            expander: &gtk4::Expander,
            section_label: &str,
        ) -> gtk4::ListBoxRow {
            use gtk4::prelude::*;

            let hbox = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
            hbox.set_margin_top(4);
            hbox.set_margin_bottom(4);
            hbox.set_margin_start(4);
            hbox.set_margin_end(4);

            // Date (fixed width)
            let date_label = gtk4::Label::new(Some(&recording.date_label));
            date_label.set_xalign(0.0);
            date_label.set_width_chars(19);
            date_label.set_max_width_chars(19);
            date_label.set_selectable(false);
            date_label.add_css_class("meta");
            hbox.append(&date_label);

            // Duration (fixed width)
            let dur_label = gtk4::Label::new(Some(&recording.duration_label));
            dur_label.set_xalign(1.0);
            dur_label.set_width_chars(8);
            dur_label.set_max_width_chars(8);
            dur_label.set_selectable(false);
            dur_label.add_css_class("dim");
            dur_label.add_css_class("meta");
            hbox.append(&dur_label);

            // Size (fixed width)
            let size_label = gtk4::Label::new(Some(&recording.size_label));
            size_label.set_xalign(1.0);
            size_label.set_width_chars(8);
            size_label.set_max_width_chars(8);
            size_label.set_selectable(false);
            size_label.add_css_class("dim");
            size_label.add_css_class("meta");
            hbox.append(&size_label);

            // Transcript preview (expanding, ellipsized)
            let transcript = gtk4::Label::new(Some(&recording.transcript_preview));
            transcript.set_xalign(0.0);
            transcript.set_hexpand(true);
            transcript.set_ellipsize(gtk4::pango::EllipsizeMode::End);
            transcript.set_max_width_chars(80);
            transcript.set_selectable(false);
            transcript.add_css_class("transcript");
            hbox.append(&transcript);

            // Play button
            let play_btn = gtk4::Button::with_label("▶");
            play_btn.set_tooltip_text(Some("Play recording"));
            play_btn.add_css_class("play-btn");
            {
                let audio_path = recording.path.clone();
                let player_ref = Rc::clone(player);
                let active_btn_ref = Rc::clone(active_play_btn);
                play_btn.connect_clicked(move |btn| {
                    let is_playing = active_btn_ref.borrow().as_ref().is_some_and(|b| b == btn);
                    if is_playing {
                        stop_playback(&player_ref, &active_btn_ref);
                    } else {
                        start_playback(&audio_path, btn, &player_ref, &active_btn_ref);
                    }
                });
            }
            hbox.append(&play_btn);

            // Folder button — open file manager with file highlighted
            let folder_btn = gtk4::Button::with_label("🖿\u{FE0E}");
            folder_btn.set_tooltip_text(Some("Show in file manager"));
            folder_btn.add_css_class("folder-btn");
            {
                let audio_path = recording.path.clone();
                let win_ref = window.clone();
                folder_btn.connect_clicked(move |_| {
                    open_in_file_manager(&audio_path, &win_ref);
                });
            }
            hbox.append(&folder_btn);

            // Delete button
            let delete_btn = gtk4::Button::with_label("🗑");
            delete_btn.set_tooltip_text(Some("Delete recording"));
            delete_btn.add_css_class("delete-btn");
            {
                let audio_path = recording.path.clone();
                let list_ref = list.clone();
                let expander_ref = expander.clone();
                let slabel = section_label.to_string();
                delete_btn.connect_clicked(move |btn| {
                    if let Err(e) = delete_recording(&audio_path) {
                        log::warn!("delete failed: {}", e);
                        return;
                    }

                    // Walk up widget tree to find the ListBoxRow
                    let mut widget: Option<gtk4::Widget> = btn.parent();
                    loop {
                        match widget {
                            Some(ref w) => {
                                if let Some(row) = w.downcast_ref::<gtk4::ListBoxRow>() {
                                    list_ref.remove(row);
                                    break;
                                }
                                widget = w.parent();
                            }
                            None => return,
                        }
                    }

                    // Count remaining rows and update expander title
                    let mut count = 0;
                    let mut child = list_ref.first_child();
                    while let Some(w) = child {
                        if w.downcast_ref::<gtk4::ListBoxRow>().is_some() {
                            count += 1;
                        }
                        child = w.next_sibling();
                    }
                    expander_ref.set_label(Some(&format!("{} ({})", slabel, count)));
                });
            }
            hbox.append(&delete_btn);

            let row = gtk4::ListBoxRow::new();
            row.set_child(Some(&hbox));
            row
        }

        /// Populate a ListBox with recording entries, updating the Expander
        /// title with the count.  Clears any existing rows first.
        fn populate_section(
            label: &str,
            recordings: &[RecordingEntry],
            list: &gtk4::ListBox,
            expander: &gtk4::Expander,
            player: &Rc<Option<WavPlayer>>,
            active_play_btn: &Rc<RefCell<Option<gtk4::Button>>>,
            window: &gtk4::Window,
        ) {
            use gtk4::prelude::*;

            // Remove existing rows
            while let Some(child) = list.first_child() {
                list.remove(&child);
            }

            for recording in recordings {
                let row = build_row(
                    recording,
                    player,
                    active_play_btn,
                    window,
                    list,
                    expander,
                    label,
                );
                list.append(&row);
            }

            // Select first row
            if let Some(first) = list.row_at_index(0) {
                list.select_row(Some(&first));
            }

            expander.set_label(Some(&format!("{} ({})", label, recordings.len())));
        }

        /// Create an Expander + ListBox pair for a section.
        fn create_section(label: &str) -> (gtk4::Expander, gtk4::ListBox) {
            use gtk4::prelude::*;

            let expander = gtk4::Expander::new(Some(label));
            expander.set_expanded(true);
            expander.add_css_class("section-expander");

            let list = gtk4::ListBox::new();
            list.set_selection_mode(gtk4::SelectionMode::Single);
            list.set_activate_on_single_click(false);

            expander.set_child(Some(&list));
            (expander, list)
        }

        // ── WAV dictation cache section (first) ──
        let (wav_expander, wav_list) = create_section("Dictation cache (0)");
        populate_section(
            "Dictation cache",
            &wav_recordings,
            &wav_list,
            &wav_expander,
            &player,
            &active_play_btn,
            &window,
        );
        sections_box.append(&wav_expander);

        // ── OGG recordings section (second) ──
        let (ogg_expander, ogg_list) = create_section("Recordings (0)");
        populate_section(
            "Recordings",
            &ogg_recordings,
            &ogg_list,
            &ogg_expander,
            &player,
            &active_play_btn,
            &window,
        );
        sections_box.append(&ogg_expander);

        scrolled.set_child(Some(&sections_box));
        root.append(&scrolled);

        // Focus the first non-empty list
        if wav_list.first_child().is_some() {
            wav_list.grab_focus();
        } else if ogg_list.first_child().is_some() {
            ogg_list.grab_focus();
        }

        // ── Inotify via gio::FileMonitor ──
        // Keep monitors alive for the lifetime of the window.
        let monitors: Rc<RefCell<Vec<gtk4::gio::FileMonitor>>> = Rc::new(RefCell::new(Vec::new()));

        /// Shared context passed to [`watch_directory`] to avoid exceeding
        /// the clippy argument-count limit.
        struct WatchCtx {
            list: gtk4::ListBox,
            expander: gtk4::Expander,
            player: Rc<Option<WavPlayer>>,
            active_play_btn: Rc<RefCell<Option<gtk4::Button>>>,
            window: gtk4::Window,
        }

        // Helper: set up a directory monitor that refreshes a section on changes.
        fn watch_directory(
            dir: &std::path::Path,
            label: &'static str,
            ctx: &WatchCtx,
            list_fn: fn() -> Result<Vec<RecordingEntry>, TalkError>,
            monitors: &Rc<RefCell<Vec<gtk4::gio::FileMonitor>>>,
        ) {
            use gtk4::prelude::*;

            let gio_dir = gtk4::gio::File::for_path(dir);
            let monitor = match gio_dir.monitor_directory(
                gtk4::gio::FileMonitorFlags::NONE,
                gtk4::gio::Cancellable::NONE,
            ) {
                Ok(m) => m,
                Err(e) => {
                    log::warn!("failed to watch {}: {}", dir.display(), e);
                    return;
                }
            };

            let list_ref = ctx.list.clone();
            let exp_ref = ctx.expander.clone();
            let player_ref = Rc::clone(&ctx.player);
            let btn_ref = Rc::clone(&ctx.active_play_btn);
            let win_ref = ctx.window.clone();

            monitor.connect_changed(move |_monitor, _file, _other, event| {
                use gtk4::gio::FileMonitorEvent;
                match event {
                    FileMonitorEvent::Created
                    | FileMonitorEvent::Deleted
                    | FileMonitorEvent::ChangesDoneHint => {}
                    _ => return,
                }
                if let Ok(entries) = list_fn() {
                    populate_section(
                        label,
                        &entries,
                        &list_ref,
                        &exp_ref,
                        &player_ref,
                        &btn_ref,
                        &win_ref,
                    );
                }
            });

            monitors.borrow_mut().push(monitor);
        }

        // Watch WAV cache directory
        if let Ok(wav_dir) = recording_cache::recordings_dir() {
            let ctx = WatchCtx {
                list: wav_list,
                expander: wav_expander,
                player: Rc::clone(&player),
                active_play_btn: Rc::clone(&active_play_btn),
                window: window.clone(),
            };
            watch_directory(
                &wav_dir,
                "Dictation cache",
                &ctx,
                list_wav_recordings,
                &monitors,
            );
        }

        // Watch OGG output directory
        if let Ok(config) = Config::load(None) {
            let ctx = WatchCtx {
                list: ogg_list,
                expander: ogg_expander,
                player: Rc::clone(&player),
                active_play_btn: Rc::clone(&active_play_btn),
                window: window.clone(),
            };
            watch_directory(
                &config.output_dir,
                "Recordings",
                &ctx,
                list_ogg_recordings,
                &monitors,
            );
        }

        // Prevent monitors from being dropped
        let _keep_monitors = monitors;
    }

    // Escape to close
    {
        let ml = main_loop.clone();
        let win = window.clone();
        let player_ref = Rc::clone(&player);
        let btn_ref = Rc::clone(&active_play_btn);
        let key_ctl = gtk4::EventControllerKey::new();
        key_ctl.connect_key_pressed(move |_, key, _, _| {
            if key == gtk4::gdk::Key::Escape {
                stop_playback(&player_ref, &btn_ref);
                win.set_visible(false);
                ml.quit();
                glib::Propagation::Stop
            } else {
                glib::Propagation::Proceed
            }
        });
        window.add_controller(key_ctl);
    }

    // Window close
    {
        let ml = main_loop.clone();
        let player_ref = Rc::clone(&player);
        let btn_ref = Rc::clone(&active_play_btn);
        window.connect_close_request(move |win| {
            stop_playback(&player_ref, &btn_ref);
            win.set_visible(false);
            ml.quit();
            glib::Propagation::Proceed
        });
    }

    crate::gtk_theme::present_centred(&window);
    main_loop.run();
    window.close();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(5.0), "0:05");
        assert_eq!(format_duration(59.0), "0:59");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(60.0), "1:00");
        assert_eq!(format_duration(125.0), "2:05");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(3661.0), "1:01:01");
        assert_eq!(format_duration(7200.0), "2:00:00");
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(0), "0 B");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1_000), "1 KB");
        assert_eq!(format_size(999_999), "999 KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1_000_000), "1.0 MB");
        assert_eq!(format_size(15_500_000), "15.5 MB");
    }

    #[test]
    fn test_wav_duration_secs_too_small() {
        // File smaller than header
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let wav = dir.path().join("tiny.wav");
        std::fs::write(&wav, b"small").expect("write");
        assert!(wav_duration_secs(&wav).is_none());
    }

    #[test]
    fn test_wav_duration_secs_valid() {
        let dir = tempfile::TempDir::new().expect("create temp dir");
        let wav = dir.path().join("test.wav");
        // 44 byte header + 32000 bytes of data = 1 second at 16kHz mono 16-bit
        let data = vec![0u8; 44 + 32_000];
        std::fs::write(&wav, &data).expect("write");
        let duration = wav_duration_secs(&wav).expect("should compute duration");
        assert!((duration - 1.0).abs() < 0.001);
    }
}
