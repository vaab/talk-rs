//! Dictate command implementation.
//!
//! Records audio, streams it to the transcription API, and pastes the result
//! into the focused application via clipboard.

mod models;
mod oneshot;
mod picker;
mod realtime;
mod text;
mod toggle;

use crate::audio::bt_profile;
use crate::audio::file_source::{OggFileSource, WavFileSource};
use crate::audio::monitor_capture::MonitorCapture;
use crate::audio::pipewire_capture::PipeWireCapture;
use crate::audio::recording_feedback::{
    RecordingBadgeTeardown, RecordingFeedback, RecordingFeedbackOptions, RecordingOverlayOptions,
};
use crate::audio::resample;
use crate::audio::{AudioCapture, CHUNK_DURATION_MS};

use crate::config::{AudioConfig, Config, Provider};
use crate::daemon;
use crate::error::TalkError;
use crate::paste::{
    default_root, focus_window, get_active_window, paste_with_root, PasteNode, PasteTiming,
    RealtimeClipboardGuard,
};
use crate::recording_cache;
use crate::telemetry::{BroadcastSink, TelemetrySink, TranscriptionEvent};
use crate::transcription;
use crate::x11::overlay::IndicatorKind;
use crate::x11::visualizer::VisualizerHandle;
use models::{resolve_model, resolve_provider};
use oneshot::dictate_oneshot;
use picker::{run_pick, PickParams};
use realtime::dictate_realtime;
use std::path::PathBuf;
use toggle::toggle_dispatch;
use tokio_util::sync::CancellationToken;

/// Options for the dictate command.
pub struct DictateOpts {
    pub chain: Option<String>,
    pub lang: Option<String>,
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

#[derive(Debug, PartialEq, Eq)]
enum DictateMode {
    Picker,
    Cached(String),
    OneShot,
    Realtime,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OutputPolicy {
    no_paste: bool,
    save: Option<PathBuf>,
    output_yaml: Option<PathBuf>,
    replace_chars: usize,
    realtime: bool,
    timestamp: bool,
}

struct DictatePlan {
    mode: DictateMode,
    policy: OutputPolicy,
    provider: Provider,
    model: String,
}

impl DictatePlan {
    fn resolve(
        opts: &DictateOpts,
        config: &Config,
        input_audio: Option<&std::path::Path>,
        cached: recording_cache::TranscriptStatus,
        replace_chars: usize,
    ) -> Self {
        let specific =
            opts.chain.is_some() || opts.provider.is_some() || opts.model.is_some() || opts.diarize;
        let mode = if opts.pick {
            DictateMode::Picker
        } else if input_audio.is_some() && !specific {
            match cached {
                recording_cache::TranscriptStatus::Available(text) => DictateMode::Cached(text),
                _ if opts.realtime => DictateMode::Realtime,
                _ => DictateMode::OneShot,
            }
        } else if opts.realtime {
            DictateMode::Realtime
        } else {
            DictateMode::OneShot
        };
        let provider = resolve_provider(opts.provider, config);
        let model = resolve_model(opts.model.as_deref(), config, provider, opts.realtime);
        Self {
            mode,
            provider,
            model,
            policy: OutputPolicy {
                no_paste: opts.no_paste,
                save: opts.save.clone(),
                output_yaml: opts.output_yaml.clone(),
                replace_chars,
                realtime: opts.realtime,
                timestamp: opts.timestamp,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputSource {
    Ogg,
    Wav,
    Monitor,
    PipeWire,
}

fn select_input_source(path: Option<&std::path::Path>, monitor: bool) -> InputSource {
    match path {
        Some(path)
            if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("ogg")) =>
        {
            InputSource::Ogg
        }
        Some(_) => InputSource::Wav,
        None if monitor => InputSource::Monitor,
        None => InputSource::PipeWire,
    }
}

async fn resolve_target_window<F, Fut>(
    explicit: Option<String>,
    daemon: bool,
    active: F,
) -> Option<String>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Option<String>>,
{
    if let Some(wid) = explicit {
        log::debug!("using target window from argument: {}", wid);
        Some(wid)
    } else if daemon {
        None
    } else {
        let wid = active().await;
        if let Some(ref w) = wid {
            log::debug!("captured active window: {}", w);
        }
        wid
    }
}

fn should_fallback_to_file(path: &std::path::Path) -> bool {
    path.is_file()
}

fn validate_dictation_options(opts: &DictateOpts) -> Result<(), TalkError> {
    if opts.diarize && opts.realtime {
        return Err(TalkError::Config(
            "--diarize is not supported with --realtime: \
             the Mistral realtime WebSocket endpoint does not support speaker diarization"
                .to_string(),
        ));
    }
    Ok(())
}

fn copy_output(source: &std::path::Path, destination: &std::path::Path) -> Result<(), TalkError> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            TalkError::Config(format!("failed to create {}: {e}", parent.display()))
        })?;
    }
    std::fs::copy(source, destination).map_err(|e| {
        TalkError::Config(format!(
            "failed to copy {} to {}: {e}",
            source.display(),
            destination.display()
        ))
    })?;
    Ok(())
}

/// Start-side counterpart of the `timing: stop +Nms <step>` log lines:
/// logs `timing: start +Nms <step>` relative to the moment `dictate`
/// began, so startup latency can be attributed per step.
struct StartTiming(std::time::Instant);

impl StartTiming {
    fn mark(&self, step: &str) {
        log::info!("timing: start +{}ms {}", self.0.elapsed().as_millis(), step);
    }
}

fn should_paste_oneshot(text: &str, realtime: bool, no_paste: bool) -> bool {
    !text.is_empty() && !realtime && !no_paste
}

struct Delivery<'a> {
    policy: &'a OutputPolicy,
    audio: &'a std::path::Path,
    provider: Provider,
    model: &'a str,
    cached: bool,
    paste_root: &'a dyn PasteNode,
    target_window: Option<&'a String>,
    paste_timing: PasteTiming,
    sink: &'a dyn TelemetrySink,
    t_stop: Option<std::time::Instant>,
    authoritative: bool,
    paste_alert: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
}

trait DeliveryStore {
    fn write_pick_if_absent(
        &self,
        audio: &std::path::Path,
        provider: Provider,
        model: &str,
        realtime: bool,
        text: &str,
    ) -> Result<(), TalkError>;
    fn get(
        &self,
        audio: &std::path::Path,
        provider: Provider,
        model: &str,
    ) -> Option<transcription::TranscriptionResult>;
    fn store(
        &self,
        audio: &std::path::Path,
        provider: Provider,
        model: &str,
        realtime: bool,
        result: &transcription::TranscriptionResult,
    ) -> Result<PathBuf, TalkError>;
    fn write_last_pointers(
        &self,
        audio: &std::path::Path,
        metadata: &std::path::Path,
    ) -> Result<(), TalkError>;
    fn rotate(&self) -> Result<(), TalkError>;
    fn write_last_paste_state(&self, target: Option<&str>, text: &str) -> Result<(), TalkError>;
}

struct RecordingDeliveryStore;

impl DeliveryStore for RecordingDeliveryStore {
    fn write_pick_if_absent(
        &self,
        audio: &std::path::Path,
        provider: Provider,
        model: &str,
        realtime: bool,
        text: &str,
    ) -> Result<(), TalkError> {
        recording_cache::write_pick_if_absent(audio, &provider.to_string(), model, realtime, text)
            .map(|_| ())
    }

    fn get(
        &self,
        audio: &std::path::Path,
        provider: Provider,
        model: &str,
    ) -> Option<transcription::TranscriptionResult> {
        recording_cache::TranscriptionCache::get(audio, provider, model)
    }

    fn store(
        &self,
        audio: &std::path::Path,
        provider: Provider,
        model: &str,
        realtime: bool,
        result: &transcription::TranscriptionResult,
    ) -> Result<PathBuf, TalkError> {
        recording_cache::TranscriptionCache::store(audio, provider, model, realtime, result)
    }

    fn write_last_pointers(
        &self,
        audio: &std::path::Path,
        metadata: &std::path::Path,
    ) -> Result<(), TalkError> {
        recording_cache::write_last_pointers(audio, metadata)
    }

    fn rotate(&self) -> Result<(), TalkError> {
        recording_cache::rotate_cache()
    }

    fn write_last_paste_state(&self, target: Option<&str>, text: &str) -> Result<(), TalkError> {
        recording_cache::write_last_paste_state(target, text).map(|_| ())
    }
}

async fn deliver(
    outcome: Result<transcription::TranscriptionResult, TalkError>,
    context: Delivery<'_>,
) -> Result<String, TalkError> {
    let root = context.paste_root;
    let target = context.target_window;
    let timing = context.paste_timing;
    let stop = context.t_stop;
    let sink = context.sink;
    let alert = context.paste_alert.clone();
    deliver_with(
        outcome,
        context,
        &RecordingDeliveryStore,
        move |text, delete| async move {
            paste_with_root(root, target, &text, delete, stop, sink, timing, alert).await
        },
    )
    .await
}

async fn deliver_with<F, Fut>(
    outcome: Result<transcription::TranscriptionResult, TalkError>,
    context: Delivery<'_>,
    store: &dyn DeliveryStore,
    paste: F,
) -> Result<String, TalkError>
where
    F: FnOnce(String, usize) -> Fut,
    Fut: std::future::Future<Output = Result<(), TalkError>>,
{
    if let Some(path) = &context.policy.save {
        copy_output(context.audio, path)?;
    }
    let result = outcome?;
    let text = transcription::format_transcription_output(&result, context.policy.timestamp)
        .trim()
        .to_string();
    if !context.cached && context.authoritative {
        if let Err(e) = store.write_pick_if_absent(
            context.audio,
            context.provider,
            context.model,
            context.policy.realtime,
            &text,
        ) {
            log::warn!("failed to write pick file: {e}");
        }
    }
    let mut result_for_cache = transcription::TranscriptionResult {
        text: text.clone(),
        ..result
    };
    if context.cached && context.policy.output_yaml.is_some() {
        if let Some(previous) = store.get(context.audio, context.provider, context.model) {
            result_for_cache.metadata = previous.metadata;
            result_for_cache.diarization = previous.diarization;
            result_for_cache.segments = previous.segments;
        }
    }
    let cache_meta_path = if context.cached && context.policy.output_yaml.is_none() {
        None
    } else {
        Some(store.store(
            context.audio,
            context.provider,
            context.model,
            context.policy.realtime,
            &result_for_cache,
        ))
    };
    if !context.cached {
        if let Some(metadata) = &cache_meta_path {
            match metadata {
                Ok(path) => {
                    if let Err(e) = store.write_last_pointers(context.audio, path) {
                        log::warn!("failed to update last recording pointers: {e}");
                    }
                }
                Err(e) => log::warn!("failed to write recording metadata: {e}"),
            }
        }
        if let Err(e) = store.rotate() {
            log::warn!("failed to rotate recording cache: {e}");
        }
    }
    if let Some(path) = &context.policy.output_yaml {
        let source = cache_meta_path
            .as_ref()
            .ok_or_else(|| TalkError::Config("missing transcription metadata".into()))?
            .as_ref()
            .map_err(|e| TalkError::Config(format!("failed to write metadata: {e}")))?;
        copy_output(source, path)?;
    }
    if should_paste_oneshot(&text, context.policy.realtime, context.policy.no_paste) {
        context.sink.emit(TranscriptionEvent::PasteStarted {
            t: std::time::Instant::now(),
        });
        paste(text.clone(), context.policy.replace_chars).await?;
        let _ = store.write_last_paste_state(context.target_window.map(String::as_str), &text);
        context.sink.emit(TranscriptionEvent::PasteCompleted {
            t: std::time::Instant::now(),
        });
    }
    if !context.policy.realtime && !text.is_empty() {
        println!("{text}");
    }
    Ok(text)
}

async fn consume_realtime_segments<F, Fut>(
    root: std::sync::Arc<dyn PasteNode>,
    mut segments: tokio::sync::mpsc::Receiver<String>,
    no_paste: bool,
    replace_chars: usize,
    mut deliver: F,
) -> String
where
    F: FnMut(std::sync::Arc<dyn PasteNode>, String, usize) -> Fut,
    Fut: std::future::Future<Output = Result<(), TalkError>>,
{
    let mut is_first = true;
    let mut tail = None;
    let mut pasted = String::new();
    while let Some(segment) = segments.recv().await {
        if no_paste || segment.is_empty() {
            continue;
        }
        let text = crate::transcription::realtime::join_segment(tail, &segment);
        let delete = if is_first { replace_chars } else { 0 };
        log::trace!("paste(realtime): {}", crate::paste::log_preview(&text));
        if let Err(e) = deliver(root.clone(), text.clone(), delete).await {
            log::warn!("per-segment paste failed: {}", e);
        } else {
            is_first = false;
            tail = text.chars().last();
            pasted.push_str(&text);
        }
    }
    pasted
}

/// Dictate: record audio, transcribe, and paste into focused application.
pub async fn dictate(opts: DictateOpts) -> Result<(), TalkError> {
    validate_dictation_options(&opts)?;
    // Toggle mode: start or stop a daemon
    if opts.toggle {
        return toggle_dispatch(&opts).await;
    }
    let t_start = StartTiming(std::time::Instant::now());

    let _daemon_owner = if opts.daemon {
        Some(daemon::dictate_slot()?.owner_guard())
    } else {
        None
    };

    // Load configuration
    let config = Config::load(None)?;
    t_start.mark("config_loaded");

    dictate_loaded(opts, config, t_start).await
}

async fn dictate_loaded(
    opts: DictateOpts,
    mut config: Config,
    t_start: StartTiming,
) -> Result<(), TalkError> {
    // Build the runtime paste-node tree from config (or fall back to
    // the default `chunk(150) → clipboard(ctrl-shift-v, 200, 400)`
    // tree when no `paste:` section is configured).  `--no-chunk-paste`
    // strips chunk wrappers from whatever tree is configured.
    let paste_root: std::sync::Arc<dyn PasteNode> = match config.paste.as_ref() {
        Some(p) => std::sync::Arc::from(p.build_root(opts.no_chunk_paste)),
        None => std::sync::Arc::from(default_root(opts.no_chunk_paste)),
    };

    // Resolve paste timing from config (defaults: 200 / 400).  Drives
    // the settle loop in `paste_with_root` and the realtime guard.
    let paste_timing = config
        .paste
        .as_ref()
        .map(|p| p.timing())
        .unwrap_or_default();

    // Determine target window: use --target-window arg (from daemon mode)
    // or capture the currently active window.
    let target_window =
        resolve_target_window(opts.target_window.clone(), opts.daemon, get_active_window).await;

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
            replace_char_count = Some(state.replacement_count_for(target_window.as_deref()));
        }
    }

    let cached_status = input_audio_file.as_deref().map_or(
        recording_cache::TranscriptStatus::NotAvailable,
        recording_cache::get_transcript,
    );
    let mut plan = DictatePlan::resolve(
        &opts,
        &config,
        input_audio_file.as_deref(),
        cached_status,
        replace_char_count.unwrap_or(0),
    );
    let chain = config
        .resolve_chain(
            crate::config::ChainCommand::Dictate,
            opts.chain.as_deref(),
            opts.provider,
            opts.model.as_deref(),
        )?
        .map(|chain| chain.eligible(opts.diarize, opts.realtime, opts.lang.as_deref()))
        .transpose()?;
    if let (Some(lang), Some(openai)) = (&opts.lang, &mut config.providers.openai) {
        openai.languages = Some(vec![lang.clone()]);
    }
    let outage_path = if chain.is_some() {
        Some(transcription::chain::outage_path()?)
    } else {
        None
    };
    if let (Some(chain), Some(path)) = (&chain, &outage_path) {
        let first = chain.first_available(path.clone()).ok_or_else(|| {
            TalkError::Config(format!("chain \"{}\" has no eligible entries", chain.name))
        })?;
        plan.provider = first.provider;
        plan.model = first.model.clone();
    }
    let policy = plan.policy;
    if plan.mode == DictateMode::Picker {
        return run_pick(
            config,
            PickParams {
                chain: opts.chain.clone(),
                input_audio_file,
                cached_brief,
                replace_char_count,
                replace_last_paste: opts.replace_last_paste,
                provider: opts.provider,
                model: opts.model,
                target_window,
                paste_root: paste_root.clone(),
                paste_timing,
            },
        )
        .await;
    }

    // Mode C: file input + default options -> consult the pick file
    // directly.  If it exists, paste its text without running any
    // capture/transcription pipeline.  Matches the `transcribe`
    // command's default branch.
    if let DictateMode::Cached(text) = plan.mode {
        let audio = input_audio_file
            .as_ref()
            .ok_or_else(|| TalkError::Config("cached transcript has no audio file".into()))?;
        let (provider, model, _streaming, _) =
            recording_cache::read_pick(audio).ok_or_else(|| {
                TalkError::Config(format!(
                    "cached transcript vanished for {}",
                    audio.display()
                ))
            })?;
        let mut cached_policy = policy.clone();
        cached_policy.realtime = false;
        deliver(
            Ok(transcription::TranscriptionResult {
                text,
                metadata: Default::default(),
                diarization: None,
                segments: None,
            }),
            Delivery {
                policy: &cached_policy,
                audio,
                provider,
                model: &model,
                cached: true,
                paste_root: paste_root.as_ref(),
                target_window: target_window.as_ref(),
                paste_timing,
                sink: &crate::telemetry::NoOpSink,
                t_stop: None,
                authoritative: false,
                paste_alert: None,
            },
        )
        .await?;
        return Ok(());
    }

    // Resolve recording feedback before GTK, capture-source construction, and
    // Bluetooth switching, preserving dictate's established sound-device
    // initialization order.  The two source branches below use these exact
    // rates (16 kHz file input, 48 kHz live PipeWire capture).
    let source = select_input_source(input_audio_file.as_deref(), opts.monitor);
    let feedback_capture_rate = match source {
        InputSource::Ogg | InputSource::Wav => AudioConfig::new().sample_rate,
        InputSource::Monitor | InputSource::PipeWire => 48_000,
    };
    let viz_mode = opts.viz.or_else(|| {
        config
            .indicators
            .as_ref()
            .and_then(|indicators| indicators.viz)
    });
    if let Some(mode) = viz_mode {
        log::info!("visualizer mode: {}", mode);
    }
    let (silence_tx, silence_rx) = std::sync::mpsc::channel::<bool>();
    let broker = std::sync::Arc::new(BroadcastSink::new(256));
    let sink: std::sync::Arc<dyn TelemetrySink> = broker.clone();
    let suppress_boop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let boop_interval_ms = config
        .indicators
        .as_ref()
        .map(|indicators| indicators.boop_interval_ms)
        .unwrap_or(5_000);
    // Diagnose inaudible start tones on Bluetooth: record which sink the
    // tone will go to and whether it was asleep BEFORE the sound player
    // opens its stream (opening it wakes the sink).  Debug-only because
    // it costs an extra PulseAudio round trip.
    if !opts.no_sounds && log::log_enabled!(log::Level::Debug) {
        match bt_profile::describe_default_sink() {
            Ok(Some(sink)) => log::debug!("start tone output sink before wake: {}", sink),
            Ok(None) => log::debug!("start tone output sink before wake: no default sink"),
            Err(e) => log::debug!("start tone output sink before wake: unavailable: {}", e),
        }
        t_start.mark("sink_probed");
    }
    let mut feedback = RecordingFeedback::new(RecordingFeedbackOptions {
        no_sounds: opts.no_sounds,
        no_boop: opts.no_boop,
        no_overlay: opts.no_overlay,
        viz: viz_mode,
        mono: config.resolved_mono(opts.mono),
        boop_interval_ms,
        capture_rate: feedback_capture_rate,
        pause_audio: true,
        suppress_boop: Some(suppress_boop.clone()),
        overlay: RecordingOverlayOptions {
            silence_tx: Some(silence_tx),
            auto_pause: !opts.no_auto_pause,
            telemetry_rx: Some(broker.subscribe()),
        },
    });
    t_start.mark("sound_player_ready");

    // Ensure GTK4/GDK4 is initialised so the overlay and visualizer can
    // query monitor geometry via GDK.  `gtk4::init()` is idempotent —
    // safe to call even if the picker path already initialised it.
    if let Err(e) = gtk4::init() {
        log::warn!(
            "GTK4 init failed (overlay/visualizer may be unavailable): {}",
            e
        );
    }
    t_start.mark("gtk_ready");

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

    let mut provider = plan.provider;
    let mut effective_model = plan.model;

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
            let capture: Box<dyn AudioCapture> = match source {
                InputSource::Ogg => Box::new(OggFileSource::new(path)?),
                InputSource::Wav => Box::new(WavFileSource::new(path, &encode_config)?),
                _ => unreachable!("file path selects file source"),
            };
            (capture, encode_config.sample_rate)
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
            if source == InputSource::Monitor {
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
    // dictate_oneshot/dictate_realtime where SIGINT has no handler
    // and the daemon becomes an unkillable orphan.
    let shutdown = CancellationToken::new();
    let shutdown_clone = shutdown.clone();
    let daemon_pid = std::process::id();
    tokio::spawn(async move {
        log::debug!(
            "daemon {}: ctrl_c handler task polled, registering handler",
            daemon_pid
        );
        let _ = tokio::signal::ctrl_c().await;
        log::debug!(
            "daemon {}: SIGINT received! cancelling shutdown token",
            daemon_pid
        );
        shutdown_clone.cancel();
    });

    log::info!(
        "starting {} transcription{}",
        if opts.realtime {
            "realtime"
        } else {
            "one-shot"
        },
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
    //    moved into dictate_oneshot / dictate_realtime, where it is
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
    t_start.mark("bt_profile_done");

    // The badge is shown first so the shortcut is acknowledged visually
    // without waiting for the tone.  The start sound is awaited before
    // capture so it cannot enter the recording.  Boop begins after
    // capture starts.
    feedback.prepare_recording();
    t_start.mark("badge_requested");
    feedback.play_start().await;
    t_start.mark("start_tone_done");
    let raw_audio_rx = capture.start()?;
    t_start.mark("capture_started");

    // Parakeet is a local backend whose model must be downloaded once
    // (~640 MB).  The transcribe pipeline never downloads silently, so
    // for the toggle/dictate flow we download it HERE — before
    // recording starts — showing a "downloading model" badge.  For
    // this non-interactive surface, selecting the Parakeet provider is
    // the consent (user-confirmed).  No-op once the model is installed.
    #[cfg(feature = "parakeet")]
    if provider == Provider::Parakeet {
        let status = crate::transcription::parakeet::consent::resolve(&config)?;
        if !status.present {
            if let Some(o) = feedback.overlay() {
                o.show(IndicatorKind::DownloadingModel);
            }
            log::info!(
                "parakeet model not found at {}; downloading before recording",
                status.model_dir.display()
            );
            crate::transcription::parakeet::model::download_model(
                &status.model_dir,
                status.variant,
            )
            .await?;
        }
    }

    // Show visualizer text panel (positioned relative to recording badge).
    // The badge itself was requested before the start tone; re-requesting
    // it here restores it after a model-download badge and is otherwise a
    // cheap state reset in the overlay thread.
    feedback.show_recording_badge();
    if let Some(ref viz) = visualizer {
        log::debug!("showing visualizer text panel");
        viz.show(crate::x11::overlay::BADGE_W);
    }
    feedback.start_boop();

    // Tee audio through the shared feedback component.  Dictate keeps its
    // existing auto-pause forwarding policy; record selects pass-through.
    let raw_for_resample = if feedback.overlay().is_some() {
        let teed = feedback.route_audio(raw_audio_rx);
        // Spawn silence notification thread: forwards silence events
        // from the overlay to the visualizer text panel and plays
        // periodic alert sounds when no audio is detected.
        let vis_push = visualizer.as_ref().map(|v| v.message_pusher());
        let alert_player = feedback.player().map(|player| player.alert_player());
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

    let mut t_stop: Option<std::time::Instant> = None;
    let result = if opts.realtime {
        // Realtime mode (--realtime): stream audio over WebSocket.
        // Each segment is pasted into the focused application as it
        // arrives, providing real-time feedback while dictating.

        // Save clipboard and focus target window before recording starts.
        // Each segment is pasted whole (no chunking) — match legacy
        // behaviour by routing through the configured paste tree with
        // chunk wrappers stripped.
        let rt_guard = if opts.no_paste {
            None
        } else {
            Some(RealtimeClipboardGuard::begin(paste_timing).await)
        };
        if !opts.no_paste {
            if let Some(ref wid) = target_window {
                log::debug!("pre-focusing target window: {}", wid);
                if !focus_window(wid).await {
                    log::warn!("could not pre-focus target window {}", wid);
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }

        // Build the realtime-flavoured root: strip chunk wrappers
        // so each segment is pasted as a single clipboard call.
        let rt_root: std::sync::Arc<dyn PasteNode> = match config.paste.as_ref() {
            Some(p) => std::sync::Arc::from(p.build_root(true)),
            None => std::sync::Arc::from(default_root(true)),
        };

        // Create segment channel for per-segment pasting
        let (seg_tx, seg_rx) = tokio::sync::mpsc::channel::<String>(32);

        // Spawn paste consumer: each segment is pasted immediately
        // through the SAME node tree used by the one-shot path.
        let rt_root_for_task = rt_root.clone();
        let no_paste = opts.no_paste;
        let replacement = replace_char_count.unwrap_or(0);
        let paste_window = target_window.clone();
        let paste_task = tokio::spawn(async move {
            if let Some(guard) = rt_guard {
                let pasted = consume_realtime_segments(
                    rt_root_for_task.clone(),
                    seg_rx,
                    no_paste,
                    replacement,
                    |root, text, delete| {
                        let window = paste_window.as_deref();
                        let guard = &guard;
                        async move {
                            guard
                                .paste_segment(
                                    root.as_ref(),
                                    &text,
                                    delete,
                                    window,
                                    &crate::telemetry::NoOpSink,
                                )
                                .await
                        }
                    },
                )
                .await;
                guard.finish().await;
                pasted
            } else {
                consume_realtime_segments(
                    rt_root_for_task,
                    seg_rx,
                    true,
                    0,
                    |_root, _text, _delete| async { Ok(()) },
                )
                .await
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
            &mut feedback,
            Some(seg_tx),
            visualizer.as_ref(),
            &shutdown,
            bt_guard,
            chain.as_ref(),
            outage_path.as_deref(),
        )
        .await
        {
            Ok((r, answered_provider, answered_model)) => {
                provider = answered_provider;
                if chain.is_some() {
                    effective_model = answered_model;
                }
                r
            }
            Err(e) => {
                if let Err(join_error) = paste_task.await {
                    log::warn!("paste task error: {}", join_error);
                }
                // Enrich model-not-found errors with available models.
                let enriched =
                    transcription::enrich_model_error(&config, provider, opts.model.as_deref(), e)
                        .await;
                return Err(enriched);
            }
        };

        // Wait for the paste task to drain every segment and run
        // its end-of-stream settle + restore (inside `rt_guard.finish()`).
        match paste_task.await {
            Ok(pasted) if !pasted.is_empty() => {
                let _ = recording_cache::write_last_paste_state(target_window.as_deref(), &pasted);
            }
            Err(e) => log::warn!("paste task error: {}", e),
            _ => {}
        }

        result
    } else {
        // One-shot mode (default): capture audio, encode, then transcribe.
        // Dictate is an autonomous pipeline (no human watching),
        // so use `Proportional` so a hung server cannot wedge the
        // pipeline forever.
        let mut transcriber = transcription::create_oneshot_transcriber(
            &config,
            provider,
            Some(&effective_model),
            opts.diarize,
            transcription::RequestTimeoutPolicy::Proportional,
        )?;
        transcriber.set_sink(sink.clone());
        if let Some(entry) = chain.as_ref().and_then(|chain| {
            chain
                .entries
                .iter()
                .find(|e| e.provider == provider && e.model == effective_model)
        }) {
            transcriber.set_retry_schedule(entry.retry_schedule());
        }

        let (stream_result, t_stop_val) = dictate_oneshot(
            &mut *capture,
            from_file,
            encode_config.clone(),
            audio_rx,
            &cache_path,
            transcriber,
            &shutdown,
            &mut feedback,
            visualizer.as_ref(),
            &config,
            provider,
            Some(&effective_model),
            opts.diarize,
            chain.is_some(),
            bt_guard,
        )
        .await;
        t_stop = t_stop_val;

        // If the one-shot transcription failed, fall back to a
        // single call to `transcribe_audio` using the saved OGG
        // file.  Retry lives inside `transcribe_audio` (see
        // `transcription::transport::retry`) — no loop here.
        match stream_result {
            Ok(mut r) => {
                if chain.is_some() {
                    r.metadata.attempts.push(transcription::chain::Attempt {
                        provider: provider.to_string(),
                        model: effective_model.clone(),
                        outcome: "success".into(),
                    });
                }
                r
            }
            Err(first_err) => {
                log::warn!("one-shot transcription failed: {}", first_err);

                if let (Some(chain), Some(path)) = (&chain, &outage_path) {
                    let continuation = async {
                        if !first_err.is_fallback_worthy() || !should_fallback_to_file(&cache_path)
                        {
                            return Err(first_err);
                        }
                        let first_index = chain
                            .entries
                            .iter()
                            .position(|e| e.provider == provider && e.model == effective_model)
                            .unwrap_or(0);
                        chain.record_busy(provider, first_err.retry_after(), path.clone());
                        if first_index + 1 == chain.entries.len() {
                            return Err(first_err);
                        }
                        let prior = vec![transcription::chain::Attempt {
                            provider: provider.to_string(),
                            model: effective_model.clone(),
                            outcome: "busy".into(),
                        }];
                        let notify = |message: &str| {
                            if let Some(viz) = &visualizer {
                                viz.push_message(message);
                            }
                        };
                        chain
                            .run_file(
                                &cache_path,
                                &config,
                                opts.diarize,
                                opts.lang.as_deref(),
                                &sink,
                                path.clone(),
                                first_index + 1,
                                prior,
                                Some(&notify),
                            )
                            .await
                    }
                    .await;
                    match continuation {
                        Ok(outcome) => {
                            provider = outcome.provider;
                            effective_model = outcome.model;
                            outcome.result
                        }
                        Err(final_err) => {
                            let final_msg = final_err.to_string();
                            log::error!("{}", final_msg);
                            if let Some(ref viz) = visualizer {
                                viz.push_message(&format!("Error: {}", final_msg));
                            }
                            if let Some(o) = feedback.overlay() {
                                o.hide();
                            }
                            return deliver(
                                Err(final_err),
                                Delivery {
                                    policy: &policy,
                                    audio: &cache_path,
                                    provider,
                                    model: &effective_model,
                                    cached: false,
                                    paste_root: paste_root.as_ref(),
                                    target_window: target_window.as_ref(),
                                    paste_timing,
                                    sink: &*sink,
                                    t_stop,
                                    authoritative: false,
                                    paste_alert: None,
                                },
                            )
                            .await
                            .map(|_| ());
                        }
                    }
                } else {
                    if !should_fallback_to_file(&cache_path) {
                        return deliver(
                            Err(first_err),
                            Delivery {
                                policy: &policy,
                                audio: &cache_path,
                                provider,
                                model: &effective_model,
                                cached: false,
                                paste_root: paste_root.as_ref(),
                                target_window: target_window.as_ref(),
                                paste_timing,
                                sink: &*sink,
                                t_stop,
                                authoritative: false,
                                paste_alert: None,
                            },
                        )
                        .await
                        .map(|_| ());
                    }

                    // Fall back to file-based transcription.  Retry is
                    // already handled by Layer 3.  Same `Proportional`
                    // policy as the one-shot path above — autonomous
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
                            retry_schedule: None,
                            language: None,
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
                            if let Some(o) = feedback.overlay() {
                                o.hide();
                            }
                            return deliver(
                                Err(final_err),
                                Delivery {
                                    policy: &policy,
                                    audio: &cache_path,
                                    provider,
                                    model: &effective_model,
                                    cached: false,
                                    paste_root: paste_root.as_ref(),
                                    target_window: target_window.as_ref(),
                                    paste_timing,
                                    sink: &*sink,
                                    t_stop,
                                    authoritative: false,
                                    paste_alert: None,
                                },
                            )
                            .await
                            .map(|_| ());
                        }
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

    // Idempotent recording-phase teardown; mode-specific paths may already
    // have cancelled the boop at the instant capture stopped.
    feedback.teardown_recording(RecordingBadgeTeardown::KeepVisible);

    // For realtime mode, play stop sound and hide visualizer here
    // (one-shot mode already did this inside dictate_oneshot on SIGINT).
    if opts.realtime {
        feedback.play_stop().await;
        if let Some(ref viz) = visualizer {
            log::debug!("hiding visualizer");
            viz.hide();
        }
        if let Some(o) = feedback.overlay() {
            log::debug!("hiding overlay");
            o.hide();
        }
    }
    // One-shot mode: overlay is still showing "Transcribing" badge —
    // it will be hidden after paste (below).

    let paste_alert: Option<std::sync::Arc<dyn Fn() + Send + Sync>> = feedback.player().map(|p| {
        let ap = p.alert_player();
        std::sync::Arc::new(move || ap.play()) as std::sync::Arc<dyn Fn() + Send + Sync>
    });
    let text = deliver(
        Ok(result),
        Delivery {
            policy: &policy,
            audio: &cache_path,
            provider,
            model: &effective_model,
            cached: false,
            paste_root: paste_root.as_ref(),
            target_window: target_window.as_ref(),
            paste_timing,
            sink: &*sink,
            t_stop,
            authoritative: opts.provider.is_none() && opts.model.is_none() && !opts.diarize,
            paste_alert,
        },
    )
    .await?;
    if text.is_empty() {
        log::warn!("empty transcription — nothing to paste");
    }
    if !opts.realtime {
        if let Some(o) = feedback.overlay() {
            o.hide();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::toggle::tests::test_opts;
    use super::{
        consume_realtime_segments, deliver, resolve_target_window, select_input_source,
        should_fallback_to_file, should_paste_oneshot, validate_dictation_options, Delivery,
        DictateMode, DictatePlan, InputSource, OutputPolicy,
    };
    use crate::config::Provider;
    use crate::error::TalkError;
    use crate::paste::{PasteCtx, PasteNode};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    struct RecordingPaste(Arc<Mutex<Vec<(String, usize)>>>);

    #[async_trait]
    impl PasteNode for RecordingPaste {
        async fn paste(&self, text: &str, ctx: &PasteCtx<'_>) -> Result<(), TalkError> {
            self.0
                .lock()
                .expect("calls lock")
                .push((text.to_string(), ctx.delete_chars_before_paste));
            Ok(())
        }
    }

    async fn record_segment(
        root: Arc<dyn PasteNode>,
        text: String,
        delete: usize,
    ) -> Result<(), TalkError> {
        let clipboard = crate::clipboard::X11Clipboard::new();
        let ctx = PasteCtx {
            target_window: None,
            delete_chars_before_paste: delete,
            t_stop: None,
            sink: &crate::telemetry::NoOpSink,
            clipboard: &clipboard,
            target_client_base: None,
            expected_target_fetches: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            alert: None,
        };
        root.paste(&text, &ctx).await
    }

    #[tokio::test]
    async fn realtime_no_paste_never_calls_delivery() {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        tx.send("Hello".to_string()).await.expect("send");
        drop(tx);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let root: Arc<dyn PasteNode> = Arc::new(RecordingPaste(calls.clone()));
        let pasted = consume_realtime_segments(root, rx, true, 5, record_segment).await;
        assert!(calls.lock().expect("calls lock").is_empty());
        assert!(pasted.is_empty());
    }

    #[tokio::test]
    async fn realtime_replacement_deletes_once_then_appends() {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        tx.send("Hello".to_string()).await.expect("send");
        tx.send("world".to_string()).await.expect("send");
        drop(tx);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let root: Arc<dyn PasteNode> = Arc::new(RecordingPaste(calls.clone()));
        let pasted = consume_realtime_segments(root, rx, false, 5, record_segment).await;
        assert_eq!(
            *calls.lock().expect("calls lock"),
            [("Hello".to_string(), 5), (" world".to_string(), 0)]
        );
        assert_eq!(pasted, "Hello world");
    }

    #[test]
    fn empty_transcript_never_reaches_oneshot_paste() {
        assert!(!should_paste_oneshot("", false, false));
        assert!(should_paste_oneshot("recognized", false, false));
        assert!(!should_paste_oneshot("recognized", true, false));
        assert!(!should_paste_oneshot("recognized", false, true));
    }

    #[test]
    fn plan_resolves_mode_and_output_policy_from_options_and_cache() {
        let config: crate::config::Config =
            serde_yaml::from_str("output_dir: /tmp\nproviders: {}\n").expect("configuration");
        let cases = [
            (false, false, false, false, DictateMode::OneShot),
            (true, false, false, false, DictateMode::Picker),
            (false, true, false, false, DictateMode::Realtime),
            (
                false,
                false,
                true,
                false,
                DictateMode::Cached("pick".into()),
            ),
            (false, false, true, true, DictateMode::OneShot),
        ];
        for (pick, realtime, file, specific, expected) in cases {
            let mut opts = test_opts();
            opts.pick = pick;
            opts.realtime = realtime;
            opts.model = specific.then(|| "override".into());
            opts.no_paste = true;
            opts.save = Some("/tmp/audio.ogg".into());
            opts.output_yaml = Some("/tmp/output.yaml".into());
            let input = file.then(|| std::path::Path::new("/tmp/input.ogg"));
            let plan = DictatePlan::resolve(
                &opts,
                &config,
                input,
                crate::recording_cache::TranscriptStatus::Available("pick".into()),
                7,
            );
            assert_eq!(plan.mode, expected);
            assert!(plan.policy.no_paste);
            assert_eq!(plan.policy.replace_chars, 7);
            assert_eq!(plan.policy.save, opts.save);
            assert_eq!(plan.policy.output_yaml, opts.output_yaml);
        }
    }

    #[tokio::test]
    async fn target_window_policy_prefers_explicit_and_never_queries_for_daemon() {
        let mut queries = 0;
        assert_eq!(
            resolve_target_window(Some("42".into()), false, || {
                queries += 1;
                async { Some("other".into()) }
            })
            .await,
            Some("42".into())
        );
        assert_eq!(
            resolve_target_window(None, true, || {
                queries += 1;
                async { Some("other".into()) }
            })
            .await,
            None
        );
        assert_eq!(queries, 0);
        assert_eq!(
            resolve_target_window(None, false, || {
                queries += 1;
                async { Some("focused".into()) }
            })
            .await,
            Some("focused".into())
        );
        assert_eq!(queries, 1);
    }

    #[test]
    fn input_source_policy_distinguishes_live_monitor_and_file_formats() {
        let cases = [
            (None, false, InputSource::PipeWire),
            (None, true, InputSource::Monitor),
            (Some("voice.ogg"), true, InputSource::Ogg),
            (Some("voice.OGG"), false, InputSource::Ogg),
            (Some("voice.wav"), false, InputSource::Wav),
        ];
        for (file, monitor, expected) in cases {
            assert_eq!(
                select_input_source(file.map(std::path::Path::new), monitor),
                expected
            );
        }
    }

    #[test]
    fn plan_resolves_provider_model_and_realtime_from_configuration_or_override() {
        let config: crate::config::Config = serde_yaml::from_str(
            "output_dir: /tmp\nproviders:\n  openai:\n    api_key: unused\n    model: configured-batch\n    realtime_model: configured-live\ntranscription:\n  default_provider: openai\n",
        ).expect("configuration");
        let cases = [
            (
                None,
                None,
                false,
                Provider::OpenAI,
                "configured-batch",
                DictateMode::OneShot,
            ),
            (
                None,
                None,
                true,
                Provider::OpenAI,
                "configured-live",
                DictateMode::Realtime,
            ),
            (
                Some(Provider::Mistral),
                Some("explicit"),
                false,
                Provider::Mistral,
                "explicit",
                DictateMode::OneShot,
            ),
        ];
        for (provider, model, realtime, expected_provider, expected_model, expected_mode) in cases {
            let mut opts = test_opts();
            opts.provider = provider;
            opts.model = model.map(str::to_owned);
            opts.realtime = realtime;
            let plan = DictatePlan::resolve(
                &opts,
                &config,
                None,
                crate::recording_cache::TranscriptStatus::NotAvailable,
                0,
            );
            assert_eq!(plan.provider, expected_provider);
            assert_eq!(plan.model, expected_model);
            assert_eq!(plan.mode, expected_mode);
        }
    }

    #[test]
    fn missing_fallback_audio_retains_original_transcription_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("not-recorded.ogg");
        assert!(!should_fallback_to_file(&missing));
        std::fs::write(&missing, b"recorded").expect("recording");
        assert!(should_fallback_to_file(&missing));
    }

    #[test]
    fn incompatible_realtime_diarization_fails_before_capture() {
        let mut opts = test_opts();
        opts.realtime = true;
        opts.diarize = true;
        assert!(matches!(
            validate_dictation_options(&opts),
            Err(TalkError::Config(_))
        ));
        opts.realtime = false;
        assert!(validate_dictation_options(&opts).is_ok());
    }

    #[tokio::test]
    async fn failed_transcription_still_saves_audio_to_nested_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("cached.ogg");
        std::fs::write(&audio, b"recorded bytes").expect("audio fixture");
        let save = dir.path().join("nested/saved.ogg");
        let policy = OutputPolicy {
            no_paste: true,
            save: Some(save.clone()),
            output_yaml: None,
            replace_chars: 0,
            realtime: false,
            timestamp: false,
        };
        let root = RecordingPaste(Arc::new(Mutex::new(Vec::new())));
        let result = deliver(
            Err(TalkError::Transcription("provider failed".into())),
            Delivery {
                policy: &policy,
                audio: &audio,
                provider: Provider::Mistral,
                model: "test-model",
                cached: false,
                paste_root: &root,
                target_window: None,
                paste_timing: Default::default(),
                sink: &crate::telemetry::NoOpSink,
                t_stop: None,
                authoritative: false,
                paste_alert: None,
            },
        )
        .await;
        assert_eq!(std::fs::read(save).expect("saved audio"), b"recorded bytes");
        assert_eq!(
            result.expect_err("failure retained").to_string(),
            "Transcription error: provider failed"
        );
    }

    #[tokio::test]
    async fn cached_pick_honors_save_yaml_and_no_paste() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("cached.ogg");
        std::fs::write(&audio, b"recorded bytes").expect("audio fixture");
        crate::recording_cache::TranscriptionCache::store(
            &audio,
            Provider::Mistral,
            "cached-model",
            false,
            &crate::transcription::TranscriptionResult {
                text: "original text".into(),
                segments: Some(vec![crate::transcription::TranscriptSegment {
                    start: 1.0,
                    end: 2.0,
                    text: "original segment".into(),
                }]),
                ..Default::default()
            },
        )
        .expect("existing provider metadata");
        let save = dir.path().join("export/audio.ogg");
        let yaml = dir.path().join("export/metadata.yaml");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let root = RecordingPaste(calls.clone());
        let policy = OutputPolicy {
            no_paste: true,
            save: Some(save.clone()),
            output_yaml: Some(yaml.clone()),
            replace_chars: 4,
            realtime: false,
            timestamp: false,
        };
        let result = crate::transcription::TranscriptionResult {
            text: "chosen transcript".into(),
            metadata: Default::default(),
            diarization: None,
            segments: None,
        };

        let text = deliver(
            Ok(result),
            Delivery {
                policy: &policy,
                audio: &audio,
                provider: Provider::Mistral,
                model: "cached-model",
                cached: true,
                paste_root: &root,
                target_window: None,
                paste_timing: Default::default(),
                sink: &crate::telemetry::NoOpSink,
                t_stop: None,
                authoritative: false,
                paste_alert: None,
            },
        )
        .await
        .expect("deliver cached pick");

        assert_eq!(text, "chosen transcript");
        assert_eq!(std::fs::read(save).expect("saved audio"), b"recorded bytes");
        let metadata = std::fs::read_to_string(yaml).expect("exported metadata");
        assert!(metadata.contains("chosen transcript"));
        assert!(metadata.contains("original segment"));
        assert!(calls.lock().expect("paste calls").is_empty());
    }

    #[tokio::test]
    async fn cached_file_dictation_skips_capture_and_provider_with_real_pick() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("recording.ogg");
        std::fs::write(&audio, b"source audio").expect("audio fixture");
        crate::recording_cache::write_pick(&audio, "openai", "saved-model", false, "saved words")
            .expect("pick");
        let config: crate::config::Config = serde_yaml::from_str(&format!(
            "output_dir: {}\nproviders: {{}}\n",
            dir.path().display()
        ))
        .expect("configuration");
        let mut opts = test_opts();
        opts.input_audio_file = Some(audio.clone());
        opts.target_window = Some("unused".into());
        opts.no_paste = true;
        super::dictate_loaded(opts, config, super::StartTiming(std::time::Instant::now()))
            .await
            .expect("cached dictation");
        assert_eq!(
            std::fs::read(&audio).expect("audio retained"),
            b"source audio"
        );
        assert_eq!(
            crate::recording_cache::read_pick(&audio)
                .expect("pick retained")
                .3,
            "saved words"
        );
        assert_eq!(std::fs::read_dir(dir.path()).expect("entries").count(), 2);
    }

    struct RecordingStore {
        events: Arc<Mutex<Vec<String>>>,
        metadata: std::path::PathBuf,
    }

    impl super::DeliveryStore for RecordingStore {
        fn write_pick_if_absent(
            &self,
            _audio: &std::path::Path,
            _provider: Provider,
            _model: &str,
            _realtime: bool,
            text: &str,
        ) -> Result<(), TalkError> {
            self.events
                .lock()
                .expect("events")
                .push(format!("pick:{text}"));
            Ok(())
        }
        fn get(
            &self,
            _audio: &std::path::Path,
            _provider: Provider,
            _model: &str,
        ) -> Option<crate::transcription::TranscriptionResult> {
            None
        }
        fn store(
            &self,
            _audio: &std::path::Path,
            _provider: Provider,
            _model: &str,
            _realtime: bool,
            result: &crate::transcription::TranscriptionResult,
        ) -> Result<std::path::PathBuf, TalkError> {
            self.events
                .lock()
                .expect("events")
                .push(format!("store:{}", result.text));
            Ok(self.metadata.clone())
        }
        fn write_last_pointers(
            &self,
            _audio: &std::path::Path,
            _metadata: &std::path::Path,
        ) -> Result<(), TalkError> {
            self.events.lock().expect("events").push("pointers".into());
            Ok(())
        }
        fn rotate(&self) -> Result<(), TalkError> {
            self.events.lock().expect("events").push("rotate".into());
            Ok(())
        }
        fn write_last_paste_state(
            &self,
            _target: Option<&str>,
            text: &str,
        ) -> Result<(), TalkError> {
            self.events
                .lock()
                .expect("events")
                .push(format!("paste-state:{text}"));
            Ok(())
        }
    }

    #[tokio::test]
    async fn successful_default_delivery_persists_pick_metadata_and_export_in_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("cache.ogg");
        std::fs::write(&audio, b"audio").expect("audio fixture");
        let metadata = dir.path().join("sidecar.yaml");
        std::fs::write(&metadata, b"metadata bytes").expect("sidecar fixture");
        let exported = dir.path().join("nested/export.yaml");
        let events = Arc::new(Mutex::new(Vec::new()));
        let store = RecordingStore {
            events: events.clone(),
            metadata,
        };
        let policy = OutputPolicy {
            no_paste: true,
            save: None,
            output_yaml: Some(exported.clone()),
            replace_chars: 0,
            realtime: false,
            timestamp: false,
        };
        let root = RecordingPaste(Arc::new(Mutex::new(Vec::new())));
        let text = super::deliver_with(
            Ok(crate::transcription::TranscriptionResult {
                text: "  chosen words  ".into(),
                ..Default::default()
            }),
            Delivery {
                policy: &policy,
                audio: &audio,
                provider: Provider::OpenAI,
                model: "batch",
                cached: false,
                paste_root: &root,
                target_window: None,
                paste_timing: Default::default(),
                sink: &crate::telemetry::NoOpSink,
                t_stop: None,
                authoritative: true,
                paste_alert: None,
            },
            &store,
            |_text, _delete| async { panic!("no paste requested") },
        )
        .await
        .expect("delivery");
        assert_eq!(text, "chosen words");
        assert_eq!(
            *events.lock().expect("events"),
            [
                "pick:chosen words",
                "store:chosen words",
                "pointers",
                "rotate"
            ]
        );
        assert_eq!(std::fs::read(exported).expect("export"), b"metadata bytes");
    }

    #[tokio::test]
    async fn explicit_model_delivery_does_not_authorize_pick_but_pastes_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("cache.ogg");
        std::fs::write(&audio, b"audio").expect("audio fixture");
        let events = Arc::new(Mutex::new(Vec::new()));
        let store = RecordingStore {
            events: events.clone(),
            metadata: dir.path().join("unused.yaml"),
        };
        let policy = OutputPolicy {
            no_paste: false,
            save: None,
            output_yaml: None,
            replace_chars: 7,
            realtime: false,
            timestamp: false,
        };
        let root = RecordingPaste(Arc::new(Mutex::new(Vec::new())));
        let pasted = Arc::new(Mutex::new(Vec::new()));
        let paste_calls = pasted.clone();
        let text = super::deliver_with(
            Ok(crate::transcription::TranscriptionResult {
                text: "selected".into(),
                ..Default::default()
            }),
            Delivery {
                policy: &policy,
                audio: &audio,
                provider: Provider::OpenAI,
                model: "explicit",
                cached: false,
                paste_root: &root,
                target_window: None,
                paste_timing: Default::default(),
                sink: &crate::telemetry::NoOpSink,
                t_stop: None,
                authoritative: false,
                paste_alert: None,
            },
            &store,
            move |text, delete| async move {
                paste_calls
                    .lock()
                    .expect("paste calls")
                    .push((text, delete));
                Ok(())
            },
        )
        .await
        .expect("delivery");
        assert_eq!(text, "selected");
        assert_eq!(
            *pasted.lock().expect("paste calls"),
            [("selected".into(), 7)]
        );
        assert_eq!(
            *events.lock().expect("events"),
            [
                "store:selected",
                "pointers",
                "rotate",
                "paste-state:selected"
            ]
        );
    }

    #[tokio::test]
    async fn metadata_export_failure_is_returned_after_recording_is_persisted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("cache.ogg");
        std::fs::write(&audio, b"audio").expect("audio fixture");
        let metadata = dir.path().join("sidecar.yaml");
        std::fs::write(&metadata, b"metadata").expect("sidecar fixture");
        let blocked = dir.path().join("not-a-directory");
        std::fs::write(&blocked, b"file").expect("blocked export parent");
        let events = Arc::new(Mutex::new(Vec::new()));
        let store = RecordingStore {
            events: events.clone(),
            metadata,
        };
        let policy = OutputPolicy {
            no_paste: false,
            save: None,
            output_yaml: Some(blocked.join("export.yaml")),
            replace_chars: 0,
            realtime: false,
            timestamp: false,
        };
        let root = RecordingPaste(Arc::new(Mutex::new(Vec::new())));
        let error = super::deliver_with(
            Ok(crate::transcription::TranscriptionResult {
                text: "words".into(),
                ..Default::default()
            }),
            Delivery {
                policy: &policy,
                audio: &audio,
                provider: Provider::Mistral,
                model: "batch",
                cached: false,
                paste_root: &root,
                target_window: None,
                paste_timing: Default::default(),
                sink: &crate::telemetry::NoOpSink,
                t_stop: None,
                authoritative: false,
                paste_alert: None,
            },
            &store,
            |_text, _delete| async { panic!("failed export must not paste") },
        )
        .await
        .expect_err("export error");
        assert!(error.to_string().contains("not-a-directory"), "{error}");
        assert_eq!(
            *events.lock().expect("events"),
            ["store:words", "pointers", "rotate"]
        );
        assert_eq!(std::fs::read(audio).expect("audio retained"), b"audio");
    }

    #[tokio::test]
    async fn realtime_final_text_is_cached_without_a_second_paste() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("cache.ogg");
        std::fs::write(&audio, b"audio").expect("audio fixture");
        let events = Arc::new(Mutex::new(Vec::new()));
        let store = RecordingStore {
            events: events.clone(),
            metadata: dir.path().join("unused.yaml"),
        };
        let policy = OutputPolicy {
            no_paste: false,
            save: None,
            output_yaml: None,
            replace_chars: 3,
            realtime: true,
            timestamp: false,
        };
        let root = RecordingPaste(Arc::new(Mutex::new(Vec::new())));
        let text = super::deliver_with(
            Ok(crate::transcription::TranscriptionResult {
                text: "live final".into(),
                ..Default::default()
            }),
            Delivery {
                policy: &policy,
                audio: &audio,
                provider: Provider::OpenAI,
                model: "live",
                cached: false,
                paste_root: &root,
                target_window: None,
                paste_timing: Default::default(),
                sink: &crate::telemetry::NoOpSink,
                t_stop: None,
                authoritative: true,
                paste_alert: None,
            },
            &store,
            |_text, _delete| async { panic!("final realtime text must not be pasted twice") },
        )
        .await
        .expect("final transcript");
        assert_eq!(text, "live final");
        assert_eq!(
            *events.lock().expect("events"),
            ["pick:live final", "store:live final", "pointers", "rotate"]
        );
    }

    #[tokio::test]
    async fn failed_save_is_an_error_even_for_successful_transcription() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("cached.ogg");
        std::fs::write(&audio, b"recorded bytes").expect("audio fixture");
        let blocked = dir.path().join("blocked");
        std::fs::write(&blocked, b"file").expect("block parent directory");
        let policy = OutputPolicy {
            no_paste: true,
            save: Some(blocked.join("audio.ogg")),
            output_yaml: None,
            replace_chars: 0,
            realtime: false,
            timestamp: false,
        };
        let root = RecordingPaste(Arc::new(Mutex::new(Vec::new())));
        let result = deliver(
            Ok(Default::default()),
            Delivery {
                policy: &policy,
                audio: &audio,
                provider: Provider::Mistral,
                model: "test",
                cached: true,
                paste_root: &root,
                target_window: None,
                paste_timing: Default::default(),
                sink: &crate::telemetry::NoOpSink,
                t_stop: None,
                authoritative: false,
                paste_alert: None,
            },
        )
        .await
        .expect_err("save error must propagate");
        assert!(result.to_string().contains("blocked"), "{result}");
    }
}
