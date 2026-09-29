//! Picker UI for selecting among multiple transcription candidates.
//!
//! Extracted from `dictate.rs` — contains the GTK4 picker window,
//! X11 window centering/raising helpers, and WAV PCM reading used
//! by the picker's realtime transcription support.

use crate::config::{Config, Provider};
use crate::error::TalkError;
use crate::transcription::{self, RealtimeTranscriber, TranscriptSegment, TranscriptionMetadata};
use std::path::PathBuf;

use super::backend::{run_realtime_transcription, spawn_transcription};
use crate::dictate::models::resolve_provider;
use crate::record::player::WavPlayer;

/// Window title used for the picker — also used for single-instance
/// detection via `x11_centre_and_raise`.
pub(super) const PICKER_TITLE: &str = "talk-rs: select transcription";

/// Candidate transcription result sent from async tasks to the GTK
/// thread via a `std::sync::mpsc` channel.
pub(super) struct PickerCandidate {
    provider: Provider,
    model: String,
    text: String,
    segments: Option<Vec<TranscriptSegment>>,
    /// Transcription metadata (latency, token usage, audio duration, etc.).
    metadata: TranscriptionMetadata,
    /// When set, the candidate failed and this is the error message.
    error: Option<String>,
    /// Whether this candidate came from a realtime (streaming) transcriber.
    streaming: bool,
}

impl PickerCandidate {
    /// Build a successful transcription candidate.
    pub(super) fn success(
        provider: Provider,
        model: String,
        text: String,
        streaming: bool,
        segments: Option<Vec<TranscriptSegment>>,
        metadata: TranscriptionMetadata,
    ) -> Self {
        Self {
            provider,
            model,
            text,
            segments,
            metadata,
            error: None,
            streaming,
        }
    }

    /// Build a failed transcription candidate.
    pub(super) fn error(
        provider: Provider,
        model: String,
        message: String,
        streaming: bool,
    ) -> Self {
        Self {
            provider,
            model,
            text: String::new(),
            segments: None,
            metadata: TranscriptionMetadata::default(),
            error: Some(message),
            streaming,
        }
    }
}

/// Messages sent from async tasks to the GTK poll loop.
pub(super) enum PickerMessage {
    /// A transcription result (success or error).
    Candidate(Box<PickerCandidate>),
    /// All initial one-shot transcription tasks have completed.
    /// The poll loop should mark any remaining one-shot spinners as
    /// failed but keep running to receive retry results.
    InitialOneShotDone,
    /// Incremental text update from a realtime transcriber.
    /// Contains the full accumulated text so GTK just replaces the label.
    StreamUpdate {
        provider: Provider,
        model: String,
        accumulated_text: String,
    },
    /// Live status update for an in-flight one-shot candidate.
    ///
    /// Emitted by [`super::backend::PickerStatusSink`] while the
    /// HTTP request is in progress.  The picker UI renders
    /// `status_text` in italic at reduced opacity in the row's
    /// transcription text area, replacing it with the final
    /// transcript when the matching [`Self::Candidate`] arrives.
    ///
    /// String values are short, lowercase, ellipsis-suffixed phase
    /// labels (e.g. `"connecting…"`, `"uploading…"`, `"waiting for
    /// server…"`, `"transcribing…"`, `"retry 2/5…"`,
    /// `"failed; retrying…"`) — the consumer treats the string as
    /// opaque and renders it verbatim.
    CandidateStatus {
        provider: Provider,
        model: String,
        /// Whether this status is for the realtime row
        /// or the one-shot row.  Required for picker row disambiguation
        /// when both rows exist for the same (provider, model) pair
        /// — without this the status would race to whichever row
        /// `position()` finds first.
        streaming: bool,
        status_text: String,
    },
}

/// Result of the picker: selected provider/model/text and whether it
/// was the pre-populated cached entry (in which case the caller should
/// skip pasting since the text is already in the target window).
pub(super) struct PickerSelection {
    pub(super) text: String,
    pub(super) is_cached: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PickerNavigation {
    Previous,
    Next,
}

pub(super) enum PickerOutcome {
    Selected(PickerSelection),
    Navigate(PickerNavigation),
    Cancelled,
}

#[derive(Clone, Copy)]
pub(super) struct PickerNavigationAvailability {
    pub(super) previous: bool,
    pub(super) next: bool,
}

pub(super) struct PickerUiInput {
    pub(super) transcribers: Vec<(Provider, String)>,
    pub(super) audio_path: PathBuf,
    pub(super) cached_entries: Vec<(Provider, String, String, bool, bool)>,
    pub(super) config: std::sync::Arc<Config>,
    pub(super) realtime_transcribers: Vec<(Provider, String, Box<dyn RealtimeTranscriber>)>,
    pub(super) deferred_candidates: Vec<(Provider, String, bool)>,
    pub(super) navigation: PickerNavigationAvailability,
}

#[derive(Clone)]
struct RealtimeSamples {
    audio_path: PathBuf,
    samples: std::sync::Arc<tokio::sync::OnceCell<Option<std::sync::Arc<Vec<i16>>>>>,
}

impl RealtimeSamples {
    fn new(audio_path: PathBuf) -> Self {
        Self {
            audio_path,
            samples: std::sync::Arc::new(tokio::sync::OnceCell::new()),
        }
    }

    async fn get(&self) -> Option<std::sync::Arc<Vec<i16>>> {
        let audio_path = self.audio_path.clone();
        self.samples
            .get_or_init(|| async move {
                #[cfg(test)]
                GTK_TEST_REALTIME_DECODES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                match tokio::task::spawn_blocking(move || {
                    crate::record::audio::read_audio_as_i16(&audio_path)
                })
                .await
                {
                    Ok(Ok(samples)) => Some(std::sync::Arc::new(samples)),
                    Ok(Err(error)) => {
                        log::warn!("failed to read audio for realtime: {}", error);
                        None
                    }
                    Err(error) => {
                        log::warn!("failed to read audio for realtime: {}", error);
                        None
                    }
                }
            })
            .await
            .clone()
    }
}

enum PickerExit {
    Selected(usize),
    Navigate(PickerNavigation),
    Cancelled,
}

struct SelectionSaveCtx {
    audio_path: PathBuf,
    dirty: bool,
    current: Option<(Provider, String, bool)>,
}

fn persist_selection_with<F>(
    ctx: &mut SelectionSaveCtx,
    text: &str,
    write: F,
) -> Result<(), TalkError>
where
    F: FnOnce(&std::path::Path, Provider, &str, bool, &str) -> Result<(), TalkError>,
{
    if !ctx.dirty {
        return Ok(());
    }
    let (provider, model, streaming) = ctx.current.clone().ok_or_else(|| {
        TalkError::Config("cannot save picker edits without a selected candidate".to_string())
    })?;
    write(&ctx.audio_path, provider, &model, streaming, text)?;
    ctx.dirty = false;
    Ok(())
}

fn persist_selection(ctx: &mut SelectionSaveCtx, text: &str) -> Result<(), TalkError> {
    #[cfg(test)]
    if GTK_TEST_SAVE_FAILURES
        .fetch_update(
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
            |remaining| remaining.checked_sub(1),
        )
        .is_ok()
    {
        return Err(TalkError::Config(
            "injected picker save failure".to_string(),
        ));
    }
    persist_selection_with(ctx, text, |path, provider, model, streaming, text| {
        crate::recording_cache::write_pick(path, &provider.to_string(), model, streaming, text)
    })
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) enum GtkTestExit {
    Navigate(PickerNavigation),
    RawNextBurst(usize),
    Wait,
    Confirm,
}

#[cfg(test)]
pub(super) struct GtkTestAction {
    pub(super) expected_provider: Provider,
    pub(super) expected_model: String,
    pub(super) expected_streaming: bool,
    pub(super) expected_text: String,
    pub(super) previous_sensitive: bool,
    pub(super) next_sensitive: bool,
    pub(super) edit_text: Option<String>,
    pub(super) fail_first_save: bool,
    pub(super) wait: Option<GtkTestWait>,
    pub(super) exit: GtkTestExit,
}

#[cfg(test)]
pub(super) struct GtkTestWait {
    pub(super) click_action: bool,
    pub(super) expected_text: String,
    pub(super) expected_action_label_before: Option<String>,
    pub(super) expected_action_label_after: Option<String>,
    pub(super) request_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub(super) expected_requests_before: Option<usize>,
    pub(super) expected_requests_after: usize,
}

#[cfg(test)]
static GTK_TEST_ACTIONS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::VecDeque<GtkTestAction>>,
> = std::sync::OnceLock::new();
#[cfg(test)]
static GTK_TEST_SAVE_FAILURES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_WINDOWS_CREATED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_WINDOWS_DESTROYED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_RECORDS_SEEN: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_STALE_SOURCE_TICKS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_ACTIVE_RECORD_GENERATION: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_WINDOW_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_WINDOW_ID_CHANGES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_PREVIOUS_BUTTON_ID: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_PREVIOUS_BUTTON_ID_CHANGES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_NEXT_BUTTON_ID: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_NEXT_BUTTON_ID_CHANGES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_UNMAPPED_NAVIGATION_BUTTONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_INVISIBLE_RECORDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_HIDDEN_TRANSITIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_REALTIME_DECODES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GTK_TEST_SESSION_FINALIZING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
type GtkTestInputThread = std::thread::JoinHandle<Result<(), String>>;
#[cfg(test)]
static GTK_TEST_INPUT_THREAD: std::sync::OnceLock<std::sync::Mutex<Option<GtkTestInputThread>>> =
    std::sync::OnceLock::new();
#[cfg(test)]
static GTK_TEST_INPUT_ERROR: std::sync::OnceLock<std::sync::Mutex<Option<String>>> =
    std::sync::OnceLock::new();

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(super) struct GtkTestMetrics {
    pub(super) remaining_actions: usize,
    pub(super) windows_created: usize,
    pub(super) windows_destroyed: usize,
    pub(super) records_seen: usize,
    pub(super) window_id_changes: usize,
    pub(super) previous_button_id_changes: usize,
    pub(super) next_button_id_changes: usize,
    pub(super) unmapped_navigation_buttons: usize,
    pub(super) invisible_records: usize,
    pub(super) hidden_transitions: usize,
    pub(super) stale_source_ticks: usize,
    pub(super) realtime_decodes: usize,
}

#[cfg(test)]
pub(super) fn install_gtk_test_actions(actions: Vec<GtkTestAction>) {
    *GTK_TEST_ACTIONS
        .get_or_init(|| std::sync::Mutex::new(std::collections::VecDeque::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = actions.into();
    GTK_TEST_SAVE_FAILURES.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_WINDOWS_CREATED.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_WINDOWS_DESTROYED.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_RECORDS_SEEN.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_STALE_SOURCE_TICKS.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_ACTIVE_RECORD_GENERATION.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_WINDOW_ID.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_WINDOW_ID_CHANGES.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_PREVIOUS_BUTTON_ID.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_PREVIOUS_BUTTON_ID_CHANGES.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_NEXT_BUTTON_ID.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_NEXT_BUTTON_ID_CHANGES.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_UNMAPPED_NAVIGATION_BUTTONS.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_INVISIBLE_RECORDS.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_HIDDEN_TRANSITIONS.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_REALTIME_DECODES.store(0, std::sync::atomic::Ordering::SeqCst);
    GTK_TEST_SESSION_FINALIZING.store(false, std::sync::atomic::Ordering::SeqCst);
    *GTK_TEST_INPUT_ERROR
        .get_or_init(|| std::sync::Mutex::new(None))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

#[cfg(test)]
pub(super) fn finish_gtk_test_input() -> Result<(), String> {
    let handle = GTK_TEST_INPUT_THREAD
        .get_or_init(|| std::sync::Mutex::new(None))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    if let Some(handle) = handle {
        handle
            .join()
            .map_err(|_| "raw picker input thread panicked".to_string())??;
    }
    if let Some(error) = GTK_TEST_INPUT_ERROR
        .get_or_init(|| std::sync::Mutex::new(None))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
    {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn gtk_test_metrics() -> GtkTestMetrics {
    let remaining = GTK_TEST_ACTIONS
        .get_or_init(|| std::sync::Mutex::new(std::collections::VecDeque::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    GtkTestMetrics {
        remaining_actions: remaining,
        windows_created: GTK_TEST_WINDOWS_CREATED.load(std::sync::atomic::Ordering::SeqCst),
        windows_destroyed: GTK_TEST_WINDOWS_DESTROYED.load(std::sync::atomic::Ordering::SeqCst),
        records_seen: GTK_TEST_RECORDS_SEEN.load(std::sync::atomic::Ordering::SeqCst),
        window_id_changes: GTK_TEST_WINDOW_ID_CHANGES.load(std::sync::atomic::Ordering::SeqCst),
        previous_button_id_changes: GTK_TEST_PREVIOUS_BUTTON_ID_CHANGES
            .load(std::sync::atomic::Ordering::SeqCst),
        next_button_id_changes: GTK_TEST_NEXT_BUTTON_ID_CHANGES
            .load(std::sync::atomic::Ordering::SeqCst),
        unmapped_navigation_buttons: GTK_TEST_UNMAPPED_NAVIGATION_BUTTONS
            .load(std::sync::atomic::Ordering::SeqCst),
        invisible_records: GTK_TEST_INVISIBLE_RECORDS.load(std::sync::atomic::Ordering::SeqCst),
        hidden_transitions: GTK_TEST_HIDDEN_TRANSITIONS.load(std::sync::atomic::Ordering::SeqCst),
        stale_source_ticks: GTK_TEST_STALE_SOURCE_TICKS.load(std::sync::atomic::Ordering::SeqCst),
        realtime_decodes: GTK_TEST_REALTIME_DECODES.load(std::sync::atomic::Ordering::SeqCst),
    }
}

#[cfg(test)]
fn register_gtk_test_window(window: &gtk4::Window) {
    use gtk4::prelude::*;

    GTK_TEST_WINDOWS_CREATED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    window.connect_visible_notify(|window| {
        if !window.is_visible()
            && !GTK_TEST_SESSION_FINALIZING.load(std::sync::atomic::Ordering::SeqCst)
        {
            GTK_TEST_HIDDEN_TRANSITIONS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    });
}

#[cfg(test)]
fn begin_gtk_test_record() -> usize {
    GTK_TEST_ACTIVE_RECORD_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1
}

#[cfg(test)]
fn mark_gtk_test_session_finalizing() {
    GTK_TEST_SESSION_FINALIZING.store(true, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
fn note_gtk_test_source_tick(generation: usize) {
    if GTK_TEST_ACTIVE_RECORD_GENERATION.load(std::sync::atomic::Ordering::SeqCst) != generation {
        GTK_TEST_STALE_SOURCE_TICKS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
fn note_gtk_test_widget_identity<T: gtk4::prelude::ObjectType>(
    widget: &T,
    first_id: &std::sync::atomic::AtomicUsize,
    changes: &std::sync::atomic::AtomicUsize,
) {
    let widget_id = widget.as_ptr() as usize;
    let initial_id = first_id
        .compare_exchange(
            0,
            widget_id,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        )
        .unwrap_or_else(|existing| existing);
    if initial_id != 0 && initial_id != widget_id {
        changes.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
fn start_gtk_test_raw_next_burst(
    window: &gtk4::Window,
    next: &gtk4::Button,
    click_count: usize,
) -> Result<(), String> {
    use gtk4::prelude::*;
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::ConnectionExt;

    let surface = window
        .surface()
        .ok_or_else(|| "picker window has no GDK surface".to_string())?;
    let xid = surface
        .downcast_ref::<gdk4_x11::X11Surface>()
        .ok_or_else(|| "picker test requires the GDK X11 backend".to_string())?
        .xid() as u32;
    let point = next
        .compute_point(
            window,
            &gtk4::graphene::Point::new(next.width() as f32 / 2.0, next.height() as f32 / 2.0),
        )
        .ok_or_else(|| "could not locate next button in picker window".to_string())?;
    let (connection, screen_num) = x11rb::connect(None)
        .map_err(|error| format!("failed to connect to isolated X11 display: {error}"))?;
    let root = connection.setup().roots[screen_num].root;
    let translated = connection
        .translate_coordinates(xid, root, point.x() as i16, point.y() as i16)
        .map_err(|error| format!("failed to request picker coordinate translation: {error}"))?
        .reply()
        .map_err(|error| format!("failed to translate picker coordinates: {error}"))?;
    let root_x = translated.dst_x.to_string();
    let root_y = translated.dst_y.to_string();

    let handle = std::thread::spawn(move || {
        let mut command = std::process::Command::new("xdotool");
        command
            .arg("mousemove")
            .arg(root_x)
            .arg(root_y)
            .arg("sleep")
            .arg("0.05");
        for click_index in 0..click_count {
            if click_index > 0 {
                command.arg("sleep").arg("0.15");
            }
            command.arg("mousedown").arg("1").arg("mouseup").arg("1");
        }
        // On the broken implementation clicks after the first navigation
        // miss the replacement button. Escape bounds the test instead of
        // leaving its per-record main loop blocked forever.
        let status = command
            .arg("sleep")
            .arg("0.5")
            .arg("key")
            .arg("Escape")
            .status()
            .map_err(|error| format!("failed to launch xdotool: {error}"))?;
        if !status.success() {
            return Err(format!("xdotool exited with {status}"));
        }
        Ok(())
    });
    *GTK_TEST_INPUT_THREAD
        .get_or_init(|| std::sync::Mutex::new(None))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(handle);
    Ok(())
}

#[cfg(test)]
struct GtkTestWindowRefs<'a> {
    audio_path: &'a std::path::Path,
    previous: &'a gtk4::Button,
    next: &'a gtk4::Button,
    buffer: &'a gtk4::TextBuffer,
    error_label: &'a gtk4::Label,
    list: &'a gtk4::ListBox,
    action_buttons: &'a std::rc::Rc<std::cell::RefCell<Vec<gtk4::Button>>>,
    selection_ctx: &'a std::rc::Rc<std::cell::RefCell<SelectionSaveCtx>>,
    window: &'a gtk4::Window,
}

#[cfg(test)]
struct GtkTestFinishRefs<'a> {
    audio_path: &'a std::path::Path,
    previous: &'a gtk4::Button,
    next: &'a gtk4::Button,
    buffer: &'a gtk4::TextBuffer,
    error_label: &'a gtk4::Label,
    list: &'a gtk4::ListBox,
    window: &'a gtk4::Window,
}

#[cfg(test)]
fn finish_gtk_test_action(action: &GtkTestAction, refs: GtkTestFinishRefs<'_>) {
    use gtk4::prelude::*;

    if let Some(text) = action.edit_text.as_ref() {
        refs.buffer.set_text(text);
    }
    let emit_exit = || match action.exit {
        GtkTestExit::Navigate(PickerNavigation::Previous) => {
            refs.previous.emit_by_name::<()>("clicked", &[])
        }
        GtkTestExit::Navigate(PickerNavigation::Next) => {
            refs.next.emit_by_name::<()>("clicked", &[])
        }
        GtkTestExit::RawNextBurst(click_count) => {
            if let Err(error) = start_gtk_test_raw_next_burst(refs.window, refs.next, click_count) {
                *GTK_TEST_INPUT_ERROR
                    .get_or_init(|| std::sync::Mutex::new(None))
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(error);
                refs.window.close();
            }
        }
        GtkTestExit::Wait => {}
        GtkTestExit::Confirm => {
            let row = refs
                .list
                .selected_row()
                .expect("test picker has selected row");
            refs.list.emit_by_name::<()>("row-activated", &[&row]);
        }
    };

    if action.fail_first_save {
        GTK_TEST_SAVE_FAILURES.store(1, std::sync::atomic::Ordering::SeqCst);
        emit_exit();
        assert!(refs.error_label.is_visible());
        assert_eq!(
            refs.buffer
                .text(&refs.buffer.start_iter(), &refs.buffer.end_iter(), false)
                .as_str(),
            action.edit_text.as_deref().unwrap_or(&action.expected_text)
        );
        assert_eq!(
            crate::recording_cache::read_pick(refs.audio_path).map(|(_, _, _, text)| text),
            Some(action.expected_text.clone())
        );
        emit_exit();
    } else {
        emit_exit();
    }
}

#[cfg(test)]
fn drive_gtk_test_window(refs: GtkTestWindowRefs<'_>) {
    use gtk4::prelude::*;

    let action = GTK_TEST_ACTIONS
        .get_or_init(|| std::sync::Mutex::new(std::collections::VecDeque::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .pop_front();
    let Some(action) = action else {
        return;
    };
    let audio_path = refs.audio_path.to_path_buf();
    let previous = refs.previous.clone();
    let next = refs.next.clone();
    let buffer = refs.buffer.clone();
    let error_label = refs.error_label.clone();
    let list = refs.list.clone();
    let action_buttons = std::rc::Rc::clone(refs.action_buttons);
    let selection_ctx = std::rc::Rc::clone(refs.selection_ctx);
    let window = refs.window.clone();
    gtk4::glib::idle_add_local_once(move || {
        GTK_TEST_RECORDS_SEEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let window_id = window.as_ptr() as usize;
        let first_id = GTK_TEST_WINDOW_ID
            .compare_exchange(
                0,
                window_id,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            )
            .unwrap_or_else(|existing| existing);
        if first_id != 0 && first_id != window_id {
            GTK_TEST_WINDOW_ID_CHANGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        note_gtk_test_widget_identity(
            &previous,
            &GTK_TEST_PREVIOUS_BUTTON_ID,
            &GTK_TEST_PREVIOUS_BUTTON_ID_CHANGES,
        );
        note_gtk_test_widget_identity(
            &next,
            &GTK_TEST_NEXT_BUTTON_ID,
            &GTK_TEST_NEXT_BUTTON_ID_CHANGES,
        );
        if !previous.is_mapped() || !next.is_mapped() {
            GTK_TEST_UNMAPPED_NAVIGATION_BUTTONS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        if !window.is_visible() || !window.is_mapped() {
            GTK_TEST_INVISIBLE_RECORDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        assert_eq!(previous.is_sensitive(), action.previous_sensitive);
        assert_eq!(next.is_sensitive(), action.next_sensitive);
        assert_eq!(
            selection_ctx.borrow().current.as_ref(),
            Some(&(
                action.expected_provider,
                action.expected_model.clone(),
                action.expected_streaming,
            ))
        );
        let displayed_text = buffer
            .text(&buffer.start_iter(), &buffer.end_iter(), false)
            .to_string();
        if let Some(wait) = action.wait.as_ref().filter(|wait| !wait.click_action) {
            assert!(
                displayed_text == action.expected_text || displayed_text == wait.expected_text,
                "unexpected picker text before auto-result: {displayed_text:?}"
            );
        } else {
            assert_eq!(displayed_text, action.expected_text);
        }
        if let Some(wait) = action.wait.as_ref() {
            let row = list.selected_row().expect("test picker has selected row");
            let row_index = row.index() as usize;
            let action_button = action_buttons
                .borrow()
                .get(row_index)
                .cloned()
                .expect("selected picker row has an action button");
            if let Some(expected) = wait.expected_action_label_before.as_deref() {
                assert_eq!(action_button.label().as_deref(), Some(expected));
            }
            if let Some(expected) = wait.expected_requests_before {
                assert_eq!(
                    wait.request_count.load(std::sync::atomic::Ordering::SeqCst),
                    expected
                );
            }
            if wait.click_action {
                action_button.emit_by_name::<()>("clicked", &[]);
            }

            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let request_count = std::sync::Arc::clone(&wait.request_count);
            let expected_requests = wait.expected_requests_after;
            let expected_text = wait.expected_text.clone();
            let expected_label = wait.expected_action_label_after.clone();
            gtk4::glib::timeout_add_local(std::time::Duration::from_millis(10), move || {
                let text_matches = buffer
                    .text(&buffer.start_iter(), &buffer.end_iter(), false)
                    .as_str()
                    == expected_text;
                let requests_match =
                    request_count.load(std::sync::atomic::Ordering::SeqCst) == expected_requests;
                let label_matches = expected_label
                    .as_deref()
                    .is_none_or(|expected| action_button.label().as_deref() == Some(expected));
                if text_matches && requests_match && label_matches {
                    finish_gtk_test_action(
                        &action,
                        GtkTestFinishRefs {
                            audio_path: &audio_path,
                            previous: &previous,
                            next: &next,
                            buffer: &buffer,
                            error_label: &error_label,
                            list: &list,
                            window: &window,
                        },
                    );
                    gtk4::glib::ControlFlow::Break
                } else if std::time::Instant::now() >= deadline {
                    panic!(
                        "picker test timed out: text={:?}, requests={}, action={:?}",
                        buffer
                            .text(&buffer.start_iter(), &buffer.end_iter(), false)
                            .as_str(),
                        request_count.load(std::sync::atomic::Ordering::SeqCst),
                        action_button.label(),
                    );
                } else {
                    gtk4::glib::ControlFlow::Continue
                }
            });
        } else {
            finish_gtk_test_action(
                &action,
                GtkTestFinishRefs {
                    audio_path: &audio_path,
                    previous: &previous,
                    next: &next,
                    buffer: &buffer,
                    error_label: &error_label,
                    list: &list,
                    window: &window,
                },
            );
        }
    });
}

fn picker_css(theme: &crate::gtk_theme::ThemeColors) -> String {
    theme.base_css(&format!(
        ".transcript {{ font-family: monospace; }} \
         .error {{ font-style: italic; color: alpha({err}, 0.8); }} \
         .retry-btn {{ min-width: 0; min-height: 0; padding: 2px 6px; font-size: 14px; }} \
         .action-btn {{ \
             min-width: 28px; min-height: 28px; \
             padding: 0; margin: 0; \
             font-family: monospace; font-weight: bold; font-size: 14px; \
         }} \
         .play-btn {{ min-width: 32px; min-height: 32px; padding: 0; font-size: 16px; }} \
         .waterfall {{ background-color: black; border-radius: 0.25em; }} \
         .copy-btn {{ min-width: 32px; min-height: 32px; padding: 0; font-size: 16px; }} \
         .editor-view {{ font-family: monospace; }} \
         textview.editor-view text {{ background-color: transparent; }} \
        ",
        err = theme.error,
    ))
}

/// Initialize GTK and install picker CSS once for the whole record session.
pub(super) fn initialize_picker_gtk() -> Result<(), TalkError> {
    gtk4::init()
        .map_err(|error| TalkError::Config(format!("failed to initialize GTK: {error}")))?;
    let theme = crate::gtk_theme::ThemeColors::resolve();
    crate::gtk_theme::load_css(&picker_css(&theme));
    Ok(())
}

type PickerNavigationCallback = std::rc::Rc<dyn Fn(PickerNavigation)>;

pub(super) struct PickerSession {
    window: gtk4::Window,
    close_button: gtk4::Button,
    waterfall_slot: gtk4::Box,
    previous_button: gtk4::Button,
    next_button: gtk4::Button,
    controls_slot: gtk4::Box,
    content_slot: gtk4::Box,
    navigation_callback: std::rc::Rc<std::cell::RefCell<Option<PickerNavigationCallback>>>,
    presented: std::cell::Cell<bool>,
    destroyed: std::cell::Cell<bool>,
}

impl PickerSession {
    pub(super) fn new() -> Self {
        use gtk4::prelude::*;

        let window = gtk4::Window::builder()
            .title(PICKER_TITLE)
            .default_width(900)
            .default_height(500)
            .decorated(false)
            .resizable(true)
            .build();
        window.set_size_request(400, 250);
        crate::gtk_theme::install_edge_resize(&window);

        let root = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
        root.set_margin_top(6);
        root.set_margin_bottom(6);
        root.set_margin_start(6);
        root.set_margin_end(6);

        let (title_bar, close_button) = crate::gtk_theme::build_title_bar();
        root.append(&title_bar);

        let play_bar = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
        play_bar.set_margin_start(4);
        play_bar.set_margin_end(4);
        play_bar.set_margin_bottom(4);

        let waterfall_slot = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
        waterfall_slot.set_hexpand(true);
        play_bar.append(&waterfall_slot);

        let previous_button = gtk4::Button::from_icon_name("go-previous-symbolic");
        previous_button.set_tooltip_text(Some("Previous recording (newer)"));
        previous_button.add_css_class("play-btn");
        previous_button.set_widget_name("picker-previous-record");
        play_bar.append(&previous_button);

        let next_button = gtk4::Button::from_icon_name("go-next-symbolic");
        next_button.set_tooltip_text(Some("Next recording (older)"));
        next_button.add_css_class("play-btn");
        next_button.set_widget_name("picker-next-record");
        play_bar.append(&next_button);

        let controls_slot = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
        play_bar.append(&controls_slot);
        root.append(&play_bar);

        let content_slot = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
        content_slot.set_vexpand(true);
        root.append(&content_slot);

        let handle = gtk4::WindowHandle::new();
        handle.set_child(Some(&root));
        window.set_child(Some(&handle));

        let navigation_callback: std::rc::Rc<std::cell::RefCell<Option<PickerNavigationCallback>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));
        {
            let callback = std::rc::Rc::clone(&navigation_callback);
            previous_button.connect_clicked(move |_| {
                if let Some(callback) = callback.borrow().as_ref() {
                    callback(PickerNavigation::Previous);
                }
            });
        }
        {
            let callback = std::rc::Rc::clone(&navigation_callback);
            next_button.connect_clicked(move |_| {
                if let Some(callback) = callback.borrow().as_ref() {
                    callback(PickerNavigation::Next);
                }
            });
        }
        #[cfg(test)]
        register_gtk_test_window(&window);
        Self {
            window,
            close_button,
            waterfall_slot,
            previous_button,
            next_button,
            controls_slot,
            content_slot,
            navigation_callback,
            presented: std::cell::Cell::new(false),
            destroyed: std::cell::Cell::new(false),
        }
    }

    fn mount(
        &self,
        waterfall: &gtk4::Widget,
        controls: &gtk4::Widget,
        content: &gtk4::Widget,
        navigation: PickerNavigationAvailability,
        navigation_callback: PickerNavigationCallback,
    ) {
        use gtk4::prelude::*;

        fn replace_child(slot: &gtk4::Box, child: &gtk4::Widget) {
            while let Some(existing) = slot.first_child() {
                slot.remove(&existing);
            }
            slot.append(child);
        }

        self.previous_button.set_sensitive(navigation.previous);
        self.next_button.set_sensitive(navigation.next);
        *self.navigation_callback.borrow_mut() = Some(navigation_callback);
        replace_child(&self.waterfall_slot, waterfall);
        replace_child(&self.controls_slot, controls);
        replace_child(&self.content_slot, content);
        if !self.presented.replace(true) {
            crate::gtk_theme::present_centred(&self.window);
        }
    }

    fn clear_navigation_callback(&self) {
        self.navigation_callback.borrow_mut().take();
    }

    pub(super) fn destroy(&self) {
        use gtk4::prelude::*;

        if self.destroyed.replace(true) {
            return;
        }
        self.clear_navigation_callback();
        #[cfg(test)]
        mark_gtk_test_session_finalizing();
        self.window.set_child(None::<&gtk4::Widget>);
        self.window.destroy();
        #[cfg(test)]
        {
            let window_widget = self.window.clone().upcast::<gtk4::Widget>();
            assert!(
                !gtk4::Window::list_toplevels()
                    .iter()
                    .any(|widget| widget == &window_widget),
                "destroyed picker window remained in GTK toplevel registry"
            );
            GTK_TEST_WINDOWS_DESTROYED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

impl Drop for PickerSession {
    fn drop(&mut self) {
        self.destroy();
    }
}

/// Escape text for use in Pango markup.
fn escape_pango(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Build Pango markup showing an inline character-level diff.
///
/// Compares `reference` (text-area content) against `candidate`
/// (AI transcription).
/// - `Equal` chunks → plain escaped text
/// - `Delete` chunks (in reference, not in candidate) → red background + strikethrough
/// - `Insert` chunks (in candidate, not in reference) → green background
///
/// Text colour is left unchanged; only the background highlights the diff.
///
/// When `reference` is empty the candidate text is returned escaped
/// but unstyled.
fn diff_markup(reference: &str, candidate: &str, del_color: &str, ins_color: &str) -> String {
    if reference.is_empty() || candidate.is_empty() {
        return escape_pango(candidate);
    }
    let chunks = dissimilar::diff(reference, candidate);
    let mut markup = String::new();
    for chunk in &chunks {
        match chunk {
            dissimilar::Chunk::Equal(text) => {
                markup.push_str(&escape_pango(text));
            }
            dissimilar::Chunk::Delete(text) => {
                markup.push_str(&format!(
                    "<span strikethrough=\"true\" background=\"{}\" bgalpha=\"50%\">",
                    del_color
                ));
                markup.push_str(&escape_pango(text));
                markup.push_str("</span>");
            }
            dissimilar::Chunk::Insert(text) => {
                markup.push_str(&format!(
                    "<span background=\"{}\" bgalpha=\"50%\">",
                    ins_color
                ));
                markup.push_str(&escape_pango(text));
                markup.push_str("</span>");
            }
        }
    }
    markup
}

/// Show a GTK4 picker window that appears immediately with a spinner,
/// then progressively fills with transcription candidates as parallel
/// API calls complete.
///
/// The window is undecorated, centered on the active monitor, and
/// always-on-top.  Layout is a 3-column table: transcript (monospace,
/// wrapping) | provider | model.
///
/// `cached_entries` contains pre-populated results that are shown
/// immediately (no spinner, no API call).  Each tuple is
/// `(provider, model, text, is_primary, streaming)` where `is_primary`
/// marks the saved selection for pre-selection (not proof of paste)
/// (selecting it again skips re-pasting).
///
/// Returns a selection, navigation request, or cancellation.
pub(super) async fn pick_with_streaming_gtk(
    session: &PickerSession,
    input: PickerUiInput,
) -> Result<PickerOutcome, TalkError> {
    let PickerUiInput {
        mut transcribers,
        audio_path,
        mut cached_entries,
        config,
        mut realtime_transcribers,
        mut deferred_candidates,
        navigation,
    } = input;
    if transcribers.is_empty()
        && cached_entries.is_empty()
        && realtime_transcribers.is_empty()
        && deferred_candidates.is_empty()
    {
        return Ok(PickerOutcome::Cancelled);
    }

    // Stable display order: sort by (provider, model, streaming) so
    // the list looks identical every time the picker opens, regardless
    // of which rows happen to have cached results.  The primary key
    // puts the config default provider first (rank 0) ahead of the
    // others (alphabetical among rank 1).
    let default_provider = resolve_provider(None, config.as_ref());
    // Rank providers: config default first (rank 0), others alphabetical
    // (rank 1 + name).  Factored out so the same ordering is applied by
    // both the per-source sorts below and the unified sort inside the
    // GTK thread.
    fn provider_rank(p: Provider, default_provider: Provider) -> (u8, String) {
        if p == default_provider {
            (0, String::new())
        } else {
            (1, p.to_string())
        }
    }
    cached_entries.sort_by(|(pa, ma, _, _, sa), (pb, mb, _, _, sb)| {
        provider_rank(*pa, default_provider)
            .cmp(&provider_rank(*pb, default_provider))
            .then(ma.cmp(mb))
            .then(sa.cmp(sb))
    });
    transcribers.sort_by(|(pa, ma), (pb, mb)| {
        provider_rank(*pa, default_provider)
            .cmp(&provider_rank(*pb, default_provider))
            .then(ma.cmp(mb))
    });
    realtime_transcribers.sort_by(|(pa, ma, _), (pb, mb, _)| {
        provider_rank(*pa, default_provider)
            .cmp(&provider_rank(*pb, default_provider))
            .then(ma.cmp(mb))
    });
    deferred_candidates.sort_by(|(pa, ma, sa), (pb, mb, sb)| {
        provider_rank(*pa, default_provider)
            .cmp(&provider_rank(*pb, default_provider))
            .then(ma.cmp(mb))
            .then(sa.cmp(sb))
    });

    // Extract (provider, model) labels before transcribers are consumed
    // so the GTK thread can pre-create rows with spinners.
    let pending_info: Vec<(Provider, String)> = transcribers.clone();
    let realtime_pending_info: Vec<(Provider, String)> = realtime_transcribers
        .iter()
        .map(|(p, m, _)| (*p, m.clone()))
        .collect();

    // Channels: transcription tasks → GTK, GTK → caller.
    let (msg_tx, msg_rx) = std::sync::mpsc::channel::<PickerMessage>();
    let (sel_tx, sel_rx) = tokio::sync::oneshot::channel::<PickerExit>();

    // Retry channel: GTK thread → async retry listener.
    // The bool indicates whether this is a streaming (realtime) retry.
    // Uses tokio unbounded channel so the receiver can `.await` — a
    // std::sync::mpsc would block the tokio worker thread and prevent
    // the runtime from shutting down when the picker closes.
    let (retry_tx, mut retry_rx) =
        tokio::sync::mpsc::unbounded_channel::<(Provider, String, bool)>();

    // Shared storage written by the GTK thread after selection,
    // read by the caller after the GTK thread finishes.
    // Tuple: (provider, model, text, is_primary, is_error, streaming)
    let results = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(
        Provider,
        String,
        String,
        bool,
        bool,
        bool,
        Option<Vec<TranscriptSegment>>,
        TranscriptionMetadata,
    )>::new()));
    let results_for_gtk = results.clone();

    // Clone audio path for the GTK play button before moving into closures.
    let audio_path_for_player = audio_path.clone();
    // Clone audio path for the waterfall spectrogram background thread.
    let audio_path_for_waterfall = audio_path.clone();
    // Clone audio path for picker selection auto-save.
    let audio_path_for_selection = audio_path.clone();
    #[cfg(test)]
    let audio_path_for_test = audio_path_for_selection.clone();

    // Clone the config Arc for the GTK thread: the per-row "T" click
    // handler may need to gate Parakeet on a download-consent dialog,
    // which calls `consent::resolve` on the config.  Only the
    // `parakeet` feature reads it, so gate the clone to avoid an
    // unused-binding warning on lean builds.  The original `config`
    // is consumed by the async transcription tasks below.
    #[cfg(feature = "parakeet")]
    let config_for_gtk = config.clone();
    let window = session.window.clone();

    // ── Per-record GTK contents ─────────────────────────────────
    let gtk_window = move || -> Result<(), TalkError> {
        use gtk4::glib;
        use gtk4::prelude::*;
        use std::cell::{Cell, RefCell};
        use std::rc::Rc;

        type SourceSlot = Rc<RefCell<Option<glib::SourceId>>>;
        fn source_slot(registry: &Rc<RefCell<Vec<SourceSlot>>>) -> SourceSlot {
            let slot = Rc::new(RefCell::new(None));
            registry.borrow_mut().push(Rc::clone(&slot));
            slot
        }
        fn remove_sources(registry: &Rc<RefCell<Vec<SourceSlot>>>) {
            for slot in registry.borrow().iter() {
                if let Some(source) = slot.borrow_mut().take() {
                    source.remove();
                }
            }
        }

        let source_registry: Rc<RefCell<Vec<SourceSlot>>> = Rc::new(RefCell::new(Vec::new()));

        let theme = crate::gtk_theme::ThemeColors::resolve();
        let error_hex = &theme.error;

        #[cfg(test)]
        let gtk_test_generation = begin_gtk_test_record();

        // CSS notes:
        //   - GtkListBox CSS node is "list", not "listbox"
        //   - Every stateful rule is duplicated with :backdrop
        //     to prevent the theme from lightening on focus loss
        //   - row:selected uses a translucent accent instead of
        //     the theme's solid #E95420
        let content = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
        content.set_vexpand(true);
        let close_btn = session.close_button.clone();

        // ── Audio player for playback button ─────────────────────
        // Initialized in background after window presentation to avoid
        // blocking the UI on cpal device probing (~1-2 s on PipeWire).
        let player: Rc<RefCell<Option<WavPlayer>>> = Rc::new(RefCell::new(None));

        // ── Waterfall spectrogram (base layer) ─────────────────────
        let waterfall_area = gtk4::DrawingArea::new();
        waterfall_area.set_hexpand(true);
        waterfall_area.set_vexpand(false);
        waterfall_area.set_content_height(32);
        waterfall_area.add_css_class("waterfall");

        type WaterfallData = Option<(Vec<Vec<f32>>, f32)>;
        let waterfall_data: Rc<RefCell<WaterfallData>> = Rc::new(RefCell::new(None));

        // Waterfall draw — renders once when data arrives, and on resize.
        {
            let data_ref = Rc::clone(&waterfall_data);
            waterfall_area.set_draw_func(move |_area, cr, width, height| {
                let w = width as usize;
                let h = height as usize;
                if w == 0 || h == 0 {
                    return;
                }
                let data = data_ref.borrow();
                if let Some((ref columns, peak)) = *data {
                    if peak > 0.0 && !columns.is_empty() {
                        if let Ok(mut surface) = gtk4::cairo::ImageSurface::create(
                            gtk4::cairo::Format::ARgb32,
                            width,
                            height,
                        ) {
                            let stride = surface.stride() as usize;
                            if let Ok(mut surf_data) = surface.data() {
                                let num_rows = crate::x11::render_util::WATERFALL_ROWS;
                                for x in 0..w {
                                    let col_idx = x * columns.len() / w;
                                    let col = &columns[col_idx];
                                    for y in 0..h {
                                        let data_row =
                                            (num_rows - 1) - (y * num_rows / h).min(num_rows - 1);
                                        let magnitude = if data_row < col.len() {
                                            col[data_row]
                                        } else {
                                            0.0
                                        };
                                        let norm = (magnitude / peak).clamp(0.0, 1.0);
                                        let brightness = if norm > 0.0 {
                                            (1.0 + norm * 9.0).log10()
                                        } else {
                                            0.0
                                        };
                                        let alpha = (brightness * 255.0) as u8;
                                        let off = y * stride + x * 4;
                                        if off + 3 < surf_data.len() {
                                            surf_data[off] = alpha;
                                            surf_data[off + 1] = alpha;
                                            surf_data[off + 2] = alpha;
                                            surf_data[off + 3] = alpha;
                                        }
                                    }
                                }
                            }
                            cr.set_source_surface(&surface, 0.0, 0.0).unwrap_or(());
                            cr.paint().unwrap_or(());
                        }
                        return;
                    }
                }
                cr.set_source_rgb(0.0, 0.0, 0.0);
                cr.paint().unwrap_or(());
            });
        }

        // ── Cursor overlay (lightweight, redraws independently) ───
        let cursor_area = gtk4::DrawingArea::new();
        cursor_area.set_hexpand(true);
        cursor_area.set_vexpand(true);

        let cursor_pos: Rc<RefCell<f64>> = Rc::new(RefCell::new(0.0));

        {
            let pos_ref = Rc::clone(&cursor_pos);
            cursor_area.set_draw_func(move |_area, cr, width, height| {
                let pos = *pos_ref.borrow();
                if pos > 0.0 {
                    let cx = (pos * width as f64).clamp(0.0, width as f64 - 1.0);
                    cr.set_source_rgba(1.0, 1.0, 1.0, 0.85);
                    cr.set_line_width(2.0);
                    cr.move_to(cx, 0.0);
                    cr.line_to(cx, height as f64);
                    cr.stroke().unwrap_or(());
                }
            });
        }

        // Stack waterfall + cursor using gtk4::Overlay.
        let wf_overlay = gtk4::Overlay::new();
        wf_overlay.set_child(Some(&waterfall_area));
        wf_overlay.add_overlay(&cursor_area);
        wf_overlay.set_hexpand(true);

        // Load waterfall data: cache hit → instant, miss → compute + persist.
        {
            let (wf_tx, wf_rx) = std::sync::mpsc::channel::<(Vec<Vec<f32>>, f32)>();
            let wf_audio_path = audio_path_for_waterfall;
            std::thread::spawn(move || {
                match crate::record::audio::load_waterfall(&wf_audio_path) {
                    Ok(result) => {
                        let _ = wf_tx.send(result);
                    }
                    Err(e) => {
                        log::warn!("waterfall: {}: {}", wf_audio_path.display(), e);
                    }
                }
            });

            let wf_data_ref = Rc::clone(&waterfall_data);
            let wf_area_ref = waterfall_area.clone();
            let source = source_slot(&source_registry);
            let source_callback = Rc::clone(&source);
            #[cfg(test)]
            let source_generation = gtk_test_generation;
            let id = glib::timeout_add_local(std::time::Duration::from_millis(50), move || {
                #[cfg(test)]
                note_gtk_test_source_tick(source_generation);
                let flow = match wf_rx.try_recv() {
                    Ok(result) => {
                        *wf_data_ref.borrow_mut() = Some(result);
                        wf_area_ref.queue_draw();
                        glib::ControlFlow::Break
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => glib::ControlFlow::Break,
                };
                if flow == glib::ControlFlow::Break {
                    *source_callback.borrow_mut() = None;
                }
                flow
            });
            *source.borrow_mut() = Some(id);
        }

        // ── Rewind button ────────────────────────────────────────
        let rewind_btn = gtk4::Button::from_icon_name("media-skip-backward-symbolic");
        rewind_btn.set_tooltip_text(Some("Rewind to start"));
        rewind_btn.add_css_class("play-btn");
        rewind_btn.set_sensitive(false); // starts at beginning

        {
            let player_ref = Rc::clone(&player);
            let pos_ref = Rc::clone(&cursor_pos);
            let wf_ref = waterfall_area.clone();
            rewind_btn.connect_clicked(move |btn| {
                if let Some(ref p) = *player_ref.borrow() {
                    p.seek(0.0);
                }
                *pos_ref.borrow_mut() = 0.0;
                btn.set_sensitive(false);
                wf_ref.queue_draw();
            });
        }

        let player_controls = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
        player_controls.append(&rewind_btn);

        // ── Play/Pause button ─────────────────────────────────────
        let play_btn = gtk4::Button::from_icon_name("media-playback-start-symbolic");
        play_btn.set_tooltip_text(Some("Play recorded audio"));
        play_btn.add_css_class("play-btn");
        if player.borrow().is_none() {
            play_btn.set_sensitive(false);
        }

        // Shared playback-active flag (avoids fragile icon-name checks).
        let playing_flag: Rc<RefCell<bool>> = Rc::new(RefCell::new(false));

        {
            let player_ref = Rc::clone(&player);
            let audio = audio_path_for_player;
            let pos_ref = Rc::clone(&cursor_pos);
            let flag = Rc::clone(&playing_flag);
            play_btn.connect_clicked(move |btn| {
                let player_guard = player_ref.borrow();
                let Some(ref p) = *player_guard else {
                    return;
                };
                if *flag.borrow() {
                    // Pause
                    p.pause();
                    *flag.borrow_mut() = false;
                    btn.set_icon_name("media-playback-start-symbolic");
                    btn.set_tooltip_text(Some("Resume playback"));
                } else if p.is_paused() || (p.has_audio() && !p.is_finished()) {
                    // Resume
                    p.resume();
                    *flag.borrow_mut() = true;
                    btn.set_icon_name("media-playback-pause-symbolic");
                    btn.set_tooltip_text(Some("Pause playback"));
                } else {
                    // Start fresh (or restart from current cursor position)
                    let pos = *pos_ref.borrow();
                    if let Err(e) = p.play(&audio) {
                        log::warn!("failed to play audio: {}", e);
                        return;
                    }
                    if pos > 0.0 && pos < 1.0 {
                        p.seek(pos);
                    }
                    *flag.borrow_mut() = true;
                    btn.set_icon_name("media-playback-pause-symbolic");
                    btn.set_tooltip_text(Some("Pause playback"));
                }
            });
        }

        // Poll for playback progress and completion.
        // Only redraws the cursor overlay — the waterfall is untouched.
        //
        // The cpal callback advances the sample position in buffer-sized
        // chunks (~10-20 ms at 48 kHz).  To avoid visible staircase
        // movement we interpolate: record the last raw progress and the
        // wall-clock time it changed, then linearly extrapolate at
        // display rate.
        {
            let player_poll = Rc::clone(&player);
            let btn_poll = play_btn.clone();
            let rewind_poll = rewind_btn.clone();
            let pos_poll = Rc::clone(&cursor_pos);
            let cursor_poll = cursor_area.clone();
            let flag_poll = Rc::clone(&playing_flag);

            // (last_raw_progress, timestamp_of_that_reading)
            let interp: Rc<RefCell<(f64, std::time::Instant)>> =
                Rc::new(RefCell::new((0.0, std::time::Instant::now())));

            let source = source_slot(&source_registry);
            #[cfg(test)]
            let source_generation = gtk_test_generation;
            let id = glib::timeout_add_local(std::time::Duration::from_millis(16), move || {
                #[cfg(test)]
                note_gtk_test_source_tick(source_generation);
                let player_guard = player_poll.borrow();
                let Some(ref p) = *player_guard else {
                    return glib::ControlFlow::Continue;
                };
                if *flag_poll.borrow() {
                    if p.is_finished() {
                        *pos_poll.borrow_mut() = 0.0;
                        *flag_poll.borrow_mut() = false;
                        *interp.borrow_mut() = (0.0, std::time::Instant::now());
                        btn_poll.set_icon_name("media-playback-start-symbolic");
                        btn_poll.set_tooltip_text(Some("Play recorded audio"));
                        rewind_poll.set_sensitive(false);
                    } else {
                        let raw = p.progress();
                        let now = std::time::Instant::now();
                        let mut st = interp.borrow_mut();

                        // When the cpal callback advances, reset the
                        // interpolation reference point.
                        if (raw - st.0).abs() > 1e-9 {
                            *st = (raw, now);
                        }

                        let dur = p.duration_secs();
                        let interpolated = if dur > 0.0 {
                            let elapsed = now.duration_since(st.1).as_secs_f64();
                            (st.0 + elapsed / dur).min(1.0)
                        } else {
                            raw
                        };
                        drop(st);

                        *pos_poll.borrow_mut() = interpolated;
                        rewind_poll.set_sensitive(interpolated > 0.0);
                    }
                    cursor_poll.queue_draw();
                } else if p.has_audio() && !p.is_finished() {
                    let progress = p.progress();
                    rewind_poll.set_sensitive(progress > 0.0);
                    // Keep reference fresh so resuming doesn't cause
                    // a jump from stale elapsed time.
                    *interp.borrow_mut() = (progress, std::time::Instant::now());
                }
                glib::ControlFlow::Continue
            });
            *source.borrow_mut() = Some(id);
        }

        player_controls.append(&play_btn);

        // ── Drag-to-seek on the waterfall ────────────────────────
        // Dragging pauses playback; releasing resumes.
        {
            let player_drag = Rc::clone(&player);
            let pos_drag = Rc::clone(&cursor_pos);
            let cursor_drag = cursor_area.clone();
            let btn_drag = play_btn.clone();
            let rewind_drag = rewind_btn.clone();
            let flag_drag = Rc::clone(&playing_flag);
            let was_playing = Rc::new(RefCell::new(false));

            let drag = gtk4::GestureDrag::new();

            {
                let player_ref = Rc::clone(&player_drag);
                let pos_ref = Rc::clone(&pos_drag);
                let cursor_ref = cursor_drag.downgrade();
                let btn_ref = btn_drag.clone();
                let flag_ref = Rc::clone(&flag_drag);
                let was_ref = Rc::clone(&was_playing);
                drag.connect_drag_begin(move |gesture, x, _y| {
                    let player_guard = player_ref.borrow();
                    let Some(ref p) = *player_guard else {
                        return;
                    };
                    let playing = *flag_ref.borrow() && !p.is_finished();
                    *was_ref.borrow_mut() = playing;
                    if playing {
                        p.pause();
                        *flag_ref.borrow_mut() = false;
                        btn_ref.set_icon_name("media-playback-start-symbolic");
                        btn_ref.set_tooltip_text(Some("Resume playback"));
                    }
                    if let Some(area) = gesture.widget().downcast_ref::<gtk4::DrawingArea>() {
                        let w = area.width() as f64;
                        if w > 0.0 {
                            let frac = (x / w).clamp(0.0, 1.0);
                            p.seek(frac);
                            *pos_ref.borrow_mut() = frac;
                            if let Some(cursor) = cursor_ref.upgrade() {
                                cursor.queue_draw();
                            }
                        }
                    }
                });
            }

            {
                let player_ref = Rc::clone(&player_drag);
                let pos_ref = Rc::clone(&pos_drag);
                let cursor_ref = cursor_drag.downgrade();
                let rewind_ref = rewind_drag.clone();
                drag.connect_drag_update(move |gesture, offset_x, _offset_y| {
                    let player_guard = player_ref.borrow();
                    let Some(ref p) = *player_guard else {
                        return;
                    };
                    if let Some(area) = gesture.widget().downcast_ref::<gtk4::DrawingArea>() {
                        let w = area.width() as f64;
                        if w > 0.0 {
                            let (start_x, _) = gesture.start_point().unwrap_or((0.0, 0.0));
                            let frac = ((start_x + offset_x) / w).clamp(0.0, 1.0);
                            p.seek(frac);
                            *pos_ref.borrow_mut() = frac;
                            rewind_ref.set_sensitive(frac > 0.0);
                            if let Some(cursor) = cursor_ref.upgrade() {
                                cursor.queue_draw();
                            }
                        }
                    }
                });
            }

            {
                let player_ref = Rc::clone(&player_drag);
                let btn_ref = btn_drag.clone();
                let flag_ref = Rc::clone(&flag_drag);
                let was_ref = Rc::clone(&was_playing);
                drag.connect_drag_end(move |_gesture, _offset_x, _offset_y| {
                    let player_guard = player_ref.borrow();
                    let Some(ref p) = *player_guard else {
                        return;
                    };
                    if *was_ref.borrow() {
                        p.resume();
                        *flag_ref.borrow_mut() = true;
                        btn_ref.set_icon_name("media-playback-pause-symbolic");
                        btn_ref.set_tooltip_text(Some("Pause playback"));
                    }
                });
            }

            // Attach drag to cursor overlay (sits on top, receives input).
            cursor_area.add_controller(drag);
        }

        // ── Editable text area + copy button ─────────────────────
        let editor_bar = gtk4::Box::new(gtk4::Orientation::Horizontal, 8);
        editor_bar.set_margin_start(4);
        editor_bar.set_margin_end(4);
        editor_bar.set_margin_bottom(4);

        let text_buffer = gtk4::TextBuffer::new(None::<&gtk4::TextTagTable>);
        let text_view = gtk4::TextView::with_buffer(&text_buffer);
        text_view.set_editable(true);
        text_view.set_wrap_mode(gtk4::WrapMode::WordChar);
        text_view.add_css_class("editor-view");
        text_view.add_css_class("transcript");

        let editor_scroll = gtk4::ScrolledWindow::builder()
            .hexpand(true)
            .max_content_height(80)
            .propagate_natural_height(true)
            .build();
        editor_scroll.set_child(Some(&text_view));
        editor_bar.append(&editor_scroll);

        let copy_btn = gtk4::Button::with_label("\u{29C9}");
        copy_btn.set_tooltip_text(Some("Copy to clipboard"));
        copy_btn.add_css_class("copy-btn");
        copy_btn.set_valign(gtk4::Align::Start);
        editor_bar.append(&copy_btn);

        content.append(&editor_bar);

        let save_error_label = gtk4::Label::new(None);
        save_error_label.set_xalign(0.0);
        save_error_label.set_wrap(true);
        save_error_label.add_css_class("error");
        save_error_label.set_visible(false);
        content.append(&save_error_label);

        let scrolled = gtk4::ScrolledWindow::builder()
            .vexpand(true)
            .hexpand(true)
            .build();
        let list = gtk4::ListBox::new();
        list.set_selection_mode(gtk4::SelectionMode::Single);
        list.set_activate_on_single_click(false);
        scrolled.set_child(Some(&list));
        content.append(&scrolled);

        let main_loop = glib::MainLoop::new(None, false);
        let sel_sender: Rc<RefCell<Option<tokio::sync::oneshot::Sender<PickerExit>>>> =
            Rc::new(RefCell::new(Some(sel_tx)));
        let retry_sender = Rc::new(RefCell::new(Some(retry_tx)));
        type CandidateList = Vec<(
            Provider,
            String,
            String,
            bool,
            bool,
            bool,
            Option<Vec<TranscriptSegment>>,
            TranscriptionMetadata,
        )>;
        let local_candidates: Rc<RefCell<CandidateList>> = Rc::new(RefCell::new(Vec::new()));
        // Transcript cells — used to swap spinner → label when results arrive.
        let transcript_cells: Rc<RefCell<Vec<gtk4::Box>>> = Rc::new(RefCell::new(Vec::new()));
        // Raw candidate texts (unmodified) for diff computation.
        let raw_texts: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        // Transcript labels for live diff updates (None for error/spinner rows).
        let transcript_labels: Rc<RefCell<Vec<Option<gtk4::Label>>>> =
            Rc::new(RefCell::new(Vec::new()));
        // Deferred row indices — rows awaiting user click.
        let deferred_indices: Rc<RefCell<std::collections::HashSet<usize>>> =
            Rc::new(RefCell::new(std::collections::HashSet::new()));
        // The initial-completion sweep must never inspect a deferred
        // row's spinner, even after its T button starts a retry.
        let mut initial_oneshot_indices = std::collections::HashSet::new();
        // Per-row "has-transcription" flag — true once the row has
        // received a non-empty transcript (cached or live).  Used to
        // prevent selecting a non-transcribed row (spinner, deferred,
        // error, "no speech detected") from wiping whatever the user
        // is currently viewing/editing in the text area.
        let has_transcription: Rc<RefCell<Vec<bool>>> = Rc::new(RefCell::new(Vec::new()));
        // Diff colours: deletion (red) from theme, insertion green (Tango palette).
        let del_color = error_hex.clone();
        let ins_color = String::from("#4e9a06");

        // ── Picker selection auto-save ──────────────────────────
        // Tracks the currently selected (provider, model, streaming)
        // and persists it to `<stem>.pick.yml` next to the recording,
        // including the current text-area content (the authoritative
        // transcription for this recording).
        // Saved: on first result arrival, after 5 s of inactivity
        // when dirty, and on picker close.

        /// Flush the pick to disk if dirty, reading the current text
        /// from `buf`.  Cancels any pending debounce timer.
        fn save_selection_now(
            ctx_ref: &Rc<RefCell<SelectionSaveCtx>>,
            pending: &Rc<RefCell<Option<glib::SourceId>>>,
            buf: &gtk4::TextBuffer,
            error_label: &gtk4::Label,
        ) -> bool {
            use gtk4::prelude::*;
            if let Some(id) = pending.borrow_mut().take() {
                id.remove();
            }
            let text = buf
                .text(&buf.start_iter(), &buf.end_iter(), false)
                .to_string();
            match persist_selection(&mut ctx_ref.borrow_mut(), &text) {
                Ok(()) => {
                    error_label.set_visible(false);
                    true
                }
                Err(error) => {
                    error_label.set_text(&format!("Could not save edits: {error}"));
                    error_label.set_visible(true);
                    false
                }
            }
        }

        /// Update `ctx.current` from a row selection.
        ///
        /// - `has_result = true`: the selected row has a transcription
        ///   result.  Mark dirty and arm the 5 s debounce save timer.
        /// - `has_result = false`: the selected row is still pending
        ///   (spinner).  Update `current` only -- do NOT mark dirty,
        ///   do NOT arm a timer.  Prevents saving `""` for a row that
        ///   hasn't transcribed yet (bug #6 in the gap analysis).
        fn mark_selection_dirty(
            ctx_ref: &Rc<RefCell<SelectionSaveCtx>>,
            pending: &Rc<RefCell<Option<glib::SourceId>>>,
            buf: &gtk4::TextBuffer,
            selection: (Provider, &str, bool),
            has_result: bool,
            error_label: &gtk4::Label,
        ) {
            let (provider, model, streaming) = selection;
            {
                let mut ctx = ctx_ref.borrow_mut();
                ctx.current = Some((provider, model.to_string(), streaming));
                if has_result {
                    ctx.dirty = true;
                }
            }
            if !has_result {
                return;
            }
            // Reset the 5 s debounce timer.
            if let Some(id) = pending.borrow_mut().take() {
                id.remove();
            }
            let ctx_clone = Rc::clone(ctx_ref);
            let pending_clone = Rc::clone(pending);
            let buf_clone = buf.clone();
            let error_clone = error_label.clone();
            let id = glib::timeout_add_local_once(std::time::Duration::from_secs(5), move || {
                *pending_clone.borrow_mut() = None;
                save_selection_now(&ctx_clone, &pending_clone, &buf_clone, &error_clone);
            });
            *pending.borrow_mut() = Some(id);
        }

        let selection_ctx: Rc<RefCell<SelectionSaveCtx>> =
            Rc::new(RefCell::new(SelectionSaveCtx {
                audio_path: audio_path_for_selection,
                dirty: false,
                current: None,
            }));
        let selection_pending: Rc<RefCell<Option<glib::SourceId>>> = Rc::new(RefCell::new(None));

        // Helper: build a row skeleton with provider + streaming + model labels.
        // Returns (hbox, transcript_cell) — caller fills the cell
        // with either a spinner or a transcript label.
        fn make_row_skeleton(
            provider: &str,
            model: &str,
            streaming: bool,
        ) -> (gtk4::Box, gtk4::Box) {
            use gtk4::prelude::*;

            let hbox = gtk4::Box::new(gtk4::Orientation::Horizontal, 12);
            hbox.set_margin_top(4);
            hbox.set_margin_bottom(4);
            hbox.set_margin_start(4);
            hbox.set_margin_end(4);

            // Transcript cell (container swapped between spinner / label)
            let cell = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
            cell.set_hexpand(true);
            hbox.append(&cell);

            // Provider (fixed width for column alignment)
            let prov_label = gtk4::Label::new(Some(provider));
            prov_label.set_xalign(0.0);
            prov_label.set_width_chars(8);
            prov_label.set_max_width_chars(8);
            prov_label.set_selectable(false);
            prov_label.add_css_class("dim");
            hbox.append(&prov_label);

            // Streaming indicator (between provider and model)
            let stream_label = gtk4::Label::new(Some(if streaming { "\u{26a1}" } else { "" }));
            stream_label.set_xalign(0.5);
            stream_label.set_width_chars(3);
            stream_label.set_max_width_chars(3);
            stream_label.set_selectable(false);
            stream_label.add_css_class("dim");
            hbox.append(&stream_label);

            // Model (fixed width for column alignment)
            let model_label = gtk4::Label::new(Some(model));
            model_label.set_xalign(0.0);
            model_label.set_width_chars(24);
            model_label.set_max_width_chars(24);
            model_label.set_ellipsize(gtk4::pango::EllipsizeMode::End);
            model_label.set_selectable(false);
            model_label.add_css_class("dim");
            hbox.append(&model_label);

            (hbox, cell)
        }

        // Helper: create a monospace transcript label.
        fn make_transcript_label(text: &str) -> gtk4::Label {
            use gtk4::prelude::*;

            let label = gtk4::Label::new(Some(text));
            label.set_xalign(0.0);
            label.set_hexpand(true);
            label.set_wrap(true);
            label.set_wrap_mode(gtk4::pango::WrapMode::WordChar);
            label.set_selectable(false);
            label.add_css_class("transcript");
            label
        }

        // ── Pre-create all rows ─────────────────────────────────
        //
        // Unified row construction: every (provider, model, streaming)
        // triple that should appear in the picker is gathered into a
        // single list, sorted once, then rendered in one pass.  Each
        // row falls into one of four kinds based on its current state
        // (already cached, one-shot pending, realtime pending, or
        // user-deferred) — but its *position* is determined solely by
        // (provider, model, streaming), so the visual order never
        // depends on which rows happen to have cached results.
        //
        // Every row also receives the unified "T" action button in the
        // trailing slot; clicking it always re-runs the transcription
        // for that row (first-time or retry — same behavior, same
        // button).

        /// Kind of row to render, chosen once at picker open time.
        enum RowKind<'a> {
            /// Cached result — render the transcript text directly.
            /// Fields: text, is_primary.
            Cached { text: &'a str, is_primary: bool },
            /// Auto-firing one-shot model — render a spinner, result
            /// arrives via the poll loop.
            PendingOneShot,
            /// Auto-firing realtime model — render an empty label,
            /// text fills in incrementally.
            PendingRealtime,
            /// User-deferred — render an empty placeholder.  The
            /// unified action button triggers the first transcription.
            Deferred,
        }

        // Build the unified row list.  Preserve primary flag for cached
        // entries (it drives initial selection below).
        let mut all_rows: Vec<(Provider, String, bool, RowKind)> = Vec::new();
        for (p, m, t, is_primary, s) in &cached_entries {
            all_rows.push((
                *p,
                m.clone(),
                *s,
                RowKind::Cached {
                    text: t.as_str(),
                    is_primary: *is_primary,
                },
            ));
        }
        for (p, m) in &pending_info {
            all_rows.push((*p, m.clone(), false, RowKind::PendingOneShot));
        }
        for (p, m) in &realtime_pending_info {
            all_rows.push((*p, m.clone(), true, RowKind::PendingRealtime));
        }
        for (p, m, s) in &deferred_candidates {
            all_rows.push((*p, m.clone(), *s, RowKind::Deferred));
        }
        // Single sort drives the visible order; matches the
        // per-source sorts done above so the order is identical
        // regardless of which bucket a row came from.
        all_rows.sort_by(|(pa, ma, sa, _), (pb, mb, sb, _)| {
            provider_rank(*pa, default_provider)
                .cmp(&provider_rank(*pb, default_provider))
                .then(ma.cmp(mb))
                .then(sa.cmp(sb))
        });

        // Unified action-button storage — one per row (same order as
        // `local_candidates`).  The poll loop enables the button when
        // a result/error arrives and disables it while a transcription
        // is in flight, and flips the icon between "T" (no
        // transcription yet) and "↻" (re-run an existing one).
        let action_buttons: Rc<RefCell<Vec<gtk4::Button>>> = Rc::new(RefCell::new(Vec::new()));

        // Flip the action button's icon + tooltip based on whether the
        // row currently holds a (non-empty) transcript.  "T" means
        // "transcribe this (first time)"; "↻" means "retry / re-run
        // the existing transcription"; "✕" means "stop the in-flight
        // request".
        fn set_action_icon(btn: &gtk4::Button, has_transcript: bool) {
            use gtk4::prelude::*;
            if has_transcript {
                btn.set_label("↻");
                btn.set_tooltip_text(Some("Re-transcribe with this model"));
            } else {
                btn.set_label("T");
                btn.set_tooltip_text(Some("Transcribe with this model"));
            }
        }

        /// Flip the action button into Stop mode for the duration
        /// of an in-flight request.  Click handlers for the row
        /// dispatch on the current label to choose between
        /// triggering a (re-)transcription and cancelling the
        /// in-flight one.
        fn set_action_icon_stop(btn: &gtk4::Button) {
            use gtk4::prelude::*;
            btn.set_label("✕");
            btn.set_tooltip_text(Some("Stop in-flight transcription"));
        }

        // Helper: build the unified action button for a given row.
        // Wires a click handler that resets the row to pending and
        // sends a (provider, model, streaming) on the retry channel.
        // Placed in the leading slot of the row's hbox — same
        // position on every row; icon toggles between "T" (no
        // transcription yet) and "↻" (retry existing result).
        //
        // The button captures the row's `is_primary` flag in
        // addition to (provider, model, streaming) so the click
        // handler resolves to THIS row even when a non-primary
        // candidate above shares the same (provider, model,
        // streaming) tuple (e.g. a cached primary entry plus a
        // deferred candidate for the same model — clicking T on
        // the deferred row would otherwise spin the primary row).
        let make_action_button = {
            let retry_tx = Rc::clone(&retry_sender);
            let cells_for_btn = Rc::clone(&transcript_cells);
            let labels_for_btn = Rc::clone(&transcript_labels);
            let raw_for_btn = Rc::clone(&raw_texts);
            let cands_for_btn = Rc::clone(&local_candidates);
            let deferred_for_btn = Rc::clone(&deferred_indices);
            let buttons_for_btn = Rc::clone(&action_buttons);
            let has_tx_for_btn = Rc::clone(&has_transcription);
            // Captured by every per-row click handler: the picker
            // window is the AlertDialog parent for the Parakeet
            // download-consent prompt; the config drives the
            // `consent::resolve` presence check.  Only the
            // `parakeet` feature ever reads these, but they live at
            // factory scope unconditionally so the closure shape
            // stays identical across cfg flavours.
            #[cfg(feature = "parakeet")]
            let window_for_btn = window.downgrade();
            #[cfg(feature = "parakeet")]
            let config_for_btn = config_for_gtk.clone();
            move |provider: Provider,
                  model: String,
                  streaming: bool,
                  is_primary_row: bool,
                  row_idx_at_create: usize|
                  -> gtk4::Button {
                let btn = gtk4::Button::with_label("T");
                btn.set_tooltip_text(Some("Transcribe with this model"));
                btn.add_css_class("action-btn");
                btn.set_valign(gtk4::Align::Center);
                {
                    let tx = Rc::clone(&retry_tx);
                    let cells_ref = Rc::clone(&cells_for_btn);
                    let labels_ref = Rc::clone(&labels_for_btn);
                    let raw_ref = Rc::clone(&raw_for_btn);
                    let cands_ref = Rc::clone(&cands_for_btn);
                    let deferred_ref = Rc::clone(&deferred_for_btn);
                    let buttons_ref = Rc::clone(&buttons_for_btn);
                    let has_tx_ref = Rc::clone(&has_tx_for_btn);
                    // Both bindings only feed the Parakeet
                    // download-consent gate below; the factory-scope
                    // captures are already cfg-gated, so just clone
                    // here under the same gate.
                    #[cfg(feature = "parakeet")]
                    let window_ref = window_for_btn.clone();
                    #[cfg(feature = "parakeet")]
                    let config_ref = config_for_btn.clone();
                    btn.connect_clicked(move |clicked| {
                        use gtk4::prelude::*;
                        log::debug!(
                            "picker click: provider={} model={} streaming={} is_primary={} row_idx_at_create={}",
                            provider, model, streaming, is_primary_row, row_idx_at_create,
                        );

                        // Stop mode: route the click to SIGUSR1 on
                        // our own PID, which fans out via the
                        // `jobs` SIGUSR1 handler to every
                        // registered in-flight token.
                        if clicked.label().as_deref() == Some("✕") {
                            use nix::sys::signal::{kill, Signal};
                            use nix::unistd::Pid;
                            let pid = Pid::from_raw(std::process::id() as i32);
                            if let Err(e) = kill(pid, Signal::SIGUSR1) {
                                log::warn!("picker: failed to send SIGUSR1: {}", e);
                            } else {
                                log::info!("picker: Stop clicked, SIGUSR1 sent");
                            }
                            return;
                        }

                        let idx = {
                            let cands = cands_ref.borrow();
                            cands.iter().position(|(p, m, _, prim, _, s, _, _)| {
                                *p == provider
                                    && *m == model
                                    && *s == streaming
                                    && *prim == is_primary_row
                            })
                        };
                        log::debug!("picker click resolved idx={:?}", idx);
                        let Some(idx) = idx else {
                            return;
                        };

                        // The post-consent action: reset the row,
                        // swap to spinner / streaming label, flip
                        // the button to Stop mode, and hand off to
                        // the async retry listener.  Wrapped in a
                        // local `FnOnce` so it can be invoked either
                        // directly (no consent needed) or from the
                        // Parakeet AlertDialog callback (after the
                        // user confirms the download).
                        let do_transcribe = {
                            let cells_ref = Rc::clone(&cells_ref);
                            let labels_ref = Rc::clone(&labels_ref);
                            let raw_ref = Rc::clone(&raw_ref);
                            let cands_ref = Rc::clone(&cands_ref);
                            let deferred_ref = Rc::clone(&deferred_ref);
                            let buttons_ref = Rc::clone(&buttons_ref);
                            let has_tx_ref = Rc::clone(&has_tx_ref);
                            let tx = Rc::clone(&tx);
                            let model = model.clone();
                            let clicked = clicked.clone();
                            move || {
                                // Reset row state: clear text, errors,
                                // segments, diff baseline, and the
                                // deferred flag.
                                {
                                    let mut cands = cands_ref.borrow_mut();
                                    if let Some((
                                        _,
                                        _,
                                        text,
                                        _,
                                        is_error,
                                        _,
                                        segments,
                                        meta,
                                    )) = cands.get_mut(idx)
                                    {
                                        text.clear();
                                        *is_error = false;
                                        *segments = None;
                                        *meta = TranscriptionMetadata::default();
                                    }
                                }
                                raw_ref.borrow_mut()[idx] = String::new();
                                deferred_ref.borrow_mut().remove(&idx);
                                if let Some(flag) = has_tx_ref.borrow_mut().get_mut(idx) {
                                    *flag = false;
                                }

                                // Swap the transcript cell to a spinner
                                // (one-shot) or empty streaming label
                                // (realtime).
                                {
                                    let cells = cells_ref.borrow();
                                    if let Some(cell) = cells.get(idx) {
                                        while let Some(child) = cell.first_child() {
                                            cell.remove(&child);
                                        }
                                        if streaming {
                                            let label = make_transcript_label("");
                                            cell.append(&label);
                                            labels_ref.borrow_mut()[idx] = Some(label);
                                        } else {
                                            let spinner = gtk4::Spinner::new();
                                            spinner.start();
                                            cell.append(&spinner);
                                            labels_ref.borrow_mut()[idx] = None;
                                        }
                                    }
                                }

                                // Flip the action button to Stop mode
                                // for the duration of the in-flight
                                // request.  The poll loop flips it
                                // back to T / ↻ when the result
                                // (or error) arrives.
                                set_action_icon_stop(&clicked);
                                if let Some(btn) = buttons_ref.borrow().get(idx) {
                                    set_action_icon_stop(btn);
                                }

                                if let Some(tx) = tx.borrow().as_ref() {
                                    let _ = tx.send((provider, model.clone(), streaming));
                                }
                            }
                        };

                        // Parakeet download-consent gate.  When the
                        // user clicks "T" on a Parakeet row and the
                        // model files are not on disk, show a modal
                        // AlertDialog asking permission to fetch
                        // ~640 MB before any download / transcription
                        // runs.  Cancel is a clean no-op (no row
                        // reset, no Stop-mode flip, no `tx.send`);
                        // Download triggers `do_transcribe` and the
                        // async retry listener performs the download
                        // before the actual transcription.
                        //
                        // Sitting BEFORE the row-state reset and
                        // spinner swap, so Cancel leaves the row
                        // exactly as it was — the button stays "T"
                        // and the transcript cell keeps its prior
                        // content.
                        #[cfg(feature = "parakeet")]
                        if provider == Provider::Parakeet {
                            let needs_consent = matches!(
                                crate::transcription::parakeet::consent::resolve(
                                    config_ref.as_ref(),
                                ),
                                Ok(ref st) if !st.present,
                            );
                            if needs_consent {
                                log::info!(
                                    "picker: Parakeet model absent — prompting for download consent",
                                );
                                let dialog = gtk4::AlertDialog::builder()
                                    .modal(true)
                                    .message("Download Parakeet speech model?")
                                    .detail(
                                        "The local Parakeet model (~640 MB) is not installed. \
                                         Download it now?",
                                    )
                                    .buttons(["Cancel", "Download"])
                                    .default_button(1)
                                    .cancel_button(0)
                                    .build();
                                let Some(window_ref) = window_ref.upgrade() else {
                                    return;
                                };
                                dialog.choose(
                                    Some(&window_ref),
                                    None::<&gtk4::gio::Cancellable>,
                                    move |res| match res {
                                        Ok(1) => {
                                            log::info!(
                                                "picker: Parakeet download consent granted",
                                            );
                                            do_transcribe();
                                        }
                                        Ok(_) => {
                                            log::info!(
                                                "picker: Parakeet download declined — aborting click",
                                            );
                                        }
                                        Err(e) => {
                                            log::warn!(
                                                "picker: Parakeet consent dialog failed: {} — aborting click",
                                                e,
                                            );
                                        }
                                    },
                                );
                                return;
                            }
                        }

                        do_transcribe();
                    });
                }
                btn
            }
        };

        // Cached entries: rows with transcript already filled.
        // The `is_primary` flag marks the saved selection for pre-selection.
        let mut selected_row = false;
        // Track which row to select, deferring the actual selection
        // until AFTER `connect_row_selected` is wired below.  Wiring
        // the handler first is critical: otherwise the initial
        // select_row fires before the handler exists, and
        // `selection_ctx.current` is never populated (bugs #1, #2,
        // #4 in the gap analysis).
        let mut index_to_select: Option<i32> = None;
        for (row_idx, (provider, model, streaming, kind)) in all_rows.iter().enumerate() {
            let (hbox, cell) = make_row_skeleton(&provider.to_string(), model, *streaming);

            // Per-row transcript cell population and `local_candidates`
            // bookkeeping.  `label_opt` is set when the cell hosts a
            // diff-capable label (cached result or realtime stream);
            // `None` means spinner or placeholder.
            let (initial_text, is_primary, is_deferred, label_opt): (
                String,
                bool,
                bool,
                Option<gtk4::Label>,
            ) = match kind {
                RowKind::Cached { text, is_primary } => {
                    let label = make_transcript_label(text);
                    if text.is_empty() {
                        label.set_markup("<i>(no speech detected)</i>");
                        label.set_opacity(0.35);
                    }
                    cell.append(&label);
                    ((*text).to_string(), *is_primary, false, Some(label))
                }
                RowKind::PendingOneShot => {
                    let spinner = gtk4::Spinner::new();
                    spinner.start();
                    cell.append(&spinner);
                    (String::new(), false, false, None)
                }
                RowKind::PendingRealtime => {
                    let label = make_transcript_label("");
                    cell.append(&label);
                    (String::new(), false, false, Some(label))
                }
                RowKind::Deferred => (String::new(), false, true, None),
            };

            // Unified action button: always present, same position
            // (leftmost), icon toggles between "T" (no transcript),
            // "↻" (retry an existing one), and "✕" (stop an
            // in-flight request).  Always sensitive — in-flight
            // rows route the click to SIGUSR1.
            let action_btn =
                make_action_button(*provider, model.clone(), *streaming, is_primary, row_idx);
            match kind {
                RowKind::PendingOneShot | RowKind::PendingRealtime => {
                    action_btn.set_sensitive(true);
                    set_action_icon_stop(&action_btn);
                }
                RowKind::Cached { text, .. } => {
                    action_btn.set_sensitive(true);
                    set_action_icon(&action_btn, !text.is_empty());
                }
                RowKind::Deferred => {
                    action_btn.set_sensitive(true);
                    set_action_icon(&action_btn, false);
                }
            }
            // Prepend: the action button occupies the leftmost column,
            // ahead of the transcript cell.
            hbox.prepend(&action_btn);

            let row = gtk4::ListBoxRow::new();
            row.set_child(Some(&hbox));
            list.append(&row);

            // Initial selection: primary cached row wins; otherwise
            // the first row in visual order.
            if is_primary || (row_idx == 0 && !selected_row) {
                index_to_select = Some(row_idx as i32);
                selected_row = true;
            }

            // Only cached rows with non-empty text start as "has
            // transcription".  Spinners, deferred rows, realtime-
            // pending rows, and "no speech detected" cached rows all
            // start false, so selecting them won't wipe the text area.
            let initial_has_transcript =
                matches!(kind, RowKind::Cached { .. }) && !initial_text.is_empty();

            local_candidates.borrow_mut().push((
                *provider,
                model.clone(),
                initial_text.clone(),
                is_primary,
                false,
                *streaming,
                None,
                TranscriptionMetadata::default(),
            ));
            transcript_cells.borrow_mut().push(cell);
            raw_texts.borrow_mut().push(initial_text);
            transcript_labels.borrow_mut().push(label_opt);
            action_buttons.borrow_mut().push(action_btn);
            has_transcription.borrow_mut().push(initial_has_transcript);
            if is_deferred {
                deferred_indices.borrow_mut().insert(row_idx);
            }
            if matches!(kind, RowKind::PendingOneShot) {
                initial_oneshot_indices.insert(row_idx);
            }
        }

        // ── Row selection → populate text area ──────────────────
        // Wired BEFORE auto-selecting so the initial selection
        // triggers mark_selection_dirty and populates the text area.
        {
            let cands_sel = Rc::clone(&local_candidates);
            let buf_sel = text_buffer.clone();
            let sel_ctx = Rc::clone(&selection_ctx);
            let sel_pending = Rc::clone(&selection_pending);
            let has_tx_sel = Rc::clone(&has_transcription);
            let save_error_sel = save_error_label.clone();
            list.connect_row_selected(move |_, row| {
                if let Some(row) = row {
                    let idx = row.index() as usize;
                    // Extract (provider, model, streaming, has_result)
                    // in a scoped borrow.  A row "has_result" when
                    // its text is non-empty OR it received an empty
                    // result (tracked elsewhere as not a spinner) --
                    // for simplicity we use `!text.is_empty()` here
                    // since empty-result rows are also non-selectable
                    // in practice until streaming updates arrive.
                    // Actually: the current row's text field is
                    // populated by the poll loop when ANY result
                    // arrives (including empty).  So "has a result"
                    // = "not an error" AND the cell is NOT a spinner.
                    // We approximate with `!is_error` only; selecting
                    // a spinner would still update current but won't
                    // arm the save timer because we pass false when
                    // text is empty AND streaming=false (pending).
                    let sel_info = {
                        let cands = cands_sel.borrow();
                        cands.get(idx).and_then(|(p, m, t, _, is_error, s, _, _)| {
                            if !*is_error {
                                // `has_result`: row has received a
                                // candidate (text may be empty).  We
                                // detect "pending one-shot row" by
                                // checking if the text is empty AND
                                // the row is not streaming -- a
                                // realtime row may legitimately have
                                // empty text while streaming.
                                let has_result = !t.is_empty() || *s;
                                Some((*p, m.clone(), *s, has_result))
                            } else {
                                None
                            }
                        })
                    };
                    if let Some((provider, model, streaming, has_result)) = sel_info {
                        mark_selection_dirty(
                            &sel_ctx,
                            &sel_pending,
                            &buf_sel,
                            (provider, &model, streaming),
                            has_result,
                            &save_error_sel,
                        );
                    }
                    // Only overwrite the text area when this row
                    // actually holds a transcription.  Selecting a
                    // non-transcribed row (spinner, deferred, error,
                    // "no speech detected") leaves the text area
                    // untouched — the user can keep viewing/editing
                    // whatever is currently there.
                    let row_has_transcript = has_tx_sel.borrow().get(idx).copied().unwrap_or(false);
                    if row_has_transcript {
                        let cands = cands_sel.borrow();
                        if let Some((_, _, text, _, is_error, _, _, _)) = cands.get(idx) {
                            if !*is_error {
                                buf_sel.set_text(text);
                            }
                        }
                    }
                }
            });
        }

        // ── Live diff update + metadata save on text-area edits ──
        {
            let raw_for_diff = Rc::clone(&raw_texts);
            let labels_for_diff = Rc::clone(&transcript_labels);
            let dc = del_color.clone();
            let ic = ins_color.clone();
            let sel_ctx_edit = Rc::clone(&selection_ctx);
            let sel_pending_edit = Rc::clone(&selection_pending);
            let save_error_edit = save_error_label.clone();
            text_buffer.connect_changed(move |buf| {
                // Diff update (existing logic).
                let reference = buf
                    .text(&buf.start_iter(), &buf.end_iter(), false)
                    .to_string();
                let texts = raw_for_diff.borrow();
                let labels = labels_for_diff.borrow();
                for (i, maybe_label) in labels.iter().enumerate() {
                    if let Some(label) = maybe_label {
                        if let Some(raw) = texts.get(i) {
                            if !raw.is_empty() {
                                let markup = diff_markup(&reference, raw, &dc, &ic);
                                label.set_markup(&markup);
                            }
                        }
                    }
                }
                // Mark dirty so the pick is saved with updated text.
                // If `current` is None (e.g. pre-populate fires before
                // the row-selected handler has set it), seed it from
                // the currently selected row so the eventual save has
                // a valid target.
                {
                    let mut ctx = sel_ctx_edit.borrow_mut();
                    ctx.dirty = true;
                    if ctx.current.is_none() {
                        // Look up selected row's provider/model.
                        // (We can't borrow `cands_sel` here since this
                        // closure lives outside that scope; the row
                        // selection handler is responsible for setting
                        // `current` via mark_selection_dirty.)
                    }
                }
                // Reset the 5 s debounce timer.
                if let Some(id) = sel_pending_edit.borrow_mut().take() {
                    id.remove();
                }
                let ctx_clone = Rc::clone(&sel_ctx_edit);
                let pending_clone = Rc::clone(&sel_pending_edit);
                let buf_clone = buf.clone();
                let error_clone = save_error_edit.clone();
                let id =
                    glib::timeout_add_local_once(std::time::Duration::from_secs(5), move || {
                        *pending_clone.borrow_mut() = None;
                        save_selection_now(&ctx_clone, &pending_clone, &buf_clone, &error_clone);
                    });
                *sel_pending_edit.borrow_mut() = Some(id);
            });
        }

        // Now that `connect_row_selected` is wired (above), perform
        // the deferred initial selection.  This fires the handler,
        // which in turn calls `mark_selection_dirty` and populates
        // the text area.
        if let Some(idx) = index_to_select {
            if let Some(row) = list.row_at_index(idx) {
                list.select_row(Some(&row));
            }
        } else if let Some(first) = list.row_at_index(0) {
            list.select_row(Some(&first));
        }

        // ── Copy button → clipboard ─────────────────────────────
        {
            let buf_copy = text_buffer.clone();
            copy_btn.connect_clicked(move |_| {
                let text = buf_copy
                    .text(&buf_copy.start_iter(), &buf_copy.end_iter(), false)
                    .to_string();
                if !text.is_empty() {
                    if let Some(display) = gtk4::gdk::Display::default() {
                        display.clipboard().set_text(&text);
                    }
                }
            });
        }

        let finished = Rc::new(Cell::new(false));
        let finish_picker: Rc<dyn Fn(PickerExit) -> bool> = {
            let finished = Rc::clone(&finished);
            let sources = Rc::clone(&source_registry);
            let selection_ctx = Rc::clone(&selection_ctx);
            let selection_pending = Rc::clone(&selection_pending);
            let buffer = text_buffer.clone();
            let save_error = save_error_label.clone();
            let player = Rc::clone(&player);
            let retry_sender = Rc::clone(&retry_sender);
            let action_buttons = Rc::clone(&action_buttons);
            let sender = Rc::clone(&sel_sender);
            let main_loop = main_loop.clone();
            let content = content.downgrade();
            let waterfall = wf_overlay.downgrade();
            let controls = player_controls.downgrade();
            Rc::new(move |exit| {
                if finished.get() {
                    return true;
                }
                if !save_selection_now(&selection_ctx, &selection_pending, &buffer, &save_error) {
                    return false;
                }
                finished.set(true);
                remove_sources(&sources);
                if let Some(ref player) = *player.borrow() {
                    player.stop();
                }
                retry_sender.borrow_mut().take();
                action_buttons.borrow_mut().clear();
                if let Some(content) = content.upgrade() {
                    content.set_sensitive(false);
                }
                if let Some(waterfall) = waterfall.upgrade() {
                    waterfall.set_sensitive(false);
                }
                if let Some(controls) = controls.upgrade() {
                    controls.set_sensitive(false);
                }
                if let Some(sender) = sender.borrow_mut().take() {
                    let _ = sender.send(exit);
                }
                main_loop.quit();
                true
            })
        };

        // Enter / double-click confirms selection (only if text is loaded)
        let row_activated_handler = {
            let cands = Rc::clone(&local_candidates);
            let buf_confirm = text_buffer.clone();
            let finish = Rc::clone(&finish_picker);
            list.connect_row_activated(move |_, row| {
                let idx = row.index() as usize;
                let ready = cands
                    .borrow()
                    .get(idx)
                    .is_some_and(|(_, _, _, _, is_error, _, _, _)| !is_error);
                if ready {
                    // Use edited text-area content if available.
                    let edited = buf_confirm
                        .text(&buf_confirm.start_iter(), &buf_confirm.end_iter(), false)
                        .to_string();
                    if !edited.is_empty() {
                        cands.borrow_mut()[idx].2 = edited;
                    }
                    finish(PickerExit::Selected(idx));
                }
            })
        };

        // Previous/next recording navigation dispatches through the
        // session-owned buttons. The callback slot changes per record,
        // while each button's GTK signal handler is connected only once.
        let navigation_callback: PickerNavigationCallback = {
            let finish = Rc::clone(&finish_picker);
            Rc::new(move |direction| {
                finish(PickerExit::Navigate(direction));
            })
        };

        // Close button cancels (same as Escape)
        let close_handler = {
            let finish = Rc::clone(&finish_picker);
            close_btn.connect_clicked(move |_| {
                finish(PickerExit::Cancelled);
            })
        };

        // Escape cancels
        let key_ctl = gtk4::EventControllerKey::new();
        let key_handler = {
            let finish = Rc::clone(&finish_picker);
            key_ctl.connect_key_pressed(move |_, key, _, _| {
                if key == gtk4::gdk::Key::Escape {
                    finish(PickerExit::Cancelled);
                    glib::Propagation::Stop
                } else {
                    glib::Propagation::Proceed
                }
            })
        };
        window.add_controller(key_ctl.clone());

        // Window close
        let window_close_handler = {
            let finish = Rc::clone(&finish_picker);
            window.connect_close_request(move |_| {
                finish(PickerExit::Cancelled);
                glib::Propagation::Stop
            })
        };

        // Build an error cell containing just the error message.
        //
        // Retry is handled by the unified "T" action button in the
        // trailing slot of the row (same button that starts the
        // first-time transcription) — no per-error retry icon here.
        let make_error_cell = {
            let raw_for_err = Rc::clone(&raw_texts);
            let labels_for_err = Rc::clone(&transcript_labels);
            move |cell: &gtk4::Box, idx: usize, msg: &str| {
                while let Some(child) = cell.first_child() {
                    cell.remove(&child);
                }
                // Clear diff state for this row.
                raw_for_err.borrow_mut()[idx] = String::new();
                labels_for_err.borrow_mut()[idx] = None;

                let err = gtk4::Label::new(Some(msg));
                err.set_xalign(0.0);
                err.set_hexpand(true);
                err.set_wrap(true);
                err.set_wrap_mode(gtk4::pango::WrapMode::WordChar);
                err.add_css_class("error");
                cell.append(&err);
            }
        };

        // Poll transcription results — update existing rows in-place.
        {
            let list = list.clone();
            let initial_oneshot_indices = initial_oneshot_indices;
            let cands = Rc::clone(&local_candidates);
            let cells = Rc::clone(&transcript_cells);
            let raw_poll = Rc::clone(&raw_texts);
            let labels_poll = Rc::clone(&transcript_labels);
            let buttons_poll = Rc::clone(&action_buttons);
            let has_tx_poll = Rc::clone(&has_transcription);
            let buf_poll = text_buffer.clone();
            let dc_poll = del_color.clone();
            let ic_poll = ins_color.clone();
            let make_err = make_error_cell.clone();
            let sel_ctx_poll = Rc::clone(&selection_ctx);
            let sel_pending_poll = Rc::clone(&selection_pending);
            let save_error_poll = save_error_label.clone();
            let source = source_slot(&source_registry);
            let source_callback = Rc::clone(&source);
            #[cfg(test)]
            let source_generation = gtk_test_generation;
            let id = glib::timeout_add_local(std::time::Duration::from_millis(50), move || {
                #[cfg(test)]
                note_gtk_test_source_tick(source_generation);
                // Cap messages per tick so the cursor timer and other
                // main-loop sources stay responsive.
                const MAX_MSGS_PER_TICK: usize = 5;
                for _ in 0..MAX_MSGS_PER_TICK {
                    match msg_rx.try_recv() {
                        Ok(PickerMessage::Candidate(c)) => {
                            // Find the pre-created row by
                            // (provider, model, streaming).
                            let idx = {
                                let cands = cands.borrow();
                                cands.iter().position(|(p, m, _, _, _, s, _, _)| {
                                    *p == c.provider && *m == c.model && *s == c.streaming
                                })
                            };
                            if let Some(idx) = idx {
                                // Clear cell content (spinner or previous error).
                                {
                                    let cells = cells.borrow();
                                    if let Some(cell) = cells.get(idx) {
                                        while let Some(child) = cell.first_child() {
                                            cell.remove(&child);
                                        }
                                    }
                                }

                                if let Some(ref err_msg) = c.error {
                                    cands.borrow_mut()[idx].4 = true;
                                    make_err(&cells.borrow()[idx], idx, err_msg);
                                    // Re-enable "T" so the user can
                                    // retry the failed transcription
                                    // (no transcript → "T", not "↻").
                                    if let Some(btn) = buttons_poll.borrow().get(idx) {
                                        btn.set_sensitive(true);
                                        set_action_icon(btn, false);
                                        // Clear any in-flight status
                                        // tooltip left over from
                                        // `CandidateStatus` events.
                                        btn.set_tooltip_text(None);
                                    }
                                    // Clear the transcription flag —
                                    // the error wiped the row.
                                    if let Some(flag) = has_tx_poll.borrow_mut().get_mut(idx) {
                                        *flag = false;
                                    }
                                } else {
                                    // Same path for empty and non-empty
                                    // text — empty is a valid result.
                                    let mut cands = cands.borrow_mut();
                                    cands[idx].2 = c.text.clone();
                                    cands[idx].6 = c.segments.clone();
                                    cands[idx].7 = c.metadata.clone();
                                    raw_poll.borrow_mut()[idx] = c.text.clone();

                                    let is_selected = list
                                        .selected_row()
                                        .is_some_and(|r| r.index() as usize == idx);
                                    let buf_empty = buf_poll
                                        .text(&buf_poll.start_iter(), &buf_poll.end_iter(), false)
                                        .is_empty();
                                    if is_selected && buf_empty {
                                        buf_poll.set_text(&c.text);
                                        // First result for the selected row —
                                        // save selection immediately.
                                        save_selection_now(
                                            &sel_ctx_poll,
                                            &sel_pending_poll,
                                            &buf_poll,
                                            &save_error_poll,
                                        );
                                    }

                                    let label = make_transcript_label("");
                                    if c.text.is_empty() {
                                        label.set_markup("<i>(no speech detected)</i>");
                                        label.set_opacity(0.35);
                                    } else {
                                        let reference = buf_poll
                                            .text(
                                                &buf_poll.start_iter(),
                                                &buf_poll.end_iter(),
                                                false,
                                            )
                                            .to_string();
                                        let markup =
                                            diff_markup(&reference, &c.text, &dc_poll, &ic_poll);
                                        label.set_markup(&markup);
                                    }
                                    cells.borrow()[idx].append(&label);
                                    labels_poll.borrow_mut()[idx] = Some(label);
                                    // Result arrived — re-enable the
                                    // button.  Flip to "↻" only when
                                    // there is actually a transcript
                                    // to re-run (empty result keeps
                                    // "T" so the user can try again).
                                    let has_text = !c.text.is_empty();
                                    if let Some(btn) = buttons_poll.borrow().get(idx) {
                                        btn.set_sensitive(true);
                                        set_action_icon(btn, has_text);
                                        // Clear any in-flight status
                                        // tooltip left over from
                                        // `CandidateStatus` events.
                                        btn.set_tooltip_text(None);
                                    }
                                    // Track whether this row now holds
                                    // a transcription, so selecting it
                                    // populates the text area (and
                                    // selecting an empty-result row
                                    // does not wipe it).
                                    if let Some(flag) = has_tx_poll.borrow_mut().get_mut(idx) {
                                        *flag = has_text;
                                    }
                                }
                            }
                        }
                        Ok(PickerMessage::StreamUpdate {
                            provider,
                            model,
                            accumulated_text,
                        }) => {
                            // Find the realtime row by
                            // (provider, model, streaming=true).
                            let idx = {
                                let cands = cands.borrow();
                                cands.iter().position(|(p, m, _, _, _, s, _, _)| {
                                    *p == provider && *m == model && *s
                                })
                            };
                            if let Some(idx) = idx {
                                raw_poll.borrow_mut()[idx] = accumulated_text.clone();
                                let reference = buf_poll
                                    .text(&buf_poll.start_iter(), &buf_poll.end_iter(), false)
                                    .to_string();
                                let markup =
                                    diff_markup(&reference, &accumulated_text, &dc_poll, &ic_poll);
                                let cells = cells.borrow();
                                if let Some(cell) = cells.get(idx) {
                                    // Reuse existing label to avoid expensive
                                    // widget destroy+create on every streaming
                                    // delta — just update the markup in place.
                                    let reused = cell
                                        .first_child()
                                        .and_then(|w| w.downcast::<gtk4::Label>().ok())
                                        .map(|label| {
                                            label.set_markup(&markup);
                                            labels_poll.borrow_mut()[idx] = Some(label);
                                        })
                                        .is_some();
                                    if !reused {
                                        // First update replacing a spinner or
                                        // error — do the full widget swap once.
                                        while let Some(child) = cell.first_child() {
                                            cell.remove(&child);
                                        }
                                        let label = make_transcript_label("");
                                        label.set_markup(&markup);
                                        cell.append(&label);
                                        labels_poll.borrow_mut()[idx] = Some(label);
                                    }
                                }
                                drop(cells);
                                cands.borrow_mut()[idx].2 = accumulated_text.clone();

                                // If this is the selected row, update
                                // the text area so the user sees the
                                // streaming transcription live, and
                                // save the pick (debounced).
                                let is_selected = list
                                    .selected_row()
                                    .is_some_and(|r| r.index() as usize == idx);
                                if is_selected {
                                    buf_poll.set_text(&accumulated_text);
                                    // The buffer change triggers
                                    // `connect_changed` which marks
                                    // the pick dirty and arms the
                                    // debounce save timer.
                                }
                                // First stream update acts as "result
                                // arrived" — re-enable the button and
                                // flip to "↻" once text exists.
                                let has_text = !accumulated_text.is_empty();
                                if let Some(btn) = buttons_poll.borrow().get(idx) {
                                    btn.set_sensitive(true);
                                    set_action_icon(btn, has_text);
                                }
                                if let Some(flag) = has_tx_poll.borrow_mut().get_mut(idx) {
                                    *flag = has_text;
                                }
                            }
                        }
                        Ok(PickerMessage::CandidateStatus {
                            provider,
                            model,
                            streaming,
                            status_text,
                        }) => {
                            // Match on (provider, model, streaming).
                            // Step 10 of the transport-consolidation
                            // plan wired the realtime path's sink so
                            // both one-shot AND realtime rows receive
                            // status updates; the `streaming` flag
                            // disambiguates them when both rows exist
                            // for the same (provider, model) pair.
                            let idx = {
                                let cands = cands.borrow();
                                cands.iter().position(|(p, m, _, _, _, s, _, _)| {
                                    *p == provider && *m == model && *s == streaming
                                })
                            };
                            if let Some(idx) = idx {
                                // Skip if a final candidate has already
                                // arrived for this row — its content
                                // takes precedence over status updates.
                                let already_has_text =
                                    has_tx_poll.borrow().get(idx).copied().unwrap_or(false);
                                let already_errored =
                                    cands.borrow().get(idx).map(|t| t.4).unwrap_or(false);
                                if already_has_text || already_errored {
                                    continue;
                                }

                                // Replace the cell content (spinner or
                                // previous status label) with an italic
                                // dimmed status line.  Pango markup:
                                // `<i>` for italic; opacity is set on
                                // the widget (Pango `alpha` only
                                // supports text colour, not the
                                // background, and renders with the
                                // theme's foreground colour anyway).
                                let cells = cells.borrow();
                                if let Some(cell) = cells.get(idx) {
                                    while let Some(child) = cell.first_child() {
                                        cell.remove(&child);
                                    }
                                    let label = make_transcript_label("");
                                    label.set_markup(&format!(
                                        "<i>{}</i>",
                                        glib::markup_escape_text(&status_text),
                                    ));
                                    label.set_opacity(0.55);
                                    cell.append(&label);
                                    // Don't store this label in
                                    // `labels_poll` — that's reserved
                                    // for transcript labels used by
                                    // diff updates.  When the final
                                    // `Candidate` arrives it clears
                                    // and replaces the cell anyway.
                                }

                                // Surface the live status via the
                                // action button's tooltip so users who
                                // hover the disabled T button see
                                // *why* it's disabled and what phase
                                // the request is in.
                                if let Some(btn) = buttons_poll.borrow().get(idx) {
                                    btn.set_tooltip_text(Some(&status_text));
                                }
                            }
                        }
                        Ok(PickerMessage::InitialOneShotDone) => {
                            // Mark rows that still show a spinner as
                            // failed — the API never responded.  Rows
                            // whose spinner was already replaced (by a
                            // transcript, placeholder, or error) are
                            // naturally skipped.
                            let cells_ref = cells.borrow();
                            let failed: Vec<usize> = {
                                let mut cands = cands.borrow_mut();
                                let indices: Vec<usize> = (0..cands.len())
                                    .filter(|i| {
                                        initial_oneshot_indices.contains(i)
                                            && cells_ref.get(*i).is_some_and(|cell| {
                                                cell.first_child().is_some_and(|w| {
                                                    w.downcast_ref::<gtk4::Spinner>().is_some()
                                                })
                                            })
                                    })
                                    .collect();
                                for &idx in &indices {
                                    cands[idx].4 = true;
                                }
                                indices
                            };
                            drop(cells_ref);
                            for idx in failed {
                                make_err(&cells.borrow()[idx], idx, "no response");
                                // Re-enable "T" so the user can retry
                                // the non-responsive model.
                                if let Some(btn) = buttons_poll.borrow().get(idx) {
                                    btn.set_sensitive(true);
                                    set_action_icon(btn, false);
                                    btn.set_tooltip_text(None);
                                }
                                // No response means no transcription.
                                if let Some(flag) = has_tx_poll.borrow_mut().get_mut(idx) {
                                    *flag = false;
                                }
                            }
                            list.grab_focus();
                            // Keep polling — retries and realtime
                            // results may still arrive.
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                            break;
                        }
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            // All senders dropped (including retry
                            // listener) — stop polling.
                            *source_callback.borrow_mut() = None;
                            return glib::ControlFlow::Break;
                        }
                    }
                }
                glib::ControlFlow::Continue
            });
            *source.borrow_mut() = Some(id);
        }

        session.mount(
            wf_overlay.upcast_ref(),
            player_controls.upcast_ref(),
            content.upcast_ref(),
            navigation,
            navigation_callback,
        );
        #[cfg(test)]
        drive_gtk_test_window(GtkTestWindowRefs {
            audio_path: &audio_path_for_test,
            previous: &session.previous_button,
            next: &session.next_button,
            buffer: &text_buffer,
            error_label: &save_error_label,
            list: &list,
            action_buttons: &action_buttons,
            selection_ctx: &selection_ctx,
            window: &window,
        });
        list.grab_focus();

        // Initialize the audio player after the window is presented so
        // the picker appears instantly instead of blocking on cpal
        // device probing (~1-2 s on PipeWire).
        #[cfg(not(test))]
        {
            let player_init = Rc::clone(&player);
            let play_btn_init = play_btn.clone();
            let source = source_slot(&source_registry);
            let source_callback = Rc::clone(&source);
            #[cfg(test)]
            let source_generation = gtk_test_generation;
            let id = glib::idle_add_local_once(move || {
                #[cfg(test)]
                note_gtk_test_source_tick(source_generation);
                *source_callback.borrow_mut() = None;
                match WavPlayer::new() {
                    Ok(p) => {
                        *player_init.borrow_mut() = Some(p);
                        play_btn_init.set_sensitive(true);
                    }
                    Err(e) => {
                        log::warn!("audio output unavailable, play button disabled: {}", e);
                    }
                }
            });
            *source.borrow_mut() = Some(id);
        }

        main_loop.run();
        session.clear_navigation_callback();
        list.disconnect(row_activated_handler);
        close_btn.disconnect(close_handler);
        key_ctl.disconnect(key_handler);
        window.remove_controller(&key_ctl);
        window.disconnect(window_close_handler);

        // Publish the candidate list so the caller can look up the
        // selected index after this thread finishes.
        if let Ok(mut r) = results_for_gtk.lock() {
            *r = local_candidates.borrow().clone();
        }

        Ok(())
    };

    // ── Parallel transcriptions ─────────────────────────────────
    // Fire initial one-shot.
    let done_tx = msg_tx.clone();
    let retry_msg_tx = msg_tx.clone();

    // Realtime consumers decode on first use. The shared OnceCell keeps
    // concurrent rows and later retries on one decode for this record.
    let realtime_samples = RealtimeSamples::new(audio_path.clone());

    // Spawn realtime transcription tasks.
    for (provider, model, transcriber) in realtime_transcribers {
        let tx = msg_tx.clone();
        let realtime_samples = realtime_samples.clone();
        tokio::spawn(async move {
            let Some(samples) = realtime_samples.get().await else {
                let _ = tx.send(PickerMessage::Candidate(Box::new(PickerCandidate::error(
                    provider,
                    model,
                    "failed to read WAV samples".into(),
                    true,
                ))));
                return;
            };
            // The cancel token is fresh per spawn site; the picker
            // GTK Stop button (Step 12b — follow-up) will wire a
            // per-row token via a message channel.  For now the
            // transport layer accepts the token and respects it,
            // unlocking cross-process cancellation via the jobs
            // module (see also `picker/mod.rs::register_local`).
            let cancel = tokio_util::sync::CancellationToken::new();
            run_realtime_transcription(transcriber, samples, tx, provider, model, cancel).await;
        });
    }

    drop(msg_tx); // only clones survive from here
    tokio::spawn({
        let audio = audio_path.clone();
        let config = config.clone();
        let tx = done_tx.clone();
        async move {
            let mut tasks = tokio::task::JoinSet::new();
            for (provider, model) in transcribers {
                spawn_transcription(
                    &mut tasks,
                    tx.clone(),
                    audio.clone(),
                    provider,
                    model,
                    config.clone(),
                    tokio_util::sync::CancellationToken::new(),
                );
            }
            drop(tx);
            while (tasks.join_next().await).is_some() {}
            // All initial one-shot tasks finished — notify GTK.
            let _ = done_tx.send(PickerMessage::InitialOneShotDone);
        }
    });

    // ── Retry listener ──────────────────────────────────────────
    // Receives (provider, model, streaming) from the GTK retry button
    // and fires a new transcription, reusing the same result channel.
    {
        let audio = audio_path;
        let config = config.clone();
        let tx = retry_msg_tx;
        let realtime_samples = realtime_samples;
        tokio::spawn(async move {
            while let Some((provider, model, streaming)) = retry_rx.recv().await {
                log::info!("retrying {}:{} (streaming={})", provider, model, streaming);
                if streaming {
                    // Realtime retry: re-stream WAV samples.
                    let Some(samples) = realtime_samples.get().await else {
                        let _ =
                            tx.send(PickerMessage::Candidate(Box::new(PickerCandidate::error(
                                provider,
                                model,
                                "WAV samples unavailable".into(),
                                true,
                            ))));
                        continue;
                    };
                    match transcription::create_realtime_transcriber(
                        config.as_ref(),
                        provider,
                        Some(&model),
                    ) {
                        Ok(transcriber) => {
                            let tx = tx.clone();
                            let model_clone = model.clone();
                            tokio::spawn(async move {
                                let cancel = tokio_util::sync::CancellationToken::new();
                                run_realtime_transcription(
                                    transcriber,
                                    samples,
                                    tx,
                                    provider,
                                    model_clone,
                                    cancel,
                                )
                                .await;
                            });
                        }
                        Err(e) => {
                            log::warn!(
                                "retry {}:{} realtime transcriber creation failed: {}",
                                provider,
                                model,
                                e
                            );
                            let _ = tx.send(PickerMessage::Candidate(Box::new(
                                PickerCandidate::error(provider, model, format!("{e}"), true),
                            )));
                        }
                    }
                } else {
                    // Parakeet download-on-demand.  The GTK click
                    // handler already obtained explicit user consent
                    // via the AlertDialog before sending us this
                    // (provider, model) pair, so it is safe to fetch
                    // the model files here without prompting again.
                    // The actual transcription via `spawn_transcription`
                    // calls `model::ensure_present` internally and
                    // would error otherwise — by downloading first we
                    // turn that would-be error into a successful
                    // transcription.  Done on the async listener
                    // task (not the GTK thread) so the UI stays
                    // responsive while the ~640 MB download runs.
                    #[cfg(feature = "parakeet")]
                    if provider == Provider::Parakeet {
                        use crate::transcription::parakeet::{consent, model as parakeet_model};
                        match consent::resolve(config.as_ref()) {
                            Ok(st) if !st.present => {
                                log::info!(
                                    "picker: downloading Parakeet model to {} (consent given via dialog)",
                                    st.model_dir.display(),
                                );
                                if let Err(e) =
                                    parakeet_model::download_model(&st.model_dir, st.variant).await
                                {
                                    log::warn!("picker: Parakeet model download failed: {}", e);
                                    let _ = tx.send(PickerMessage::Candidate(Box::new(
                                        PickerCandidate::error(
                                            provider,
                                            model.clone(),
                                            format!("{e}"),
                                            false,
                                        ),
                                    )));
                                    continue;
                                }
                            }
                            Ok(_) => {}
                            Err(e) => {
                                log::warn!("picker: Parakeet consent resolve failed: {}", e);
                            }
                        }
                    }
                    let mut tasks = tokio::task::JoinSet::new();
                    spawn_transcription(
                        &mut tasks,
                        tx.clone(),
                        audio.clone(),
                        provider,
                        model,
                        config.clone(),
                        tokio_util::sync::CancellationToken::new(),
                    );
                    while (tasks.join_next().await).is_some() {}
                }
            }
            // retry_tx dropped (GTK closed) — exit.
        });
    }

    // ── Wait for result ─────────────────────────────────────────
    gtk_window()?;

    let picker_exit = sel_rx
        .await
        .map_err(|_| TalkError::Config("GTK picker closed unexpectedly".into()))?;

    match picker_exit {
        PickerExit::Selected(idx) => {
            let items = results
                .lock()
                .map_err(|_| TalkError::Config("results lock poisoned".into()))?;
            if idx < items.len() {
                let (_p, _m, t, cached, _is_error, _s, _segments, _metadata) = items[idx].clone();
                Ok(PickerOutcome::Selected(PickerSelection {
                    text: t,
                    is_cached: cached,
                }))
            } else {
                Ok(PickerOutcome::Cancelled)
            }
        }
        PickerExit::Navigate(direction) => Ok(PickerOutcome::Navigate(direction)),
        PickerExit::Cancelled => Ok(PickerOutcome::Cancelled),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcription::TranscriptionEvent;

    struct ScriptedRealtime {
        events: Vec<TranscriptionEvent>,
        connect_error: bool,
    }

    #[async_trait::async_trait]
    impl crate::transcription::RealtimeTranscriber for ScriptedRealtime {
        async fn validate(&self) -> Result<(), TalkError> {
            Ok(())
        }

        async fn transcribe_realtime(
            &self,
            _audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
        ) -> Result<tokio::sync::mpsc::Receiver<TranscriptionEvent>, TalkError> {
            if self.connect_error {
                return Err(TalkError::Transcription("upgrade rejected".into()));
            }
            let (tx, rx) = tokio::sync::mpsc::channel(8);
            for event in &self.events {
                tx.try_send(event.clone()).expect("scripted event fits");
            }
            Ok(rx)
        }
    }

    #[tokio::test]
    async fn realtime_adapter_preserves_final_text_and_timed_segments() {
        let (tx, rx) = std::sync::mpsc::channel();
        crate::dictate::picker::backend::run_realtime_transcription(
            Box::new(ScriptedRealtime {
                events: vec![
                    TranscriptionEvent::TextDelta {
                        text: "provisional".into(),
                    },
                    TranscriptionEvent::SegmentDelta {
                        text: "corrected words".into(),
                        start: Some(0.25),
                        end: Some(1.5),
                    },
                    TranscriptionEvent::Done,
                ],
                connect_error: false,
            }),
            std::sync::Arc::new(vec![100; 480]),
            tx,
            Provider::OpenAI,
            "live-model".into(),
            tokio_util::sync::CancellationToken::new(),
        )
        .await;

        match rx.recv().expect("provisional update") {
            PickerMessage::StreamUpdate {
                provider,
                model,
                accumulated_text,
            } => {
                assert_eq!(
                    (provider, model.as_str(), accumulated_text.as_str()),
                    (Provider::OpenAI, "live-model", "provisional")
                );
            }
            _ => panic!("expected provisional update"),
        }
        match rx.recv().expect("corrected update") {
            PickerMessage::StreamUpdate {
                accumulated_text, ..
            } => {
                assert_eq!(accumulated_text, "corrected words");
            }
            _ => panic!("expected corrected update"),
        }
        match rx.recv().expect("final candidate") {
            PickerMessage::Candidate(candidate) => {
                assert_eq!(candidate.provider, Provider::OpenAI);
                assert_eq!(candidate.model, "live-model");
                assert_eq!(candidate.text, "corrected words");
                assert!(candidate.streaming);
                assert_eq!(candidate.error, None);
                assert_eq!(
                    candidate.segments.map(|segments| segments
                        .into_iter()
                        .map(|segment| (segment.start, segment.end, segment.text))
                        .collect::<Vec<_>>()),
                    Some(vec![(0.25, 1.5, "corrected words".to_string())])
                );
            }
            _ => panic!("expected final candidate"),
        }
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn realtime_adapter_reports_upgrade_failure_as_error_candidate() {
        let (tx, rx) = std::sync::mpsc::channel();
        crate::dictate::picker::backend::run_realtime_transcription(
            Box::new(ScriptedRealtime {
                events: Vec::new(),
                connect_error: true,
            }),
            std::sync::Arc::new(vec![]),
            tx,
            Provider::OpenAI,
            "live-model".into(),
            tokio_util::sync::CancellationToken::new(),
        )
        .await;
        match rx.recv().expect("error candidate") {
            PickerMessage::Candidate(candidate) => {
                assert_eq!(candidate.text, "");
                assert_eq!(
                    candidate.error.as_deref(),
                    Some("Transcription error: upgrade rejected")
                );
                assert!(candidate.streaming);
            }
            _ => panic!("expected error candidate"),
        }
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn concurrent_realtime_sample_consumers_share_one_decode() {
        GTK_TEST_REALTIME_DECODES.store(0, std::sync::atomic::Ordering::SeqCst);
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sine_440_0.5s_mono.m4a");
        let samples = RealtimeSamples::new(path);

        let (first, second) = tokio::join!(samples.get(), samples.get());
        let first = first.expect("first consumer receives decoded samples");
        let second = second.expect("second consumer receives decoded samples");

        assert!(!first.is_empty());
        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert_eq!(
            GTK_TEST_REALTIME_DECODES.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[test]
    fn persist_selection_clears_dirty_after_successful_write() {
        let temp = tempfile::TempDir::new().expect("tempdir");
        let audio_path = temp.path().join("recording.ogg");
        std::fs::write(&audio_path, b"").expect("audio fixture");
        let mut ctx = SelectionSaveCtx {
            audio_path: audio_path.clone(),
            dirty: true,
            current: Some((Provider::Mistral, "voxtral-test".to_string(), false)),
        };

        persist_selection(&mut ctx, "edited transcript").expect("save selection");

        assert!(!ctx.dirty);
        assert_eq!(
            crate::recording_cache::read_pick(&audio_path),
            Some((
                Provider::Mistral,
                "voxtral-test".to_string(),
                false,
                "edited transcript".to_string(),
            ))
        );
    }

    #[test]
    fn persist_selection_keeps_dirty_when_write_fails() {
        let mut ctx = SelectionSaveCtx {
            audio_path: PathBuf::from("/fixture/recording.ogg"),
            dirty: true,
            current: Some((Provider::OpenAI, "whisper-test".to_string(), false)),
        };

        let result = persist_selection_with(&mut ctx, "edited transcript", |_, _, _, _, _| {
            Err(TalkError::Config("injected write failure".to_string()))
        });

        assert!(
            matches!(result, Err(TalkError::Config(message)) if message == "injected write failure")
        );
        assert!(ctx.dirty);
    }
}
