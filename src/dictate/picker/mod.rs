//! Pick-mode logic for the dictate command.
//!
//! When `--pick` is passed, the user selects among multiple transcription
//! providers/models via a GTK picker window.  Cached results are reused
//! when available.

mod backend;
mod ui;

use crate::config::{Config, Provider};
use crate::error::TalkError;
use crate::paste::{paste_with_root, PasteNode, PasteTiming};
use crate::recording_cache;
use crate::transcription::{self, RealtimeTranscriber};
use crate::x11::x11_centre_and_raise;
use std::path::PathBuf;

use super::models::{build_retry_candidates, resolve_model, resolve_provider};
use ui::{
    initialize_picker_gtk, pick_with_streaming_gtk, PickerNavigation, PickerNavigationAvailability,
    PickerOutcome, PickerSelection, PickerSession, PickerUiInput, PICKER_TITLE,
};

type PickerCandidateKey = (Provider, String, bool);

struct PickRecordContext<'a> {
    cached_brief: Option<&'a recording_cache::RecordingMetadataBrief>,
    is_original: bool,
    reached_by_navigation: bool,
    navigation: PickerNavigationAvailability,
}

#[derive(Clone)]
struct PickerPosition {
    audio_path: PathBuf,
    original_audio: PathBuf,
    reached_by_navigation: bool,
}

impl PickerPosition {
    fn new(audio_path: PathBuf, original_audio: PathBuf) -> Self {
        Self {
            audio_path,
            original_audio,
            reached_by_navigation: false,
        }
    }

    fn is_original(&self) -> bool {
        self.audio_path == self.original_audio
    }
}

enum PickerTransition {
    Open(PickerPosition),
    Stay,
    Gone,
    Cancelled,
    Selected(PickerSelection),
}

fn transition_picker(
    mut position: PickerPosition,
    outcome: PickerOutcome,
    target: Option<PathBuf>,
    target_available: bool,
    current_available: bool,
) -> PickerTransition {
    match outcome {
        PickerOutcome::Cancelled => PickerTransition::Cancelled,
        PickerOutcome::Selected(selection) => PickerTransition::Selected(selection),
        PickerOutcome::Navigate(_) => match target {
            Some(path) if target_available => {
                position.audio_path = path;
                position.reached_by_navigation = true;
                PickerTransition::Open(position)
            }
            _ if current_available => PickerTransition::Stay,
            _ => PickerTransition::Gone,
        },
    }
}

struct PreparedRecord {
    cached_entries: Vec<PickerCachedEntry>,
    oneshot_candidates: Vec<(Provider, String)>,
    realtime_candidates: Vec<(Provider, String)>,
    deferred_candidates: Vec<PickerCandidateKey>,
}

/// Parameters for the pick-mode path.
pub(crate) struct PickParams {
    pub input_audio_file: Option<PathBuf>,
    pub cached_brief: Option<recording_cache::RecordingMetadataBrief>,
    pub replace_char_count: Option<usize>,
    pub replace_last_paste: bool,
    pub provider: Option<Provider>,
    pub model: Option<String>,
    pub target_window: Option<String>,
    /// Pre-built paste-node tree shared with the dictate one-shot
    /// path.  `Arc` rather than `Box` so the same root can be cloned
    /// into both the one-shot and pick paths.
    pub paste_root: std::sync::Arc<dyn PasteNode>,
    pub paste_timing: PasteTiming,
}

/// Run pick mode: show a GTK picker with cached and live transcriptions.
///
/// Returns `Ok(())` when the user selects a result (or cancels).
pub(crate) async fn run_pick(config: Config, params: PickParams) -> Result<(), TalkError> {
    // Single-instance: if a picker window is already open, just
    // raise and focus it instead of opening a second one.
    if x11_centre_and_raise(PICKER_TITLE) {
        log::info!("picker already open — raised existing window");
        return Ok(());
    }

    let audio_path = params.input_audio_file.clone().ok_or_else(|| {
        TalkError::Config("--pick requires --input-audio-file or --retry-last".to_string())
    })?;
    if !audio_path.exists() {
        return Err(TalkError::Config(format!(
            "input audio file not found: {}",
            audio_path.display()
        )));
    }

    let original_audio = std::fs::canonicalize(&audio_path).map_err(|error| {
        TalkError::Config(format!(
            "failed to resolve input audio file {}: {}",
            audio_path.display(),
            error
        ))
    })?;
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        initialize_picker_gtk()?;
        let session = PickerSession::new();
        let result = handle.block_on(run_pick_loop(
            std::sync::Arc::new(config),
            params,
            original_audio,
            &session,
        ));
        session.destroy();
        result
    })
    .await
    .map_err(|error| TalkError::Config(format!("GTK picker task failed: {error}")))?
}

async fn run_pick_loop(
    config: std::sync::Arc<Config>,
    params: PickParams,
    original_audio: PathBuf,
    session: &PickerSession,
) -> Result<(), TalkError> {
    let selection = drive_picker(config.as_ref(), original_audio, |position, navigation| {
        let config = config.clone();
        let params = &params;
        async move {
            let is_original = position.is_original();
            run_pick_record(
                config,
                params,
                position.audio_path,
                PickRecordContext {
                    cached_brief: if is_original {
                        params.cached_brief.as_ref()
                    } else {
                        None
                    },
                    is_original,
                    reached_by_navigation: position.reached_by_navigation,
                    navigation,
                },
                session,
            )
            .await
        }
    })
    .await?;
    session.destroy();
    if let Some(selection) = selection {
        paste_picker_selection(&params, selection).await?;
    }
    Ok(())
}

async fn drive_picker<F, Fut>(
    config: &Config,
    original_audio: PathBuf,
    mut show: F,
) -> Result<Option<PickerSelection>, TalkError>
where
    F: FnMut(PickerPosition, PickerNavigationAvailability) -> Fut,
    Fut: std::future::Future<Output = Result<PickerOutcome, TalkError>>,
{
    let mut position = PickerPosition::new(original_audio.clone(), original_audio);
    loop {
        let audio_path = &position.audio_path;
        let navigation = crate::record::recording_navigation(audio_path, &config.output_dir)?;
        let navigation_availability = PickerNavigationAvailability {
            previous: navigation
                .as_ref()
                .and_then(|value| value.previous.as_ref())
                .is_some(),
            next: navigation
                .as_ref()
                .and_then(|value| value.next.as_ref())
                .is_some(),
        };
        let outcome = show(position.clone(), navigation_availability).await?;

        let target = if let PickerOutcome::Navigate(direction) = &outcome {
            let refreshed = crate::record::recording_navigation(audio_path, &config.output_dir)?;
            refreshed.and_then(|value| match direction {
                PickerNavigation::Previous => value.previous,
                PickerNavigation::Next => value.next,
            })
        } else {
            None
        };
        let target_available = target.as_ref().is_some_and(|path| path.is_file());
        let current_available = audio_path.is_file();
        let transition = transition_picker(
            position.clone(),
            outcome,
            target.clone(),
            target_available,
            current_available,
        );
        match transition {
            PickerTransition::Cancelled | PickerTransition::Gone => {
                return Ok(None);
            }
            PickerTransition::Open(next) => {
                position = next;
                continue;
            }
            PickerTransition::Stay => {
                if let Some(path) = target {
                    log::warn!(
                        "picker navigation target became unavailable: {}",
                        path.display()
                    );
                } else {
                    log::debug!("picker navigation reached a collection boundary");
                }
            }
            PickerTransition::Selected(selection) => {
                return Ok(Some(selection));
            }
        }
    }
}

async fn run_pick_record(
    config: std::sync::Arc<Config>,
    params: &PickParams,
    audio_path: PathBuf,
    context: PickRecordContext<'_>,
    session: &PickerSession,
) -> Result<PickerOutcome, TalkError> {
    present_record(config, params, audio_path, context, |input| {
        pick_with_streaming_gtk(session, input)
    })
    .await
}

async fn present_record<F, Fut>(
    config: std::sync::Arc<Config>,
    params: &PickParams,
    audio_path: PathBuf,
    context: PickRecordContext<'_>,
    present: F,
) -> Result<PickerOutcome, TalkError>
where
    F: FnOnce(PickerUiInput) -> Fut,
    Fut: std::future::Future<Output = Result<PickerOutcome, TalkError>>,
{
    let PickRecordContext {
        cached_brief,
        is_original,
        reached_by_navigation,
        navigation,
    } = context;
    let prepared = prepare_record_input(
        config.as_ref(),
        params,
        &audio_path,
        cached_brief,
        reached_by_navigation,
    )
    .await;
    let PreparedRecord {
        cached_entries,
        oneshot_candidates: oneshot_filtered,
        realtime_candidates: realtime_filtered,
        deferred_candidates: deferred,
    } = prepared;

    let mut rt_transcribers: Vec<(Provider, String, Box<dyn RealtimeTranscriber>)> = Vec::new();
    for (provider, model) in realtime_filtered {
        match transcription::create_realtime_transcriber(config.as_ref(), provider, Some(&model)) {
            Ok(t) => rt_transcribers.push((provider, model, t)),
            Err(e) => log::warn!("skipping realtime {}:{}: {}", provider, model, e),
        }
    }
    if oneshot_filtered.is_empty()
        && cached_entries.is_empty()
        && rt_transcribers.is_empty()
        && deferred.is_empty()
    {
        return Err(TalkError::Transcription(
            "no transcription providers available".to_string(),
        ));
    }

    let mut outcome = present(PickerUiInput {
        transcribers: oneshot_filtered,
        audio_path,
        cached_entries,
        config: config.clone(),
        realtime_transcribers: rt_transcribers,
        deferred_candidates: deferred,
        navigation,
    })
    .await?;
    if !is_original {
        if let PickerOutcome::Selected(selection) = &mut outcome {
            selection.is_cached = false;
        }
    }
    Ok(outcome)
}

async fn prepare_record_input(
    config: &Config,
    params: &PickParams,
    audio_path: &std::path::Path,
    cached_brief: Option<&recording_cache::RecordingMetadataBrief>,
    reached_by_navigation: bool,
) -> PreparedRecord {
    // Read the authoritative pick (user-confirmed selection + edited text).
    // This is the ONLY cross-provider source of truth for the picker's
    // selection state.  Sidecars are per-model internals probed below
    // via `transcribe_audio(allow_api=false)`.
    let pick = recording_cache::read_pick(audio_path);
    let selected_key: Option<(Provider, String, bool)> = pick
        .as_ref()
        .map(|(p, m, s, _)| (*p, m.clone(), *s))
        .or_else(|| {
            cached_brief.and_then(|b| {
                Some((
                    b.provider.as_deref()?.parse().ok()?,
                    b.model.as_deref()?.to_string(),
                    false,
                ))
            })
        });

    let mut all_entries: Vec<(Provider, String, String, bool)> = Vec::new();

    // Seed entry for the pick itself: the user's authoritative choice
    // goes first so it can be pre-selected.
    if let Some((p, m, s, t)) = pick.as_ref() {
        all_entries.push((*p, m.clone(), t.clone(), *s));
    }

    let candidates = build_retry_candidates(config, params.provider, params.model.as_deref());
    log::debug!("picker candidates: {} total", candidates.len());
    for (p, m, s) in &candidates {
        log::debug!("  candidate: {}:{} (streaming={})", p, m, s);
    }

    // Probe the per-model sidecar cache for each one-shot candidate via
    // Layer 3 with `allow_api=false`.  Hits populate `all_entries`
    // (no spinner needed).  Misses return `CacheOnly` — those models
    // remain uncached and will be shown as deferred buttons or the
    // default model's auto-firing row.
    for (p, m, streaming) in &candidates {
        if *streaming {
            continue; // realtime models have no sidecar cache
        }
        // Skip if we already have it (e.g. from the pick file).
        if all_entries
            .iter()
            .any(|(ep, em, _, es)| ep == p && em == m && !*es)
        {
            continue;
        }
        let sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink> =
            std::sync::Arc::new(crate::telemetry::NoOpSink);
        // `allow_api=false` short-circuits before any HTTP call, so
        // the `policy` here is purely a typing requirement — the
        // wall-clock branch in `send_once` is never reached.  Pass
        // `Proportional` (the function-default flavour) so this
        // call site does not look like it is asking for picker
        // semantics it cannot use.
        match transcription::transcribe_audio(
            audio_path,
            config,
            *p,
            Some(m),
            false,
            transcription::TranscribeOptions {
                allow_api: false,
                policy: transcription::RequestTimeoutPolicy::Proportional,
                cancel_token: None,
                skip_legacy_lock: false,
            },
            &sink,
        )
        .await
        {
            Ok(result) => {
                all_entries.push((*p, m.clone(), result.text, false));
            }
            Err(TalkError::CacheOnly) => {
                log::debug!("  sidecar cache miss: {}:{}", p, m);
            }
            Err(e) => {
                log::debug!("  sidecar probe failed: {}:{} — {}", p, m, e);
            }
        }
    }

    // Build cached_entries with is_primary flag. The authoritative saved
    // pick goes first so it is pre-selected on every record. A saved
    // selection does not establish that text was delivered to this target.
    // Tuple: (provider, model, text, is_primary, streaming)
    let cached_entries = prioritize_selected_entry(all_entries, selected_key);

    log::debug!(
        "picker cache: {} cached entries (primary={})",
        cached_entries.len(),
        cached_entries.iter().filter(|(_, _, _, p, _)| *p).count(),
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

    // Filter out every (provider, model, streaming) triple that
    // already has a cached result — no need to re-transcribe.
    let filtered = uncached_candidates(candidates, &cached_entries);
    log::debug!(
        "picker: {} transcribers needed (after filtering)",
        filtered.len(),
    );
    for (p, m, s) in &filtered {
        log::debug!("  needs API call: {}:{} (streaming={})", p, m, s);
    }

    // Resolve the default model so only it is transcribed immediately.
    // All other candidates are deferred — shown in the UI with a
    // "transcribe" button that the user can click on demand.
    let default_provider = resolve_provider(params.provider, config);
    let default_model = resolve_model(params.model.as_deref(), config, default_provider, false);
    log::debug!(
        "picker default model: {}:{} (one-shot)",
        default_provider,
        default_model,
    );

    let is_default = |p: &Provider, m: &str| *p == default_provider && m == default_model;

    // Arrow navigation must never send a paid request automatically.
    // Cached rows were already removed above; every remaining row stays
    // available through its existing T action.
    let (mut default_filtered, mut deferred) =
        split_transcription_candidates(filtered, is_default, !reached_by_navigation);

    // Parakeet immediate-default fallback.  When the default model
    // is Parakeet and its files are not on disk, demote the
    // candidate from the immediate set to the deferred list — so
    // the picker opens with a clickable "T" row instead of
    // auto-firing a transcription that would just error out (the
    // pipeline NEVER downloads silently).  The user then clicks T
    // and the picker's GTK click handler shows the consent
    // AlertDialog before the async retry listener fetches the model
    // and runs the transcription.  This keeps "open the picker" a
    // non-intrusive action (no auto-dialog at open) while still
    // surfacing the model as a first-class candidate.
    #[cfg(feature = "parakeet")]
    {
        let mut i = 0;
        while i < default_filtered.len() {
            if default_filtered[i].0 == Provider::Parakeet {
                let present = crate::transcription::parakeet::consent::resolve(config)
                    .map(|s| s.present)
                    .unwrap_or(true); // resolve failed → let the
                                      // transcribe path surface a
                                      // clean error instead of
                                      // silently deferring.
                if !present {
                    let cand = default_filtered.remove(i);
                    log::debug!(
                        "picker: default Parakeet model {} absent — deferring until user clicks T",
                        cand.1,
                    );
                    deferred.push(cand);
                    continue;
                }
            }
            i += 1;
        }
    }

    log::debug!(
        "picker: {} immediate, {} deferred",
        default_filtered.len(),
        deferred.len(),
    );

    // Split default candidates into one-shot and realtime groups.
    let oneshot_filtered: Vec<(Provider, String)> = default_filtered
        .iter()
        .filter(|(_, _, s)| !s)
        .map(|(p, m, _)| (*p, m.clone()))
        .collect();
    let realtime_filtered: Vec<(Provider, String)> = default_filtered
        .iter()
        .filter(|(_, _, s)| *s)
        .map(|(p, m, _)| (*p, m.clone()))
        .collect();

    PreparedRecord {
        cached_entries,
        oneshot_candidates: oneshot_filtered,
        realtime_candidates: realtime_filtered,
        deferred_candidates: deferred,
    }
}

type PickerCachedEntry = (Provider, String, String, bool, bool);

fn prioritize_selected_entry(
    mut entries: Vec<(Provider, String, String, bool)>,
    selected: Option<PickerCandidateKey>,
) -> Vec<PickerCachedEntry> {
    let mut cached = Vec::new();
    if let Some((provider, model, streaming)) = selected {
        if let Some(index) = entries
            .iter()
            .position(|(p, m, _, s)| *p == provider && *m == model && *s == streaming)
        {
            let (p, m, text, s) = entries.remove(index);
            cached.push((p, m, text, true, s));
        }
    }
    cached.extend(
        entries
            .into_iter()
            .map(|(p, m, text, s)| (p, m, text, false, s)),
    );
    cached
}

fn uncached_candidates(
    candidates: Vec<PickerCandidateKey>,
    cached: &[PickerCachedEntry],
) -> Vec<PickerCandidateKey> {
    candidates
        .into_iter()
        .filter(|(p, m, s)| {
            let dominated = cached
                .iter()
                .any(|(cp, cm, _, _, cs)| cp == p && cm == m && cs == s);
            if dominated {
                log::debug!("  filtered out (cached): {}:{} (streaming={})", p, m, s);
            }
            !dominated
        })
        .collect()
}

fn split_transcription_candidates<F>(
    filtered: Vec<PickerCandidateKey>,
    is_default: F,
    auto_transcribe: bool,
) -> (Vec<PickerCandidateKey>, Vec<PickerCandidateKey>)
where
    F: Fn(&Provider, &str) -> bool,
{
    let mut immediate = Vec::new();
    let mut deferred = Vec::new();
    for (provider, model, streaming) in filtered {
        if auto_transcribe && is_default(&provider, &model) {
            immediate.push((provider, model, streaming));
        } else {
            deferred.push((provider, model, streaming));
        }
    }
    (immediate, deferred)
}

async fn paste_picker_selection(
    params: &PickParams,
    selection: PickerSelection,
) -> Result<(), TalkError> {
    paste_picker_selection_with(params, selection, |text, delete_chars| async move {
        paste_with_root(
            params.paste_root.as_ref(),
            params.target_window.as_ref(),
            &text,
            delete_chars,
            None,
            &crate::telemetry::NoOpSink,
            params.paste_timing,
            None,
        )
        .await
    })
    .await
}

async fn paste_picker_selection_with<F, Fut>(
    params: &PickParams,
    selection: PickerSelection,
    paste: F,
) -> Result<(), TalkError>
where
    F: FnOnce(String, usize) -> Fut,
    Fut: std::future::Future<Output = Result<(), TalkError>>,
{
    // Selection is auto-saved by the picker UI (on first result,
    // debounced on row change, and on close).

    // A saved pick records the user's selection, not a paste into the
    // current target. Confirming it still delivers the selected text.
    if selection.is_cached {
        log::debug!("saved pick selected for delivery to current target");
    }

    let delete_chars = if params.replace_last_paste {
        // Prefer the paste-state file (written after every paste,
        // including picker selections) over recording metadata so
        // that successive picker replacements delete the correct
        // number of characters.
        replacement_char_count(
            recording_cache::read_last_paste_state()?.as_ref(),
            params.target_window.as_deref(),
            params.replace_char_count,
        )
    } else {
        0
    };

    paste(selection.text.clone(), delete_chars).await?;
    let _ =
        recording_cache::write_last_paste_state(params.target_window.as_deref(), &selection.text);
    println!("{}", selection.text);
    Ok(())
}

fn replacement_char_count(
    last_paste: Option<&recording_cache::LastPasteState>,
    target_window: Option<&str>,
    fallback: Option<usize>,
) -> usize {
    last_paste.map_or_else(
        || fallback.unwrap_or(0),
        |state| state.replacement_count_for(target_window),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, OpenAIConfig, ProvidersConfig};
    use crate::paste::PasteCtx;
    use crate::recording_cache::TranscriptionCache;
    use crate::transcription::TranscriptionResult;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    const PICKER_TEST_MODEL: &str = "aaa-picker-test";
    static PICKER_CACHE_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[test]
    fn saved_pick_stays_primary_and_suppresses_matching_candidate_only() {
        let picked = (Provider::OpenAI, "gpt-transcribe".to_string(), false);
        let cached = prioritize_selected_entry(
            vec![
                (Provider::Mistral, "voxtral".into(), "other".into(), false),
                (picked.0, picked.1.clone(), "user edit".into(), false),
            ],
            Some(picked.clone()),
        );
        assert_eq!(
            cached[0],
            (
                Provider::OpenAI,
                "gpt-transcribe".into(),
                "user edit".into(),
                true,
                false
            )
        );
        assert_eq!(
            cached[1],
            (
                Provider::Mistral,
                "voxtral".into(),
                "other".into(),
                false,
                false
            )
        );
        let remaining = uncached_candidates(
            vec![picked, (Provider::OpenAI, "gpt-transcribe".into(), true)],
            &cached,
        );
        assert_eq!(
            remaining,
            vec![(Provider::OpenAI, "gpt-transcribe".into(), true)]
        );
    }

    #[tokio::test]
    async fn confirming_saved_pick_from_input_file_pastes_to_current_target() {
        let _lock = PICKER_CACHE_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("temp cache");
        let _cache = EnvGuard::set("XDG_CACHE_HOME", dir.path());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut params = openai_picker_params(PathBuf::from("unused.ogg"));
        params.paste_root = Arc::new(RecordingPaste {
            calls: calls.clone(),
        });
        paste_picker_selection(
            &params,
            PickerSelection {
                text: "saved selection".into(),
                is_cached: true,
            },
        )
        .await
        .expect("cached selection");
        assert_eq!(*calls.lock().expect("paste calls"), vec!["saved selection"]);
    }

    #[test]
    fn replacement_prefers_last_paste_character_count() {
        let state = recording_cache::LastPasteState {
            timestamp: String::new(),
            char_count: 11,
            window_id: Some("42".into()),
            text: String::new(),
        };
        assert_eq!(
            replacement_char_count(Some(&state), Some("42"), Some(4)),
            11
        );
        assert_eq!(replacement_char_count(Some(&state), Some("43"), Some(4)), 0);
        assert_eq!(replacement_char_count(None, Some("42"), Some(4)), 4);
        assert_eq!(replacement_char_count(None, None, None), 0);
    }

    struct RecordingPaste {
        calls: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl PasteNode for RecordingPaste {
        async fn paste(&self, text: &str, _ctx: &PasteCtx<'_>) -> Result<(), TalkError> {
            self.calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(text.to_string());
            Ok(())
        }
    }

    struct EnvGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &std::path::Path) -> Self {
            let previous = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(value) = self.previous.as_ref() {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    struct ValidationCacheGuard {
        _path: EnvGuard,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl ValidationCacheGuard {
        fn new(path: &std::path::Path) -> Self {
            let lock = crate::transcription::transport::validate_cache::__TEST_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let path = EnvGuard::set("TALK_RS_VALIDATE_CACHE_PATH", path);
            crate::transcription::transport::validate_cache::__test_reset();
            Self {
                _path: path,
                _lock: lock,
            }
        }
    }

    impl Drop for ValidationCacheGuard {
        fn drop(&mut self) {
            crate::transcription::transport::validate_cache::__test_reset();
        }
    }

    #[derive(Clone)]
    struct CountingTranscriptionResponse {
        requests: Arc<std::sync::atomic::AtomicUsize>,
        text: &'static str,
    }

    impl Respond for CountingTranscriptionResponse {
        fn respond(&self, _request: &Request) -> ResponseTemplate {
            self.requests
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"text": self.text}))
        }
    }

    async fn mount_openai_picker_api(
        server: &MockServer,
        requests: Arc<std::sync::atomic::AtomicUsize>,
        text: &'static str,
        expected_transcriptions: u64,
    ) {
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"id": PICKER_TEST_MODEL}]
            })))
            .expect(1)
            .mount(server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(CountingTranscriptionResponse { requests, text })
            .expect(expected_transcriptions)
            .mount(server)
            .await;
    }

    fn openai_picker_config(output_dir: PathBuf, server: &MockServer) -> Config {
        Config {
            output_dir,
            providers: ProvidersConfig {
                mistral: None,
                openai: Some(OpenAIConfig {
                    api_key: "sk-picker-test".to_string(),
                    url: Some(server.uri()),
                    model: PICKER_TEST_MODEL.to_string(),
                    realtime_model: "gpt-live-transcribe".to_string(),
                    prompt: None,
                    keywords: None,
                    languages: None,
                    realtime_delay: None,
                }),
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        }
    }

    fn openai_picker_params(audio_path: PathBuf) -> PickParams {
        PickParams {
            input_audio_file: Some(audio_path),
            cached_brief: None,
            replace_char_count: None,
            replace_last_paste: false,
            provider: Some(Provider::OpenAI),
            model: Some(PICKER_TEST_MODEL.to_string()),
            target_window: None,
            paste_root: Arc::new(RecordingPaste {
                calls: Arc::new(Mutex::new(Vec::new())),
            }),
            paste_timing: PasteTiming::default(),
        }
    }

    #[test]
    fn navigated_candidates_defer_default_without_reordering_rows() {
        let candidates = vec![
            (Provider::OpenAI, "aaa".to_string(), false),
            (Provider::OpenAI, PICKER_TEST_MODEL.to_string(), false),
            (Provider::OpenAI, "zzz".to_string(), true),
        ];
        let is_default = |provider: &Provider, model: &str| {
            *provider == Provider::OpenAI && model == PICKER_TEST_MODEL
        };

        let (initial_immediate, initial_deferred) =
            split_transcription_candidates(candidates.clone(), is_default, true);
        assert_eq!(
            initial_immediate,
            vec![(Provider::OpenAI, PICKER_TEST_MODEL.to_string(), false)]
        );
        assert_eq!(
            initial_deferred,
            vec![
                (Provider::OpenAI, "aaa".to_string(), false),
                (Provider::OpenAI, "zzz".to_string(), true),
            ]
        );

        let (navigated_immediate, navigated_deferred) =
            split_transcription_candidates(candidates.clone(), is_default, false);
        assert!(navigated_immediate.is_empty());
        assert_eq!(navigated_deferred, candidates);
    }

    #[test]
    fn navigation_transitions_include_return_to_original_and_terminal_outcomes() {
        let original = PathBuf::from("/recordings/original.ogg");
        let neighbor = PathBuf::from("/recordings/neighbor.ogg");
        let initial = PickerPosition::new(original.clone(), original.clone());
        let forward = transition_picker(
            initial,
            PickerOutcome::Navigate(PickerNavigation::Next),
            Some(neighbor.clone()),
            true,
            true,
        );
        let PickerTransition::Open(next) = forward else {
            panic!("expected navigation");
        };
        assert_eq!(next.audio_path, neighbor);
        assert!(next.reached_by_navigation);
        assert!(!next.is_original());
        let back = transition_picker(
            next,
            PickerOutcome::Navigate(PickerNavigation::Previous),
            Some(original.clone()),
            true,
            true,
        );
        let PickerTransition::Open(returned) = back else {
            panic!("expected return");
        };
        assert!(returned.is_original());
        assert!(returned.reached_by_navigation);
        assert!(matches!(
            transition_picker(
                returned.clone(),
                PickerOutcome::Navigate(PickerNavigation::Previous),
                None,
                false,
                true
            ),
            PickerTransition::Stay
        ));
        assert!(matches!(
            transition_picker(
                returned.clone(),
                PickerOutcome::Navigate(PickerNavigation::Next),
                Some(neighbor),
                false,
                false
            ),
            PickerTransition::Gone
        ));
        assert!(matches!(
            transition_picker(
                returned.clone(),
                PickerOutcome::Cancelled,
                None,
                false,
                true
            ),
            PickerTransition::Cancelled
        ));
        assert!(matches!(
            transition_picker(returned, PickerOutcome::Selected(PickerSelection { text: "chosen".into(), is_cached: true }), None, false, true),
            PickerTransition::Selected(PickerSelection { text, .. }) if text == "chosen"
        ));
    }

    #[tokio::test]
    async fn record_preparation_uses_pick_and_sidecar_without_paid_requests() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("record.ogg");
        std::fs::write(&audio, b"audio").expect("audio fixture");
        recording_cache::write_pick(&audio, "openai", PICKER_TEST_MODEL, false, "edited")
            .expect("saved pick");
        TranscriptionCache::store(
            &audio,
            Provider::OpenAI,
            "whisper-1",
            false,
            &TranscriptionResult {
                text: "other cached".into(),
                ..Default::default()
            },
        )
        .expect("sidecar");
        let config = Config {
            output_dir: dir.path().to_path_buf(),
            providers: ProvidersConfig {
                mistral: None,
                openai: Some(OpenAIConfig {
                    api_key: "unused".into(),
                    url: None,
                    model: PICKER_TEST_MODEL.into(),
                    realtime_model: "gpt-live-transcribe".into(),
                    prompt: None,
                    keywords: None,
                    languages: None,
                    realtime_delay: None,
                }),
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };
        let params = openai_picker_params(audio.clone());
        let initial = prepare_record_input(&config, &params, &audio, None, false).await;
        assert_eq!(initial.cached_entries[0].2, "edited");
        assert!(initial.cached_entries[0].3);
        assert!(initial
            .cached_entries
            .iter()
            .any(|(_, model, text, _, _)| model == "whisper-1" && text == "other cached"));
        assert!(!initial
            .oneshot_candidates
            .iter()
            .any(|(_, model)| model == PICKER_TEST_MODEL));
        let navigated = prepare_record_input(&config, &params, &audio, None, true).await;
        assert!(navigated.oneshot_candidates.is_empty());
        assert!(navigated
            .deferred_candidates
            .iter()
            .any(|(_, model, streaming)| model != PICKER_TEST_MODEL && !streaming));
    }

    #[tokio::test]
    async fn default_candidate_is_immediate_only_on_first_open() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("record.ogg");
        std::fs::write(&audio, b"audio").expect("audio fixture");
        let config = Config {
            output_dir: dir.path().to_path_buf(),
            providers: ProvidersConfig {
                mistral: None,
                openai: Some(OpenAIConfig {
                    api_key: "unused".into(),
                    url: None,
                    model: PICKER_TEST_MODEL.into(),
                    realtime_model: "gpt-live-transcribe".into(),
                    prompt: None,
                    keywords: None,
                    languages: None,
                    realtime_delay: None,
                }),
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };
        let params = openai_picker_params(audio.clone());
        let initial = prepare_record_input(&config, &params, &audio, None, false).await;
        assert_eq!(
            initial.oneshot_candidates,
            vec![(Provider::OpenAI, PICKER_TEST_MODEL.into())]
        );
        assert!(!initial
            .deferred_candidates
            .iter()
            .any(|(_, model, _)| model == PICKER_TEST_MODEL));
        let returned = prepare_record_input(&config, &params, &audio, None, true).await;
        assert!(returned.oneshot_candidates.is_empty());
        assert!(returned
            .deferred_candidates
            .iter()
            .any(|(_, model, streaming)| model == PICKER_TEST_MODEL && !streaming));
    }

    #[tokio::test]
    async fn record_loop_navigates_both_ways_and_confirms_only_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        let original = dir.path().join("2026-04-02T10-00-00+0200.ogg");
        let older = dir.path().join("2026-04-01T10-00-00+0200.ogg");
        std::fs::write(&original, b"audio").expect("original");
        std::fs::write(&older, b"audio").expect("older");
        let config = Config {
            output_dir: dir.path().to_path_buf(),
            providers: ProvidersConfig {
                mistral: None,
                openai: None,
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };
        let actions = std::collections::VecDeque::from([
            PickerOutcome::Navigate(PickerNavigation::Next),
            PickerOutcome::Navigate(PickerNavigation::Previous),
            PickerOutcome::Selected(PickerSelection {
                text: "confirmed".into(),
                is_cached: true,
            }),
        ]);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_by_presenter = seen.clone();
        let mut actions = actions;
        let selection = drive_picker(&config, original.clone(), move |position, availability| {
            seen_by_presenter.lock().expect("seen").push((
                position.audio_path.clone(),
                position.is_original(),
                position.reached_by_navigation,
                availability.previous,
                availability.next,
            ));
            let action = actions.pop_front().expect("one action per record");
            async move { Ok(action) }
        })
        .await
        .expect("loop")
        .expect("confirmed selection");
        assert_eq!(selection.text, "confirmed");
        assert_eq!(
            *seen.lock().expect("seen"),
            vec![
                (original.clone(), true, false, false, true),
                (older, false, true, true, false),
                (original, true, true, false, true),
            ]
        );
    }

    #[tokio::test]
    async fn record_loop_cancel_never_selects_or_pastes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let original = dir.path().join("2026-04-02T10-00-00+0200.ogg");
        std::fs::write(&original, b"audio").expect("original");
        let config = Config {
            output_dir: dir.path().to_path_buf(),
            providers: ProvidersConfig {
                mistral: None,
                openai: None,
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };
        let selection = drive_picker(&config, original, |_position, availability| {
            assert!(!availability.previous && !availability.next);
            async { Ok(PickerOutcome::Cancelled) }
        })
        .await
        .expect("cancel");
        assert!(selection.is_none());
    }

    #[tokio::test]
    async fn record_loop_stops_when_current_recording_disappears_during_navigation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let original = dir.path().join("2026-04-02T10-00-00+0200.ogg");
        let older = dir.path().join("2026-04-01T10-00-00+0200.ogg");
        std::fs::write(&original, b"audio").expect("original");
        std::fs::write(&older, b"audio").expect("older");
        let config = Config {
            output_dir: dir.path().to_path_buf(),
            providers: ProvidersConfig {
                mistral: None,
                openai: None,
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };
        let selected = drive_picker(&config, original.clone(), move |position, availability| {
            assert!(availability.next);
            std::fs::remove_file(position.audio_path).expect("simulate disappearing record");
            async { Ok(PickerOutcome::Navigate(PickerNavigation::Next)) }
        })
        .await
        .expect("navigation should end cleanly");
        assert!(selected.is_none());
        assert!(older.is_file());
    }

    #[tokio::test]
    async fn record_presentation_receives_cached_pick_and_navigated_selection_is_not_cached() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("record.ogg");
        std::fs::write(&audio, b"audio").expect("audio fixture");
        recording_cache::write_pick(&audio, "openai", PICKER_TEST_MODEL, false, "saved words")
            .expect("pick");
        let config = Arc::new(Config {
            output_dir: dir.path().to_path_buf(),
            providers: ProvidersConfig {
                mistral: None,
                openai: Some(OpenAIConfig {
                    api_key: "unused".into(),
                    url: None,
                    model: PICKER_TEST_MODEL.into(),
                    realtime_model: "gpt-live-transcribe".into(),
                    prompt: None,
                    keywords: None,
                    languages: None,
                    realtime_delay: None,
                }),
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        });
        let params = openai_picker_params(audio.clone());
        let result = present_record(
            config,
            &params,
            audio,
            PickRecordContext {
                cached_brief: None,
                is_original: false,
                reached_by_navigation: true,
                navigation: PickerNavigationAvailability {
                    previous: false,
                    next: false,
                },
            },
            |input| async move {
                assert_eq!(input.cached_entries[0].2, "saved words");
                assert!(input.transcribers.is_empty());
                assert!(input
                    .deferred_candidates
                    .iter()
                    .any(|(_, model, _)| model == "whisper-1"));
                Ok(PickerOutcome::Selected(PickerSelection {
                    text: "saved words".into(),
                    is_cached: true,
                }))
            },
        )
        .await
        .expect("presented");
        assert!(
            matches!(result, PickerOutcome::Selected(PickerSelection { text, is_cached: false }) if text == "saved words")
        );
    }

    #[tokio::test]
    async fn picker_confirmation_replaces_previous_paste_only_after_success() {
        let _lock = PICKER_CACHE_ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("tempdir");
        let _cache = EnvGuard::set("XDG_CACHE_HOME", dir.path());
        let mut params = openai_picker_params(dir.path().join("unused.ogg"));
        params.target_window = Some("42".into());
        paste_picker_selection_with(
            &params,
            PickerSelection {
                text: "old".into(),
                is_cached: false,
            },
            |text, delete| async move {
                assert_eq!((text.as_str(), delete), ("old", 0));
                Ok(())
            },
        )
        .await
        .expect("first paste");
        params.replace_last_paste = true;
        let failure = paste_picker_selection_with(
            &params,
            PickerSelection {
                text: "failed".into(),
                is_cached: true,
            },
            |text, delete| async move {
                assert_eq!((text.as_str(), delete), ("failed", 3));
                Err(TalkError::Clipboard("paste rejected".into()))
            },
        )
        .await;
        assert!(matches!(failure, Err(TalkError::Clipboard(_))));
        paste_picker_selection_with(
            &params,
            PickerSelection {
                text: "new".into(),
                is_cached: true,
            },
            |text, delete| async move {
                assert_eq!((text.as_str(), delete), ("new", 3));
                Ok(())
            },
        )
        .await
        .expect("replacement");
        assert_eq!(
            recording_cache::read_last_paste_state()
                .expect("state")
                .expect("paste")
                .text,
            "new"
        );
    }

    #[tokio::test]
    async fn sidecar_matching_metadata_brief_is_primary_without_a_pick() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("record.ogg");
        std::fs::write(&audio, b"audio").expect("audio fixture");
        TranscriptionCache::store(
            &audio,
            Provider::OpenAI,
            "whisper-1",
            false,
            &TranscriptionResult {
                text: "sidecar words".into(),
                ..Default::default()
            },
        )
        .expect("sidecar");
        let config = Config {
            output_dir: dir.path().to_path_buf(),
            providers: ProvidersConfig {
                mistral: None,
                openai: Some(OpenAIConfig {
                    api_key: "unused".into(),
                    url: None,
                    model: PICKER_TEST_MODEL.into(),
                    realtime_model: "gpt-live-transcribe".into(),
                    prompt: None,
                    keywords: None,
                    languages: None,
                    realtime_delay: None,
                }),
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };
        let params = openai_picker_params(audio.clone());
        let brief = recording_cache::RecordingMetadataBrief {
            transcript: "sidecar words".into(),
            provider: Some("openai".into()),
            model: Some("whisper-1".into()),
        };
        let prepared = prepare_record_input(&config, &params, &audio, Some(&brief), false).await;
        assert_eq!(
            prepared.cached_entries[0],
            (
                Provider::OpenAI,
                "whisper-1".into(),
                "sidecar words".into(),
                true,
                false
            )
        );
        assert!(!prepared
            .deferred_candidates
            .iter()
            .any(|(_, model, _)| model == "whisper-1"));
        let invalid_brief = recording_cache::RecordingMetadataBrief {
            provider: Some("unknown".into()),
            ..brief
        };
        let without_primary =
            prepare_record_input(&config, &params, &audio, Some(&invalid_brief), false).await;
        assert!(without_primary.cached_entries.iter().all(|entry| !entry.3));
    }

    #[cfg(feature = "parakeet")]
    #[tokio::test]
    async fn missing_local_default_is_deferred_without_model_download() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("record.ogg");
        std::fs::write(&audio, b"audio").expect("audio fixture");
        let config = Config {
            output_dir: dir.path().to_path_buf(),
            providers: ProvidersConfig {
                mistral: None,
                openai: None,
                parakeet: Some(crate::config::ParakeetConfig {
                    model_dir: Some(dir.path().join("missing-model")),
                    ..Default::default()
                }),
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };
        let params = PickParams {
            provider: Some(Provider::Parakeet),
            model: None,
            ..openai_picker_params(audio.clone())
        };
        let prepared = prepare_record_input(&config, &params, &audio, None, false).await;
        assert!(prepared.oneshot_candidates.is_empty());
        assert_eq!(
            prepared.deferred_candidates,
            vec![(
                Provider::Parakeet,
                "parakeet-tdt-0.6b-v3-int8".into(),
                false
            )]
        );
        assert!(!dir.path().join("missing-model").exists());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires an isolated GTK display"]
    async fn run_pick_loop_reuses_visible_window_and_preserves_record_state() {
        use super::ui::{
            gtk_test_metrics, install_gtk_test_actions, GtkTestAction, GtkTestExit, GtkTestMetrics,
        };

        let temp = tempfile::TempDir::new().expect("tempdir");
        let _cache_home = EnvGuard::set("XDG_CACHE_HOME", &temp.path().join("cache"));
        let output = temp.path().join("output");
        std::fs::create_dir_all(&output).expect("output directory");
        let newest = output.join("2026-04-03T10-00-00+0200.ogg");
        let middle = output.join("2026-04-02T10-00-00+0200.ogg");
        let oldest = output.join("2026-04-01T10-00-00+0200.ogg");
        for path in [&newest, &middle, &oldest] {
            std::fs::write(path, b"").expect("audio fixture");
            TranscriptionCache::store(
                path,
                Provider::Mistral,
                "voxtral-mini-2507",
                false,
                &TranscriptionResult {
                    text: "default row".to_string(),
                    ..TranscriptionResult::default()
                },
            )
            .expect("default sidecar");
        }
        recording_cache::write_pick(
            &newest,
            "openai",
            "gpt-live-transcribe",
            true,
            "newest saved",
        )
        .expect("newest pick");
        recording_cache::write_pick(&middle, "openai", "gpt-transcribe", false, "middle saved")
            .expect("middle pick");
        recording_cache::write_pick(
            &oldest,
            "mistral",
            "voxtral-mini-2602",
            true,
            "oldest edited",
        )
        .expect("oldest pick");

        install_gtk_test_actions(vec![
            GtkTestAction {
                expected_provider: Provider::OpenAI,
                expected_model: "gpt-transcribe".to_string(),
                expected_streaming: false,
                expected_text: "middle saved".to_string(),
                previous_sensitive: true,
                next_sensitive: true,
                edit_text: Some("middle edited".to_string()),
                fail_first_save: true,
                wait: None,
                exit: GtkTestExit::Navigate(PickerNavigation::Next),
            },
            GtkTestAction {
                expected_provider: Provider::Mistral,
                expected_model: "voxtral-mini-2602".to_string(),
                expected_streaming: true,
                expected_text: "oldest edited".to_string(),
                previous_sensitive: true,
                next_sensitive: false,
                edit_text: None,
                fail_first_save: false,
                wait: None,
                exit: GtkTestExit::Navigate(PickerNavigation::Previous),
            },
            GtkTestAction {
                expected_provider: Provider::OpenAI,
                expected_model: "gpt-transcribe".to_string(),
                expected_streaming: false,
                expected_text: "middle edited".to_string(),
                previous_sensitive: true,
                next_sensitive: true,
                edit_text: None,
                fail_first_save: false,
                wait: None,
                exit: GtkTestExit::Navigate(PickerNavigation::Previous),
            },
            GtkTestAction {
                expected_provider: Provider::OpenAI,
                expected_model: "gpt-live-transcribe".to_string(),
                expected_streaming: true,
                expected_text: "newest saved".to_string(),
                previous_sensitive: false,
                next_sensitive: true,
                edit_text: None,
                fail_first_save: false,
                wait: None,
                exit: GtkTestExit::Navigate(PickerNavigation::Next),
            },
            GtkTestAction {
                expected_provider: Provider::OpenAI,
                expected_model: "gpt-transcribe".to_string(),
                expected_streaming: false,
                expected_text: "middle edited".to_string(),
                previous_sensitive: true,
                next_sensitive: true,
                edit_text: None,
                fail_first_save: false,
                wait: None,
                exit: GtkTestExit::Navigate(PickerNavigation::Next),
            },
            GtkTestAction {
                expected_provider: Provider::Mistral,
                expected_model: "voxtral-mini-2602".to_string(),
                expected_streaming: true,
                expected_text: "oldest edited".to_string(),
                previous_sensitive: true,
                next_sensitive: false,
                edit_text: None,
                fail_first_save: false,
                wait: None,
                exit: GtkTestExit::Confirm,
            },
        ]);

        let calls = Arc::new(Mutex::new(Vec::new()));
        let config = Config {
            output_dir: output,
            providers: ProvidersConfig {
                mistral: None,
                openai: None,
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };
        run_pick(
            config,
            PickParams {
                input_audio_file: Some(middle.clone()),
                cached_brief: None,
                replace_char_count: None,
                replace_last_paste: false,
                provider: None,
                model: None,
                target_window: None,
                paste_root: Arc::new(RecordingPaste {
                    calls: Arc::clone(&calls),
                }),
                paste_timing: PasteTiming::default(),
            },
        )
        .await
        .expect("picker loop");

        assert_eq!(
            *calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec!["oldest edited".to_string()]
        );
        assert_eq!(
            recording_cache::read_pick(&middle),
            Some((
                Provider::OpenAI,
                "gpt-transcribe".to_string(),
                false,
                "middle edited".to_string(),
            ))
        );
        assert_eq!(
            gtk_test_metrics(),
            GtkTestMetrics {
                remaining_actions: 0,
                windows_created: 1,
                windows_destroyed: 1,
                records_seen: 6,
                window_id_changes: 0,
                previous_button_id_changes: 0,
                next_button_id_changes: 0,
                unmapped_navigation_buttons: 0,
                invisible_records: 0,
                hidden_transitions: 0,
                stale_source_ticks: 0,
                realtime_decodes: 0,
            }
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires an isolated GTK display and xdotool"]
    async fn rapid_stationary_next_clicks_navigate_once_per_click() {
        use super::ui::{
            finish_gtk_test_input, gtk_test_metrics, install_gtk_test_actions, GtkTestAction,
            GtkTestExit,
        };

        let temp = tempfile::TempDir::new().expect("tempdir");
        let _cache_home = EnvGuard::set("XDG_CACHE_HOME", &temp.path().join("cache"));
        let output = temp.path().join("output");
        std::fs::create_dir_all(&output).expect("output directory");
        let recordings: Vec<PathBuf> = (1..=6)
            .rev()
            .map(|day| output.join(format!("2026-04-{day:02}T10-00-00+0200.ogg")))
            .collect();
        for (index, path) in recordings.iter().enumerate() {
            std::fs::write(path, b"").expect("audio fixture");
            recording_cache::write_pick(
                path,
                "mistral",
                "voxtral-mini-2507",
                false,
                &format!("recording {index}"),
            )
            .expect("recording pick");
        }

        let mut actions = Vec::new();
        for index in 0..6 {
            actions.push(GtkTestAction {
                expected_provider: Provider::Mistral,
                expected_model: "voxtral-mini-2507".to_string(),
                expected_streaming: false,
                expected_text: format!("recording {index}"),
                previous_sensitive: index > 0,
                next_sensitive: index < 5,
                edit_text: None,
                fail_first_save: false,
                wait: None,
                exit: match index {
                    0 => GtkTestExit::RawNextBurst(5),
                    5 => GtkTestExit::Confirm,
                    _ => GtkTestExit::Wait,
                },
            });
        }
        install_gtk_test_actions(actions);

        let calls = Arc::new(Mutex::new(Vec::new()));
        let config = Config {
            output_dir: output,
            providers: ProvidersConfig {
                mistral: None,
                openai: None,
                parakeet: None,
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };
        run_pick(
            config,
            PickParams {
                input_audio_file: Some(recordings[0].clone()),
                cached_brief: None,
                replace_char_count: None,
                replace_last_paste: false,
                provider: None,
                model: None,
                target_window: None,
                paste_root: Arc::new(RecordingPaste {
                    calls: Arc::clone(&calls),
                }),
                paste_timing: PasteTiming::default(),
            },
        )
        .await
        .expect("picker loop");
        finish_gtk_test_input().expect("raw picker input");

        assert_eq!(gtk_test_metrics().records_seen, 6);
        assert_eq!(
            *calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec!["recording 5".to_string()]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires an isolated GTK display"]
    async fn initial_uncached_record_auto_transcribes_default_model_once() {
        use super::ui::{install_gtk_test_actions, GtkTestAction, GtkTestExit, GtkTestWait};

        let temp = tempfile::TempDir::new().expect("tempdir");
        let _cache_home = EnvGuard::set("XDG_CACHE_HOME", &temp.path().join("cache"));
        let _validate_cache = ValidationCacheGuard::new(&temp.path().join("validate-cache.yaml"));
        let output = temp.path().join("output");
        std::fs::create_dir_all(&output).expect("output directory");
        let original = output.join("2026-04-02T10-00-00+0200.ogg");
        std::fs::write(&original, b"fixture audio").expect("audio fixture");
        let server = MockServer::start().await;
        let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        mount_openai_picker_api(&server, Arc::clone(&requests), "initial result", 1).await;
        install_gtk_test_actions(vec![GtkTestAction {
            expected_provider: Provider::OpenAI,
            expected_model: PICKER_TEST_MODEL.to_string(),
            expected_streaming: false,
            expected_text: String::new(),
            previous_sensitive: false,
            next_sensitive: false,
            edit_text: None,
            fail_first_save: false,
            wait: Some(GtkTestWait {
                click_action: false,
                expected_text: "initial result".to_string(),
                expected_action_label_before: None,
                expected_action_label_after: Some("↻".to_string()),
                request_count: Arc::clone(&requests),
                expected_requests_before: None,
                expected_requests_after: 1,
            }),
            exit: GtkTestExit::Confirm,
        }]);

        run_pick(
            openai_picker_config(output, &server),
            openai_picker_params(original),
        )
        .await
        .expect("picker");
        assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires an isolated GTK display"]
    async fn navigated_uncached_record_waits_for_t_before_transcribing() {
        use super::ui::{install_gtk_test_actions, GtkTestAction, GtkTestExit, GtkTestWait};

        let temp = tempfile::TempDir::new().expect("tempdir");
        let _cache_home = EnvGuard::set("XDG_CACHE_HOME", &temp.path().join("cache"));
        let _validate_cache = ValidationCacheGuard::new(&temp.path().join("validate-cache.yaml"));
        let output = temp.path().join("output");
        std::fs::create_dir_all(&output).expect("output directory");
        let original = output.join("2026-04-02T10-00-00+0200.ogg");
        let navigated = output.join("2026-04-01T10-00-00+0200.ogg");
        std::fs::write(&original, b"fixture audio").expect("original fixture");
        std::fs::write(&navigated, b"fixture audio").expect("navigated fixture");
        TranscriptionCache::store(
            &original,
            Provider::OpenAI,
            PICKER_TEST_MODEL,
            false,
            &TranscriptionResult {
                text: "cached original".to_string(),
                ..TranscriptionResult::default()
            },
        )
        .expect("original sidecar");
        let server = MockServer::start().await;
        let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        mount_openai_picker_api(&server, Arc::clone(&requests), "on-demand result", 1).await;
        install_gtk_test_actions(vec![
            GtkTestAction {
                expected_provider: Provider::OpenAI,
                expected_model: PICKER_TEST_MODEL.to_string(),
                expected_streaming: false,
                expected_text: "cached original".to_string(),
                previous_sensitive: false,
                next_sensitive: true,
                edit_text: None,
                fail_first_save: false,
                wait: None,
                exit: GtkTestExit::Navigate(PickerNavigation::Next),
            },
            GtkTestAction {
                expected_provider: Provider::OpenAI,
                expected_model: PICKER_TEST_MODEL.to_string(),
                expected_streaming: false,
                expected_text: String::new(),
                previous_sensitive: true,
                next_sensitive: false,
                edit_text: None,
                fail_first_save: false,
                wait: Some(GtkTestWait {
                    click_action: true,
                    expected_text: "on-demand result".to_string(),
                    expected_action_label_before: Some("T".to_string()),
                    expected_action_label_after: Some("↻".to_string()),
                    request_count: Arc::clone(&requests),
                    expected_requests_before: Some(0),
                    expected_requests_after: 1,
                }),
                exit: GtkTestExit::Confirm,
            },
        ]);

        run_pick(
            openai_picker_config(output, &server),
            openai_picker_params(original),
        )
        .await
        .expect("picker");
        assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires an isolated GTK display"]
    async fn navigating_back_to_original_does_not_auto_transcribe_again() {
        use super::ui::{install_gtk_test_actions, GtkTestAction, GtkTestExit, GtkTestWait};

        let temp = tempfile::TempDir::new().expect("tempdir");
        let _cache_home = EnvGuard::set("XDG_CACHE_HOME", &temp.path().join("cache"));
        let _validate_cache = ValidationCacheGuard::new(&temp.path().join("validate-cache.yaml"));
        let output = temp.path().join("output");
        std::fs::create_dir_all(&output).expect("output directory");
        let original = output.join("2026-04-02T10-00-00+0200.ogg");
        let neighbor = output.join("2026-04-01T10-00-00+0200.ogg");
        std::fs::write(&original, b"fixture audio").expect("original fixture");
        std::fs::write(&neighbor, b"fixture audio").expect("neighbor fixture");
        TranscriptionCache::store(
            &neighbor,
            Provider::OpenAI,
            PICKER_TEST_MODEL,
            false,
            &TranscriptionResult {
                text: "cached neighbor".to_string(),
                ..TranscriptionResult::default()
            },
        )
        .expect("neighbor sidecar");
        let server = MockServer::start().await;
        let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        mount_openai_picker_api(&server, Arc::clone(&requests), "original result", 1).await;
        install_gtk_test_actions(vec![
            GtkTestAction {
                expected_provider: Provider::OpenAI,
                expected_model: PICKER_TEST_MODEL.to_string(),
                expected_streaming: false,
                expected_text: String::new(),
                previous_sensitive: false,
                next_sensitive: true,
                edit_text: None,
                fail_first_save: false,
                wait: Some(GtkTestWait {
                    click_action: false,
                    expected_text: "original result".to_string(),
                    expected_action_label_before: None,
                    expected_action_label_after: Some("↻".to_string()),
                    request_count: Arc::clone(&requests),
                    expected_requests_before: None,
                    expected_requests_after: 1,
                }),
                exit: GtkTestExit::Navigate(PickerNavigation::Next),
            },
            GtkTestAction {
                expected_provider: Provider::OpenAI,
                expected_model: PICKER_TEST_MODEL.to_string(),
                expected_streaming: false,
                expected_text: "cached neighbor".to_string(),
                previous_sensitive: true,
                next_sensitive: false,
                edit_text: None,
                fail_first_save: false,
                wait: None,
                exit: GtkTestExit::Navigate(PickerNavigation::Previous),
            },
            GtkTestAction {
                expected_provider: Provider::OpenAI,
                expected_model: PICKER_TEST_MODEL.to_string(),
                expected_streaming: false,
                expected_text: "original result".to_string(),
                previous_sensitive: false,
                next_sensitive: true,
                edit_text: None,
                fail_first_save: false,
                wait: None,
                exit: GtkTestExit::Confirm,
            },
        ]);

        run_pick(
            openai_picker_config(output, &server),
            openai_picker_params(original),
        )
        .await
        .expect("picker");
        assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
