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
    let mut audio_path = original_audio.clone();
    let mut reached_by_navigation = false;
    loop {
        let navigation = crate::record::recording_navigation(&audio_path, &config.output_dir)?;
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
        let is_original = audio_path == original_audio;
        let cached_brief = if is_original {
            params.cached_brief.as_ref()
        } else {
            None
        };
        let outcome = run_pick_record(
            config.clone(),
            &params,
            audio_path.clone(),
            PickRecordContext {
                cached_brief,
                is_original,
                reached_by_navigation,
                navigation: navigation_availability,
            },
            session,
        )
        .await?;

        match outcome {
            PickerOutcome::Cancelled => {
                session.destroy();
                return Ok(());
            }
            PickerOutcome::Navigate(direction) => {
                let refreshed =
                    crate::record::recording_navigation(&audio_path, &config.output_dir)?;
                let target = refreshed.and_then(|value| match direction {
                    PickerNavigation::Previous => value.previous,
                    PickerNavigation::Next => value.next,
                });
                match target {
                    Some(path) if path.is_file() => {
                        audio_path = path;
                        // Every arrow-opened record is opt-in for paid
                        // transcription, including the original on return.
                        reached_by_navigation = true;
                        continue;
                    }
                    Some(path) => {
                        log::warn!(
                            "picker navigation target became unavailable: {}",
                            path.display()
                        );
                    }
                    None => {
                        log::debug!("picker navigation reached a collection boundary");
                    }
                }
                if !audio_path.is_file() {
                    return Ok(());
                }
            }
            PickerOutcome::Selected(selection) => {
                session.destroy();
                return paste_picker_selection(&params, selection).await;
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
    let PickRecordContext {
        cached_brief,
        is_original,
        reached_by_navigation,
        navigation,
    } = context;
    // Read the authoritative pick (user-confirmed selection + edited text).
    // This is the ONLY cross-provider source of truth for the picker's
    // selection state.  Sidecars are per-model internals probed below
    // via `transcribe_audio(allow_api=false)`.
    let pick = recording_cache::read_pick(&audio_path);
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

    let candidates =
        build_retry_candidates(config.as_ref(), params.provider, params.model.as_deref());
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
            &audio_path,
            config.as_ref(),
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
    // pick goes first so it is pre-selected on every record. Whether that
    // selection should skip paste is decided separately at outcome handoff:
    // only the original record can represent text already in the target.
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

    // Resolve the default model so only it is transcribed immediately.
    // All other candidates are deferred — shown in the UI with a
    // "transcribe" button that the user can click on demand.
    let default_provider = resolve_provider(params.provider, config.as_ref());
    let default_model = resolve_model(
        params.model.as_deref(),
        config.as_ref(),
        default_provider,
        false,
    );
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
                let present = crate::transcription::parakeet::consent::resolve(config.as_ref())
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

    // Create realtime transcribers for the default model only.
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

    let mut outcome = pick_with_streaming_gtk(
        session,
        PickerUiInput {
            transcribers: oneshot_filtered,
            audio_path,
            cached_entries,
            config: config.clone(),
            realtime_transcribers: rt_transcribers,
            deferred_candidates: deferred,
            navigation,
        },
    )
    .await?;
    if !is_original {
        if let PickerOutcome::Selected(selection) = &mut outcome {
            selection.is_cached = false;
        }
    }
    Ok(outcome)
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
    // Selection is auto-saved by the picker UI (on first result,
    // debounced on row change, and on close).

    // If the user selected the cached entry, nothing to do — the
    // text is already in the target window.
    if selection.is_cached {
        log::info!("cached entry selected — no paste needed");
        return Ok(());
    }

    let delete_chars = if params.replace_last_paste {
        // Prefer the paste-state file (written after every paste,
        // including picker selections) over recording metadata so
        // that successive picker replacements delete the correct
        // number of characters.
        recording_cache::read_last_paste_state()?
            .map(|s| s.char_count)
            .or(params.replace_char_count)
            .unwrap_or(0)
    } else {
        0
    };

    paste_with_root(
        params.paste_root.as_ref(),
        params.target_window.as_ref(),
        &selection.text,
        delete_chars,
        None,
        &crate::telemetry::NoOpSink,
        params.paste_timing,
        // Picker paste path has no sound player wired.
        None,
    )
    .await?;
    let _ =
        recording_cache::write_last_paste_state(params.target_window.as_deref(), &selection.text);
    println!("{}", selection.text);
    Ok(())
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
