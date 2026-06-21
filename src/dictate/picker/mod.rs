//! Pick-mode logic for the dictate command.
//!
//! When `--pick` is passed, the user selects among multiple transcription
//! providers/models via a GTK picker window.  Cached results are reused
//! when available.

mod backend;
mod ui;

use crate::config::{Config, PasteShortcut, Provider};
use crate::error::TalkError;
use crate::paste::{paste_text_to_target, PasteTiming};
use crate::recording_cache;
use crate::transcription::{self, RealtimeTranscriber};
use crate::x11::x11_centre_and_raise;
use std::path::PathBuf;

use super::models::{build_retry_candidates, resolve_model, resolve_provider};
use ui::{pick_with_streaming_gtk, PICKER_TITLE};

/// Parameters for the pick-mode path.
pub(crate) struct PickParams {
    pub input_audio_file: Option<PathBuf>,
    pub cached_brief: Option<recording_cache::RecordingMetadataBrief>,
    pub replace_char_count: Option<usize>,
    pub replace_last_paste: bool,
    pub provider: Option<Provider>,
    pub model: Option<String>,
    pub target_window: Option<String>,
    pub paste_chunk_chars: usize,
    pub paste_shortcut: PasteShortcut,
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

    // Read the authoritative pick (user-confirmed selection + edited text).
    // This is the ONLY cross-provider source of truth for the picker's
    // selection state.  Sidecars are per-model internals probed below
    // via `transcribe_audio(allow_api=false)`.
    let pick = recording_cache::read_pick(&audio_path);
    let selected_key: Option<(Provider, String, bool)> = pick
        .as_ref()
        .map(|(p, m, s, _)| (*p, m.clone(), *s))
        .or_else(|| {
            params.cached_brief.as_ref().and_then(|b| {
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

    let config = std::sync::Arc::new(config);

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

    // Build cached_entries with is_primary flag.  The selected
    // (already-pasted) entry goes first so it is pre-selected
    // in the GTK list.
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

    // Split filtered candidates into default (immediate) and deferred.
    let mut default_filtered: Vec<(Provider, String, bool)> = Vec::new();
    let mut deferred: Vec<(Provider, String, bool)> = Vec::new();
    for (p, m, s) in filtered {
        if is_default(&p, &m) {
            default_filtered.push((p, m, s));
        } else {
            deferred.push((p, m, s));
        }
    }

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
    let batch_filtered: Vec<(Provider, String)> = default_filtered
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

    if batch_filtered.is_empty()
        && cached_entries.is_empty()
        && rt_transcribers.is_empty()
        && deferred.is_empty()
    {
        return Err(TalkError::Transcription(
            "no transcription providers available".to_string(),
        ));
    }

    let selected = pick_with_streaming_gtk(
        batch_filtered,
        audio_path,
        cached_entries,
        config.clone(),
        rt_transcribers,
        deferred,
    )
    .await?;
    let selection = match selected {
        Some(s) => s,
        None => return Ok(()),
    };

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

    paste_text_to_target(
        params.target_window.as_ref(),
        &selection.text,
        delete_chars,
        params.paste_chunk_chars,
        None,
        &crate::telemetry::NoOpSink,
        params.paste_shortcut,
        params.paste_timing,
    )
    .await?;
    let _ =
        recording_cache::write_last_paste_state(params.target_window.as_deref(), &selection.text);
    println!("{}", selection.text);
    Ok(())
}
