//! Pick-mode logic for the dictate command.
//!
//! When `--pick` is passed, the user selects among multiple transcription
//! providers/models via a GTK picker window.  Cached results are reused
//! when available.

use crate::core::config::{Config, Provider};
use crate::core::error::TalkError;
use crate::core::picker_cache;
use crate::core::recording_cache;
use crate::core::transcription::{self, BatchTranscriber, RealtimeTranscriber};
use crate::paste::paste_text_to_target;
use crate::x11::x11_centre_and_raise;
use std::path::PathBuf;

use super::models::build_retry_candidates;
use super::picker::{pick_with_streaming_gtk, PICKER_TITLE};

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

    // Load previously cached picker results for this audio file
    // so that reopening the picker skips API calls entirely.
    let picker_cache_data = picker_cache::read(&audio_path);

    // Determine the "already pasted" (provider, model, streaming) triple.
    // The picker cache's `selected` field takes precedence
    // (the user may have picked a different model last time).
    // Fall back to the recording metadata otherwise.
    let selected_key: Option<(Provider, String, bool)> = picker_cache_data
        .selected
        .as_ref()
        .and_then(|s| {
            Some((
                s.provider.parse::<Provider>().ok()?,
                s.model.clone(),
                s.streaming,
            ))
        })
        .or_else(|| {
            params.cached_brief.as_ref().and_then(|b| {
                Some((
                    b.provider.as_deref()?.parse().ok()?,
                    b.model.as_deref()?.to_string(),
                    false,
                ))
            })
        });

    // Collect all available cached transcriptions.
    // Tuple: (provider, model, text, streaming)
    let mut all_entries: Vec<(Provider, String, String, bool)> = Vec::new();

    // From recording metadata (always batch / streaming=false).
    if let Some(ref brief) = params.cached_brief {
        if let (Some(ps), Some(m)) = (brief.provider.as_deref(), brief.model.as_deref()) {
            if let Ok(p) = ps.parse::<Provider>() {
                all_entries.push((p, m.to_string(), brief.transcript.clone(), false));
            }
        }
    }

    // From picker cache results (skip duplicates).
    for cr in &picker_cache_data.results {
        if let Ok(p) = cr.provider.parse::<Provider>() {
            if !all_entries
                .iter()
                .any(|(ep, em, _, es)| *ep == p && *em == cr.model && *es == cr.streaming)
            {
                all_entries.push((p, cr.model.clone(), cr.text.clone(), cr.streaming));
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
        "picker cache: {} cached entries (primary={}, from_cache={})",
        cached_entries.len(),
        cached_entries.iter().filter(|(_, _, _, p, _)| *p).count(),
        picker_cache_data.results.len(),
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

    let candidates = build_retry_candidates(&config, params.provider, params.model.as_deref());
    log::debug!("picker candidates: {} total", candidates.len());
    for (p, m, s) in &candidates {
        log::debug!("  candidate: {}:{} (streaming={})", p, m, s);
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

    // Split filtered candidates into batch and realtime groups.
    let batch_filtered: Vec<(Provider, String)> = filtered
        .iter()
        .filter(|(_, _, s)| !s)
        .map(|(p, m, _)| (*p, m.clone()))
        .collect();
    let realtime_filtered: Vec<(Provider, String)> = filtered
        .iter()
        .filter(|(_, _, s)| *s)
        .map(|(p, m, _)| (*p, m.clone()))
        .collect();

    // Create batch transcribers before entering GTK (needs &Config).
    let mut transcribers: Vec<(Provider, String, Box<dyn BatchTranscriber>)> = Vec::new();
    for (provider, model) in batch_filtered {
        match transcription::create_batch_transcriber(&config, provider, Some(&model), false) {
            Ok(t) => transcribers.push((provider, model, t)),
            Err(e) => log::warn!("skipping batch {}:{}: {}", provider, model, e),
        }
    }

    // Create realtime transcribers.
    let mut rt_transcribers: Vec<(Provider, String, Box<dyn RealtimeTranscriber>)> = Vec::new();
    for (provider, model) in realtime_filtered {
        match transcription::create_realtime_transcriber(&config, provider, Some(&model)) {
            Ok(t) => rt_transcribers.push((provider, model, t)),
            Err(e) => log::warn!("skipping realtime {}:{}: {}", provider, model, e),
        }
    }

    if transcribers.is_empty() && cached_entries.is_empty() && rt_transcribers.is_empty() {
        return Err(TalkError::Transcription(
            "no transcription providers available".to_string(),
        ));
    }

    let audio_path_for_selection = audio_path.clone();
    let selected = pick_with_streaming_gtk(
        transcribers,
        audio_path,
        cached_entries,
        config,
        rt_transcribers,
    )
    .await?;
    let selection = match selected {
        Some(s) => s,
        None => return Ok(()),
    };

    // Record which entry the user selected so it appears
    // pre-selected the next time the picker opens.
    if let Err(e) = picker_cache::write_selected(
        &audio_path_for_selection,
        &selection.provider.to_string(),
        &selection.model,
        selection.streaming,
    ) {
        log::warn!("failed to update picker selection: {}", e);
    }

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
    )
    .await?;
    let _ =
        recording_cache::write_last_paste_state(params.target_window.as_deref(), &selection.text);
    println!("{}", selection.text);
    Ok(())
}
