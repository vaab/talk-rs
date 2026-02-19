//! Dictate command implementation.
//!
//! Records audio, streams it to the transcription API, and pastes the result
//! into the focused application via clipboard.

use crate::core::audio::file_source::WavFileSource;
use crate::core::audio::indicator::SoundPlayer;
use crate::core::audio::pipewire_capture::PipeWireCapture;
use crate::core::audio::resample;
use crate::core::audio::{AudioCapture, AudioWriter, OggOpusWriter, WavWriter, CHUNK_DURATION_MS};
use crate::core::clipboard::{Clipboard, X11Clipboard};
use crate::core::config::{AudioConfig, Config, Provider};
use crate::core::daemon::{self, DaemonStatus};
use crate::core::error::TalkError;
use crate::core::overlay::{IndicatorKind, OverlayHandle};
use crate::core::picker_cache;
use crate::core::recording_cache;
use crate::core::transcription::{
    self, BatchTranscriber, MistralProviderMetadata, OpenAIProviderMetadata,
    OpenAIRealtimeMetadata, ProviderSpecificMetadata, TranscriptionEvent, TranscriptionMetadata,
    TranscriptionResult,
};
use crate::core::visualizer::VisualizerHandle;
use std::os::unix::process::CommandExt as _;
use std::path::PathBuf;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

/// Options for the dictate command.
pub struct DictateOpts {
    pub save: Option<PathBuf>,
    pub output_yaml: Option<PathBuf>,
    pub input_audio_file: Option<PathBuf>,
    pub retry_last: bool,
    pub pick: bool,
    pub replace_last_paste: bool,
    pub provider: Option<Provider>,
    pub model: Option<String>,
    pub realtime: bool,
    pub toggle: bool,
    pub no_sounds: bool,
    pub no_overlay: bool,
    pub amplitude: bool,
    pub spectrum: bool,
    pub daemon: bool,
    pub target_window: Option<String>,
    pub verbose: u8,
}

/// Resolve the effective provider from CLI override or config default.
fn resolve_provider(cli_provider: Option<Provider>, config: &Config) -> Provider {
    if let Some(p) = cli_provider {
        return p;
    }
    config
        .transcription
        .as_ref()
        .map(|t| t.default_provider)
        .unwrap_or(Provider::Mistral)
}

/// Resolve the effective model name from CLI override or config default.
fn resolve_model(
    cli_model: Option<&str>,
    config: &Config,
    provider: Provider,
    realtime: bool,
) -> String {
    if let Some(m) = cli_model {
        return m.to_string();
    }
    match provider {
        Provider::Mistral => config
            .providers
            .mistral
            .as_ref()
            .map(|c| c.model.clone())
            .unwrap_or_else(|| "voxtral-mini-2507".to_string()),
        Provider::OpenAI => {
            if realtime {
                config
                    .providers
                    .openai
                    .as_ref()
                    .map(|c| c.realtime_model.clone())
                    .unwrap_or_else(|| "gpt-4o-mini-transcribe".to_string())
            } else {
                config
                    .providers
                    .openai
                    .as_ref()
                    .map(|c| c.model.clone())
                    .unwrap_or_else(|| "whisper-1".to_string())
            }
        }
    }
}

/// Known OpenAI models for the `/v1/audio/transcriptions` endpoint.
const OPENAI_TRANSCRIPTION_MODELS: &[&str] =
    &["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"];

/// Known Mistral models for the `/v1/audio/transcriptions` endpoint.
///
/// `voxtral-mini-latest` aliases to `voxtral-mini-2602` for
/// transcription; we use explicit version names so the user can
/// compare results between generations.
const MISTRAL_TRANSCRIPTION_MODELS: &[&str] = &["voxtral-mini-2507", "voxtral-mini-2602"];

/// Push all known transcription models for `provider` into `out`.
fn add_known_models(out: &mut Vec<(Provider, String)>, provider: Provider) {
    let models = match provider {
        Provider::OpenAI => OPENAI_TRANSCRIPTION_MODELS,
        Provider::Mistral => MISTRAL_TRANSCRIPTION_MODELS,
    };
    for m in models {
        out.push((provider, (*m).to_string()));
    }
}

fn build_retry_candidates(
    config: &Config,
    cli_provider: Option<Provider>,
    cli_model: Option<&str>,
) -> Vec<(Provider, String)> {
    let mut out: Vec<(Provider, String)> = Vec::new();

    // If the user explicitly specified a model, include it even if it
    // is not in the known list (e.g. a dated snapshot or new model).
    if let (Some(provider), Some(model)) = (cli_provider, cli_model) {
        out.push((provider, model.to_string()));
    }

    match cli_provider {
        Some(provider) => {
            // Specific provider requested: add all known models for it.
            add_known_models(&mut out, provider);
        }
        None => {
            // No provider filter: add all known models for every
            // provider, plus the config defaults (in case the user
            // configured a model we do not list).
            add_known_models(&mut out, Provider::OpenAI);
            add_known_models(&mut out, Provider::Mistral);
            out.push((
                Provider::OpenAI,
                resolve_model(None, config, Provider::OpenAI, false),
            ));
            out.push((
                Provider::Mistral,
                resolve_model(None, config, Provider::Mistral, false),
            ));
        }
    }

    out.sort_by(|a, b| {
        a.0.to_string()
            .cmp(&b.0.to_string())
            .then_with(|| a.1.cmp(&b.1))
    });
    out.dedup();
    out
}

/// Window title used for the picker — also used for single-instance
/// detection via `x11_centre_and_raise`.
const PICKER_TITLE: &str = "talk-rs: select transcription";

/// Candidate transcription result sent from async tasks to the GTK
/// thread via a `std::sync::mpsc` channel.
struct PickerCandidate {
    provider: Provider,
    model: String,
    text: String,
    /// When set, the candidate failed and this is the error message.
    error: Option<String>,
}

/// Messages sent from async tasks to the GTK poll loop.
enum PickerMessage {
    /// A transcription result (success or error).
    Candidate(PickerCandidate),
    /// All initial transcription tasks have completed.
    /// The poll loop should mark any remaining spinners as failed
    /// but keep running to receive retry results.
    InitialBatchDone,
}

/// Centre a known X11 window on the monitor containing the mouse
/// pointer, set it always-on-top, and activate it.
///
/// This is the core positioning helper — callers supply the XID
/// directly (from GDK or from a title search).
fn x11_centre_and_raise_xid(wid: u32) -> bool {
    use x11rb::connection::Connection;
    use x11rb::protocol::randr::ConnectionExt as _;
    use x11rb::protocol::xproto::*;

    let (conn, screen_num) = match x11rb::connect(None) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;

    // ── Intern atoms ────────────────────────────────────────────
    let atom_names: &[&[u8]] = &[
        b"_NET_WM_STATE",
        b"_NET_WM_STATE_ABOVE",
        b"_NET_ACTIVE_WINDOW",
    ];
    let cookies: Vec<_> = atom_names
        .iter()
        .map(|n| conn.intern_atom(false, n))
        .collect::<Vec<_>>();
    let mut atoms = Vec::new();
    for cookie in cookies {
        let cookie = match cookie {
            Ok(c) => c,
            Err(_) => return false,
        };
        let atom = match cookie.reply() {
            Ok(r) => r.atom,
            Err(_) => return false,
        };
        atoms.push(atom);
    }
    let a_wm_state = atoms[0];
    let a_above = atoms[1];
    let a_active = atoms[2];

    // ── Query pointer position ──────────────────────────────────
    let pointer = match conn.query_pointer(root) {
        Ok(cookie) => match cookie.reply() {
            Ok(p) => p,
            Err(_) => return false,
        },
        Err(_) => return false,
    };
    let px = pointer.root_x as i32;
    let py = pointer.root_y as i32;

    // ── Find monitor at pointer via RandR ───────────────────────
    let (mon_x, mon_y, mon_w, mon_h) = {
        let default = (0i32, 0i32, 1920i32, 1080i32);
        let resources = match conn.randr_get_screen_resources(root) {
            Ok(cookie) => match cookie.reply() {
                Ok(r) => r,
                Err(_) => return false,
            },
            Err(_) => return false,
        };

        let mut found = default;
        let mut any_monitor = false;
        for &crtc in &resources.crtcs {
            let info = match conn.randr_get_crtc_info(crtc, 0) {
                Ok(cookie) => match cookie.reply() {
                    Ok(i) => i,
                    Err(_) => continue,
                },
                Err(_) => continue,
            };
            if info.width == 0 || info.height == 0 {
                continue;
            }
            let cx = info.x as i32;
            let cy = info.y as i32;
            let cw = info.width as i32;
            let ch = info.height as i32;

            if !any_monitor {
                found = (cx, cy, cw, ch);
                any_monitor = true;
            }

            if px >= cx && px < cx + cw && py >= cy && py < cy + ch {
                found = (cx, cy, cw, ch);
                break;
            }
        }
        found
    };

    // ── Get physical window geometry ────────────────────────────
    let geom = match conn.get_geometry(wid) {
        Ok(cookie) => match cookie.reply() {
            Ok(g) => g,
            Err(_) => return false,
        },
        Err(_) => return false,
    };

    let win_w = geom.width as i32;
    let win_h = geom.height as i32;
    if win_w == 0 || win_h == 0 {
        return false;
    }

    // ── Centre window on monitor ────────────────────────────────
    let x = mon_x + (mon_w - win_w) / 2;
    let y = mon_y + (mon_h - win_h) / 2;

    let _ = conn.configure_window(wid, &ConfigureWindowAux::new().x(x).y(y));

    // ── Set always-on-top (_NET_WM_STATE_ADD ABOVE) ─────────────
    let above_event = ClientMessageEvent::new(32, wid, a_wm_state, [1u32, a_above, 0, 0, 0]);
    let _ = conn.send_event(
        false,
        root,
        EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
        above_event,
    );

    // ── Activate window (_NET_ACTIVE_WINDOW) ────────────────────
    let activate_event = ClientMessageEvent::new(32, wid, a_active, [1u32, 0, 0, 0, 0]);
    let _ = conn.send_event(
        false,
        root,
        EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
        activate_event,
    );

    let _ = conn.flush();
    true
}

/// Find the X11 window whose `_NET_WM_NAME` matches `title`, centre
/// it on the monitor containing the mouse pointer, set it
/// always-on-top, and activate it.
///
/// Used for single-instance detection (raising an already-open
/// picker).  For newly created windows, prefer
/// [`x11_centre_and_raise_xid`] with the XID obtained from
/// [`gdk4_x11`] — it avoids the `_NET_CLIENT_LIST` race entirely.
fn x11_centre_and_raise(title: &str) -> bool {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;

    let (conn, screen_num) = match x11rb::connect(None) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;

    // ── Intern atoms for title search ───────────────────────────
    let atom_names: &[&[u8]] = &[b"_NET_CLIENT_LIST", b"_NET_WM_NAME", b"UTF8_STRING"];
    let cookies: Vec<_> = atom_names
        .iter()
        .map(|n| conn.intern_atom(false, n))
        .collect::<Vec<_>>();
    let mut atoms = Vec::new();
    for cookie in cookies {
        let cookie = match cookie {
            Ok(c) => c,
            Err(_) => return false,
        };
        let atom = match cookie.reply() {
            Ok(r) => r.atom,
            Err(_) => return false,
        };
        atoms.push(atom);
    }
    let a_client_list = atoms[0];
    let a_wm_name = atoms[1];
    let a_utf8 = atoms[2];

    // ── Find window by _NET_WM_NAME ─────────────────────────────
    let client_list = match conn.get_property(false, root, a_client_list, AtomEnum::WINDOW, 0, 1024)
    {
        Ok(cookie) => match cookie.reply() {
            Ok(prop) => prop,
            Err(_) => return false,
        },
        Err(_) => return false,
    };

    let wid_vec: Vec<u32> = match client_list.value32() {
        Some(iter) => iter.collect(),
        None => return false,
    };

    for &wid in &wid_vec {
        // Try _NET_WM_NAME (UTF-8) first.
        if let Ok(cookie) = conn.get_property(false, wid, a_wm_name, a_utf8, 0, 256) {
            if let Ok(prop) = cookie.reply() {
                if String::from_utf8_lossy(&prop.value) == title {
                    return x11_centre_and_raise_xid(wid);
                }
            }
        }
        // Fallback: WM_NAME (Latin-1).
        if let Ok(cookie) =
            conn.get_property(false, wid, AtomEnum::WM_NAME, AtomEnum::STRING, 0, 256)
        {
            if let Ok(prop) = cookie.reply() {
                if String::from_utf8_lossy(&prop.value) == title {
                    return x11_centre_and_raise_xid(wid);
                }
            }
        }
    }

    false
}

/// Result of the picker: selected provider/model/text and whether it
/// was the pre-populated cached entry (in which case the caller should
/// skip pasting since the text is already in the target window).
struct PickerSelection {
    provider: Provider,
    model: String,
    text: String,
    is_cached: bool,
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
/// `(provider, model, text, is_primary)` where `is_primary` marks the
/// entry that was already pasted in a previous run (selecting it again
/// skips re-pasting).
///
/// Returns `Some(PickerSelection)` for the selected candidate, or
/// `None` if the user cancelled.
async fn pick_with_streaming_gtk(
    mut transcribers: Vec<(Provider, String, Box<dyn BatchTranscriber>)>,
    audio_path: PathBuf,
    mut cached_entries: Vec<(Provider, String, String, bool)>,
    config: Config,
) -> Result<Option<PickerSelection>, TalkError> {
    if transcribers.is_empty() && cached_entries.is_empty() {
        return Ok(None);
    }

    // Stable display order: sort by (provider, model) so the list
    // looks identical every time the picker opens.
    cached_entries.sort_by(|(pa, ma, _, _), (pb, mb, _, _)| {
        pa.to_string().cmp(&pb.to_string()).then(ma.cmp(mb))
    });
    transcribers
        .sort_by(|(pa, ma, _), (pb, mb, _)| pa.to_string().cmp(&pb.to_string()).then(ma.cmp(mb)));

    // Extract (provider, model) labels before transcribers are consumed
    // so the GTK thread can pre-create rows with spinners.
    let pending_info: Vec<(Provider, String)> = transcribers
        .iter()
        .map(|(p, m, _)| (*p, m.clone()))
        .collect();

    // Channels: transcription tasks → GTK, GTK → caller.
    let (msg_tx, msg_rx) = std::sync::mpsc::channel::<PickerMessage>();
    let (sel_tx, sel_rx) = tokio::sync::oneshot::channel::<Option<usize>>();

    // Retry channel: GTK thread → async retry listener.
    let (retry_tx, retry_rx) = std::sync::mpsc::channel::<(Provider, String)>();

    // Shared storage written by the GTK thread after selection,
    // read by the caller after the GTK thread finishes.
    let results = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(
        Provider,
        String,
        String,
        bool,
        bool,
    )>::new()));
    let results_for_gtk = results.clone();

    // ── GTK window ──────────────────────────────────────────────
    let gtk_handle = tokio::task::spawn_blocking(move || -> Result<(), TalkError> {
        use gtk4::glib;
        use gtk4::prelude::*;
        use std::cell::RefCell;
        use std::rc::Rc;

        gtk4::init().map_err(|e| TalkError::Config(format!("failed to initialize GTK: {}", e)))?;

        // Resolve theme colours to concrete hex values so that the
        // entire CSS is state-independent (no :backdrop surprises).
        //
        // GTK4 themes expose @theme_base_color but NOT libadwaita's
        // @view_bg_color; we approximate the latter by darkening
        // (dark themes, ×0.77) or barely tinting (light themes,
        // ×0.98) the base colour.  The selection highlight reuses
        // @theme_selected_bg_color at 32 % opacity — the same
        // treatment libadwaita applies in Nautilus/Files.
        #[allow(deprecated)]
        let probe = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
        #[allow(deprecated)]
        let ctx = probe.style_context();

        let rgba_to_hex = |c: &gtk4::gdk::RGBA| -> String {
            format!(
                "#{:02x}{:02x}{:02x}",
                (c.red() * 255.0).round() as u8,
                (c.green() * 255.0).round() as u8,
                (c.blue() * 255.0).round() as u8,
            )
        };

        #[allow(deprecated)]
        let view_bg_hex = if let Some(base) = ctx.lookup_color("theme_base_color") {
            let luma = 0.299 * base.red() + 0.587 * base.green() + 0.114 * base.blue();
            let factor: f32 = if luma < 0.5 { 0.77 } else { 0.98 };
            let to_u8 =
                |v: f32| -> u8 { ((v * factor * 255.0).round() as i32).clamp(0, 255) as u8 };
            format!(
                "#{:02x}{:02x}{:02x}",
                to_u8(base.red()),
                to_u8(base.green()),
                to_u8(base.blue()),
            )
        } else {
            "@theme_base_color".to_string()
        };

        #[allow(deprecated)]
        let error_hex = ctx
            .lookup_color("error_color")
            .map(|c| rgba_to_hex(&c))
            .unwrap_or_else(|| "@error_color".to_string());

        // Selection: accent at 32 % opacity (matches libadwaita).
        #[allow(deprecated)]
        let sel_hex = ctx
            .lookup_color("theme_selected_bg_color")
            .map(|c| {
                format!(
                    "rgba({},{},{},0.32)",
                    (c.red() * 255.0).round() as u8,
                    (c.green() * 255.0).round() as u8,
                    (c.blue() * 255.0).round() as u8,
                )
            })
            .unwrap_or_else(|| "alpha(@theme_selected_bg_color, 0.32)".to_string());

        let window = gtk4::Window::builder()
            .title(PICKER_TITLE)
            .default_width(900)
            .default_height(500)
            .decorated(false)
            .build();

        // CSS notes:
        //   - GtkListBox CSS node is "list", not "listbox"
        //   - Every stateful rule is duplicated with :backdrop
        //     to prevent the theme from lightening on focus loss
        //   - row:selected uses a translucent accent instead of
        //     the theme's solid #E95420
        let css = gtk4::CssProvider::new();
        css.load_from_data(&format!(
            "* {{ font-size: 13px; }} \
             window, window:backdrop {{ border: 1px solid alpha(white, 0.15); border-radius: 12px; background-color: {bg}; }} \
             scrolledwindow, viewport, list, \
             scrolledwindow:backdrop, viewport:backdrop, list:backdrop {{ background-color: transparent; background-image: none; border-radius: 10px; }} \
             row:not(:selected), row:not(:selected):backdrop {{ background-color: transparent; background-image: none; }} \
             row:selected, row:selected:backdrop {{ background-color: {sel}; }} \
             .transcript {{ font-family: monospace; }} \
             .dim {{ opacity: 0.55; }} \
             .error {{ font-style: italic; color: alpha({err}, 0.8); }} \
             .retry-btn {{ min-width: 0; min-height: 0; padding: 2px 6px; font-size: 14px; }} \
             row {{ border-radius: 8px; }}",
            bg = view_bg_hex,
            err = error_hex,
            sel = sel_hex,
        ));
        if let Some(display) = gtk4::gdk::Display::default() {
            gtk4::style_context_add_provider_for_display(
                &display,
                &css,
                gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
            );
        }

        let root = gtk4::Box::new(gtk4::Orientation::Vertical, 4);
        root.set_margin_top(6);
        root.set_margin_bottom(6);
        root.set_margin_start(6);
        root.set_margin_end(6);

        let scrolled = gtk4::ScrolledWindow::builder()
            .vexpand(true)
            .hexpand(true)
            .build();
        let list = gtk4::ListBox::new();
        list.set_selection_mode(gtk4::SelectionMode::Single);
        list.set_activate_on_single_click(false);
        scrolled.set_child(Some(&list));
        root.append(&scrolled);

        // WindowHandle enables dragging the window from any non-
        // interactive area (help label, table background, margins).
        let handle = gtk4::WindowHandle::new();
        handle.set_child(Some(&root));
        window.set_child(Some(&handle));

        let main_loop = glib::MainLoop::new(None, false);
        let sel_sender: Rc<RefCell<Option<tokio::sync::oneshot::Sender<Option<usize>>>>> =
            Rc::new(RefCell::new(Some(sel_tx)));
        type CandidateList = Vec<(Provider, String, String, bool, bool)>;
        let local_candidates: Rc<RefCell<CandidateList>> = Rc::new(RefCell::new(Vec::new()));
        // Transcript cells — used to swap spinner → label when results arrive.
        let transcript_cells: Rc<RefCell<Vec<gtk4::Box>>> = Rc::new(RefCell::new(Vec::new()));

        // Helper: build a row skeleton with provider + model labels.
        // Returns (hbox, transcript_cell) — caller fills the cell
        // with either a spinner or a transcript label.
        fn make_row_skeleton(provider: &str, model: &str) -> (gtk4::Box, gtk4::Box) {
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

        // Cached entries: rows with transcript already filled.
        // The `is_primary` flag marks the entry that was already
        // pasted in a previous run — it gets pre-selected.
        let mut selected_row = false;
        for (i, (provider, model, text, is_primary)) in cached_entries.iter().enumerate() {
            let (hbox, cell) = make_row_skeleton(&provider.to_string(), model);
            cell.append(&make_transcript_label(text));

            let row = gtk4::ListBoxRow::new();
            row.set_child(Some(&hbox));
            list.append(&row);

            // Select the primary entry, or the very first one.
            if *is_primary || (i == 0 && !selected_row) {
                list.select_row(Some(&row));
                selected_row = true;
            }

            local_candidates.borrow_mut().push((
                *provider,
                model.clone(),
                text.clone(),
                *is_primary,
                false,
            ));
            transcript_cells.borrow_mut().push(cell);
        }

        // Pending entries: row with spinner in transcript cell.
        for (provider, model) in &pending_info {
            let (hbox, cell) = make_row_skeleton(&provider.to_string(), model);
            let spinner = gtk4::Spinner::new();
            spinner.start();
            cell.append(&spinner);

            let row = gtk4::ListBoxRow::new();
            row.set_child(Some(&hbox));
            list.append(&row);

            local_candidates.borrow_mut().push((
                *provider,
                model.clone(),
                String::new(),
                false,
                false,
            ));
            transcript_cells.borrow_mut().push(cell);
        }

        // Select first row when there are no cached entries.
        if !selected_row {
            if let Some(first) = list.row_at_index(0) {
                list.select_row(Some(&first));
            }
        }

        // Enter / double-click confirms selection (only if text is loaded)
        {
            let sel = Rc::clone(&sel_sender);
            let ml = main_loop.clone();
            let cands = Rc::clone(&local_candidates);
            let win = window.clone();
            list.connect_row_activated(move |_, row| {
                let idx = row.index() as usize;
                let ready = cands
                    .borrow()
                    .get(idx)
                    .is_some_and(|(_, _, text, _, is_error)| !text.is_empty() && !is_error);
                if ready {
                    win.set_visible(false);
                    if let Some(tx) = sel.borrow_mut().take() {
                        let _ = tx.send(Some(idx));
                    }
                    ml.quit();
                }
            });
        }

        // Escape cancels
        {
            let sel = Rc::clone(&sel_sender);
            let ml = main_loop.clone();
            let win = window.clone();
            let key_ctl = gtk4::EventControllerKey::new();
            key_ctl.connect_key_pressed(move |_, key, _, _| {
                if key == gtk4::gdk::Key::Escape {
                    win.set_visible(false);
                    if let Some(tx) = sel.borrow_mut().take() {
                        let _ = tx.send(None);
                    }
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
            let sel = Rc::clone(&sel_sender);
            let ml = main_loop.clone();
            window.connect_close_request(move |win| {
                win.set_visible(false);
                if let Some(tx) = sel.borrow_mut().take() {
                    let _ = tx.send(None);
                }
                ml.quit();
                glib::Propagation::Proceed
            });
        }

        // Build an error cell with a retry button (↻).
        //
        // Clicking the button replaces the error with a spinner and
        // sends a retry request through `retry_tx`.
        let make_error_cell = {
            let retry_tx = retry_tx.clone();
            let cands_for_err = Rc::clone(&local_candidates);
            let cells_for_err = Rc::clone(&transcript_cells);
            move |cell: &gtk4::Box, idx: usize, msg: &str| {
                while let Some(child) = cell.first_child() {
                    cell.remove(&child);
                }
                let hbox = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);

                let err = gtk4::Label::new(Some(msg));
                err.set_xalign(0.0);
                err.set_hexpand(true);
                err.add_css_class("error");
                hbox.append(&err);

                let retry_btn = gtk4::Button::with_label("↻");
                retry_btn.set_tooltip_text(Some("Retry transcription"));
                retry_btn.add_css_class("retry-btn");
                {
                    let tx = retry_tx.clone();
                    let cands_ref = Rc::clone(&cands_for_err);
                    let cells_ref = Rc::clone(&cells_for_err);
                    retry_btn.connect_clicked(move |_| {
                        let mut cands = cands_ref.borrow_mut();
                        if let Some((provider, model, text, _, is_error)) = cands.get_mut(idx) {
                            let _ = tx.send((*provider, model.clone()));
                            // Reset row state.
                            text.clear();
                            *is_error = false;
                            // Replace error with spinner.
                            let cells = cells_ref.borrow();
                            if let Some(cell) = cells.get(idx) {
                                while let Some(child) = cell.first_child() {
                                    cell.remove(&child);
                                }
                                let spinner = gtk4::Spinner::new();
                                spinner.start();
                                cell.append(&spinner);
                            }
                        }
                    });
                }
                hbox.append(&retry_btn);
                cell.append(&hbox);
            }
        };

        // Poll transcription results — update existing rows in-place.
        {
            let list = list.clone();
            let cands = Rc::clone(&local_candidates);
            let cells = Rc::clone(&transcript_cells);
            let make_err = make_error_cell.clone();
            glib::timeout_add_local(std::time::Duration::from_millis(50), move || {
                loop {
                    match msg_rx.try_recv() {
                        Ok(PickerMessage::Candidate(c)) => {
                            // Find the pre-created row by (provider, model).
                            let idx = {
                                let cands = cands.borrow();
                                cands
                                    .iter()
                                    .position(|(p, m, _, _, _)| *p == c.provider && *m == c.model)
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
                                } else {
                                    let mut cands = cands.borrow_mut();
                                    cands[idx].2 = c.text.clone();
                                    cells.borrow()[idx].append(&make_transcript_label(&c.text));
                                }
                            }
                        }
                        Ok(PickerMessage::InitialBatchDone) => {
                            // Mark remaining spinners as failed.
                            let failed: Vec<usize> = {
                                let mut cands = cands.borrow_mut();
                                let indices: Vec<usize> = cands
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, (_, _, text, _, is_error))| {
                                        text.is_empty() && !*is_error
                                    })
                                    .map(|(i, _)| i)
                                    .collect();
                                for &idx in &indices {
                                    cands[idx].4 = true;
                                }
                                indices
                            };
                            for idx in failed {
                                make_err(&cells.borrow()[idx], idx, "no response");
                            }
                            list.grab_focus();
                            // Keep polling — retries may still arrive.
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                            return glib::ControlFlow::Continue;
                        }
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            // All senders dropped (including retry
                            // listener) — stop polling.
                            return glib::ControlFlow::Break;
                        }
                    }
                }
            });
        }

        // Present invisible, then centre + reveal synchronously in
        // the `map` signal.  The XID is obtained directly from GDK
        // (no `_NET_CLIENT_LIST` race).
        window.set_opacity(0.0);

        {
            let window_reveal = window.clone();
            window.connect_map(move |w| {
                let xid = w
                    .surface()
                    .and_then(|s| s.downcast_ref::<gdk4_x11::X11Surface>().map(|xs| xs.xid()));
                if let Some(xid) = xid {
                    #[allow(clippy::cast_possible_truncation)]
                    let xid32 = xid as u32;
                    x11_centre_and_raise_xid(xid32);
                }
                window_reveal.set_opacity(1.0);
            });
        }

        window.present();
        list.grab_focus();
        main_loop.run();
        window.close();

        // Publish the candidate list so the caller can look up the
        // selected index after this thread finishes.
        if let Ok(mut r) = results_for_gtk.lock() {
            *r = local_candidates.borrow().clone();
        }

        Ok(())
    });

    // ── Parallel transcriptions ─────────────────────────────────
    let audio_path_for_cache = audio_path.clone();

    // Helper: run one transcription and send the result.
    fn spawn_transcription(
        tasks: &mut tokio::task::JoinSet<()>,
        tx: std::sync::mpsc::Sender<PickerMessage>,
        audio: PathBuf,
        provider: Provider,
        model: String,
        transcriber: Box<dyn BatchTranscriber>,
    ) {
        tasks.spawn(async move {
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(8),
                transcriber.transcribe_file(&audio),
            )
            .await;
            let msg = match result {
                Ok(Ok(res)) => {
                    let text = res.text.trim().to_string();
                    if text.is_empty() {
                        return;
                    }
                    PickerMessage::Candidate(PickerCandidate {
                        provider,
                        model,
                        text,
                        error: None,
                    })
                }
                Ok(Err(e)) => {
                    log::warn!("candidate {}:{} failed: {}", provider, model, e);
                    PickerMessage::Candidate(PickerCandidate {
                        provider,
                        model,
                        text: String::new(),
                        error: Some(format!("{e}")),
                    })
                }
                Err(_) => {
                    log::warn!("candidate {}:{} timed out after 8s", provider, model);
                    PickerMessage::Candidate(PickerCandidate {
                        provider,
                        model,
                        text: String::new(),
                        error: Some("timed out".into()),
                    })
                }
            };
            let _ = tx.send(msg);
        });
    }

    // Fire initial batch.
    let done_tx = msg_tx.clone();
    let retry_msg_tx = msg_tx.clone();
    drop(msg_tx); // only clones survive from here
    tokio::spawn({
        let audio = audio_path.clone();
        let tx = done_tx.clone();
        async move {
            let mut tasks = tokio::task::JoinSet::new();
            for (provider, model, transcriber) in transcribers {
                spawn_transcription(
                    &mut tasks,
                    tx.clone(),
                    audio.clone(),
                    provider,
                    model,
                    transcriber,
                );
            }
            drop(tx);
            while (tasks.join_next().await).is_some() {}
            // All initial tasks finished — notify GTK.
            let _ = done_tx.send(PickerMessage::InitialBatchDone);
        }
    });

    // ── Retry listener ──────────────────────────────────────────
    // Receives (provider, model) from the GTK retry button and
    // fires a new transcription, reusing the same result channel.
    {
        let audio = audio_path;
        let tx = retry_msg_tx;
        tokio::spawn(async move {
            while let Ok((provider, model)) = retry_rx.recv() {
                log::info!("retrying {}:{}", provider, model);
                match transcription::create_batch_transcriber(&config, provider, Some(&model)) {
                    Ok(transcriber) => {
                        let mut tasks = tokio::task::JoinSet::new();
                        spawn_transcription(
                            &mut tasks,
                            tx.clone(),
                            audio.clone(),
                            provider,
                            model,
                            transcriber,
                        );
                        while (tasks.join_next().await).is_some() {}
                    }
                    Err(e) => {
                        log::warn!(
                            "retry {}:{} transcriber creation failed: {}",
                            provider,
                            model,
                            e
                        );
                        let _ = tx.send(PickerMessage::Candidate(PickerCandidate {
                            provider,
                            model,
                            text: String::new(),
                            error: Some(format!("{e}")),
                        }));
                    }
                }
            }
            // retry_tx dropped (GTK closed) — exit.
        });
    }

    // ── Wait for result ─────────────────────────────────────────
    let selected_index = sel_rx
        .await
        .map_err(|_| TalkError::Config("GTK picker closed unexpectedly".into()))?;

    gtk_handle
        .await
        .map_err(|e| TalkError::Config(format!("GTK picker task failed: {}", e)))??;

    // Persist all successful transcription results to the picker
    // cache so that reopening the picker for the same audio file
    // returns instantly without any API calls.
    {
        let items = results
            .lock()
            .map_err(|_| TalkError::Config("results lock poisoned".into()))?;
        let to_cache: Vec<picker_cache::CachedResult> = items
            .iter()
            .filter(|(_, _, text, _, is_error)| !text.is_empty() && !*is_error)
            .map(|(p, m, t, _, _)| picker_cache::CachedResult {
                provider: p.to_string(),
                model: m.clone(),
                text: t.clone(),
            })
            .collect();
        if let Err(e) = picker_cache::write_results(&audio_path_for_cache, &to_cache) {
            log::warn!("failed to write picker cache: {}", e);
        }
    }

    match selected_index {
        Some(idx) => {
            let items = results
                .lock()
                .map_err(|_| TalkError::Config("results lock poisoned".into()))?;
            if idx < items.len() {
                let (p, m, t, cached, _is_error) = items[idx].clone();
                Ok(Some(PickerSelection {
                    provider: p,
                    model: m,
                    text: t,
                    is_cached: cached,
                }))
            } else {
                Ok(None)
            }
        }
        None => Ok(None),
    }
}

/// Maximum number of attempts to focus the target window.
const FOCUS_MAX_RETRIES: u32 = 5;

/// Initial delay between focus retry attempts (doubles each retry).
const FOCUS_INITIAL_DELAY_MS: u64 = 50;

/// Attempt to focus the target window and verify the active window
/// matches.  Retries with exponential backoff to give the window
/// manager time to settle after destroying a transient window (e.g.
/// the GTK picker).
///
/// Returns `Ok(())` when the target window is confirmed active, or
/// `Err` if focus could not be established after all retries.
async fn ensure_focus(window_id: &str) -> Result<(), TalkError> {
    let mut delay_ms = FOCUS_INITIAL_DELAY_MS;

    for attempt in 1..=FOCUS_MAX_RETRIES {
        focus_window(window_id).await;
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;

        if let Some(active) = get_active_window().await {
            if active == window_id {
                log::debug!("target window {} focused (attempt {})", window_id, attempt);
                return Ok(());
            }
            log::debug!(
                "focus attempt {}/{}: expected {}, got {}",
                attempt,
                FOCUS_MAX_RETRIES,
                window_id,
                active,
            );
        } else {
            log::debug!(
                "focus attempt {}/{}: could not determine active window",
                attempt,
                FOCUS_MAX_RETRIES,
            );
        }

        delay_ms *= 2;
    }

    Err(TalkError::Clipboard(format!(
        "could not focus target window {} after {} attempts \
         — aborting to avoid sending keys to the wrong window",
        window_id, FOCUS_MAX_RETRIES,
    )))
}

async fn paste_text_to_target(
    target_window: Option<&String>,
    text: &str,
    delete_chars_before_paste: usize,
) -> Result<(), TalkError> {
    let clipboard = X11Clipboard::new();

    if let Some(wid) = target_window {
        log::debug!("refocusing target window: {}", wid);
        ensure_focus(wid).await?;
    }

    if delete_chars_before_paste > 0 {
        log::info!("deleting {} chars before paste", delete_chars_before_paste);
        simulate_backspace(delete_chars_before_paste).await?;
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }

    let saved_clipboard = clipboard.get_text().await.ok();
    clipboard.set_text(text).await?;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    simulate_paste().await?;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    if let Some(saved) = saved_clipboard {
        let _ = clipboard.set_text(&saved).await;
    }

    Ok(())
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

/// Simulate deleting the previous text by sending repeated BackSpace.
async fn simulate_backspace(count: usize) -> Result<(), TalkError> {
    if count == 0 {
        return Ok(());
    }

    let repeat = count.to_string();
    let output = tokio::process::Command::new("xdotool")
        .args(["key", "--delay", "0", "--repeat", &repeat, "BackSpace"])
        .output()
        .await
        .map_err(|e| TalkError::Clipboard(format!("failed to run xdotool: {e}")))?;

    if !output.status.success() {
        return Err(TalkError::Clipboard(
            "xdotool backspace simulation failed".to_string(),
        ));
    }

    Ok(())
}

/// Dictate: record audio, transcribe, and paste into focused application.
pub async fn dictate(opts: DictateOpts) -> Result<(), TalkError> {
    // Toggle mode: start or stop a daemon
    if opts.toggle {
        return toggle_dispatch(
            opts.provider,
            opts.model,
            opts.realtime,
            opts.no_sounds,
            opts.no_overlay,
            opts.amplitude,
            opts.spectrum,
            opts.save.as_deref(),
            opts.verbose,
        )
        .await;
    }

    let save_path = opts.save;

    // Load configuration
    let config = Config::load(None)?;

    // Determine target window: use --target-window arg (from daemon mode)
    // or capture the currently active window.
    let target_window = if let Some(wid) = opts.target_window {
        log::debug!("using target window from argument: {}", wid);
        Some(wid)
    } else if !opts.daemon {
        let wid = get_active_window().await;
        if let Some(ref w) = wid {
            log::debug!("captured active window: {}", w);
        }
        wid
    } else {
        None
    };

    let mut input_audio_file = opts.input_audio_file.clone();
    let mut replace_char_count: Option<usize> = None;
    let mut cached_brief: Option<recording_cache::RecordingMetadataBrief> = None;
    if opts.retry_last {
        let last_audio = recording_cache::last_recording_path()
            .or_else(|_| recording_cache::latest_recording_path())?;
        let last_meta = recording_cache::last_metadata_path()
            .ok()
            .or(recording_cache::metadata_path_for_recording(&last_audio)?);
        if let Some(meta) = last_meta {
            if let Ok(previous) = recording_cache::read_metadata_brief(&meta) {
                replace_char_count = Some(previous.transcript.chars().count());
                cached_brief = Some(previous);
            }
        }
        input_audio_file = Some(last_audio);
    }

    if opts.pick {
        // Single-instance: if a picker window is already open, just
        // raise and focus it instead of opening a second one.
        if x11_centre_and_raise(PICKER_TITLE) {
            log::info!("picker already open — raised existing window");
            return Ok(());
        }

        if opts.realtime {
            return Err(TalkError::Config(
                "--pick is currently supported only in batch mode".to_string(),
            ));
        }

        let audio_path = input_audio_file.clone().ok_or_else(|| {
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

        // Determine the "already pasted" (provider, model) pair.
        // The picker cache's `selected` field takes precedence
        // (the user may have picked a different model last time).
        // Fall back to the recording metadata otherwise.
        let selected_key: Option<(Provider, String)> = picker_cache_data
            .selected
            .as_ref()
            .and_then(|s| Some((s.provider.parse::<Provider>().ok()?, s.model.clone())))
            .or_else(|| {
                cached_brief.as_ref().and_then(|b| {
                    Some((
                        b.provider.as_deref()?.parse().ok()?,
                        b.model.as_deref()?.to_string(),
                    ))
                })
            });

        // Collect all available cached transcriptions.
        let mut all_entries: Vec<(Provider, String, String)> = Vec::new();

        // From recording metadata.
        if let Some(ref brief) = cached_brief {
            if let (Some(ps), Some(m)) = (brief.provider.as_deref(), brief.model.as_deref()) {
                if let Ok(p) = ps.parse::<Provider>() {
                    all_entries.push((p, m.to_string(), brief.transcript.clone()));
                }
            }
        }

        // From picker cache results (skip duplicates).
        for cr in &picker_cache_data.results {
            if let Ok(p) = cr.provider.parse::<Provider>() {
                if !all_entries
                    .iter()
                    .any(|(ep, em, _)| *ep == p && *em == cr.model)
                {
                    all_entries.push((p, cr.model.clone(), cr.text.clone()));
                }
            }
        }

        // Build cached_entries with is_primary flag.  The selected
        // (already-pasted) entry goes first so it is pre-selected
        // in the GTK list.
        let mut cached_entries: Vec<(Provider, String, String, bool)> = Vec::new();
        if let Some((ref sp, ref sm)) = selected_key {
            if let Some(idx) = all_entries.iter().position(|(p, m, _)| p == sp && m == sm) {
                let (p, m, t) = all_entries.remove(idx);
                cached_entries.push((p, m, t, true));
            }
        }
        for (p, m, t) in all_entries {
            cached_entries.push((p, m, t, false));
        }

        log::debug!(
            "picker cache: {} cached entries (primary={}, from_cache={})",
            cached_entries.len(),
            cached_entries.iter().filter(|(_, _, _, p)| *p).count(),
            picker_cache_data.results.len(),
        );
        for (p, m, _, is_primary) in &cached_entries {
            log::debug!("  cached: {}:{} (primary={})", p, m, is_primary);
        }

        let candidates = build_retry_candidates(&config, opts.provider, opts.model.as_deref());
        log::debug!("picker candidates: {} total", candidates.len());
        for (p, m) in &candidates {
            log::debug!("  candidate: {}:{}", p, m);
        }

        // Filter out every (provider, model) pair that already has a
        // cached result — no need to re-transcribe.
        let filtered: Vec<(Provider, String)> = candidates
            .into_iter()
            .filter(|(p, m)| {
                let dominated = cached_entries
                    .iter()
                    .any(|(cp, cm, _, _)| cp == p && cm == m);
                if dominated {
                    log::debug!("  filtered out (cached): {}:{}", p, m);
                }
                !dominated
            })
            .collect();
        log::debug!(
            "picker: {} transcribers needed (after filtering)",
            filtered.len(),
        );
        for (p, m) in &filtered {
            log::debug!("  needs API call: {}:{}", p, m);
        }

        // Create all transcribers before entering GTK (needs &Config).
        let mut transcribers: Vec<(Provider, String, Box<dyn BatchTranscriber>)> = Vec::new();
        for (provider, model) in filtered {
            match transcription::create_batch_transcriber(&config, provider, Some(&model)) {
                Ok(t) => transcribers.push((provider, model, t)),
                Err(e) => log::warn!("skipping {}:{}: {}", provider, model, e),
            }
        }

        if transcribers.is_empty() && cached_entries.is_empty() {
            return Err(TalkError::Transcription(
                "no transcription providers available".to_string(),
            ));
        }

        let audio_path_for_selection = audio_path.clone();
        let selected =
            pick_with_streaming_gtk(transcribers, audio_path, cached_entries, config).await?;
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
        ) {
            log::warn!("failed to update picker selection: {}", e);
        }

        // If the user selected the cached entry, nothing to do — the
        // text is already in the target window.
        if selection.is_cached {
            log::info!("cached entry selected — no paste needed");
            return Ok(());
        }

        let delete_chars = if opts.replace_last_paste {
            // Prefer the paste-state file (written after every paste,
            // including picker selections) over recording metadata so
            // that successive picker replacements delete the correct
            // number of characters.
            recording_cache::read_last_paste_state()?
                .map(|s| s.char_count)
                .or(replace_char_count)
                .unwrap_or(0)
        } else {
            0
        };

        paste_text_to_target(target_window.as_ref(), &selection.text, delete_chars).await?;
        let _ = recording_cache::write_last_paste_state(target_window.as_deref(), &selection.text);
        println!("{}", selection.text);
        return Ok(());
    }

    // Initialize sound player (single-channel with preemption)
    let player = if opts.no_sounds {
        log::debug!("sound indicators disabled");
        None
    } else {
        match SoundPlayer::new() {
            Ok(p) => {
                log::debug!("sound player initialized");
                Some(p)
            }
            Err(e) => {
                log::warn!("sound indicators unavailable: {}", e);
                None
            }
        }
    };

    // Initialize overlay (visual indicator on X11)
    let overlay = if opts.no_overlay {
        log::debug!("visual overlay disabled");
        None
    } else {
        match OverlayHandle::new() {
            Ok(h) => {
                log::debug!("overlay initialized");
                Some(h)
            }
            Err(e) => {
                log::warn!("visual overlay unavailable: {}", e);
                None
            }
        }
    };

    // Initialize visualizer (amplitude / spectrum panels)
    let visualizer = if opts.amplitude || opts.spectrum {
        match VisualizerHandle::new(opts.amplitude, opts.spectrum) {
            Ok(h) => {
                log::debug!(
                    "visualizer initialized (amplitude={}, spectrum={})",
                    opts.amplitude,
                    opts.spectrum,
                );
                Some(h)
            }
            Err(e) => {
                log::warn!("visualizer unavailable: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Play start sound
    if let Some(ref p) = player {
        log::debug!("playing start sound");
        p.play_start().await;
    }

    // Show recording indicator
    if let Some(ref o) = overlay {
        log::debug!("showing recording overlay");
        o.show(IndicatorKind::Recording);
    }

    // Show visualizer panels (positioned relative to 182px recording badge)
    if let Some(ref viz) = visualizer {
        log::debug!("showing visualizer");
        viz.show(182);
    }

    // Start boop loop (every 5 seconds)
    let boop_token = player
        .as_ref()
        .map(|p| p.start_boop_loop(std::time::Duration::from_secs(5)));

    // Generate cache recording path (always, even without --save)
    let (cache_wav_path, cache_timestamp) = recording_cache::generate_recording_path()?;
    log::info!("cache recording: {}", cache_wav_path.display());

    let provider = resolve_provider(opts.provider, &config);
    let effective_model = resolve_model(opts.model.as_deref(), &config, provider, opts.realtime);

    // Create audio source: live microphone or WAV file input.
    //
    // For live capture, record at the device's native rate (typically
    // 48 kHz) and downsample to 16 kHz with a proper anti-aliasing
    // filter.  WAV file input is assumed to be 16 kHz already.
    let encode_config = AudioConfig::new(); // 16 kHz target for encoder
    let from_file = input_audio_file.is_some();
    let (mut capture, capture_rate): (Box<dyn AudioCapture>, u32) =
        if let Some(ref path) = input_audio_file {
            log::info!("using audio file input: {}", path.display());
            (
                Box::new(WavFileSource::new(path, &encode_config)?),
                encode_config.sample_rate,
            )
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
            log::info!(
                "capture at {}Hz (PipeWire), target {}Hz",
                rate,
                encode_config.sample_rate
            );
            (
                Box::new(PipeWireCapture::new(capture_config)) as Box<dyn AudioCapture>,
                rate,
            )
        };

    log::info!(
        "starting {} transcription{}",
        if opts.realtime { "realtime" } else { "batch" },
        if from_file { " (from file)" } else { "" }
    );

    let result = if opts.realtime {
        // Realtime mode (--realtime): stream audio over WebSocket.
        // Each segment is pasted into the focused application as it
        // arrives, providing real-time feedback while dictating.

        // Save clipboard and focus target window before recording starts
        let rt_clipboard = X11Clipboard::new();
        let saved_clipboard = rt_clipboard.get_text().await.ok();
        if let Some(ref wid) = target_window {
            log::debug!("pre-focusing target window: {}", wid);
            if !focus_window(wid).await {
                log::warn!("could not pre-focus target window {}", wid);
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // Create segment channel for per-segment pasting
        let (seg_tx, mut seg_rx) = tokio::sync::mpsc::channel::<String>(32);

        // Spawn paste consumer: each segment is pasted immediately
        let paste_task = tokio::spawn(async move {
            let paste_clip = X11Clipboard::new();
            let mut is_first = true;
            while let Some(segment) = seg_rx.recv().await {
                let paste_text = if is_first {
                    is_first = false;
                    segment
                } else {
                    format!(" {}", segment)
                };
                if let Err(e) = paste_clip.set_text(&paste_text).await {
                    log::warn!("per-segment clipboard set failed: {}", e);
                    continue;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                if let Err(e) = simulate_paste().await {
                    log::warn!("per-segment paste failed: {}", e);
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        });

        // Player and boop_token are passed so the stop sound fires
        // immediately when the user toggles — not after the WebSocket
        // finishes collecting transcription results.
        let raw_audio_rx = capture.start()?;
        let capture_chunk = (capture_rate as usize * CHUNK_DURATION_MS as usize) / 1000;
        let audio_rx = resample::spawn_resample_task(
            capture_rate,
            encode_config.sample_rate,
            raw_audio_rx,
            capture_chunk,
        )?;
        let result = dictate_realtime(
            config,
            provider,
            opts.model.as_deref(),
            &cache_wav_path,
            audio_rx,
            &mut *capture,
            from_file,
            player.as_ref(),
            boop_token.as_ref(),
            Some(seg_tx),
            visualizer.as_ref(),
        )
        .await?;

        // Wait for all pending pastes to complete
        if let Err(e) = paste_task.await {
            log::warn!("paste task error: {}", e);
        }

        // Restore original clipboard
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        if let Some(saved) = saved_clipboard {
            log::debug!("restoring original clipboard");
            let _ = rt_clipboard.set_text(&saved).await;
        }

        result
    } else {
        // Batch mode (default): capture audio, encode, then transcribe
        let transcriber =
            transcription::create_batch_transcriber(&config, provider, opts.model.as_deref())?;

        // Pre-flight: verify provider connectivity and model validity
        // before starting audio capture.
        log::info!("validating {} provider configuration", provider);
        transcriber.validate().await?;

        let raw_audio_rx = capture.start()?;
        let capture_chunk = (capture_rate as usize * CHUNK_DURATION_MS as usize) / 1000;
        let audio_rx = resample::spawn_resample_task(
            capture_rate,
            encode_config.sample_rate,
            raw_audio_rx,
            capture_chunk,
        )?;

        dictate_streaming(
            &mut *capture,
            from_file,
            encode_config.clone(),
            audio_rx,
            &cache_wav_path,
            transcriber,
        )
        .await?
    };

    // Stop boop loop (idempotent — may already be cancelled by
    // dictate_realtime for realtime mode).
    if let Some(token) = boop_token {
        log::debug!("stopping boop loop");
        token.cancel();
    }

    // Hide recording indicator and visualizer
    if let Some(ref o) = overlay {
        log::debug!("hiding overlay");
        o.hide();
    }
    if let Some(ref viz) = visualizer {
        log::debug!("hiding visualizer");
        viz.hide();
    }

    // For batch mode (default), play stop sound here (realtime mode
    // already played it inside dictate_realtime on SIGINT).
    if !opts.realtime {
        if let Some(ref p) = player {
            log::debug!("playing stop sound");
            p.play_stop().await;
        }
    }

    let metadata = result.metadata;
    let text = result.text.trim().to_string();

    // Write recording cache metadata and rotate old entries.
    let cache_wav_filename = cache_wav_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("recording.wav");
    let cache_meta_path = recording_cache::write_metadata(
        &cache_timestamp,
        provider,
        &effective_model,
        opts.realtime,
        &text,
        cache_wav_filename,
        &metadata,
    );
    if let Err(ref e) = cache_meta_path {
        log::warn!("failed to write recording metadata: {}", e);
    }
    if let Err(e) = recording_cache::rotate_cache() {
        log::warn!("failed to rotate recording cache: {}", e);
    }

    // Copy cache WAV to --save path if specified
    if let Some(ref path) = save_path {
        if let Err(e) = std::fs::copy(&cache_wav_path, path) {
            log::warn!("failed to copy recording to {}: {}", path.display(), e);
        } else {
            log::info!("audio saved to: {}", path.display());
        }
    }

    // Copy cache metadata YAML to --output-yaml path if specified
    if let Some(ref yaml_path) = opts.output_yaml {
        match cache_meta_path {
            Ok(ref src) => {
                if let Err(e) = std::fs::copy(src, yaml_path) {
                    log::warn!("failed to copy metadata to {}: {}", yaml_path.display(), e);
                } else {
                    log::info!("metadata YAML saved to: {}", yaml_path.display());
                }
            }
            Err(_) => {
                log::warn!("skipping --output-yaml: cache metadata was not written");
            }
        }
    }

    if text.is_empty() {
        log::warn!("empty transcription — nothing to paste");
        return Ok(());
    }

    log::info!("transcription: {}", text);

    // Paste into focused application (batch mode only;
    // realtime mode pastes per-segment during recording)
    if !opts.realtime {
        paste_text_to_target(target_window.as_ref(), &text, 0).await?;
        let _ = recording_cache::write_last_paste_state(target_window.as_deref(), &text);
    }

    // Print transcription to stdout (batch mode only;
    // realtime mode already prints segments as they arrive)
    if !opts.realtime {
        println!("{}", text);
    }

    // If running as daemon, clean up PID file on normal exit
    if opts.daemon {
        if let Ok(pid_file) = daemon::pid_path() {
            let _ = daemon::remove_pid_file(&pid_file);
        }
    }

    Ok(())
}

/// Toggle dispatch: start a new daemon or stop a running one.
#[allow(clippy::too_many_arguments)]
async fn toggle_dispatch(
    provider: Option<Provider>,
    model: Option<String>,
    realtime: bool,
    no_sounds: bool,
    no_overlay: bool,
    amplitude: bool,
    spectrum: bool,
    save: Option<&std::path::Path>,
    verbose: u8,
) -> Result<(), TalkError> {
    let pid_file = daemon::pid_path()?;

    // Acquire exclusive lock to prevent race between concurrent toggle calls
    let _lock = daemon::acquire_lock()?;

    match daemon::check_status(&pid_file)? {
        DaemonStatus::NotRunning => {
            toggle_start(
                &pid_file, provider, model, realtime, no_sounds, no_overlay, amplitude, spectrum,
                save, verbose,
            )
            .await
        }
        DaemonStatus::Running { pid } => toggle_stop(pid, &pid_file),
    }
}

/// Start a new daemon: capture window, spawn detached dictate process, write PID.
#[allow(clippy::too_many_arguments)]
async fn toggle_start(
    pid_file: &std::path::Path,
    provider: Option<Provider>,
    model: Option<String>,
    realtime: bool,
    no_sounds: bool,
    no_overlay: bool,
    amplitude: bool,
    spectrum: bool,
    save: Option<&std::path::Path>,
    verbose: u8,
) -> Result<(), TalkError> {
    // Pre-flight: validate provider/model before spawning the daemon
    // so the user gets immediate feedback on misconfiguration.
    let config = Config::load(None)?;
    let effective_provider = resolve_provider(provider, &config);
    log::info!("validating {} provider configuration", effective_provider);
    if realtime {
        let t = transcription::create_realtime_transcriber(
            &config,
            effective_provider,
            model.as_deref(),
        )?;
        t.validate().await?;
    } else {
        let t =
            transcription::create_batch_transcriber(&config, effective_provider, model.as_deref())?;
        t.validate().await?;
    }

    // Capture active window before spawning daemon
    let target_window = get_active_window().await;

    // Find our own executable
    let exe = std::env::current_exe()
        .map_err(|e| TalkError::Config(format!("failed to determine current executable: {}", e)))?;

    // Build daemon command: talk-rs [-v...] dictate --daemon [--realtime] [--no-sounds] [--target-window=WID]
    let mut cmd = std::process::Command::new(&exe);

    // Forward verbosity level (before subcommand)
    if verbose > 0 {
        cmd.arg(format!("-{}", "v".repeat(verbose as usize)));
    }

    cmd.arg("dictate").arg("--daemon");

    if let Some(p) = provider {
        cmd.arg("--provider").arg(p.to_string());
    }

    if let Some(ref m) = model {
        cmd.arg("--model").arg(m);
    }

    if realtime {
        cmd.arg("--realtime");
    }

    if no_sounds {
        cmd.arg("--no-sounds");
    }

    if no_overlay {
        cmd.arg("--no-overlay");
    }

    if amplitude {
        cmd.arg("--amplitude");
    }

    if spectrum {
        cmd.arg("--spectrum");
    }

    if let Some(path) = save {
        cmd.arg("--save").arg(path);
    }

    if let Some(ref wid) = target_window {
        cmd.arg("--target-window").arg(wid);
    }

    // Redirect stdout/stderr to log file
    let log_file_path = daemon::log_path()?;
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_path)
        .map_err(|e| {
            TalkError::Config(format!(
                "failed to open log file {}: {}",
                log_file_path.display(),
                e
            ))
        })?;
    let log_stderr = log_file
        .try_clone()
        .map_err(|e| TalkError::Config(format!("failed to clone log file handle: {}", e)))?;

    cmd.stdout(std::process::Stdio::from(log_file));
    cmd.stderr(std::process::Stdio::from(log_stderr));
    cmd.stdin(std::process::Stdio::null());

    // Create new process group (equivalent to setsid for signal isolation)
    cmd.process_group(0);

    let child = cmd
        .spawn()
        .map_err(|e| TalkError::Config(format!("failed to spawn daemon process: {}", e)))?;

    let child_pid = child.id();
    daemon::write_pid_file(pid_file, child_pid)?;

    log::info!(
        "dictation started (PID {}, logs: {})",
        child_pid,
        log_file_path.display()
    );

    Ok(())
}

/// Stop a running daemon: SIGINT, wait, SIGTERM fallback, clean up PID file.
fn toggle_stop(pid: u32, pid_file: &std::path::Path) -> Result<(), TalkError> {
    log::info!("stopping dictation (PID {})", pid);
    daemon::stop_daemon(pid, pid_file)?;
    log::info!("dictation stopped");
    Ok(())
}

/// Flush completed sentences from the live buffer to stdout.
///
/// Scans the buffer for sentence-ending punctuation (`.` `!` `?` `。` `！` `？`)
/// followed by whitespace. Everything up to and including the punctuation is
/// emitted as a line on stdout and appended to `segments`. The remainder stays
/// in the buffer for further accumulation.
fn flush_sentences(buffer: &mut String, segments: &mut Vec<String>) {
    loop {
        // Find the earliest sentence-ending punctuation followed by whitespace.
        let boundary = buffer.char_indices().position(|(i, ch)| {
            if matches!(ch, '。' | '！' | '？') {
                // CJK sentence-ending punctuation: always a boundary
                // (no space expected between CJK sentences)
                true
            } else if matches!(ch, '.' | '!' | '?') {
                // Latin sentence-ending punctuation: require whitespace
                // or end-of-string after it to avoid splitting "3.14"
                let after = i + ch.len_utf8();
                after >= buffer.len() || buffer[after..].starts_with(|c: char| c.is_whitespace())
            } else {
                false
            }
        });

        let Some(pos) = boundary else {
            break;
        };

        // Convert char position back to byte offset (including the punctuation char)
        let (byte_offset, punct_char) = buffer.char_indices().nth(pos).unwrap_or((0, '.'));
        let split_at = byte_offset + punct_char.len_utf8();

        let sentence = buffer[..split_at].trim().to_string();
        if !sentence.is_empty() {
            println!("{}", sentence);
            segments.push(sentence);
        }

        // Remove the emitted sentence + any leading whitespace from the remainder
        let remainder = buffer[split_at..].trim_start().to_string();
        // Clear stderr live preview
        let blank = " ".repeat(buffer.len());
        eprint!("\r{}\r", blank);

        *buffer = remainder;
        if !buffer.is_empty() {
            eprint!("\r{}", buffer);
        }
    }
}

/// Realtime dictation mode via WebSocket.
///
/// Streams raw PCM audio to the transcription API and receives
/// incremental transcription events. Returns the accumulated text.
///
/// Audio is always tee'd to `cache_wav_path` so the recording is
/// cached for later review.
///
/// `player` and `boop_token` are passed so that when recording stops
/// (SIGINT), the stop sound fires immediately — the user hears it the
/// instant they toggle, not after the WebSocket finishes.
///
/// When `visualizer` is provided, the live transcription text is pushed
/// to the text overlay as words arrive.
#[allow(clippy::too_many_arguments)]
async fn dictate_realtime(
    config: Config,
    provider: Provider,
    model: Option<&str>,
    cache_wav_path: &std::path::Path,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    capture: &mut dyn AudioCapture,
    from_file: bool,
    player: Option<&SoundPlayer>,
    boop_token: Option<&CancellationToken>,
    segment_tx: Option<tokio::sync::mpsc::Sender<String>>,
    visualizer: Option<&VisualizerHandle>,
) -> Result<TranscriptionResult, TalkError> {
    // Create and validate the transcriber before starting audio capture
    // so the user gets immediate feedback on misconfiguration.
    let transcriber = transcription::create_realtime_transcriber(&config, provider, model)?;
    log::info!("validating {} provider configuration", provider);
    transcriber.validate().await?;

    // Always tee audio to the cache WAV for recording cache.
    log::info!("caching audio to: {}", cache_wav_path.display());
    let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
    let wav_task = tokio::spawn(audio_tee_to_wav(
        audio_rx,
        fwd_tx,
        cache_wav_path.to_path_buf(),
        AudioConfig::new(),
    ));
    let audio_rx = fwd_rx;

    let mut event_rx = transcriber.transcribe_realtime(audio_rx).await?;
    let started = std::time::Instant::now();

    if from_file {
        log::info!("transcribing audio file (realtime)...");
    } else {
        log::info!("recording (realtime)... press Ctrl+C to stop");
    }

    let capture_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let capture_stop_clone = capture_stop.clone();

    // Spawn Ctrl+C handler that stops capture (also works for file
    // input — early abort by the user).
    let ctrlc_task = tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        capture_stop_clone.store(true, std::sync::atomic::Ordering::Release);
    });

    // Completed sentences/phrases emitted so far.
    let mut segments: Vec<String> = Vec::new();
    // Buffer for the current in-progress phrase (live TextDelta).
    let mut current_line = String::new();
    let mut detected_language: Option<String> = None;
    let mut unknown_event_types: Vec<String> = Vec::new();
    let mut event_counts: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    let mut api_segment_count: usize = 0;
    let mut session_id: Option<String> = None;
    let mut conversation_id: Option<String> = None;
    let mut last_rate_limits: Option<serde_json::Value> = None;
    let mut ws_upgrade_headers: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();

    let bump = |key: &str, counts: &mut std::collections::BTreeMap<String, u64>| {
        let entry = counts.entry(key.to_string()).or_insert(0);
        *entry += 1;
    };

    loop {
        // Check if Ctrl+C was pressed — stop capture to trigger end-of-audio
        if capture_stop.load(std::sync::atomic::Ordering::Acquire) {
            log::info!("stopping recording");

            // Immediate audible + visual feedback: the user hears the
            // stop sound the instant they toggle, not after the
            // transcription WebSocket finishes.
            if let Some(token) = boop_token {
                token.cancel();
            }
            if let Some(p) = player {
                let stop = p.sounds.stop.clone();
                p.play(&stop);
            }

            capture.stop()?;
            // Reset so we don't stop again
            capture_stop.store(false, std::sync::atomic::Ordering::Release);
        }

        tokio::select! {
            event = event_rx.recv() => {
                match event {
                    Some(TranscriptionEvent::TextDelta { text }) => {
                        bump("text_delta", &mut event_counts);
                        current_line.push_str(&text);
                        eprint!("\r{}", current_line);

                        // Push live text to the overlay.
                        if let Some(viz) = visualizer {
                            let mut live = segments.join(" ");
                            if !live.is_empty() && !current_line.is_empty() {
                                live.push(' ');
                            }
                            live.push_str(&current_line);
                            viz.set_text(&live);
                        }

                        // Flush completed sentences from the buffer.
                        // Split on sentence-ending punctuation followed by
                        // whitespace or end-of-string.
                        let prev_count = segments.len();
                        flush_sentences(&mut current_line, &mut segments);
                        if let Some(ref tx) = segment_tx {
                            for seg in &segments[prev_count..] {
                                let _ = tx.send(seg.clone()).await;
                            }
                        }
                    }
                    Some(TranscriptionEvent::SegmentDelta { text, .. }) => {
                        bump("segment_delta", &mut event_counts);
                        api_segment_count += 1;
                        // If the API sends segment events, use them as
                        // authoritative sentence boundaries.
                        let segment_text = text.trim().to_string();
                        if !segment_text.is_empty() {
                            println!("{}", segment_text);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(segment_text.clone()).await;
                            }
                            segments.push(segment_text);
                        }
                        let blank = " ".repeat(current_line.len());
                        eprint!("\r{}\r", blank);
                        current_line.clear();

                        // Update overlay with completed segments.
                        if let Some(viz) = visualizer {
                            viz.set_text(&segments.join(" "));
                        }
                    }
                    Some(TranscriptionEvent::Done) => {
                        bump("done", &mut event_counts);
                        // Flush any trailing text that didn't end with punctuation
                        let trailing = current_line.trim().to_string();
                        if !trailing.is_empty() {
                            println!("{}", trailing);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(trailing.clone()).await;
                            }
                            segments.push(trailing);
                        }
                        eprintln!();
                        break;
                    }
                    Some(TranscriptionEvent::Error { message }) => {
                        bump("error", &mut event_counts);
                        return Err(TalkError::Transcription(format!(
                            "Realtime transcription error: {}",
                            message
                        )));
                    }
                    Some(TranscriptionEvent::SessionCreated) => {
                        bump("session_created", &mut event_counts);
                        log::debug!("session created event received");
                    }
                    Some(TranscriptionEvent::SessionInfo { session_id: sid, conversation_id: cid }) => {
                        bump("session_info", &mut event_counts);
                        if sid.is_some() {
                            session_id = sid;
                        }
                        if cid.is_some() {
                            conversation_id = cid;
                        }
                    }
                    Some(TranscriptionEvent::RateLimitsUpdated { raw }) => {
                        bump("rate_limits_updated", &mut event_counts);
                        last_rate_limits = Some(raw);
                    }
                    Some(TranscriptionEvent::TransportMetadata { headers }) => {
                        bump("transport_metadata", &mut event_counts);
                        ws_upgrade_headers.extend(headers);
                    }
                    Some(TranscriptionEvent::Language { language }) => {
                        bump("language", &mut event_counts);
                        log::info!("detected language: {}", language);
                        detected_language = Some(language);
                    }
                    Some(TranscriptionEvent::Unknown { event_type, .. }) => {
                        bump("unknown", &mut event_counts);
                        if let Some(kind) = event_type {
                            bump(&format!("event:{kind}"), &mut event_counts);
                            if !unknown_event_types.contains(&kind) {
                                unknown_event_types.push(kind);
                            }
                        }
                    }
                    None => {
                        // Channel closed without Done event
                        let trailing = current_line.trim().to_string();
                        if !trailing.is_empty() {
                            println!("{}", trailing);
                            if let Some(ref tx) = segment_tx {
                                let _ = tx.send(trailing.clone()).await;
                            }
                            segments.push(trailing);
                        }
                        eprintln!();
                        break;
                    }
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                // Periodic check for Ctrl+C flag
            }
        }
    }

    ctrlc_task.abort();

    // Wait for WAV tee task to finish writing
    match wav_task.await {
        Ok(Ok(())) => log::debug!("cache WAV saved"),
        Ok(Err(e)) => log::warn!("cache WAV write error: {}", e),
        Err(e) => log::warn!("cache WAV task panicked: {}", e),
    }

    let provider_specific = match provider {
        Provider::OpenAI => Some(ProviderSpecificMetadata::OpenAI(OpenAIProviderMetadata {
            model: model.map(str::to_string),
            usage_raw: None,
            rate_limit_headers: std::collections::BTreeMap::new(),
            unknown_event_types,
            realtime: Some(OpenAIRealtimeMetadata {
                session_id,
                conversation_id,
                event_counts,
                last_rate_limits,
                ws_upgrade_headers: ws_upgrade_headers.clone(),
            }),
        })),
        Provider::Mistral => Some(ProviderSpecificMetadata::Mistral(MistralProviderMetadata {
            model: model.map(str::to_string),
            usage_raw: None,
            unknown_event_types,
        })),
    };

    Ok(TranscriptionResult {
        text: segments.join(" "),
        metadata: TranscriptionMetadata {
            request_latency_ms: None,
            session_elapsed_ms: Some(started.elapsed().as_millis() as u64),
            request_id: ws_upgrade_headers.get("x-request-id").cloned(),
            provider_processing_ms: ws_upgrade_headers
                .get("openai-processing-ms")
                .and_then(|s| s.parse::<u64>().ok()),
            detected_language,
            audio_seconds: None,
            segment_count: Some(if api_segment_count > 0 {
                api_segment_count
            } else {
                segments.len()
            }),
            word_count: None,
            token_usage: None,
            provider_specific,
        },
    })
}

/// Tee audio from `source` into both a WAV file and a forwarding channel.
///
/// Each `Vec<i16>` chunk is forwarded to `fwd_tx` for the transcriber
/// and then written as raw PCM s16le to the WAV file.  The WAV is an
/// exact mirror of what the API received — not more, not less.  When the
/// source channel closes, the WAV header is patched with the final size.
async fn audio_tee_to_wav(
    mut source: tokio::sync::mpsc::Receiver<Vec<i16>>,
    fwd_tx: tokio::sync::mpsc::Sender<Vec<i16>>,
    wav_path: PathBuf,
    audio_config: AudioConfig,
) -> Result<(), TalkError> {
    let mut writer = WavWriter::new(audio_config);
    let header = writer.header()?;

    let mut file = tokio::fs::File::create(&wav_path)
        .await
        .map_err(TalkError::Io)?;
    file.write_all(&header).await.map_err(TalkError::Io)?;

    let mut total_samples: u64 = 0;

    while let Some(pcm_chunk) = source.recv().await {
        // Forward to transcriber first.  Only write to the debug WAV
        // what was successfully forwarded, so the capture is an exact
        // mirror of what the API received.
        let wav_chunk = pcm_chunk.clone();
        if fwd_tx.send(pcm_chunk).await.is_err() {
            log::debug!("transcriber channel closed, stopping debug WAV");
            break;
        }

        total_samples += wav_chunk.len() as u64;
        let pcm_bytes = writer.write_pcm(&wav_chunk)?;
        file.write_all(&pcm_bytes).await.map_err(TalkError::Io)?;
    }

    // No drain loop — the WAV contains exactly what was forwarded.

    // Signal end-of-audio to transcriber before WAV finalisation.
    drop(fwd_tx);

    // Patch WAV header with actual data size
    let final_header = writer.finalize()?;
    file.seek(std::io::SeekFrom::Start(0))
        .await
        .map_err(TalkError::Io)?;
    file.write_all(&final_header).await.map_err(TalkError::Io)?;
    file.sync_all().await.map_err(TalkError::Io)?;

    log::info!(
        "debug WAV: {} samples ({:.1}s) saved to {}",
        total_samples,
        total_samples as f64 / 16000.0,
        wav_path.display()
    );

    Ok(())
}

/// Batch dictation mode.
///
/// Encodes audio and streams it directly to a single transcription request.
/// Audio is also tee'd to `cache_wav_path` for the recording cache.
///
/// When `from_file` is true, the function waits for the audio source to
/// exhaust naturally (in addition to allowing Ctrl+C for early abort).
async fn dictate_streaming(
    capture: &mut dyn AudioCapture,
    from_file: bool,
    audio_config: AudioConfig,
    audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    cache_wav_path: &std::path::Path,
    transcriber: Box<dyn BatchTranscriber>,
) -> Result<TranscriptionResult, TalkError> {
    // Tee raw PCM to cache WAV before encoding
    log::info!("caching audio to: {}", cache_wav_path.display());
    let (fwd_tx, fwd_rx) = tokio::sync::mpsc::channel::<Vec<i16>>(100);
    let cache_wav_task = tokio::spawn(audio_tee_to_wav(
        audio_rx,
        fwd_tx,
        cache_wav_path.to_path_buf(),
        audio_config.clone(),
    ));

    // Create channel for streaming encoded audio to transcriber
    let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(25);

    // Oneshot to signal when encoding is done (file source exhausted
    // or capture stopped).  Enables the stop logic to race Ctrl+C
    // against natural completion for file input.
    let (encode_done_tx, encode_done_rx) = tokio::sync::oneshot::channel::<()>();

    // Spawn encode task: PCM → OGG Opus → stream_tx
    let encode_task = tokio::spawn(async move {
        let mut rx = fwd_rx;
        let mut writer = match OggOpusWriter::new(audio_config) {
            Ok(writer) => writer,
            Err(err) => {
                log::error!("error creating audio writer: {}", err);
                return Err(err);
            }
        };

        let header = match writer.header() {
            Ok(bytes) => bytes,
            Err(err) => {
                log::error!("error creating audio header: {}", err);
                return Err(err);
            }
        };
        if stream_tx.send(header).await.is_err() {
            log::warn!("transcription stream closed during header send");
            let _ = encode_done_tx.send(());
            return Ok::<(), TalkError>(());
        }

        while let Some(pcm_chunk) = rx.recv().await {
            let encoded_data = match writer.write_pcm(&pcm_chunk) {
                Ok(data) => data,
                Err(err) => {
                    log::error!("error encoding audio: {}", err);
                    let _ = encode_done_tx.send(());
                    return Err(err);
                }
            };
            if !encoded_data.is_empty()
                && stream_tx.send(encoded_data).await.is_err()
            {
                log::warn!("transcription stream closed during audio send");
                break;
            }
        }

        // Finalize writer
        match writer.finalize() {
            Ok(remaining) => {
                if !remaining.is_empty() {
                    let _ = stream_tx.send(remaining).await;
                }
            }
            Err(err) => {
                log::error!("error finalizing audio writer: {}", err);
                let _ = encode_done_tx.send(());
                return Err(err);
            }
        }

        // stream_tx is dropped here, closing the channel
        let _ = encode_done_tx.send(());
        Ok::<(), TalkError>(())
    });

    // Spawn transcription task
    let transcribe_task =
        tokio::spawn(async move { transcriber.transcribe_stream(stream_rx, "audio.ogg").await });

    // Wait for recording to end: Ctrl+C for live mic, natural
    // completion for file input, or Ctrl+C to abort file early.
    if from_file {
        log::info!("transcribing audio file (batch)...");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                log::info!("aborting");
                capture.stop()?;
            }
            _ = encode_done_rx => {
                log::info!("audio file playback complete");
            }
        }
    } else {
        log::info!("recording (batch)... press Ctrl+C to stop and transcribe");
        tokio::signal::ctrl_c()
            .await
            .map_err(|err| TalkError::Audio(format!("Failed to listen for Ctrl+C: {}", err)))?;
        log::info!("stopping recording");
        capture.stop()?;
    }

    // Wait for cache WAV and encode tasks
    match cache_wav_task.await {
        Ok(Ok(())) => log::debug!("cache WAV task completed"),
        Ok(Err(err)) => log::warn!("cache WAV write error: {}", err),
        Err(err) => log::warn!("cache WAV task panicked: {}", err),
    }
    match encode_task.await {
        Ok(Ok(())) => log::debug!("encode task completed"),
        Ok(Err(err)) => log::error!("encode error: {}", err),
        Err(err) => log::error!("encode task panicked: {}", err),
    }

    // Wait for transcription result
    let result = match transcribe_task.await {
        Ok(Ok(result)) => result,
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

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dictate_pipeline_with_mocks() {
        use crate::core::audio::mock::MockAudioCapture;
        use crate::core::audio::{AudioCapture, AudioWriter, OggOpusWriter};
        use crate::core::clipboard::MockClipboard;
        use crate::core::config::AudioConfig;
        use crate::core::transcription::{BatchTranscriber, MockBatchTranscriber};

        let audio_config = AudioConfig::new();

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
        let transcriber = MockBatchTranscriber::new("Hello world from dictation");
        let transcribe_task =
            tokio::spawn(
                async move { transcriber.transcribe_stream(stream_rx, "audio.ogg").await },
            );

        // Wait for encode to finish
        encode_task.await.expect("encode task");

        // Stop capture
        capture.stop().expect("stop capture");

        // Get transcription
        let result = transcribe_task
            .await
            .expect("transcribe task")
            .expect("transcription");
        let text = result.text;
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

    #[test]
    fn test_flush_sentences_single_sentence() {
        let mut buf = "Hello world.".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["Hello world."]);
        assert_eq!(buf, "");
    }

    #[test]
    fn test_flush_sentences_trailing_partial() {
        let mut buf = "First sentence. And then".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["First sentence."]);
        assert_eq!(buf, "And then");
    }

    #[test]
    fn test_flush_sentences_multiple() {
        let mut buf = "One. Two! Three? Rest".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["One.", "Two!", "Three?"]);
        assert_eq!(buf, "Rest");
    }

    #[test]
    fn test_flush_sentences_no_punctuation() {
        let mut buf = "no punctuation here".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert!(segs.is_empty());
        assert_eq!(buf, "no punctuation here");
    }

    #[test]
    fn test_flush_sentences_chinese_punctuation() {
        let mut buf = "你好。世界".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["你好。"]);
        assert_eq!(buf, "世界");
    }

    #[test]
    fn test_flush_sentences_period_at_end() {
        let mut buf = "End of text.".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["End of text."]);
        assert_eq!(buf, "");
    }
}
