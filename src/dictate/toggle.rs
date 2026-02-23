//! Toggle dispatch: start or stop a dictation daemon.
//!
//! Extracted from `dictate.rs` — handles the `--toggle` flag logic:
//! spawn a new daemon process or stop a running one, with provider
//! validation after spawn.

use super::models::resolve_provider;
use crate::cli::action::paste::get_active_window;
use crate::core::config::{Config, Provider};
use crate::core::daemon::{self, DaemonStatus};
use crate::core::error::TalkError;
use crate::core::transcription;
use crate::core::visualizer::VisualizerHandle;
use std::os::unix::process::CommandExt as _;

/// Minimum time (ms) to keep a validation error visible in the target
/// window before returning, so the user has time to read it.
const VALIDATION_ERROR_DISPLAY_MS: u64 = 2000;

/// Toggle dispatch: start a new daemon or stop a running one.
#[allow(clippy::too_many_arguments)]
pub async fn toggle_dispatch(
    provider: Option<Provider>,
    model: Option<String>,
    diarize: bool,
    realtime: bool,
    no_sounds: bool,
    no_boop: bool,
    no_chunk_paste: bool,
    monitor: bool,
    no_overlay: bool,
    amplitude: bool,
    spectrum: bool,
    save: Option<&std::path::Path>,
    verbose: u8,
) -> Result<(), TalkError> {
    let pid_file = daemon::pid_path()?;

    // Phase 1 — under lock: check status + spawn daemon / stop daemon.
    // The lock is scoped so it is released as soon as the PID file is
    // written (start) or removed (stop).  This avoids blocking a
    // subsequent toggle press while the network validation is in flight.
    let spawn_ctx: Option<SpawnContext> = {
        let _lock = daemon::acquire_lock()?;

        let status = daemon::check_status(&pid_file)?;
        daemon::trace(&format!(
            "[DBG] toggle_dispatch: check_status = {:?}",
            status
        ));
        match status {
            DaemonStatus::NotRunning => Some(
                toggle_spawn(
                    &pid_file,
                    provider,
                    model,
                    diarize,
                    realtime,
                    no_sounds,
                    no_boop,
                    no_chunk_paste,
                    monitor,
                    no_overlay,
                    amplitude,
                    spectrum,
                    save,
                    verbose,
                )
                .await?,
            ),
            DaemonStatus::Running { pid } => {
                toggle_stop(pid, &pid_file)?;
                None
            }
        }
    }; // _lock dropped — second toggle can proceed immediately

    // Phase 2 — without lock: validate the provider configuration.
    // The daemon is already recording; validation runs concurrently.
    if let Some(ctx) = spawn_ctx {
        toggle_validate(ctx).await?;
    }

    Ok(())
}

/// Everything [`toggle_validate`] needs after the daemon has been
/// spawned (and the lock released).
struct SpawnContext {
    child_pid: u32,
    pid_file: std::path::PathBuf,
    config: Config,
    effective_provider: Provider,
    model: Option<String>,
    diarize: bool,
    realtime: bool,
}

/// Spawn the daemon process and write the PID file.
///
/// This runs **under the exclusive lock** and must stay fast — no
/// network I/O.  Returns context needed for the subsequent validation
/// phase.
#[allow(clippy::too_many_arguments)]
async fn toggle_spawn(
    pid_file: &std::path::Path,
    provider: Option<Provider>,
    model: Option<String>,
    diarize: bool,
    realtime: bool,
    no_sounds: bool,
    no_boop: bool,
    no_chunk_paste: bool,
    monitor: bool,
    no_overlay: bool,
    amplitude: bool,
    spectrum: bool,
    save: Option<&std::path::Path>,
    verbose: u8,
) -> Result<SpawnContext, TalkError> {
    // Instant local checks only — no network I/O.
    let config = Config::load(None)?;
    let effective_provider = resolve_provider(provider, &config);

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

    if diarize {
        cmd.arg("--diarize");
    }

    if realtime {
        cmd.arg("--realtime");
    }

    if no_sounds {
        cmd.arg("--no-sounds");
    }

    if no_boop {
        cmd.arg("--no-boop");
    }

    if no_chunk_paste {
        cmd.arg("--no-chunk-paste");
    }

    if monitor {
        cmd.arg("--monitor");
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

    // Spawn daemon immediately — recording starts without waiting for
    // network validation.
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

    Ok(SpawnContext {
        child_pid,
        pid_file: pid_file.to_path_buf(),
        config,
        effective_provider,
        model,
        diarize,
        realtime,
    })
}

/// Validate the provider configuration after the daemon has been
/// spawned.  Runs **without the lock** so a concurrent toggle (stop)
/// is not blocked by the network round-trip.
///
/// On failure the daemon is killed and the error message is pasted
/// into the target window where the transcription would normally
/// appear.
async fn toggle_validate(ctx: SpawnContext) -> Result<(), TalkError> {
    log::info!(
        "validating {} provider configuration",
        ctx.effective_provider
    );

    let validation_result: Result<(), TalkError> = async {
        if ctx.realtime {
            let t = transcription::create_realtime_transcriber(
                &ctx.config,
                ctx.effective_provider,
                ctx.model.as_deref(),
            )?;
            t.validate().await
        } else {
            let t = transcription::create_batch_transcriber(
                &ctx.config,
                ctx.effective_provider,
                ctx.model.as_deref(),
                ctx.diarize,
            )?;
            t.validate().await
        }
    }
    .await;

    if let Err(ref e) = validation_result {
        log::warn!("provider validation failed, stopping daemon: {}", e);
        let _ = daemon::stop_if_owner(ctx.child_pid, &ctx.pid_file);

        // Show error in a temporary visualizer overlay so the user gets
        // visible feedback without injecting text into their window.
        let error_msg = format!("[talk-rs] {e}");
        match VisualizerHandle::new(false, false, true) {
            Ok(viz) => {
                viz.show(0);
                viz.set_text(&error_msg);
                tokio::time::sleep(std::time::Duration::from_millis(
                    VALIDATION_ERROR_DISPLAY_MS,
                ))
                .await;
            }
            Err(viz_err) => {
                log::warn!("could not show error overlay ({}): {}", viz_err, error_msg);
            }
        }
    }

    validation_result
}

/// Signal a running daemon to stop and remove the PID file.
///
/// Returns immediately — the daemon finishes its transcription and
/// paste in the background, so a new toggle-on can proceed without
/// waiting.
fn toggle_stop(pid: u32, pid_file: &std::path::Path) -> Result<(), TalkError> {
    daemon::trace(&format!(
        "[DBG] toggle_stop: sending SIGINT to daemon PID {}",
        pid
    ));
    daemon::signal_daemon(pid, pid_file)?;
    daemon::trace(&format!(
        "[DBG] toggle_stop: signal_daemon returned OK for PID {}",
        pid
    ));
    Ok(())
}
