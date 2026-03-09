//! Toggle dispatch: start or stop a dictation daemon.
//!
//! Extracted from `dictate.rs` — handles the `--toggle` flag logic:
//! spawn a new daemon process or stop a running one.

use crate::config::{Provider, VizMode};
use crate::daemon::{self, DaemonStatus};
use crate::error::TalkError;
use crate::paste::get_active_window;
use std::os::unix::process::CommandExt as _;

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
    viz: Option<VizMode>,
    bw: bool,
    save: Option<&std::path::Path>,
    verbose: u8,
) -> Result<(), TalkError> {
    let pid_file = daemon::pid_path()?;
    let _lock = daemon::acquire_lock()?;

    let status = daemon::check_status(&pid_file)?;
    daemon::trace(&format!(
        "[DBG] toggle_dispatch: check_status = {:?}",
        status
    ));
    match status {
        DaemonStatus::NotRunning => {
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
                viz,
                bw,
                save,
                verbose,
            )
            .await?;
        }
        DaemonStatus::Running { pid } => {
            toggle_stop(pid, &pid_file)?;
        }
    }

    Ok(())
}

/// Spawn the daemon process and write the PID file.
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
    viz: Option<VizMode>,
    bw: bool,
    save: Option<&std::path::Path>,
    verbose: u8,
) -> Result<(), TalkError> {
    // Capture active window before spawning daemon
    let target_window = get_active_window().await;

    // Find our own executable
    let exe = std::env::current_exe()
        .map_err(|e| TalkError::Config(format!("failed to determine current executable: {}", e)))?;

    // Build daemon command: talk-rs [-v...] dictate --daemon [flags] [--target-window=WID]
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

    if let Some(mode) = viz {
        cmd.arg("--viz").arg(mode.to_string());
    }

    if bw {
        cmd.arg("--bw");
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

    // Spawn daemon — recording starts immediately.
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
