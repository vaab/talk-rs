use crate::daemon::{self, ToggleOutcome};
use crate::error::TalkError;
use std::path::PathBuf;

pub struct RecordToggleOpts {
    pub file: Option<PathBuf>,
    pub monitor: bool,
    pub no_bt_auto_switch: bool,
    pub verbose: u8,
}

fn build_daemon_args(opts: &RecordToggleOpts) -> Vec<String> {
    let mut args = Vec::new();
    if opts.verbose > 0 {
        args.push(format!("-{}", "v".repeat(opts.verbose as usize)));
    }
    args.push("record".to_string());
    args.push("--daemon".to_string());
    if opts.monitor {
        args.push("--monitor".to_string());
    }
    if opts.no_bt_auto_switch {
        args.push("--no-bt-auto-switch".to_string());
    }
    if let Some(path) = &opts.file {
        args.push(path.to_string_lossy().into_owned());
    }
    args
}

pub async fn toggle_dispatch(opts: &RecordToggleOpts) -> Result<(), TalkError> {
    let slot = daemon::record_slot()?;
    let outcome =
        daemon::toggle_current_executable(&slot, || async { Ok(build_daemon_args(opts)) }).await?;
    match outcome {
        ToggleOutcome::Started { pid, log_path } => log::info!(
            "recording started (PID {}, logs: {})",
            pid,
            log_path.display()
        ),
        ToggleOutcome::Signalled { pid } => slot.trace(&format!(
            "[DBG] record toggle sent SIGINT to daemon PID {pid}"
        )),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn record_child_args_forward_only_record_options() {
        let opts = RecordToggleOpts {
            file: Some(PathBuf::from("/tmp/meeting.ogg")),
            monitor: true,
            no_bt_auto_switch: true,
            verbose: 2,
        };

        let args = build_daemon_args(&opts);

        assert_eq!(
            args,
            vec![
                "-vv",
                "record",
                "--daemon",
                "--monitor",
                "--no-bt-auto-switch",
                "/tmp/meeting.ogg",
            ]
        );
        assert_eq!(args.iter().filter(|arg| *arg == "--daemon").count(), 1);
        assert!(!args.iter().any(|arg| arg == "--toggle"));
        assert!(!args.iter().any(|arg| arg == "--ui"));
    }

    #[test]
    fn record_child_args_keep_optional_path_absent() {
        let opts = RecordToggleOpts {
            file: None,
            monitor: false,
            no_bt_auto_switch: false,
            verbose: 0,
        };

        assert_eq!(build_daemon_args(&opts), vec!["record", "--daemon"]);
    }
}
