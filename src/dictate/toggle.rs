//! Toggle dispatch: start or stop a dictation daemon.
//!
//! Extracted from `dictate.rs` — handles the `--toggle` flag logic:
//! spawn a new daemon process or stop a running one.

use crate::daemon::{self, ToggleOutcome};
use crate::dictate::DictateOpts;
use crate::error::TalkError;
use crate::paste::get_active_window;

/// Build the CLI argument list for the daemon process.
///
/// This is a pure function to enable unit testing of argument forwarding.
fn build_daemon_args(opts: &DictateOpts, target_window: Option<String>) -> Vec<String> {
    let mut args = Vec::new();

    // Forward verbosity level (before subcommand)
    if opts.verbose > 0 {
        args.push(format!("-{}", "v".repeat(opts.verbose as usize)));
    }

    args.push("dictate".to_string());
    args.push("--daemon".to_string());

    if let Some(p) = opts.provider {
        args.push("--provider".to_string());
        args.push(p.to_string());
    }

    if let Some(ref m) = opts.model {
        args.push("--model".to_string());
        args.push(m.clone());
    }

    if opts.diarize {
        args.push("--diarize".to_string());
    }

    if opts.timestamp {
        args.push("--timestamp".to_string());
    }

    if opts.realtime {
        args.push("--realtime".to_string());
    }

    if opts.no_sounds {
        args.push("--no-sounds".to_string());
    }

    if opts.no_boop {
        args.push("--no-boop".to_string());
    }

    if opts.no_chunk_paste {
        args.push("--no-chunk-paste".to_string());
    }

    if opts.no_paste {
        args.push("--no-paste".to_string());
    }

    if opts.monitor {
        args.push("--monitor".to_string());
    }

    if opts.no_overlay {
        args.push("--no-overlay".to_string());
    }

    if opts.no_auto_pause {
        args.push("--no-auto-pause".to_string());
    }

    if let Some(mode) = opts.viz {
        args.push("--viz".to_string());
        args.push(mode.to_string());
    }

    if opts.mono {
        args.push("--mono".to_string());
    }

    if opts.upload_format != crate::transcription::UploadFormat::Wav {
        args.push("--upload-format".to_string());
        args.push(format!("{:?}", opts.upload_format).to_lowercase());
    }

    if opts.no_bt_auto_switch {
        args.push("--no-bt-auto-switch".to_string());
    }

    if let Some(ref path) = opts.save {
        args.push("--save".to_string());
        args.push(path.to_string_lossy().to_string());
    }

    if let Some(ref path) = opts.output_yaml {
        args.push("--output-yaml".to_string());
        args.push(path.to_string_lossy().to_string());
    }

    if let Some(ref path) = opts.input_audio_file {
        args.push("--input-audio-file".to_string());
        args.push(path.to_string_lossy().to_string());
    }

    if opts.retry_last {
        args.push("--retry-last".to_string());
    }

    if opts.pick {
        args.push("--pick".to_string());
    }

    if opts.replace_last_paste {
        args.push("--replace-last-paste".to_string());
    }

    if let Some(ref wid) = target_window {
        args.push("--target-window".to_string());
        args.push(wid.clone());
    }

    args
}

/// Toggle dispatch: start a new daemon or stop a running one.
pub async fn toggle_dispatch(opts: &DictateOpts) -> Result<(), TalkError> {
    let slot = daemon::dictate_slot()?;
    let outcome = daemon::toggle_current_executable(&slot, || async {
        let target_window = get_active_window().await;
        Ok(build_daemon_args(opts, target_window))
    })
    .await?;
    match outcome {
        ToggleOutcome::Started { pid, log_path } => log::info!(
            "dictation started (PID {}, logs: {})",
            pid,
            log_path.display()
        ),
        ToggleOutcome::Signalled { pid } => slot.trace(&format!(
            "[DBG] dictate toggle sent SIGINT to daemon PID {pid}"
        )),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_opts() -> DictateOpts {
        DictateOpts {
            save: None,
            output_yaml: None,
            input_audio_file: None,
            retry_last: false,
            pick: false,
            replace_last_paste: false,
            provider: None,
            model: None,
            diarize: false,
            timestamp: false,
            realtime: false,
            toggle: false,
            no_sounds: false,
            no_boop: false,
            no_chunk_paste: false,
            no_paste: false,
            monitor: false,
            no_overlay: false,
            no_auto_pause: false,
            viz: None,
            mono: false,
            upload_format: crate::transcription::UploadFormat::Wav,
            no_bt_auto_switch: false,
            daemon: false,
            target_window: None,
            verbose: 0,
        }
    }

    #[test]
    fn test_build_daemon_args_includes_timestamp() {
        let mut opts = test_opts();
        opts.timestamp = true;
        let args = build_daemon_args(&opts, None);
        assert!(
            args.contains(&"--timestamp".to_string()),
            "--timestamp should be forwarded to daemon"
        );
    }

    #[test]
    fn test_build_daemon_args_includes_no_paste() {
        let mut opts = test_opts();
        opts.no_paste = true;
        let args = build_daemon_args(&opts, None);
        assert!(
            args.contains(&"--no-paste".to_string()),
            "--no-paste should be forwarded to daemon"
        );
    }

    #[test]
    fn test_build_daemon_args_includes_pick() {
        let mut opts = test_opts();
        opts.pick = true;
        let args = build_daemon_args(&opts, None);
        assert!(
            args.contains(&"--pick".to_string()),
            "--pick should be forwarded to daemon"
        );
    }

    #[test]
    fn test_build_daemon_args_includes_retry_last() {
        let mut opts = test_opts();
        opts.retry_last = true;
        let args = build_daemon_args(&opts, None);
        assert!(
            args.contains(&"--retry-last".to_string()),
            "--retry-last should be forwarded to daemon"
        );
    }

    #[test]
    fn test_build_daemon_args_includes_replace_last_paste() {
        let mut opts = test_opts();
        opts.replace_last_paste = true;
        let args = build_daemon_args(&opts, None);
        assert!(
            args.contains(&"--replace-last-paste".to_string()),
            "--replace-last-paste should be forwarded to daemon"
        );
    }

    #[test]
    fn test_build_daemon_args_includes_input_audio_file() {
        let mut opts = test_opts();
        opts.input_audio_file = Some(PathBuf::from("/tmp/test.ogg"));
        let args = build_daemon_args(&opts, None);
        let idx = args
            .iter()
            .position(|a| a == "--input-audio-file")
            .expect("--input-audio-file should be present");
        assert_eq!(args[idx + 1], "/tmp/test.ogg");
    }

    #[test]
    fn test_build_daemon_args_includes_output_yaml() {
        let mut opts = test_opts();
        opts.output_yaml = Some(PathBuf::from("/tmp/out.yaml"));
        let args = build_daemon_args(&opts, None);
        let idx = args
            .iter()
            .position(|a| a == "--output-yaml")
            .expect("--output-yaml should be present");
        assert_eq!(args[idx + 1], "/tmp/out.yaml");
    }

    #[test]
    fn test_build_daemon_args_does_not_include_toggle() {
        let mut opts = test_opts();
        opts.toggle = true;
        opts.daemon = true;
        let args = build_daemon_args(&opts, None);
        assert!(
            !args.contains(&"--toggle".to_string()),
            "--toggle should NOT be forwarded (already handled)"
        );
        // Verify --daemon appears exactly once
        let daemon_count = args.iter().filter(|a| *a == "--daemon").count();
        assert_eq!(daemon_count, 1, "--daemon should appear exactly once");
    }

    #[test]
    fn test_build_daemon_args_includes_target_window() {
        let opts = test_opts();
        let args = build_daemon_args(&opts, Some("0x12345678".to_string()));
        let idx = args
            .iter()
            .position(|a| a == "--target-window")
            .expect("--target-window should be present");
        assert_eq!(args[idx + 1], "0x12345678");
    }

    #[test]
    fn test_build_daemon_args_omits_unset_flags() {
        let opts = test_opts();
        let args = build_daemon_args(&opts, None);
        assert!(!args.contains(&"--timestamp".to_string()));
        assert!(!args.contains(&"--no-paste".to_string()));
        assert!(!args.contains(&"--diarize".to_string()));
        assert!(!args.contains(&"--realtime".to_string()));
        assert!(!args.contains(&"--no-sounds".to_string()));
    }

    #[test]
    fn test_build_daemon_args_exact_vector() {
        let mut opts = test_opts();
        opts.verbose = 2;
        opts.provider = Some(crate::config::Provider::OpenAI);
        opts.model = Some("gpt-test".to_string());
        opts.diarize = true;
        opts.timestamp = true;
        opts.realtime = true;
        opts.no_sounds = true;
        opts.no_boop = true;
        opts.no_chunk_paste = true;
        opts.no_paste = true;
        opts.monitor = true;
        opts.no_overlay = true;
        opts.no_auto_pause = true;
        opts.viz = Some(crate::config::VizMode::Waterfall);
        opts.mono = true;
        opts.upload_format = crate::transcription::UploadFormat::Ogg;
        opts.no_bt_auto_switch = true;
        opts.save = Some(PathBuf::from("/tmp/save.ogg"));
        opts.output_yaml = Some(PathBuf::from("/tmp/output.yaml"));
        opts.input_audio_file = Some(PathBuf::from("/tmp/input.ogg"));
        opts.retry_last = true;
        opts.pick = true;
        opts.replace_last_paste = true;
        opts.toggle = true;
        opts.daemon = true;

        let args = build_daemon_args(&opts, Some("0x1234".to_string()));

        assert_eq!(
            args,
            vec![
                "-vv",
                "dictate",
                "--daemon",
                "--provider",
                "openai",
                "--model",
                "gpt-test",
                "--diarize",
                "--timestamp",
                "--realtime",
                "--no-sounds",
                "--no-boop",
                "--no-chunk-paste",
                "--no-paste",
                "--monitor",
                "--no-overlay",
                "--no-auto-pause",
                "--viz",
                "waterfall",
                "--mono",
                "--upload-format",
                "ogg",
                "--no-bt-auto-switch",
                "--save",
                "/tmp/save.ogg",
                "--output-yaml",
                "/tmp/output.yaml",
                "--input-audio-file",
                "/tmp/input.ogg",
                "--retry-last",
                "--pick",
                "--replace-last-paste",
                "--target-window",
                "0x1234",
            ]
        );
        assert_eq!(args.iter().filter(|arg| *arg == "--daemon").count(), 1);
        assert!(!args.iter().any(|arg| arg == "--toggle"));
    }
}
