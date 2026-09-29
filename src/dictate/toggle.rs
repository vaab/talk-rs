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

    if let Some(chain) = &opts.chain {
        args.extend(["--chain".to_string(), chain.clone()]);
    }
    if let Some(lang) = &opts.lang {
        args.extend(["--lang".to_string(), lang.clone()]);
    }

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
pub(super) mod tests {
    use super::*;
    use std::path::PathBuf;

    pub(in crate::dictate) fn test_opts() -> DictateOpts {
        DictateOpts {
            chain: None,
            lang: None,
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
    fn toggle_forwards_chain_and_language() {
        let mut opts = test_opts();
        opts.chain = Some("french".into());
        opts.lang = Some("fr".into());
        assert_eq!(
            build_daemon_args(&opts, None),
            ["dictate", "--daemon", "--chain", "french", "--lang", "fr"]
        );
    }

    #[test]
    fn each_forwarded_flag_has_exact_daemon_arguments() {
        type FlagCase = (&'static str, fn(&mut DictateOpts), &'static [&'static str]);
        let cases: &[FlagCase] = &[
            ("timestamp", |o| o.timestamp = true, &["--timestamp"]),
            ("no_paste", |o| o.no_paste = true, &["--no-paste"]),
            ("pick", |o| o.pick = true, &["--pick"]),
            ("retry_last", |o| o.retry_last = true, &["--retry-last"]),
            (
                "replace",
                |o| o.replace_last_paste = true,
                &["--replace-last-paste"],
            ),
            (
                "input",
                |o| o.input_audio_file = Some("/tmp/test.ogg".into()),
                &["--input-audio-file", "/tmp/test.ogg"],
            ),
            (
                "yaml",
                |o| o.output_yaml = Some("/tmp/out.yaml".into()),
                &["--output-yaml", "/tmp/out.yaml"],
            ),
            (
                "provider",
                |o| o.provider = Some(crate::config::Provider::OpenAI),
                &["--provider", "openai"],
            ),
            (
                "model",
                |o| o.model = Some("test".into()),
                &["--model", "test"],
            ),
            ("diarize", |o| o.diarize = true, &["--diarize"]),
            ("realtime", |o| o.realtime = true, &["--realtime"]),
            ("no_sounds", |o| o.no_sounds = true, &["--no-sounds"]),
            ("no_boop", |o| o.no_boop = true, &["--no-boop"]),
            (
                "no_chunk",
                |o| o.no_chunk_paste = true,
                &["--no-chunk-paste"],
            ),
            ("monitor", |o| o.monitor = true, &["--monitor"]),
            ("no_overlay", |o| o.no_overlay = true, &["--no-overlay"]),
            (
                "no_auto_pause",
                |o| o.no_auto_pause = true,
                &["--no-auto-pause"],
            ),
            (
                "viz",
                |o| o.viz = Some(crate::config::VizMode::Waterfall),
                &["--viz", "waterfall"],
            ),
            ("mono", |o| o.mono = true, &["--mono"]),
            (
                "upload",
                |o| o.upload_format = crate::transcription::UploadFormat::Ogg,
                &["--upload-format", "ogg"],
            ),
            (
                "bt",
                |o| o.no_bt_auto_switch = true,
                &["--no-bt-auto-switch"],
            ),
            (
                "save",
                |o| o.save = Some("/tmp/save.ogg".into()),
                &["--save", "/tmp/save.ogg"],
            ),
        ];
        for (name, change, extra) in cases {
            let mut opts = test_opts();
            change(&mut opts);
            let mut expected = vec!["dictate", "--daemon"];
            expected.extend_from_slice(extra);
            assert_eq!(build_daemon_args(&opts, None), expected, "{name}");
        }
        assert_eq!(
            build_daemon_args(&test_opts(), Some("0x1234".into())),
            ["dictate", "--daemon", "--target-window", "0x1234"]
        );
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
