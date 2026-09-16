use crate::cli::def::Commands;
use std::path::PathBuf;

pub use crate::dictate::{dictate, DictateOpts};
pub use crate::record::ui::record_ui;
pub use crate::record::{record, record_daemon, RecordOpts};
pub use crate::speak::{speak, SpeakOpts};
pub use crate::transcribe::transcribe;

pub async fn dispatch(command: Commands, verbose: u8) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Record {
            file,
            monitor,
            ui,
            toggle,
            no_bt_auto_switch,
            daemon,
        } => {
            if ui {
                record_ui().await?;
            } else if toggle {
                crate::record::toggle::toggle_dispatch(&crate::record::toggle::RecordToggleOpts {
                    file: file.map(PathBuf::from),
                    monitor,
                    no_bt_auto_switch,
                    verbose,
                })
                .await?;
            } else {
                let args = file.map(|f| vec![f]).unwrap_or_default();
                let opts = RecordOpts {
                    args,
                    monitor,
                    no_bt_auto_switch,
                };
                if daemon {
                    record_daemon(opts).await?;
                } else {
                    record(opts).await?;
                }
            }
        }
        Commands::Transcribe {
            input,
            output,
            provider,
            model,
            diarize,
            timestamp,
        } => {
            let mut args = vec![input];
            if let Some(output_file) = output {
                args.push(output_file);
            }
            transcribe(args, provider, model, diarize, timestamp).await?;
        }
        Commands::Speak {
            text,
            file,
            provider,
            voice,
            lang,
            speed,
            output,
            force,
        } => {
            speak(SpeakOpts {
                text,
                file: file.map(PathBuf::from),
                provider,
                voice,
                lang,
                speed,
                output: output.map(PathBuf::from),
                force,
            })
            .await?;
        }
        Commands::Dictate {
            save,
            output_yaml,
            input_audio_file,
            retry_last,
            pick,
            replace_last_paste,
            provider,
            model,
            diarize,
            timestamp,
            realtime,
            toggle,
            no_sounds,
            no_boop,
            no_chunk_paste,
            no_paste,
            monitor,
            no_overlay,
            no_auto_pause,
            viz,
            mono,
            upload_format,
            no_bt_auto_switch,
            daemon,
            target_window,
        } => {
            dictate(DictateOpts {
                save: save.map(PathBuf::from),
                output_yaml: output_yaml.map(PathBuf::from),
                input_audio_file: input_audio_file.map(PathBuf::from),
                retry_last,
                pick,
                replace_last_paste,
                provider,
                model,
                diarize,
                timestamp,
                realtime,
                toggle,
                no_sounds,
                no_boop,
                no_chunk_paste,
                no_paste,
                monitor,
                no_overlay,
                no_auto_pause,
                viz,
                mono,
                upload_format,
                no_bt_auto_switch,
                daemon,
                target_window,
                verbose,
            })
            .await?;
        }
    }
    Ok(())
}
