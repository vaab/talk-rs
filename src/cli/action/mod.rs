mod record;
mod record_ui;
mod transcribe;

use crate::cli::def::Commands;
use std::path::PathBuf;

pub use crate::dictate::{dictate, DictateOpts};
pub use record::record;
pub use record_ui::record_ui;
pub use transcribe::transcribe;

pub async fn dispatch(command: Commands, verbose: u8) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Record { file, monitor, ui } => {
            if ui {
                record_ui().await?;
            } else {
                let args = file.map(|f| vec![f]).unwrap_or_default();
                record(args, monitor).await?;
            }
        }
        Commands::Transcribe {
            input,
            output,
            provider,
            model,
            diarize,
        } => {
            let mut args = vec![input];
            if let Some(output_file) = output {
                args.push(output_file);
            }
            transcribe(args, provider, model, diarize).await?;
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
            realtime,
            toggle,
            no_sounds,
            no_boop,
            no_chunk_paste,
            monitor,
            no_overlay,
            amplitude,
            spectrum,
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
                realtime,
                toggle,
                no_sounds,
                no_boop,
                no_chunk_paste,
                monitor,
                no_overlay,
                amplitude,
                spectrum,
                daemon,
                target_window,
                verbose,
            })
            .await?;
        }
    }
    Ok(())
}
