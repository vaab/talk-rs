mod dictate;
mod record;
mod transcribe;

use crate::cli::def::Commands;
use std::path::PathBuf;

pub use dictate::{dictate, DictateOpts};
pub use record::record;
pub use transcribe::transcribe;

pub async fn dispatch(command: Commands, verbose: u8) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Record { file } => {
            let args = file.map(|f| vec![f]).unwrap_or_default();
            record(args).await?;
        }
        Commands::Transcribe {
            input,
            output,
            provider,
            model,
        } => {
            let mut args = vec![input];
            if let Some(output_file) = output {
                args.push(output_file);
            }
            transcribe(args, provider, model).await?;
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
            realtime,
            toggle,
            no_sounds,
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
                realtime,
                toggle,
                no_sounds,
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
