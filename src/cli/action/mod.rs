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
