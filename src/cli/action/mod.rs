mod dictate;
mod record;
mod transcribe;

use crate::cli::def::Commands;

pub use dictate::dictate;
pub use record::record;
pub use transcribe::transcribe;

pub async fn dispatch(command: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Record { file } => {
            let args = file.map(|f| vec![f]).unwrap_or_default();
            record(args).await?;
        }
        Commands::Transcribe { input, output } => {
            let mut args = vec![input];
            if let Some(output_file) = output {
                args.push(output_file);
            }
            transcribe(args).await?;
        }
        Commands::Dictate {
            file,
            batch,
            toggle,
            no_sounds,
            daemon,
            target_window,
        } => {
            let args = file.map(|f| vec![f]).unwrap_or_default();
            dictate(args, batch, toggle, no_sounds, daemon, target_window).await?;
        }
    }
    Ok(())
}
