mod record;

use crate::cli::def::Commands;

pub use record::record;

pub async fn dispatch(command: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Record { file } => {
            let args = file.map(|f| vec![f]).unwrap_or_default();
            record(args).await?;
        }
    }
    Ok(())
}
