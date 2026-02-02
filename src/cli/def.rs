use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "talk-rs", version, about = "Talk CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Record audio from the system and save to a file
    Record {
        /// Output file path (optional, defaults to memo-YYYY-MM-DD-HH-MM-SS.ogg)
        #[arg(value_name = "FILE")]
        file: Option<String>,
    },
}
