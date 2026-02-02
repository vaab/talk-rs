mod action;
pub mod def;

use clap::Parser;
use def::Cli;

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let _cli = Cli::parse();

    action::dispatch().await
}
