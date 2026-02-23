pub(crate) mod action;
pub mod def;
mod log;

use clap::Parser;
use def::Cli;

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    log::setup(cli.verbose)?;

    action::dispatch(cli.command, cli.verbose).await
}
