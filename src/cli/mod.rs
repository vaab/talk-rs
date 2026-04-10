mod action;
pub mod def;
mod log;

use clap::Parser;
use def::Cli;

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Propagate --log-file into the environment so child processes
    // (picker, daemon) inherit it via clap's `env` attribute.
    if let Some(ref path) = cli.log_file {
        std::env::set_var("TALK_RS_LOG_FILE", path);
    }

    log::setup(cli.verbose, cli.log_file.as_deref())?;

    action::dispatch(cli.command, cli.verbose).await
}
