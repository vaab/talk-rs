use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "talk-rs", version, about = "Talk CLI")]
pub struct Cli {}

#[derive(Debug)]
pub enum Commands {}
