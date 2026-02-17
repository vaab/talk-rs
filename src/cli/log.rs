//! Logging setup with verbosity-based level control.
//!
//! Follows the `-v` / `-vv` / `-vvv` convention:
//! - 0 (default): Warn
//! - 1 (`-v`):    Info
//! - 2 (`-vv`):   Debug
//! - 3+ (`-vvv`): Trace
//!
//! Output goes to stderr with colored single-letter level prefixes
//! and dotted target names (e.g. `I talk_rs.core.transcription: ...`).

use std::io;

/// Initialize the global logger with the given verbosity level.
pub fn setup(verbosity: u8) -> Result<(), String> {
    use colored::Colorize;

    let base_config = fern::Dispatch::new();

    let base_config = match verbosity {
        0 => base_config.level(log::LevelFilter::Warn),
        1 => base_config.level(log::LevelFilter::Info),
        2 => base_config.level(log::LevelFilter::Debug),
        _3_or_more => base_config.level(log::LevelFilter::Trace),
    };

    let stderr_config = fern::Dispatch::new()
        .format(
            move |out: fern::FormatCallback,
                  message: &std::fmt::Arguments,
                  record: &log::Record| {
                let level = match record.level() {
                    log::Level::Error => "E".bright_red(),
                    log::Level::Warn => "W".bright_yellow(),
                    log::Level::Info => "I".bright_green(),
                    log::Level::Debug => "D".blue(),
                    log::Level::Trace => "T".black(),
                };
                let target = record.target().replace("::", ".").yellow();

                out.finish(format_args!("{} {}: {}", level, target, message));
            },
        )
        .chain(io::stderr());

    base_config
        .chain(stderr_config)
        .apply()
        .map_err(|e| e.to_string())
}
