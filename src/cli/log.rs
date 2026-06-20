//! Logging setup with verbosity-based level control.
//!
//! Follows the `-v` / `-vv` / `-vvv` convention:
//! - 0 (default): Warn
//! - 1 (`-v`):    Info
//! - 2 (`-vv`):   Debug
//! - 3+ (`-vvv`): Trace
//!
//! Output always goes to stderr (colored, timestamped).  When
//! `--log-file <PATH>` (or `TALK_RS_LOG_FILE`) is set, a second
//! plain-text copy is appended to the given file at Info level or
//! above.

use std::io;
use std::path::Path;

/// Initialize the global logger with the given verbosity level.
///
/// When `log_file` is `Some`, logs are additionally written to that
/// path (plain text, always at Info level or above).
/// Map a `-v` repeat count to the stderr log level.
///
/// `0` → Warn, `1` (`-v`) → Info, `2` (`-vv`) → Debug, `3+` (`-vvv`)
/// → Trace.
fn base_level_for(verbosity: u8) -> log::LevelFilter {
    match verbosity {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _3_or_more => log::LevelFilter::Trace,
    }
}

/// Level for the `--log-file` sink: at least Info (so the file stays
/// useful even at the default Warn-only verbosity), but as verbose as
/// `base_level` when that is more verbose than Info.  This is what lets
/// `-vvv` push Trace (and thus the paste diagnostics) into the file.
fn file_level_for(base_level: log::LevelFilter) -> log::LevelFilter {
    base_level.max(log::LevelFilter::Info)
}

pub fn setup(verbosity: u8, log_file: Option<&str>) -> Result<(), String> {
    use colored::Colorize;

    let base_level = base_level_for(verbosity);

    // ── stderr (colored) ────────────────────────────────────────
    //
    // Per-sink level lives on this child dispatch (NOT on the root):
    // in `fern`, a parent `Dispatch::level()` is a hard gate that
    // children cannot exceed in verbosity, so the root must stay at
    // its pass-all default and each sink carries its own level.
    let stderr_config = fern::Dispatch::new()
        .level(base_level)
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
                let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
                out.finish(format_args!("{} {} {}: {}", now, level, target, message));
            },
        )
        .chain(io::stderr());

    // Root dispatch carries NO level (defaults to Trace = pass-all);
    // filtering happens per child sink above and below.
    let mut base_config = fern::Dispatch::new().chain(stderr_config);

    // ── log file (plain) ────────────────────────────────────────
    //
    // The file captures at least Info (so it stays useful even at
    // the default Warn-only verbosity), but follows the requested
    // verbosity when it is MORE verbose than Info.  In particular
    // `-vvv` lets Trace reach the file — required for the paste
    // diagnostics, which are emitted at Trace.
    if let Some(path) = log_file {
        let file_level = file_level_for(base_level);
        let file = open_log_file(Path::new(path))?;
        let file_config = fern::Dispatch::new()
            .level(file_level)
            .format(
                |out: fern::FormatCallback, message: &std::fmt::Arguments, record: &log::Record| {
                    let level = match record.level() {
                        log::Level::Error => "E",
                        log::Level::Warn => "W",
                        log::Level::Info => "I",
                        log::Level::Debug => "D",
                        log::Level::Trace => "T",
                    };
                    let target = record.target().replace("::", ".");
                    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
                    out.finish(format_args!("{} {} {}: {}", now, level, target, message));
                },
            )
            .chain(file);
        base_config = base_config.chain(file_config);
    }

    base_config.apply().map_err(|e| e.to_string())
}

/// Maximum log file size before truncation (2 MiB).
const MAX_LOG_SIZE: u64 = 2 * 1024 * 1024;

/// Open (or create) the log file for appending.  Truncates if the
/// file exceeds [`MAX_LOG_SIZE`] so logs don't grow unbounded.
fn open_log_file(path: &Path) -> Result<std::fs::File, String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create log directory: {}", e))?;
        }
    }

    // Truncate if over threshold.
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > MAX_LOG_SIZE {
            let _ = std::fs::remove_file(path);
        }
    }

    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("cannot open log file {}: {}", path.display(), e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::LevelFilter;

    #[test]
    fn test_base_level_for_verbosity() {
        assert_eq!(base_level_for(0), LevelFilter::Warn);
        assert_eq!(base_level_for(1), LevelFilter::Info);
        assert_eq!(base_level_for(2), LevelFilter::Debug);
        assert_eq!(base_level_for(3), LevelFilter::Trace);
        // Anything above 3 stays at the most verbose level.
        assert_eq!(base_level_for(4), LevelFilter::Trace);
        assert_eq!(base_level_for(255), LevelFilter::Trace);
    }

    #[test]
    fn test_file_level_floors_at_info() {
        // At the default Warn-only verbosity the file still captures
        // Info so it stays useful.
        assert_eq!(file_level_for(LevelFilter::Warn), LevelFilter::Info);
        assert_eq!(file_level_for(LevelFilter::Info), LevelFilter::Info);
    }

    #[test]
    fn test_file_level_follows_more_verbose_request() {
        // `-vv` / `-vvv` push the file sink past Info so Debug / Trace
        // (and thus the paste diagnostics) reach the file.
        assert_eq!(file_level_for(LevelFilter::Debug), LevelFilter::Debug);
        assert_eq!(file_level_for(LevelFilter::Trace), LevelFilter::Trace);
    }

    #[test]
    fn test_vvv_routes_trace_to_file() {
        // End-to-end of the level math: `-vvv` means the file sink is
        // at Trace, which is what makes `-vvv` capture paste diagnostics
        // in `--log-file`.
        let base = base_level_for(3);
        assert_eq!(file_level_for(base), LevelFilter::Trace);
    }
}
