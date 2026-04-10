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
pub fn setup(verbosity: u8, log_file: Option<&str>) -> Result<(), String> {
    use colored::Colorize;

    let base_level = match verbosity {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _3_or_more => log::LevelFilter::Trace,
    };

    // ── stderr (colored) ────────────────────────────────────────
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
                let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
                out.finish(format_args!("{} {} {}: {}", now, level, target, message));
            },
        )
        .chain(io::stderr());

    let mut base_config = fern::Dispatch::new().level(base_level).chain(stderr_config);

    // ── log file (plain, always ≥ Info) ─────────────────────────
    if let Some(path) = log_file {
        let file = open_log_file(Path::new(path))?;
        let file_config = fern::Dispatch::new()
            .level(log::LevelFilter::Info)
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
