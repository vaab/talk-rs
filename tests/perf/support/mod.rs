//! Shared performance-harness pieces.
//!
//! - [`fixtures`]: speech-like WAV/OGG and recordings-library factory.
//! - [`runner`]: isolated `talk-rs` runs, `timing:` / `perf-counter:`
//!   parsing, child CPU / peak RSS.
//! - [`provider`]: wiremock provider behind a counting TCP proxy.
//! - [`display`]: isolated X display (Xvfb or headless Weston+Xwayland).
//! - [`Metrics`]: machine-readable result files for `bin/perf-report`.

#![allow(dead_code)] // each perf test uses a different subset of the harness

pub mod display;
pub mod fixtures;
pub mod provider;
pub mod runner;

use std::fmt::Write as _;

/// True when the binary was built with `--features perf-counters`.
pub fn counters_enabled() -> bool {
    cfg!(feature = "perf-counters")
}

/// Skip (return early) unless counters are compiled in.
#[macro_export]
macro_rules! require_counters {
    () => {
        if !$crate::support::counters_enabled() {
            eprintln!("SKIP: needs `--features perf-counters`");
            return;
        }
    };
}

/// Skip unless environment variable `$name` is set; returns its value.
pub fn gate(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => {
            eprintln!("SKIP: set {name} to run this measurement");
            None
        }
    }
}

/// One measurement: `item` (shortlist id) × `cut` (scenario).
///
/// Written to `$TALK_RS_PERF_OUT/<item>--<cut>.yaml` when that
/// variable is set, and always echoed to stderr.
pub struct Metrics {
    item: &'static str,
    cut: &'static str,
    values: Vec<(String, f64)>,
}

impl Metrics {
    pub fn new(item: &'static str, cut: &'static str) -> Self {
        Self {
            item,
            cut,
            values: Vec::new(),
        }
    }

    /// Record `name` (snake_case counter names are written dash-cased,
    /// the harness's YAML key convention).
    pub fn set(&mut self, name: &str, value: impl Into<f64>) -> &mut Self {
        self.values.push((name.replace('_', "-"), value.into()));
        self
    }

    /// Copy selected `perf-counter:` values from a run report.
    pub fn counters(&mut self, report: &runner::RunReport, names: &[&str]) -> &mut Self {
        for name in names {
            let value = report.counter(name);
            self.set(name, value as f64);
        }
        self
    }

    pub fn write(&self) {
        let mut body = format!("item: {}\ncut: {}\nmetrics:\n", self.item, self.cut);
        for (name, value) in &self.values {
            let _ = writeln!(body, "  {name}: {}", fmt_value(*value));
        }
        eprintln!("perf {}/{}:\n{body}", self.item, self.cut);
        if let Some(dir) = std::env::var_os("TALK_RS_PERF_OUT") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("perf output dir");
            std::fs::write(dir.join(format!("{}--{}.yaml", self.item, self.cut)), body)
                .expect("write perf metrics");
        }
    }
}

fn fmt_value(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v:.3}")
    }
}

/// Median of a sample (for repeated timing runs).
pub fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Config for a sandbox whose providers both point at `provider_url`,
/// with an optional chain `perf` made of `chain` entries.
pub fn config_yaml(output_dir: &std::path::Path, provider_url: &str, chain: &[&str]) -> String {
    let mut yaml = format!(
        "output_dir: {}\nproviders:\n  mistral:\n    api_key: fake\n    url: {provider_url}\n    model: voxtral-mini-2602\n  openai:\n    api_key: fake\n    url: {provider_url}\n    model: gpt-transcribe\nindicators:\n  boop_interval_ms: 0\n  visual_overlay: false\n",
        output_dir.display()
    );
    if !chain.is_empty() {
        yaml.push_str("transcription:\n  default_provider: mistral\n  chains:\n    perf:\n");
        for entry in chain {
            yaml.push_str(&format!("      - {entry}\n"));
        }
    }
    yaml
}

/// The fallback chain used by every "first provider fails" cut:
/// two busy entries, the third answers.  Outage memory is
/// per-provider and dictate records the live failure before walking
/// the rest, so the third entry must not reuse the first provider.
pub const CHAIN_3: [&str; 3] = [
    "mistral/voxtral-mini-2602",
    "openai/gpt-transcribe",
    "openai/gpt-4o-transcribe",
];
