//! Shared performance-harness pieces.
//!
//! - [`fixtures`]: speech-like WAV/OGG and recordings-library factory.
//! - [`runner`]: isolated `talk-rs` runs, `timing:` / `perf-counter:`
//!   parsing, child CPU / peak RSS.
//! - [`provider`]: wiremock provider behind a counting TCP proxy.
//! - [`display`]: isolated X display (Xvfb or headless Weston+Xwayland).
//! - [`Metrics`]: machine-readable result files for `bin/perf-report`.

#![allow(dead_code)] // each perf test uses a different subset of the harness

pub mod audio;
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

/// Gate a local-model cut: `$name` must name a directory that the
/// PRODUCTION presence check accepts.  An unset variable skips; a set
/// but incomplete directory FAILS (talk-rs would take its consent /
/// download path there, which a measurement must never trigger).
pub fn model_gate(name: &str, kind: talk_rs::perf_counters::LocalModel) -> Option<ModelDir> {
    let dir = std::path::PathBuf::from(gate(name)?);
    assert!(
        talk_rs::perf_counters::model_present(kind, &dir),
        "{name}={} is not a complete {kind:?} model (production presence check failed)",
        dir.display()
    );
    let before = tree_state(&dir);
    Some(ModelDir { dir, before })
}

/// A validated local-model directory; dropping it asserts the run left
/// the tree unchanged (no download, no derived-file write).
pub struct ModelDir {
    pub dir: std::path::PathBuf,
    before: Vec<(String, u64, std::time::SystemTime)>,
}

impl ModelDir {
    pub fn path(&self) -> &str {
        self.dir.to_str().expect("utf-8 model dir")
    }

    pub fn assert_unchanged(&self) {
        assert_eq!(
            tree_state(&self.dir),
            self.before,
            "the model directory was modified during the measurement"
        );
    }
}

/// (relative path, size, mtime) of every file under `root`, sorted.
fn tree_state(root: &std::path::Path) -> Vec<(String, u64, std::time::SystemTime)> {
    fn walk(
        root: &std::path::Path,
        dir: &std::path::Path,
        out: &mut Vec<(String, u64, std::time::SystemTime)>,
    ) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                walk(root, &path, out);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string();
                out.push((
                    rel,
                    meta.len(),
                    meta.modified().unwrap_or(std::time::UNIX_EPOCH),
                ));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    // Sibling lock files written by the model fetcher count as changes.
    if let (Some(parent), Some(name)) = (root.parent(), root.file_name()) {
        let lock = parent.join(format!("{}.lock", name.to_string_lossy()));
        if let Ok(meta) = std::fs::metadata(&lock) {
            out.push((
                "../<lock>".into(),
                meta.len(),
                meta.modified().unwrap_or(std::time::UNIX_EPOCH),
            ));
        }
    }
    out.sort();
    out
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
        let mut body = format!(
            "item: {}\ncut: {}\nstatus: ran\nmetrics:\n",
            self.item, self.cut
        );
        for (name, value) in &self.values {
            assert!(
                value.is_finite(),
                "{}/{}: metric {name} is {value}",
                self.item,
                self.cut
            );
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

/// Number of repetitions for timing metrics (`TALK_RS_PERF_REPEAT`,
/// default 5).  Work counters are deterministic and taken from the
/// first run.
pub fn repeats() -> usize {
    std::env::var("TALK_RS_PERF_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n >= 1)
        .unwrap_or(5)
}

/// Nearest-rank percentile (`p` in 0..=100) of a sample.
pub fn percentile(values: &[f64], p: f64) -> f64 {
    assert!(!values.is_empty(), "percentile of an empty sample");
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

impl Metrics {
    /// Record a repeated timing sample as `<name>` (median),
    /// `<name>-p90`, `<name>-spread` (max − min) and `<name>-n`.  The
    /// comparator scores `<name>`, so every timing target is a median.
    pub fn timing(&mut self, name: &str, samples: &[f64]) -> &mut Self {
        let (lo, hi) = samples
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        self.set(name, percentile(samples, 50.0))
            .set(&format!("{name}_p90"), percentile(samples, 90.0))
            .set(&format!("{name}_spread"), hi - lo)
            .set(&format!("{name}_n"), samples.len() as f64)
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
