//! talk-rs performance measurement suite.
//!
//! One test binary (`cargo test --test perf`) so the harness costs a
//! single link.  Each module measures one shortlisted improvement on a
//! realistic cut, asserts the invariants every experiment must keep,
//! and records its metrics with [`support::Metrics`] (machine-readable
//! when `TALK_RS_PERF_OUT` is set).  `bin/perf-report` drives it.
//!
//! Work counters need `--features perf-counters`; tests that read them
//! skip (pass) without it so a plain `cargo test` stays green.
//! Long, display, and local-model cuts are `#[ignore]`d and/or gated
//! by environment variables (see `perf/README.org`).

#![cfg(all(feature = "capture", feature = "ui"))]

mod support;

mod perf_dictate;
mod perf_models;
mod perf_record_ui;
mod perf_transcribe;
mod perf_transport;
