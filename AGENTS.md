# talk-rs Agent Guide

## Project Snapshot

- `talk-rs` is a production Rust CLI for Linux voice dictation.
- Main flow: record audio, transcribe with provider backends, paste into the focused app.
- Keep code modular: `src/cli/` for command parsing/dispatch, `src/core/` for runtime logic.

## Architecture Boundaries

- Keep `src/cli/` thin: argument parsing, command routing, and user-facing command wiring.
- Keep `src/core/` as the source of truth for business logic (audio, transcription, overlay, daemon, cache, picker).
- Do not leak CLI concerns into core modules.
- Prefer focused modules and explicit `Result`-based error propagation.

## Recording Readiness (durable user requirement)

- The "recording" cue (red `IndicatorKind::Recording` badge, start tone
  as "speak now") must only be given after capture readiness is
  PROVEN: the capture stream, explicitly targeted at the resolved
  microphone source (the Bluetooth headset's HFP source once switched),
  reached the Streaming state and delivered its first buffer.
- Never use a sleep, fixed delay, or timeout as evidence of readiness.
  Timeouts may exist only as error limits: when the proof is missing,
  fail loudly and never show the recording cue.
- No sample delivered by capture after its first buffer may be dropped
  or trimmed before the recording pipeline (the start tone being
  faintly captured is accepted).
- Do not reorder the startup sequence (neutral badge → headset source →
  proven capture → start tone emitted → red badge) without the user's
  explicit authorization.

## Development Commands

- **Prerequisite**: run `./autogen.sh` before any build, test, or lint command. It generates version metadata required by the build.
- Format: `cargo fmt`
- Lint: `cargo clippy --all-targets`
- Test: `cargo test`
- Build: `cargo build`
- Release build: `cargo build --release`

## Coding Rules

- No `unwrap()` or `expect()` in project code.
- No dead-code pragmas without a short justification comment.
- No trailing whitespace in tracked files.
- Keep behavior changes covered by tests (unit and/or integration as appropriate).
- Update docs/config examples when changing CLI flags, config schema, or user-visible behavior.

## Verification Checklist

Before finishing a change, run and validate:

- [ ] `cargo fmt`
- [ ] `cargo clippy --all-targets`
- [ ] `cargo test`
- [ ] `cargo build`
- [ ] `cargo build --release` before asking for user approval
- [ ] No new `unwrap()`/`expect()` in non-test code
- [ ] No unexplained dead-code pragmas
- [ ] No trailing whitespace (e.g. `grep -r "[[:space:]]$" src/ tests/`)

## Agent File Convention

- Keep this file as the project instruction source of truth.
- Keep `CLAUDE.md` as a symlink to `AGENTS.md`.
