# talk-rs Agent Notes

## Scope

- Rust CLI scaffold for talk-rs.
- Keep modules separated between CLI and core logic.

## Commands

- Format: `cargo fmt`
- Lint: `cargo clippy --all-targets`
- Build: `cargo build`
- Test: `cargo test`

## Conventions

- No `unwrap()` in project code.
- Prefer explicit error handling with `Result`.
- Keep `src/cli/` focused on argument parsing and dispatch.
