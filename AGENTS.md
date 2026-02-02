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
- No dead code pragmas without explanatory comments.
- No trailing whitespace in any file.

## Review Checklist (Per Commit)

Before each commit, verify:
- [ ] `cargo fmt` - Code is formatted
- [ ] `cargo clippy --all-targets` - No warnings
- [ ] `cargo test` - All tests pass
- [ ] `cargo build` - Builds successfully
- [ ] No `unwrap()` or `expect()` in new code
- [ ] No dead code pragmas (or commented explanation if necessary)
- [ ] No trailing whitespace (check with `grep -r "[[:space:]]$" src/`)
- [ ] All technical terms in commit message wrapped in double-backticks
- [ ] Meta-goals reviewed (modularity, zero defaults, streaming, extensibility)
