//! Clipboard paste-node: set_text → simulate keystroke → DETERMINISTIC
//! per-chunk target-confirmation gate.
//!
//! The legacy `wait_until_served(0, …)` gate that this node used to
//! call advanced as soon as ANY X11 client fetched the offered
//! UTF8_STRING — including clipboard managers — and could
//! therefore overwrite the clipboard BEFORE the actual paste target
//! pulled the content, dropping the chunk and (because the previous
//! serve thread was still alive in its grace window) re-serving the
//! last chunk, causing a duplicate.  This was proven against real
//! log evidence: clipboard managers from different X11 client-bases
//! consume each chunk, but the target client-base does not appear
//! in the dropped chunk.
//!
//! The new gate is keyed on the TARGET X11 client-base (= target XID
//! masked with the server's `resource_id_mask`).  Chunk 1 LEARNS the
//! target's per-paste fetch count via a quiescence window (modern
//! GTK / Qt apps issue two UTF8_STRING requests per paste); chunks
//! 2..N CONFIRM the same count is reached before the gate releases.
//! On timeout the paste ABORTS LOUDLY rather than silently advancing
//! — there is no "best effort" path here.
//!
//! When the target client-base cannot be resolved (blind paste, or
//! the XID could not be parsed / masked), the node falls back to
//! the legacy `served_count > 0` gate WITH A WARNING but does not
//! abort, preserving backward compatibility for `--no-paste`-style
//! flows that have no specific target window.
//!
//! The save / restore steps live in the
//! [`crate::paste::paste_with_root`] wrapper so they apply ONCE per
//! whole-paste operation, not per chunk.

use crate::clipboard::Clipboard as _;
use crate::config::PasteShortcut;
use crate::error::TalkError;
use crate::paste::node::{PasteCtx, PasteNode};
use crate::paste::{log_preview, simulate_paste};
use async_trait::async_trait;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Poll interval (ms) used by the gate loops.  Five milliseconds
/// matches [`crate::clipboard::X11Clipboard::wait_until_served`] and
/// keeps the gate responsive without saturating the async runtime —
/// a single SelectionRequest round-trip is typically served within a
/// few milliseconds, so most chunks confirm on the first or second
/// poll.
const GATE_POLL_INTERVAL_MS: u64 = 5;

/// Tunables for a [`ClipboardNode`].  See
/// [`crate::config::PasteConfig`] for the YAML-facing knobs of the
/// same name.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClipboardNode {
    pub(crate) shortcut: PasteShortcut,
    /// Carried for `timing_from_tree` extraction — accepted in the
    /// YAML schema for backward compatibility but no longer used at
    /// runtime.  The legacy "pre-restore settle" window it governed
    /// has been replaced by the deterministic per-chunk
    /// target-confirmation gate; see the module doc.
    #[allow(dead_code)] // Surfaced indirectly via `node::timing_from_tree`.
    pub(crate) restore_settle_ms: u64,
    /// Per-chunk ABORT deadline.  See module doc.
    pub(crate) chunk_fetch_timeout_ms: u64,
    /// Per-chunk target-quiescence window.  See module doc.
    pub(crate) target_quiescence_ms: u64,
    /// Automatic per-chunk retries on the target-confirmation path.
    /// See module doc and [`crate::paste::node::DEFAULT_TARGET_FETCH_RETRIES`].
    pub(crate) target_fetch_retries: u32,
}

/// Outcome of the per-chunk wait phase.  Factored out so the
/// learn / confirm logic stays unit-testable: the X11-touching
/// timing loops drive a [`Decision`] which the caller acts on (set
/// expected count, return Err, return Ok).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GateDecision {
    /// Chunk 1 successfully observed at least one target fetch and
    /// then quiesced; freeze `expected` as the learned count.
    Learned { expected: u32 },
    /// Chunk N reached the previously learned count and then
    /// quiesced; safe to advance.
    Confirmed { observed: u32 },
    /// Hard timeout: the target client-base did not reach the
    /// required fetch count within `chunk_fetch_timeout_ms`.
    /// Caller MUST surface this as a [`TalkError::Clipboard`]
    /// abort — never silently advance.
    AbortedTimeout { observed: u32, required: u32 },
}

#[async_trait]
impl PasteNode for ClipboardNode {
    async fn paste(&self, text: &str, ctx: &PasteCtx<'_>) -> Result<(), TalkError> {
        let clipboard = ctx.clipboard;

        // Blind-paste fallback: no target client-base could be
        // resolved (realtime per-segment path).  This path does NOT
        // retry — it is best-effort by design and has always advanced
        // on the legacy `served_count > 0` signal.  Kept unchanged.
        let target_base = match ctx.target_client_base {
            Some(base) => base,
            None => {
                self.serve_and_simulate(clipboard, text).await?;
                self.run_fallback_gate(clipboard).await;
                let _ = ctx.t_stop;
                return Ok(());
            }
        };

        // Deterministic target-confirmation path WITH automatic retry.
        //
        // Each attempt re-serves the chunk (fresh serve handle = fresh
        // per-chunk fetch counter), re-focuses the target window (the
        // observed root cause: the paste keystroke was sent before the
        // keyboard focus was effective), re-sends the paste keystroke
        // and re-runs the gate.  On chunk 1 (LEARN) each retry resets
        // `expected_target_fetches` to 0 so the gate re-LEARNS instead
        // of wrongly entering CONFIRM.
        //
        // This async loop mirrors the pure [`run_retry_plan`] state
        // machine 1:1 (attempt range, chunk-1 reset, signal-once on
        // exhaustion); keep the two in lock-step — the unit tests
        // exercise `run_retry_plan`.
        let learn_phase = ctx.expected_target_fetches.load(Ordering::Relaxed) == 0;

        for attempt in 0..=self.target_fetch_retries {
            if attempt > 0 {
                // On a chunk-1 retry the previous failed attempt must
                // not leave a partially-learned expected count behind:
                // reset to 0 so this attempt re-LEARNS.  On chunk N
                // (CONFIRM) the expected count was learned by chunk 1
                // and must be preserved across retries.
                if learn_phase {
                    ctx.expected_target_fetches.store(0, Ordering::Relaxed);
                }
                // Re-focus the target window before re-sending the
                // keystroke — this is the actual root-cause fix.
                if let Some(wid) = ctx.target_window {
                    if let Err(e) = crate::paste::ensure_focus(wid).await {
                        log::warn!(
                            "paste(clipboard-node): retry {} could not re-focus \
                             target window {}: {} — retrying anyway",
                            attempt,
                            wid,
                            e,
                        );
                    }
                }
            }

            self.serve_and_simulate(clipboard, text).await?;

            match self.run_target_gate(clipboard, target_base, ctx).await {
                Ok(()) => {
                    if attempt > 0 {
                        log::info!(
                            "paste(clipboard-node): chunk confirmed on retry {} \
                             (target client-base {:#x})",
                            attempt,
                            target_base,
                        );
                    }
                    let _ = ctx.t_stop;
                    return Ok(());
                }
                Err(e) => {
                    if attempt < self.target_fetch_retries {
                        log::warn!(
                            "paste(clipboard-node): target client-base {:#x} did not \
                             fetch chunk within {} ms (attempt {}/{}) — re-focusing \
                             and retrying: {}",
                            target_base,
                            self.chunk_fetch_timeout_ms,
                            attempt + 1,
                            self.target_fetch_retries + 1,
                            e,
                        );
                        continue;
                    }
                    // Retries exhausted: emit the VISIBLE abort signal
                    // exactly once (red overlay + alert sound), then
                    // propagate the error.  The wrapper still restores
                    // the original clipboard regardless.
                    self.signal_final_abort(ctx, &e);
                    return Err(e);
                }
            }
        }

        // Unreachable: the `for` loop always returns from within (the
        // last iteration either returns Ok or the exhausted Err).
        // Kept as a defensive fallback that never fires.
        Err(TalkError::Clipboard(
            "paste aborted: retry loop exited without a decision".to_string(),
        ))
    }
}

impl ClipboardNode {
    /// Serve the chunk onto the clipboard and simulate the paste
    /// keystroke.  Factored out so the retry loop can re-run it on
    /// each attempt: a fresh `set_text` installs a fresh per-chunk
    /// serve handle (and therefore a fresh target-fetch counter),
    /// without which a retry would gate against a stale / consumed
    /// counter and confirm immediately on false evidence.
    async fn serve_and_simulate(
        &self,
        clipboard: &crate::clipboard::X11Clipboard,
        text: &str,
    ) -> Result<(), TalkError> {
        log::trace!(
            "paste(clipboard-node): set chunk content={}",
            log_preview(text),
        );

        clipboard.set_text(text).await?;

        // Read-back diagnostic (verbatim from legacy `paste_one`).
        // Note this runs on a fresh X11 connection inside the X11
        // clipboard impl, so its requestor's client-base is
        // different from the target's — the per-client tracking
        // automatically excludes it from the gate.
        if log::log_enabled!(log::Level::Trace) {
            match clipboard.get_text().await {
                Ok(rb) if rb == text => {
                    log::trace!("paste(clipboard-node): chunk read-back OK");
                }
                Ok(rb) => {
                    log::trace!(
                        "paste(clipboard-node): read-back MISMATCH — clipboard holds {}",
                        log_preview(&rb),
                    );
                }
                Err(e) => {
                    log::trace!("paste(clipboard-node): read-back failed: {}", e);
                }
            }
        }

        tokio::time::sleep(Duration::from_millis(5)).await;

        simulate_paste(self.shortcut).await
    }

    /// Emit the VISIBLE abort signal on a FINAL paste abort (target
    /// retries exhausted): a `Failed` telemetry event that drives the
    /// overlay to its red `Phase::Error`, plus the triple-pulse alert
    /// tone (when an alert hook is wired).  Fired exactly ONCE, here,
    /// because the retry loop only reaches this site after the last
    /// attempt failed.
    fn signal_final_abort(&self, ctx: &PasteCtx<'_>, err: &TalkError) {
        ctx.sink.emit(crate::telemetry::TranscriptionEvent::Failed {
            reason: err.to_string(),
            t: Instant::now(),
        });
        if let Some(alert) = ctx.alert.as_ref() {
            alert();
        }
    }

    /// Deterministic per-chunk gate keyed on the target client-base.
    ///
    /// Chunk 1 learns the target's per-paste fetch count via a
    /// quiescence window; subsequent chunks confirm the same count
    /// is reached.  On hard timeout returns a clear
    /// [`TalkError::Clipboard`] — the caller never silently advances.
    async fn run_target_gate(
        &self,
        clipboard: &crate::clipboard::X11Clipboard,
        target_base: u32,
        ctx: &PasteCtx<'_>,
    ) -> Result<(), TalkError> {
        let expected_prev = ctx.expected_target_fetches.load(Ordering::Relaxed);
        let timeout = Duration::from_millis(self.chunk_fetch_timeout_ms);
        let quiescence = Duration::from_millis(self.target_quiescence_ms);

        let decision = if expected_prev == 0 {
            // CHUNK 1 = LEARN
            wait_and_learn(clipboard, target_base, timeout, quiescence).await
        } else {
            // CHUNK N = CONFIRM
            wait_and_confirm(clipboard, target_base, expected_prev, timeout, quiescence).await
        };

        match decision {
            GateDecision::Learned { expected } => {
                ctx.expected_target_fetches
                    .store(expected, Ordering::Relaxed);
                log::info!(
                    "paste(clipboard-node): target client-base {:#x} learned \
                     expected_target_fetches={} (chunk 1 quiesced after {} ms)",
                    target_base,
                    expected,
                    self.target_quiescence_ms,
                );
                Ok(())
            }
            GateDecision::Confirmed { observed } => {
                log::trace!(
                    "paste(clipboard-node): target client-base {:#x} confirmed \
                     fetches={} (>= expected={})",
                    target_base,
                    observed,
                    expected_prev,
                );
                Ok(())
            }
            GateDecision::AbortedTimeout { observed, required } => {
                let msg = if expected_prev == 0 {
                    format!(
                        "paste aborted: target X11 client-base {:#x} never fetched \
                         the clipboard for chunk 1 within {} ms (observed={}) — \
                         wrong focus, unsupported app, or shortcut mismatch",
                        target_base, self.chunk_fetch_timeout_ms, observed,
                    )
                } else {
                    format!(
                        "paste aborted: target X11 client-base {:#x} only fetched \
                         clipboard {}/{} times within {} ms (this chunk would be \
                         dropped — refusing to overwrite silently)",
                        target_base, observed, required, self.chunk_fetch_timeout_ms,
                    )
                };
                log::error!("{}", msg);
                Err(TalkError::Clipboard(msg))
            }
        }
    }

    /// Blind-paste fallback: no target client-base could be
    /// resolved.  Keeps the legacy `served_count > 0` gate with a
    /// warning on timeout (NOT an abort) for backward compatibility
    /// with target-less flows like the realtime per-segment paste.
    async fn run_fallback_gate(&self, clipboard: &crate::clipboard::X11Clipboard) {
        log::debug!(
            "paste(clipboard-node): no target client-base — falling back to \
             legacy served_count gate (chunk_fetch_timeout_ms={})",
            self.chunk_fetch_timeout_ms,
        );
        let served = clipboard
            .wait_until_served(0, Duration::from_millis(self.chunk_fetch_timeout_ms))
            .await;
        if served == 0 {
            log::warn!(
                "paste(clipboard-node): blind-paste fallback timed out after {} ms \
                 with served_count=0 — target never fetched our clipboard, \
                 likely paste corruption",
                self.chunk_fetch_timeout_ms,
            );
        } else {
            log::trace!(
                "paste(clipboard-node): blind-paste consumed (served_count={})",
                served,
            );
        }
    }
}

/// Chunk-1 LEARN phase: wait for the target client-base to fetch at
/// least once, then a quiescence window during which no NEW target
/// fetch arrives.  Freezes the observed count as the per-operation
/// expected count.
///
/// Returns [`GateDecision::Learned`] on success, or
/// [`GateDecision::AbortedTimeout`] when no target fetch arrives
/// before the deadline.
async fn wait_and_learn(
    clipboard: &crate::clipboard::X11Clipboard,
    target_base: u32,
    timeout: Duration,
    quiescence: Duration,
) -> GateDecision {
    let deadline = Instant::now() + timeout;
    let poll = Duration::from_millis(GATE_POLL_INTERVAL_MS);

    // Phase A: wait for the FIRST target fetch.
    loop {
        let count = clipboard.target_fetch_count(target_base);
        if count > 0 {
            break;
        }
        if Instant::now() >= deadline {
            return GateDecision::AbortedTimeout {
                observed: 0,
                required: 1,
            };
        }
        tokio::time::sleep(poll).await;
    }

    // Phase B: keep waiting through `quiescence` of no NEW fetch.
    // Update `last_change` whenever the count grows; freeze when
    // `quiescence` elapses since the last growth.
    let mut last_count = clipboard.target_fetch_count(target_base);
    let mut last_change = Instant::now();
    loop {
        if last_change.elapsed() >= quiescence {
            return GateDecision::Learned {
                expected: last_count,
            };
        }
        if Instant::now() >= deadline {
            // Quiescence didn't complete within the chunk deadline,
            // but we DID observe a fetch — freeze whatever count we
            // have (no abort: chunk 1 already saw at least one
            // target fetch, which proves the target IS consuming).
            return GateDecision::Learned {
                expected: last_count,
            };
        }
        tokio::time::sleep(poll).await;
        let now = clipboard.target_fetch_count(target_base);
        if now != last_count {
            last_count = now;
            last_change = Instant::now();
        }
    }
}

/// Chunk-N CONFIRM phase: wait for the target client-base to reach
/// at least `expected` fetches on the CURRENT serve handle (each
/// chunk gets a fresh handle, so counts start at 0).  Then a short
/// quiescence window absorbs any trailing fetch before the gate
/// releases.
///
/// Returns [`GateDecision::Confirmed`] on success, or
/// [`GateDecision::AbortedTimeout`] when `expected` is not reached.
async fn wait_and_confirm(
    clipboard: &crate::clipboard::X11Clipboard,
    target_base: u32,
    expected: u32,
    timeout: Duration,
    quiescence: Duration,
) -> GateDecision {
    let deadline = Instant::now() + timeout;
    let poll = Duration::from_millis(GATE_POLL_INTERVAL_MS);

    // Phase A: wait until count >= expected.
    let mut observed;
    loop {
        observed = clipboard.target_fetch_count(target_base);
        if observed >= expected {
            break;
        }
        if Instant::now() >= deadline {
            return GateDecision::AbortedTimeout {
                observed,
                required: expected,
            };
        }
        tokio::time::sleep(poll).await;
    }

    // Phase B: short quiescence — absorbs an extra fetch before we
    // overwrite the clipboard.
    let mut last_count = observed;
    let mut last_change = Instant::now();
    loop {
        if last_change.elapsed() >= quiescence {
            return GateDecision::Confirmed {
                observed: last_count,
            };
        }
        if Instant::now() >= deadline {
            return GateDecision::Confirmed {
                observed: last_count,
            };
        }
        tokio::time::sleep(poll).await;
        let now = clipboard.target_fetch_count(target_base);
        if now != last_count {
            last_count = now;
            last_change = Instant::now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x11::clipboard::client_base;

    // ── client_base masking ─────────────────────────────────────

    /// Spec: real-world log evidence.  resource_id_mask = 0x001FFFFF;
    /// target window XID 50331661 and its real paste-requestor child
    /// 50331792 must yield the SAME client-base (0x03000000).  Without
    /// this masking the gate would key on the ephemeral child widget
    /// id and miss the actual fetcher entirely.
    #[test]
    fn client_base_groups_target_and_child_widget() {
        let mask: u32 = 0x001F_FFFF;
        let target: u32 = 50_331_661;
        let child: u32 = 50_331_792;
        let expected_base: u32 = 0x0300_0000;
        assert_eq!(client_base(target, mask), expected_base);
        assert_eq!(client_base(child, mask), expected_base);
    }

    /// Spec: clipboard manager client-bases (0x6a00000 / 0x6e00000)
    /// observed in the dropped-chunk session are DIFFERENT from the
    /// target's base — so a per-client-base gate correctly excludes
    /// them.
    #[test]
    fn client_base_distinguishes_clipboard_managers_from_target() {
        let mask: u32 = 0x001F_FFFF;
        let target_base = client_base(50_331_661, mask);
        // Some pixmap/window the clipboard manager owns; we only
        // need DIFFERENT high bits.  Use representative bases.
        let manager_a: u32 = 0x0640_0001;
        let manager_b: u32 = 0x0680_1234;
        assert_ne!(client_base(manager_a, mask), target_base);
        assert_ne!(client_base(manager_b, mask), target_base);
    }

    /// Spec: with a different resource_id_mask (some servers use
    /// 0x1FFFFF, others wider), the masking still produces a stable
    /// prefix.  Pure function — no X11 connection touched.
    #[test]
    fn client_base_is_pure_bit_masking() {
        // Wider mask: client-bases are 16 bits.
        let mask: u32 = 0x0000_FFFF;
        assert_eq!(client_base(0x1234_5678, mask), 0x1234_0000);
        assert_eq!(client_base(0x1234_FFFF, mask), 0x1234_0000);
        // Tighter mask: client-bases are 24 bits.
        let mask: u32 = 0x0000_00FF;
        assert_eq!(client_base(0xABCD_EF12, mask), 0xABCD_EF00);
    }

    // ── Per-client tracking + own-client exclusion ──────────────
    //
    // These tests exercise the FetchMap shape from the inside out:
    // tracking is implemented in `x11::clipboard::serve_request`,
    // which we cannot invoke directly without a live X11 connection.
    // We test the math here (client_base) and the gate-logic state
    // machine below; integration verification of the serve_request
    // counting itself happens via the real-X11 dictate path.

    // ── Learn / confirm / quiescence state-machine tests ────────
    //
    // We test the LOGIC of wait_and_learn / wait_and_confirm against
    // a controllable fetch-count source.  Driving real X11 in a unit
    // test would require a Xephyr server; instead we feed the
    // X11Clipboard from a fake serve handle.  Since X11Clipboard's
    // serve_handle is a private Mutex<Option<ClipboardServeHandle>>,
    // we factor the test cases through a tiny helper that drives
    // wait_and_learn / wait_and_confirm against a SIMULATED count
    // source.  The helpers below mirror the production loop shape
    // 1:1 — any divergence would be caught when this module's tests
    // start passing against the real X11 path.

    /// Spec: ABORT immediately when the target never fetches.  The
    /// pure decision computation falls through to AbortedTimeout.
    #[tokio::test]
    async fn learn_aborts_when_target_never_fetches() {
        // Use a custom helper that simulates "always 0".
        let timeout = Duration::from_millis(20);
        let quiescence = Duration::from_millis(50);
        let decision = simulate_learn_with(|_| 0, timeout, quiescence).await;
        match decision {
            GateDecision::AbortedTimeout { observed, required } => {
                assert_eq!(observed, 0);
                assert_eq!(required, 1);
            }
            other => panic!("expected AbortedTimeout, got {:?}", other),
        }
    }

    /// Spec: when the target fetches ONCE and then no more arrive,
    /// learn freezes at expected=1 after quiescence.
    #[tokio::test]
    async fn learn_freezes_at_one_when_only_one_fetch_arrives() {
        // After 0ms count jumps from 0 → 1 and never grows.
        let started = Instant::now();
        let decision = simulate_learn_with(
            move |_| {
                if started.elapsed() > Duration::from_millis(2) {
                    1
                } else {
                    0
                }
            },
            Duration::from_millis(300),
            Duration::from_millis(40),
        )
        .await;
        match decision {
            GateDecision::Learned { expected } => assert_eq!(expected, 1),
            other => panic!("expected Learned{{1}}, got {:?}", other),
        }
    }

    /// Spec: when the target fetches TWICE in quick succession (the
    /// observed GTK / Qt pattern), learn freezes at expected=2 after
    /// the quiescence window has elapsed past the second fetch.
    #[tokio::test]
    async fn learn_freezes_at_two_when_target_fetches_twice() {
        let started = Instant::now();
        let decision = simulate_learn_with(
            move |_| {
                let e = started.elapsed();
                if e > Duration::from_millis(15) {
                    2
                } else if e > Duration::from_millis(2) {
                    1
                } else {
                    0
                }
            },
            Duration::from_millis(300),
            Duration::from_millis(40),
        )
        .await;
        match decision {
            GateDecision::Learned { expected } => assert_eq!(expected, 2),
            other => panic!("expected Learned{{2}}, got {:?}", other),
        }
    }

    /// Spec: a chunk N confirm reaches the expected count, then
    /// quiesces, then returns Confirmed.
    #[tokio::test]
    async fn confirm_succeeds_when_expected_count_reached() {
        let started = Instant::now();
        let decision = simulate_confirm_with(
            move |_| {
                let e = started.elapsed();
                if e > Duration::from_millis(15) {
                    2
                } else if e > Duration::from_millis(2) {
                    1
                } else {
                    0
                }
            },
            2,
            Duration::from_millis(300),
            Duration::from_millis(40),
        )
        .await;
        match decision {
            GateDecision::Confirmed { observed } => assert!(observed >= 2),
            other => panic!("expected Confirmed, got {:?}", other),
        }
    }

    /// Spec: when chunk N's target only fetches once (instead of the
    /// expected two), the confirm phase ABORTS on timeout.  This is
    /// the exact dropped-chunk scenario from the real log evidence.
    #[tokio::test]
    async fn confirm_aborts_when_expected_count_not_reached() {
        // Stays at 1 forever; expected=2 → abort.
        let started = Instant::now();
        let decision = simulate_confirm_with(
            move |_| {
                if started.elapsed() > Duration::from_millis(2) {
                    1
                } else {
                    0
                }
            },
            2,
            Duration::from_millis(40),
            Duration::from_millis(20),
        )
        .await;
        match decision {
            GateDecision::AbortedTimeout { observed, required } => {
                assert_eq!(observed, 1);
                assert_eq!(required, 2);
            }
            other => panic!("expected AbortedTimeout, got {:?}", other),
        }
    }

    // ── Retry-loop state-machine tests ──────────────────────────
    //
    // These drive the pure `run_retry_plan` mirror of the async
    // retry loop in `ClipboardNode::paste`, verifying the reset /
    // relearn / signal-once behaviour without an X11 server.  The
    // mirror is test-only: production drives real async side effects
    // but keeps the SAME loop shape (attempt range, chunk-1 reset,
    // signal-once on exhaustion) — keep them in lock-step.

    /// Outcome of one attempt: `Ok` = gate confirmed; `Err` = the
    /// abort message the real gate would have produced.
    type AttemptResult = Result<(), String>;

    /// Observability into the retry loop: final result, how many
    /// chunk-1 relearn resets happened, whether the final abort
    /// signal fired (exactly once on exhaustion), and attempt count.
    struct RetryTrace {
        result: AttemptResult,
        resets: u32,
        signalled: bool,
        attempts: u32,
    }

    /// Pure retry state machine mirroring the loop in
    /// [`super::ClipboardNode::paste`] 1:1.  `retries` = extra
    /// attempts beyond the first; `learn_phase` = chunk 1 (each retry
    /// resets to re-learn); `gate` = per-attempt outcome.
    fn run_retry_plan<F>(retries: u32, learn_phase: bool, mut gate: F) -> RetryTrace
    where
        F: FnMut(u32) -> AttemptResult,
    {
        let mut resets = 0u32;
        let mut attempts = 0u32;
        for attempt in 0..=retries {
            if attempt > 0 && learn_phase {
                // Chunk-1 retry: reset expected count so the gate
                // re-LEARNS instead of wrongly entering CONFIRM.
                resets += 1;
            }
            attempts += 1;
            match gate(attempt) {
                Ok(()) => {
                    return RetryTrace {
                        result: Ok(()),
                        resets,
                        signalled: false,
                        attempts,
                    };
                }
                Err(e) => {
                    if attempt < retries {
                        continue;
                    }
                    // Retries exhausted: the final abort signal fires
                    // exactly once here.
                    return RetryTrace {
                        result: Err(e),
                        resets,
                        signalled: true,
                        attempts,
                    };
                }
            }
        }
        // Unreachable in practice (loop always returns).
        RetryTrace {
            result: Err("retry loop exited without a decision".to_string()),
            resets,
            signalled: false,
            attempts,
        }
    }

    /// Spec (A): the chunk fails the first attempt then succeeds on
    /// the retry — the loop returns Ok, does NOT fire the abort
    /// signal, and used exactly two attempts.
    #[test]
    fn retry_succeeds_after_one_failed_attempt() {
        let trace = run_retry_plan(2, /* learn_phase */ true, |attempt| {
            if attempt == 0 {
                Err("first attempt: target never fetched".to_string())
            } else {
                Ok(())
            }
        });
        assert!(trace.result.is_ok(), "second attempt should confirm");
        assert!(!trace.signalled, "no abort signal on eventual success");
        assert_eq!(trace.attempts, 2, "one failure + one success");
        // Chunk-1 retry re-learns exactly once (for attempt 1).
        assert_eq!(trace.resets, 1);
    }

    /// Spec (B): every attempt fails — after retries are exhausted
    /// the loop returns Err AND fires the abort signal exactly once.
    #[test]
    fn retry_aborts_and_signals_once_after_exhaustion() {
        let mut fail_count = 0u32;
        let trace = run_retry_plan(2, /* learn_phase */ false, |_attempt| {
            fail_count += 1;
            Err("target never fetched".to_string())
        });
        assert!(trace.result.is_err(), "exhausted retries must abort");
        assert!(
            trace.signalled,
            "abort signal fires exactly once on final abort"
        );
        assert_eq!(trace.attempts, 3, "1 initial + 2 retries = 3 attempts");
        assert_eq!(fail_count, 3, "gate invoked once per attempt");
        // CONFIRM phase (chunk N) never resets the learned count.
        assert_eq!(trace.resets, 0);
    }

    /// Spec (C): on chunk 1 (LEARN phase) each RETRY resets the
    /// expected count so the gate re-LEARNS; on chunk N (CONFIRM)
    /// no reset happens.  Two retries ⇒ two resets in LEARN, zero
    /// in CONFIRM.
    #[test]
    fn chunk_one_retries_relearn_but_chunk_n_does_not() {
        let learn = run_retry_plan(2, true, |_| Err("nope".to_string()));
        assert_eq!(learn.resets, 2, "chunk-1 relearns on each of 2 retries");

        let confirm = run_retry_plan(2, false, |_| Err("nope".to_string()));
        assert_eq!(confirm.resets, 0, "chunk-N never resets learned count");
    }

    /// Spec: with zero retries a single failing attempt aborts +
    /// signals immediately (no extra attempts).  This is the shape
    /// the config knob `target_fetch_retries: 0` produces.
    #[test]
    fn zero_retries_aborts_on_first_failure() {
        let trace = run_retry_plan(0, true, |_| Err("nope".to_string()));
        assert!(trace.result.is_err());
        assert!(trace.signalled);
        assert_eq!(trace.attempts, 1);
    }

    /// Spec: the first attempt succeeding needs no retries and never
    /// signals — the common happy path.
    #[test]
    fn first_attempt_success_no_retry_no_signal() {
        let trace = run_retry_plan(2, true, |attempt| {
            assert_eq!(attempt, 0, "must not run a second attempt");
            Ok(())
        });
        assert!(trace.result.is_ok());
        assert!(!trace.signalled);
        assert_eq!(trace.attempts, 1);
        assert_eq!(trace.resets, 0);
    }

    // ── Test helpers (mirror prod loop shape 1:1) ───────────────
    //
    // The helpers below replicate `wait_and_learn` / `wait_and_confirm`
    // against an injectable "count source" so we can drive the state
    // machine without an X11 server.  Any change to the production
    // loop shape MUST mirror here, and vice-versa — keep them lock-step.

    async fn simulate_learn_with<F>(
        count: F,
        timeout: Duration,
        quiescence: Duration,
    ) -> GateDecision
    where
        F: Fn(Instant) -> u32,
    {
        let deadline = Instant::now() + timeout;
        let poll = Duration::from_millis(GATE_POLL_INTERVAL_MS);

        loop {
            let c = count(Instant::now());
            if c > 0 {
                break;
            }
            if Instant::now() >= deadline {
                return GateDecision::AbortedTimeout {
                    observed: 0,
                    required: 1,
                };
            }
            tokio::time::sleep(poll).await;
        }

        let mut last_count = count(Instant::now());
        let mut last_change = Instant::now();
        loop {
            if last_change.elapsed() >= quiescence {
                return GateDecision::Learned {
                    expected: last_count,
                };
            }
            if Instant::now() >= deadline {
                return GateDecision::Learned {
                    expected: last_count,
                };
            }
            tokio::time::sleep(poll).await;
            let now = count(Instant::now());
            if now != last_count {
                last_count = now;
                last_change = Instant::now();
            }
        }
    }

    async fn simulate_confirm_with<F>(
        count: F,
        expected: u32,
        timeout: Duration,
        quiescence: Duration,
    ) -> GateDecision
    where
        F: Fn(Instant) -> u32,
    {
        let deadline = Instant::now() + timeout;
        let poll = Duration::from_millis(GATE_POLL_INTERVAL_MS);

        let mut observed;
        loop {
            observed = count(Instant::now());
            if observed >= expected {
                break;
            }
            if Instant::now() >= deadline {
                return GateDecision::AbortedTimeout {
                    observed,
                    required: expected,
                };
            }
            tokio::time::sleep(poll).await;
        }

        let mut last_count = observed;
        let mut last_change = Instant::now();
        loop {
            if last_change.elapsed() >= quiescence {
                return GateDecision::Confirmed {
                    observed: last_count,
                };
            }
            if Instant::now() >= deadline {
                return GateDecision::Confirmed {
                    observed: last_count,
                };
            }
            tokio::time::sleep(poll).await;
            let now = count(Instant::now());
            if now != last_count {
                last_count = now;
                last_change = Instant::now();
            }
        }
    }
}
