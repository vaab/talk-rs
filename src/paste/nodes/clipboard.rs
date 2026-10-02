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
//! On timeout with no target fetch, the shortcut may be retried. If
//! any target fetch was served, retry could duplicate inserted text,
//! so the paste ABORTS LOUDLY rather than re-injecting or advancing.
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
    #[allow(dead_code)] // Surfaced indirectly via `paste_config::timing_from_tree`.
    pub(crate) restore_settle_ms: u64,
    /// Per-chunk ABORT deadline.  See module doc.
    pub(crate) chunk_fetch_timeout_ms: u64,
    /// Per-chunk target-quiescence window.  See module doc.
    pub(crate) target_quiescence_ms: u64,
    /// Automatic per-chunk retries on the target-confirmation path.
    /// See module doc and [`crate::paste_config::DEFAULT_TARGET_FETCH_RETRIES`].
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

async fn focus_then_inject<FocusFuture, Inject, InjectFuture>(
    focus: FocusFuture,
    inject: Inject,
) -> Result<(), TalkError>
where
    FocusFuture: std::future::Future<Output = Result<(), TalkError>>,
    Inject: FnOnce() -> InjectFuture,
    InjectFuture: std::future::Future<Output = Result<(), TalkError>>,
{
    focus.await?;
    inject().await
}

/// Owns the retry boundary shared by X11 delivery and scripted observers.
/// Once the target fetched even once, a second shortcut might duplicate text.
async fn run_target_attempts<F, Fut>(
    retries: u32,
    mut attempt_fn: F,
) -> Result<GateDecision, TalkError>
where
    F: FnMut(u32) -> Fut,
    Fut: std::future::Future<Output = Result<GateDecision, TalkError>>,
{
    for attempt in 0..=retries {
        let decision = attempt_fn(attempt).await?;
        match decision {
            GateDecision::Learned { .. } | GateDecision::Confirmed { .. } => return Ok(decision),
            GateDecision::AbortedTimeout { observed: 0, .. } if attempt < retries => continue,
            GateDecision::AbortedTimeout { .. } => return Ok(decision),
        }
    }
    Err(TalkError::Clipboard(
        "paste retry loop exhausted".to_string(),
    ))
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

        // Deterministic target-confirmation path with zero-fetch retry.
        //
        // Each zero-fetch retry re-serves the chunk (fresh serve handle =
        // fresh per-chunk fetch counter) and re-focuses before re-sending.
        // Once any target fetch occurred, no second shortcut is safe. On
        // chunk 1 (LEARN) each retry resets
        // `expected_target_fetches` to 0 so the gate re-LEARNS instead
        // of wrongly entering CONFIRM.
        //
        let learn_phase = ctx.expected_target_fetches.load(Ordering::Relaxed) == 0;
        let result = run_target_attempts(self.target_fetch_retries, |attempt| async move {
            if attempt > 0 {
                // On a chunk-1 retry the previous failed attempt must
                // not leave a partially-learned expected count behind:
                // reset to 0 so this attempt re-LEARNS.  On chunk N
                // (CONFIRM) the expected count was learned by chunk 1
                // and must be preserved across retries.
                if learn_phase {
                    ctx.expected_target_fetches.store(0, Ordering::Relaxed);
                }
            }

            let injection_result = if attempt > 0 {
                if let Some(wid) = ctx.target_window {
                    focus_then_inject(crate::paste::refocus(wid), || {
                        self.serve_and_simulate(clipboard, text)
                    })
                    .await
                } else {
                    self.serve_and_simulate(clipboard, text).await
                }
            } else {
                self.serve_and_simulate(clipboard, text).await
            };
            if let Err(error) = injection_result {
                log::error!(
                    "paste(clipboard-node): retry {} focus/shortcut attempt failed: {}",
                    attempt,
                    error,
                );
                return Err(error);
            }
            Ok(self.run_target_gate(clipboard, target_base, ctx).await)
        })
        .await;
        let result = match result {
            Ok(GateDecision::Learned { .. } | GateDecision::Confirmed { .. }) => Ok(()),
            Ok(GateDecision::AbortedTimeout { observed, required }) => {
                let reason = if observed == 0 {
                    "no target fetch after all shortcut attempts"
                } else {
                    "not retrying after a target fetch: the chunk may already be inserted"
                };
                Err(TalkError::Clipboard(format!(
                    "paste aborted: target X11 client-base {target_base:#x} fetched clipboard \
                     {observed}/{required} times within {} ms; {reason}",
                    self.chunk_fetch_timeout_ms
                )))
            }
            Err(error) => Err(error),
        };
        if let Err(ref error) = result {
            self.signal_final_abort(ctx, error);
        }
        let _ = ctx.t_stop;
        result
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

    /// Emit the VISIBLE abort signal on a final paste abort: a `Failed`
    /// telemetry event that drives the
    /// overlay to its red `Phase::Error`, plus the triple-pulse alert
    /// tone (when an alert hook is wired).  Fired exactly ONCE, here,
    /// because the caller reaches this site only on failure.
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
    /// [`GateDecision::AbortedTimeout`] — the caller never silently advances.
    async fn run_target_gate(
        &self,
        clipboard: &crate::clipboard::X11Clipboard,
        target_base: u32,
        ctx: &PasteCtx<'_>,
    ) -> GateDecision {
        let expected_prev = ctx.expected_target_fetches.load(Ordering::Relaxed);
        let timeout = Duration::from_millis(self.chunk_fetch_timeout_ms);
        let quiescence = Duration::from_millis(self.target_quiescence_ms);

        let decision = if expected_prev == 0 {
            // CHUNK 1 = LEARN
            wait_and_learn(
                || clipboard.target_fetch_count(target_base),
                timeout,
                quiescence,
            )
            .await
        } else {
            // CHUNK N = CONFIRM
            wait_and_confirm(
                || clipboard.target_fetch_count(target_base),
                expected_prev,
                timeout,
                quiescence,
            )
            .await
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
                decision
            }
            GateDecision::Confirmed { observed } => {
                log::trace!(
                    "paste(clipboard-node): target client-base {:#x} confirmed \
                     fetches={} (>= expected={})",
                    target_base,
                    observed,
                    expected_prev,
                );
                decision
            }
            GateDecision::AbortedTimeout { .. } => decision,
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
    fetch_count: impl Fn() -> u32,
    timeout: Duration,
    quiescence: Duration,
) -> GateDecision {
    let deadline = Instant::now() + timeout;
    let poll = Duration::from_millis(GATE_POLL_INTERVAL_MS);

    // Phase A: wait for the FIRST target fetch.
    loop {
        let count = fetch_count();
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
    let mut last_count = fetch_count();
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
        let now = fetch_count();
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
    fetch_count: impl Fn() -> u32,
    expected: u32,
    timeout: Duration,
    quiescence: Duration,
) -> GateDecision {
    let deadline = Instant::now() + timeout;
    let poll = Duration::from_millis(GATE_POLL_INTERVAL_MS);

    // Phase A: wait until count >= expected.
    let mut observed;
    loop {
        observed = fetch_count();
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
        let now = fetch_count();
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

    #[tokio::test]
    async fn retry_focus_failure_stops_before_injection() {
        let injection_count = std::sync::atomic::AtomicU32::new(0);
        let result = focus_then_inject(
            async {
                Err(TalkError::Clipboard(
                    "target focus remained elsewhere".to_string(),
                ))
            },
            || async {
                injection_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        )
        .await;

        assert!(result.is_err());
        assert_eq!(injection_count.load(Ordering::Relaxed), 0);
    }

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
    // Production wait loops accept a fetch-count observer, so these
    // tests drive the same logic as X11 with a deterministic source.

    /// Spec: ABORT immediately when the target never fetches.  The
    /// pure decision computation falls through to AbortedTimeout.
    #[tokio::test]
    async fn learn_aborts_when_target_never_fetches() {
        // Use a custom helper that simulates "always 0".
        let timeout = Duration::from_millis(20);
        let quiescence = Duration::from_millis(50);
        let decision = wait_and_learn(|| 0, timeout, quiescence).await;
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
        let decision = wait_and_learn(
            move || {
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
        let decision = wait_and_learn(
            move || {
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
        let decision = wait_and_confirm(
            move || {
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
        let decision = wait_and_confirm(
            move || {
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

    // Retry assertions below exercise the production attempt loop directly.

    #[tokio::test]
    async fn partial_target_fetch_never_injects_the_same_chunk_twice() {
        let learned =
            run_target_attempts(2, |_| async { Ok(GateDecision::Learned { expected: 2 }) })
                .await
                .expect("first chunk learns fetch count");
        assert_eq!(learned, GateDecision::Learned { expected: 2 });
        let mut injections = Vec::new();
        let result = run_target_attempts(2, |_| {
            injections.push("chunk");
            async {
                Ok(GateDecision::AbortedTimeout {
                    observed: 1,
                    required: 2,
                })
            }
        })
        .await;
        assert!(matches!(
            result,
            Ok(GateDecision::AbortedTimeout {
                observed: 1,
                required: 2
            })
        ));
        assert_eq!(injections, vec!["chunk"]);
    }

    #[tokio::test]
    async fn zero_fetch_attempt_can_retry_and_confirm() {
        let mut injections = 0;
        let result = run_target_attempts(2, |_| {
            injections += 1;
            let observed = injections;
            async move {
                if observed == 1 {
                    Ok(GateDecision::AbortedTimeout {
                        observed: 0,
                        required: 2,
                    })
                } else {
                    Ok(GateDecision::Confirmed { observed: 2 })
                }
            }
        })
        .await;
        assert!(matches!(
            result,
            Ok(GateDecision::Confirmed { observed: 2 })
        ));
        assert_eq!(injections, 2);
    }

    #[tokio::test]
    async fn zero_fetch_exhausts_only_after_configured_attempts() {
        let mut injections = 0;
        let result = run_target_attempts(2, |_| {
            injections += 1;
            async {
                Ok(GateDecision::AbortedTimeout {
                    observed: 0,
                    required: 1,
                })
            }
        })
        .await;
        assert!(matches!(
            result,
            Ok(GateDecision::AbortedTimeout {
                observed: 0,
                required: 1
            })
        ));
        assert_eq!(injections, 3);
    }
}
