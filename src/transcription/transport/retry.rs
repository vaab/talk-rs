//! Shared retry primitive for transcription transport calls.
//!
//! Used by **both** batch HTTP POST and realtime WebSocket upgrade.
//! Every outgoing provider API call wraps itself in
//! [`with_retry`] — there is no "without retry" code path.
//!
//! # Why it lives here
//!
//! Retry is a property of "open a connection and wait for an
//! acknowledged response".  That applies identically to an HTTP
//! POST and to a WebSocket upgrade handshake (which IS an HTTP
//! request).  The retry logic, budget, bail-on-model-error
//! semantics, and telemetry event emission are therefore shared,
//! implemented once, and invoked polymorphically by each provider
//! transport.

use crate::config::Provider;
use crate::error::TalkError;
use crate::telemetry::{TelemetrySink, TranscriptionEvent};
use crate::transcription::is_model_error;
use std::future::Future;
use std::sync::Arc;

/// Maximum number of retry attempts before giving up.
///
/// Matches the budget used by the dictate retry loop before this
/// primitive existed.  Do not change without understanding the
/// downstream visualizer pacing: each attempt emits a
/// [`TranscriptionEvent::RetryScheduled`] event that the UI
/// surfaces to the user.
pub(crate) const MAX_RETRIES: u32 = 5;

/// Execute a batch transcription call with retry semantics shared
/// across providers and transports.
///
/// The retry loop:
///
/// 1. Calls `call()` once.
/// 2. On `Ok(result)` → return immediately.
/// 3. On `Err(e)` where [`is_model_error`] returns `true` → bail
///    immediately.  Model-not-found is permanent; retrying wastes
///    the user's time.
/// 4. On any other `Err(e)` → emit
///    [`TranscriptionEvent::RetryScheduled`] on `sink` and retry,
///    up to [`MAX_RETRIES`] additional attempts.
///
/// After [`MAX_RETRIES`] exhausted retries, returns the last
/// error.
///
/// # Parameters
///
/// - `provider`: used by [`is_model_error`] to check provider-specific
///   error patterns.
/// - `sink`: receives `RetryScheduled` events before each retry.
///   Callers who do not care pass
///   [`crate::telemetry::NoOpSink`] wrapped in `Arc`.
/// - `call`: an idempotent async closure that performs the actual
///   transport operation (HTTP POST or WS upgrade).  Invoked at
///   most `MAX_RETRIES + 1` times.
///
/// # Returns
///
/// The successful result, or the final error after all retries
/// are exhausted.
pub(crate) async fn with_retry<T, F, Fut>(
    provider: Provider,
    sink: &Arc<dyn TelemetrySink>,
    call: F,
) -> Result<T, TalkError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, TalkError>>,
{
    let mut last_err: Option<TalkError> = None;

    // Attempt 0 is the initial call; attempts 1..=MAX_RETRIES are retries.
    for attempt in 0..=MAX_RETRIES {
        if attempt > 0 {
            let reason = last_err.as_ref().map(|e| e.to_string()).unwrap_or_default();
            sink.emit(TranscriptionEvent::RetryScheduled {
                attempt,
                max: MAX_RETRIES,
                reason,
                t: std::time::Instant::now(),
            });
        }

        match call().await {
            Ok(result) => return Ok(result),
            Err(e) if is_model_error(provider, &e) => return Err(e),
            Err(e) => last_err = Some(e),
        }
    }

    Err(last_err
        .unwrap_or_else(|| TalkError::Transcription("transcription failed after retries".into())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::NoOpSink;
    use crate::transcription::TranscriptionResult;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Simple in-memory sink that counts `RetryScheduled` events.
    struct CountingSink {
        retries: AtomicU32,
    }

    impl CountingSink {
        fn new() -> Self {
            Self {
                retries: AtomicU32::new(0),
            }
        }

        fn retry_count(&self) -> u32 {
            self.retries.load(Ordering::SeqCst)
        }
    }

    impl TelemetrySink for CountingSink {
        fn emit(&self, event: TranscriptionEvent) {
            if matches!(event, TranscriptionEvent::RetryScheduled { .. }) {
                self.retries.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    #[tokio::test]
    async fn succeeds_on_first_try_emits_no_retries() {
        let counting = Arc::new(CountingSink::new());
        let sink: Arc<dyn TelemetrySink> = counting.clone();

        let result = with_retry(Provider::OpenAI, &sink, || async {
            Ok(TranscriptionResult {
                text: "hello".into(),
                ..Default::default()
            })
        })
        .await;

        assert_eq!(result.expect("ok").text, "hello");
        assert_eq!(counting.retry_count(), 0);
    }

    #[tokio::test]
    async fn retries_on_transient_errors_then_succeeds() {
        let counting = Arc::new(CountingSink::new());
        let sink: Arc<dyn TelemetrySink> = counting.clone();
        let call_count = Arc::new(AtomicU32::new(0));

        let result = with_retry(Provider::OpenAI, &sink, || {
            let count = call_count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(TalkError::Transcription(format!(
                        "transient error on attempt {}",
                        n
                    )))
                } else {
                    Ok(TranscriptionResult {
                        text: "success".into(),
                        ..Default::default()
                    })
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "success");
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
        // 2 retries emitted (attempts 1 and 2; attempt 0 is initial
        // call and does not emit).
        assert_eq!(counting.retry_count(), 2);
    }

    #[tokio::test]
    async fn bails_immediately_on_model_error() {
        let counting = Arc::new(CountingSink::new());
        let sink: Arc<dyn TelemetrySink> = counting.clone();
        let call_count = Arc::new(AtomicU32::new(0));

        let result: Result<TranscriptionResult, TalkError> =
            with_retry(Provider::OpenAI, &sink, || {
                let count = call_count.clone();
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    // OpenAI model error pattern: `model_not_found`.
                    Err(TalkError::Transcription(
                        "model_not_found: whisper-xyz".into(),
                    ))
                }
            })
            .await;

        assert!(result.is_err());
        // Exactly one attempt — no retries on permanent error.
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        assert_eq!(counting.retry_count(), 0);
    }

    #[tokio::test]
    async fn exhausts_max_retries_then_returns_last_error() {
        let counting = Arc::new(CountingSink::new());
        let sink: Arc<dyn TelemetrySink> = counting.clone();
        let call_count = Arc::new(AtomicU32::new(0));

        let result: Result<TranscriptionResult, TalkError> =
            with_retry(Provider::OpenAI, &sink, || {
                let count = call_count.clone();
                async move {
                    let n = count.fetch_add(1, Ordering::SeqCst);
                    Err(TalkError::Transcription(format!(
                        "persistent error on attempt {}",
                        n
                    )))
                }
            })
            .await;

        assert!(result.is_err());
        // Total attempts = MAX_RETRIES + 1 (initial + N retries).
        assert_eq!(call_count.load(Ordering::SeqCst), MAX_RETRIES + 1);
        // Retries emitted = MAX_RETRIES.
        assert_eq!(counting.retry_count(), MAX_RETRIES);
    }

    #[tokio::test]
    async fn no_op_sink_does_not_panic() {
        let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);
        let call_count = Arc::new(AtomicU32::new(0));

        let _ = with_retry(Provider::Mistral, &sink, || {
            let count = call_count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 1 {
                    Err(TalkError::Transcription("transient".into()))
                } else {
                    Ok(TranscriptionResult::default())
                }
            }
        })
        .await;

        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }
}
