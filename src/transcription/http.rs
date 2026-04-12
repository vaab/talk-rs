//! Shared HTTP helpers for batch transcription backends.
//!
//! Contains common constants, client construction, JSON parsing
//! utilities, and model validation logic used by both the Mistral
//! and OpenAI providers.

use crate::error::TalkError;
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

/// Timeout for the lightweight model-listing preflight check.
pub(crate) const VALIDATE_TIMEOUT: Duration = Duration::from_secs(10);

/// TCP connect timeout — fail fast when the server is unreachable.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// Kernel-level unacknowledged-data timeout (Linux only).
///
/// If transmitted data (upload bytes, TCP ACKs) goes unacknowledged
/// for this duration, the kernel forcefully closes the socket.  This
/// is the primary mechanism for detecting silent VPN/network drops
/// during active data transfer — unlike TCP keepalive, it fires even
/// when the send buffer is full.
#[cfg(target_os = "linux")]
const TCP_USER_TIMEOUT: Duration = Duration::from_secs(3);

/// TCP keepalive idle time — start sending probes after this much
/// silence on an otherwise idle connection.
const TCP_KEEPALIVE: Duration = Duration::from_secs(5);

/// Interval between TCP keepalive probes once probing starts.
const TCP_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(1);

/// Number of unanswered keepalive probes before declaring the
/// connection dead.
const TCP_KEEPALIVE_RETRIES: u32 = 3;

/// Build an HTTP client tuned for VPN-hostile networks.
///
/// The client uses aggressive idle and kernel-level timeouts to
/// detect dead connections within a few seconds.  Transcription
/// request paths additionally wrap individual calls in a
/// [`proportional_timeout`] so a server that accepts the connection
/// and then hangs on the application layer (where TCP-level defences
/// cannot help) is bounded by a payload-sized wall clock.
pub(crate) fn build_client() -> Result<Client, TalkError> {
    // NOTE: The client-wide `read_timeout` is intentionally omitted.
    // In reqwest 0.13 it acts as a non-resetting wall-clock timer
    // during the upload + wait-for-headers phase, which would kill
    // legitimate requests where upload + server processing exceeds
    // the value.  Dead connections are detected by `tcp_user_timeout`
    // (during active transfer) and TCP keepalive (during idle
    // phases).  Per-request wall-clock caps for slow-but-alive
    // servers are applied at each call site via [`proportional_timeout`].
    let builder = Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .tcp_keepalive(TCP_KEEPALIVE)
        .tcp_keepalive_interval(TCP_KEEPALIVE_INTERVAL)
        .tcp_keepalive_retries(TCP_KEEPALIVE_RETRIES);

    #[cfg(target_os = "linux")]
    let builder = builder.tcp_user_timeout(TCP_USER_TIMEOUT);

    builder
        .build()
        .map_err(|e| TalkError::Config(format!("failed to build HTTP client: {}", e)))
}

/// Floor for [`proportional_timeout`].
///
/// Tiny audio payloads still need to withstand a couple of seconds of
/// TLS handshake + server processing without tripping their own
/// request timeout.
const REQUEST_TIMEOUT_FLOOR_SECS: u64 = 3;

/// Divisor for [`proportional_timeout`].
///
/// Controls how the per-KB timeout budget scales.  `KB_DIVISOR = 10`
/// gives `1 s per 10 KB of audio`, which (for typical OGG Opus at
/// ~4 KB/s of encoded audio) corresponds to roughly `2.5 ×` the
/// recording duration — generous enough for legitimate server
/// processing, tight enough to catch application-layer hangs within
/// seconds for typical short dictations.
const REQUEST_TIMEOUT_KB_DIVISOR: u64 = 10;

/// Compute a per-request wall-clock timeout proportional to the audio
/// payload size.
///
/// Formula: `max(REQUEST_TIMEOUT_FLOOR_SECS, audio_bytes / 1024 / REQUEST_TIMEOUT_KB_DIVISOR)`
/// seconds.
///
/// # Rationale
///
/// Transcription latency loosely scales with audio duration, which
/// itself scales with payload size for a fixed encoding.  A fixed
/// wall-clock timeout either (a) kills legitimate long requests or
/// (b) lets short requests hang for minutes when the server accepts
/// the connection and then silently stalls (TCP-level defences cannot
/// detect this).  A proportional cap gives small requests a generous
/// fixed floor, large requests enough headroom, and bounds the
/// worst-case user wait to something comfortably below the observed
/// 168 s application-layer hang.
///
/// # Examples
///
/// | Audio size | Timeout |
/// |------------|---------|
/// | < 30 KB    | 3 s     |
/// | 100 KB     | 10 s    |
/// | 150 KB     | 15 s    |
/// | 1 MB       | 102 s   |
pub(crate) fn proportional_timeout(audio_bytes: u64) -> Duration {
    let kb = audio_bytes / 1024;
    let secs = std::cmp::max(REQUEST_TIMEOUT_FLOOR_SECS, kb / REQUEST_TIMEOUT_KB_DIVISOR);
    Duration::from_secs(secs)
}

/// Extract an optional `u64` from a JSON object by key.
///
/// Handles both unsigned and non-negative signed integer values.
pub(crate) fn parse_u64_field(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(|v| {
        v.as_u64().or_else(|| {
            v.as_i64()
                .and_then(|n| if n >= 0 { Some(n as u64) } else { None })
        })
    })
}

// ── Model validation ────────────────────────────────────────────────

/// Response from a `/v1/models` endpoint (same shape for Mistral and OpenAI).
#[derive(Debug, Deserialize)]
pub(crate) struct ModelsResponse {
    pub data: Vec<ModelInfo>,
}

/// A single entry in the models list.
#[derive(Debug, Deserialize)]
pub(crate) struct ModelInfo {
    pub id: String,
}

/// Validate that `model` is available at `api_base/v1/models`.
///
/// `provider_name` is used only in error messages (e.g. `"Mistral"`,
/// `"OpenAI"`).  `is_transcription_model` filters the available models
/// to suggest transcription-relevant alternatives on failure.
pub(crate) async fn validate_model(
    provider_name: &str,
    api_key: &str,
    model: &str,
    api_base: &str,
    is_transcription_model: fn(&str) -> bool,
) -> Result<(), TalkError> {
    let models_url = format!("{}/v1/models", api_base);

    let client = build_client()?;
    let response = client
        .get(&models_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .timeout(VALIDATE_TIMEOUT)
        .send()
        .await
        .map_err(|e| {
            TalkError::Config(format!("Failed to connect to {} API: {}", provider_name, e))
        })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(TalkError::Config(format!(
            "{} API error ({}): {}",
            provider_name, status, body
        )));
    }

    let models: ModelsResponse = response.json().await.map_err(|e| {
        TalkError::Config(format!(
            "Failed to parse {} models response: {}",
            provider_name, e
        ))
    })?;

    if models.data.iter().any(|m| m.id == model) {
        return Ok(());
    }

    // Model not found — collect transcription-relevant alternatives.
    let mut transcription_models: Vec<&str> = models
        .data
        .iter()
        .map(|m| m.id.as_str())
        .filter(|id| is_transcription_model(id))
        .collect();
    transcription_models.sort();

    if transcription_models.is_empty() {
        Err(TalkError::Config(format!(
            "Model '{}' not found in {} account",
            model, provider_name
        )))
    } else {
        Err(TalkError::Config(format!(
            "Model '{}' not found. Available transcription models: {}",
            model,
            transcription_models.join(", ")
        )))
    }
}

/// Enrich a model error with available transcription model suggestions.
///
/// Returns the error unchanged if it is not a model error or if
/// suggestions cannot be fetched.
pub(crate) async fn enrich_model_error(
    error: TalkError,
    api_key: &str,
    model: &str,
    api_base: &str,
    is_model_error: fn(&TalkError) -> bool,
    is_transcription_model: fn(&str) -> bool,
) -> TalkError {
    if !is_model_error(&error) {
        return error;
    }
    match super::model_suggestions::fetch_transcription_models(
        api_key,
        api_base,
        is_transcription_model,
    )
    .await
    {
        Ok(models) if !models.is_empty() => TalkError::Transcription(format!(
            "Model '{}' not found. Available transcription models: {}",
            model,
            models.join(", ")
        )),
        Ok(_) => TalkError::Transcription(format!(
            "Model '{}' not found (no transcription models available in account)",
            model
        )),
        Err(e) => {
            log::warn!("could not fetch model suggestions: {}", e);
            error
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_timeout_returns_floor_for_small_audio() {
        assert_eq!(proportional_timeout(0), Duration::from_secs(3));
        assert_eq!(proportional_timeout(1024), Duration::from_secs(3));
        assert_eq!(proportional_timeout(10 * 1024), Duration::from_secs(3));
        // 29 KB: 29 / 10 = 2 < floor(3), so floor applies.
        assert_eq!(proportional_timeout(29 * 1024), Duration::from_secs(3));
    }

    #[test]
    fn proportional_timeout_transitions_above_floor_at_30kb() {
        // 30 KB: 30 / 10 = 3 == floor; tied, returns 3.
        assert_eq!(proportional_timeout(30 * 1024), Duration::from_secs(3));
        // 31 KB: 31 / 10 = 3 == floor; still 3.
        assert_eq!(proportional_timeout(31 * 1024), Duration::from_secs(3));
        // 40 KB: 40 / 10 = 4 > floor; now scales with size.
        assert_eq!(proportional_timeout(40 * 1024), Duration::from_secs(4));
    }

    #[test]
    fn proportional_timeout_scales_linearly_with_kb() {
        // 100 KB → 10 s
        assert_eq!(proportional_timeout(100 * 1024), Duration::from_secs(10));
        // 147 KB → 14 s (the known 168s hang case gets bounded here)
        assert_eq!(proportional_timeout(147 * 1024), Duration::from_secs(14));
        // 500 KB → 50 s
        assert_eq!(proportional_timeout(500 * 1024), Duration::from_secs(50));
        // 1 MB → 102 s
        assert_eq!(proportional_timeout(1024 * 1024), Duration::from_secs(102));
    }

    #[test]
    fn proportional_timeout_handles_large_audio_without_overflow() {
        // 16 MB should yield a sane (if generous) value.
        let t = proportional_timeout(16 * 1024 * 1024);
        assert_eq!(t, Duration::from_secs(1638));
    }

    #[test]
    fn proportional_timeout_rounds_down_on_non_round_kb() {
        // 1500 bytes = 1 KB (integer division), floor still applies.
        assert_eq!(proportional_timeout(1500), Duration::from_secs(3));
        // 45_678 bytes = 44 KB, 44 / 10 = 4 > floor.
        assert_eq!(proportional_timeout(45_678), Duration::from_secs(4));
    }
}
