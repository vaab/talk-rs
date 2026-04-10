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
/// detect dead connections within a few seconds, rather than relying
/// on large overall request timeouts.  Individual requests should
/// **not** set their own `.timeout()` — the per-frame and kernel
/// timeouts handle stalls automatically.
pub(crate) fn build_client() -> Result<Client, TalkError> {
    // NOTE: `read_timeout` is intentionally omitted.  In reqwest 0.13
    // it acts as a non-resetting wall-clock timer during the
    // upload + wait-for-headers phase, which would kill legitimate
    // requests where upload + server processing exceeds the value.
    // Dead connections are detected by `tcp_user_timeout` (during
    // active transfer) and TCP keepalive (during idle phases).
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
