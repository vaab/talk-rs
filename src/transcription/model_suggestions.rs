//! Cached model-list fetching with retries.
//!
//! Provides a generic utility for fetching `/v1/models` from any
//! provider API, filtering results, and caching them in-process.
//!
//! Provider-specific logic (error detection, model filters) lives in
//! the provider modules (`mistral.rs`, `openai.rs`), not here.

use crate::error::TalkError;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Cache TTL — model lists rarely change.
const CACHE_TTL: Duration = Duration::from_secs(3600);

/// Timeout for a single `/v1/models` request.
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum number of retry attempts (matches the transcription retry
/// count in `dictate/mod.rs`).
const MAX_RETRIES: u32 = 5;

/// Delay between retry attempts.
const RETRY_DELAY: Duration = Duration::from_millis(500);

// ── Cache ───────────────────────────────────────────────────────────

struct CachedModels {
    models: Vec<String>,
    fetched_at: Instant,
}

static MODELS_CACHE: OnceLock<Mutex<HashMap<String, CachedModels>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<String, CachedModels>> {
    MODELS_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── Fetching ────────────────────────────────────────────────────────

/// Fetch transcription-relevant model names from a provider API.
///
/// Calls `GET {api_base}/v1/models`, filters results with the
/// caller-supplied `filter` predicate, and caches the result
/// in-process with a TTL.  Retries up to [`MAX_RETRIES`] times
/// on transient failures.
///
/// The caller provides the filter function so this utility has zero
/// provider-specific knowledge.
pub(crate) async fn fetch_transcription_models(
    api_key: &str,
    api_base: &str,
    filter: fn(&str) -> bool,
) -> Result<Vec<String>, TalkError> {
    let cache_key = api_base.to_string();

    // Fast path: fresh cache hit.
    if let Ok(guard) = cache().lock() {
        if let Some(cached) = guard.get(&cache_key) {
            if cached.fetched_at.elapsed() < CACHE_TTL {
                return Ok(cached.models.clone());
            }
        }
    }

    // Fetch with retries.
    let client = Client::new();
    let models_url = format!("{}/v1/models", api_base);
    let mut last_err: Option<TalkError> = None;

    for attempt in 0..MAX_RETRIES {
        if attempt > 0 {
            tokio::time::sleep(RETRY_DELAY).await;
        }

        match fetch_models_once(&client, &models_url, api_key, filter).await {
            Ok(models) => {
                // Update cache.
                if let Ok(mut guard) = cache().lock() {
                    guard.insert(
                        cache_key,
                        CachedModels {
                            models: models.clone(),
                            fetched_at: Instant::now(),
                        },
                    );
                }
                return Ok(models);
            }
            Err(e) => {
                log::warn!(
                    "models fetch attempt {}/{} failed: {}",
                    attempt + 1,
                    MAX_RETRIES,
                    e
                );
                last_err = Some(e);
            }
        }
    }

    // All retries exhausted — return stale cache if we have one.
    if let Ok(guard) = cache().lock() {
        if let Some(cached) = guard.get(&cache_key) {
            log::warn!("all model-list retries failed; using stale cache");
            return Ok(cached.models.clone());
        }
    }

    Err(last_err.unwrap_or_else(|| TalkError::Config("failed to fetch model list".to_string())))
}

/// Single attempt to fetch and filter the model list.
async fn fetch_models_once(
    client: &Client,
    url: &str,
    api_key: &str,
    filter: fn(&str) -> bool,
) -> Result<Vec<String>, TalkError> {
    #[derive(Deserialize)]
    struct ModelsResponse {
        data: Vec<ModelInfo>,
    }

    #[derive(Deserialize)]
    struct ModelInfo {
        id: String,
    }

    let response = client
        .get(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .timeout(FETCH_TIMEOUT)
        .send()
        .await
        .map_err(|e| TalkError::Config(format!("models list request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(TalkError::Config(format!(
            "models API error ({}): {}",
            status, body
        )));
    }

    let models: ModelsResponse = response
        .json()
        .await
        .map_err(|e| TalkError::Config(format!("failed to parse models response: {}", e)))?;

    let mut result: Vec<String> = models
        .data
        .iter()
        .map(|m| m.id.as_str())
        .filter(|id| filter(id))
        .map(String::from)
        .collect();
    result.sort();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fetch_transcription_models_with_mock() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"id": "voxtral-mini-2507"},
                    {"id": "voxtral-mini-2602"},
                    {"id": "mistral-large-latest"},
                    {"id": "some-transcription-model"}
                ]
            })))
            .mount(&mock_server)
            .await;

        fn mistral_filter(id: &str) -> bool {
            id.contains("voxtral") || id.contains("transcri")
        }

        let result =
            fetch_transcription_models("test-key", &mock_server.uri(), mistral_filter).await;

        assert!(result.is_ok());
        let models = result.unwrap();
        assert!(models.contains(&"voxtral-mini-2507".to_string()));
        assert!(models.contains(&"voxtral-mini-2602".to_string()));
        assert!(models.contains(&"some-transcription-model".to_string()));
        // Non-transcription model should be filtered out.
        assert!(!models.contains(&"mistral-large-latest".to_string()));
    }

    #[tokio::test]
    async fn test_fetch_caches_result() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let mock = Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("authorization", "Bearer cache-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"id": "voxtral-mini-2507"}]
            })))
            .expect(1) // Should only be called once due to caching.
            .mount_as_scoped(&mock_server)
            .await;

        fn filter(id: &str) -> bool {
            id.contains("voxtral")
        }

        let base = mock_server.uri();
        let r1 = fetch_transcription_models("cache-key", &base, filter).await;
        let r2 = fetch_transcription_models("cache-key", &base, filter).await;

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert_eq!(r1.unwrap(), r2.unwrap());

        // Drop the scoped mock to verify expectations.
        drop(mock);
    }
}
