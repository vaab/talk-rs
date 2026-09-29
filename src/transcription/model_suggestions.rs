//! Cached model-list fetching with retries.
//!
//! Provides a generic utility for fetching `/v1/models` from any
//! provider API, filtering results, and caching them in-process.
//!
//! Provider-specific logic (error detection, model filters) lives in
//! the provider modules (`mistral.rs`, `openai.rs`), not here.

use crate::error::TalkError;
use crate::telemetry::{NoOpSink, TelemetrySink};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use super::transport::{self, Method, Request, RequestBody};
use tokio_util::sync::CancellationToken;

/// Cache TTL — model lists rarely change.
const CACHE_TTL: Duration = Duration::from_secs(3600);

/// Wall-clock budget for a single `/v1/models` request.  The
/// transport's connection-retry schedule covers transient network
/// issues; this wall-clock guards against a server that accepts
/// the connection but stalls.
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

// ── Cache ───────────────────────────────────────────────────────────

struct CachedModels {
    models: Vec<String>,
    fetched_at: Instant,
}

type CacheKey = (String, String);

/// Model lists are cached per (API endpoint, credential) because
/// different credentials may have different model entitlements.
static MODELS_CACHE: OnceLock<Mutex<HashMap<CacheKey, CachedModels>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<CacheKey, CachedModels>> {
    MODELS_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── Fetching ────────────────────────────────────────────────────────

/// Fetch transcription-relevant model names from a provider API.
///
/// Calls `GET {api_base}/v1/models` via the consolidated
/// [`super::transport::http_request`] (which owns the retry loop),
/// filters results with the caller-supplied `filter` predicate, and
/// caches the result in-process with a TTL per (API endpoint,
/// credential), since credentials may have different model
/// entitlements.
///
/// On transport failure, returns the stale cache if one exists,
/// otherwise propagates the structured failure.
pub(crate) async fn fetch_transcription_models(
    api_key: &str,
    api_base: &str,
    filter: fn(&str) -> bool,
) -> Result<Vec<String>, TalkError> {
    let cache_key = (api_base.to_string(), api_key.to_string());

    // Fast path: fresh cache hit.
    if let Ok(guard) = cache().lock() {
        if let Some(cached) = guard.get(&cache_key) {
            if cached.fetched_at.elapsed() < CACHE_TTL {
                return Ok(filtered_models(&cached.models, filter));
            }
        }
    }

    let models_url = format!("{}/v1/models", api_base);
    let req = Request {
        retry_schedule: Default::default(),
        method: Method::Get,
        url: models_url,
        headers: vec![("Authorization".into(), format!("Bearer {}", api_key))],
        body: RequestBody::Empty,
        // Provider tag is just for error display; the model
        // suggestion fetch is shared across providers.
        provider: crate::config::Provider::Mistral,
        provider_name: "model-suggestions".into(),
        phase: crate::error::PipelinePhase::Validate,
        wall_clock: Some(FETCH_TIMEOUT),
    };

    let sink: Arc<dyn TelemetrySink> = Arc::new(NoOpSink);

    match transport::http_request(req, &sink, CancellationToken::new()).await {
        Ok(response) => {
            if !(200..300).contains(&response.status) {
                let body = String::from_utf8_lossy(&response.body).to_string();
                let err =
                    TalkError::Config(format!("models API error ({}): {}", response.status, body));
                return fallback_to_stale_cache(&cache_key, filter, err);
            }

            let parsed: ModelsResponse = match serde_json::from_slice(&response.body) {
                Ok(p) => p,
                Err(e) => {
                    let err = TalkError::Config(format!("failed to parse models response: {}", e));
                    return fallback_to_stale_cache(&cache_key, filter, err);
                }
            };

            let mut raw: Vec<String> = parsed.data.iter().map(|m| m.id.clone()).collect();
            raw.sort();

            // Update cache.
            if let Ok(mut guard) = cache().lock() {
                guard.insert(
                    cache_key,
                    CachedModels {
                        models: raw.clone(),
                        fetched_at: Instant::now(),
                    },
                );
            }
            Ok(filtered_models(&raw, filter))
        }
        Err(pf) => {
            let err: TalkError = pf.into();
            fallback_to_stale_cache(&cache_key, filter, err)
        }
    }
}

/// Return a stale cached entry if one exists, otherwise propagate
/// the supplied error.
fn filtered_models(models: &[String], filter: fn(&str) -> bool) -> Vec<String> {
    models.iter().filter(|m| filter(m)).cloned().collect()
}

fn fallback_to_stale_cache(
    cache_key: &CacheKey,
    filter: fn(&str) -> bool,
    err: TalkError,
) -> Result<Vec<String>, TalkError> {
    if let Ok(guard) = cache().lock() {
        if let Some(cached) = guard.get(cache_key) {
            log::warn!("model-list fetch failed ({}); using stale cache", err);
            return Ok(filtered_models(&cached.models, filter));
        }
    }
    Err(err)
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    id: String,
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

    #[tokio::test]
    async fn cached_raw_models_are_filtered_for_each_caller() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let mock = Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"id": "batch-model"}, {"id": "stream-model"}]
            })))
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        let batch = fetch_transcription_models("distinct-filter-key", &server.uri(), |id| {
            id.starts_with("batch")
        })
        .await
        .unwrap();
        let stream = fetch_transcription_models("distinct-filter-key", &server.uri(), |id| {
            id.starts_with("stream")
        })
        .await
        .unwrap();
        assert_eq!(batch, vec!["batch-model"]);
        assert_eq!(stream, vec!["stream-model"]);
        drop(mock);
    }
}
