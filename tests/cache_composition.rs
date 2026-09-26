use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use talk_rs::config::{Config, Provider};
use talk_rs::error::TalkError;
use talk_rs::recording_cache::{
    acquire_model_lock, acquire_pick_lock, get_transcript, is_model_locked, read_pick,
    release_model_lock, release_pick_lock, write_pick, TranscriptStatus,
};
use talk_rs::telemetry::{NoOpSink, TelemetrySink};
use talk_rs::transcription::{
    produce_transcript, read_cached_transcript, transcribe_audio, TranscribeOptions,
};
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Match, Mock, MockServer, Request, ResponseTemplate};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct ModelBody(&'static str);

impl Match for ModelBody {
    fn matches(&self, request: &Request) -> bool {
        let needle = format!("name=\"model\"\r\n\r\n{}\r\n", self.0);
        request
            .body
            .windows(needle.len())
            .any(|window| window == needle.as_bytes())
    }
}

struct Fixture {
    _lock: MutexGuard<'static, ()>,
    dir: TempDir,
    previous: Option<std::ffi::OsString>,
    audio: PathBuf,
    config: Config,
}

impl Fixture {
    fn new(server: &MockServer) -> Self {
        assert!(server.uri().starts_with("http://127.0.0.1:"));
        let lock = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let dir = TempDir::new().expect("tempdir");
        let previous = std::env::var_os("TALK_RS_VALIDATE_CACHE_PATH");
        // SAFETY: all tests in this integration-test binary hold ENV_LOCK.
        unsafe {
            std::env::set_var(
                "TALK_RS_VALIDATE_CACHE_PATH",
                dir.path().join("validate.yml"),
            )
        };
        let config_path = dir.path().join("config.yaml");
        std::fs::write(
            &config_path,
            format!(
                "output_dir: {}\nproviders:\n  openai:\n    api_key: test-key\n    url: {}\n    model: gpt-transcribe\ntranscription:\n  default_provider: openai\n",
                dir.path().display(),
                server.uri()
            ),
        )
        .expect("config yaml");
        let config = Config::load(Some(&config_path)).expect("load temp config");
        let audio = dir.path().join("speech.wav");
        write_wav(&audio);
        Self {
            _lock: lock,
            dir,
            previous,
            audio,
            config,
        }
    }

    fn sink() -> Arc<dyn TelemetrySink> {
        Arc::new(NoOpSink)
    }

    async fn transcribe(&self, model: Option<&str>, allow_api: bool) -> Result<String, TalkError> {
        transcribe_audio(
            &self.audio,
            &self.config,
            Provider::OpenAI,
            model,
            false,
            TranscribeOptions {
                allow_api,
                ..Default::default()
            },
            &Self::sink(),
        )
        .await
        .map(|result| result.text)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // SAFETY: ENV_LOCK remains held until after this Drop completes.
        unsafe {
            match self.previous.take() {
                Some(value) => std::env::set_var("TALK_RS_VALIDATE_CACHE_PATH", value),
                None => std::env::remove_var("TALK_RS_VALIDATE_CACHE_PATH"),
            }
        }
    }
}

fn write_wav(path: &Path) {
    let samples = [0i16; 320];
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::from(&b"RIFF"[..]);
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16_000u32.to_le_bytes());
    wav.extend_from_slice(&32_000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(path, wav).expect("wav");
}

async fn mock_models(server: &MockServer) {
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"id": "gpt-transcribe"}, {"id": "whisper-1"}]
        })))
        .mount(server)
        .await;
}

#[tokio::test]
async fn cache_only_and_held_model_lock_never_contact_provider() {
    let server = MockServer::start().await;
    let fixture = Fixture::new(&server);
    assert!(matches!(
        fixture.transcribe(None, false).await,
        Err(TalkError::CacheOnly)
    ));
    acquire_model_lock(&fixture.audio, Provider::OpenAI, "gpt-transcribe", false)
        .expect("hold lock");
    assert!(matches!(
        fixture.transcribe(None, true).await,
        Err(TalkError::ModelInProgress)
    ));
    release_model_lock(&fixture.audio, Provider::OpenAI, "gpt-transcribe", false)
        .expect("release lock");
    assert!(server
        .received_requests()
        .await
        .expect("requests")
        .is_empty());
}

#[tokio::test]
async fn config_to_sidecar_to_cache_hit_keeps_models_separate() {
    let server = MockServer::start().await;
    let fixture = Fixture::new(&server);
    mock_models(&server).await;
    for (model, text) in [("gpt-transcribe", "first"), ("whisper-1", "second")] {
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(ModelBody(model))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"text": text})),
            )
            .expect(1)
            .mount(&server)
            .await;
    }

    assert_eq!(
        fixture.transcribe(None, true).await.expect("first"),
        "first"
    );
    assert!(fixture
        .dir
        .path()
        .join("speech_openai_gpt-transcribe_oneshot.yml")
        .exists());
    assert_eq!(
        fixture.transcribe(None, true).await.expect("cached"),
        "first"
    );
    assert_eq!(
        fixture
            .transcribe(Some("whisper-1"), true)
            .await
            .expect("override"),
        "second"
    );
    assert_eq!(
        fixture
            .transcribe(Some("whisper-1"), false)
            .await
            .expect("cached override"),
        "second"
    );
    assert!(fixture
        .dir
        .path()
        .join("speech_openai_whisper-1_oneshot.yml")
        .exists());
    server.verify().await;
}

#[tokio::test]
async fn provider_failure_releases_model_lock_for_next_attempt() {
    let server = MockServer::start().await;
    let fixture = Fixture::new(&server);
    mock_models(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad audio"))
        .expect(2)
        .mount(&server)
        .await;
    assert!(matches!(
        fixture.transcribe(None, true).await,
        Err(TalkError::Pipeline(_))
    ));
    assert!(!is_model_locked(
        &fixture.audio,
        Provider::OpenAI,
        "gpt-transcribe",
        false
    ));
    assert!(matches!(
        fixture.transcribe(None, true).await,
        Err(TalkError::Pipeline(_))
    ));
    server.verify().await;
}

#[tokio::test]
async fn pick_precedes_sidecar_and_user_edit_never_contacts_provider_again() {
    let server = MockServer::start().await;
    let fixture = Fixture::new(&server);
    mock_models(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"text": "  original  "})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let text = produce_transcript(
        &fixture.audio,
        &fixture.config,
        Provider::OpenAI,
        None,
        &Fixture::sink(),
    )
    .await
    .expect("produce");
    assert_eq!(text, "original");
    assert_eq!(
        get_transcript(&fixture.audio),
        TranscriptStatus::Available("original".into())
    );
    write_pick(
        &fixture.audio,
        "openai",
        "gpt-transcribe",
        false,
        "edited by user",
    )
    .expect("edit pick");
    assert_eq!(
        read_cached_transcript(&fixture.audio, &fixture.config),
        Some("edited by user".into())
    );
    assert_eq!(
        produce_transcript(
            &fixture.audio,
            &fixture.config,
            Provider::OpenAI,
            None,
            &Fixture::sink()
        )
        .await
        .expect("existing pick"),
        "edited by user"
    );
    assert_eq!(read_pick(&fixture.audio).expect("pick").3, "edited by user");
    server.verify().await;
}

#[tokio::test]
async fn producer_failure_surfaces_error_and_releases_pick_lock() {
    let server = MockServer::start().await;
    let fixture = Fixture::new(&server);
    acquire_pick_lock(&fixture.audio).expect("hold pick lock");
    assert!(matches!(
        produce_transcript(
            &fixture.audio,
            &fixture.config,
            Provider::OpenAI,
            None,
            &Fixture::sink()
        )
        .await,
        Err(TalkError::TranscriptInProgress)
    ));
    release_pick_lock(&fixture.audio).expect("release held lock");
    mock_models(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad audio"))
        .expect(1)
        .mount(&server)
        .await;
    let result = produce_transcript(
        &fixture.audio,
        &fixture.config,
        Provider::OpenAI,
        None,
        &Fixture::sink(),
    )
    .await;
    assert!(matches!(result, Err(TalkError::Pipeline(_))));
    assert_eq!(
        get_transcript(&fixture.audio),
        TranscriptStatus::NotAvailable
    );
    acquire_pick_lock(&fixture.audio).expect("failure released lock");
    server.verify().await;
}
