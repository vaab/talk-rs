# Multi-Provider Transcription — Design

## Problem

Voxtral realtime is unreliable.  We need to:

1. Support multiple providers (Mistral, OpenAI, future others)
2. Support multiple modes per provider (batch file-upload, streaming/realtime)
3. Send audio to **several providers simultaneously** and compare results
4. Make provider/model selectable via config AND CLI flags
5. Keep the current best path (Mistral v1 batch, full file upload) as default

## Current Architecture

```
Transcriber trait (async_trait)
├── transcribe_file(&Path) -> String          # batch: read file, POST multipart
└── transcribe_stream(Receiver<Vec<u8>>) -> String  # batch: stream encoded bytes, POST multipart

MistralTranscriber         implements Transcriber    (batch file-upload)
MistralRealtimeTranscriber  standalone struct         (WebSocket, raw PCM in, event stream out)
MockTranscriber            implements Transcriber    (testing)
```

**Problems with current design:**
- Realtime transcriber is not behind a trait — can't swap providers
- `Transcriber` trait returns `String` — no incremental events for realtime
- PCM (`Vec<i16>`) vs encoded bytes (`Vec<u8>`) split is baked in
- Config only has `providers.mistral` — no provider selection
- No CLI flag for provider/model override

## Provider Capabilities Matrix

| Provider | Batch (file) | Batch streaming (upload) | Realtime (WebSocket) | Realtime (SSE) |
|----------|-------------|-------------------------|---------------------|----------------|
| Mistral  | ✅ v1 API    | ✅ v1 API (same endpoint) | ✅ Voxtral Realtime WS | ❌ |
| OpenAI   | ✅ Whisper API | ✅ same endpoint          | ✅ Realtime API (ephemeral token) | ✅ `stream=true` on batch |

### OpenAI Models

| Model | Batch | SSE stream | Realtime WS | Segments | Notes |
|-------|-------|------------|-------------|----------|-------|
| `whisper-1` | ✅ | ❌ | ✅ | `verbose_json` | Cheapest, most format options |
| `gpt-4o-transcribe` | ✅ | ✅ | ✅ | via SSE events | Higher quality |
| `gpt-4o-mini-transcribe` | ✅ | ✅ | ✅ | via SSE events | Lower cost than 4o |

### Mistral Models

| Model | Batch | Realtime WS | Notes |
|-------|-------|-------------|-------|
| `voxtral-mini-latest` | ✅ | ❌ | Default batch model |
| `voxtral-mini-transcribe-realtime-2602` | ❌ | ✅ | Hardcoded realtime model |

## Proposed Architecture

### Two Traits, Not One

The batch and realtime paths are fundamentally different:
- Batch: file/bytes in → single String out (request/response)
- Realtime: PCM stream in → event stream out (bidirectional, incremental)

Forcing them into one trait creates awkward no-ops.  Split them:

```rust
/// Batch transcription: file or byte-stream in, full text out.
#[async_trait]
pub trait BatchTranscriber: Send + Sync {
    /// Transcribe a complete audio file from disk.
    async fn transcribe_file(&self, audio_path: &Path) -> Result<String, TalkError>;

    /// Transcribe from a stream of encoded audio bytes (OGG, WAV, etc.).
    async fn transcribe_stream(
        &self,
        audio_stream: Receiver<Vec<u8>>,
        file_name: &str,
    ) -> Result<String, TalkError>;
}

/// Realtime transcription: raw PCM in, incremental events out.
#[async_trait]
pub trait RealtimeTranscriber: Send + Sync {
    /// Connect and start streaming.  Returns a channel of events.
    async fn transcribe_realtime(
        &self,
        audio_rx: Receiver<Vec<i16>>,
    ) -> Result<Receiver<TranscriptionEvent>, TalkError>;
}
```

### Provider Enum (for Config + CLI)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Mistral,
    OpenAI,
}
```

### Factory Pattern

Instead of constructing `MistralTranscriber` directly in `dictate.rs`,
use a factory that reads config + CLI overrides:

```rust
pub fn create_batch_transcriber(config: &Config, overrides: &TranscriptionOverrides)
    -> Result<Box<dyn BatchTranscriber>, TalkError>;

pub fn create_realtime_transcriber(config: &Config, overrides: &TranscriptionOverrides)
    -> Result<Box<dyn RealtimeTranscriber>, TalkError>;
```

Where `TranscriptionOverrides` carries CLI flags:

```rust
pub struct TranscriptionOverrides {
    pub provider: Option<Provider>,
    pub model: Option<String>,
}
```

### Multi-Provider Comparison Mode

For grading/benchmarking, run multiple providers in parallel on the
same audio.  This requires:

1. The audio is captured once (or read from file once)
2. The audio data is **tee'd** to N providers simultaneously
3. Results are collected and displayed side-by-side

```rust
pub async fn transcribe_multi(
    providers: Vec<Box<dyn BatchTranscriber>>,
    audio_path: &Path,
) -> Vec<(String, Result<String, TalkError>)>;
```

For realtime comparison, each provider gets its own copy of the PCM
stream via broadcast or fan-out channel.

### New CLI Flags

On `dictate` and `transcribe`:

```
--provider <mistral|openai>     Override config default provider
--model <model-name>            Override config default model
```

On `transcribe` (and possibly `dictate --batch`):

```
--compare <provider1,provider2>  Send to multiple providers, show results side-by-side
```

### Config Changes

```yaml
providers:
  mistral:
    api_key: "..."
    model: "voxtral-mini-latest"          # batch model
    realtime_model: "voxtral-mini-transcribe-realtime-2602"  # realtime model
    context_bias: "word1,word2"
  openai:                                 # NEW
    api_key: "..."
    model: "whisper-1"                    # batch model
    realtime_model: "gpt-4o-mini-transcribe"  # realtime model

transcription:                            # NEW section
  default_provider: mistral               # which provider to use by default
  default_mode: batch                     # batch | realtime
```

`providers.openai` is optional — only required if you select OpenAI.
`providers.mistral.api_key` validation should only fire if Mistral is
actually used (not unconditionally like today).

### OpenAI Implementation

**`OpenAITranscriber`** (batch):
- Endpoint: `POST https://api.openai.com/v1/audio/transcriptions`
- Auth: `Authorization: Bearer {api_key}`
- Multipart form: `file`, `model`, `response_format=json`, optional `language`
- Response: `{"text": "..."}`
- File size limit: 25 MB
- Supported formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm

**`OpenAIRealtimeTranscriber`** (WebSocket):
- Step 1: `POST https://api.openai.com/v1/realtime/transcription_sessions` → ephemeral token
- Step 2: Connect WSS with token
- Audio format: pcm16, 24kHz (note: different from Mistral's 16kHz)
- Events: `conversation.item.input_audio_transcription.delta`, `.completed`

## File Plan

```
src/core/transcription/
├── mod.rs              # Traits (BatchTranscriber, RealtimeTranscriber), Provider enum, factory, TranscriptionEvent
├── mistral.rs          # MistralBatchTranscriber, MistralRealtimeTranscriber
├── openai.rs           # OpenAIBatchTranscriber, OpenAIRealtimeTranscriber     ← NEW
├── realtime.rs         # REMOVE: merge into mistral.rs (it's Mistral-specific)
└── multi.rs            # Multi-provider comparison runner                      ← NEW

src/core/config.rs      # Add OpenAIConfig, transcription defaults, lazy validation
src/cli/def.rs          # Add --provider, --model, --compare flags
src/cli/action/dictate.rs   # Use factory instead of hardcoded MistralTranscriber
src/cli/action/transcribe.rs # Use factory, add --compare support
```

## Migration Steps

1. **Rename traits** — `Transcriber` → `BatchTranscriber`, extract `RealtimeTranscriber`
2. **Rename impls** — `MistralTranscriber` → `MistralBatchTranscriber`
3. **Add config** — `OpenAIConfig`, `transcription` section, lazy validation
4. **Add CLI flags** — `--provider`, `--model`
5. **Add factory** — `create_batch_transcriber()`, `create_realtime_transcriber()`
6. **Wire up dictate/transcribe** — use factory instead of hardcoded constructors
7. **Implement OpenAI batch** — `OpenAIBatchTranscriber`
8. **Implement OpenAI realtime** — `OpenAIRealtimeTranscriber` (if needed)
9. **Add multi-provider** — `--compare` flag, parallel execution, side-by-side output
10. **Tests** — mock servers for OpenAI, factory tests, multi-provider tests

## Decisions (2026-02-18)

- **Default mode → batch**.  `talk-rs dictate` = batch (file upload).
  `--realtime` to opt into WebSocket streaming.  The old `--batch` flag
  is removed (batch is the default now).
- **Comparison**: `--compare mistral,openai` flag on `transcribe` and
  `dictate --batch`.  Both batch and realtime comparison eventually.
- **Benchmark workflow**:
  - `--save-benchmark` saves the captured WAV for later replay
  - `--input-audio-file <path>` replays a saved OGG/WAV file through
    the transcription pipeline instead of recording from the mic
  - These two flags enable a record-once / replay-many workflow for
    grading providers.
- **Implementation order**: incremental.
  1. Flip default to batch, add `--realtime` flag (validate both modes)
  2. Trait refactor + config + factory
  3. OpenAI batch implementation
  4. `--provider` / `--model` CLI flags
  5. `--compare` + `--save-benchmark` + `--input-audio-file`
  6. OpenAI realtime
  7. Multi-provider realtime comparison

## Open Questions

- OpenAI realtime uses 24kHz PCM — do we resample, or reject if capture is 16kHz?
- Should comparison output be structured (JSON) or human-readable table?
