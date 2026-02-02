# Talk-RS Architectural Decisions

## Task 1.6: Audio Encoder Implementation

### Decision 1: Trait-Based Encoder Architecture
**Decision**: Use trait `AudioEncoder` with pluggable implementations
**Rationale**:
- Enables swapping codecs (Opus, AAC, FLAC) without changing client code
- Supports testing with MockEncoder
- Follows Rust idioms for abstraction
- Aligns with AudioCapture trait pattern

**Alternatives Considered**:
- Enum-based dispatch: Less flexible, harder to add new codecs
- Direct Opus usage: Tightly coupled, harder to test

**Impact**: Enables future codec support without refactoring

---

### Decision 2: Stateful Encoding with Internal Buffering
**Decision**: Encoders maintain internal buffer for frame-based codecs
**Rationale**:
- Opus requires fixed frame sizes (10, 20, 40, 60ms)
- Audio capture produces variable-sized chunks
- Buffering decouples capture from encoding
- Flush pattern ensures all data is output

**Alternatives Considered**:
- Require caller to buffer: Shifts complexity to client
- Fixed-size chunks from capture: Constrains capture design

**Impact**: Encoder handles frame alignment transparently

---

### Decision 3: Configuration from AudioConfig
**Decision**: All encoder parameters (sample_rate, channels, bitrate) from AudioConfig
**Rationale**:
- Single source of truth for audio parameters
- Enables configuration-driven behavior
- Supports environment variable overrides
- Consistent with capture module design

**Alternatives Considered**:
- Hardcoded defaults: Inflexible, harder to test
- Per-encoder configuration: Duplicates config logic

**Impact**: Encoders are fully configurable without code changes

---

### Decision 4: Mono/Stereo Only Support
**Decision**: Limit Opus encoder to 1 (mono) and 2 (stereo) channels
**Rationale**:
- Voice recording typically mono or stereo
- Opus supports these natively
- Simplifies error handling
- Matches typical use cases

**Alternatives Considered**:
- Support all Opus channel counts: Adds complexity for rare cases
- Automatic downmix: Loses information

**Impact**: Clear error for unsupported channel counts

---

### Decision 5: 20ms Frame Duration
**Decision**: Use 20ms frames (matches CHUNK_DURATION_MS)
**Rationale**:
- Aligns with audio capture chunk duration
- Good balance between latency and compression
- Standard for voice codecs
- Reduces buffering overhead

**Alternatives Considered**:
- 10ms: Lower latency, higher overhead
- 40ms: Higher latency, better compression
- 60ms: Too high latency for interactive use

**Impact**: Minimal buffering, predictable latency

---

### Decision 6: Zero-Padding on Flush
**Decision**: Pad incomplete final frame with zeros
**Rationale**:
- Ensures all samples are encoded
- Opus requires complete frames
- Zeros are silent, minimal artifact
- Simple to implement

**Alternatives Considered**:
- Discard incomplete frame: Loses audio data
- Repeat last sample: More complex, similar artifacts
- Silence detection: Adds complexity

**Impact**: No audio loss at stream end

---

### Decision 7: MockEncoder as Pass-Through
**Decision**: MockEncoder converts i16 to bytes without compression
**Rationale**:
- Enables testing without codec overhead
- Useful for CI/CD without libopus
- Follows MockAudioCapture pattern
- Simple to understand and verify

**Alternatives Considered**:
- Synthetic compression: More realistic but complex
- No-op encoder: Doesn't test byte conversion

**Impact**: Fast tests, clear test behavior

---

### Decision 8: unsafe impl Send for OpusEncoder
**Decision**: Manually implement Send for OpusEncoder
**Rationale**:
- opus::Encoder is Send but not explicitly marked
- Safe because no interior mutability or raw pointers
- Required for async/tokio compatibility
- Documented with explanatory comment

**Alternatives Considered**:
- Wrap in Arc<Mutex<>>: Unnecessary overhead
- Don't implement Send: Breaks async usage

**Impact**: Encoder can be used in async contexts

---

### Decision 9: Comprehensive Unit Tests
**Decision**: Include 10 encoder-specific tests
**Rationale**:
- Catches bugs early
- Documents expected behavior
- Enables refactoring with confidence
- Tests both happy path and error cases

**Test Coverage**:
- MockEncoder: encode, flush, buffer behavior
- OpusEncoder: creation, mono, stereo, flush, invalid channels
- Roundtrip: encode → decode → verify

**Impact**: High confidence in encoder correctness

---

### Decision 10: Three-Commit Strategy
**Decision**: Split implementation into 3 atomic commits
**Rationale**:
- Trait definition: Core abstraction
- MockEncoder: Testing support
- OpusEncoder: Real implementation
- Each commit is independently reviewable
- Follows established pattern from audio module

**Commits**:
1. `new: [audio] add ``AudioEncoder`` trait definition`
2. `new: [audio] add ``MockEncoder`` for testing`
3. `new: [audio] add ``OpusEncoder`` implementation`

**Impact**: Clear commit history, easier debugging

---

## Meta-Goals Alignment

### Modularity & Swappability ✓
- AudioEncoder trait enables pluggable backends
- MockEncoder for testing
- OpusEncoder for production
- Future codecs can be added without changes to trait

### Zero Hardcoded Defaults ✓
- All values from AudioConfig
- No magic numbers in encoder
- Configuration-driven behavior
- Environment variable overrides supported

### Streaming-First Design ✓
- Stateful encoding with buffering
- Processes chunks as they arrive
- Flush pattern for finalization
- Matches capture module design

### Future Extensibility ✓
- Trait-based design accommodates new codecs
- Configuration pattern supports new parameters
- Error handling extensible for new error types
- No breaking changes needed for new implementations
