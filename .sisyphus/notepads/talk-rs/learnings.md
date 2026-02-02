# Talk-RS Learnings

## Audio Encoder Implementation (Task 1.6)

### Trait Design Pattern
- `AudioEncoder` trait with `Send` bound for thread safety
- Two methods: `encode(&mut self, pcm: &[i16]) -> Result<Vec<u8>, TalkError>` and `flush(&mut self) -> Result<Vec<u8>, TalkError>`
- Stateful encoding with internal buffering for frame-based codecs
- Flush pattern ensures all buffered data is output before finalization

### MockEncoder Implementation
- Simple pass-through encoder for testing
- Converts i16 PCM samples to little-endian bytes
- Maintains internal buffer for flush() behavior
- Useful for testing pipelines without real codec overhead
- `#[allow(dead_code)]` on config field with comment explaining future use

### OpusEncoder Implementation
- Uses `opus` crate (v0.3) for Voip application
- Frame-based encoding: 20ms chunks (320 samples at 16kHz mono, 640 stereo)
- Bitrate configuration from `AudioConfig` via `set_bitrate()`
- Supports mono (1 channel) and stereo (2 channels) only
- Proper error handling for unsupported channel counts
- `unsafe impl Send` with explanatory comment (opus::Encoder is Send)
- Padding with zeros in flush() to complete final frame

### Configuration Pattern
- All encoder parameters come from `AudioConfig` (sample_rate, channels, bitrate)
- No hardcoded values - enables flexibility for different audio profiles
- Bitrate is set explicitly on Opus encoder for quality control

### Testing Strategy
- Mock encoder tests: Verify byte conversion and buffer behavior
- Opus encoder tests: Creation, mono/stereo encoding, flush behavior
- Roundtrip test: Encode PCM → Opus → decode → verify similarity
- Invalid channel count test: Ensure proper error handling
- All tests use helper function `test_config()` for consistency

### Error Handling
- `TalkError::Audio` variant for all encoding errors
- Descriptive error messages for debugging
- Proper error propagation with `?` operator

### Code Quality
- Comprehensive documentation with examples
- Proper use of `#[derive(Debug)]` for OpusEncoder
- No trailing whitespace
- Clippy clean
- All 20 tests pass (10 audio-specific)

## Audio Module Architecture

### Module Organization
- `mod.rs`: Trait definitions and re-exports
- `mock.rs`: MockAudioCapture for testing
- `cpal_capture.rs`: Real CPAL-based capture
- `encoder.rs`: AudioEncoder trait and implementations

### Constants
- `CHUNK_DURATION_MS = 20`: Standard chunk duration
- `CHANNEL_CAPACITY = 25`: Bounded channel capacity for backpressure

### Re-export Pattern
- Public re-exports in `mod.rs` for convenience: `pub use encoder::{AudioEncoder, MockEncoder, OpusEncoder};`
- Allows `use crate::core::audio::AudioEncoder` instead of `use crate::core::audio::encoder::AudioEncoder`

## Opus Codec Details

### Frame Sizes
- Valid Voip frame sizes: 10ms, 20ms, 40ms, 60ms
- Implementation uses 20ms (matches CHUNK_DURATION_MS)
- Frame size calculation: `(sample_rate * duration_ms) / 1000`

### Bitrate Configuration
- Opus supports variable bitrate (VBR) and constant bitrate (CBR)
- Implementation uses `opus::Bitrate::Bits(config.bitrate as i32)`
- Typical values: 32kbps (mono), 64kbps (stereo)

### Channel Support
- Mono: `opus::Channels::Mono`
- Stereo: `opus::Channels::Stereo`
- No support for 5.1, 7.1, or other surround formats

## Commit Message Convention

### Format
- `{type}: [{module}] add ``technical-term`` description`
- Types: `new:`, `chg:`, `fix:`, `test:`, `doc:`, `pkg:`
- Technical terms wrapped in double-backticks
- Use HEREDOC format to avoid shell escaping

### Example
```
new: [audio] add ``AudioEncoder`` trait with ``MockEncoder`` and ``OpusEncoder`` implementations
```

## Pre-Commit Checklist
1. `cargo fmt` - Format code
2. `cargo clippy --all-targets` - Check warnings
3. `cargo build` - Verify compilation
4. `cargo test` - All tests must pass
5. `grep -r "[[:space:]]$" src/` - Check trailing whitespace
6. Review dead_code pragmas with comments
7. Verify meta-goals compliance
8. Backtick audit of commit message
