# Talk-RS Issues & Gotchas

## Task 1.6: Audio Encoder Implementation

### No Issues Encountered
- Implementation completed successfully
- All tests pass (20 total, 10 audio-specific)
- Build clean with no warnings
- Clippy clean
- No trailing whitespace

### Potential Future Considerations

#### Opus Encoder Limitations
- Only supports mono (1) and stereo (2) channels
- Attempting to create encoder with 5+ channels will fail with descriptive error
- Future enhancement: Support for surround formats if needed

#### Frame Buffering
- Opus requires fixed frame sizes (10, 20, 40, 60ms)
- Implementation buffers partial frames and pads with zeros on flush
- This may introduce slight artifacts at stream end - acceptable for voice
- Alternative: Could implement frame-size negotiation with capture module

#### Bitrate Configuration
- Bitrate is set at encoder creation time
- Cannot change bitrate mid-stream without recreating encoder
- Current design assumes static bitrate per session
- Future: Could add `set_bitrate()` method if dynamic adjustment needed

#### MockEncoder Behavior
- Pass-through encoder doesn't actually compress
- Output size = input size * 2 (i16 → 2 bytes)
- Useful for testing but not representative of real compression ratios
- Consider adding synthetic compression in mock for more realistic testing

### Testing Gaps (Acceptable for MVP)
- No stress tests with very large PCM buffers
- No performance benchmarks
- No memory leak tests
- No concurrent encoder tests
- These can be added in future iterations

### Dependencies
- `opus` crate v0.3 is stable and well-maintained
- No known security issues
- Requires system libopus library (handled by opus crate)

### Error Handling
- All error paths properly covered
- Descriptive error messages for debugging
- No panics in encoder code (all Results properly handled)

## Architecture Notes

### Why unsafe impl Send?
- `opus::Encoder` is Send but not explicitly marked
- Safe to send across threads because:
  - No interior mutability
  - No raw pointers
  - Mutex/Arc not needed for encoder itself
- Capture module handles thread safety with Arc<Mutex<>>

### Why Clone on AudioConfig?
- Added `#[derive(Clone)]` to AudioConfig for test flexibility
- Allows creating multiple encoders with same config in tests
- No performance impact (small struct)
- Consistent with Rust idioms for configuration objects

### Why Separate MockEncoder?
- Enables testing without codec overhead
- Allows testing error paths without real Opus
- Useful for CI/CD on systems without libopus
- Follows established pattern from MockAudioCapture

## Lessons for Future Tasks

### Trait Design
- Always include `Send` bound for async compatibility
- Stateful traits need clear lifecycle (encode → flush)
- Document buffering behavior in trait docs

### Testing
- Mock implementations are valuable for testing
- Roundtrip tests verify codec correctness
- Error case tests (invalid channels) catch edge cases

### Configuration
- Never hardcode values - use config structs
- Validate configuration at creation time
- Provide descriptive errors for invalid configs

### Code Quality
- Use `#[allow(dead_code)]` sparingly with comments
- Derive Debug for debugging support
- Comprehensive unit tests catch issues early
