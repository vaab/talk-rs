//! Transcription interfaces and implementations.
//!
//! This module provides traits and implementations for transcribing audio
//! to text using various backends (Mistral, OpenAI).
//!
//! Two traits model the two modes of operation:
//!
//! - `OneShotTranscriber`: file or byte-stream in, full text out.
//! - `RealtimeTranscriber`: raw PCM stream in, incremental event stream out.

use crate::config::{Config, Provider};
use crate::error::TalkError;
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Audio format for one-shot uploads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum UploadFormat {
    /// Uncompressed WAV (default).
    #[default]
    Wav,
    /// OGG Opus compressed audio (smaller uploads).
    Ogg,
}

/// Body source for a one-shot transcription request.
pub(crate) enum TranscriptionBody {
    /// Full audio file on disk. The transport reads it.
    File(PathBuf),
    /// Chunks arriving through a channel — used during
    /// record-while-upload streaming. The transport collects them.
    //
    // The `Pipe` variant is only *constructed* by the live
    // record-while-upload path (`dictate`), which requires `capture`.
    // In a build without `capture` the variant is only ever pattern-
    // matched (e.g. by the parakeet backend), never constructed — but
    // it remains part of the transcription-transport contract, hence
    // the scoped allow.
    #[cfg_attr(not(feature = "capture"), allow(dead_code))]
    Pipe {
        chunks: tokio::sync::mpsc::Receiver<Vec<u8>>,
        file_name: String,
    },
}

/// Normalize an on-disk audio file into the API-optimal upload form:
/// **16 kHz mono OGG/Opus**.
///
/// Both transcription providers (Mistral Voxtral and OpenAI Whisper /
/// gpt-4o-transcribe) operate internally at 16 kHz mono — they
/// downsample and downmix any input server-side, and accept nothing
/// above 8 kHz audio bandwidth.  Uploading the original (potentially
/// 48 kHz stereo, e.g. a high-fidelity `record` output) wastes
/// bandwidth, increases latency, and can hit OpenAI's 25 MB cap, all
/// for zero accuracy benefit.  So we always re-encode to 16 kHz mono
/// before sending.
///
/// Returns `(bytes, file_name)` where `file_name` carries an `.ogg`
/// extension so the multipart upload advertises the correct format.
///
/// On any decode/encode failure (e.g. an exotic container we cannot
/// decode locally), this falls back to the original file bytes with a
/// warning, so an upload that worked before this normalization existed
/// keeps working.
pub(crate) fn normalize_file_for_upload(
    path: &std::path::Path,
) -> Result<(Vec<u8>, String), TalkError> {
    if !path.exists() {
        return Err(TalkError::Transcription(format!(
            "Audio file not found: {}",
            path.display()
        )));
    }

    crate::perf_counters::incr(crate::perf_counters::Counter::NormalizeCalls);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("audio");
    let normalized_name = format!("{stem}.ogg");

    match encode_16k_mono_ogg(path) {
        Ok(bytes) => {
            log::info!(
                "upload normalization: {} -> 16kHz mono ogg ({} bytes)",
                path.display(),
                bytes.len()
            );
            Ok((bytes, normalized_name))
        }
        Err(err) => {
            // Never break a previously-working upload: fall back to the
            // raw file bytes (with the original file name).
            log::warn!(
                "upload normalization failed for {} ({}); uploading original bytes",
                path.display(),
                err
            );
            let bytes = std::fs::read(path)
                .map_err(|e| TalkError::Transcription(format!("Failed to read audio file: {e}")))?;
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("audio.wav")
                .to_string();
            Ok((bytes, file_name))
        }
    }
}

/// Decode any supported audio file to 16 kHz mono PCM and re-encode it
/// as an OGG/Opus byte stream (16 kHz mono).
fn encode_16k_mono_ogg(path: &std::path::Path) -> Result<Vec<u8>, TalkError> {
    use crate::audio::{AudioWriter, OggOpusWriter};

    // `read_audio_as_i16` decodes wav/ogg/opus/m4a/mp4/aac, resampling
    // to 16 kHz and downmixing to mono — exactly the target form.
    let pcm = crate::record::audio::read_audio_as_i16(path)?;
    crate::perf_counters::incr(crate::perf_counters::Counter::UploadEncodes);

    // 16 kHz mono is the hardcoded transcription profile.
    let mut writer = OggOpusWriter::new(crate::config::AudioConfig::new())?;
    let mut out = writer.header()?;
    out.extend_from_slice(&writer.write_pcm(&pcm)?);
    out.extend_from_slice(&writer.finalize()?);
    Ok(out)
}

/// The upload payload prepared for one version of an audio file.
#[derive(Debug)]
pub(crate) struct PreparedUpload {
    /// Bytes of the multipart file part, shared by every attempt.
    pub(crate) bytes: std::sync::Arc<Vec<u8>>,
    /// File name advertised in the multipart file part.
    pub(crate) file_name: String,
}

/// Prepare an on-disk audio file for upload: the same result as
/// [`normalize_file_for_upload`], with three differences in how the
/// work is done.
///
/// - **Off the async runtime**: the decode/encode runs on tokio's
///   blocking pool, so a long file never stalls the worker that also
///   drives timers, sockets and the UI feedback of the caller.
/// - **Once per file version**: the result is memoised per source
///   version (see [`UploadPreparations`]), and concurrent callers for
///   the same version share one preparation.  A fallback chain, or a
///   picker asking several models, prepares the file once instead of
///   once per provider.  Replacing or editing the file changes its
///   version, so the next call prepares the new content.
/// - **No re-encode of talk-rs's own dictation recordings**: a file
///   directly in the recordings cache directory whose stream proves it
///   was written by talk-rs at exactly the upload profile (see
///   [`is_own_upload_profile_recording`]) is uploaded as is.  Location
///   or extension alone prove nothing: any other file is normalized,
///   including recordings written before the encoder settings were
///   recorded in the stream.
///
/// `cancel`, when it fires, abandons the wait (the caller gets a
/// "cancelled" error); a preparation another caller shares continues
/// on the blocking pool and stays memoised.
pub(crate) async fn prepare_file_upload(
    path: &Path,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<std::sync::Arc<PreparedUpload>, TalkError> {
    static PROCESS: std::sync::OnceLock<UploadPreparations> = std::sync::OnceLock::new();
    PROCESS
        .get_or_init(|| UploadPreparations::new(PREPARED_UPLOADS_KEPT))
        .prepare(path, cancel, prepare_upload_blocking)
        .await
}

/// Number of prepared uploads kept for reuse: a chain or a picker
/// works on one recording at a time; the bound keeps a long-lived
/// process (picker, recordings browser) from accumulating payloads.
const PREPARED_UPLOADS_KEPT: usize = 4;
const PREPARED_UPLOADS_BYTES: usize = 32 * 1024 * 1024;

type PreparedSlot = std::sync::Arc<tokio::sync::OnceCell<std::sync::Arc<PreparedUpload>>>;

/// Identity of one version of a source file: its resolved path, inode
/// and the size and timestamps any rewrite changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceVersion {
    path: PathBuf,
    input_path: PathBuf,
    direct_file: bool,
    dev: u64,
    ino: u64,
    pub(crate) len: u64,
    pub(crate) mtime: (i64, i64),
    ctime: (i64, i64),
}

impl SourceVersion {
    pub(crate) fn cache_identity(&self) -> (u64, u64, (i64, i64)) {
        (self.dev, self.ino, self.ctime)
    }

    pub(crate) async fn of(path: &Path) -> Result<Self, TalkError> {
        use std::os::unix::fs::MetadataExt;
        let inspect_error = |e: std::io::Error| {
            if e.kind() == std::io::ErrorKind::NotFound {
                TalkError::Transcription(format!("Audio file not found: {}", path.display()))
            } else {
                TalkError::Transcription(format!(
                    "Failed to inspect audio file {}: {e}",
                    path.display()
                ))
            }
        };
        let canonical = tokio::fs::canonicalize(path).await.map_err(inspect_error)?;
        let meta = tokio::fs::metadata(&canonical)
            .await
            .map_err(inspect_error)?;
        let direct_file = !tokio::fs::symlink_metadata(path)
            .await
            .map_err(inspect_error)?
            .file_type()
            .is_symlink();
        let current = tokio::fs::metadata(path).await.map_err(inspect_error)?;
        if (
            meta.dev(),
            meta.ino(),
            meta.len(),
            meta.mtime(),
            meta.mtime_nsec(),
            meta.ctime(),
            meta.ctime_nsec(),
        ) != (
            current.dev(),
            current.ino(),
            current.len(),
            current.mtime(),
            current.mtime_nsec(),
            current.ctime(),
            current.ctime_nsec(),
        ) {
            return Err(TalkError::Transcription(
                "audio file changed while inspecting it".into(),
            ));
        }
        Ok(Self {
            path: canonical,
            input_path: path.to_path_buf(),
            direct_file,
            dev: meta.dev(),
            ino: meta.ino(),
            len: meta.len(),
            mtime: (meta.mtime(), meta.mtime_nsec()),
            ctime: (meta.ctime(), meta.ctime_nsec()),
        })
    }
}

/// Upload preparations memoised per source version, most recent last,
/// at most `kept` of them.
pub(crate) struct UploadPreparations {
    kept: usize,
    slots: std::sync::Arc<std::sync::Mutex<Vec<(SourceVersion, PreparedSlot)>>>,
    active: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>>,
    #[cfg(test)]
    shared_attachments: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    completed_maintenance: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl UploadPreparations {
    pub(crate) fn new(kept: usize) -> Self {
        Self {
            kept,
            slots: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            active: std::sync::OnceLock::new(),
            #[cfg(test)]
            shared_attachments: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            completed_maintenance: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// The prepared upload of `path`'s current version, produced by
    /// `prepare` on the blocking pool the first time it is asked for
    /// (see [`prepare_file_upload`]).  A failed preparation is not
    /// memoised: the next caller tries again.
    pub(crate) async fn prepare(
        &self,
        path: &Path,
        cancel: &tokio_util::sync::CancellationToken,
        prepare: fn(&Path) -> Result<PreparedUpload, TalkError>,
    ) -> Result<std::sync::Arc<PreparedUpload>, TalkError> {
        for _ in 0..3 {
            let version = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(TalkError::Transcription("cancelled by caller".into())),
                version = SourceVersion::of(path) => version?,
            };
            let slot = self.slot(version.clone());
            let slots = self.slots.clone();
            let kept = self.kept;
            #[cfg(test)]
            let maintenance = self.completed_maintenance.clone();
            let path = path.to_path_buf();
            let permit = self
                .active
                .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(4)))
                .clone();
            // The waiter owns initialization independently of its caller.
            let wait = tokio::spawn(async move {
                let prepared = slot
                    .get_or_try_init(|| async move {
                        let _permit = permit.acquire_owned().await.map_err(|e| {
                            TalkError::Transcription(format!(
                                "upload preparation capacity closed: {e}"
                            ))
                        })?;
                        let (prepared, work) = tokio::task::spawn_blocking(move || {
                            crate::perf_counters::measure_thread_work(|| prepare(&path))
                        })
                        .await
                        .map_err(|e| {
                            TalkError::Transcription(format!("upload preparation failed: {e}"))
                        })?;
                        crate::perf_counters::credit_thread_work(work);
                        prepared.map(std::sync::Arc::new)
                    })
                    .await
                    .cloned();
                if prepared.is_ok() {
                    let mut entries = slots.lock().unwrap_or_else(|p| p.into_inner());
                    Self::trim_slots(&mut entries, kept);
                }
                #[cfg(test)]
                maintenance.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                prepared
            });
            let prepared = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(TalkError::Transcription("cancelled by caller".into())),
                prepared = wait => prepared
                    .map_err(|e| TalkError::Transcription(format!("upload preparation failed: {e}")))?,
            };
            let prepared = prepared?;
            let current = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(TalkError::Transcription("cancelled by caller".into())),
                current = SourceVersion::of(&version.input_path) => current?,
            };
            if current == version {
                return Ok(prepared);
            }
            self.remove(&version);
        }
        Err(TalkError::Transcription(
            "audio file changed while it was being prepared".into(),
        ))
    }

    /// The slot of `version`, created if needed.  Older versions of the
    /// same file can never be asked for again and are dropped.
    fn slot(&self, version: SourceVersion) -> PreparedSlot {
        let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        slots.retain(|(v, cell)| {
            v.input_path != version.input_path || *v == version || !cell.initialized()
        });
        if let Some(index) = slots.iter().position(|(v, _)| *v == version) {
            let entry = slots.remove(index);
            let slot = entry.1.clone();
            slots.push(entry);
            #[cfg(test)]
            self.shared_attachments
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return slot;
        }
        let slot = PreparedSlot::default();
        slots.push((version, slot.clone()));
        Self::trim_slots(&mut slots, self.kept);
        slot
    }

    fn remove(&self, version: &SourceVersion) {
        let mut slots = self.slots.lock().unwrap_or_else(|p| p.into_inner());
        slots.retain(|(v, _)| v != version);
    }

    fn trim_slots(slots: &mut Vec<(SourceVersion, PreparedSlot)>, kept: usize) {
        let mut completed = slots.iter().filter(|(_, s)| s.initialized()).count();
        let mut bytes: usize = slots
            .iter()
            .filter_map(|(_, s)| s.get())
            .map(|p| p.bytes.len())
            .sum();
        while completed > kept || bytes > PREPARED_UPLOADS_BYTES {
            let Some(index) = slots.iter().position(|(_, s)| s.initialized()) else {
                break;
            };
            let (_, removed) = slots.remove(index);
            if let Some(payload) = removed.get() {
                bytes -= payload.bytes.len();
            }
            completed -= 1;
        }
    }
}

/// The blocking part of [`prepare_file_upload`].
fn prepare_upload_blocking(path: &Path) -> Result<PreparedUpload, TalkError> {
    let recordings = crate::recording_cache::recordings_dir().ok();
    prepare_upload_from(path, recordings.as_deref())
}

/// The local-profile eligibility fast path for files directly in `recordings` (the
/// dictation cache directory), else [`normalize_file_for_upload`].
fn prepare_upload_from(
    path: &Path,
    recordings: Option<&Path>,
) -> Result<PreparedUpload, TalkError> {
    if path.extension().is_some_and(|ext| ext == "ogg")
        && recordings.is_some_and(|dir| is_directly_in(path, dir))
    {
        if let Ok(bytes) = std::fs::read(path) {
            if is_own_upload_profile_recording(&bytes) {
                log::info!(
                    "upload: {} is a talk-rs recording at the upload profile; sending it as is ({} bytes)",
                    path.display(),
                    bytes.len()
                );
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("audio.ogg")
                    .to_string();
                return Ok(PreparedUpload {
                    bytes: std::sync::Arc::new(bytes),
                    file_name,
                });
            }
        }
    }
    let (bytes, file_name) = normalize_file_for_upload(path)?;
    Ok(PreparedUpload {
        bytes: std::sync::Arc::new(bytes),
        file_name,
    })
}

/// Whether `path` lies directly in `dir` (symlinks resolved on both
/// sides).
fn is_directly_in(path: &Path, dir: &Path) -> bool {
    if std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
        return false;
    }
    let (Ok(dir), Ok(parent)) = (
        dir.canonicalize(),
        path.parent().unwrap_or(Path::new(".")).canonicalize(),
    ) else {
        return false;
    };
    parent == dir
}

/// Whether `bytes` is eligible as one complete Ogg Opus stream at the
/// writer's upload profile (the one
/// [`encode_16k_mono_ogg`] would produce: Voip, 16 kHz, mono, same
/// bitrate), so re-encoding it would only lose quality.
///
/// Checks: a single logical stream; a mono, mapping-family-0
/// `OpusHead` (pre-skip is not checked); `OpusTags` carrying the encoder
/// profile hint (not authentication of the writer);
/// every page intact (CRC-checked by the reader); and an
/// end-of-stream page, so an interrupted recording is still repaired
/// by the normal path.
pub(crate) fn is_own_upload_profile_recording(bytes: &[u8]) -> bool {
    use crate::audio::writer::{encoder_profile, opus_tags_encoder};

    if !ogg_pages_are_contiguous(bytes) {
        return false;
    }
    let upload_profile =
        encoder_profile(opus::Application::Voip, &crate::config::AudioConfig::new());
    let mut reader = ogg::reading::PacketReader::new(std::io::Cursor::new(bytes));
    let packet = |reader: &mut ogg::reading::PacketReader<_>| reader.read_packet().ok().flatten();
    let Some(head) = packet(&mut reader) else {
        return false;
    };
    let serial = head.stream_serial();
    let head_ok = head.first_in_stream()
        && head.data.len() == 19
        && head.data.starts_with(b"OpusHead")
        && head.data[8] == 1
        && head.data[9] == 1
        && head.data[12..16] == 48_000u32.to_le_bytes()
        && head.data[16..18] == 0i16.to_le_bytes()
        && head.data[18] == 0;
    if !head_ok {
        return false;
    }
    let Some(tags) = packet(&mut reader) else {
        return false;
    };
    if tags.stream_serial() != serial || opus_tags_encoder(&tags.data) != Some(&upload_profile) {
        return false;
    }
    loop {
        match reader.read_packet() {
            Ok(Some(p)) if p.stream_serial() != serial => return false,
            Ok(Some(p)) if p.last_in_stream() => {
                // Nothing may follow the end of the stream.
                return matches!(reader.read_packet(), Ok(None));
            }
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => return false,
        }
    }
}

fn ogg_pages_are_contiguous(mut bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    while !bytes.is_empty() {
        if bytes.len() < 27 || &bytes[..4] != b"OggS" || bytes[4] != 0 {
            return false;
        }
        let segments = bytes[26] as usize;
        let Some(laces) = bytes.get(27..27 + segments) else {
            return false;
        };
        let size = 27 + segments + laces.iter().map(|&lace| lace as usize).sum::<usize>();
        let Some(rest) = bytes.get(size..) else {
            return false;
        };
        bytes = rest;
    }
    true
}

/// Per-request wall-clock-timeout policy for one-shot transcription.
///
/// Two distinct call contexts demand opposite defaults:
///
/// - Autonomous pipelines (dictate end-of-recording, `transcribe`
///   CLI) MUST not hang forever — there is no human watching to
///   abort.  These callers pick [`Self::Proportional`].
/// - Interactive callers (the GTK picker row) prefer "wait as long
///   as it takes" — the user can dismiss the picker if a candidate
///   is taking forever.  These callers pick [`Self::UserAttended`].
///
/// The policy is stored on each `OneShotTranscriber` (chosen at
/// construction via `with_policy`) and consulted by `send_once`
/// when it issues the HTTP request.  In both cases the
/// `connect_timeout` from `build_client()` and the kernel-level TCP
/// defences (TCP keepalive, `tcp_user_timeout` on Linux) still
/// fire, so dead connections cannot hang either path indefinitely
/// — only legitimately slow servers benefit from `UserAttended`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestTimeoutPolicy {
    /// Cap each attempt at `proportional_timeout(file_len)` so an
    /// unattended pipeline cannot hang forever.  This is the legacy
    /// behaviour and remains the default of [`MistralOneShotTranscriber::new`]
    /// / [`OpenAIOneShotTranscriber::new`].
    #[default]
    Proportional,
    /// No per-request wall-clock cap; rely solely on
    /// `connect_timeout` plus TCP-level defences.  Used by the
    /// picker so a slow-responding server can take its time without
    /// being killed mid-transcription.  Cancellation is the user's
    /// responsibility (close the picker, or — once wired — a
    /// per-row cancel button).
    UserAttended,
}

impl RequestTimeoutPolicy {
    pub(crate) fn wall_clock(self, audio_bytes: u64) -> Option<std::time::Duration> {
        match self {
            Self::Proportional => Some(transport::http::proportional_timeout(audio_bytes)),
            Self::UserAttended => None,
        }
    }
}

/// Caller-controlled options for [`transcribe_audio`].
///
/// Bundles the orthogonal axes that affect a single transcription
/// call so the function signature stays readable as new axes
/// appear.  Each field is a typed concept that means something
/// specific at the call site:
///
/// - `allow_api`: whether the cache-miss path may hit the network.
///   `false` = cache-only probe (returns [`TalkError::CacheOnly`]
///   on miss); `true` = full pipeline with API call.
/// - `policy`: per-request wall-clock policy.  See
///   [`RequestTimeoutPolicy`].
///
/// Construct via the public fields directly — there is no builder.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// Permit network I/O on cache miss.  `false` makes the call
    /// cache-only (returns [`TalkError::CacheOnly`] on miss);
    /// `true` runs the full pipeline including API.
    pub allow_api: bool,
    /// Per-request wall-clock-timeout policy.  See
    /// [`RequestTimeoutPolicy`] for variant semantics.
    pub policy: RequestTimeoutPolicy,
    /// Cancellation token to thread into the transport's request
    /// loop.  When the token fires, the in-flight HTTP request
    /// aborts within milliseconds.  Defaults to `None` (no
    /// cancellation wired); the picker / streaming orchestrator
    /// supplies a real token when it wants the user to be able to
    /// stop the request.
    pub cancel_token: Option<tokio_util::sync::CancellationToken>,
    /// Skip `acquire_model_lock`/`release_model_lock` (the legacy
    /// per-model lock).  Used by call sites that already register
    /// the job via [`crate::transcription::jobs::register_local`]
    /// (which writes the same lock file with a richer YAML
    /// payload) so we don't conflict-on-self.
    pub skip_legacy_lock: bool,
    pub retry_schedule: Option<transport::RetrySchedule>,
    pub language: Option<String>,
}

pub(crate) mod catalog;
pub mod chain;
pub mod jobs;
pub mod mistral;
pub mod model_suggestions;
pub mod openai;
pub mod openai_realtime;
pub(crate) mod outage;
#[cfg(feature = "parakeet")]
pub mod parakeet;
pub mod realtime;
pub mod transport;

pub use mistral::MistralOneShotTranscriber;
pub use openai::OpenAIOneShotTranscriber;
pub use openai_realtime::OpenAIRealtimeTranscriber;
#[cfg(feature = "parakeet")]
pub use parakeet::ParakeetOneShotTranscriber;
pub use realtime::{MistralRealtimeTranscriber, OrderedItemTranscript, TranscriptionEvent};

/// Result type for transcription operations.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionResult {
    /// The transcribed text from the audio file.
    pub text: String,
    /// Optional metadata extracted from provider responses/headers.
    pub metadata: TranscriptionMetadata,
    /// Speaker-diarized segments, when diarization was requested and
    /// supported by the provider.
    pub diarization: Option<Vec<DiarizationSegment>>,
    /// Time-localized transcript segments, when the provider returned
    /// them.  Independent of diarization — a diarized response may
    /// populate both `diarization` and `segments`.  Used to reconstruct
    /// timelines in downstream consumers such as `activity-memo`.
    pub segments: Option<Vec<TranscriptSegment>>,
}

/// A single speaker-attributed segment from diarization.
///
/// Provider-agnostic: each provider maps its native speaker labels
/// into this structure.
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizationSegment {
    /// Speaker identifier (e.g. `"SPEAKER_00"`).
    pub speaker: String,
    /// Segment start time in seconds.
    pub start: f64,
    /// Segment end time in seconds.
    pub end: f64,
    /// Transcribed text for this segment.
    pub text: String,
}

/// A time-localized transcript segment without speaker attribution.
///
/// Provider-agnostic: each one-shot/realtime backend maps its native
/// segment representation into this shape.  Unlike `DiarizationSegment`,
/// a `TranscriptSegment` does not carry a speaker identifier — it is
/// just a start/end window around a piece of text, suitable for
/// timeline reconstruction and sub-minute granularity in consumers.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptSegment {
    /// Segment start time in seconds, from the beginning of the recording.
    pub start: f64,
    /// Segment end time in seconds, from the beginning of the recording.
    pub end: f64,
    /// Transcribed text for this segment.
    pub text: String,
}

/// Extract generic transcript segments from a raw provider response.
///
/// Accepts a slice of `serde_json::Value` (as returned by both Mistral
/// Voxtral and OpenAI Whisper under `verbose_json`) and pulls out
/// `start`, `end`, and `text` from every segment that has them.
/// Segments with missing timing or empty text are skipped.  Returns
/// `None` when no usable segments survive the filter so that call
/// sites can leave `TranscriptionResult.segments` as `None` rather
/// than `Some(vec![])`.
pub(crate) fn parse_transcript_segments(
    segments: &[serde_json::Value],
) -> Option<Vec<TranscriptSegment>> {
    let mut result = Vec::new();
    for seg in segments {
        let start = seg.get("start").and_then(|v| v.as_f64());
        let end = seg.get("end").and_then(|v| v.as_f64());
        let text = seg.get("text").and_then(|v| v.as_str()).unwrap_or("");
        if let (Some(start), Some(end)) = (start, end) {
            if !text.is_empty() {
                result.push(TranscriptSegment {
                    start,
                    end,
                    text: text.to_string(),
                });
            }
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Format transcription output, using diarized segments when available.
///
/// When diarization segments are present, each line is prefixed with
/// `[SPEAKER_ID]`.  Adjacent segments from the same speaker are merged
/// into a single block.  When no diarization is present, returns the
/// plain transcript text.
///
/// When `timestamp` is true and diarization is present, each line is
/// prefixed with `[HH:MM:SS]` before the speaker label.
pub fn format_transcription_output(result: &TranscriptionResult, timestamp: bool) -> String {
    let Some(ref segments) = result.diarization else {
        return result.text.clone();
    };

    if segments.is_empty() {
        return result.text.clone();
    }

    let mut lines = Vec::new();
    let mut current_speaker: Option<&str> = None;
    let mut current_texts: Vec<&str> = Vec::new();
    let mut current_start: f64 = 0.0;

    for seg in segments {
        if current_speaker == Some(seg.speaker.as_str()) {
            current_texts.push(seg.text.trim());
        } else {
            // Flush previous speaker block
            if let Some(speaker) = current_speaker {
                if timestamp {
                    lines.push(format!(
                        "[{}] {} {}",
                        format_timestamp(current_start),
                        speaker,
                        current_texts.join(" ")
                    ));
                } else {
                    lines.push(format!("[{}] {}", speaker, current_texts.join(" ")));
                }
            }
            current_speaker = Some(&seg.speaker);
            current_start = seg.start;
            current_texts.clear();
            current_texts.push(seg.text.trim());
        }
    }
    // Flush last block
    if let Some(speaker) = current_speaker {
        if timestamp {
            lines.push(format!(
                "[{}] {} {}",
                format_timestamp(current_start),
                speaker,
                current_texts.join(" ")
            ));
        } else {
            lines.push(format!("[{}] {}", speaker, current_texts.join(" ")));
        }
    }

    lines.join("\n")
}

/// Format seconds as HH:MM:SS.
fn format_timestamp(seconds: f64) -> String {
    let total_secs = seconds as u64;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, secs)
}

/// Provider-agnostic metadata that can be written to YAML.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionMetadata {
    /// Chain attempts in order; empty for legacy single-model calls.
    pub attempts: Vec<chain::Attempt>,
    /// End-to-end API call latency measured client-side.
    pub request_latency_ms: Option<u64>,
    /// End-to-end realtime session duration measured client-side.
    pub session_elapsed_ms: Option<u64>,
    /// Request identifier from provider response headers.
    pub request_id: Option<String>,
    /// Provider-side processing duration in milliseconds when available.
    pub provider_processing_ms: Option<u64>,
    /// Detected language code if returned by provider.
    pub detected_language: Option<String>,
    /// Audio duration reported by provider usage/response.
    pub audio_seconds: Option<f64>,
    /// Number of transcript segments returned by provider.
    pub segment_count: Option<usize>,
    /// Number of word-level timestamps returned by provider.
    pub word_count: Option<usize>,
    /// Token usage summary when available.
    pub token_usage: Option<TokenUsage>,
    /// Provider-specific payload for advanced diagnostics.
    pub provider_specific: Option<ProviderSpecificMetadata>,
}

/// Common token usage summary.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Provider-specific metadata captured from API responses.
#[derive(Debug, Clone)]
pub enum ProviderSpecificMetadata {
    OpenAI(OpenAIProviderMetadata),
    Mistral(MistralProviderMetadata),
}

/// OpenAI-specific metadata.
#[derive(Debug, Clone, Default)]
pub struct OpenAIProviderMetadata {
    pub model: Option<String>,
    pub usage_raw: Option<serde_json::Value>,
    pub rate_limit_headers: BTreeMap<String, String>,
    pub unknown_event_types: Vec<String>,
    pub realtime: Option<OpenAIRealtimeMetadata>,
}

/// OpenAI realtime-specific metadata.
#[derive(Debug, Clone, Default)]
pub struct OpenAIRealtimeMetadata {
    pub session_id: Option<String>,
    pub conversation_id: Option<String>,
    pub event_counts: BTreeMap<String, u64>,
    pub last_rate_limits: Option<serde_json::Value>,
    pub ws_upgrade_headers: BTreeMap<String, String>,
}

/// Mistral-specific metadata.
#[derive(Debug, Clone, Default)]
pub struct MistralProviderMetadata {
    pub model: Option<String>,
    pub usage_raw: Option<serde_json::Value>,
    pub unknown_event_types: Vec<String>,
}

// ── One-shot trait ──────────────────────────────────────────────────

/// One-shot transcription: file or byte-stream in, full text out.
///
/// Implementations should handle file I/O, API communication, and error
/// handling.  All implementations must be `Send + Sync` for use in async
/// contexts.
#[async_trait]
pub(crate) trait OneShotTranscriber: Send + Sync {
    /// Pre-flight check: verify API connectivity and model validity.
    ///
    /// Called before starting audio capture so the user gets immediate
    /// feedback when a provider is misconfigured or a model name is
    /// invalid.  Implementations should make a lightweight API call
    /// (e.g. list available models) and return a helpful error with
    /// available alternatives on failure.
    async fn validate(&self) -> Result<(), TalkError>;

    /// Fetch a transcription from either a file or an encoded stream.
    async fn fetch_transcription(
        &self,
        body: TranscriptionBody,
    ) -> Result<TranscriptionResult, TalkError>;

    /// Inject a telemetry sink for event emission during HTTP calls.
    ///
    /// The default implementation is a no-op — override in concrete
    /// types that support telemetry.  Called by the orchestrator
    /// (`dictate/mod.rs`) right after construction, before any
    /// `fetch_transcription` is invoked.
    fn set_sink(&mut self, _sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink>) {}

    /// Inject a cancellation token for the transcription request.
    ///
    /// When the token fires, the in-flight ``http_request`` aborts
    /// within milliseconds.  Used by the picker Stop button and by
    /// SIGUSR1-routed cross-process cancellation via
    /// [`crate::transcription::jobs::cancel_remote`].
    fn set_cancel_token(&mut self, _token: tokio_util::sync::CancellationToken) {}
    fn set_retry_schedule(&mut self, _schedule: transport::RetrySchedule) {}
}

// ── Realtime trait ───────────────────────────────────────────────────

/// Realtime transcription: raw PCM stream in, incremental events out.
//
// The trait is driven exclusively by the live dictation path, which
// requires both `capture` and `ui`.  In a headless build the trait
// methods are never invoked (the concrete realtime transcribers still
// expose inherent `transcribe_realtime` methods for library
// consumers), so the dead-code allow is scoped to that configuration.
#[cfg_attr(not(all(feature = "capture", feature = "ui")), allow(dead_code))]
#[async_trait]
pub(crate) trait RealtimeTranscriber: Send + Sync {
    /// Pre-flight check: verify API connectivity and model validity.
    ///
    /// Same purpose as [`OneShotTranscriber::validate`]: called by
    /// `dictate_realtime` right after the transcriber is created and
    /// before the streaming session is opened, so a bad key or model
    /// surfaces as an immediate, enriched error rather than from inside
    /// the streaming loop.  Capture has already started at that point
    /// (audio is buffered meanwhile); the check gates the provider
    /// session, not the microphone.
    async fn validate(&self) -> Result<(), TalkError>;

    /// Connect and start streaming.  Returns a channel of events.
    async fn transcribe_realtime(
        &self,
        audio_rx: tokio::sync::mpsc::Receiver<Vec<i16>>,
    ) -> Result<tokio::sync::mpsc::Receiver<TranscriptionEvent>, TalkError>;

    /// Inject a telemetry sink for event emission during the WS
    /// upgrade phase (and, later, the in-session frame events).
    fn set_sink(&mut self, _sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink>) {}

    /// Inject a cancellation token for the realtime session.
    ///
    /// When the token fires, the in-flight WebSocket upgrade or
    /// active session aborts.  See [`OneShotTranscriber::set_cancel_token`]
    /// for the wiring rationale.
    fn set_cancel_token(&mut self, _token: tokio_util::sync::CancellationToken) {}
    fn set_retry_schedule(&mut self, _schedule: transport::RetrySchedule) {}
}

// ── Error detection / enrichment dispatchers ────────────────────────

/// Check if an error is a model-not-found error for the given provider.
///
/// Detection order:
///
/// 1. **Structural fast path** — if the error is a
///    [`TalkError::Pipeline`] carrying a
///    [`crate::error::PipelineFailureKind::ModelRejected`], return
///    `true` regardless of provider.  Producers (validate-path
///    HTTP code) emit this variant directly, so consumers can
///    detect model-rejection without parsing strings.
/// 2. **Legacy string-matching fallback** — for any pre-migration
///    call sites that still surface model-not-found as
///    `TalkError::Config(String)` / `TalkError::Transcription(String)`,
///    dispatch to the provider's own detection logic.
///
/// The structural path runs first so a future refactor cannot
/// accidentally regress retry/bail behaviour by skipping the
/// legacy string match — structural producers always win.
pub fn is_model_error(provider: Provider, error: &TalkError) -> bool {
    use crate::error::PipelineFailureKind;
    if let TalkError::Pipeline(pf) = error {
        if matches!(pf.kind, PipelineFailureKind::ModelRejected { .. }) {
            return true;
        }
    }
    match provider {
        Provider::Mistral => mistral::is_model_error(error),
        Provider::OpenAI => openai::is_model_error(error),
        // Parakeet is a local backend — no remote model-rejection concept.
        Provider::Parakeet => false,
    }
}

/// Enrich a model error with available transcription model suggestions.
///
/// Dispatches to the provider module's own enrichment logic.  Returns
/// the error unchanged if it is not a model error or if the provider
/// section is not configured.
pub async fn enrich_model_error(
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    error: TalkError,
) -> TalkError {
    match provider {
        Provider::Mistral => {
            let Some(ref cfg) = config.providers.mistral else {
                return error;
            };
            let model_name = model.unwrap_or(&cfg.model);
            let api_base = cfg.url.as_deref().unwrap_or(mistral::API_BASE);
            mistral::enrich_model_error(error, &cfg.api_key, model_name, api_base).await
        }
        Provider::OpenAI => {
            let Some(ref cfg) = config.providers.openai else {
                return error;
            };
            let model_name = model.unwrap_or(&cfg.model);
            let api_base = cfg.url.as_deref().unwrap_or(openai::API_BASE);
            openai::enrich_model_error(error, &cfg.api_key, model_name, api_base).await
        }
        // Parakeet is local; no API model catalog to enrich against —
        // pass the error through unchanged.
        Provider::Parakeet => error,
    }
}

// ── Factory ──────────────────────────────────────────────────────────

/// Create a one-shot transcriber for the given provider.
///
/// When `model` is `Some`, it overrides the config default for that
/// provider (the `--model` CLI flag).  When `diarize` is `true`, the
/// transcriber will request speaker diarization (if supported by the
/// provider).  The `policy` is stored on the transcriber and
/// consulted by its `send_once` implementation to decide whether to
/// attach a per-request wall-clock timeout — see
/// [`RequestTimeoutPolicy`] for variant semantics.
pub(crate) fn create_oneshot_transcriber(
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    diarize: bool,
    policy: RequestTimeoutPolicy,
) -> Result<Box<dyn OneShotTranscriber>, TalkError> {
    create_oneshot_transcriber_with_language(config, provider, model, diarize, policy, None)
}

fn create_oneshot_transcriber_with_language(
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    diarize: bool,
    policy: RequestTimeoutPolicy,
    language: Option<&str>,
) -> Result<Box<dyn OneShotTranscriber>, TalkError> {
    match provider {
        Provider::Mistral => {
            let mut cfg = config.providers.mistral.clone().ok_or_else(|| {
                TalkError::Config(
                    "Mistral provider selected but providers.mistral is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.mistral.api_key is required".to_string(),
                ));
            }
            if let Some(m) = model {
                cfg.model = m.to_string();
            }
            Ok(Box::new(MistralOneShotTranscriber::with_policy(
                cfg, diarize, policy,
            )?))
        }
        Provider::OpenAI => {
            let cfg = config.providers.openai.as_ref().ok_or_else(|| {
                TalkError::Config(
                    "OpenAI provider selected but providers.openai is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.openai.api_key is required".to_string(),
                ));
            }
            let cfg = override_openai_batch_model(cfg, model);
            let mut cfg = cfg;
            if let Some(language) = language {
                cfg.languages = Some(vec![language.to_string()]);
            }
            Ok(Box::new(OpenAIOneShotTranscriber::with_policy(
                cfg, policy,
            )?))
        }
        // Phase 3: local Parakeet backend.  Construction is cheap
        // (path resolution only); the heavy model load happens on
        // first `fetch_transcription`.  `validate` only CHECKS model
        // presence (model::ensure_present) — it never downloads.  The
        // download is an explicit, consented step driven by each entry
        // surface (CLI / toggle / picker) before transcription begins.
        #[cfg(feature = "parakeet")]
        Provider::Parakeet => {
            let mut cfg = config.providers.parakeet.clone().unwrap_or_default();
            if let Some(m) = model {
                cfg.model = Some(m.to_string());
            }
            Ok(Box::new(ParakeetOneShotTranscriber::with_policy(
                cfg, policy,
            )?))
        }
        #[cfg(not(feature = "parakeet"))]
        Provider::Parakeet => Err(TalkError::Config(
            "talk-rs was built without the 'parakeet' feature; rebuild without \
             --no-default-features to enable the local Parakeet backend"
                .to_string(),
        )),
    }
}

fn override_openai_batch_model(
    config: &crate::config::OpenAIConfig,
    model: Option<&str>,
) -> crate::config::OpenAIConfig {
    let mut config = config.clone();
    if let Some(model) = model {
        config.model = model.to_string();
    }
    config
}

/// Read the transcript for a recording from the cache, falling
/// through all layers without making any API call.
///
/// Waterfall (in priority order):
///
/// 1. **Pick file** (``<stem>.pick.yml``) -- the authoritative
///    transcript, possibly user-edited.  Returned if present.
/// 2. **Default-provider / default-model sidecar** -- the
///    one-shot-mode cache for the default transcription model.
///    Read synchronously, no API call.
/// 3. **None** -- no transcript cached.
///
/// Used by consumers that want to display a transcript if one is
/// cheaply available but MUST NOT trigger network I/O.  The record
/// UI, for example, uses this to decide between "show transcript"
/// and "show audio player".
///
/// This function is synchronous because every step is a local
/// filesystem read.  It never calls the transcription API.
pub fn read_cached_transcript(audio_path: &std::path::Path, config: &Config) -> Option<String> {
    use crate::recording_cache::{get_transcript, TranscriptStatus};

    // Layer 1: pick file (authoritative).
    match get_transcript(audio_path) {
        TranscriptStatus::Available(text) => return Some(text),
        // In-progress or unavailable: fall through to sidecar probe.
        TranscriptStatus::InProgress | TranscriptStatus::NotAvailable => {}
    }

    default_sidecar_transcript(audio_path, config)
}

/// Resolve lock, pick, then default-model sidecar without rereading the pick.
pub(crate) fn cached_transcript_status(
    audio_path: &std::path::Path,
    config: &Config,
) -> crate::recording_cache::TranscriptStatus {
    use crate::recording_cache::{get_transcript, TranscriptStatus};
    match get_transcript(audio_path) {
        TranscriptStatus::NotAvailable => default_sidecar_transcript(audio_path, config)
            .map(TranscriptStatus::Available)
            .unwrap_or(TranscriptStatus::NotAvailable),
        status => status,
    }
}

fn default_sidecar_transcript(audio_path: &std::path::Path, config: &Config) -> Option<String> {
    let provider = config
        .transcription
        .as_ref()
        .map(|t| t.default_provider)
        .unwrap_or(Provider::Mistral);
    let effective_model = resolve_effective_model(config, provider, None);
    crate::recording_cache::TranscriptionCache::get(audio_path, provider, &effective_model)
        .map(|r| r.text)
}

pub(crate) fn is_default_cached_sidecar_for(
    audio_path: &std::path::Path,
    sidecar: &std::path::Path,
    config: &Config,
) -> bool {
    let provider = config
        .transcription
        .as_ref()
        .map(|t| t.default_provider)
        .unwrap_or(Provider::Mistral);
    let model = resolve_effective_model(config, provider, None);
    crate::recording_cache::TranscriptionCache::is_sidecar_for(
        audio_path, sidecar, provider, &model,
    )
}

/// Produce the authoritative transcript for a recording (Layer 2).
///
/// 1. Calls [`crate::recording_cache::get_transcript`]:
///    - [`crate::recording_cache::TranscriptStatus::Available`] -> returns the text.
///    - [`crate::recording_cache::TranscriptStatus::InProgress`] -> returns
///      [`TalkError::TranscriptInProgress`].
///    - [`crate::recording_cache::TranscriptStatus::NotAvailable`] -> continues.
/// 2. Acquires the pick lock.
/// 3. Calls [`transcribe_audio`] with `allow_api = true`.
/// 4. Writes the pick file with the resulting text.
/// 5. Releases the pick lock.
///
/// The pick lock is always released even on error paths.
pub async fn produce_transcript(
    audio_path: &std::path::Path,
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    sink: &std::sync::Arc<dyn crate::telemetry::TelemetrySink>,
) -> Result<String, TalkError> {
    use crate::recording_cache::{self, TranscriptStatus};

    match recording_cache::get_transcript(audio_path) {
        TranscriptStatus::Available(text) => return Ok(text),
        TranscriptStatus::InProgress => return Err(TalkError::TranscriptInProgress),
        TranscriptStatus::NotAvailable => {}
    }

    recording_cache::acquire_pick_lock(audio_path)?;

    let effective_model = resolve_effective_model(config, provider, model);

    // `produce_transcript` is called from autonomous backends
    // (recording cache layer 2 producer) — pick `Proportional` so a
    // hung server cannot wedge the producer indefinitely.
    let result = transcribe_audio(
        audio_path,
        config,
        provider,
        model,
        false,
        TranscribeOptions {
            allow_api: true,
            policy: RequestTimeoutPolicy::Proportional,
            cancel_token: None,
            skip_legacy_lock: false,
            retry_schedule: None,
            language: None,
        },
        sink,
    )
    .await;

    let final_result = match result {
        Ok(r) => {
            let text = r.text.trim().to_string();
            if let Err(e) = recording_cache::write_pick(
                audio_path,
                &provider.to_string(),
                &effective_model,
                false,
                &text,
            ) {
                log::warn!("failed to write pick file: {}", e);
            }
            Ok(text)
        }
        Err(e) => Err(e),
    };

    if let Err(e) = recording_cache::release_pick_lock(audio_path) {
        log::warn!("failed to release pick lock: {}", e);
    }

    final_result
}

/// Transcribe an audio file.
///
/// THE single transcription entry point for one-shot-from-file
/// transcription (Layer 3).  Checks the sidecar cache first; on
/// miss and when `allow_api` is true, acquires a per-model lock,
/// calls the provider API, stores the sidecar, releases the lock.
///
/// Callers never see cache or lock files.
///
/// # Errors
///
/// - [`TalkError::CacheOnly`]: sidecar miss and `allow_api = false`.
/// - [`TalkError::ModelInProgress`]: per-model lock already held by
///   another producer.
pub async fn transcribe_audio(
    audio_path: &Path,
    config: &Config,
    provider: Provider,
    model: Option<&str>,
    diarize: bool,
    options: TranscribeOptions,
    sink: &std::sync::Arc<dyn crate::telemetry::TelemetrySink>,
) -> Result<TranscriptionResult, TalkError> {
    let TranscribeOptions {
        allow_api,
        policy,
        cancel_token,
        skip_legacy_lock,
        retry_schedule,
        language,
    } = options;
    use crate::recording_cache::{self, TranscriptionCache};

    let effective_model = resolve_effective_model(config, provider, model);

    if !diarize {
        if let Some(cached) = TranscriptionCache::get(audio_path, provider, &effective_model) {
            log::info!(
                "transcription cache hit for {}:{} on {}",
                provider,
                effective_model,
                audio_path.display()
            );
            return Ok(cached);
        }
    }

    if !allow_api {
        log::debug!(
            "transcription cache miss for {}:{} on {} — API call forbidden",
            provider,
            effective_model,
            audio_path.display()
        );
        return Err(TalkError::CacheOnly);
    }

    // Capture source identity before taking the model lock; an inspection
    // error must not leave that lock behind.
    let upload_source = if provider.is_local() {
        None
    } else {
        Some(SourceVersion::of(audio_path).await?)
    };

    // Acquire per-model lock before calling the API, unless the
    // caller has already registered via `jobs::register_local`
    // (which writes the same lock with a richer payload).
    if !skip_legacy_lock {
        recording_cache::acquire_model_lock(audio_path, provider, &effective_model, false)?;
    }

    log::info!(
        "transcription cache miss for {}:{} on {} — calling API",
        provider,
        effective_model,
        audio_path.display()
    );

    // Wrap API call in a closure so we can always release the lock.
    let api_result = async {
        let mut transcriber = create_oneshot_transcriber_with_language(
            config,
            provider,
            model,
            diarize,
            policy,
            language.as_deref(),
        )?;
        transcriber.set_sink(sink.clone());
        if let Some(schedule) = retry_schedule {
            transcriber.set_retry_schedule(schedule);
        }
        if let Some(token) = cancel_token {
            transcriber.set_cancel_token(token);
        }
        transcriber.validate().await?;
        transcriber
            .fetch_transcription(TranscriptionBody::File(audio_path.to_path_buf()))
            .await
    }
    .await;

    // Lock release is skipped when the caller owns the lock via
    // `jobs::register_local` (the `LocalJob` Drop removes the
    // file).  Otherwise release here.
    let release_lock = |context: &str| {
        if skip_legacy_lock {
            return;
        }
        if let Err(e) =
            recording_cache::release_model_lock(audio_path, provider, &effective_model, false)
        {
            log::warn!("failed to release model lock {}: {}", context, e);
        }
    };

    match api_result {
        Ok(result) => {
            if let Some(source) = &upload_source {
                if !matches!(SourceVersion::of(audio_path).await, Ok(current) if current == *source)
                {
                    release_lock("after source change");
                    return Err(TalkError::Transcription(
                        "audio file changed while transcription was in progress".into(),
                    ));
                }
            }
            let stored = if let Some(source) = &upload_source {
                TranscriptionCache::store_verified(
                    audio_path,
                    provider,
                    &effective_model,
                    false,
                    &result,
                    source,
                )
                .await
            } else {
                TranscriptionCache::store(audio_path, provider, &effective_model, false, &result)
            };
            if let Err(e) = stored {
                if matches!(&e, TalkError::Transcription(message) if message.starts_with("audio file changed"))
                {
                    release_lock("after source change during publication");
                    return Err(e);
                }
                log::warn!("failed to cache transcription result: {}", e);
            }
            release_lock("after success");
            Ok(result)
        }
        Err(e) => {
            release_lock("after error");
            Err(e)
        }
    }
}

/// Resolve the effective model name from CLI override or config default.
fn resolve_effective_model(config: &Config, provider: Provider, model: Option<&str>) -> String {
    if let Some(m) = model {
        return m.to_string();
    }
    match provider {
        Provider::Mistral => config
            .providers
            .mistral
            .as_ref()
            .map(|c| c.model.clone())
            .unwrap_or_else(|| "voxtral-mini-latest".to_string()),
        Provider::OpenAI => config
            .providers
            .openai
            .as_ref()
            .map(|c| c.model.clone())
            .unwrap_or_else(|| "gpt-transcribe".to_string()),
        Provider::Parakeet => config
            .providers
            .parakeet
            .as_ref()
            .map(|c| c.resolved_model_name())
            .unwrap_or_else(|| "parakeet-tdt-0.6b-v3-int8".to_string()),
    }
}

/// Create a realtime transcriber for the given provider.
///
/// When `model` is `Some`, it overrides the config default for that
/// provider's realtime model (the `--model` CLI flag).
//
// Only the live dictation path (`dictate` / picker) constructs a
// realtime transcriber; that path requires both `capture` and `ui`.
// In a headless build it is never called, but the factory + the
// `RealtimeTranscriber` trait remain part of the exposed realtime
// API surface, so the allow is scoped to the headless configuration.
#[cfg_attr(not(all(feature = "capture", feature = "ui")), allow(dead_code))]
pub(crate) fn create_realtime_transcriber(
    config: &Config,
    provider: Provider,
    model: Option<&str>,
) -> Result<Box<dyn RealtimeTranscriber>, TalkError> {
    match provider {
        Provider::Mistral => {
            let cfg = config.providers.mistral.clone().ok_or_else(|| {
                TalkError::Config(
                    "Mistral provider selected but providers.mistral is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.mistral.api_key is required".to_string(),
                ));
            }
            let mut transcriber = MistralRealtimeTranscriber::new(cfg);
            if let Some(model) = model {
                transcriber.set_model(model.to_string());
            }
            Ok(Box::new(transcriber))
        }
        Provider::OpenAI => {
            let mut cfg = config.providers.openai.clone().ok_or_else(|| {
                TalkError::Config(
                    "OpenAI provider selected but providers.openai is not configured".to_string(),
                )
            })?;
            if cfg.api_key.is_empty() {
                return Err(TalkError::Config(
                    "providers.openai.api_key is required".to_string(),
                ));
            }
            if let Some(m) = model {
                cfg.realtime_model = m.to_string();
            }
            Ok(Box::new(OpenAIRealtimeTranscriber::new(cfg)))
        }
        // Parakeet is one-shot-only; reject realtime selection cleanly.
        Provider::Parakeet => Err(TalkError::Config(
            "parakeet provider has no realtime mode; use one-shot transcription (omit --realtime)"
                .to_string(),
        )),
    }
}

// ── Mock ─────────────────────────────────────────────────────────────

/// Mock one-shot transcriber for testing.
///
/// Returns a hardcoded transcription result without making any API calls.
pub struct MockOneShotTranscriber {
    /// The text to return when transcribe is called.
    pub response_text: String,
}

impl MockOneShotTranscriber {
    /// Create a new mock transcriber with the given response text.
    pub fn new(response_text: impl Into<String>) -> Self {
        Self {
            response_text: response_text.into(),
        }
    }
}

#[async_trait]
impl OneShotTranscriber for MockOneShotTranscriber {
    async fn validate(&self) -> Result<(), TalkError> {
        Ok(())
    }

    async fn fetch_transcription(
        &self,
        body: TranscriptionBody,
    ) -> Result<TranscriptionResult, TalkError> {
        if let TranscriptionBody::Pipe { mut chunks, .. } = body {
            while chunks.recv().await.is_some() {}
        }

        Ok(TranscriptionResult {
            text: self.response_text.clone(),
            metadata: TranscriptionMetadata::default(),
            diarization: None,
            segments: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute path to a shipped test fixture.
    fn fixture(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests");
        p.push("fixtures");
        p.push(name);
        p
    }

    /// Read the channel count + input sample rate out of an OGG/Opus
    /// byte stream's OpusHead packet (RFC 7845).  Returns
    /// `(channels, declared_rate)`.  Note: Opus always declares 48000
    /// in OpusHead regardless of the encoder's internal rate, so the
    /// meaningful assertion for "16 kHz mono" is the channel count
    /// here plus the decoder rate verified separately.
    fn opus_head_channels(ogg: &[u8]) -> u8 {
        // Find "OpusHead" magic; channel count is the byte right after
        // the 8-magic + 1 version bytes = offset+9.
        let pos = ogg
            .windows(8)
            .position(|w| w == b"OpusHead")
            .expect("OpusHead present");
        ogg[pos + 9]
    }

    #[test]
    fn test_normalize_stereo_44k_m4a_to_mono_ogg() {
        // The stereo 44.1 kHz AAC fixture must come out as a mono OGG.
        let path = fixture("sine_440_0.5s_stereo.m4a");
        assert!(path.exists(), "fixture missing: {}", path.display());

        let (bytes, name) = normalize_file_for_upload(&path).expect("normalization should succeed");

        // Filename advertises ogg.
        assert!(name.ends_with(".ogg"), "expected .ogg name, got {name}");
        // Valid OGG/Opus stream, mono.
        assert_eq!(&bytes[0..4], b"OggS", "should be an OGG stream");
        assert_eq!(opus_head_channels(&bytes), 1, "must be downmixed to mono");

        // Decode it back as 16 kHz mono — proves the encoder produced a
        // stream a 16 kHz mono Opus decoder accepts and that it carries
        // real audio (non-silent).
        let mut decoder = opus::Decoder::new(16_000, opus::Channels::Mono)
            .expect("decoder creation should succeed");
        // Pull the first audio packet out of the OGG container.
        let mut packet_reader =
            ogg::reading::PacketReader::new(std::io::Cursor::new(bytes.clone()));
        // Skip the two header packets (OpusHead, OpusTags).
        let _ = packet_reader.read_packet_expected().expect("OpusHead");
        let _ = packet_reader.read_packet_expected().expect("OpusTags");
        let first_audio = packet_reader
            .read_packet_expected()
            .expect("at least one audio packet");
        let mut out = vec![0i16; 16_000]; // up to 1s
        let decoded = decoder
            .decode(&first_audio.data, &mut out, false)
            .expect("decode should succeed");
        assert!(decoded > 0, "decoded frame should be non-empty");
    }

    #[test]
    fn test_normalize_mono_m4a_stays_mono_ogg() {
        let path = fixture("sine_440_0.5s_mono.m4a");
        assert!(path.exists(), "fixture missing: {}", path.display());

        let (bytes, name) = normalize_file_for_upload(&path).expect("normalization should succeed");
        assert!(name.ends_with(".ogg"));
        assert_eq!(&bytes[0..4], b"OggS");
        assert_eq!(opus_head_channels(&bytes), 1);
    }

    #[test]
    fn test_normalize_missing_file_errors() {
        let err = normalize_file_for_upload(&PathBuf::from("/nonexistent/audio.m4a"))
            .expect_err("missing file should error");
        assert!(
            err.to_string().contains("not found"),
            "expected not-found error, got: {err}"
        );
    }

    #[test]
    fn test_normalize_undecodable_falls_back_to_raw_bytes() {
        // A file with an unsupported extension cannot be decoded; the
        // normalizer must fall back to uploading the raw bytes rather
        // than failing (never break a previously-working upload).
        let dir = tempfile::TempDir::new().expect("tmp dir");
        let path = dir.path().join("weird.bin");
        std::fs::write(&path, b"not really audio").expect("write");

        let (bytes, name) = normalize_file_for_upload(&path).expect("fallback should succeed");
        assert_eq!(bytes, b"not really audio");
        assert_eq!(name, "weird.bin", "fallback keeps original file name");
    }

    #[tokio::test]
    async fn test_mock_transcriber_returns_text() {
        let mock = MockOneShotTranscriber::new("Hello, world!");
        let result = mock
            .fetch_transcription(TranscriptionBody::File(PathBuf::from("/tmp/test.wav")))
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "Hello, world!");
    }

    #[tokio::test]
    async fn test_mock_transcriber_stream() {
        let mock = MockOneShotTranscriber::new("Streamed transcription");
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        // Send some fake audio data
        tx.send(vec![0u8; 100]).await.unwrap();
        tx.send(vec![1u8; 200]).await.unwrap();
        drop(tx); // Close the channel

        let result = mock
            .fetch_transcription(TranscriptionBody::Pipe {
                chunks: rx,
                file_name: "test.wav".to_string(),
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "Streamed transcription");
    }

    #[tokio::test]
    async fn test_mock_transcriber_ignores_path() {
        let mock = MockOneShotTranscriber::new("Fixed response");
        let result1 = mock
            .fetch_transcription(TranscriptionBody::File(PathBuf::from("/path/one.wav")))
            .await;
        let result2 = mock
            .fetch_transcription(TranscriptionBody::File(PathBuf::from("/path/two.wav")))
            .await;

        assert_eq!(result1.unwrap().text, "Fixed response");
        assert_eq!(result2.unwrap().text, "Fixed response");
    }

    #[test]
    fn test_provider_from_str() {
        assert_eq!("mistral".parse::<Provider>().unwrap(), Provider::Mistral);
        assert_eq!("openai".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert_eq!("OpenAI".parse::<Provider>().unwrap(), Provider::OpenAI);
        assert!("unknown".parse::<Provider>().is_err());
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(Provider::Mistral.to_string(), "mistral");
        assert_eq!(Provider::OpenAI.to_string(), "openai");
    }

    #[test]
    fn test_format_plain_text_without_diarization() {
        let result = TranscriptionResult {
            text: "Hello world.".to_string(),
            metadata: Default::default(),
            diarization: None,
            segments: None,
        };
        assert_eq!(format_transcription_output(&result, false), "Hello world.");
    }

    #[test]
    fn test_format_diarized_output() {
        let result = TranscriptionResult {
            text: "Hello. I am fine.".to_string(),
            metadata: Default::default(),
            diarization: Some(vec![
                DiarizationSegment {
                    speaker: "SPEAKER_00".to_string(),
                    start: 0.0,
                    end: 1.5,
                    text: "Hello.".to_string(),
                },
                DiarizationSegment {
                    speaker: "SPEAKER_01".to_string(),
                    start: 1.5,
                    end: 3.0,
                    text: "I am fine.".to_string(),
                },
            ]),
            segments: None,
        };
        assert_eq!(
            format_transcription_output(&result, false),
            "[SPEAKER_00] Hello.\n[SPEAKER_01] I am fine."
        );
    }

    #[test]
    fn test_format_diarized_merges_same_speaker() {
        let result = TranscriptionResult {
            text: "Hello. How are you? I am fine.".to_string(),
            metadata: Default::default(),
            diarization: Some(vec![
                DiarizationSegment {
                    speaker: "SPEAKER_00".to_string(),
                    start: 0.0,
                    end: 1.0,
                    text: "Hello.".to_string(),
                },
                DiarizationSegment {
                    speaker: "SPEAKER_00".to_string(),
                    start: 1.0,
                    end: 2.0,
                    text: "How are you?".to_string(),
                },
                DiarizationSegment {
                    speaker: "SPEAKER_01".to_string(),
                    start: 2.0,
                    end: 3.5,
                    text: "I am fine.".to_string(),
                },
            ]),
            segments: None,
        };
        assert_eq!(
            format_transcription_output(&result, false),
            "[SPEAKER_00] Hello. How are you?\n[SPEAKER_01] I am fine."
        );
    }

    #[test]
    fn test_format_diarized_empty_segments() {
        let result = TranscriptionResult {
            text: "Hello world.".to_string(),
            metadata: Default::default(),
            diarization: Some(vec![]),
            segments: None,
        };
        // Empty segments → fall back to plain text
        assert_eq!(format_transcription_output(&result, false), "Hello world.");
    }

    #[test]
    fn test_format_diarized_with_timestamps() {
        let result = TranscriptionResult {
            text: "Hello. I am fine.".to_string(),
            metadata: Default::default(),
            diarization: Some(vec![
                DiarizationSegment {
                    speaker: "speaker_1".to_string(),
                    start: 0.0,
                    end: 1.5,
                    text: "Hello.".to_string(),
                },
                DiarizationSegment {
                    speaker: "speaker_2".to_string(),
                    start: 1.5,
                    end: 3.0,
                    text: "I am fine.".to_string(),
                },
            ]),
            segments: None,
        };
        assert_eq!(
            format_transcription_output(&result, true),
            "[00:00:00] speaker_1 Hello.\n[00:00:01] speaker_2 I am fine."
        );
    }

    #[test]
    fn test_format_diarized_with_timestamps_merges_same_speaker() {
        let result = TranscriptionResult {
            text: "Hello. How are you? I am fine.".to_string(),
            metadata: Default::default(),
            diarization: Some(vec![
                DiarizationSegment {
                    speaker: "speaker_1".to_string(),
                    start: 0.0,
                    end: 1.0,
                    text: "Hello.".to_string(),
                },
                DiarizationSegment {
                    speaker: "speaker_1".to_string(),
                    start: 1.0,
                    end: 2.0,
                    text: "How are you?".to_string(),
                },
                DiarizationSegment {
                    speaker: "speaker_2".to_string(),
                    start: 2.0,
                    end: 3.5,
                    text: "I am fine.".to_string(),
                },
            ]),
            segments: None,
        };
        assert_eq!(
            format_transcription_output(&result, true),
            "[00:00:00] speaker_1 Hello. How are you?\n[00:00:02] speaker_2 I am fine."
        );
    }

    #[test]
    fn test_format_timestamp_helper() {
        assert_eq!(format_timestamp(0.0), "00:00:00");
        assert_eq!(format_timestamp(1.5), "00:00:01");
        assert_eq!(format_timestamp(61.0), "00:01:01");
        assert_eq!(format_timestamp(3661.0), "01:01:01");
    }

    #[test]
    fn test_parse_transcript_segments_voxtral_shape() {
        // Real Voxtral response shape: segments with start/end/text
        // and speaker_id: null when diarization is not requested.
        let raw = serde_json::json!([
            {
                "text": "Du coup je viens de corriger.",
                "start": 1.2,
                "end": 10.7,
                "type": "transcription_segment",
                "speaker_id": null
            },
            {
                "text": " Le premier c'était un bug.",
                "start": 11.9,
                "end": 24.5,
                "type": "transcription_segment",
                "speaker_id": null
            }
        ]);
        let slice = raw.as_array().expect("array literal is array");
        let parsed = parse_transcript_segments(slice).expect("some segments");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].start, 1.2);
        assert_eq!(parsed[0].end, 10.7);
        assert_eq!(parsed[0].text, "Du coup je viens de corriger.");
        assert_eq!(parsed[1].start, 11.9);
        assert_eq!(parsed[1].end, 24.5);
    }

    #[test]
    fn test_parse_transcript_segments_whisper_verbose_json_shape() {
        // Whisper verbose_json shape adds extra fields we ignore:
        // `id`, `seek`, `tokens`, `temperature`, `avg_logprob`,
        // `compression_ratio`, `no_speech_prob`.  Only start/end/text
        // matter for us.
        let raw = serde_json::json!([
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 3.2,
                "text": " Hello world.",
                "tokens": [50364, 2425, 1002, 13, 50524],
                "temperature": 0.0,
                "avg_logprob": -0.3,
                "compression_ratio": 1.1,
                "no_speech_prob": 0.01
            }
        ]);
        let slice = raw.as_array().expect("array literal is array");
        let parsed = parse_transcript_segments(slice).expect("some segments");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].start, 0.0);
        assert_eq!(parsed[0].end, 3.2);
        assert_eq!(parsed[0].text, " Hello world.");
    }

    #[test]
    fn test_parse_transcript_segments_skips_malformed() {
        // Segments without start/end, or with empty text, are skipped.
        // Returns None when nothing usable survives.
        let raw = serde_json::json!([
            { "text": "no timing here" },
            { "start": 0.0, "text": "no end here" },
            { "start": 0.0, "end": 1.0, "text": "" },
            { "start": 5.0, "end": 6.0 }
        ]);
        let slice = raw.as_array().expect("array literal is array");
        assert!(parse_transcript_segments(slice).is_none());
    }

    #[test]
    fn test_parse_transcript_segments_mixed_good_and_bad() {
        // One good segment among several malformed ones survives.
        let raw = serde_json::json!([
            { "text": "no timing" },
            { "start": 1.0, "end": 2.5, "text": "valid" },
            { "start": 0.0, "end": 1.0, "text": "" }
        ]);
        let slice = raw.as_array().expect("array literal is array");
        let parsed = parse_transcript_segments(slice).expect("some segments");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].text, "valid");
    }

    #[test]
    fn test_parse_transcript_segments_empty_input() {
        let parsed = parse_transcript_segments(&[]);
        assert!(parsed.is_none());
    }

    #[test]
    fn test_transcription_result_default_has_no_segments() {
        let result = TranscriptionResult::default();
        assert!(result.segments.is_none());
        assert!(result.diarization.is_none());
        assert_eq!(result.text, "");
    }

    #[test]
    fn openai_batch_override_clones_config_and_replaces_only_model() {
        let original = crate::config::OpenAIConfig {
            api_key: "key".to_string(),
            url: Some("https://example.test".to_string()),
            model: "gpt-transcribe".to_string(),
            realtime_model: "gpt-live-transcribe".to_string(),
            prompt: Some("prompt".to_string()),
            keywords: Some(vec!["keyword".to_string()]),
            languages: Some(vec!["fr".to_string()]),
            realtime_delay: Some(crate::config::OpenAIRealtimeDelay::Low),
        };

        let overridden = override_openai_batch_model(&original, Some("whisper-1"));
        assert_eq!(overridden.model, "whisper-1");
        assert_eq!(overridden.api_key, original.api_key);
        assert_eq!(overridden.url, original.url);
        assert_eq!(overridden.realtime_model, original.realtime_model);
        assert_eq!(overridden.prompt, original.prompt);
        assert_eq!(overridden.keywords, original.keywords);
        assert_eq!(overridden.languages, original.languages);
        assert_eq!(overridden.realtime_delay, original.realtime_delay);
    }

    fn minimal_config() -> Config {
        use tempfile::NamedTempFile;
        let yaml = r#"
output_dir: /tmp/test-output
providers:
  mistral:
    api_key: test
"#;
        let mut file = NamedTempFile::new().expect("tmp file");
        std::io::Write::write_all(&mut file, yaml.as_bytes()).expect("write");
        Config::load(Some(file.path())).expect("load")
    }

    #[tokio::test]
    async fn test_produce_transcript_returns_existing_pick() {
        let dir = tempfile::TempDir::new().expect("tmp dir");
        let audio_path = dir.path().join("has-pick.ogg");
        std::fs::write(&audio_path, b"fake").expect("write audio");
        crate::recording_cache::write_pick(
            &audio_path,
            "openai",
            "whisper-1",
            false,
            "cached text",
        )
        .expect("write pick");

        let config = minimal_config();
        crate::recording_cache::TranscriptionCache::store(
            &audio_path,
            Provider::Mistral,
            "voxtral-mini-2507",
            false,
            &TranscriptionResult {
                text: "older provider output".into(),
                ..Default::default()
            },
        )
        .expect("write lower-priority sidecar");
        assert_eq!(
            read_cached_transcript(&audio_path, &config),
            Some("cached text".into())
        );
        let result = produce_transcript(
            &audio_path,
            &config,
            Provider::OpenAI,
            None,
            &(std::sync::Arc::new(crate::telemetry::NoOpSink)
                as std::sync::Arc<dyn crate::telemetry::TelemetrySink>),
        )
        .await;
        assert_eq!(result.unwrap(), "cached text");
    }

    #[tokio::test]
    async fn test_produce_transcript_returns_in_progress_when_locked() {
        let dir = tempfile::TempDir::new().expect("tmp dir");
        let audio_path = dir.path().join("locked.ogg");
        std::fs::write(&audio_path, b"fake").expect("write audio");
        crate::recording_cache::acquire_pick_lock(&audio_path).expect("lock");

        let config = minimal_config();
        let result = produce_transcript(
            &audio_path,
            &config,
            Provider::OpenAI,
            None,
            &(std::sync::Arc::new(crate::telemetry::NoOpSink)
                as std::sync::Arc<dyn crate::telemetry::TelemetrySink>),
        )
        .await;
        assert!(matches!(result, Err(TalkError::TranscriptInProgress)));
    }

    #[test]
    fn read_cached_transcript_falls_back_to_default_sidecar_without_network() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let audio = dir.path().join("fallback.ogg");
        std::fs::write(&audio, b"audio").expect("write audio");
        let config = minimal_config();
        crate::recording_cache::TranscriptionCache::store(
            &audio,
            Provider::Mistral,
            "voxtral-mini-2507",
            false,
            &TranscriptionResult {
                text: "sidecar transcript".into(),
                ..Default::default()
            },
        )
        .expect("write sidecar");

        assert_eq!(
            read_cached_transcript(&audio, &config),
            Some("sidecar transcript".into())
        );
    }

    #[tokio::test]
    async fn test_produce_transcript_prefers_pick_over_lock_only_when_lock_absent() {
        // When both pick and lock exist, lock wins -> InProgress.
        // When only pick exists, pick wins -> return text.
        let dir = tempfile::TempDir::new().expect("tmp dir");
        let audio_path = dir.path().join("pick-only.ogg");
        std::fs::write(&audio_path, b"fake").expect("write audio");
        crate::recording_cache::write_pick(&audio_path, "openai", "whisper-1", false, "x")
            .expect("write pick");

        let config = minimal_config();
        let result = produce_transcript(
            &audio_path,
            &config,
            Provider::OpenAI,
            None,
            &(std::sync::Arc::new(crate::telemetry::NoOpSink)
                as std::sync::Arc<dyn crate::telemetry::TelemetrySink>),
        )
        .await;
        assert_eq!(result.unwrap(), "x");
    }

    /// Factory wiring: with the `parakeet` feature enabled (the
    /// default), `create_oneshot_transcriber(.., Provider::Parakeet, ..)`
    /// must return `Ok(boxed)` without touching the filesystem or the
    /// network.  The expensive model load is deferred to
    /// `validate` / `fetch_transcription`.
    #[cfg(feature = "parakeet")]
    #[test]
    fn create_oneshot_transcriber_parakeet_constructs_without_io() {
        use crate::config::{ParakeetConfig, ParakeetVariant, ProvidersConfig};

        let tmp = tempfile::TempDir::new().expect("tmp dir");
        let parakeet_cfg = ParakeetConfig {
            variant: ParakeetVariant::Int8,
            // Empty dir — must NOT error at construction time.
            model_dir: Some(tmp.path().to_path_buf()),
            num_threads: 1,
            model: None,
        };
        let config = Config {
            output_dir: tmp.path().to_path_buf(),
            providers: ProvidersConfig {
                mistral: None,
                openai: None,
                parakeet: Some(parakeet_cfg),
                kokoro: None,
            },
            indicators: None,
            transcription: None,
            speak: None,
            paste: None,
            audio: None,
            recording: None,
        };

        let t = create_oneshot_transcriber(
            &config,
            Provider::Parakeet,
            None,
            false,
            RequestTimeoutPolicy::Proportional,
        )
        .expect("parakeet construction must succeed");
        // We only need to confirm the boxed value exists — the
        // concrete type stays private behind the trait.
        let _: Box<dyn OneShotTranscriber> = t;
    }

    /// End-to-end guard on the pre-upload normalization step.
    ///
    /// `encode_16k_mono_ogg` is the one call site that hands a whole
    /// decoded file to `OggOpusWriter::write_pcm` in a single call, so
    /// it is where the historical quadratic front-drain in `write_pcm`
    /// actually hurt: a 2h22m memo spent ~132 minutes memmoving its PCM
    /// buffer before contacting the API at all (see the comment on the
    /// cursor loop in `src/audio/writer.rs`).
    ///
    /// A short real OGG/Opus file exercises the same whole-buffer path
    /// without encoding twenty minutes of audio twice. The deterministic
    /// guard on the underlying quadratic cause lives in
    /// `audio::writer::tests::test_write_pcm_bulk_call_does_not_memmove_quadratically`.
    #[test]
    fn encode_16k_mono_ogg_normalizes_a_recording_quickly() {
        use crate::audio::{AudioWriter, OggOpusWriter};

        const SECONDS: usize = 2;
        const RATE: usize = 16_000;

        let dir = tempfile::TempDir::new().expect("tmp dir");
        let path = dir.path().join("talkrs-perf-check.ogg");

        // Build the fixture with the project's own writer rather than an
        // external tool, so the test has no runtime dependency on ffmpeg.
        let pcm: Vec<i16> = (0..SECONDS * RATE)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                ((t * 440.0 * std::f32::consts::TAU).sin() * 12000.0) as i16
            })
            .collect();
        let mut writer =
            OggOpusWriter::new(crate::config::AudioConfig::new()).expect("writer construction");
        let mut fixture = writer.header().expect("header");
        fixture.extend_from_slice(&writer.write_pcm(&pcm).expect("encode fixture"));
        fixture.extend_from_slice(&writer.finalize().expect("finalize fixture"));
        std::fs::write(&path, &fixture).expect("write fixture");

        let started = std::time::Instant::now();
        let out = encode_16k_mono_ogg(&path).expect("normalization must succeed");
        let elapsed = started.elapsed();

        assert!(!out.is_empty(), "normalization produced no bytes");
        assert_eq!(
            &out[0..4],
            b"OggS",
            "normalized output must be an OGG stream"
        );
        eprintln!(
            "encode_16k_mono_ogg: {SECONDS}s recording ({} bytes in, {} bytes out) in {:?}",
            fixture.len(),
            out.len(),
            elapsed
        );
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "normalizing a {SECONDS}s recording took {elapsed:?}; the pre-upload step is too slow"
        );
    }
}

/// Specification of upload preparation: the provenance fast path, the
/// per-version memo, concurrency and cancellation.
#[cfg(test)]
mod upload_prepare_tests {
    use super::*;
    use crate::audio::{AudioWriter, OggOpusWriter};
    use crate::perf_counters::{thread_value, Counter};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio_util::sync::CancellationToken;

    fn tone(seconds: f64, rate: u32) -> Vec<i16> {
        (0..(seconds * rate as f64) as usize)
            .map(|i| {
                let t = i as f32 / rate as f32;
                ((t * 330.0 * std::f32::consts::TAU).sin() * 9_000.0) as i16
            })
            .collect()
    }

    /// A finalized stream from `writer` carrying `pcm`.
    fn ogg(mut writer: OggOpusWriter, pcm: &[i16]) -> Vec<u8> {
        let mut bytes = writer.header().expect("header");
        bytes.extend(writer.write_pcm(pcm).expect("pcm"));
        bytes.extend(writer.finalize().expect("finalize"));
        bytes
    }

    /// What dictation writes into the recordings cache.
    fn dictation_ogg(seconds: f64) -> Vec<u8> {
        ogg(
            OggOpusWriter::new(crate::config::AudioConfig::new()).expect("writer"),
            &tone(seconds, 16_000),
        )
    }

    /// What `talk-rs record` writes (48 kHz, Audio application).
    fn recording_ogg(seconds: f64) -> Vec<u8> {
        let config = crate::config::AudioConfig {
            sample_rate: 48_000,
            channels: 1,
            bitrate: 64_000,
        };
        ogg(
            OggOpusWriter::new_for_recording(config).expect("writer"),
            &tone(seconds, 48_000),
        )
    }

    /// Prepare `name` (holding `bytes`) placed directly in a stand-in
    /// recordings directory, or next to it; returns the upload and the
    /// re-encodes it took.
    fn prepare_placed(bytes: &[u8], in_recordings: bool) -> (PreparedUpload, u64) {
        let root = tempfile::tempdir().expect("tmp");
        let recordings = root.path().join("recordings");
        std::fs::create_dir_all(&recordings).expect("dir");
        let dir = if in_recordings {
            recordings.clone()
        } else {
            root.path().to_path_buf()
        };
        let path = dir.join("2026-10-01T09-00-00+0200.ogg");
        std::fs::write(&path, bytes).expect("write");
        let encodes = thread_value(Counter::UploadEncodes);
        let prepared = prepare_upload_from(&path, Some(&recordings)).expect("prepare");
        (prepared, thread_value(Counter::UploadEncodes) - encodes)
    }

    #[test]
    fn own_dictation_recording_is_uploaded_as_is() {
        let original = dictation_ogg(1.5);
        let (prepared, encodes) = prepare_placed(&original, true);
        assert_eq!(encodes, 0);
        assert_eq!(*prepared.bytes, original);
        assert_eq!(prepared.file_name, "2026-10-01T09-00-00+0200.ogg");
    }

    #[test]
    fn upload_profile_recording_outside_the_cache_is_normalized() {
        let original = dictation_ogg(1.5);
        let (prepared, encodes) = prepare_placed(&original, false);
        assert_eq!(encodes, 1);
        assert_ne!(*prepared.bytes, original);
    }

    #[test]
    fn other_profile_in_the_cache_is_normalized() {
        let original = recording_ogg(1.5);
        let (prepared, encodes) = prepare_placed(&original, true);
        assert_eq!(encodes, 1);
        assert_ne!(*prepared.bytes, original);
        assert!(is_own_upload_profile_recording(&prepared.bytes));
    }

    #[test]
    fn unfinished_or_damaged_recording_in_the_cache_is_normalized() {
        let original = dictation_ogg(1.5);
        // Interrupted recording: no end-of-stream page.
        let last_page = original
            .windows(4)
            .rposition(|w| w == b"OggS")
            .expect("pages");
        let (prepared, encodes) = prepare_placed(&original[..last_page], true);
        assert_eq!(encodes, 1);
        assert_ne!(*prepared.bytes, &original[..last_page]);
        // One flipped byte in the audio: page checksum fails.
        let mut damaged = original.clone();
        let mid = damaged.len() / 2;
        damaged[mid] ^= 0x55;
        assert!(!is_own_upload_profile_recording(&damaged));
        // Something appended after the end of the stream.
        let mut chained = original.clone();
        chained.extend_from_slice(&dictation_ogg(0.5));
        assert!(!is_own_upload_profile_recording(&chained));
        assert!(is_own_upload_profile_recording(&original));
        let mut junk = original.clone();
        junk.extend_from_slice(b"unframed trailing data");
        assert!(!is_own_upload_profile_recording(&junk));
    }

    #[test]
    fn remuxed_header_and_foreign_serial_cannot_claim_the_upload_profile() {
        use ogg::{PacketWriteEndInfo, PacketWriter};
        let original = dictation_ogg(0.1);
        for changed in [Some(8), Some(9), Some(12), Some(16), Some(18), None] {
            let mut reader = ogg::reading::PacketReader::new(std::io::Cursor::new(&original));
            let mut writer = PacketWriter::new(Vec::new());
            let mut index = 0;
            while let Some(mut packet) = reader.read_packet().expect("packet") {
                if index == 0 {
                    if let Some(byte) = changed {
                        packet.data[byte] ^= 1;
                    }
                }
                let serial = if changed.is_none() && index > 1 {
                    43
                } else {
                    42
                };
                let end = if packet.last_in_stream() {
                    PacketWriteEndInfo::EndStream
                } else {
                    PacketWriteEndInfo::EndPage
                };
                let granule = packet.absgp_page();
                writer
                    .write_packet(packet.data, serial, end, granule)
                    .expect("write");
                index += 1;
            }
            assert!(
                !is_own_upload_profile_recording(writer.inner_mut()),
                "changed {changed:?}"
            );
        }
    }

    #[test]
    fn non_ogg_bytes_are_never_own_recordings() {
        assert!(!is_own_upload_profile_recording(b""));
        assert!(!is_own_upload_profile_recording(b"not really audio"));
        let (prepared, _) = prepare_placed(b"not really audio", true);
        // Undecodable: original bytes under the original name.
        assert_eq!(*prepared.bytes, b"not really audio");
        assert_eq!(prepared.file_name, "2026-10-01T09-00-00+0200.ogg");
    }

    #[test]
    fn symlinked_cache_parent_keeps_the_fast_path_but_file_alias_does_not() {
        let root = tempfile::tempdir().expect("tmp");
        let recordings = root.path().join("recordings");
        std::fs::create_dir(&recordings).expect("recordings");
        let link = root.path().join("cache-link");
        std::os::unix::fs::symlink(&recordings, &link).expect("link");
        let file = link.join("voice.ogg");
        let bytes = dictation_ogg(0.1);
        std::fs::write(&file, &bytes).expect("write");
        assert_eq!(
            *prepare_upload_from(&file, Some(&recordings))
                .expect("fast")
                .bytes,
            bytes
        );

        let alias = root.path().join("alias.ogg");
        std::os::unix::fs::symlink(&file, &alias).expect("alias");
        assert_ne!(
            *prepare_upload_from(&alias, Some(&recordings))
                .expect("normalize")
                .bytes,
            bytes
        );
    }

    #[tokio::test]
    async fn aliases_keep_their_own_decode_and_upload_names_in_both_orders() {
        let root = tempfile::tempdir().expect("tmp");
        let target = root.path().join("source.wav");
        let mut writer = crate::audio::WavWriter::new(crate::config::AudioConfig::new());
        let mut wav = writer.header().expect("header");
        wav.extend(writer.write_pcm(&[1000; 320]).expect("pcm"));
        let header = writer.finalize().expect("finalize");
        wav[..header.len()].copy_from_slice(&header);
        std::fs::write(&target, &wav).expect("write");
        let spoken = root.path().join("spoken.wav");
        let opaque = root.path().join("opaque.bin");
        std::os::unix::fs::symlink(&target, &spoken).expect("link");
        std::os::unix::fs::symlink(&target, &opaque).expect("link");
        for raw_first in [true, false] {
            let memo = UploadPreparations::new(2);
            let cancel = CancellationToken::new();
            let paths = if raw_first {
                [&opaque, &spoken]
            } else {
                [&spoken, &opaque]
            };
            for path in paths {
                let prepared = memo
                    .prepare(path, &cancel, prepare_upload_blocking)
                    .await
                    .expect("prepared");
                if path == &opaque {
                    assert_eq!(prepared.file_name, "opaque.bin");
                    assert_eq!(*prepared.bytes, wav);
                } else {
                    assert_eq!(prepared.file_name, "spoken.ogg");
                    assert!(is_own_upload_profile_recording(&prepared.bytes));
                }
            }
        }
    }

    #[tokio::test]
    async fn changed_during_preparation_returns_current_bytes() {
        static RUNS: AtomicUsize = AtomicUsize::new(0);
        fn prepare(path: &Path) -> Result<PreparedUpload, TalkError> {
            let bytes = std::fs::read(path)?;
            if RUNS.fetch_add(1, Ordering::SeqCst) == 0 {
                std::fs::write(path, b"second").map_err(TalkError::Io)?;
            }
            Ok(PreparedUpload {
                bytes: std::sync::Arc::new(bytes),
                file_name: "audio.bin".into(),
            })
        }
        let root = tempfile::tempdir().expect("tmp");
        let path = root.path().join("audio.bin");
        std::fs::write(&path, b"first").expect("write");
        let memo = UploadPreparations::new(2);
        let prepared = memo
            .prepare(&path, &CancellationToken::new(), prepare)
            .await
            .expect("prepared");
        assert_eq!(&*prepared.bytes, b"second");
        assert_eq!(RUNS.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn already_cancelled_does_not_schedule_preparation() {
        static RUNS: AtomicUsize = AtomicUsize::new(0);
        fn prepare(_: &Path) -> Result<PreparedUpload, TalkError> {
            RUNS.fetch_add(1, Ordering::SeqCst);
            Err(TalkError::Transcription("unexpected work".into()))
        }
        let root = tempfile::tempdir().expect("tmp");
        let path = root.path().join("audio.bin");
        std::fs::write(&path, b"audio").expect("write");
        let cancel = CancellationToken::new();
        cancel.cancel();
        let memo = UploadPreparations::new(2);
        let err = memo
            .prepare(&path, &cancel, prepare)
            .await
            .expect_err("cancelled");
        assert!(err.to_string().contains("cancelled"), "{err}");
        assert_eq!(RUNS.load(Ordering::SeqCst), 0);
    }

    // ── Memo ──────────────────────────────────────────────────────

    /// A preparation that records each run and takes `DELAY`.
    macro_rules! counting_prepare {
        ($runs:ident, $delay_ms:expr) => {{
            static $runs: AtomicUsize = AtomicUsize::new(0);
            fn prepare(path: &Path) -> Result<PreparedUpload, TalkError> {
                $runs.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis($delay_ms));
                Ok(PreparedUpload {
                    bytes: std::sync::Arc::new(std::fs::read(path)?),
                    file_name: "x.ogg".into(),
                })
            }
            (
                &$runs,
                prepare as fn(&Path) -> Result<PreparedUpload, TalkError>,
            )
        }};
    }

    #[tokio::test]
    async fn same_version_is_prepared_once_and_a_new_version_again() {
        let (runs, prepare) = counting_prepare!(RUNS, 0);
        let memo = UploadPreparations::new(2);
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("memo.ogg");
        std::fs::write(&path, b"first").expect("write");
        let cancel = CancellationToken::new();
        let a = memo.prepare(&path, &cancel, prepare).await.expect("a");
        let b = memo.prepare(&path, &cancel, prepare).await.expect("b");
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert!(std::sync::Arc::ptr_eq(&a, &b));
        assert_eq!(*a.bytes, b"first");

        // Replaced by a new file of the same size: new inode/ctime.
        let tmp = dir.path().join("memo.tmp");
        std::fs::write(&tmp, b"secnd").expect("write");
        std::fs::rename(&tmp, &path).expect("rename");
        let c = memo.prepare(&path, &cancel, prepare).await.expect("c");
        assert_eq!(*c.bytes, b"secnd");
        // Rewritten in place.
        std::fs::write(&path, b"third, longer").expect("write");
        let d = memo.prepare(&path, &cancel, prepare).await.expect("d");
        assert_eq!(*d.bytes, b"third, longer");
        assert_eq!(runs.load(Ordering::SeqCst), 3);
        // Only the current version of a file is kept.
        assert_eq!(memo.slots.lock().expect("lock").len(), 1);
    }

    #[tokio::test]
    async fn memo_is_bounded_and_keeps_aliases_distinct() {
        let (runs, prepare) = counting_prepare!(RUNS, 0);
        let memo = UploadPreparations::new(2);
        let dir = tempfile::tempdir().expect("tmp");
        let cancel = CancellationToken::new();
        let paths: Vec<PathBuf> = (0..3)
            .map(|i| {
                let p = dir.path().join(format!("{i}.ogg"));
                std::fs::write(&p, format!("audio {i}")).expect("write");
                p
            })
            .collect();
        let link = dir.path().join("link.ogg");
        std::os::unix::fs::symlink(&paths[0], &link).expect("symlink");
        memo.prepare(&paths[0], &cancel, prepare).await.expect("0");
        memo.prepare(&link, &cancel, prepare).await.expect("link");
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        for p in &paths[1..] {
            memo.prepare(p, &cancel, prepare).await.expect("p");
        }
        assert_eq!(memo.slots.lock().expect("lock").len(), 2);
        // The oldest was evicted, so it is prepared again.
        memo.prepare(&paths[0], &cancel, prepare)
            .await
            .expect("0 again");
        assert_eq!(runs.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn oversized_raw_payload_is_delivered_but_not_retained() {
        static RUNS: AtomicUsize = AtomicUsize::new(0);
        fn prepare(_: &Path) -> Result<PreparedUpload, TalkError> {
            RUNS.fetch_add(1, Ordering::SeqCst);
            Ok(PreparedUpload {
                bytes: std::sync::Arc::new(vec![1; PREPARED_UPLOADS_BYTES + 1]),
                file_name: "raw.bin".into(),
            })
        }
        let root = tempfile::tempdir().expect("tmp");
        let path = root.path().join("raw.bin");
        std::fs::write(&path, b"source").expect("write");
        let memo = UploadPreparations::new(2);
        for expected in 1..=2 {
            let prepared = memo
                .prepare(&path, &CancellationToken::new(), prepare)
                .await
                .expect("payload");
            assert_eq!(prepared.bytes.len(), PREPARED_UPLOADS_BYTES + 1);
            assert_eq!(RUNS.load(Ordering::SeqCst), expected);
            assert!(memo.slots.lock().expect("lock").is_empty());
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_preparations_trim_on_detached_completion() {
        use std::sync::{Condvar, Mutex};
        static GATE: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());
        static COMPLETED: AtomicUsize = AtomicUsize::new(0);
        fn prepare(_: &Path) -> Result<PreparedUpload, TalkError> {
            let mut released = GATE.0.lock().expect("gate");
            while !*released {
                released = GATE.1.wait(released).expect("gate wait");
            }
            drop(released);
            let bytes = std::sync::Arc::new(vec![0; 4 * 1024 * 1024]);
            COMPLETED.fetch_add(1, Ordering::SeqCst);
            Ok(PreparedUpload {
                bytes,
                file_name: "audio.bin".into(),
            })
        }

        let root = tempfile::tempdir().expect("tmp");
        let memo = std::sync::Arc::new(UploadPreparations::new(PREPARED_UPLOADS_KEPT));
        let mut callers = Vec::new();
        let mut tokens = Vec::new();
        for n in 0..10 {
            let path = root.path().join(format!("{n}.bin"));
            std::fs::write(&path, b"source").expect("write");
            let token = CancellationToken::new();
            tokens.push(token.clone());
            let owner = memo.clone();
            callers.push(tokio::spawn(async move {
                owner.prepare(&path, &token, prepare).await
            }));
        }
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while memo.slots.lock().expect("slots").len() != 10 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("registered");
        for token in tokens {
            token.cancel();
        }
        for caller in callers {
            assert!(caller.await.expect("caller").is_err());
        }
        *GATE.0.lock().expect("gate") = true;
        GATE.1.notify_all();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while memo.completed_maintenance.load(Ordering::SeqCst) != 10 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("completed");
        assert_eq!(COMPLETED.load(Ordering::SeqCst), 10);
        let slots = memo.slots.lock().expect("slots");
        let entries = slots.iter().filter(|(_, slot)| slot.initialized()).count();
        let bytes: usize = slots
            .iter()
            .filter_map(|(_, slot)| slot.get())
            .map(|upload| upload.bytes.len())
            .sum();
        assert!(
            entries <= PREPARED_UPLOADS_KEPT,
            "retained {entries} entries"
        );
        assert!(bytes <= PREPARED_UPLOADS_BYTES, "retained {bytes} bytes");
    }

    #[tokio::test]
    async fn publication_does_not_stamp_old_response_with_replacement_identity() {
        let root = tempfile::tempdir().expect("tmp");
        let path = root.path().join("recording.wav");
        std::fs::write(&path, b"audio A").expect("write A");
        let uploaded = SourceVersion::of(&path).await.expect("version A");
        let result = TranscriptionResult {
            text: "text for A".into(),
            ..Default::default()
        };
        assert_eq!(SourceVersion::of(&path).await.expect("checked A"), uploaded);
        let replacement = root.path().join("replacement.wav");
        std::fs::write(&replacement, b"different replacement audio B").expect("write B");
        std::fs::rename(&replacement, &path).expect("replace");
        assert!(crate::recording_cache::TranscriptionCache::store_verified(
            &path,
            Provider::OpenAI,
            "gpt-transcribe",
            false,
            &result,
            &uploaded,
        )
        .await
        .is_err());
        assert!(
            crate::recording_cache::TranscriptionCache::get(
                &path,
                Provider::OpenAI,
                "gpt-transcribe",
            )
            .is_none(),
            "replacement B must not receive A's transcript"
        );
    }

    #[tokio::test]
    async fn verified_sidecar_rejects_replacement_with_matching_size_and_mtime() {
        let root = tempfile::tempdir().expect("tmp");
        let path = root.path().join("recording.wav");
        std::fs::write(&path, b"audio A").expect("write A");
        let uploaded = SourceVersion::of(&path).await.expect("version A");
        let modified = std::fs::metadata(&path)
            .expect("metadata A")
            .modified()
            .expect("mtime A");
        let result = TranscriptionResult {
            text: "text for A".into(),
            ..Default::default()
        };
        crate::recording_cache::TranscriptionCache::store_verified(
            &path,
            Provider::OpenAI,
            "gpt-transcribe",
            false,
            &result,
            &uploaded,
        )
        .await
        .expect("store A");
        assert_eq!(
            crate::recording_cache::TranscriptionCache::get(
                &path,
                Provider::OpenAI,
                "gpt-transcribe",
            )
            .expect("A sidecar")
            .text,
            "text for A"
        );
        let replacement = root.path().join("replacement.wav");
        std::fs::write(&replacement, b"audio B").expect("write B");
        std::fs::File::open(&replacement)
            .expect("open B")
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .expect("match mtime");
        std::fs::rename(&replacement, &path).expect("replace");
        assert!(
            crate::recording_cache::TranscriptionCache::get(
                &path,
                Provider::OpenAI,
                "gpt-transcribe",
            )
            .is_none(),
            "matching size/mtime must not authenticate a different inode"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn active_slots_remain_shared_under_eviction_pressure() {
        use std::collections::HashMap;
        use std::sync::{mpsc, Mutex, OnceLock};
        type Gate = (mpsc::Sender<()>, mpsc::Receiver<()>);
        static GATES: OnceLock<Mutex<HashMap<PathBuf, Gate>>> = OnceLock::new();
        static RUNS: AtomicUsize = AtomicUsize::new(0);
        fn prepare(path: &Path) -> Result<PreparedUpload, TalkError> {
            RUNS.fetch_add(1, Ordering::SeqCst);
            let gate = GATES
                .get()
                .expect("gates")
                .lock()
                .expect("lock")
                .remove(path);
            if let Some((entered, release)) = gate {
                entered.send(()).expect("entered");
                release.recv().expect("release");
            }
            Ok(PreparedUpload {
                bytes: std::sync::Arc::new(std::fs::read(path)?),
                file_name: "audio.bin".into(),
            })
        }
        let root = tempfile::tempdir().expect("tmp");
        let memo = std::sync::Arc::new(UploadPreparations::new(2));
        let mut jobs = Vec::new();
        let mut gates = Vec::new();
        for n in 0..3 {
            let path = root.path().join(format!("{n}.bin"));
            std::fs::write(&path, b"audio").expect("write");
            let (entered, ready) = mpsc::channel();
            let (release, waiting) = mpsc::channel();
            GATES
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .expect("lock")
                .insert(path.clone(), (entered, waiting));
            gates.push((ready, release));
            let memo = memo.clone();
            jobs.push(tokio::spawn(async move {
                memo.prepare(&path, &CancellationToken::new(), prepare)
                    .await
            }));
        }
        for (ready, _) in &gates {
            tokio::task::block_in_place(|| ready.recv().expect("started"));
        }
        let attached_before = memo.shared_attachments.load(Ordering::SeqCst);
        let first = root.path().join("0.bin");
        let duplicate = {
            let memo = memo.clone();
            tokio::spawn(async move {
                memo.prepare(&first, &CancellationToken::new(), prepare)
                    .await
            })
        };
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while memo.shared_attachments.load(Ordering::SeqCst) == attached_before {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("duplicate attached to existing slot");
        for (_, release) in gates {
            release.send(()).expect("release");
        }
        let first = jobs.remove(0).await.expect("join").expect("first");
        let same = duplicate.await.expect("join").expect("same");
        assert!(std::sync::Arc::ptr_eq(&first, &same));
        for job in jobs {
            job.await.expect("join").expect("prepared");
        }
        assert_eq!(RUNS.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_provider_waiters_do_not_post_or_abort_a_survivor() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let root = tempfile::tempdir().expect("tmp");
        let source = root.path().join("recording.wav");
        let mut writer = crate::audio::WavWriter::new(crate::config::AudioConfig::new());
        let mut wav = writer.header().expect("header");
        wav.extend(writer.write_pcm(&vec![1000; 16_000 * 30]).expect("pcm"));
        let header = writer.finalize().expect("finalize");
        wav[..header.len()].copy_from_slice(&header);
        std::fs::write(&source, wav).expect("write");

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"text": "ok"})),
            )
            .mount(&server)
            .await;
        let mistral_config = crate::config::MistralConfig {
            api_key: "test".into(),
            url: Some(server.uri()),
            model: "voxtral-mini-2602".into(),
            context_bias: None,
            tts_model: "voxtral-mini-tts-latest".into(),
            tts_voice: None,
            tts_voices: None,
        };
        let openai_config = crate::config::OpenAIConfig {
            api_key: "test".into(),
            url: Some(server.uri()),
            model: "gpt-transcribe".into(),
            realtime_model: "gpt-live-transcribe".into(),
            prompt: None,
            keywords: None,
            languages: None,
            realtime_delay: None,
        };
        let first_cancel = CancellationToken::new();
        let second_cancel = CancellationToken::new();
        let mut first =
            MistralOneShotTranscriber::new(mistral_config.clone(), false).expect("mistral");
        first.set_cancel_token(first_cancel.clone());
        let mut second = OpenAIOneShotTranscriber::new(openai_config).expect("openai");
        second.set_cancel_token(second_cancel.clone());
        let first_path = source.clone();
        let a = tokio::spawn(async move {
            first
                .fetch_transcription(TranscriptionBody::File(first_path))
                .await
        });
        let second_path = source.clone();
        let b = tokio::spawn(async move {
            second
                .fetch_transcription(TranscriptionBody::File(second_path))
                .await
        });
        let survivor = MistralOneShotTranscriber::new(mistral_config, false).expect("survivor");
        let c = tokio::spawn(async move {
            survivor
                .fetch_transcription(TranscriptionBody::File(source))
                .await
        });
        tokio::task::yield_now().await;
        first_cancel.cancel();
        second_cancel.cancel();
        for waiter in [a, b] {
            let result = tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
                .await
                .expect("prompt cancel")
                .expect("join");
            assert!(result
                .expect_err("cancelled")
                .to_string()
                .contains("cancelled"));
        }
        assert_eq!(c.await.expect("join").expect("result").text, "ok");
        assert_eq!(server.received_requests().await.expect("requests").len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_callers_share_one_preparation_and_survive_a_cancelled_one() {
        let (runs, prepare) = counting_prepare!(RUNS, 300);
        let memo = std::sync::Arc::new(UploadPreparations::new(2));
        let dir = tempfile::tempdir().expect("tmp");
        let path = std::sync::Arc::new(dir.path().join("memo.ogg"));
        std::fs::write(&*path, b"audio").expect("write");

        // The first caller starts the preparation, then is cancelled.
        let first_cancel = CancellationToken::new();
        let first = {
            let (memo, path, cancel) = (memo.clone(), path.clone(), first_cancel.clone());
            tokio::spawn(async move { memo.prepare(&path, &cancel, prepare).await })
        };
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let others: Vec<_> = (0..3)
            .map(|_| {
                let (memo, path) = (memo.clone(), path.clone());
                tokio::spawn(async move {
                    memo.prepare(&path, &CancellationToken::new(), prepare)
                        .await
                })
            })
            .collect();
        first_cancel.cancel();
        let cancelled = first.await.expect("join");
        assert!(
            matches!(&cancelled, Err(TalkError::Transcription(m)) if m.contains("cancelled")),
            "{cancelled:?}"
        );
        for other in others {
            let prepared = other.await.expect("join").expect("prepared");
            assert_eq!(*prepared.bytes, b"audio");
        }
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn failed_preparation_is_not_memoised_and_missing_file_errors() {
        static FAIL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        static RUNS: AtomicUsize = AtomicUsize::new(0);
        fn prepare(_: &Path) -> Result<PreparedUpload, TalkError> {
            RUNS.fetch_add(1, Ordering::SeqCst);
            if FAIL.swap(false, Ordering::SeqCst) {
                return Err(TalkError::Transcription("read failed".into()));
            }
            Ok(PreparedUpload {
                bytes: std::sync::Arc::new(b"ok".to_vec()),
                file_name: "x.ogg".into(),
            })
        }
        let memo = UploadPreparations::new(2);
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("memo.ogg");
        std::fs::write(&path, b"audio").expect("write");
        let cancel = CancellationToken::new();
        assert!(memo.prepare(&path, &cancel, prepare).await.is_err());
        let ok = memo.prepare(&path, &cancel, prepare).await.expect("retry");
        assert_eq!(*ok.bytes, b"ok");
        assert_eq!(RUNS.load(Ordering::SeqCst), 2);

        let err = memo
            .prepare(&dir.path().join("absent.ogg"), &cancel, prepare)
            .await
            .expect_err("missing");
        assert!(err.to_string().contains("not found"), "{err}");
        assert_eq!(RUNS.load(Ordering::SeqCst), 2);
    }

    /// The preparation runs on the blocking pool: a heartbeat on a
    /// single-worker runtime keeps ticking while a slow one runs.
    #[test]
    fn preparation_does_not_block_the_runtime_worker() {
        let (_, prepare) = counting_prepare!(RUNS, 800);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("runtime");
        let max_gap = runtime.block_on(async {
            let dir = tempfile::tempdir().expect("tmp");
            let path = dir.path().join("memo.ogg");
            std::fs::write(&path, b"audio").expect("write");
            let memo = std::sync::Arc::new(UploadPreparations::new(2));
            let task = tokio::spawn(async move {
                memo.prepare(&path, &CancellationToken::new(), prepare)
                    .await
                    .map(|_| ())
            });
            let mut last = std::time::Instant::now();
            let mut max_gap = std::time::Duration::ZERO;
            while !task.is_finished() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                let now = std::time::Instant::now();
                max_gap = max_gap.max(now - last);
                last = now;
            }
            task.await.expect("join").expect("prepared");
            max_gap
        });
        assert!(
            max_gap < std::time::Duration::from_millis(400),
            "worker blocked for {max_gap:?}"
        );
    }
}

/// Performance harness, item `upload-normalize-once`: the upload
/// preparation contract a memoised / provenance-aware implementation
/// must keep, measured through the public `transcribe_audio` path
/// against a local mock provider (validation cache isolated).
///
/// - Foreign input (48 kHz **stereo** WAV) is uploaded as a 16 kHz
///   mono Ogg Opus stream carrying the same audio (strict decode,
///   envelope correlation), with an explicit multipart length.
/// - False provenance: a 48 kHz Audio-profile OGG named and placed like
///   a cache file must still be converted to the upload profile.
/// - Source replacement between two calls re-prepares from the new
///   content (the second upload carries the NEW audio).
/// - Undecodable input still uploads the original bytes and name.
/// - Three concurrent callers on the same file all receive complete,
///   identical-content uploads.
/// - `upload_encodes` counts actual re-encodes (reported per cut).
#[cfg(test)]
mod perf_upload_prepare {
    use super::*;
    use crate::audio::perf_audio::{decode_ogg_strict, envelope_correlation};
    use crate::audio::{AudioWriter, OggOpusWriter, WavWriter};
    use crate::perf_counters::{thread_value, Counter};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct Isolation {
        _lock: std::sync::MutexGuard<'static, ()>,
        _dir: tempfile::TempDir,
    }

    fn isolate() -> Isolation {
        let lock = transport::validate_cache::__TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let dir = tempfile::tempdir().expect("tmp");
        std::env::set_var("TALK_RS_VALIDATE_CACHE_PATH", dir.path().join("v.yaml"));
        transport::validate_cache::__test_reset();
        Isolation {
            _lock: lock,
            _dir: dir,
        }
    }

    /// Speech-like PCM whose envelope depends on `seed`.
    fn speech(seconds: f64, rate: u32, seed: f32) -> Vec<i16> {
        (0..(seconds * rate as f64) as usize)
            .map(|i| {
                let t = i as f32 / rate as f32;
                let env = (t * (2.1 + seed) * std::f32::consts::TAU).sin().abs();
                (env * (t * 210.0 * std::f32::consts::TAU).sin() * 16_000.0) as i16
            })
            .collect()
    }

    fn write_stereo_wav(path: &std::path::Path, mono: &[i16]) {
        let mut w = WavWriter::new(crate::config::AudioConfig {
            sample_rate: 48_000,
            channels: 2,
            bitrate: 0,
        });
        let stereo: Vec<i16> = mono.iter().flat_map(|&s| [s, s]).collect();
        let mut bytes = w.header().expect("header");
        bytes.extend(w.write_pcm(&stereo).expect("pcm"));
        let h = w.finalize().expect("finalize");
        bytes[..h.len()].copy_from_slice(&h);
        std::fs::write(path, bytes).expect("write wav");
    }

    fn write_recording_ogg(path: &std::path::Path, pcm48: &[i16]) {
        let mut w = OggOpusWriter::new_for_recording(crate::config::AudioConfig {
            sample_rate: 48_000,
            channels: 1,
            bitrate: 64_000,
        })
        .expect("writer");
        let mut bytes = w.header().expect("header");
        bytes.extend(w.write_pcm(pcm48).expect("pcm"));
        bytes.extend(w.finalize().expect("finalize"));
        std::fs::write(path, bytes).expect("write ogg");
    }

    fn downsample(pcm48: &[i16]) -> Vec<i16> {
        pcm48.iter().step_by(3).copied().collect()
    }

    async fn provider() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"id": "voxtral-mini-2602"}, {"id": "voxtral-mini-2507"}, {"id": "voxtral-mini-latest"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"text": "ok"})),
            )
            .mount(&server)
            .await;
        server
    }

    fn config(dir: &std::path::Path, url: &str) -> Config {
        serde_yaml::from_str(&format!(
            "output_dir: {}\nproviders:\n  mistral:\n    api_key: fake\n    url: {url}\n    model: voxtral-mini-2602\n",
            dir.display()
        ))
        .expect("config")
    }

    /// (file name, content length header, file bytes) of each upload.
    async fn uploads(server: &MockServer) -> Vec<(String, u64, Vec<u8>)> {
        let mut out = Vec::new();
        for r in server.received_requests().await.unwrap_or_default() {
            if r.method.as_str() != "POST" {
                continue;
            }
            let ct = r
                .headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            let boundary = format!(
                "--{}",
                ct.split("boundary=").nth(1).unwrap_or("").trim_matches('"')
            );
            let len = r
                .headers
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let body = &r.body;
            let find = |h: &[u8], n: &[u8], from: usize| {
                h[from..]
                    .windows(n.len())
                    .position(|w| w == n)
                    .map(|p| p + from)
            };
            let mut pos = 0;
            while let Some(start) = find(body, boundary.as_bytes(), pos) {
                if body[start + boundary.len()..].starts_with(b"--") {
                    break; // closing delimiter
                }
                let head_end = find(body, b"\r\n\r\n", start).expect("part headers");
                let headers = String::from_utf8_lossy(&body[start..head_end]).to_string();
                let next = find(body, boundary.as_bytes(), head_end).unwrap_or(body.len());
                if headers.contains("name=\"file\"") {
                    let name = headers
                        .split("filename=\"")
                        .nth(1)
                        .and_then(|v| v.split('"').next())
                        .unwrap_or("")
                        .to_string();
                    let data = body[head_end + 4..next]
                        .strip_suffix(b"\r\n")
                        .unwrap_or(&body[head_end + 4..next]);
                    out.push((name, len, data.to_vec()));
                }
                pos = start + boundary.len();
            }
        }
        out
    }

    async fn transcribe(path: &std::path::Path, config: &Config) {
        let sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink> =
            std::sync::Arc::new(crate::telemetry::NoOpSink);
        let r = transcribe_audio(
            path,
            config,
            Provider::Mistral,
            None,
            false,
            TranscribeOptions {
                allow_api: true,
                ..TranscribeOptions::default()
            },
            &sink,
        )
        .await
        .expect("transcription");
        assert_eq!(r.text, "ok");
    }

    fn assert_upload_profile(what: &str, bytes: &[u8], reference16: &[i16]) {
        let head = bytes
            .windows(8)
            .position(|w| w == b"OpusHead")
            .expect("OpusHead");
        assert_eq!(bytes[head + 9], 1, "{what}: mono");
        let decoded = decode_ogg_strict(bytes);
        // A lossy source carries up to one Opus frame (20 ms = 320
        // samples) of codec delay through decode → re-encode.
        let diff = decoded.len().abs_diff(reference16.len());
        assert!(
            diff <= 320,
            "{what}: {} vs {} samples",
            decoded.len(),
            reference16.len()
        );
        let r = envelope_correlation(reference16, &decoded);
        assert!(r > 0.95, "{what}: wrong audio (r={r:.3})");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn perf_prepare_foreign_stereo_wav_is_converted() {
        let _iso = isolate();
        let dir = tempfile::tempdir().expect("tmp");
        let server = provider().await;
        let pcm48 = speech(4.0, 48_000, 0.0);
        let input = dir.path().join("import.wav");
        write_stereo_wav(&input, &pcm48);
        let encodes = thread_value(Counter::UploadEncodes);
        transcribe(&input, &config(dir.path(), &server.uri())).await;
        let ups = uploads(&server).await;
        assert_eq!(ups.len(), 1);
        let (name, len, bytes) = &ups[0];
        assert_eq!(name, "import.ogg");
        assert!(
            *len > bytes.len() as u64,
            "explicit Content-Length covers the part"
        );
        assert_upload_profile("stereo wav", bytes, &downsample(&pcm48));
        crate::perf_counters::record_metrics(
            "upload-normalize-once",
            "prepare-foreign-stereo-wav",
            &[(
                "upload_encodes",
                (thread_value(Counter::UploadEncodes) - encodes) as f64,
            )],
        );
    }

    #[tokio::test]
    async fn changed_source_during_http_is_not_cached_as_the_new_recording() {
        let _iso = isolate();
        let dir = tempfile::tempdir().expect("tmp");
        let input = dir.path().join("moving.wav");
        write_stereo_wav(&input, &speech(0.1, 48_000, 0.0));
        let replacement = dir.path().join("replacement.wav");
        write_stereo_wav(&replacement, &speech(0.2, 48_000, 1.0));
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"id": "voxtral-mini-2602"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(move |_: &wiremock::Request| {
                std::fs::rename(&replacement, &input).expect("replace after upload");
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"text": "old audio"}))
            })
            .mount(&server)
            .await;
        let audio = dir.path().join("moving.wav");
        let sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink> =
            std::sync::Arc::new(crate::telemetry::NoOpSink);
        let err = transcribe_audio(
            &audio,
            &config(dir.path(), &server.uri()),
            Provider::Mistral,
            None,
            false,
            TranscribeOptions {
                allow_api: true,
                ..TranscribeOptions::default()
            },
            &sink,
        )
        .await
        .expect_err("replaced audio must not be attributed to current file");
        assert!(err.to_string().contains("changed"), "{err}");
        assert!(crate::recording_cache::TranscriptionCache::get(
            &audio,
            Provider::Mistral,
            "voxtral-mini-2602"
        )
        .is_none());
    }

    /// A 48 kHz recording-profile OGG sitting where a dictation cache
    /// file would, with a cache-like name: provenance is NOT proven by
    /// location/extension, so it must still be converted.
    #[tokio::test(flavor = "current_thread")]
    async fn perf_prepare_false_provenance_ogg_is_converted() {
        let _iso = isolate();
        let dir = tempfile::tempdir().expect("tmp");
        let recordings = dir.path().join("cache/talk-rs/recordings");
        std::fs::create_dir_all(&recordings).expect("dirs");
        let server = provider().await;
        let pcm48 = speech(4.0, 48_000, 0.3);
        let input = recordings.join("2026-09-30T10-00-00+0200.ogg");
        write_recording_ogg(&input, &pcm48);
        let original = std::fs::read(&input).expect("bytes");
        transcribe(&input, &config(dir.path(), &server.uri())).await;
        let (_, _, bytes) = uploads(&server).await.remove(0);
        assert_ne!(bytes, original, "48 kHz input uploaded unconverted");
        assert_upload_profile("48k ogg", &bytes, &downsample(&pcm48));
    }

    /// Replacing the source between two calls (new content, same name)
    /// must upload the new audio, never a stale prepared artifact.
    #[tokio::test(flavor = "current_thread")]
    async fn perf_prepare_source_replacement_reprepares() {
        let _iso = isolate();
        let dir = tempfile::tempdir().expect("tmp");
        let server = provider().await;
        let cfg = config(dir.path(), &server.uri());
        let input = dir.path().join("memo.wav");
        let first = speech(3.0, 48_000, 0.0);
        write_stereo_wav(&input, &first);
        transcribe(&input, &cfg).await;
        // Drop the transcript sidecar so the second call reaches the API.
        for e in std::fs::read_dir(dir.path()).expect("dir").flatten() {
            if e.path().extension().is_some_and(|x| x == "yml") {
                let _ = std::fs::remove_file(e.path());
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        let second = speech(3.5, 48_000, 1.7);
        write_stereo_wav(&input, &second);
        transcribe(&input, &cfg).await;
        let ups = uploads(&server).await;
        assert_eq!(ups.len(), 2);
        assert_upload_profile("after replacement", &ups[1].2, &downsample(&second));
    }

    /// Transcript-cache hit: no upload and no preparation at all.
    #[tokio::test(flavor = "current_thread")]
    async fn perf_prepare_cache_hit_does_no_work() {
        let _iso = isolate();
        let dir = tempfile::tempdir().expect("tmp");
        let server = provider().await;
        let cfg = config(dir.path(), &server.uri());
        let input = dir.path().join("memo.wav");
        write_stereo_wav(&input, &speech(2.0, 48_000, 0.0));
        transcribe(&input, &cfg).await;
        let encodes = thread_value(Counter::UploadEncodes);
        let decodes = thread_value(Counter::AudioFileDecodes);
        transcribe(&input, &cfg).await;
        assert_eq!(uploads(&server).await.len(), 1);
        assert_eq!(thread_value(Counter::UploadEncodes), encodes);
        assert_eq!(thread_value(Counter::AudioFileDecodes), decodes);
    }

    /// Undecodable input: raw bytes and original name are uploaded.
    #[tokio::test(flavor = "current_thread")]
    async fn perf_prepare_undecodable_uploads_raw_bytes() {
        let _iso = isolate();
        let dir = tempfile::tempdir().expect("tmp");
        let server = provider().await;
        let input = dir.path().join("weird.bin");
        std::fs::write(&input, b"not really audio").expect("write");
        transcribe(&input, &config(dir.path(), &server.uri())).await;
        let (name, _, bytes) = uploads(&server).await.remove(0);
        assert_eq!(name, "weird.bin");
        assert_eq!(bytes, b"not really audio");
    }

    /// Three concurrent callers for different models on the same file
    /// (the picker's pattern) each upload the complete converted audio.
    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn perf_prepare_concurrent_callers_get_complete_uploads() {
        let _iso = isolate();
        let dir = tempfile::tempdir().expect("tmp");
        let server = provider().await;
        let cfg = std::sync::Arc::new(config(dir.path(), &server.uri()));
        let pcm48 = speech(3.0, 48_000, 0.5);
        let input = std::sync::Arc::new(dir.path().join("memo.wav"));
        write_stereo_wav(&input, &pcm48);
        let mut tasks = Vec::new();
        for model in [
            "voxtral-mini-2602",
            "voxtral-mini-2507",
            "voxtral-mini-latest",
        ] {
            let (cfg, input) = (cfg.clone(), input.clone());
            tasks.push(tokio::spawn(async move {
                let sink: std::sync::Arc<dyn crate::telemetry::TelemetrySink> =
                    std::sync::Arc::new(crate::telemetry::NoOpSink);
                transcribe_audio(
                    &input,
                    &cfg,
                    Provider::Mistral,
                    Some(model),
                    false,
                    TranscribeOptions {
                        allow_api: true,
                        ..TranscribeOptions::default()
                    },
                    &sink,
                )
                .await
            }));
        }
        for t in tasks {
            assert_eq!(t.await.expect("join").expect("transcription").text, "ok");
        }
        let ups = uploads(&server).await;
        assert_eq!(ups.len(), 3);
        for (i, (_, _, bytes)) in ups.iter().enumerate() {
            assert_upload_profile(&format!("caller {i}"), bytes, &downsample(&pcm48));
        }
    }
}
