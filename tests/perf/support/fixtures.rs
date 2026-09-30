//! Fixture factory: speech-like audio and recording libraries,
//! generated at test time with the project's own writers.

use std::path::{Path, PathBuf};

use talk_rs::audio::{AudioWriter, OggOpusWriter, WavWriter};
use talk_rs::config::AudioConfig;

/// Deterministic speech-like mono PCM: a harmonic "voice" (120-180 Hz
/// fundamental, formant-weighted overtones) amplitude-modulated into
/// ~300 ms syllables with short pauses, plus low noise.  Loud enough
/// for the overlay's RMS / live-audio detectors, not clipped.
pub fn speech_pcm(seconds: f64, rate: u32) -> Vec<i16> {
    let n = (seconds * rate as f64) as usize;
    let mut seed = 0x2545_f491_u32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / rate as f64;
        let syllable = (t / 0.37).floor();
        let phase = (t / 0.37).fract();
        let envelope = if phase < 0.8 {
            (std::f64::consts::PI * phase / 0.8).sin()
        } else {
            0.0
        };
        let f0 = 120.0 + 60.0 * ((syllable * 1.7).sin() * 0.5 + 0.5);
        let mut v = 0.0;
        for (k, weight) in [(1.0, 1.0), (2.0, 0.6), (3.0, 0.45), (5.0, 0.3), (8.0, 0.15)] {
            v += weight * (2.0 * std::f64::consts::PI * f0 * k * t).sin();
        }
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let noise = (seed as f64 / u32::MAX as f64 - 0.5) * 0.02;
        out.push(((v / 2.5 * envelope + noise) * 12_000.0) as i16);
    }
    out
}

/// Write 16-bit mono PCM as a WAV file.
pub fn write_wav(path: &Path, pcm: &[i16], rate: u32) {
    let mut writer = WavWriter::new(AudioConfig {
        sample_rate: rate,
        channels: 1,
        bitrate: 32_000,
    });
    let mut bytes = writer.header().expect("WAV header");
    bytes.extend(writer.write_pcm(pcm).expect("WAV audio"));
    let header = writer.finalize().expect("final WAV header");
    bytes[..header.len()].copy_from_slice(&header);
    std::fs::write(path, bytes).expect("write WAV fixture");
}

/// Encoding profile for an OGG fixture.
#[derive(Clone, Copy)]
pub enum OggProfile {
    /// 16 kHz mono Voip — exactly what dictate's cache writer produces.
    Transcription,
    /// 48 kHz mono Audio — what `talk-rs record` produces.
    Recording,
}

/// Write an Opus OGG file with the project writer, streaming the PCM
/// in 20 ms chunks so long fixtures never hold all PCM in memory.
pub fn write_ogg(path: &Path, seconds: f64, profile: OggProfile) {
    let (rate, mut writer) = match profile {
        OggProfile::Transcription => (
            16_000,
            OggOpusWriter::new(AudioConfig::new()).expect("ogg writer"),
        ),
        OggProfile::Recording => {
            let cfg = AudioConfig {
                sample_rate: 48_000,
                channels: 1,
                bitrate: 64_000,
            };
            (
                48_000,
                OggOpusWriter::new_for_recording(cfg).expect("ogg writer"),
            )
        }
    };
    let mut bytes = writer.header().expect("ogg header");
    let block = 60.0;
    let mut done = 0.0;
    while done < seconds {
        let len = (seconds - done).min(block);
        let pcm = speech_pcm(len, rate);
        bytes.extend(writer.write_pcm(&pcm).expect("ogg audio"));
        done += len;
    }
    bytes.extend(writer.finalize().expect("ogg finalize"));
    std::fs::write(path, bytes).expect("write OGG fixture");
}

/// Bump when any generator in this module changes its output: cached
/// fixtures are keyed by this version, profile and duration, and their
/// content hash is verified on every use.
pub const FIXTURE_VERSION: u32 = 2;

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Long fixtures are costly to encode; keep them once per checkout
/// under `target/perf-fixtures/`, keyed by generator version, profile
/// and duration, with a content hash (`.fnv`) checked on every use so a
/// stale or damaged file is regenerated rather than silently reused.
pub fn cached_ogg(name: &str, seconds: f64, profile: OggProfile) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/perf-fixtures");
    std::fs::create_dir_all(&dir).expect("fixture cache dir");
    let tag = match profile {
        OggProfile::Transcription => "upload16k",
        OggProfile::Recording => "record48k",
    };
    let key = format!("{name}-v{FIXTURE_VERSION}-{tag}-{seconds}s");
    let path = dir.join(format!("{key}.ogg"));
    let hash_path = dir.join(format!("{key}.fnv"));
    let valid = std::fs::read(&path)
        .ok()
        .zip(std::fs::read_to_string(&hash_path).ok())
        .is_some_and(|(bytes, hash)| hash.trim() == format!("{:016x}", fnv64(&bytes)));
    if !valid {
        let tmp = dir.join(format!("{key}.ogg.partial"));
        write_ogg(&tmp, seconds, profile);
        let bytes = std::fs::read(&tmp).expect("fixture bytes");
        std::fs::rename(&tmp, &path).expect("publish fixture");
        std::fs::write(&hash_path, format!("{:016x}\n", fnv64(&bytes))).expect("fixture hash");
    }
    path
}

/// Shape of a synthetic recordings library.
pub struct LibrarySpec {
    pub recordings: usize,
    /// Recordings without a `.pick.yml` (they get an audio player bar
    /// unless they carry a pick lock).
    pub without_pick: usize,
    /// Among the no-pick OGG rows: valid `.wf` waveform caches (warm).
    pub waveform_warm: usize,
    /// … `.wf` caches older than their audio (stale → recomputed).
    pub waveform_stale: usize,
    /// … truncated `.wf` caches (corrupt → recomputed).
    pub waveform_corrupt: usize,
    /// No-pick rows that are imported `.mp4` (AAC) files.  (The browser
    /// lists `.ogg`, `.m4a`, `.mp4` and `.aac` in `output_dir`; `.wav`
    /// only in the dictation cache.)
    pub imported_mp4: usize,
    /// No-pick rows that are imported M4A (AAC) files.
    pub imported_m4a: usize,
    /// No-pick rows that are long recordings (`long_seconds`).
    pub long_rows: usize,
    pub long_seconds: f64,
    /// Rows with a pick lock and no pick ("transcription ongoing").
    pub in_progress: usize,
    pub seconds_each: f64,
}

impl LibrarySpec {
    /// The user's measured library shape — 1504 audio files, 1304
    /// picks, 883 `.wf` — scaled to 1500 rows, with the realistic mix
    /// of formats, waveform-cache states and lengths among the ~200
    /// rows rendered as audio player bars.
    pub fn user_like() -> Self {
        Self {
            recordings: 1500,
            without_pick: 200,
            waveform_warm: 60,
            waveform_stale: 10,
            waveform_corrupt: 10,
            imported_mp4: 10,
            imported_m4a: 10,
            long_rows: 1,
            long_seconds: 600.0,
            in_progress: 5,
            seconds_each: 2.0,
        }
    }
}

/// What the recordings browser must show for one library entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectedRow {
    /// A transcript row showing this text.
    Transcript(String),
    /// An audio player bar (no pick).
    Player,
    /// "(transcription ongoing)" (pick lock, no pick).
    InProgress,
}

/// A built library: every audio path with what its row must show.
pub struct Library {
    pub rows: Vec<(PathBuf, ExpectedRow)>,
    /// The long no-pick recordings (Play cut targets).
    pub long_rows: Vec<PathBuf>,
}

impl Library {
    /// Expected display order: newest first by file name.
    pub fn expected_order(&self) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = self.rows.iter().map(|(p, _)| p.clone()).collect();
        paths.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
        paths
    }
}

/// Write a `.wf` waveform cache in the browser's binary format
/// (u32 columns, u32 rows, f32 peak, column-major f32 data).
fn write_waveform(path: &Path, columns: u32, rows: u32) {
    let mut bytes = Vec::new();
    bytes.extend(columns.to_le_bytes());
    bytes.extend(rows.to_le_bytes());
    bytes.extend(0.5f32.to_le_bytes());
    for i in 0..columns * rows {
        bytes.extend(((i % 17) as f32 / 17.0).to_le_bytes());
    }
    std::fs::write(path, bytes).expect("waveform cache");
}

fn set_mtime(path: &Path, when: std::time::SystemTime) {
    let file = std::fs::File::options()
        .write(true)
        .open(path)
        .expect("open for mtime");
    file.set_modified(when).expect("set mtime");
}

/// Build a `YYYY/MM/<timestamp>.<ext>` library matching `spec`.  OGG
/// rows share one encoded payload (copied), so 1500 entries take about
/// a second; long rows hard-link a cached fixture.
pub fn build_library(root: &Path, spec: &LibrarySpec) -> Library {
    let template = root.join(".template.ogg");
    write_ogg(&template, spec.seconds_each, OggProfile::Recording);
    let payload = std::fs::read(&template).expect("template");
    std::fs::remove_file(&template).expect("remove template");
    let m4a_payload = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sine_440_0.5s_mono.m4a"),
    )
    .expect("m4a fixture");
    let long = (spec.long_rows > 0)
        .then(|| cached_ogg("library-long", spec.long_seconds, OggProfile::Recording));
    let past = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);

    let mut library = Library {
        rows: Vec::with_capacity(spec.recordings),
        long_rows: Vec::new(),
    };
    // No-pick rows are split, in order, into: in-progress, long, MP4,
    // M4A, then OGG with warm / stale / corrupt / no waveform cache.
    let mut cursor = 0;
    let mut take = |n: usize| {
        let range = cursor..cursor + n;
        cursor += n;
        range
    };
    let in_progress = take(spec.in_progress);
    let long_range = take(spec.long_rows);
    let mp4_range = take(spec.imported_mp4);
    let m4a_range = take(spec.imported_m4a);
    let warm = take(spec.waveform_warm);
    let stale = take(spec.waveform_stale);
    let corrupt = take(spec.waveform_corrupt);
    assert!(
        cursor <= spec.without_pick,
        "library spec: too many special no-pick rows"
    );

    for i in 0..spec.recordings {
        let day = i / 24;
        let month = 1 + (day / 28) % 12;
        let dom = 1 + day % 28;
        let year = 2025 + day / (28 * 12);
        let dir = root.join(format!("{year:04}/{month:02}"));
        std::fs::create_dir_all(&dir).expect("library dir");
        let stem = format!(
            "{year:04}-{month:02}-{dom:02}T{:02}-{:02}-00+0200",
            i % 24,
            (i * 7) % 60
        );
        let ext = if mp4_range.contains(&i) {
            "mp4"
        } else if m4a_range.contains(&i) {
            "m4a"
        } else {
            "ogg"
        };
        let audio = dir.join(format!("{stem}.{ext}"));
        match (ext, &long) {
            ("m4a" | "mp4", _) => std::fs::write(&audio, &m4a_payload).expect("aac row"),
            (_, Some(long)) if long_range.contains(&i) => {
                std::fs::hard_link(long, &audio)
                    .or_else(|_| std::fs::copy(long, &audio).map(|_| ()))
                    .expect("long row");
                library.long_rows.push(audio.clone());
            }
            _ => std::fs::write(&audio, &payload).expect("ogg row"),
        }
        let expected = if i >= spec.without_pick {
            let text = format!("recording number {i} transcript text");
            std::fs::write(
                dir.join(format!("{stem}.pick.yml")),
                format!(
                    "provider: mistral\nmodel: voxtral-mini-2602\nstreaming: false\ntext: {text}\n"
                ),
            )
            .expect("pick file");
            ExpectedRow::Transcript(text)
        } else if in_progress.contains(&i) {
            std::fs::write(dir.join(format!("{stem}.pick-lock.yml")), "").expect("pick lock");
            ExpectedRow::InProgress
        } else {
            let wf = dir.join(format!("{stem}.wf"));
            if warm.contains(&i) {
                set_mtime(&audio, past);
                write_waveform(&wf, 256, 64);
            } else if stale.contains(&i) {
                write_waveform(&wf, 256, 64);
                set_mtime(&wf, past);
            } else if corrupt.contains(&i) {
                set_mtime(&audio, past);
                std::fs::write(&wf, [1u8, 0, 0]).expect("corrupt wf");
            }
            ExpectedRow::Player
        };
        library.rows.push((audio, expected));
    }
    library
}

/// Duration (s) of an Opus OGG byte stream from its last page granule.
pub fn ogg_duration(bytes: &[u8]) -> f64 {
    let mut granule = 0u64;
    let mut pos = 0;
    while let Some(off) = bytes[pos..].windows(4).position(|w| w == b"OggS") {
        let page = pos + off;
        if page + 27 > bytes.len() {
            break;
        }
        let g = u64::from_le_bytes(bytes[page + 6..page + 14].try_into().unwrap_or([0; 8]));
        if g != u64::MAX {
            granule = g;
        }
        pos = page + 4;
    }
    // Opus granules count 48 kHz samples; pre-skip is 312 for libopus.
    granule.saturating_sub(312) as f64 / 48_000.0
}

/// OGG bytes with stream serials and page CRCs zeroed, so two encodes
/// of the same audio compare equal despite random stream serials.
pub fn ogg_without_serials(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    let mut pos = 0;
    while let Some(off) = out[pos..].windows(4).position(|w| w == b"OggS") {
        let page = pos + off;
        if page + 27 > out.len() {
            break;
        }
        out[page + 14..page + 18].fill(0);
        out[page + 22..page + 26].fill(0);
        pos = page + 4;
    }
    out
}

/// The single `.ogg` recording written under `dir` (dictate's cache).
pub fn only_ogg_in(dir: &Path) -> PathBuf {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|x| x == "ogg")
                && !p
                    .symlink_metadata()
                    .is_ok_and(|m| m.file_type().is_symlink())
        })
        .collect();
    assert_eq!(
        found.len(),
        1,
        "expected one cache recording in {}",
        dir.display()
    );
    found.remove(0)
}
