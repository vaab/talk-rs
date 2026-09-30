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

/// Long fixtures are costly to encode; keep them once per checkout
/// under `target/perf-fixtures/` (keyed by name) instead of per test.
pub fn cached_ogg(name: &str, seconds: f64, profile: OggProfile) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/perf-fixtures");
    std::fs::create_dir_all(&dir).expect("fixture cache dir");
    let path = dir.join(format!("{name}.ogg"));
    if !path.exists() {
        let tmp = dir.join(format!("{name}.ogg.partial"));
        write_ogg(&tmp, seconds, profile);
        std::fs::rename(&tmp, &path).expect("publish fixture");
    }
    path
}

/// Shape of a synthetic recordings library.
pub struct LibrarySpec {
    pub recordings: usize,
    /// Recordings without a `.pick.yml` (they get an audio player bar).
    pub without_pick: usize,
    /// Recordings (among those without pick) that already have `.wf`.
    pub with_waterfall: usize,
    pub seconds_each: f64,
}

impl LibrarySpec {
    /// The user's measured library shape: 1504 audio files, 1304
    /// picks, i.e. ~200 rows rendered as audio player bars.
    pub fn user_like() -> Self {
        Self {
            recordings: 1500,
            without_pick: 200,
            with_waterfall: 0,
            seconds_each: 2.0,
        }
    }
}

/// Build a `YYYY/MM/<timestamp>.ogg` library.  All recordings share one
/// encoded payload (copied), so building 1500 entries takes a second.
pub fn build_library(root: &Path, spec: &LibrarySpec) -> Vec<PathBuf> {
    let template = root.join(".template.ogg");
    write_ogg(&template, spec.seconds_each, OggProfile::Recording);
    let payload = std::fs::read(&template).expect("template");
    std::fs::remove_file(&template).expect("remove template");
    let mut paths = Vec::with_capacity(spec.recordings);
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
        let audio = dir.join(format!("{stem}.ogg"));
        std::fs::write(&audio, &payload).expect("library audio");
        if i >= spec.without_pick {
            std::fs::write(
                dir.join(format!("{stem}.pick.yml")),
                format!(
                    "provider: mistral\nmodel: voxtral-mini-2602\nstreaming: false\ntext: recording number {i} transcript text\n"
                ),
            )
            .expect("pick file");
        }
        paths.push(audio);
    }
    paths
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
