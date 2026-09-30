//! Test-only audio verification shared by the performance cuts:
//! strict Ogg Opus parsing/decoding and a content comparator that is
//! meaningful for a perceptual codec.

/// Strictly parse and decode an Ogg Opus stream; panics on any
/// structural problem.  Returns 16 kHz mono PCM trimmed to the
/// final granule (pre-skip removed), i.e. exactly what a
/// conforming player outputs.
pub(crate) fn decode_ogg_strict(bytes: &[u8]) -> Vec<i16> {
    let mut reader = ogg::reading::PacketReader::new(std::io::Cursor::new(bytes));
    let head = reader
        .read_packet()
        .expect("ogg read")
        .expect("OpusHead packet");
    assert!(head.first_in_stream(), "OpusHead must open the stream");
    assert_eq!(&head.data[..8], b"OpusHead");
    assert_eq!(head.data[9], 1, "mono");
    let pre_skip = u16::from_le_bytes([head.data[10], head.data[11]]) as u64;
    let serial = head.stream_serial();
    let tags = reader
        .read_packet()
        .expect("ogg read")
        .expect("OpusTags packet");
    assert_eq!(&tags.data[..8], b"OpusTags");
    let mut decoder = opus::Decoder::new(16_000, opus::Channels::Mono).expect("decoder");
    let mut pcm = Vec::new();
    let mut last_granule = 0u64;
    let mut saw_eos = false;
    while let Some(packet) = reader.read_packet().expect("ogg read") {
        assert!(!saw_eos, "packet after end of stream");
        assert_eq!(packet.stream_serial(), serial, "single logical stream");
        let g = packet.absgp_page();
        assert!(g >= last_granule, "granule went backwards");
        last_granule = g;
        let mut out = vec![0i16; 5760];
        let n = decoder
            .decode(&packet.data, &mut out, false)
            .expect("opus decode");
        pcm.extend_from_slice(&out[..n]);
        saw_eos = packet.last_in_stream();
    }
    assert!(saw_eos, "stream not finalized (no EOS page)");
    // Granules count 48 kHz samples; the decoder runs at 16 kHz.
    let skip = (pre_skip / 3) as usize;
    let total = (last_granule.saturating_sub(pre_skip) / 3) as usize;
    assert!(
        pcm.len() >= skip + total,
        "granule claims more audio than decoded"
    );
    pcm[skip..skip + total].to_vec()
}

/// 10 ms RMS envelope.
fn envelope(pcm: &[i16]) -> Vec<f64> {
    pcm.chunks(160)
        .map(|c| (c.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / c.len() as f64).sqrt())
        .collect()
}

/// Correlation of the 10 ms energy envelopes of `reference` and
/// `decoded` (best over a ±30 ms lag).  Opus Voip is perceptual,
/// so sample-level error is meaningless; the envelope tracks the
/// content, its order and its silences, and drops towards 0 for
/// silence, noise, reordered or truncated audio.
pub(crate) fn envelope_correlation(reference: &[i16], decoded: &[i16]) -> f64 {
    let a = envelope(reference);
    let b = envelope(decoded);
    let mut best = f64::MIN;
    for lag in -3i64..=3 {
        let pairs: Vec<(f64, f64)> = (0..a.len() as i64)
            .filter_map(|i| {
                let j = i + lag;
                (j >= 0 && (j as usize) < b.len()).then(|| (a[i as usize], b[j as usize]))
            })
            .collect();
        let n = pairs.len() as f64;
        if n < 2.0 {
            continue;
        }
        let (ma, mb) = (
            pairs.iter().map(|p| p.0).sum::<f64>() / n,
            pairs.iter().map(|p| p.1).sum::<f64>() / n,
        );
        let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
        for (x, y) in &pairs {
            cov += (x - ma) * (y - mb);
            va += (x - ma).powi(2);
            vb += (y - mb).powi(2);
        }
        best = best.max(cov / (va.sqrt() * vb.sqrt()).max(1e-9));
    }
    best
}

/// Speech-like reference used by the comparator self-test.
#[cfg(test)]
fn reference() -> Vec<i16> {
    (0..50_000)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            let env = (t * 2.7 * std::f32::consts::TAU).sin().abs();
            (env * (t * 190.0 * std::f32::consts::TAU).sin() * 20_000.0) as i16
        })
        .collect()
}

/// The comparator rejects what the guards exist to catch.
#[test]
fn perf_single_encode_comparator_rejects_wrong_audio() {
    let reference = reference();
    assert!(envelope_correlation(&reference, &reference) > 0.999);
    let silence = vec![0i16; reference.len()];
    assert!(envelope_correlation(&reference, &silence) < 0.5);
    let mut reordered = reference[reference.len() / 2..].to_vec();
    reordered.extend_from_slice(&reference[..reference.len() / 2]);
    assert!(envelope_correlation(&reference, &reordered) < 0.9);
    let tone: Vec<i16> = (0..reference.len())
        .map(|i| ((i as f32 * 0.07).sin() * 8000.0) as i16)
        .collect();
    assert!(envelope_correlation(&reference, &tone) < 0.5);
}
