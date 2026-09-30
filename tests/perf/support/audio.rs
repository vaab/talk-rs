//! Strict audio verification for binary cuts (mirror of the crate's
//! test-only `audio::perf_audio`, which integration tests cannot see).

/// Strictly parse and decode an Ogg Opus stream (one logical stream,
/// OpusHead/OpusTags, monotonic granules, EOS); panics on any
/// structural problem.  Returns mono PCM at `rate` trimmed to the final
/// granule, i.e. exactly what a conforming player outputs.
pub fn decode_ogg_strict(bytes: &[u8], rate: u32) -> Vec<i16> {
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
    let mut decoder = opus::Decoder::new(rate, opus::Channels::Mono).expect("decoder");
    let mut pcm = Vec::new();
    let (mut last_granule, mut saw_eos) = (0u64, false);
    while let Some(packet) = reader.read_packet().expect("ogg read") {
        assert!(!saw_eos, "packet after end of stream");
        assert_eq!(packet.stream_serial(), serial, "single logical stream");
        let g = packet.absgp_page();
        assert!(g >= last_granule, "granule went backwards");
        last_granule = g;
        let mut out = vec![0i16; 5760 * 3];
        let n = decoder
            .decode(&packet.data, &mut out, false)
            .expect("opus decode");
        pcm.extend_from_slice(&out[..n]);
        saw_eos = packet.last_in_stream();
    }
    assert!(saw_eos, "stream not finalized (no EOS page)");
    let ratio = 48_000 / rate as u64;
    let skip = (pre_skip / ratio) as usize;
    let total = (last_granule.saturating_sub(pre_skip) / ratio) as usize;
    assert!(
        pcm.len() >= skip + total,
        "granule claims more audio than decoded"
    );
    pcm[skip..skip + total].to_vec()
}

fn envelope(pcm: &[i16], rate: u32) -> Vec<f64> {
    pcm.chunks((rate / 100) as usize)
        .map(|c| (c.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / c.len() as f64).sqrt())
        .collect()
}

/// Correlation of 10 ms energy envelopes, best over a ±30 ms lag: high
/// for the same content in the same order; low for silence, noise,
/// reordered or truncated audio.
pub fn envelope_correlation(reference: &[i16], decoded: &[i16], rate: u32) -> f64 {
    let (a, b) = (envelope(reference, rate), envelope(decoded, rate));
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
        let ma = pairs.iter().map(|p| p.0).sum::<f64>() / n;
        let mb = pairs.iter().map(|p| p.1).sum::<f64>() / n;
        let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
        for (x, y) in &pairs {
            c += (x - ma) * (y - mb);
            va += (x - ma).powi(2);
            vb += (y - mb).powi(2);
        }
        best = best.max(c / (va.sqrt() * vb.sqrt()).max(1e-9));
    }
    best
}
