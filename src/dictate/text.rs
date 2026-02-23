//! Text utilities for dictation: sentence splitting with terminal feedback.

/// Flush completed sentences from the live buffer to stdout.
///
/// Scans the buffer for sentence-ending punctuation (`.` `!` `?` `。` `！` `？`)
/// followed by whitespace. Everything up to and including the punctuation is
/// emitted as a line on stdout and appended to `segments`. The remainder stays
/// in the buffer for further accumulation.
pub(super) fn flush_sentences(buffer: &mut String, segments: &mut Vec<String>) {
    loop {
        // Find the earliest sentence-ending punctuation followed by whitespace.
        let boundary = buffer.char_indices().position(|(i, ch)| {
            if matches!(ch, '。' | '！' | '？') {
                // CJK sentence-ending punctuation: always a boundary
                // (no space expected between CJK sentences)
                true
            } else if matches!(ch, '.' | '!' | '?') {
                // Latin sentence-ending punctuation: require whitespace
                // or end-of-string after it to avoid splitting "3.14"
                let after = i + ch.len_utf8();
                after >= buffer.len() || buffer[after..].starts_with(|c: char| c.is_whitespace())
            } else {
                false
            }
        });

        let Some(pos) = boundary else {
            break;
        };

        // Convert char position back to byte offset (including the punctuation char)
        let (byte_offset, punct_char) = buffer.char_indices().nth(pos).unwrap_or((0, '.'));
        let split_at = byte_offset + punct_char.len_utf8();

        let sentence = buffer[..split_at].trim().to_string();
        if !sentence.is_empty() {
            println!("{}", sentence);
            segments.push(sentence);
        }

        // Remove the emitted sentence + any leading whitespace from the remainder
        let remainder = buffer[split_at..].trim_start().to_string();
        // Clear stderr live preview
        let blank = " ".repeat(buffer.len());
        eprint!("\r{}\r", blank);

        *buffer = remainder;
        if !buffer.is_empty() {
            eprint!("\r{}", buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flush_sentences_single_sentence() {
        let mut buf = "Hello world.".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["Hello world."]);
        assert_eq!(buf, "");
    }

    #[test]
    fn test_flush_sentences_trailing_partial() {
        let mut buf = "First sentence. And then".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["First sentence."]);
        assert_eq!(buf, "And then");
    }

    #[test]
    fn test_flush_sentences_multiple() {
        let mut buf = "One. Two! Three? Rest".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["One.", "Two!", "Three?"]);
        assert_eq!(buf, "Rest");
    }

    #[test]
    fn test_flush_sentences_no_punctuation() {
        let mut buf = "no punctuation here".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert!(segs.is_empty());
        assert_eq!(buf, "no punctuation here");
    }

    #[test]
    fn test_flush_sentences_chinese_punctuation() {
        let mut buf = "你好。世界".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["你好。"]);
        assert_eq!(buf, "世界");
    }

    #[test]
    fn test_flush_sentences_period_at_end() {
        let mut buf = "End of text.".to_string();
        let mut segs: Vec<String> = Vec::new();
        flush_sentences(&mut buf, &mut segs);
        assert_eq!(segs, vec!["End of text."]);
        assert_eq!(buf, "");
    }
}
