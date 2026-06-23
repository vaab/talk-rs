//! Chunk paste-node: split into word-bounded sub-chunks and forward
//! each to a child node.  Emits cumulative `PasteProgress` telemetry,
//! matching legacy behaviour byte-for-byte.

use crate::error::TalkError;
use crate::paste::node::{PasteCtx, PasteNode};
use crate::paste::split_into_char_chunks;
use crate::telemetry::TranscriptionEvent;
use async_trait::async_trait;

pub(crate) struct ChunkNode {
    pub(crate) chunk_chars: usize,
    pub(crate) child: Box<dyn PasteNode>,
}

#[async_trait]
impl PasteNode for ChunkNode {
    async fn paste(&self, text: &str, ctx: &PasteCtx<'_>) -> Result<(), TalkError> {
        let total_chars = text.len() as u64;

        if self.chunk_chars == 0 {
            // `chunk_chars: 0` ⇒ single-shot (legacy semantics).
            self.child.paste(text, ctx).await?;
            ctx.sink.emit(TranscriptionEvent::PasteProgress {
                chars_pasted: total_chars,
                total_chars,
                t: std::time::Instant::now(),
            });
            return Ok(());
        }

        let chunks = split_into_char_chunks(text, self.chunk_chars);
        let n_chunks = chunks.len();
        log::trace!("paste(chunk-node): split into {} chunk(s)", n_chunks);

        let mut chars_pasted: u64 = 0;
        for (idx, chunk) in chunks.iter().enumerate() {
            log::trace!(
                "paste(chunk-node): chunk {}/{} ({} bytes)",
                idx + 1,
                n_chunks,
                chunk.len(),
            );
            self.child.paste(chunk, ctx).await?;
            chars_pasted += chunk.len() as u64;
            ctx.sink.emit(TranscriptionEvent::PasteProgress {
                chars_pasted,
                total_chars,
                t: std::time::Instant::now(),
            });
        }

        Ok(())
    }
}
