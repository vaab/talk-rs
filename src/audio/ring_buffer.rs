//! Fixed-capacity ring buffer of `f32` audio samples.
//!
//! A tiny wrap-around sample buffer shared between the audio capture
//! tee and the X11 overlay visualizer.  Lives in the core `audio`
//! module (no heavy dependencies) so the capture tee can use it
//! without pulling in the `ui` feature; the X11 rendering code
//! re-exports it from `x11::render_util` for backwards compatibility.

pub struct RingBuffer {
    data: Vec<f32>,
    write_pos: usize,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            write_pos: 0,
            capacity,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            self.data[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
    }

    pub fn read_last(&self, n: usize) -> Vec<f32> {
        let n = n.min(self.capacity);
        let mut out = Vec::with_capacity(n);
        let start = (self.write_pos + self.capacity - n) % self.capacity;
        for i in 0..n {
            out.push(self.data[(start + i) % self.capacity]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_push_and_read() {
        let mut rb = RingBuffer::new(8);
        rb.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = rb.read_last(5);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn ring_buffer_wraps_around() {
        let mut rb = RingBuffer::new(4);
        rb.push(&[1.0, 2.0, 3.0, 4.0]);
        rb.push(&[5.0, 6.0]);
        let out = rb.read_last(4);
        assert_eq!(out, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn ring_buffer_read_more_than_capacity() {
        let mut rb = RingBuffer::new(4);
        rb.push(&[1.0, 2.0]);
        let out = rb.read_last(10);
        // Clamped to capacity
        assert_eq!(out.len(), 4);
    }
}
