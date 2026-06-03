//! Streaming decoder must be lossless: feeding hidden states through the
//! sliding-window `decode_streaming` must reproduce the audio produced by
//! decoding the whole sequence at once with `decode_all`.

use soprano_core::audio::convert::f32_to_i16;
use soprano_core::inference::decoder::{decode_all, decode_streaming};
use soprano_core::inference::session::{load_session, HIDDEN_DIM};
use soprano_core::{AudioSink, ExecutionProvider, SinkError};

struct Collector(Vec<i16>);
impl AudioSink for Collector {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        self.0.extend_from_slice(samples);
        Ok(samples.len())
    }
    fn available(&self) -> usize {
        usize::MAX
    }
    fn on_sentence_complete(&mut self, _: usize) {}
    fn on_drain_complete(&mut self) {}
    fn on_error(&mut self, _: String) {}
}

fn decoder_path() -> Option<std::path::PathBuf> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    ["soprano_decoder_f16.onnx", "soprano_decoder.onnx"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists())
}

// Deterministic synthetic hidden states (no backbone needed).
fn synth(n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..HIDDEN_DIM)
                .map(|d| (((i * 31 + d * 7) % 97) as f32 / 97.0) - 0.5)
                .collect()
        })
        .collect()
}

#[test]
fn streaming_decode_matches_whole_decode() {
    let Some(path) = decoder_path() else {
        eprintln!("Skipping: decoder model not found");
        return;
    };
    let mut session = load_session(&path, &ExecutionProvider::Cpu).expect("load decoder");

    // Span several windows (CHUNK_SIZE=8) including a non-aligned tail.
    let hidden = synth(45);

    let whole: Vec<i16> = decode_all(&mut session, &hidden)
        .expect("decode_all")
        .iter()
        .map(|&s| f32_to_i16(s))
        .collect();

    let mut sink = Collector(Vec::new());
    decode_streaming(&mut session, &hidden, &mut sink).expect("decode_streaming");
    let streamed = sink.0;

    assert_eq!(
        streamed.len(),
        whole.len(),
        "streamed length {} != whole-decode length {}",
        streamed.len(),
        whole.len()
    );

    let max_diff = streamed
        .iter()
        .zip(&whole)
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 1,
        "streamed audio diverges from whole decode: max abs i16 diff = {max_diff}"
    );
}
