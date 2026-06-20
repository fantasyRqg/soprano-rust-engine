//! Public streaming-decoder contract: every write a `StreamingDecode` makes
//! carries the stream's tag, and a stream that produces no audio writes nothing.
//! (The lossless-vs-whole-decode guarantee is verified by the in-crate unit
//! tests in `decoder.rs`, which can reach the private one-shot `run_decoder`.)

use soprano_core::inference::decoder::StreamingDecode;
use soprano_core::inference::session::{load_session, HIDDEN_DIM, STREAM_HOLDBACK_FRAMES};
use soprano_core::{AudioSink, ExecutionProvider, SinkError};

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

/// Records every `write` as `(tag, sample_count)` so a test can assert which
/// tag (if any) audio was attributed to.
struct TagRecorder(Vec<(u64, usize)>);
impl AudioSink for TagRecorder {
    fn write(&mut self, tag: u64, samples: &[i16]) -> Result<usize, SinkError> {
        self.0.push((tag, samples.len()));
        Ok(samples.len())
    }
    fn available(&self) -> usize {
        usize::MAX
    }
    fn on_drain_complete(&mut self) {}
    fn on_error(&mut self, _: u64, _: String) {}
}

/// Regression guard for the tagged-buffer contract (deterministic, no backbone):
/// a stream that synthesizes no audio must write nothing at all, so a sentence
/// the model collapses to zero frames produces no tagged audio rather than a
/// boundary the host can't represent. Mirrors the worker's hallucination path,
/// where `finish()` is deliberately skipped and the held-back frames dropped.
#[test]
fn zero_output_stream_writes_no_tagged_audio() {
    let Some(path) = decoder_path() else {
        eprintln!("Skipping: decoder model not found");
        return;
    };
    let mut session = load_session(&path, &ExecutionProvider::Cpu).expect("load decoder");

    // Fewer frames buffered than the holdback margin (CHUNK_SIZE + margin), so
    // nothing is ever emittable mid-stream; then drop the stream without
    // finishing — exactly what the worker does on a hallucination run.
    let hidden = synth(STREAM_HOLDBACK_FRAMES);
    let mut sink = TagRecorder(Vec::new());
    let no_cancel = || false;
    {
        let mut stream = StreamingDecode::new(&mut session, STREAM_HOLDBACK_FRAMES, 7);
        for h in &hidden {
            assert!(
                stream.push(h, &mut sink, &no_cancel).expect("push"),
                "push should not signal stop without cancellation"
            );
        }
        // No finish() — the held-back tail is intentionally dropped.
        assert_eq!(
            stream.total_samples_written(),
            0,
            "nothing should be emittable while still inside the holdback margin"
        );
    }
    assert!(
        sink.0.is_empty(),
        "a zero-output stream must not write any tagged audio, got {:?}",
        sink.0
    );
}

/// Every sample a stream writes must carry that stream's tag, so the host can
/// attribute played audio to the right sentence with no separate boundary
/// callback to correlate or lose.
#[test]
fn tagged_writes_carry_the_streams_tag() {
    let Some(path) = decoder_path() else {
        eprintln!("Skipping: decoder model not found");
        return;
    };
    let mut session = load_session(&path, &ExecutionProvider::Cpu).expect("load decoder");

    // Enough frames to clear the holdback margin and emit several windows.
    let hidden = synth(STREAM_HOLDBACK_FRAMES + 40);
    let mut sink = TagRecorder(Vec::new());
    let no_cancel = || false;
    {
        let mut stream = StreamingDecode::new(&mut session, STREAM_HOLDBACK_FRAMES, 42);
        for h in &hidden {
            assert!(
                stream.push(h, &mut sink, &no_cancel).expect("push"),
                "push should not signal stop without cancellation"
            );
        }
        assert!(
            stream.finish(&mut sink, &no_cancel).expect("finish"),
            "finish should not signal stop without cancellation"
        );
    }

    let total: usize = sink.0.iter().map(|(_, n)| n).sum();
    assert!(total > 0, "expected this stream to produce audio");
    assert!(
        sink.0.iter().all(|&(tag, _)| tag == 42),
        "every write must carry the stream's tag (42), got {:?}",
        sink.0
    );
}
