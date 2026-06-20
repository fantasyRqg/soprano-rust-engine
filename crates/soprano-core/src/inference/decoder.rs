//! Vocos decoder inference with sliding window for streaming.
//! Port of the decoder logic from soprano-web-onnx/onnx-streaming.js

use std::time::Instant;

use ort::session::Session;
use ort::value::Tensor;

use super::session::*;
use crate::audio::convert::f32_to_i16;
use crate::audio::sink::{AudioSink, SinkError};
use crate::profile::profile_log;

fn write_f32_pcm(
    sink: &mut dyn AudioSink,
    tag: u64,
    samples: &[f32],
    total_samples_written: &mut usize,
    should_cancel: &impl Fn() -> bool,
) -> Result<bool, String> {
    if samples.is_empty() {
        return Ok(true);
    }

    let i16_samples: Vec<i16> = samples.iter().map(|&s| f32_to_i16(s)).collect();
    let mut written = 0;
    while written < i16_samples.len() {
        if should_cancel() {
            *total_samples_written += written;
            return Ok(false);
        }

        match sink.write(tag, &i16_samples[written..]) {
            Ok(0) => return Err("sink write made no progress".to_string()),
            Ok(n) => written += n,
            // A closed sink aborts the feed like a cancellation: keep the
            // partial count but stop decoding — running inference against a
            // dead sink wastes work and lets sample offsets drift from what
            // the sink actually received.
            Err(SinkError::Closed) => {
                *total_samples_written += written;
                return Ok(false);
            }
            Err(e) => return Err(format!("sink write error: {}", e)),
        }
    }
    *total_samples_written += i16_samples.len();
    Ok(true)
}

/// Run the decoder on hidden states and return PCM f32 audio.
pub(crate) fn run_decoder(
    session: &mut Session,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<f32>, String> {
    if hidden_states.is_empty() {
        return Ok(Vec::new());
    }

    let num_tokens = hidden_states.len();

    // Transpose: (num_tokens, HIDDEN_DIM) → (1, HIDDEN_DIM, num_tokens)
    let mut decoder_input = vec![0.0f32; HIDDEN_DIM * num_tokens];
    for (w, hs) in hidden_states.iter().enumerate() {
        for d in 0..HIDDEN_DIM {
            decoder_input[d * num_tokens + w] = hs[d];
        }
    }

    let input_tensor = Tensor::from_array((
        [1usize, HIDDEN_DIM, num_tokens],
        decoder_input.into_boxed_slice(),
    ))
    .map_err(|e| format!("failed to create decoder input: {}", e))?;

    let input_name = session
        .inputs()
        .first()
        .map(|i| i.name())
        .unwrap_or("hidden_states");

    let outputs = session
        .run(vec![(
            std::borrow::Cow::from(input_name.to_string()),
            input_tensor.into_dyn(),
        )])
        .map_err(|e| format!("decoder inference failed: {}", e))?;

    let (_shape, audio_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("failed to extract audio: {}", e))?;

    Ok(audio_data.to_vec())
}

/// Incremental sliding-window decoder. Hidden states are pushed one token at a
/// time as the backbone produces them; each [`push`](Self::push) flushes any
/// windows that have gained enough right context (and cleared the holdback
/// margin) so audio streams out while generation is still running. Call
/// [`finish`](Self::finish) once generation ends cleanly to drain the margin.
///
/// The emitted audio is bit-for-bit identical to decoding the whole sequence at
/// once (`decode_all`): every window carries `RECEPTIVE_FIELD` tokens of
/// context on both sides, so there are no seams to crossfade.
pub struct StreamingDecode<'s> {
    session: &'s mut Session,
    /// All hidden states produced so far. ~2 KB/token, ≤ MAX_NEW_TOKENS, so the
    /// full buffer stays under ~1 MB — not worth a sliding window.
    hidden: Vec<Vec<f32>>,
    /// Index of the next frame to emit.
    next_frame: usize,
    /// Frames kept un-emitted behind the generation frontier (see
    /// `STREAM_HOLDBACK_FRAMES`). 0 emits as soon as right context exists.
    holdback: usize,
    total_samples_written: usize,
    /// App-defined tag this stream's audio belongs to, attached to every
    /// `AudioSink::write`. A stream that writes nothing (e.g. a hallucination
    /// run whose held-back frames are dropped) simply emits no tagged audio.
    tag: u64,
}

impl<'s> StreamingDecode<'s> {
    pub fn new(session: &'s mut Session, holdback: usize, tag: u64) -> Self {
        Self {
            session,
            hidden: Vec::new(),
            next_frame: 0,
            holdback,
            total_samples_written: 0,
            tag,
        }
    }

    pub fn total_samples_written(&self) -> usize {
        self.total_samples_written
    }

    /// Append one token's hidden state and flush any now-emittable windows.
    /// Returns `Ok(false)` if the sink stopped accepting (closed or cancelled),
    /// in which case the caller should abandon this stream.
    pub fn push(
        &mut self,
        hidden: &[f32],
        sink: &mut dyn AudioSink,
        should_cancel: &impl Fn() -> bool,
    ) -> Result<bool, String> {
        self.hidden.push(hidden.to_vec());
        self.drain(false, sink, should_cancel)
    }

    /// Drain every remaining buffered frame, ignoring the holdback margin. Call
    /// once generation has ended *cleanly* (EOS or token limit). On a
    /// hallucination/cancel the caller must NOT call this — the held-back frames
    /// are meant to be dropped.
    pub fn finish(
        &mut self,
        sink: &mut dyn AudioSink,
        should_cancel: &impl Fn() -> bool,
    ) -> Result<bool, String> {
        self.drain(true, sink, should_cancel)
    }

    fn drain(
        &mut self,
        final_flush: bool,
        sink: &mut dyn AudioSink,
        should_cancel: &impl Fn() -> bool,
    ) -> Result<bool, String> {
        let avail = self.hidden.len();
        // The decoder emits (W - 1) frames for a window of W tokens, so fewer
        // than 2 tokens yields no audio.
        if avail < 2 {
            return Ok(true);
        }
        let total_frames = avail - 1;

        while self.next_frame < total_frames {
            if should_cancel() {
                return Ok(false);
            }

            let chunk_end = (self.next_frame + CHUNK_SIZE).min(total_frames);

            // Mid-stream we only emit windows that have a full RECEPTIVE_FIELD of
            // right context AND sit behind the holdback margin; the final flush
            // drains the tail regardless. The margin (>= RECEPTIVE_FIELD) keeps
            // a detection window of frames un-emitted so a late hallucination
            // never reaches the sink.
            if !final_flush {
                let margin = self.holdback.max(RECEPTIVE_FIELD);
                if chunk_end + margin > avail {
                    break;
                }
            }

            // Decode a window with RECEPTIVE_FIELD tokens of context on BOTH
            // sides — lossless, no seam crossfade needed.
            let w0 = self.next_frame.saturating_sub(RECEPTIVE_FIELD);
            let w1 = (chunk_end + RECEPTIVE_FIELD).min(avail);
            let t_window = Instant::now();
            let audio = run_decoder(self.session, &self.hidden[w0..w1])?;
            profile_log!(
                "decoder window: {}ms ({} tokens decoded for {} frames emitted)",
                t_window.elapsed().as_millis(),
                w1 - w0,
                chunk_end - self.next_frame
            );

            // Emit frames [next_frame, chunk_end); local index = global - w0.
            let start = (self.next_frame - w0) * SAMPLES_PER_TOKEN;
            let end = ((chunk_end - w0) * SAMPLES_PER_TOKEN).min(audio.len());
            if start < end
                && !write_f32_pcm(
                    sink,
                    self.tag,
                    &audio[start..end],
                    &mut self.total_samples_written,
                    should_cancel,
                )?
            {
                return Ok(false);
            }

            self.next_frame = chunk_end;
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    //! The sliding-window streaming decoder must be lossless: feeding hidden
    //! states through `StreamingDecode` must reproduce the audio produced by
    //! decoding the whole sequence at once with `run_decoder` (the oracle).
    use super::*;
    use crate::audio::convert::f32_to_i16;
    use crate::inference::session::{load_session, HIDDEN_DIM, STREAM_HOLDBACK_FRAMES};
    use crate::ExecutionProvider;

    struct Collector(Vec<i16>);
    impl AudioSink for Collector {
        fn write(&mut self, _tag: u64, samples: &[i16]) -> Result<usize, SinkError> {
            self.0.extend_from_slice(samples);
            Ok(samples.len())
        }
        fn available(&self) -> usize {
            usize::MAX
        }
        fn on_drain_complete(&mut self) {}
        fn on_error(&mut self, _: u64, _: String) {}
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

    fn whole_decode(session: &mut Session, hidden: &[Vec<f32>]) -> Vec<i16> {
        run_decoder(session, hidden)
            .expect("run_decoder")
            .iter()
            .map(|&s| f32_to_i16(s))
            .collect()
    }

    fn assert_lossless(streamed: &[i16], whole: &[i16], label: &str) {
        assert_eq!(
            streamed.len(),
            whole.len(),
            "{label} length {} != whole-decode length {}",
            streamed.len(),
            whole.len()
        );
        let max_diff = streamed
            .iter()
            .zip(whole)
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 1,
            "{label} audio diverges from whole decode: max abs i16 diff = {max_diff}"
        );
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
        let whole = whole_decode(&mut session, &hidden);

        // No holdback: stream the whole pre-computed buffer through the decoder.
        let mut sink = Collector(Vec::new());
        {
            let mut stream = StreamingDecode::new(&mut session, 0, 0);
            for h in &hidden {
                assert!(stream.push(h, &mut sink, &|| false).expect("push"));
            }
            assert!(stream.finish(&mut sink, &|| false).expect("finish"));
        }
        assert_lossless(&sink.0, &whole, "streamed");
    }

    // Interleaved decoding (one hidden state pushed at a time, with a holdback
    // margin so hallucination runs never reach the sink) must, on a clean finish,
    // drain every buffered frame and reproduce the whole-decode audio exactly.
    #[test]
    fn interleaved_decode_with_holdback_matches_whole_decode() {
        let Some(path) = decoder_path() else {
            eprintln!("Skipping: decoder model not found");
            return;
        };
        let mut session = load_session(&path, &ExecutionProvider::Cpu).expect("load decoder");

        // Enough tokens to span the holdback margin plus several emitted windows.
        let hidden = synth(STREAM_HOLDBACK_FRAMES + 40);
        let whole = whole_decode(&mut session, &hidden);

        let mut sink = Collector(Vec::new());
        {
            let mut stream = StreamingDecode::new(&mut session, STREAM_HOLDBACK_FRAMES, 0);
            for h in &hidden {
                assert!(
                    stream.push(h, &mut sink, &|| false).expect("push"),
                    "push should not signal stop without cancellation"
                );
            }
            assert!(
                stream.finish(&mut sink, &|| false).expect("finish"),
                "finish should not signal stop without cancellation"
            );
        }
        assert_lossless(&sink.0, &whole, "interleaved");
    }
}
