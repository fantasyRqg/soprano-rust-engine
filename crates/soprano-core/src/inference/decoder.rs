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
fn run_decoder(session: &mut Session, hidden_states: &[Vec<f32>]) -> Result<Vec<f32>, String> {
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

/// Run the decoder on all hidden states at once (non-streaming).
pub fn decode_all(session: &mut Session, hidden_states: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    run_decoder(session, hidden_states)
}

/// Run the decoder in streaming mode with a sliding window.
/// Writes PCM i16 chunks to the provided AudioSink as they become available.
pub fn decode_streaming(
    session: &mut Session,
    hidden_states: &[Vec<f32>],
    sink: &mut dyn AudioSink,
) -> Result<usize, String> {
    Ok(
        decode_streaming_cancellable(session, hidden_states, sink, || false)?
            .expect("non-cancellable decoding cannot be cancelled"),
    )
}

/// Decode a complete hidden-state buffer in one streaming pass. Equivalent to
/// pushing every token through [`StreamingDecode`] with no holdback and then
/// finishing — kept as the reference entry point exercised by the lossless
/// test. Returns `None` if cancelled mid-stream.
pub(crate) fn decode_streaming_cancellable(
    session: &mut Session,
    hidden_states: &[Vec<f32>],
    sink: &mut dyn AudioSink,
    should_cancel: impl Fn() -> bool,
) -> Result<Option<usize>, String> {
    let mut stream = StreamingDecode::new(session, 0, 0);
    for hidden in hidden_states {
        if !stream.push(hidden, sink, &should_cancel)? {
            return Ok(None);
        }
    }
    if !stream.finish(sink, &should_cancel)? {
        return Ok(None);
    }
    Ok(Some(stream.total_samples_written()))
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
