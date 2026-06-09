//! Vocos decoder inference with sliding window for streaming.
//! Port of the decoder logic from soprano-web-onnx/onnx-streaming.js

use ort::session::Session;
use ort::value::Tensor;

use super::session::*;
use crate::audio::convert::f32_to_i16;
use crate::audio::sink::{AudioSink, SinkError};

fn write_f32_pcm(
    sink: &mut dyn AudioSink,
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

        match sink.write(&i16_samples[written..]) {
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

pub(crate) fn decode_streaming_cancellable(
    session: &mut Session,
    hidden_states: &[Vec<f32>],
    sink: &mut dyn AudioSink,
    should_cancel: impl Fn() -> bool,
) -> Result<Option<usize>, String> {
    if hidden_states.is_empty() {
        return Ok(Some(0));
    }

    let total_tokens = hidden_states.len();
    // The decoder emits (W - 1) audio frames of SAMPLES_PER_TOKEN each for a
    // window of W tokens, so a single token yields no audio.
    if total_tokens < 2 {
        return Ok(Some(0));
    }
    let total_frames = total_tokens - 1;
    let mut total_samples_written = 0;
    let mut offset = 0;

    while offset < total_frames {
        if should_cancel() {
            return Ok(None);
        }

        let chunk_end = (offset + CHUNK_SIZE).min(total_frames);

        // Decode a window with RECEPTIVE_FIELD tokens of context on BOTH sides.
        // With context on each side, the frames we emit are bit-for-bit the same
        // as decoding the whole sequence at once — lossless streaming, no need to
        // crossfade seams (the previous left-only window decoded each chunk's
        // trailing frames without right context, producing audible seams).
        let w0 = offset.saturating_sub(RECEPTIVE_FIELD);
        let w1 = (chunk_end + RECEPTIVE_FIELD).min(total_tokens);
        let audio = run_decoder(session, &hidden_states[w0..w1])?;

        // Emit frames [offset, chunk_end); local frame index = global index - w0.
        let start = (offset - w0) * SAMPLES_PER_TOKEN;
        let end = ((chunk_end - w0) * SAMPLES_PER_TOKEN).min(audio.len());
        if start < end
            && !write_f32_pcm(
                sink,
                &audio[start..end],
                &mut total_samples_written,
                &should_cancel,
            )?
        {
            return Ok(None);
        }

        offset = chunk_end;
    }

    Ok(Some(total_samples_written))
}
