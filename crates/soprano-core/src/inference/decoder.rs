//! Vocos decoder inference with sliding window for streaming.
//! Port of the decoder logic from soprano-web-onnx/onnx-streaming.js

use ort::session::Session;
use ort::value::Tensor;

use super::session::*;
use crate::audio::convert::f32_to_i16;
use crate::audio::sink::{AudioSink, SinkError};

/// Run the decoder on hidden states and return PCM f32 audio.
fn run_decoder(
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

    let input_tensor = Tensor::from_array(
        ([1usize, HIDDEN_DIM, num_tokens], decoder_input.into_boxed_slice())
    ).map_err(|e| format!("failed to create decoder input: {}", e))?;

    let input_name = session.inputs().first()
        .map(|i| i.name())
        .unwrap_or("hidden_states");

    let outputs = session.run(
        vec![(std::borrow::Cow::from(input_name.to_string()), input_tensor.into_dyn())]
    ).map_err(|e| format!("decoder inference failed: {}", e))?;

    let (_shape, audio_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("failed to extract audio: {}", e))?;

    Ok(audio_data.to_vec())
}

/// Run the decoder on all hidden states at once (non-streaming).
pub fn decode_all(
    session: &mut Session,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<f32>, String> {
    run_decoder(session, hidden_states)
}

/// Run the decoder in streaming mode with a sliding window.
/// Writes PCM i16 chunks to the provided AudioSink as they become available.
pub fn decode_streaming(
    session: &mut Session,
    hidden_states: &[Vec<f32>],
    sink: &mut dyn AudioSink,
) -> Result<usize, String> {
    if hidden_states.is_empty() {
        return Ok(0);
    }

    let total_tokens = hidden_states.len();
    let mut total_samples_written = 0;
    let mut offset = 0;

    while offset < total_tokens {
        // Determine window: receptive field + chunk
        let rf_start = if offset >= RECEPTIVE_FIELD {
            offset - RECEPTIVE_FIELD
        } else {
            0
        };
        let chunk_end = (offset + CHUNK_SIZE).min(total_tokens);
        let window = &hidden_states[rf_start..chunk_end];

        // Run decoder on this window
        let audio = run_decoder(session, window)?;

        // Extract only the chunk portion (skip receptive field audio)
        let rf_tokens_in_window = offset - rf_start;
        let chunk_tokens = chunk_end - offset;

        let audio_start = rf_tokens_in_window * SAMPLES_PER_TOKEN;
        let audio_end = (rf_tokens_in_window + chunk_tokens) * SAMPLES_PER_TOKEN;
        let audio_end = audio_end.min(audio.len());
        let audio_start = audio_start.min(audio_end);

        if audio_start < audio_end {
            let chunk_audio = &audio[audio_start..audio_end];

            // Convert f32 → i16 and write to sink (blocks if full = backpressure)
            let i16_samples: Vec<i16> = chunk_audio.iter().map(|&s| f32_to_i16(s)).collect();
            let mut written = 0;
            while written < i16_samples.len() {
                match sink.write(&i16_samples[written..]) {
                    Ok(n) => written += n,
                    Err(SinkError::Closed) => return Ok(total_samples_written),
                    Err(e) => return Err(format!("sink write error: {}", e)),
                }
            }
            total_samples_written += i16_samples.len();
        }

        offset = chunk_end;
    }

    Ok(total_samples_written)
}
