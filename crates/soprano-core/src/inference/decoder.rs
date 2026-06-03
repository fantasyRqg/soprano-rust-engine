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
) -> Result<(), String> {
    if samples.is_empty() {
        return Ok(());
    }

    let i16_samples: Vec<i16> = samples.iter().map(|&s| f32_to_i16(s)).collect();
    let mut written = 0;
    while written < i16_samples.len() {
        match sink.write(&i16_samples[written..]) {
            Ok(n) => written += n,
            Err(SinkError::Closed) => {
                *total_samples_written += written;
                return Ok(());
            }
            Err(e) => return Err(format!("sink write error: {}", e)),
        }
    }
    *total_samples_written += i16_samples.len();
    Ok(())
}

fn crossfade(prev_tail: &[f32], next_head: &[f32]) -> Vec<f32> {
    let overlap = prev_tail.len().min(next_head.len());
    let mut blended = Vec::with_capacity(overlap);

    if overlap == 0 {
        return blended;
    }

    for i in 0..overlap {
        let t = i as f32 / overlap as f32;
        blended.push(prev_tail[i] * (1.0 - t) + next_head[i] * t);
    }

    blended
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
    if hidden_states.is_empty() {
        return Ok(0);
    }

    let total_tokens = hidden_states.len();
    let mut total_samples_written = 0;
    let mut offset = 0;
    let mut pending_tail: Vec<f32> = Vec::new();

    while offset < total_tokens {
        // Determine window: receptive field + chunk
        let rf_start = offset.saturating_sub(RECEPTIVE_FIELD);
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
            let is_last_chunk = chunk_end == total_tokens;
            let crossfade_len = STREAM_CROSSFADE_SAMPLES.min(chunk_audio.len());

            if pending_tail.is_empty() {
                if is_last_chunk || chunk_audio.len() <= crossfade_len {
                    write_f32_pcm(sink, chunk_audio, &mut total_samples_written)?;
                } else {
                    let stable_len = chunk_audio.len() - crossfade_len;
                    write_f32_pcm(sink, &chunk_audio[..stable_len], &mut total_samples_written)?;
                    pending_tail = chunk_audio[stable_len..].to_vec();
                }
            } else {
                let overlap = pending_tail.len().min(crossfade_len);
                let blended = crossfade(&pending_tail[..overlap], &chunk_audio[..overlap]);
                write_f32_pcm(sink, &blended, &mut total_samples_written)?;

                if is_last_chunk {
                    write_f32_pcm(sink, &chunk_audio[overlap..], &mut total_samples_written)?;
                    pending_tail.clear();
                } else {
                    let tail_len =
                        STREAM_CROSSFADE_SAMPLES.min(chunk_audio.len().saturating_sub(overlap));
                    let body_end = chunk_audio.len() - tail_len;
                    if overlap < body_end {
                        write_f32_pcm(
                            sink,
                            &chunk_audio[overlap..body_end],
                            &mut total_samples_written,
                        )?;
                    }
                    pending_tail = chunk_audio[body_end..].to_vec();
                }
            }
        }

        offset = chunk_end;
    }

    if !pending_tail.is_empty() {
        write_f32_pcm(sink, &pending_tail, &mut total_samples_written)?;
    }

    Ok(total_samples_written)
}
