//! Autoregressive backbone inference with KV cache management.
//! Port of the generation loop from soprano-web-onnx/onnx-streaming.js

use std::borrow::Cow;

use ndarray::Array4;
use ort::session::Session;
use ort::value::{DynValue, Tensor};

use super::sampler::{sample, SamplingParams};
use super::session::*;
use crate::text::tokenizer::TOKEN_STOP;

/// Result of a single backbone generation run.
pub struct BackboneOutput {
    /// Collected hidden states from generated tokens, shape (num_tokens, HIDDEN_DIM).
    pub hidden_states: Vec<Vec<f32>>,
    /// Generated token IDs (excluding prompt tokens).
    pub generated_tokens: Vec<u32>,
    /// Whether generation was stopped by hallucination detection.
    pub hallucinated: bool,
}

/// Hallucination detection: track consecutive tokens with similar hidden states.
struct HallucinationDetector {
    prev_hidden: Option<Vec<f32>>,
    consecutive_similar: usize,
    threshold: f32,
    max_consecutive: usize,
}

impl HallucinationDetector {
    fn new() -> Self {
        Self {
            prev_hidden: None,
            consecutive_similar: 0,
            threshold: 300.0,
            max_consecutive: 16,
        }
    }

    fn check(&mut self, hidden: &[f32]) -> bool {
        if let Some(ref prev) = self.prev_hidden {
            let diff: f32 = prev.iter().zip(hidden).map(|(a, b)| (a - b).abs()).sum();
            if diff < self.threshold {
                self.consecutive_similar += 1;
                if self.consecutive_similar > self.max_consecutive {
                    return true;
                }
            } else {
                self.consecutive_similar = 0;
            }
        }
        self.prev_hidden = Some(hidden.to_vec());
        false
    }
}

/// Run autoregressive generation on the backbone model.
pub fn generate(
    session: &mut Session,
    input_ids: &[i64],
    params: &SamplingParams,
) -> Result<BackboneOutput, String> {
    Ok(generate_cancellable(session, input_ids, params, || false)?
        .expect("non-cancellable generation cannot be cancelled"))
}

pub(crate) fn generate_cancellable(
    session: &mut Session,
    input_ids: &[i64],
    params: &SamplingParams,
    should_cancel: impl Fn() -> bool,
) -> Result<Option<BackboneOutput>, String> {
    let mut rng = rand::rng();
    let prompt_len = input_ids.len();

    // Initialize seen tokens mask for repetition penalty
    let mut seen_tokens = vec![false; VOCAB_SIZE];
    for &id in input_ids {
        let id = id as usize;
        if id < VOCAB_SIZE {
            seen_tokens[id] = true;
        }
    }

    // Initialize KV cache as empty tensors: (1, NUM_KV_HEADS, 0, HEAD_DIM)
    // Use ndarray to create 0-dim tensors (tuple form rejects dim=0)
    // Option slots so each step can move the value into the session inputs
    // and put the (owned) present.* output back without copying.
    let mut kv_cache: Vec<Option<DynValue>> = Vec::with_capacity(NUM_LAYERS * 2);
    for _ in 0..NUM_LAYERS * 2 {
        let empty = Array4::<f32>::zeros((1, NUM_KV_HEADS, 0, HEAD_DIM));
        let tensor =
            Tensor::from_array(empty).map_err(|e| format!("failed to create empty KV: {}", e))?;
        kv_cache.push(Some(tensor.into_dyn()));
    }

    // KV input/output names, resolved once outside the generation loop.
    let kv_names: Vec<[String; 4]> = (0..NUM_LAYERS)
        .map(|i| {
            [
                format!("past_key_values.{}.key", i),
                format!("past_key_values.{}.value", i),
                format!("present.{}.key", i),
                format!("present.{}.value", i),
            ]
        })
        .collect();

    let mut current_ids: Vec<i64> = input_ids.to_vec();
    let mut seq_len = prompt_len;
    let mut hidden_states_buffer: Vec<Vec<f32>> = Vec::new();
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut hallucination_detector = HallucinationDetector::new();

    // Find output indices by name
    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();
    let logits_idx = output_names
        .iter()
        .position(|n| n == "logits")
        .ok_or("logits output not found")?;
    let hidden_idx = output_names
        .iter()
        .position(|n| n == "last_hidden_state")
        .ok_or("last_hidden_state output not found")?;

    for _step in 0..MAX_NEW_TOKENS {
        if should_cancel() {
            return Ok(None);
        }

        let input_len = current_ids.len();

        // Build input tensors
        let ids_tensor = Tensor::from_array((
            [1i64, input_len as i64],
            current_ids.clone().into_boxed_slice(),
        ))
        .map_err(|e| format!("input_ids: {}", e))?;

        let mask_tensor = Tensor::from_array((
            [1i64, seq_len as i64],
            vec![1i64; seq_len].into_boxed_slice(),
        ))
        .map_err(|e| format!("attention_mask: {}", e))?;

        let pos_ids: Vec<i64> = if input_len == 1 {
            vec![(seq_len - 1) as i64]
        } else {
            (0..input_len as i64).collect()
        };
        let pos_tensor = Tensor::from_array(([1i64, input_len as i64], pos_ids.into_boxed_slice()))
            .map_err(|e| format!("position_ids: {}", e))?;

        // Build named inputs
        let mut inputs: Vec<(Cow<str>, DynValue)> = Vec::new();
        inputs.push(("input_ids".into(), ids_tensor.into_dyn()));
        inputs.push(("attention_mask".into(), mask_tensor.into_dyn()));
        inputs.push(("position_ids".into(), pos_tensor.into_dyn()));

        // Add KV cache inputs — move the values out of their slots
        for (i, names) in kv_names.iter().enumerate() {
            let k_val = kv_cache[i * 2].take().expect("KV key slot empty");
            let v_val = kv_cache[i * 2 + 1].take().expect("KV value slot empty");
            inputs.push((names[0].as_str().into(), k_val));
            inputs.push((names[1].as_str().into(), v_val));
        }

        // Run backbone inference
        let mut outputs = session
            .run(inputs)
            .map_err(|e| format!("backbone inference failed: {}", e))?;

        if should_cancel() {
            return Ok(None);
        }

        // Extract logits for the last token position and sample the next
        // token while the immutable borrow of `outputs` is live.
        let next_token = {
            let (logits_shape, logits_data) = outputs[logits_idx]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("failed to extract logits: {}", e))?;
            let last_pos = logits_shape[1] as usize - 1;
            let vocab = logits_shape[2] as usize;
            let logits_offset = last_pos * vocab;
            let logits_slice = &logits_data[logits_offset..logits_offset + vocab];
            sample(logits_slice, &seen_tokens, params, &mut rng)
        };

        // Extract hidden state for the last token position
        let (hidden_shape, hidden_data) = outputs[hidden_idx]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("failed to extract hidden states: {}", e))?;
        let hidden_last_pos = hidden_shape[1] as usize - 1;
        let hidden_dim = hidden_shape[2] as usize;
        let hidden_offset = hidden_last_pos * hidden_dim;
        let hidden_vec: Vec<f32> = hidden_data[hidden_offset..hidden_offset + hidden_dim].to_vec();

        // Update KV cache by moving the owned present.{i}.key/value outputs
        // out of `outputs` — no copy of the (growing) cache per step.
        for (i, names) in kv_names.iter().enumerate() {
            kv_cache[i * 2] = Some(
                outputs
                    .remove(&names[2])
                    .ok_or_else(|| format!("{} not found in outputs", names[2]))?,
            );
            kv_cache[i * 2 + 1] = Some(
                outputs
                    .remove(&names[3])
                    .ok_or_else(|| format!("{} not found in outputs", names[3]))?,
            );
        }
        let finished = next_token == TOKEN_STOP;

        // Match Python logic exactly:
        // Python checks EOS first, then collects hidden state if not EOS
        if finished {
            break;
        }

        // Collect hidden state from this model call's output
        // (Python: hidden_states.append(last_hidden[0, -1, :]))
        hidden_states_buffer.push(hidden_vec.clone());
        generated_tokens.push(next_token);

        if (next_token as usize) < VOCAB_SIZE {
            seen_tokens[next_token as usize] = true;
        }

        // Hallucination detection
        if hallucination_detector.check(&hidden_vec) {
            return Ok(Some(BackboneOutput {
                hidden_states: hidden_states_buffer,
                generated_tokens,
                hallucinated: true,
            }));
        }

        // Next step: single token input
        current_ids = vec![next_token as i64];
        seq_len += 1;
    }

    Ok(Some(BackboneOutput {
        hidden_states: hidden_states_buffer,
        generated_tokens,
        hallucinated: false,
    }))
}
