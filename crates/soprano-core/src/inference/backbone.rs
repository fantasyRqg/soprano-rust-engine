//! Autoregressive backbone inference with KV cache management.
//! Port of the generation loop from soprano-web-onnx/onnx-streaming.js

use std::borrow::Cow;
use std::time::{Duration, Instant};

use ndarray::Array4;
use ort::session::Session;
use ort::value::{DynValue, Tensor};

use super::sampler::{sample, SamplingParams};
use super::session::*;
use crate::profile::profile_log;
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
            max_consecutive: HALLUCINATION_MAX_CONSECUTIVE,
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

/// How a generation run ended.
pub(crate) enum GenEnd {
    /// Reached EOS or the token limit with no problem.
    Completed,
    /// Hallucination detector tripped; the run was cut short.
    Hallucinated,
    /// `should_cancel` fired or the per-token callback asked to stop.
    Aborted,
}

/// Outcome of a streaming generation run, for the coordinator in `tts`.
pub(crate) enum StreamOutcome {
    Completed { generated: usize },
    Hallucinated { generated: usize },
    Aborted,
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

/// Generate and collect all hidden states/tokens into a `BackboneOutput`.
/// Returns `None` if cancelled. On hallucination the returned buffer includes
/// the degenerate token (the caller discards it).
pub(crate) fn generate_cancellable(
    session: &mut Session,
    input_ids: &[i64],
    params: &SamplingParams,
    should_cancel: impl Fn() -> bool,
) -> Result<Option<BackboneOutput>, String> {
    let mut hidden_states = Vec::new();
    let mut generated_tokens = Vec::new();
    let end = generate_core(
        session,
        input_ids,
        params,
        should_cancel,
        |token, hidden| {
            generated_tokens.push(token);
            hidden_states.push(hidden.to_vec());
            Ok(true)
        },
    )?;
    Ok(match end {
        GenEnd::Aborted => None,
        GenEnd::Hallucinated => Some(BackboneOutput {
            hidden_states,
            generated_tokens,
            hallucinated: true,
        }),
        GenEnd::Completed => Some(BackboneOutput {
            hidden_states,
            generated_tokens,
            hallucinated: false,
        }),
    })
}

/// Generate and hand each token's hidden state to `on_hidden` as soon as it is
/// produced, so a decoder can stream audio while generation continues.
/// `on_hidden` returns `Ok(false)` to stop early (e.g. the sink closed).
pub(crate) fn generate_streaming_cancellable(
    session: &mut Session,
    input_ids: &[i64],
    params: &SamplingParams,
    should_cancel: impl Fn() -> bool,
    mut on_hidden: impl FnMut(&[f32]) -> Result<bool, String>,
) -> Result<StreamOutcome, String> {
    let mut generated = 0usize;
    let end = generate_core(
        session,
        input_ids,
        params,
        should_cancel,
        |_token, hidden| {
            generated += 1;
            on_hidden(hidden)
        },
    )?;
    Ok(match end {
        GenEnd::Completed => StreamOutcome::Completed { generated },
        GenEnd::Hallucinated => StreamOutcome::Hallucinated { generated },
        GenEnd::Aborted => StreamOutcome::Aborted,
    })
}

/// Core autoregressive loop. `on_step` is invoked with each generated token and
/// its hidden state (after the EOS check, before hallucination detection — so a
/// hallucination run's tokens are delivered but the EOS token is not). It
/// returns `Ok(false)` to abort generation.
fn generate_core(
    session: &mut Session,
    input_ids: &[i64],
    params: &SamplingParams,
    should_cancel: impl Fn() -> bool,
    mut on_step: impl FnMut(u32, &[f32]) -> Result<bool, String>,
) -> Result<GenEnd, String> {
    let mut rng = rand::rng();
    let prompt_len = input_ids.len();
    let t_total = Instant::now();
    let mut prefill_ms: u128 = 0;
    let mut step_time = Duration::ZERO;
    let mut step_count: usize = 0;

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
            return Ok(GenEnd::Aborted);
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
        let t_run = Instant::now();
        let mut outputs = session
            .run(inputs)
            .map_err(|e| format!("backbone inference failed: {}", e))?;
        if _step == 0 {
            prefill_ms = t_run.elapsed().as_millis();
            profile_log!(
                "backbone prefill: {}ms ({} prompt tokens)",
                prefill_ms,
                prompt_len
            );
        } else {
            step_time += t_run.elapsed();
            step_count += 1;
        }

        if should_cancel() {
            return Ok(GenEnd::Aborted);
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

        if (next_token as usize) < VOCAB_SIZE {
            seen_tokens[next_token as usize] = true;
        }

        // Hand the hidden state to the callback (collect or stream). Done
        // before hallucination detection so the run's tokens are delivered;
        // the streaming decoder holds them back behind its margin and drops
        // them if the detector then trips.
        if !on_step(next_token, &hidden_vec)? {
            return Ok(GenEnd::Aborted);
        }

        // Hallucination detection
        if hallucination_detector.check(&hidden_vec) {
            return Ok(GenEnd::Hallucinated);
        }

        // Next step: single token input
        current_ids = vec![next_token as i64];
        seq_len += 1;
    }

    profile_log!(
        "backbone total: {}ms (prefill={}ms, {} decode steps, avg {:.1}ms/step)",
        t_total.elapsed().as_millis(),
        prefill_ms,
        step_count,
        if step_count > 0 {
            step_time.as_secs_f64() * 1000.0 / step_count as f64
        } else {
            0.0
        }
    );

    Ok(GenEnd::Completed)
}
