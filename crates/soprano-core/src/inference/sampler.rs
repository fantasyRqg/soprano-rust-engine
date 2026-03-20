//! Token sampling: greedy, top-k, top-p, temperature, repetition penalty.
//! Ported from soprano/tts.py and soprano-web-onnx/onnx-streaming.js

use rand::Rng;

/// Sampling parameters.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Temperature for sampling. 0.0 = greedy.
    pub temperature: f32,
    /// Top-k sampling. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Repetition penalty applied to already-seen tokens.
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 0.95,
            repetition_penalty: 1.2,
        }
    }
}

/// Sample the next token from logits.
///
/// `logits` is the raw logits for the last position, shape `[vocab_size]`.
/// `seen_tokens` is a set of token IDs that have been generated so far
/// (used for repetition penalty).
pub fn sample(
    logits: &[f32],
    seen_tokens: &[bool],
    params: &SamplingParams,
    rng: &mut impl Rng,
) -> u32 {
    let vocab_size = logits.len();

    // Apply repetition penalty and temperature
    let use_rep_penalty = params.repetition_penalty != 1.0 && !seen_tokens.is_empty();
    let inv_temp = if params.temperature > 0.0 {
        1.0 / params.temperature
    } else {
        1.0 // greedy mode, temperature doesn't matter
    };
    let inv_rep_penalty = if use_rep_penalty {
        1.0 / params.repetition_penalty
    } else {
        1.0
    };

    // Greedy mode: just find argmax
    if params.temperature <= 0.0 {
        return greedy_sample(logits, seen_tokens, use_rep_penalty, params.repetition_penalty, inv_rep_penalty);
    }

    // Sampling mode
    let k = if params.top_k > 0 {
        params.top_k.min(vocab_size)
    } else {
        vocab_size
    };

    if k < vocab_size {
        // Top-k path: use partial sort to find top-k tokens
        top_k_sample(
            logits, seen_tokens, params, use_rep_penalty,
            inv_temp, inv_rep_penalty, k, rng,
        )
    } else {
        // Full softmax path
        full_softmax_sample(
            logits, seen_tokens, params, use_rep_penalty,
            inv_temp, inv_rep_penalty, rng,
        )
    }
}

fn apply_penalty(score: f32, is_seen: bool, use_rep: bool, rep_penalty: f32, inv_rep: f32) -> f32 {
    if use_rep && is_seen {
        if score < 0.0 {
            score * rep_penalty
        } else {
            score * inv_rep
        }
    } else {
        score
    }
}

fn greedy_sample(
    logits: &[f32],
    seen_tokens: &[bool],
    use_rep: bool,
    rep_penalty: f32,
    inv_rep: f32,
) -> u32 {
    let mut best_id = 0u32;
    let mut best_score = f32::NEG_INFINITY;
    for (i, &logit) in logits.iter().enumerate() {
        let is_seen = seen_tokens.get(i).copied().unwrap_or(false);
        let s = apply_penalty(logit, is_seen, use_rep, rep_penalty, inv_rep);
        if s > best_score {
            best_score = s;
            best_id = i as u32;
        }
    }
    best_id
}

fn top_k_sample(
    logits: &[f32],
    seen_tokens: &[bool],
    params: &SamplingParams,
    use_rep: bool,
    inv_temp: f32,
    inv_rep: f32,
    k: usize,
    rng: &mut impl Rng,
) -> u32 {
    // Collect (score, token_id) for all tokens with penalties applied
    let mut scored: Vec<(f32, u32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| {
            let is_seen = seen_tokens.get(i).copied().unwrap_or(false);
            let s = apply_penalty(logit, is_seen, use_rep, params.repetition_penalty, inv_rep) * inv_temp;
            (s, i as u32)
        })
        .collect();

    // Partial sort to get top-k (using select_nth_unstable for O(n))
    let pivot = scored.len() - k;
    scored.select_nth_unstable_by(pivot, |a, b| a.0.partial_cmp(&b.0).unwrap());
    let top_k = &mut scored[pivot..];

    // Sort top-k by score descending
    top_k.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Softmax over top-k
    let max_score = top_k[0].0;
    let exp_scores: Vec<f64> = top_k
        .iter()
        .map(|(s, _)| ((*s - max_score) as f64).exp())
        .collect();
    let sum_exp: f64 = exp_scores.iter().sum();

    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return top_k[0].1;
    }

    // Top-p filtering
    let keep = if params.top_p < 1.0 {
        let threshold = params.top_p as f64 * sum_exp;
        let mut cumulative = 0.0;
        let mut kept = 0;
        for &w in &exp_scores {
            cumulative += w;
            kept += 1;
            if kept >= 1 && cumulative >= threshold {
                break;
            }
        }
        kept
    } else {
        exp_scores.len()
    };

    let total_weight: f64 = exp_scores[..keep].iter().sum();

    // Weighted random sample
    let mut r = rng.random::<f64>() * total_weight;
    for i in 0..keep {
        r -= exp_scores[i];
        if r <= 0.0 {
            return top_k[i].1;
        }
    }
    top_k[0].1
}

fn full_softmax_sample(
    logits: &[f32],
    seen_tokens: &[bool],
    params: &SamplingParams,
    use_rep: bool,
    inv_temp: f32,
    inv_rep: f32,
    rng: &mut impl Rng,
) -> u32 {
    // Find max score for numerical stability
    let mut max_score = f32::NEG_INFINITY;
    let mut best_id = 0u32;
    let scores: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| {
            let is_seen = seen_tokens.get(i).copied().unwrap_or(false);
            let s = apply_penalty(logit, is_seen, use_rep, params.repetition_penalty, inv_rep) * inv_temp;
            if s > max_score {
                max_score = s;
                best_id = i as u32;
            }
            s
        })
        .collect();

    let sum_exp: f64 = scores.iter().map(|&s| ((s - max_score) as f64).exp()).sum();

    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return best_id;
    }

    let mut r = rng.random::<f64>() * sum_exp;
    for (i, &s) in scores.iter().enumerate() {
        r -= ((s - max_score) as f64).exp();
        if r <= 0.0 {
            return i as u32;
        }
    }
    best_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_selects_max() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let seen = vec![false; 5];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = rand::rng();
        assert_eq!(sample(&logits, &seen, &params, &mut rng), 3);
    }

    #[test]
    fn test_repetition_penalty_suppresses() {
        let logits = vec![0.9, 0.8, 0.1];
        let mut seen = vec![false; 3];
        seen[0] = true; // token 0 has been seen
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 5.0, // strong penalty
            ..Default::default()
        };
        let mut rng = rand::rng();
        // Token 0 had highest logit but is penalized, so token 1 should win
        assert_eq!(sample(&logits, &seen, &params, &mut rng), 1);
    }

    #[test]
    fn test_sampling_with_temperature() {
        let mut logits = vec![0.0f32; 100];
        logits[42] = 100.0; // Make token 42 overwhelmingly likely
        let seen = vec![false; 100];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let mut rng = rand::rng();
        // With such a dominant logit, sampling should almost always pick 42
        let result = sample(&logits, &seen, &params, &mut rng);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_top_k_limits_candidates() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let seen = vec![false; 5];
        let params = SamplingParams {
            temperature: 0.01, // Very low temp = nearly greedy
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let mut rng = rand::rng();
        let result = sample(&logits, &seen, &params, &mut rng);
        // Should be one of the top 2: token 3 or 4
        assert!(result == 3 || result == 4);
    }
}
