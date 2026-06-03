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

    // Greedy mode: just find argmax
    if params.temperature <= 0.0 {
        return greedy_sample(logits, seen_tokens, params);
    }

    let mut scored: Vec<(f32, u32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| {
            let is_seen = seen_tokens.get(i).copied().unwrap_or(false);
            (adjusted_score(logit, is_seen, params), i as u32)
        })
        .collect();

    let k = if params.top_k > 0 {
        params.top_k.min(vocab_size)
    } else {
        vocab_size
    };

    if k < vocab_size {
        let pivot = scored.len() - k;
        scored.select_nth_unstable_by(pivot, |a, b| a.0.total_cmp(&b.0));
        scored = scored.split_off(pivot);
    }

    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    sample_from_sorted_scores(&scored, params.top_p, rng)
}

fn adjusted_score(score: f32, is_seen: bool, params: &SamplingParams) -> f32 {
    let use_rep = params.repetition_penalty != 1.0;
    let score = if use_rep && is_seen {
        if score < 0.0 {
            score * params.repetition_penalty
        } else {
            score / params.repetition_penalty
        }
    } else {
        score
    };

    if params.temperature > 0.0 {
        score / params.temperature
    } else {
        score
    }
}

fn greedy_sample(logits: &[f32], seen_tokens: &[bool], params: &SamplingParams) -> u32 {
    let mut best_id = 0u32;
    let mut best_score = f32::NEG_INFINITY;
    for (i, &logit) in logits.iter().enumerate() {
        let is_seen = seen_tokens.get(i).copied().unwrap_or(false);
        let s = adjusted_score(logit, is_seen, params);
        if s > best_score {
            best_score = s;
            best_id = i as u32;
        }
    }
    best_id
}

fn sample_from_sorted_scores(
    scored: &[(f32, u32)],
    top_p: f32,
    rng: &mut impl Rng,
) -> u32 {
    if scored.is_empty() {
        return 0;
    }

    let max_score = scored[0].0;
    let exp_scores: Vec<f64> = scored
        .iter()
        .map(|(s, _)| ((*s - max_score) as f64).exp())
        .collect();
    let sum_exp: f64 = exp_scores.iter().sum();

    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return scored[0].1;
    }

    // Top-p filtering
    let keep = if top_p < 1.0 {
        let threshold = top_p.max(0.0) as f64 * sum_exp;
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
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return scored[0].1;
    }

    // Weighted random sample
    let mut r = rng.random::<f64>() * total_weight;
    for i in 0..keep {
        r -= exp_scores[i];
        if r <= 0.0 {
            return scored[i].1;
        }
    }
    scored[0].1
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

    #[test]
    fn test_top_p_applies_without_top_k() {
        let logits = vec![5.0, 4.9, 4.8, 1.0];
        let seen = vec![false; 4];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.01,
            repetition_penalty: 1.0,
        };
        let mut rng = rand::rng();

        assert_eq!(sample(&logits, &seen, &params, &mut rng), 0);
    }
}
