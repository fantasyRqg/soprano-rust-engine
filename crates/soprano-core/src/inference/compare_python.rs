//! Compare Rust engine output against the Python ONNX reference.
//! Run `cd soprano && uv run python -c "..."` first to generate
//! /tmp/soprano_ref_*.npy, then `cargo test -p soprano-core compare`.
//!
//! Drives the internal `generate_core` (collecting tokens + hidden states) and
//! `run_decoder` (whole-buffer decode) primitives — the same primitives the
//! streaming path is built on — so the comparison needs no public one-shot API.

use crate::inference::backbone::{self, GenEnd};
use crate::inference::decoder;
use crate::inference::sampler::SamplingParams;
use crate::inference::session::load_session;
use crate::text::normalizer;
use crate::text::tokenizer::SopranoTokenizer;
use crate::ExecutionProvider;

fn models_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn models_available() -> bool {
    let dir = models_dir();
    dir.join("tokenizer.json").exists()
        && (dir.join("soprano_backbone_kv_f16.onnx").exists()
            || dir.join("soprano_backbone_kv.onnx").exists())
}

fn ref_data_available() -> bool {
    std::path::Path::new("/tmp/soprano_ref_tokens.npy").exists()
}

/// Skip the npy header and return the data offset.
fn npy_data_offset(bytes: &[u8]) -> usize {
    if &bytes[..6] == b"\x93NUMPY" {
        let major = bytes[6];
        let header_len = if major == 1 {
            u16::from_le_bytes([bytes[8], bytes[9]]) as usize
        } else {
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
        };
        (if major == 1 { 10 } else { 12 }) + header_len
    } else {
        0
    }
}

/// Load a 1D npy file as Vec<i64> (tokens are stored as int64).
fn load_npy_i64(path: &str) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap();
    let data = &bytes[npy_data_offset(&bytes)..];
    data.chunks(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn load_npy_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let data = &bytes[npy_data_offset(&bytes)..];
    data.chunks(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn test_compare_tokens_with_python() {
    if !models_available() || !ref_data_available() {
        eprintln!("Skipping: models or reference data not found");
        eprintln!("Run Python reference first to generate /tmp/soprano_ref_tokens.npy");
        return;
    }

    let dir = models_dir();

    // Python does NOT run text normalization for the e2e test; it uses the raw
    // prompt `[STOP][TEXT]{text}[START]`. Match that exactly.
    let text = "Hello world.";
    let normalized = normalizer::normalize(text);
    eprintln!("Rust normalized: {:?}", normalized);
    let python_prompt = "[STOP][TEXT]Hello world.[START]";
    eprintln!("Python prompt:   {:?}", python_prompt);

    let tokenizer = SopranoTokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let rust_ids = tokenizer.encode(&normalized).unwrap();
    let python_ids = tokenizer.encode(python_prompt).unwrap();
    eprintln!("Rust token IDs:   {:?}", rust_ids);
    eprintln!("Python token IDs: {:?}", python_ids);
    if rust_ids != python_ids {
        eprintln!("WARNING: Rust normalization produces different tokens than Python raw text!");
    }

    // Run the backbone with the SAME input as Python (bypassing normalization),
    // collecting tokens and hidden states straight off the core loop.
    let input_ids: Vec<i64> = python_ids.iter().map(|&id| id as i64).collect();
    eprintln!("Running backbone with Python-matching input_ids...");

    let mut backbone_session = load_session(
        if dir.join("soprano_backbone_kv_f16.onnx").exists() {
            dir.join("soprano_backbone_kv_f16.onnx")
        } else {
            dir.join("soprano_backbone_kv.onnx")
        },
        &ExecutionProvider::Cpu,
    )
    .unwrap();

    let params = SamplingParams {
        temperature: 0.0, // greedy to match Python
        top_k: 0,
        top_p: 0.95,
        repetition_penalty: 1.2,
    };

    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut hidden_states: Vec<Vec<f32>> = Vec::new();
    let end = backbone::generate_core(
        &mut backbone_session,
        &input_ids,
        &params,
        || false,
        |token, hidden| {
            generated_tokens.push(token);
            hidden_states.push(hidden.to_vec());
            Ok(true)
        },
    )
    .unwrap();
    let hallucinated = matches!(end, GenEnd::Hallucinated);

    eprintln!(
        "Rust generated {} tokens: {:?}",
        generated_tokens.len(),
        generated_tokens
    );
    eprintln!("Rust hidden states: {} vectors", hidden_states.len());
    eprintln!("Hallucinated: {}", hallucinated);

    let ref_tokens = load_npy_i64("/tmp/soprano_ref_tokens.npy");
    let ref_tokens_u32: Vec<u32> = ref_tokens.iter().map(|&t| t as u32).collect();
    eprintln!(
        "Python ref tokens ({}): {:?}",
        ref_tokens_u32.len(),
        ref_tokens_u32
    );

    let match_len = generated_tokens.len().min(ref_tokens_u32.len());
    let first_mismatch = generated_tokens
        .iter()
        .zip(&ref_tokens_u32)
        .take(match_len)
        .position(|(r, p)| r != p);
    if let Some(pos) = first_mismatch {
        eprintln!(
            "MISMATCH at token {}: rust={}, python={}",
            pos, generated_tokens[pos], ref_tokens_u32[pos]
        );
    } else if generated_tokens.len() != ref_tokens_u32.len() {
        eprintln!(
            "Token count differs: rust={}, python={}",
            generated_tokens.len(),
            ref_tokens_u32.len()
        );
    } else {
        eprintln!("ALL TOKENS MATCH!");
    }

    // Run the decoder over the whole hidden buffer and compare audio (SNR).
    if !hidden_states.is_empty() {
        let mut decoder_session = load_session(
            if dir.join("soprano_decoder_f16.onnx").exists() {
                dir.join("soprano_decoder_f16.onnx")
            } else {
                dir.join("soprano_decoder.onnx")
            },
            &ExecutionProvider::Cpu,
        )
        .unwrap();

        let audio = decoder::run_decoder(&mut decoder_session, &hidden_states).unwrap();
        eprintln!(
            "Rust audio: {} samples ({:.2}s)",
            audio.len(),
            audio.len() as f64 / 32000.0
        );

        let ref_audio = load_npy_f32("/tmp/soprano_ref_audio.npy");
        eprintln!(
            "Python ref audio: {} samples ({:.2}s)",
            ref_audio.len(),
            ref_audio.len() as f64 / 32000.0
        );

        let min_len = audio.len().min(ref_audio.len());
        if min_len > 0 {
            let signal_power: f64 = ref_audio[..min_len]
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                / min_len as f64;
            let noise_power: f64 = audio[..min_len]
                .iter()
                .zip(&ref_audio[..min_len])
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / min_len as f64;

            if noise_power > 0.0 {
                let snr = 10.0 * (signal_power / noise_power).log10();
                eprintln!("SNR: {:.1} dB (>30 dB = PASS)", snr);
                assert!(snr > 30.0, "SNR too low: {:.1} dB", snr);
            } else {
                eprintln!("Outputs are identical (infinite SNR)");
            }

            let max_err: f32 = audio[..min_len]
                .iter()
                .zip(&ref_audio[..min_len])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("Max error: {:.6}", max_err);
        }
    }

    // First N tokens must match (exact for greedy decoding). Small end-divergence
    // is acceptable due to f16 float differences between Python and Rust ORT.
    let min_len = generated_tokens.len().min(ref_tokens_u32.len());
    let matching = generated_tokens[..min_len]
        .iter()
        .zip(&ref_tokens_u32[..min_len])
        .take_while(|(a, b)| a == b)
        .count();
    let match_pct = matching as f64 / ref_tokens_u32.len() as f64 * 100.0;
    eprintln!(
        "Token match: {}/{} ({:.0}%)",
        matching,
        ref_tokens_u32.len(),
        match_pct
    );
    assert!(
        match_pct >= 95.0,
        "too few matching tokens: {:.0}%",
        match_pct
    );
}
