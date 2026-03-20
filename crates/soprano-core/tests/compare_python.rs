//! Compare Rust engine output against Python ONNX reference.
//! Run `cd soprano && uv run python -c "..."` first to generate /tmp/soprano_ref_*.npy

use soprano_core::inference::backbone;
use soprano_core::inference::decoder;
use soprano_core::inference::sampler::SamplingParams;
use soprano_core::inference::session::load_session;
use soprano_core::text::tokenizer::SopranoTokenizer;
use soprano_core::text::normalizer;

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

/// Load a 1D npy file as Vec<i64> (tokens are stored as int64).
fn load_npy_i64(path: &str) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap();
    // Simple npy parser for 1D int64 arrays
    // Skip header, find data
    let header_end = bytes.windows(1).position(|w| w[0] == b'\n' && bytes[..bytes.len()].contains(&0x0A)).unwrap_or(0);
    // Find the \n after the header dict
    let mut pos = 0;
    // npy format: magic(6) + version(2) + header_len(2/4) + header
    if &bytes[..6] == b"\x93NUMPY" {
        let major = bytes[6];
        let _minor = bytes[7];
        let header_len = if major == 1 {
            u16::from_le_bytes([bytes[8], bytes[9]]) as usize
        } else {
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
        };
        pos = if major == 1 { 10 } else { 12 };
        pos += header_len;
    }
    let data = &bytes[pos..];
    data.chunks(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn load_npy_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let mut pos = 0;
    if &bytes[..6] == b"\x93NUMPY" {
        let major = bytes[6];
        let header_len = if major == 1 {
            u16::from_le_bytes([bytes[8], bytes[9]]) as usize
        } else {
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
        };
        pos = if major == 1 { 10 } else { 12 };
        pos += header_len;
    }
    let data = &bytes[pos..];
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

    // Step 1: Normalize text the same way Python does
    // Python uses: prompt = f'[STOP][TEXT]{text}[START]'
    // Our normalizer adds extra processing. Let's match Python exactly.
    let text = "Hello world.";
    let normalized = normalizer::normalize(text);
    eprintln!("Rust normalized: {:?}", normalized);

    // Python prompt for "Hello world." is "[STOP][TEXT]Hello world.[START]"
    // Note: Python does NOT run text normalization for the e2e test!
    // It uses raw text directly: prompt = f'[STOP][TEXT]{text}[START]'
    let python_prompt = "[STOP][TEXT]Hello world.[START]";
    eprintln!("Python prompt:   {:?}", python_prompt);

    // Step 2: Tokenize both
    let tokenizer = SopranoTokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let rust_ids = tokenizer.encode(&normalized).unwrap();
    let python_ids = tokenizer.encode(python_prompt).unwrap();

    eprintln!("Rust token IDs:   {:?}", rust_ids);
    eprintln!("Python token IDs: {:?}", python_ids);

    // Check if normalization changes the tokens
    if rust_ids != python_ids {
        eprintln!("WARNING: Rust normalization produces different tokens than Python raw text!");
        eprintln!("  Rust normalized text: {:?}", normalized);
        eprintln!("  Python raw prompt:    {:?}", python_prompt);
    }

    // Step 3: Run backbone with the SAME input as Python (bypass normalization)
    let input_ids: Vec<i64> = python_ids.iter().map(|&id| id as i64).collect();
    eprintln!("Running backbone with Python-matching input_ids...");

    let mut backbone_session = load_session(
        if dir.join("soprano_backbone_kv_f16.onnx").exists() {
            dir.join("soprano_backbone_kv_f16.onnx")
        } else {
            dir.join("soprano_backbone_kv.onnx")
        }
    ).unwrap();

    let params = SamplingParams {
        temperature: 0.0, // greedy to match Python
        top_k: 0,
        top_p: 0.95,
        repetition_penalty: 1.2,
    };

    let output = backbone::generate(&mut backbone_session, &input_ids, &params).unwrap();

    eprintln!("Rust generated {} tokens: {:?}", output.generated_tokens.len(), output.generated_tokens);
    eprintln!("Rust hidden states: {} vectors", output.hidden_states.len());
    eprintln!("Hallucinated: {}", output.hallucinated);

    // Load Python reference tokens
    let ref_tokens = load_npy_i64("/tmp/soprano_ref_tokens.npy");
    let ref_tokens_u32: Vec<u32> = ref_tokens.iter().map(|&t| t as u32).collect();
    eprintln!("Python ref tokens ({}): {:?}", ref_tokens_u32.len(), ref_tokens_u32);

    // Compare tokens
    let match_len = output.generated_tokens.len().min(ref_tokens_u32.len());
    let mut first_mismatch = None;
    for i in 0..match_len {
        if output.generated_tokens[i] != ref_tokens_u32[i] {
            first_mismatch = Some(i);
            break;
        }
    }

    if let Some(pos) = first_mismatch {
        eprintln!("MISMATCH at token {}: rust={}, python={}",
            pos, output.generated_tokens[pos], ref_tokens_u32[pos]);
    } else if output.generated_tokens.len() != ref_tokens_u32.len() {
        eprintln!("Token count differs: rust={}, python={}",
            output.generated_tokens.len(), ref_tokens_u32.len());
    } else {
        eprintln!("ALL TOKENS MATCH!");
    }

    // Step 4: Run decoder and compare audio
    if !output.hidden_states.is_empty() {
        let mut decoder_session = load_session(
            if dir.join("soprano_decoder_f16.onnx").exists() {
                dir.join("soprano_decoder_f16.onnx")
            } else {
                dir.join("soprano_decoder.onnx")
            }
        ).unwrap();

        let audio = decoder::decode_all(&mut decoder_session, &output.hidden_states).unwrap();
        eprintln!("Rust audio: {} samples ({:.2}s)", audio.len(), audio.len() as f64 / 32000.0);

        // Load Python reference audio
        let ref_audio = load_npy_f32("/tmp/soprano_ref_audio.npy");
        eprintln!("Python ref audio: {} samples ({:.2}s)", ref_audio.len(), ref_audio.len() as f64 / 32000.0);

        // Compute SNR
        let min_len = audio.len().min(ref_audio.len());
        if min_len > 0 {
            let signal_power: f64 = ref_audio[..min_len].iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / min_len as f64;
            let noise_power: f64 = audio[..min_len].iter().zip(&ref_audio[..min_len])
                .map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>() / min_len as f64;

            if noise_power > 0.0 {
                let snr = 10.0 * (signal_power / noise_power).log10();
                eprintln!("SNR: {:.1} dB (>30 dB = PASS)", snr);
                assert!(snr > 30.0, "SNR too low: {:.1} dB", snr);
            } else {
                eprintln!("Outputs are identical (infinite SNR)");
            }

            let max_err: f32 = audio[..min_len].iter().zip(&ref_audio[..min_len])
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("Max error: {:.6}", max_err);
        }
    }

    // Check that the first N tokens match (exact match expected for greedy decoding).
    // Small divergence at the end is acceptable due to f16 floating point differences
    // in ONNX Runtime between Python and Rust bindings.
    let min_len = output.generated_tokens.len().min(ref_tokens_u32.len());
    let matching = output.generated_tokens[..min_len].iter()
        .zip(&ref_tokens_u32[..min_len])
        .take_while(|(a, b)| a == b)
        .count();
    let match_pct = matching as f64 / ref_tokens_u32.len() as f64 * 100.0;
    eprintln!("Token match: {}/{} ({:.0}%)", matching, ref_tokens_u32.len(), match_pct);
    assert!(match_pct >= 95.0, "too few matching tokens: {:.0}%", match_pct);
}
