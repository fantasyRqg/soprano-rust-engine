//! ONNX session management for Soprano backbone and decoder models.

use ort::session::Session;
use std::path::Path;

use crate::tts::ExecutionProvider;

/// Model configuration constants.
pub const NUM_LAYERS: usize = 17;
pub const NUM_KV_HEADS: usize = 1;
pub const HEAD_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 512;
pub const VOCAB_SIZE: usize = 8192;
pub const MAX_NEW_TOKENS: usize = 512;
pub const SAMPLE_RATE: u32 = 32000;
/// Audio samples generated per token by the decoder.
pub const SAMPLES_PER_TOKEN: usize = 2048;

/// Streaming decoder constants.
pub const RECEPTIVE_FIELD: usize = 4;
pub const CHUNK_SIZE: usize = 8;

/// Backbone hallucination detector: number of consecutive near-identical
/// hidden states that trips detection (see `backbone::HallucinationDetector`).
pub const HALLUCINATION_MAX_CONSECUTIVE: usize = 16;

/// Frames the interleaved decoder holds back before releasing them to the
/// sink. Hallucination is detected *after* the model has already emitted the
/// degenerate run, so we keep at least one full detection window un-emitted:
/// if generation trips the detector, these held-back frames are dropped and
/// never reach the sink (matching the non-streaming "discard on hallucination"
/// behavior), while everything ahead of the margin has already streamed out.
/// On a clean end-of-generation the margin is fully drained, so it adds buffer
/// depth, not lost audio.
pub const STREAM_HOLDBACK_FRAMES: usize = HALLUCINATION_MAX_CONSECUTIVE + 2;

/// Load an ONNX model session from file with the given execution provider.
pub fn load_session(path: impl AsRef<Path>, ep: &ExecutionProvider) -> Result<Session, String> {
    let builder = Session::builder()
        .map_err(|e| format!("failed to create session builder: {e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| format!("failed to set optimization level: {e}"))?;

    let mut builder = match ep {
        #[cfg(feature = "nnapi")]
        ExecutionProvider::Nnapi => builder
            .with_execution_providers([ort::ep::nnapi::NNAPI::default().build()])
            .map_err(|e| format!("failed to register NNAPI EP: {e}"))?,
        #[cfg(feature = "xnnpack")]
        ExecutionProvider::Xnnpack => builder
            .with_execution_providers([ort::ep::xnnpack::XNNPACK::default().build()])
            .map_err(|e| format!("failed to register XNNPACK EP: {e}"))?,
        #[cfg(feature = "coreml")]
        ExecutionProvider::CoreMl => builder
            .with_execution_providers([ort::ep::coreml::CoreML::default().build()])
            .map_err(|e| format!("failed to register CoreML EP: {e}"))?,
        _ => builder, // CPU fallback
    };

    builder
        .commit_from_file(path.as_ref())
        .map_err(|e| format!("failed to load ONNX model: {e}"))
}
