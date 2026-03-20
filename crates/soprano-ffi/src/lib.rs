//! UniFFI bindings for the Soprano TTS engine.
//!
//! Wraps soprano-core types into UniFFI-compatible interfaces.
//! The AudioSink trait is exposed as a callback interface that receives
//! PCM data as raw bytes (little-endian i16).

use std::sync::{Arc, Mutex};
use soprano_core::{AudioSink, SinkError};

uniffi::setup_scaffolding!();

// ─── Error types ───────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    #[error("model loading failed: {msg}")]
    ModelLoadError { msg: String },
    #[error("input too long: {token_count} tokens exceeds max {max_tokens}")]
    InputTooLong { token_count: u32, max_tokens: u32 },
    #[error("inference error: {msg}")]
    InferenceError { msg: String },
    #[error("hallucination detected")]
    Hallucination,
    #[error("tokenization error: {msg}")]
    TokenizationError { msg: String },
}

impl From<soprano_core::SopranoError> for FfiError {
    fn from(e: soprano_core::SopranoError) -> Self {
        match e {
            soprano_core::SopranoError::ModelLoadError(msg) => FfiError::ModelLoadError { msg },
            soprano_core::SopranoError::InputTooLong { token_count, max_tokens } => {
                FfiError::InputTooLong {
                    token_count: token_count as u32,
                    max_tokens: max_tokens as u32,
                }
            }
            soprano_core::SopranoError::NoSink => {
                FfiError::InferenceError { msg: "no sink attached".to_string() }
            }
            soprano_core::SopranoError::InferenceError(msg) => FfiError::InferenceError { msg },
            soprano_core::SopranoError::Hallucination => FfiError::Hallucination,
            soprano_core::SopranoError::TokenizationError(msg) => {
                FfiError::TokenizationError { msg }
            }
        }
    }
}

// ─── Config & result types ─────────────────────────────────────────────────

/// Execution provider for ONNX Runtime inference.
#[derive(uniffi::Enum)]
pub enum ExecutionProvider {
    /// Default CPU execution.
    Cpu,
    /// Android NNAPI (delegates to NPU/GPU).
    Nnapi,
    /// XNNPACK optimized CPU kernels for ARM.
    Xnnpack,
}

impl From<&ExecutionProvider> for soprano_core::ExecutionProvider {
    fn from(ep: &ExecutionProvider) -> Self {
        match ep {
            ExecutionProvider::Cpu => soprano_core::ExecutionProvider::Cpu,
            ExecutionProvider::Nnapi => soprano_core::ExecutionProvider::Nnapi,
            ExecutionProvider::Xnnpack => soprano_core::ExecutionProvider::Xnnpack,
        }
    }
}

#[derive(uniffi::Record)]
pub struct SopranoConfig {
    pub model_path: String,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub execution_provider: ExecutionProvider,
}

#[derive(uniffi::Record)]
pub struct EstimateResult {
    pub pcm_samples: u64,
    pub pcm_bytes: u64,
    pub duration_ms: u64,
}

// ─── Callback interface — app implements this ──────────────────────────────

/// FFI callback interface for receiving audio data.
/// PCM data is delivered as raw bytes (little-endian i16 samples, 32kHz mono).
#[uniffi::export(with_foreign)]
pub trait FfiAudioSink: Send + Sync {
    /// Write PCM data bytes to the sink. Must block if buffer is full.
    /// Returns number of bytes written.
    fn write_pcm(&self, pcm_data: Vec<u8>) -> i64;

    /// Available space in bytes.
    fn available_bytes(&self) -> u64;

    /// Called when a sentence finishes synthesis.
    fn on_sentence_complete(&self, sentence_index: u32);

    /// Called when all queued sentences are done.
    fn on_drain_complete(&self);

    /// Called on inference error.
    fn on_error(&self, message: String);
}

/// Adapter that bridges FfiAudioSink (callback interface) to core AudioSink trait.
struct SinkAdapter {
    inner: Arc<dyn FfiAudioSink>,
}

impl AudioSink for SinkAdapter {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        // Convert i16 slice to bytes (little-endian)
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let bytes_written = self.inner.write_pcm(bytes);
        if bytes_written < 0 {
            return Err(SinkError::Closed);
        }
        // Convert bytes written back to samples
        Ok((bytes_written as usize) / 2)
    }

    fn available(&self) -> usize {
        (self.inner.available_bytes() as usize) / 2
    }

    fn on_sentence_complete(&mut self, sentence_index: usize) {
        self.inner.on_sentence_complete(sentence_index as u32);
    }

    fn on_drain_complete(&mut self) {
        self.inner.on_drain_complete();
    }

    fn on_error(&mut self, error: String) {
        self.inner.on_error(error);
    }
}

// ─── Main engine object ────────────────────────────────────────────────────

#[derive(uniffi::Object)]
pub struct SopranoTts {
    inner: Mutex<soprano_core::SopranoTTS>,
}

#[uniffi::export]
impl SopranoTts {
    /// Create a new TTS engine with the given config and audio sink callback.
    #[uniffi::constructor]
    pub fn new(config: SopranoConfig, sink: Arc<dyn FfiAudioSink>) -> Result<Arc<Self>, FfiError> {
        let core_config = soprano_core::SopranoConfig {
            model_path: config.model_path,
            temperature: config.temperature,
            top_k: config.top_k as usize,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            execution_provider: (&config.execution_provider).into(),
        };

        let adapter = SinkAdapter { inner: sink };
        let engine = soprano_core::SopranoTTS::new(core_config, Box::new(adapter))
            .map_err(FfiError::from)?;

        Ok(Arc::new(Self {
            inner: Mutex::new(engine),
        }))
    }

    /// Estimate worst-case output size for a text.
    pub fn estimate(&self, text: String) -> EstimateResult {
        let engine = self.inner.lock().unwrap();
        let est = engine.estimate(&text);
        EstimateResult {
            pcm_samples: est.pcm_samples as u64,
            pcm_bytes: est.pcm_bytes as u64,
            duration_ms: est.duration_ms,
        }
    }

    /// Feed text for synthesis. Non-blocking — queues internally.
    /// Returns error if input exceeds 512 tokens after normalization.
    pub fn feed(&self, text: String) -> Result<(), FfiError> {
        let engine = self.inner.lock().unwrap();
        engine.feed(&text).map_err(FfiError::from)
    }

    /// Stop current inference and discard queued sentences.
    pub fn flush(&self) {
        let engine = self.inner.lock().unwrap();
        engine.flush();
    }

    /// Block until all queued sentences finish writing to sink.
    pub fn drain(&self) {
        let engine = self.inner.lock().unwrap();
        engine.drain();
    }

    /// Update sampling parameters (takes effect on next sentence).
    pub fn set_params(
        &self,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        repetition_penalty: f32,
    ) {
        let mut engine = self.inner.lock().unwrap();
        engine.set_params(temperature, top_k as usize, top_p, repetition_penalty);
    }
}
