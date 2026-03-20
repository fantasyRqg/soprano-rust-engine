//! Public API for the Soprano TTS engine.

use std::path::Path;
use std::sync::mpsc;
use std::thread;

use ort::session::Session;
use thiserror::Error;

use crate::audio::sink::AudioSink;
use crate::inference::backbone;
use crate::inference::decoder;
use crate::inference::sampler::SamplingParams;
use crate::inference::session::*;
use crate::text::normalizer;
use crate::text::tokenizer::{SopranoTokenizer, MAX_TOKENS};

/// Execution provider for ONNX Runtime inference.
#[derive(Debug, Clone, Copy, Default)]
pub enum ExecutionProvider {
    /// Default CPU execution.
    #[default]
    Cpu,
    /// Android NNAPI (delegates to NPU/GPU). Falls back to CPU for unsupported ops.
    Nnapi,
    /// XNNPACK optimized CPU kernels for ARM.
    Xnnpack,
    /// Apple CoreML (delegates to ANE/GPU on iOS/macOS).
    CoreMl,
}

/// Configuration for the Soprano TTS engine.
pub struct SopranoConfig {
    /// Path to directory containing backbone.onnx, decoder.onnx, tokenizer.json.
    pub model_path: String,
    /// Temperature for sampling. 0.0 = greedy (default).
    pub temperature: f32,
    /// Top-k sampling. 0 = disabled (default).
    pub top_k: usize,
    /// Top-p (nucleus) sampling threshold. Default 0.95.
    pub top_p: f32,
    /// Repetition penalty. Default 1.2.
    pub repetition_penalty: f32,
    /// Execution provider for inference acceleration. Default: Cpu.
    pub execution_provider: ExecutionProvider,
}

impl Default for SopranoConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            temperature: 0.0,
            top_k: 0,
            top_p: 0.95,
            repetition_penalty: 1.2,
            execution_provider: ExecutionProvider::Cpu,
        }
    }
}

/// Estimated output size for a text input.
pub struct EstimateResult {
    /// Estimated number of i16 PCM samples.
    pub pcm_samples: usize,
    /// Estimated bytes (pcm_samples * 2).
    pub pcm_bytes: usize,
    /// Estimated audio duration in milliseconds.
    pub duration_ms: u64,
}

#[derive(Debug, Error)]
pub enum SopranoError {
    #[error("model loading failed: {0}")]
    ModelLoadError(String),
    #[error("input too long: {token_count} tokens exceeds max {max_tokens}")]
    InputTooLong { token_count: usize, max_tokens: usize },
    #[error("no sink attached — call set_sink() before feed()")]
    NoSink,
    #[error("inference error: {0}")]
    InferenceError(String),
    #[error("hallucination detected")]
    Hallucination,
    #[error("tokenization error: {0}")]
    TokenizationError(String),
}

/// Internal message for the worker thread.
enum WorkerMsg {
    Feed { text: String },
    Flush,
    Drain { done_tx: mpsc::Sender<()> },
    UpdateParams(SamplingParams),
    Shutdown,
}

/// The Soprano TTS engine.
pub struct SopranoTTS {
    worker_tx: mpsc::Sender<WorkerMsg>,
    worker_handle: Option<thread::JoinHandle<()>>,
    sampling_params: SamplingParams,
}

impl SopranoTTS {
    /// Load models from the configured model_path and start the worker thread.
    pub fn new(config: SopranoConfig, sink: Box<dyn AudioSink>) -> Result<Self, SopranoError> {
        let model_path = Path::new(&config.model_path);

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = SopranoTokenizer::from_file(&tokenizer_path)
            .map_err(SopranoError::ModelLoadError)?;

        // Load ONNX sessions
        let backbone_path = find_backbone_model(model_path)?;
        let decoder_path = find_decoder_model(model_path)?;

        let mut backbone_session = load_session(&backbone_path, &config.execution_provider)
            .map_err(SopranoError::ModelLoadError)?;
        let mut decoder_session = load_session(&decoder_path, &config.execution_provider)
            .map_err(SopranoError::ModelLoadError)?;

        let sampling_params = SamplingParams {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
        };

        let (tx, rx) = mpsc::channel::<WorkerMsg>();
        let initial_params = sampling_params.clone();

        let handle = thread::spawn(move || {
            worker_loop(rx, &tokenizer, &mut backbone_session, &mut decoder_session, sink, initial_params);
        });

        Ok(Self {
            worker_tx: tx,
            worker_handle: Some(handle),
            sampling_params,
        })
    }

    /// Estimate worst-case upper bound for output size.
    pub fn estimate(&self, text: &str) -> EstimateResult {
        // Heuristic: ~1 token per 3-4 chars, max 512 tokens, 2048 samples per token
        let estimated_tokens = (text.len() / 3).min(MAX_NEW_TOKENS);
        let pcm_samples = estimated_tokens * SAMPLES_PER_TOKEN;
        let pcm_bytes = pcm_samples * 2;
        let duration_ms = (pcm_samples as u64 * 1000) / SAMPLE_RATE as u64;
        EstimateResult {
            pcm_samples,
            pcm_bytes,
            duration_ms,
        }
    }

    /// Feed text for synthesis. Non-blocking — queues internally.
    /// Engine normalizes, tokenizes, and errors if >512 tokens.
    pub fn feed(&self, text: &str) -> Result<(), SopranoError> {
        self.worker_tx
            .send(WorkerMsg::Feed { text: text.to_string() })
            .map_err(|_| SopranoError::InferenceError("worker thread died".to_string()))
    }

    /// Stop current inference and discard queued sentences.
    pub fn flush(&self) {
        let _ = self.worker_tx.send(WorkerMsg::Flush);
    }

    /// Block until all queued sentences finish writing to sink.
    pub fn drain(&self) {
        let (done_tx, done_rx) = mpsc::channel();
        if self.worker_tx.send(WorkerMsg::Drain { done_tx }).is_ok() {
            let _ = done_rx.recv();
        }
    }

    /// Update inference parameters (takes effect on next sentence).
    pub fn set_params(
        &mut self,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) {
        self.sampling_params = SamplingParams {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        };
        let _ = self.worker_tx.send(WorkerMsg::UpdateParams(self.sampling_params.clone()));
    }
}

impl Drop for SopranoTTS {
    fn drop(&mut self) {
        let _ = self.worker_tx.send(WorkerMsg::Shutdown);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Worker thread that processes the sentence queue.
fn worker_loop(
    rx: mpsc::Receiver<WorkerMsg>,
    tokenizer: &SopranoTokenizer,
    backbone: &mut Session,
    decoder: &mut Session,
    mut sink: Box<dyn AudioSink>,
    mut params: SamplingParams,
) {
    let mut sentence_index = 0usize;

    loop {
        match rx.recv() {
            Ok(WorkerMsg::Feed { text }) => {
                // Normalize text
                let normalized = normalizer::normalize(&text);

                // Tokenize
                let token_ids = match tokenizer.encode(&normalized) {
                    Ok(ids) => ids,
                    Err(e) => {
                        sink.on_error(format!("tokenization error: {}", e));
                        continue;
                    }
                };

                // Check length limit
                if token_ids.len() > MAX_TOKENS {
                    sink.on_error(format!(
                        "input too long: {} tokens exceeds max {}",
                        token_ids.len(),
                        MAX_TOKENS
                    ));
                    continue;
                }

                // Convert to i64 for ONNX
                let input_ids: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();

                // Run backbone generation
                let backbone_output = match backbone::generate(backbone, &input_ids, &params) {
                    Ok(out) => out,
                    Err(e) => {
                        sink.on_error(format!("backbone error: {}", e));
                        continue;
                    }
                };

                if backbone_output.hallucinated {
                    sink.on_error("hallucination detected".to_string());
                    sentence_index += 1;
                    sink.on_sentence_complete(sentence_index);
                    continue;
                }

                // Run decoder with streaming
                if !backbone_output.hidden_states.is_empty() {
                    if let Err(e) = decoder::decode_streaming(
                        decoder,
                        &backbone_output.hidden_states,
                        &mut *sink,
                    ) {
                        sink.on_error(format!("decoder error: {}", e));
                    }
                }

                sentence_index += 1;
                sink.on_sentence_complete(sentence_index);
            }
            Ok(WorkerMsg::Flush) => {
                // Drain remaining messages from the channel
                while let Ok(msg) = rx.try_recv() {
                    if let WorkerMsg::Shutdown = msg {
                        return;
                    }
                }
                sentence_index = 0;
            }
            Ok(WorkerMsg::Drain { done_tx }) => {
                let _ = done_tx.send(());
                sink.on_drain_complete();
            }
            Ok(WorkerMsg::UpdateParams(new_params)) => {
                params = new_params;
            }
            Ok(WorkerMsg::Shutdown) | Err(_) => {
                return;
            }
        }
    }
}

/// Find backbone model file (try f16 first, then f32).
fn find_backbone_model(model_dir: &Path) -> Result<std::path::PathBuf, SopranoError> {
    let candidates = [
        "soprano_backbone_kv_f16.onnx",
        "soprano_backbone_kv.onnx",
    ];
    for name in candidates {
        let path = model_dir.join(name);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(SopranoError::ModelLoadError(format!(
        "backbone model not found in {}",
        model_dir.display()
    )))
}

/// Find decoder model file (try f16 first, then f32).
fn find_decoder_model(model_dir: &Path) -> Result<std::path::PathBuf, SopranoError> {
    let candidates = [
        "soprano_decoder_f16.onnx",
        "soprano_decoder.onnx",
    ];
    for name in candidates {
        let path = model_dir.join(name);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(SopranoError::ModelLoadError(format!(
        "decoder model not found in {}",
        model_dir.display()
    )))
}
