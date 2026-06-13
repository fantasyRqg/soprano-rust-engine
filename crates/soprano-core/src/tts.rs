//! Public API for the Soprano TTS engine.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Instant;

use ort::session::Session;
use thiserror::Error;

use crate::audio::sink::AudioSink;
use crate::inference::backbone;
use crate::inference::decoder;
use crate::inference::sampler::SamplingParams;
use crate::inference::session::*;
use crate::profile::profile_log;
use crate::text::chunker;
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

/// Errors returned synchronously from the public API. Synthesis failures
/// (tokenization, length limits, inference, hallucination) happen on the
/// worker thread and are delivered through `AudioSink::on_error` instead.
#[derive(Debug, Error)]
pub enum SopranoError {
    #[error("model loading failed: {0}")]
    ModelLoadError(String),
    #[error("inference error: {0}")]
    InferenceError(String),
}

/// Internal message for the worker thread.
enum WorkerMsg {
    Feed { text: String, tag: u64, epoch: u64 },
    Flush,
    Drain { done_tx: mpsc::Sender<()> },
    UpdateParams(SamplingParams),
    Shutdown,
}

/// The Soprano TTS engine.
pub struct SopranoTTS {
    worker_tx: mpsc::Sender<WorkerMsg>,
    worker_handle: Option<thread::JoinHandle<()>>,
    /// Monotonic stream generation. Each feed is stamped with the epoch at
    /// submission time; `flush()` bumps it. The worker skips feeds from older
    /// epochs and cancels in-flight synthesis when the epoch moves, so feeds
    /// submitted *after* a flush are never discarded by it.
    epoch: Arc<AtomicU64>,
}

impl SopranoTTS {
    /// Load models from the configured model_path and start the worker thread.
    pub fn new(config: SopranoConfig, sink: Box<dyn AudioSink>) -> Result<Self, SopranoError> {
        let model_path = Path::new(&config.model_path);

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer =
            SopranoTokenizer::from_file(&tokenizer_path).map_err(SopranoError::ModelLoadError)?;

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
        let epoch = Arc::new(AtomicU64::new(0));
        let worker_epoch = Arc::clone(&epoch);

        let handle = thread::spawn(move || {
            worker_loop(
                rx,
                &tokenizer,
                &mut backbone_session,
                &mut decoder_session,
                sink,
                sampling_params,
                worker_epoch,
            );
        });

        Ok(Self {
            worker_tx: tx,
            worker_handle: Some(handle),
            epoch,
        })
    }

    /// Estimate a worst-case upper bound for output size.
    ///
    /// Text is normalized and chunked exactly like `feed` does, and each chunk
    /// can generate up to `MAX_NEW_TOKENS` regardless of its character count,
    /// so the bound is per-chunk, not per-text.
    pub fn estimate(&self, text: &str) -> EstimateResult {
        let normalized = normalizer::normalize(text);
        let num_chunks = chunker::chunk_normalized(&normalized).len().max(1);
        let pcm_samples = num_chunks * MAX_NEW_TOKENS * SAMPLES_PER_TOKEN;
        let pcm_bytes = pcm_samples * 2;
        let duration_ms = (pcm_samples as u64 * 1000) / SAMPLE_RATE as u64;
        EstimateResult {
            pcm_samples,
            pcm_bytes,
            duration_ms,
        }
    }

    /// Feed text for synthesis. Non-blocking — queues internally.
    ///
    /// `tag` is an opaque app-defined identifier echoed back via
    /// `AudioSink::on_sentence_start` at the sample where this feed's audio
    /// begins. The engine never interprets it.
    ///
    /// This only returns queueing errors, such as a stopped worker. Synthesis
    /// errors from normalization, tokenization, inference, or decoding are
    /// delivered asynchronously through `AudioSink::on_error`.
    pub fn feed(&self, text: &str, tag: u64) -> Result<(), SopranoError> {
        self.worker_tx
            .send(WorkerMsg::Feed {
                text: text.to_string(),
                tag,
                epoch: self.epoch.load(Ordering::SeqCst),
            })
            .map_err(|_| SopranoError::InferenceError("worker thread died".to_string()))
    }

    /// Request cancellation of current inference and discard sentences queued
    /// before this call. Feeds submitted after `flush()` returns are kept.
    ///
    /// Cancellation is checked between ONNX calls and decoder writes. If a sink
    /// blocks inside `write()`, cancellation takes effect after that call returns.
    pub fn flush(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        let _ = self.worker_tx.send(WorkerMsg::Flush);
    }

    /// Block until all queued sentences finish writing to sink, or until a
    /// pending flush has discarded the queue.
    pub fn drain(&self) {
        let (done_tx, done_rx) = mpsc::channel();
        if self.worker_tx.send(WorkerMsg::Drain { done_tx }).is_ok() {
            let _ = done_rx.recv();
        }
    }

    /// Update inference parameters (takes effect on next sentence).
    pub fn set_params(&self, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32) {
        let _ = self.worker_tx.send(WorkerMsg::UpdateParams(SamplingParams {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        }));
    }
}

/// Joins the worker thread. In-flight synthesis is cancelled at its next
/// check, but a sink currently blocked inside `write()` must return before
/// the join can complete — see the `AudioSink::write` contract.
impl Drop for SopranoTTS {
    fn drop(&mut self) {
        // Bump the epoch so in-flight synthesis cancels at its next check.
        self.epoch.fetch_add(1, Ordering::SeqCst);
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
    epoch: Arc<AtomicU64>,
) {
    let mut total_samples_written: u64 = 0;

    loop {
        match rx.recv() {
            Ok(WorkerMsg::Feed {
                text,
                tag,
                epoch: feed_epoch,
            }) => {
                // Stale feed: a flush happened after it was queued.
                if feed_epoch != epoch.load(Ordering::SeqCst) {
                    continue;
                }
                let is_cancelled = || feed_epoch != epoch.load(Ordering::SeqCst);
                let t_feed = Instant::now();

                // Normalize text
                let normalized = normalizer::normalize(&text);
                let chunks = chunker::chunk_normalized(&normalized);
                profile_log!(
                    "feed tag={}: normalize+chunk {}ms, {} chunks",
                    tag,
                    t_feed.elapsed().as_millis(),
                    chunks.len()
                );

                if chunks.is_empty() {
                    sink.on_error(tag, "normalized input was empty".to_string());
                    continue;
                }

                // One marker per feed: the next samples written belong to this
                // feed, starting at the current cumulative sample offset.
                sink.on_sentence_start(tag, total_samples_written);

                let mut cancelled = false;
                for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
                    if is_cancelled() {
                        cancelled = true;
                        break;
                    }
                    profile_log!(
                        "chunk {} start at +{}ms ({} chars)",
                        chunk_idx,
                        t_feed.elapsed().as_millis(),
                        chunk.len()
                    );

                    // Tokenize. Any chunk error aborts the rest of the feed:
                    // synthesizing later chunks would play audio with an
                    // unannounced gap in the middle of the utterance.
                    let token_ids = match tokenizer.encode(&chunk) {
                        Ok(ids) => ids,
                        Err(e) => {
                            sink.on_error(tag, format!("tokenization error: {}", e));
                            break;
                        }
                    };

                    // Check length limit
                    if token_ids.len() > MAX_TOKENS {
                        sink.on_error(
                            tag,
                            format!(
                                "input too long: {} tokens exceeds max {}",
                                token_ids.len(),
                                MAX_TOKENS
                            ),
                        );
                        break;
                    }

                    // Convert to i64 for ONNX
                    let input_ids: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();

                    // Interleave generation and decoding: each hidden state is
                    // pushed into the streaming decoder as the backbone produces
                    // it, so audio starts after roughly the holdback margin
                    // instead of after the whole chunk is generated. The decoder
                    // holds back STREAM_HOLDBACK_FRAMES so a hallucination run
                    // (detected only after the fact) never reaches the sink.
                    let mut stream =
                        decoder::StreamingDecode::new(&mut *decoder, STREAM_HOLDBACK_FRAMES);
                    let mut first_audio_logged = false;
                    let outcome = backbone::generate_streaming_cancellable(
                        &mut *backbone,
                        &input_ids,
                        &params,
                        is_cancelled,
                        |hidden| {
                            let cont = stream.push(hidden, &mut *sink, &is_cancelled)?;
                            if !first_audio_logged && stream.total_samples_written() > 0 {
                                profile_log!(
                                    "chunk {} first audio at +{}ms",
                                    chunk_idx,
                                    t_feed.elapsed().as_millis()
                                );
                                first_audio_logged = true;
                            }
                            Ok(cont)
                        },
                    );

                    match outcome {
                        Ok(backbone::StreamOutcome::Completed { generated }) => {
                            profile_log!(
                                "chunk {} backbone done at +{}ms ({} tokens), draining decoder",
                                chunk_idx,
                                t_feed.elapsed().as_millis(),
                                generated
                            );
                            // Clean end: drain the held-back margin.
                            match stream.finish(&mut *sink, &is_cancelled) {
                                Ok(true) => {
                                    total_samples_written += stream.total_samples_written() as u64;
                                    profile_log!(
                                        "chunk {} decode done at +{}ms ({} samples)",
                                        chunk_idx,
                                        t_feed.elapsed().as_millis(),
                                        stream.total_samples_written()
                                    );
                                }
                                Ok(false) => {
                                    cancelled = true;
                                    break;
                                }
                                Err(e) => {
                                    sink.on_error(tag, format!("decoder error: {}", e));
                                    break;
                                }
                            }
                        }
                        Ok(backbone::StreamOutcome::Hallucinated { generated }) => {
                            profile_log!(
                                "chunk {} hallucinated at +{}ms ({} tokens); dropping held-back tail",
                                chunk_idx,
                                t_feed.elapsed().as_millis(),
                                generated
                            );
                            // The degenerate run is still buffered behind the
                            // holdback margin; by NOT calling finish() those
                            // frames are dropped and never reach the sink.
                            sink.on_error(tag, "hallucination detected".to_string());
                            break;
                        }
                        Ok(backbone::StreamOutcome::Aborted) => {
                            cancelled = true;
                            break;
                        }
                        Err(e) => {
                            sink.on_error(tag, format!("backbone error: {}", e));
                            break;
                        }
                    }
                }

                if cancelled {
                    continue;
                }
            }
            Ok(WorkerMsg::Flush) => {
                // Stale feeds are skipped by the epoch check as they dequeue;
                // the flush itself only resets the stream's sample offset.
                total_samples_written = 0;
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
    let candidates = ["soprano_backbone_kv_f16.onnx", "soprano_backbone_kv.onnx"];
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
    let candidates = ["soprano_decoder_f16.onnx", "soprano_decoder.onnx"];
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
