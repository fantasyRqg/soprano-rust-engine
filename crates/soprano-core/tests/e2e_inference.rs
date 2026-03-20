//! End-to-end integration test for the Soprano TTS engine.

use std::sync::{Arc, Mutex, Condvar};
use soprano_core::{AudioSink, SinkError, SopranoConfig, SopranoTTS};

/// Simple test sink that collects all audio samples.
struct CollectorSink {
    samples: Arc<Mutex<Vec<i16>>>,
    sentences_completed: Arc<Mutex<Vec<usize>>>,
    drain_complete: Arc<(Mutex<bool>, Condvar)>,
    errors: Arc<Mutex<Vec<String>>>,
    max_samples: usize,
}

impl CollectorSink {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: Arc::new(Mutex::new(Vec::new())),
            sentences_completed: Arc::new(Mutex::new(Vec::new())),
            drain_complete: Arc::new((Mutex::new(false), Condvar::new())),
            errors: Arc::new(Mutex::new(Vec::new())),
            max_samples,
        }
    }

    fn samples(&self) -> Arc<Mutex<Vec<i16>>> {
        self.samples.clone()
    }

    fn errors(&self) -> Arc<Mutex<Vec<String>>> {
        self.errors.clone()
    }

    fn sentences(&self) -> Arc<Mutex<Vec<usize>>> {
        self.sentences_completed.clone()
    }
}

impl AudioSink for CollectorSink {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        let mut buf = self.samples.lock().unwrap();
        let available = self.max_samples.saturating_sub(buf.len());
        let to_write = samples.len().min(available);
        if to_write == 0 && !samples.is_empty() {
            // Buffer full — in a real app this would block.
            // For testing, just accept all.
            buf.extend_from_slice(samples);
            return Ok(samples.len());
        }
        buf.extend_from_slice(&samples[..to_write]);
        Ok(to_write)
    }

    fn available(&self) -> usize {
        let buf = self.samples.lock().unwrap();
        self.max_samples.saturating_sub(buf.len())
    }

    fn on_sentence_complete(&mut self, sentence_index: usize) {
        self.sentences_completed.lock().unwrap().push(sentence_index);
    }

    fn on_drain_complete(&mut self) {
        let (lock, cvar) = &*self.drain_complete;
        let mut done = lock.lock().unwrap();
        *done = true;
        cvar.notify_all();
    }

    fn on_error(&mut self, error: String) {
        self.errors.lock().unwrap().push(error);
    }
}

fn models_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
}

fn models_available() -> bool {
    let dir = models_dir();
    dir.join("tokenizer.json").exists()
        && (dir.join("soprano_backbone_kv_f16.onnx").exists()
            || dir.join("soprano_backbone_kv.onnx").exists())
        && (dir.join("soprano_decoder_f16.onnx").exists()
            || dir.join("soprano_decoder.onnx").exists())
}

#[test]
fn test_e2e_basic_synthesis() {
    if !models_available() {
        eprintln!("Skipping e2e test: model files not found in {:?}", models_dir());
        return;
    }

    let sink = CollectorSink::new(1_000_000); // ~15 seconds of audio
    let samples_ref = sink.samples();
    let errors_ref = sink.errors();
    let sentences_ref = sink.sentences();

    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        temperature: 0.0, // greedy for deterministic results
        ..Default::default()
    };

    let engine = SopranoTTS::new(config, Box::new(sink)).expect("failed to create engine");

    // Feed a short sentence
    engine.feed("Hello world.").expect("feed failed");
    engine.drain();

    let samples = samples_ref.lock().unwrap();
    let errors = errors_ref.lock().unwrap();
    let sentences = sentences_ref.lock().unwrap();

    // Print diagnostics
    eprintln!("Generated {} audio samples ({:.2}s at 32kHz)",
        samples.len(),
        samples.len() as f64 / 32000.0
    );
    eprintln!("Errors: {:?}", *errors);
    eprintln!("Sentences completed: {:?}", *sentences);

    // Should have generated some audio
    assert!(!samples.is_empty(), "no audio samples generated");
    assert!(samples.len() > 1000, "too few samples: {}", samples.len());

    // Should have completed one sentence
    assert_eq!(sentences.len(), 1, "expected 1 sentence complete callback");

    // Should have no errors
    assert!(errors.is_empty(), "unexpected errors: {:?}", *errors);

    // Audio should be within i16 range (sanity check)
    let max_abs = samples.iter().map(|s| s.abs() as u16).max().unwrap_or(0);
    eprintln!("Max absolute sample value: {}", max_abs);
    assert!(max_abs > 0, "all samples are zero — likely a bug");
}

#[test]
fn test_e2e_input_too_long() {
    if !models_available() {
        return;
    }

    let sink = CollectorSink::new(1_000_000);
    let errors_ref = sink.errors();

    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        ..Default::default()
    };

    let engine = SopranoTTS::new(config, Box::new(sink)).expect("failed to create engine");

    // Feed a very long string that should exceed 512 tokens
    let long_text = "hello world. ".repeat(500);
    engine.feed(&long_text).expect("feed should succeed (async)");
    engine.drain();

    let errors = errors_ref.lock().unwrap();
    eprintln!("Errors for long input: {:?}", *errors);
    assert!(!errors.is_empty(), "expected an error for overly long input");
    assert!(errors[0].contains("too long"), "expected 'too long' error, got: {}", errors[0]);
}

#[test]
fn test_e2e_estimate() {
    if !models_available() {
        return;
    }

    let sink = CollectorSink::new(100);
    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        ..Default::default()
    };

    let engine = SopranoTTS::new(config, Box::new(sink)).expect("failed to create engine");

    let est = engine.estimate("Hello world.");
    eprintln!("Estimate: {} samples, {} bytes, {}ms",
        est.pcm_samples, est.pcm_bytes, est.duration_ms);

    assert!(est.pcm_samples > 0);
    assert_eq!(est.pcm_bytes, est.pcm_samples * 2);
    assert!(est.duration_ms > 0);
}
