//! End-to-end integration test for the Soprano TTS engine.

use soprano_core::{AudioSink, SinkError, SopranoConfig, SopranoTTS};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::Duration;

/// Test sink that collects audio and records every on_sentence_start marker.
/// Each marker stores `(tag, reported_offset, samples_received_at_marker)` so
/// tests can assert the reported offset matches the sink's own received count
/// at the instant the marker fired.
struct CollectorSink {
    samples: Arc<Mutex<Vec<i16>>>,
    starts: Arc<Mutex<Vec<(u64, u64, u64)>>>,
    drain_complete: Arc<(Mutex<bool>, Condvar)>,
    errors: Arc<Mutex<Vec<String>>>,
    max_samples: usize,
}

impl CollectorSink {
    fn new(max_samples: usize) -> Self {
        Self {
            samples: Arc::new(Mutex::new(Vec::new())),
            starts: Arc::new(Mutex::new(Vec::new())),
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

    fn starts(&self) -> Arc<Mutex<Vec<(u64, u64, u64)>>> {
        self.starts.clone()
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

    fn on_sentence_start(&mut self, tag: u64, sample_offset: u64) {
        let received = self.samples.lock().unwrap().len() as u64;
        self.starts.lock().unwrap().push((tag, sample_offset, received));
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
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
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
        eprintln!(
            "Skipping e2e test: model files not found in {:?}",
            models_dir()
        );
        return;
    }

    let sink = CollectorSink::new(1_000_000); // ~15 seconds of audio
    let samples_ref = sink.samples();
    let errors_ref = sink.errors();
    let starts_ref = sink.starts();

    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        temperature: 0.0, // greedy for deterministic results
        ..Default::default()
    };

    let engine = SopranoTTS::new(config, Box::new(sink)).expect("failed to create engine");

    // Two separate feeds → exactly two start markers, one per feed.
    engine.feed("Hello world.", 100).expect("feed 1 failed");
    engine.feed("Enough of this.", 200).expect("feed 2 failed");
    engine.drain();

    let samples = samples_ref.lock().unwrap();
    let errors = errors_ref.lock().unwrap();
    let starts = starts_ref.lock().unwrap();

    // Print diagnostics
    eprintln!(
        "Generated {} audio samples ({:.2}s at 32kHz)",
        samples.len(),
        samples.len() as f64 / 32000.0
    );
    eprintln!("Errors: {:?}", *errors);
    eprintln!("Start markers (tag, offset, received): {:?}", *starts);

    // Should have generated some audio
    assert!(!samples.is_empty(), "no audio samples generated");
    assert!(samples.len() > 1000, "too few samples: {}", samples.len());

    // Two feeds → two start markers, tags echoed in order.
    assert_eq!(starts.len(), 2, "expected 2 start markers");
    assert_eq!(starts[0].0, 100, "first marker tag");
    assert_eq!(starts[1].0, 200, "second marker tag");
    // First feed starts at sample 0.
    assert_eq!(starts[0].1, 0, "first marker offset must be 0");
    // Second feed starts after the first feed's audio (> 0).
    assert!(starts[1].1 > 0, "second marker offset must be > 0");
    // Invariant: reported offset == samples the sink had received at marker time.
    for (tag, offset, received) in starts.iter() {
        assert_eq!(
            offset, received,
            "marker tag {} offset {} != samples received {}",
            tag, offset, received
        );
    }

    // Should have no errors
    assert!(errors.is_empty(), "unexpected errors: {:?}", *errors);

    // Audio should be within i16 range (sanity check)
    let max_abs = samples.iter().map(|&s| s.unsigned_abs()).max().unwrap_or(0);
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

    // Feed a single oversized sentence that should still exceed 512 tokens.
    let long_text = "hello ".repeat(700);
    engine
        .feed(&long_text, 0)
        .expect("feed should succeed (async)");
    engine.drain();

    let errors = errors_ref.lock().unwrap();
    eprintln!("Errors for long input: {:?}", *errors);
    assert!(
        !errors.is_empty(),
        "expected an error for overly long input"
    );
    assert!(
        errors[0].contains("too long"),
        "expected 'too long' error, got: {}",
        errors[0]
    );
}

#[test]
fn test_e2e_flush_then_drain_completes() {
    if !models_available() {
        return;
    }

    let sink = CollectorSink::new(1_000_000);
    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        ..Default::default()
    };

    let engine = SopranoTTS::new(config, Box::new(sink)).expect("failed to create engine");
    engine
        .feed(&"hello ".repeat(700), 0)
        .expect("feed should succeed");
    engine.flush();

    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        engine.drain();
        let _ = done_tx.send(());
    });

    assert!(
        done_rx.recv_timeout(Duration::from_secs(10)).is_ok(),
        "flush followed by drain did not complete"
    );
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
    eprintln!(
        "Estimate: {} samples, {} bytes, {}ms",
        est.pcm_samples, est.pcm_bytes, est.duration_ms
    );

    assert!(est.pcm_samples > 0);
    assert_eq!(est.pcm_bytes, est.pcm_samples * 2);
    assert!(est.duration_ms > 0);
}

#[test]
fn test_e2e_one_marker_per_feed() {
    if !models_available() {
        return;
    }

    let sink = CollectorSink::new(1_000_000);
    let starts_ref = sink.starts();

    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        temperature: 0.0,
        ..Default::default()
    };

    let engine = SopranoTTS::new(config, Box::new(sink)).expect("failed to create engine");

    // A single feed containing two sentences — the chunker may split it into
    // multiple internal chunks, but the marker must fire exactly once per feed.
    engine
        .feed("Hello world. Enough of this.", 42)
        .expect("feed failed");
    engine.drain();

    let starts = starts_ref.lock().unwrap();
    eprintln!("Start markers: {:?}", *starts);
    assert_eq!(starts.len(), 1, "one feed must produce exactly one marker");
    assert_eq!(starts[0].0, 42, "marker tag");
    assert_eq!(starts[0].1, 0, "single feed starts at offset 0");
}

#[test]
fn test_e2e_offset_resets_after_flush() {
    if !models_available() {
        return;
    }

    let sink = CollectorSink::new(1_000_000);
    let starts_ref = sink.starts();

    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        temperature: 0.0,
        ..Default::default()
    };

    let engine = SopranoTTS::new(config, Box::new(sink)).expect("failed to create engine");

    // First feed produces some audio, advancing the cumulative offset.
    engine.feed("Hello world.", 1).expect("feed 1 failed");
    engine.drain();

    // Flush resets the stream; drain() blocks until the worker has processed
    // the flush, so the next feed starts from a reset offset of 0.
    engine.flush();
    engine.drain();

    engine.feed("Enough of this.", 2).expect("feed 2 failed");
    engine.drain();

    let starts = starts_ref.lock().unwrap();
    eprintln!("Start markers across flush: {:?}", *starts);
    assert_eq!(starts.len(), 2, "expected 2 markers");
    assert_eq!(starts[0].0, 1);
    assert_eq!(starts[1].0, 2);
    // Both feeds start at offset 0 because flush reset the counter between them.
    assert_eq!(starts[0].1, 0, "first feed offset");
    assert_eq!(starts[1].1, 0, "offset must reset to 0 after flush");
}
