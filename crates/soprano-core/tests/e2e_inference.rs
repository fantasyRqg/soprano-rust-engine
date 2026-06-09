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
    errors: Arc<Mutex<Vec<(u64, String)>>>,
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

    fn errors(&self) -> Arc<Mutex<Vec<(u64, String)>>> {
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
        self.starts
            .lock()
            .unwrap()
            .push((tag, sample_offset, received));
    }

    fn on_drain_complete(&mut self) {
        let (lock, cvar) = &*self.drain_complete;
        let mut done = lock.lock().unwrap();
        *done = true;
        cvar.notify_all();
    }

    fn on_error(&mut self, tag: u64, error: String) {
        self.errors.lock().unwrap().push((tag, error));
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
        errors[0].1.contains("too long"),
        "expected 'too long' error, got: {}",
        errors[0].1
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

    // estimate() documents a worst-case upper bound. Long text is split into
    // many chunks, each of which can generate up to MAX_NEW_TOKENS, so the
    // estimate must not be capped at a single chunk's worth of samples.
    let single_chunk_cap = 512 * 2048; // MAX_NEW_TOKENS * SAMPLES_PER_TOKEN
    let long_text = "This is a sentence that will become its own chunk. ".repeat(200);
    let est_long = engine.estimate(&long_text);
    eprintln!(
        "Long-text estimate: {} samples (single-chunk cap: {})",
        est_long.pcm_samples, single_chunk_cap
    );
    assert!(
        est_long.pcm_samples > single_chunk_cap,
        "estimate for ~10k chars of text is capped at one chunk ({} samples) — not an upper bound",
        est_long.pcm_samples
    );
}

#[test]
fn test_e2e_one_marker_per_feed() {
    if !models_available() {
        return;
    }

    // Precondition: this input must split into multiple chunks, otherwise the
    // "one marker per multi-chunk feed" property is not actually exercised.
    let chunk_count =
        soprano_core::chunk_normalized(&soprano_core::normalize("Hello world. Enough of this."))
            .len();
    assert!(
        chunk_count > 1,
        "test input must produce multiple chunks to be meaningful, got {chunk_count}"
    );
    eprintln!("Precondition chunk_count: {chunk_count}");

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
fn test_e2e_feed_after_flush_is_not_discarded() {
    if !models_available() {
        return;
    }

    let sink = CollectorSink::new(10_000_000);
    let starts_ref = sink.starts();
    let errors_ref = sink.errors();

    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        temperature: 0.0,
        ..Default::default()
    };

    let engine = SopranoTTS::new(config, Box::new(sink)).expect("failed to create engine");

    // A long single chunk (no punctuation → no chunk splits) keeps the worker
    // busy in the backbone loop for a while.
    let long_text = "hello world again and again we keep talking ".repeat(4);
    engine.feed(&long_text, 1).expect("feed 1 failed");

    // Wait until the worker has actually picked up feed 1 (its marker fires
    // just before synthesis starts).
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while starts_ref.lock().unwrap().is_empty() {
        assert!(
            std::time::Instant::now() < deadline,
            "feed 1 never started synthesizing"
        );
        std::thread::sleep(Duration::from_millis(10));
    }

    // Barge-in: cancel the current utterance and immediately say something new.
    // The new feed is queued behind the Flush message; it must survive it.
    engine.flush();
    engine.feed("Enough of this.", 99).expect("feed 2 failed");
    engine.drain();

    let starts = starts_ref.lock().unwrap();
    let errors = errors_ref.lock().unwrap();
    eprintln!("Start markers: {:?}", *starts);
    eprintln!("Errors: {:?}", *errors);

    assert!(
        starts.iter().any(|&(tag, _, _)| tag == 99),
        "feed sent after flush() was discarded — no marker for tag 99: {:?}",
        *starts
    );
    assert!(errors.is_empty(), "unexpected errors: {:?}", *errors);
}

#[test]
fn test_e2e_chunk_error_aborts_rest_of_feed() {
    if !models_available() {
        return;
    }

    let sink = CollectorSink::new(10_000_000);
    let samples_ref = sink.samples();
    let errors_ref = sink.errors();

    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        temperature: 0.0,
        ..Default::default()
    };

    let engine = SopranoTTS::new(config, Box::new(sink)).expect("failed to create engine");

    // First chunk exceeds the 512-token limit and errors; the trailing
    // sentence forms a second, valid chunk. Synthesizing it anyway would
    // play a fragment with everything before it silently missing, so the
    // whole feed must abort at the first chunk error.
    let text = format!("{}. And a short tail sentence.", "hello ".repeat(700));
    engine.feed(&text, 7).expect("feed failed");
    engine.drain();

    let samples = samples_ref.lock().unwrap();
    let errors = errors_ref.lock().unwrap();
    eprintln!("Errors: {:?}", *errors);
    eprintln!("Samples after failed chunk: {}", samples.len());

    assert!(!errors.is_empty(), "expected a 'too long' error");
    assert_eq!(errors[0].0, 7, "error must carry the failed feed's tag");
    assert!(
        errors[0].1.contains("too long"),
        "expected 'too long' error, got: {}",
        errors[0].1
    );
    assert!(
        samples.is_empty(),
        "feed continued past a failed chunk: {} samples synthesized after the error",
        samples.len()
    );
}

/// Sink that accepts the first write, then reports Closed forever.
struct ClosingSink {
    write_calls: Arc<Mutex<u64>>,
    drain_complete: Arc<(Mutex<bool>, Condvar)>,
}

impl AudioSink for ClosingSink {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        let mut calls = self.write_calls.lock().unwrap();
        *calls += 1;
        if *calls == 1 {
            Ok(samples.len())
        } else {
            Err(SinkError::Closed)
        }
    }

    fn available(&self) -> usize {
        0
    }

    fn on_sentence_start(&mut self, _tag: u64, _sample_offset: u64) {}

    fn on_drain_complete(&mut self) {
        let (lock, cvar) = &*self.drain_complete;
        *lock.lock().unwrap() = true;
        cvar.notify_all();
    }

    fn on_error(&mut self, _tag: u64, _error: String) {}
}

#[test]
fn test_e2e_closed_sink_stops_synthesis() {
    if !models_available() {
        return;
    }

    let write_calls = Arc::new(Mutex::new(0u64));
    let sink = ClosingSink {
        write_calls: write_calls.clone(),
        drain_complete: Arc::new((Mutex::new(false), Condvar::new())),
    };

    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        temperature: 0.0,
        ..Default::default()
    };

    let engine = SopranoTTS::new(config, Box::new(sink)).expect("failed to create engine");

    // Long single chunk → many decode windows → many write attempts if the
    // engine keeps going against a closed sink.
    let long_text = "hello world again and again we keep talking ".repeat(4);
    engine.feed(&long_text, 1).expect("feed failed");
    engine.drain();

    let calls = *write_calls.lock().unwrap();
    eprintln!("write() calls against closing sink: {}", calls);
    assert!(
        calls <= 2,
        "engine kept synthesizing against a closed sink: {} write calls (expected ≤ 2)",
        calls
    );
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
