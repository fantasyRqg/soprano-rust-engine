//! Regression test: flush() must be able to interrupt a blocked drain().
//!
//! The drain() call blocks until all queued audio has been written, which with
//! a real audio sink happens at playback rate. If the FFI object serializes
//! calls (e.g. behind a Mutex), a Stop button calling flush() blocks until the
//! entire utterance has played — a UI hang / ANR on mobile.

use soprano_ffi::{ExecutionProvider, FfiAudioSink, SopranoConfig, SopranoTts};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

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

/// Sink that simulates real-time playback backpressure: every write blocks
/// for a while, like an audio buffer that's full.
struct SlowSink {
    started: AtomicBool,
}

impl FfiAudioSink for SlowSink {
    fn write_pcm(&self, _tag: u64, pcm_data: Vec<u8>) -> i64 {
        self.started.store(true, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(200));
        pcm_data.len() as i64
    }

    fn available_bytes(&self) -> u64 {
        0
    }

    fn on_drain_complete(&self) {}
    fn on_error(&self, _tag: u64, _message: String) {}
}

#[test]
fn flush_interrupts_blocked_drain() {
    if !models_available() {
        eprintln!("Skipping: model files not found in {:?}", models_dir());
        return;
    }

    let sink = Arc::new(SlowSink {
        started: AtomicBool::new(false),
    });

    let config = SopranoConfig {
        model_path: models_dir().to_string_lossy().to_string(),
        temperature: 0.0,
        top_k: 0,
        top_p: 0.95,
        repetition_penalty: 1.2,
        execution_provider: ExecutionProvider::Cpu,
    };

    let engine = SopranoTts::new(config, sink.clone()).expect("failed to create engine");

    // Long single chunk (no punctuation) so synthesis runs for a while.
    let long_text = "hello world again and again we keep talking ".repeat(4);
    engine.feed(long_text, 1).expect("feed failed");

    // Wait until audio is actually flowing into the (slow) sink.
    let deadline = Instant::now() + Duration::from_secs(60);
    while !sink.started.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "synthesis never started");
        std::thread::sleep(Duration::from_millis(10));
    }

    // Block a background thread in drain(), like an app's speak task.
    let engine_for_drain = engine.clone();
    let (drain_done_tx, drain_done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        engine_for_drain.drain();
        let _ = drain_done_tx.send(());
    });

    // Give the drain thread time to enter drain() and block.
    std::thread::sleep(Duration::from_millis(300));

    // Stop button: flush() must return promptly even while drain() is blocked.
    let t0 = Instant::now();
    engine.flush();
    let flush_elapsed = t0.elapsed();
    assert!(
        flush_elapsed < Duration::from_secs(2),
        "flush() blocked for {:?} behind drain() — cancellation cannot interrupt playback",
        flush_elapsed
    );

    // And the cancellation must let drain() finish quickly afterwards.
    assert!(
        drain_done_rx.recv_timeout(Duration::from_secs(30)).is_ok(),
        "drain() did not complete after flush()"
    );
}
