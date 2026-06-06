# Feed Tags & Playback-Timeline Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the app attach an opaque `u64` tag to each `feed` and receive it back via a new `on_sentence_start(tag, sample_offset)` callback at the exact PCM sample where that sentence's audio begins, enabling in-sync sentence highlighting and stop/resume.

**Architecture:** Replace the per-chunk `on_sentence_complete(index)` callback with a per-feed `on_sentence_start(tag, sample_offset)` callback. The worker thread tracks a running `total_samples_written: u64` (summing the decoder's reported sample counts) and fires the callback once per feed, before decoding that feed's first chunk, with the current cumulative sample count. The PCM byte path is unchanged; only integers travel on the callback. The offset resets to 0 on `flush`.

**Tech Stack:** Rust workspace (`soprano-core`, `soprano-ffi`, `soprano-cli`, `soprano-bench`), UniFFI 0.31 for Swift/Kotlin/Python bindings, ONNX Runtime (`ort`) for inference. Tests are model-gated integration tests under `crates/soprano-core/tests/` that skip when model files are absent.

**Spec:** `docs/specs/2026-06-06-feed-tags-playback-tracking-design.md`

---

## File Structure

Files modified (no new files):

- `crates/soprano-core/src/audio/sink.rs` — `AudioSink` trait: remove `on_sentence_complete`, add `on_sentence_start`.
- `crates/soprano-core/src/tts.rs` — `WorkerMsg::Feed` gains `tag`; `feed()` gains `tag`; `worker_loop` tracks `total_samples_written` and fires `on_sentence_start`; `flush` resets the counter.
- `crates/soprano-ffi/src/lib.rs` — `FfiAudioSink` trait, `SinkAdapter`, and `feed()` updated to match.
- `crates/soprano-cli/src/main.rs` — `WavSink` sink-method update; `feed` call gains a tag.
- `crates/soprano-bench/src/main.rs` — `CountingSink` sink-method update; `feed` call gains a tag.
- `crates/soprano-core/tests/decoder_streaming.rs` — `Collector` test sink method update.
- `crates/soprano-core/tests/e2e_inference.rs` — `CollectorSink` rewritten to record start markers; feed calls and assertions updated; new behavioral tests added.

**Note on TDD ordering:** The trait signature change is cross-cutting — it breaks compilation in every implementor at once, so the "test fails to compile" state is trivial. Tasks 1–2 are mechanical signature/plumbing refactors verified by `cargo build`. The real behavior (per-feed marker, correct `sample_offset`, flush reset) is gated by the behavioral tests in Task 3, which is the meaningful test-first step for the logic. Task 3 asserts the core invariant: **at the instant `on_sentence_start(tag, offset)` fires, `offset` equals the number of samples the sink has already received.**

---

## Task 1: Update the core `AudioSink` trait and all in-tree implementors

Mechanical signature change so the workspace compiles before behavior is wired.

**Files:**
- Modify: `crates/soprano-core/src/audio/sink.rs:23-24`
- Modify: `crates/soprano-cli/src/main.rs:80`
- Modify: `crates/soprano-bench/src/main.rs:76`
- Modify: `crates/soprano-core/tests/decoder_streaming.rs:19`

- [ ] **Step 1: Change the trait method in `sink.rs`**

In `crates/soprano-core/src/audio/sink.rs`, replace the `on_sentence_complete` declaration (lines 23-24):

```rust
    /// Called when a sentence finishes.
    fn on_sentence_complete(&mut self, sentence_index: usize);
```

with:

```rust
    /// Called once per `feed`, just before that sentence's audio begins
    /// streaming to the sink. `tag` is the app-defined value passed to `feed`.
    /// `sample_offset` is the cumulative i16 sample count (mono, 32kHz) written
    /// to this sink since the start of the current stream (resets on flush) —
    /// i.e. the sample index at which this sentence's audio begins.
    fn on_sentence_start(&mut self, tag: u64, sample_offset: u64);
```

- [ ] **Step 2: Update `WavSink` in the CLI**

In `crates/soprano-cli/src/main.rs:80`, replace:

```rust
    fn on_sentence_complete(&mut self, _sentence_index: usize) {}
```

with:

```rust
    fn on_sentence_start(&mut self, _tag: u64, _sample_offset: u64) {}
```

- [ ] **Step 3: Update `CountingSink` in the bench**

In `crates/soprano-bench/src/main.rs:76`, replace:

```rust
    fn on_sentence_complete(&mut self, _: usize) {}
```

with:

```rust
    fn on_sentence_start(&mut self, _: u64, _: u64) {}
```

- [ ] **Step 4: Update `Collector` in the streaming test**

In `crates/soprano-core/tests/decoder_streaming.rs:19`, replace:

```rust
    fn on_sentence_complete(&mut self, _: usize) {}
```

with:

```rust
    fn on_sentence_start(&mut self, _: u64, _: u64) {}
```

- [ ] **Step 5: Verify the core crate compiles (FFI + e2e test still broken — expected)**

Run: `cargo build -p soprano-core -p soprano-cli -p soprano-bench`
Expected: PASS. (`soprano-ffi` and the `e2e_inference` test are updated in later tasks and are NOT built here.)

- [ ] **Step 6: Commit**

```bash
git add crates/soprano-core/src/audio/sink.rs crates/soprano-cli/src/main.rs crates/soprano-bench/src/main.rs crates/soprano-core/tests/decoder_streaming.rs
git commit -m "refactor: replace AudioSink::on_sentence_complete with on_sentence_start"
```

---

## Task 2: Thread tag + sample offset through `tts.rs`

**Files:**
- Modify: `crates/soprano-core/src/tts.rs:93-99` (WorkerMsg)
- Modify: `crates/soprano-core/src/tts.rs:179-185` (feed)
- Modify: `crates/soprano-core/src/tts.rs:245-361` (worker_loop)
- Modify: `crates/soprano-cli/src/main.rs:139` (feed call)
- Modify: `crates/soprano-bench/src/main.rs:133` (feed call)

- [ ] **Step 1: Add `tag` to `WorkerMsg::Feed`**

In `crates/soprano-core/src/tts.rs`, change the `Feed` variant (line 94):

```rust
    Feed { text: String, tag: u64 },
```

- [ ] **Step 2: Add `tag` to the public `feed` method**

Replace the `feed` method (lines 179-185):

```rust
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
            })
            .map_err(|_| SopranoError::InferenceError("worker thread died".to_string()))
    }
```

- [ ] **Step 3: Track samples and fire the marker in `worker_loop`**

In `crates/soprano-core/src/tts.rs`, change the counter declaration (line 245):

```rust
    let mut sentence_index = 0usize;
```

to:

```rust
    let mut total_samples_written: u64 = 0;
```

Then replace the entire `Ok(WorkerMsg::Feed { text }) => { ... }` arm (lines 249-343) with:

```rust
            Ok(WorkerMsg::Feed { text, tag }) => {
                if cancel_flag.load(Ordering::SeqCst) {
                    continue;
                }

                // Normalize text
                let normalized = normalizer::normalize(&text);
                let chunks = chunker::chunk_normalized(&normalized);

                if chunks.is_empty() {
                    sink.on_error("normalized input was empty".to_string());
                    continue;
                }

                // One marker per feed: the next samples written belong to this
                // feed, starting at the current cumulative sample offset.
                sink.on_sentence_start(tag, total_samples_written);

                let mut cancelled = false;
                for chunk in chunks {
                    if cancel_flag.load(Ordering::SeqCst) {
                        cancelled = true;
                        break;
                    }

                    // Tokenize
                    let token_ids = match tokenizer.encode(&chunk) {
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
                    let backbone_output = match backbone::generate_cancellable(
                        backbone,
                        &input_ids,
                        &params,
                        || cancel_flag.load(Ordering::SeqCst),
                    ) {
                        Ok(Some(out)) => out,
                        Ok(None) => {
                            cancelled = true;
                            break;
                        }
                        Err(e) => {
                            sink.on_error(format!("backbone error: {}", e));
                            continue;
                        }
                    };

                    if backbone_output.hallucinated {
                        sink.on_error("hallucination detected".to_string());
                        continue;
                    }

                    // Run decoder with streaming
                    if !backbone_output.hidden_states.is_empty() {
                        match decoder::decode_streaming_cancellable(
                            decoder,
                            &backbone_output.hidden_states,
                            &mut *sink,
                            || cancel_flag.load(Ordering::SeqCst),
                        ) {
                            Ok(Some(n)) => {
                                total_samples_written += n as u64;
                            }
                            Ok(None) => {
                                cancelled = true;
                                break;
                            }
                            Err(e) => {
                                sink.on_error(format!("decoder error: {}", e));
                            }
                        }
                    }
                }

                if cancelled {
                    continue;
                }
            }
```

Key changes from the original: `tag` destructured; `on_sentence_start(tag, total_samples_written)` fired once before the chunk loop; the hallucination branch no longer increments a counter or fires a completion callback (just `on_error` + `continue`); the decoder match captures `n` and accumulates it; the per-chunk `sentence_index += 1; sink.on_sentence_complete(...)` is removed.

- [ ] **Step 4: Reset the counter on flush**

In the `Ok(WorkerMsg::Flush)` arm, replace (line 359):

```rust
                sentence_index = 0;
```

with:

```rust
                total_samples_written = 0;
```

- [ ] **Step 5: Update the CLI feed call**

In `crates/soprano-cli/src/main.rs:139`, replace `engine.feed(&text)` with `engine.feed(&text, 0)`:

```rust
    engine.feed(&text, 0).unwrap_or_else(|e| {
```

(The CLI synthesizes one blob and ignores markers, so a constant tag of `0` is fine.)

- [ ] **Step 6: Update the bench feed call**

In `crates/soprano-bench/src/main.rs:133`, replace `engine.feed(&cli.text)` with `engine.feed(&cli.text, 0)`:

```rust
        if let Err(e) = engine.feed(&cli.text, 0) {
```

- [ ] **Step 7: Verify core/cli/bench compile**

Run: `cargo build -p soprano-core -p soprano-cli -p soprano-bench`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/soprano-core/src/tts.rs crates/soprano-cli/src/main.rs crates/soprano-bench/src/main.rs
git commit -m "feat: thread feed tag and emit on_sentence_start with sample offset"
```

---

## Task 3: Behavioral tests in `e2e_inference.rs`

Rewrite the test sink to record start markers, then test the per-feed semantics, the offset invariant, and the flush reset. These are model-gated (skip without models), matching the existing tests.

**Files:**
- Modify: `crates/soprano-core/tests/e2e_inference.rs`

- [ ] **Step 1: Rewrite `CollectorSink` to record start markers**

Replace the struct, its `impl CollectorSink` block, and its `impl AudioSink` block (lines 8-77) with:

```rust
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
```

- [ ] **Step 2: Update `test_e2e_basic_synthesis` to use start markers**

In `test_e2e_basic_synthesis`, replace the `sentences_ref` binding (line 105):

```rust
    let sentences_ref = sink.sentences();
```

with:

```rust
    let starts_ref = sink.starts();
```

Replace the single feed (lines 116-118):

```rust
    engine
        .feed("Hello world. Enough of this.")
        .expect("feed failed");
```

with two tagged feeds:

```rust
    // Two separate feeds → exactly two start markers, one per feed.
    engine.feed("Hello world.", 100).expect("feed 1 failed");
    engine.feed("Enough of this.", 200).expect("feed 2 failed");
```

Replace the `sentences` binding and its diagnostics/assertions (lines 123, 132, 138-139):

```rust
    let sentences = sentences_ref.lock().unwrap();
```
```rust
    eprintln!("Sentences completed: {:?}", *sentences);
```
```rust
    // Should have completed two chunked sentences
    assert_eq!(sentences.len(), 2, "expected 2 sentence complete callbacks");
```

with:

```rust
    let starts = starts_ref.lock().unwrap();
```
```rust
    eprintln!("Start markers (tag, offset, received): {:?}", *starts);
```
```rust
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
```

- [ ] **Step 3: Run the updated basic-synthesis test (skips without models)**

Run: `cargo test -p soprano-core --test e2e_inference test_e2e_basic_synthesis -- --nocapture`
Expected: PASS (or a "Skipping e2e test: model files not found" line and PASS if models are absent).

- [ ] **Step 4: Add a test that one multi-sentence feed yields exactly one marker**

Append this test to `crates/soprano-core/tests/e2e_inference.rs`:

```rust
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
```

- [ ] **Step 5: Run the per-feed test**

Run: `cargo test -p soprano-core --test e2e_inference test_e2e_one_marker_per_feed -- --nocapture`
Expected: PASS (or skip-and-PASS without models).

- [ ] **Step 6: Add a test that the offset resets after flush**

Append this test to `crates/soprano-core/tests/e2e_inference.rs`:

```rust
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
```

- [ ] **Step 7: Run the flush-reset test**

Run: `cargo test -p soprano-core --test e2e_inference test_e2e_offset_resets_after_flush -- --nocapture`
Expected: PASS (or skip-and-PASS without models).

- [ ] **Step 8: Run the full e2e test file**

Run: `cargo test -p soprano-core --test e2e_inference -- --nocapture`
Expected: all tests PASS (or skip-and-PASS without models).

- [ ] **Step 9: Commit**

```bash
git add crates/soprano-core/tests/e2e_inference.rs
git commit -m "test: cover on_sentence_start markers, per-feed semantics, flush reset"
```

---

## Task 4: Update the FFI layer

**Files:**
- Modify: `crates/soprano-ffi/src/lib.rs:109-110` (trait method)
- Modify: `crates/soprano-ffi/src/lib.rs:140-142` (adapter)
- Modify: `crates/soprano-ffi/src/lib.rs:196-199` (feed)

- [ ] **Step 1: Change the `FfiAudioSink` trait method**

In `crates/soprano-ffi/src/lib.rs`, replace (lines 109-110):

```rust
    /// Called when a sentence finishes synthesis.
    fn on_sentence_complete(&self, sentence_index: u32);
```

with:

```rust
    /// Called once per `feed`, just before that sentence's audio begins
    /// streaming. `tag` is the value passed to `feed`; `sample_offset` is the
    /// i16 sample index (mono, 32kHz, resets on flush) where this sentence's
    /// audio starts. Build a (sample_offset → tag) timeline and compare against
    /// your audio player's played-sample cursor to highlight in sync.
    fn on_sentence_start(&self, tag: u64, sample_offset: u64);
```

- [ ] **Step 2: Update the `SinkAdapter` forward**

Replace the adapter method (lines 140-142):

```rust
    fn on_sentence_complete(&mut self, sentence_index: usize) {
        self.inner.on_sentence_complete(sentence_index as u32);
    }
```

with (no narrowing — `u64` end-to-end):

```rust
    fn on_sentence_start(&mut self, tag: u64, sample_offset: u64) {
        self.inner.on_sentence_start(tag, sample_offset);
    }
```

- [ ] **Step 3: Add `tag` to the FFI `feed`**

Replace the `feed` method (lines 194-199):

```rust
    /// Feed text for synthesis. Non-blocking — queues internally.
    /// Synthesis errors are delivered asynchronously through `on_error`.
    pub fn feed(&self, text: String) -> Result<(), FfiError> {
        let engine = self.inner.lock().unwrap();
        engine.feed(&text).map_err(FfiError::from)
    }
```

with:

```rust
    /// Feed text for synthesis. Non-blocking — queues internally.
    /// `tag` is an opaque app-defined identifier echoed back via
    /// `on_sentence_start` at the sample where this feed's audio begins.
    /// Synthesis errors are delivered asynchronously through `on_error`.
    pub fn feed(&self, text: String, tag: u64) -> Result<(), FfiError> {
        let engine = self.inner.lock().unwrap();
        engine.feed(&text, tag).map_err(FfiError::from)
    }
```

- [ ] **Step 4: Verify the FFI crate compiles**

Run: `cargo build -p soprano-ffi`
Expected: PASS. (This regenerates the in-crate UniFFI scaffolding via the proc-macros. Swift/Kotlin/Python binding files are regenerated by the platform build scripts under `examples/ios/build-rust.sh` and `examples/android/build-rust.sh`; foreign sink implementations must rename `onSentenceComplete` → `onSentenceStart(tag, sampleOffset)` and pass a tag to `feed`.)

- [ ] **Step 5: Commit**

```bash
git add crates/soprano-ffi/src/lib.rs
git commit -m "feat(ffi): expose feed tag and on_sentence_start across the FFI boundary"
```

---

## Task 5: Full-workspace verification

**Files:** none (verification only)

- [ ] **Step 1: Format**

Run: `cargo fmt --all`
Expected: no output; if files change, `git add -A && git commit -m "style: cargo fmt"`.

- [ ] **Step 2: Build the whole workspace**

Run: `cargo build --workspace`
Expected: PASS.

- [ ] **Step 3: Clippy across all targets**

Run: `cargo clippy --workspace --all-targets`
Expected: no errors (no new warnings introduced).

- [ ] **Step 4: Run the full test suite**

Run: `cargo test --workspace -- --nocapture`
Expected: PASS. Model-gated tests print a "Skipping" line if models are absent; that is still a PASS.

- [ ] **Step 5: Confirm no `on_sentence_complete` references remain**

Run: `grep -rn "on_sentence_complete" crates --include="*.rs"`
Expected: no output (empty result).

- [ ] **Step 6: Final commit (if fmt or any cleanup changed files)**

```bash
git add -A
git commit -m "chore: workspace verification for feed-tags-playback-tracking" --allow-empty
```

---

## Self-Review Notes

- **Spec coverage:** `feed(text, tag)` → Task 2/4. `on_sentence_start(tag, sample_offset)` added + `on_sentence_complete` removed → Task 1/2/4. Once-per-feed marker before first chunk → Task 2 Step 3, tested Task 3 Step 4. `sample_offset` = cumulative i16 samples, offset invariant → Task 2 Step 3, tested Task 3 Step 2. Reset on flush → Task 2 Step 4, tested Task 3 Step 6. `sentence_index` deleted → Task 2 Step 3. Zero-audio/hallucination edge (marker still fires, `on_error` reports) → Task 2 Step 3 (hallucination branch keeps `on_error`, drops the counter).
- **Type consistency:** `tag: u64` and `sample_offset: u64` are used identically in `AudioSink`, `FfiAudioSink`, `SinkAdapter`, `WorkerMsg::Feed`, and both `feed` signatures. Decoder return `Ok(Some(n))` where `n: usize` is cast `as u64` before accumulation.
- **No placeholders:** every code/command step contains concrete content.
