# soprano-core — Design Notes

This document explains the **non-obvious internal designs** of the core engine —
the pieces where the code is doing something subtle for a reason. For build
instructions, the public API, and platform setup, see the [root README](../../README.md).

Audience: anyone modifying the inference pipeline, the streaming/cancellation
logic, or the `AudioSink` contract.

---

## 1. The journey of one `feed()`

```
feed(text, tag)                       [public API, non-blocking]
        │  (mpsc channel)
        ▼
worker thread ── normalize ── chunk ──┐
                                      │  for each chunk:
                                      ▼
                       tokenize → backbone (autoregressive) ──┐
                                                              │ hidden states, one per token
                                                              ▼
                                              streaming decoder (windowed, lossless)
                                                              │ PCM i16
                                                              ▼
                                                         AudioSink::write
```

The whole pipeline runs on a **single dedicated worker thread**. `feed()` only
enqueues a message and returns; everything below happens off the caller's thread.

Module map:

| Concern | Module |
|---|---|
| Public facade, worker loop, flush/drain | `tts.rs` |
| Text normalization | `text/normalizer.rs` |
| Sentence-like chunking | `text/chunker.rs` |
| BPE tokenizer + special tokens | `text/tokenizer.rs` |
| Autoregressive generation + KV cache | `inference/backbone.rs` |
| Token sampling | `inference/sampler.rs` |
| Vocos decoder, windowed + streaming | `inference/decoder.rs` |
| Model/streaming constants | `inference/session.rs` |
| `AudioSink` trait + errors | `audio/sink.rs` |
| f32 → i16 PCM conversion | `audio/convert.rs` |
| Env-gated timing instrumentation | `profile.rs` |

---

## 2. Worker thread & message protocol

`SopranoTTS::new` spawns one worker thread holding the two ONNX `Session`s
(backbone + decoder) and the `Box<dyn AudioSink>`. ONNX `Session`s are heavy and
not cheaply shareable, so keeping them thread-local on a single worker avoids
locking and lets generation and decoding share state freely.

The public methods are thin wrappers that send a `WorkerMsg` over an mpsc channel:

- `Feed { text, tag, epoch }` — synthesize text
- `Flush` — reset the stream's sample offset (see §3)
- `Drain { done_tx }` — block the caller until the queue is empty
- `UpdateParams` — swap sampling parameters for the next chunk
- `Shutdown` — sent from `Drop`

Because there is exactly one consumer, message ordering is the queue order, and
backpressure naturally propagates: if `AudioSink::write` blocks, the worker
blocks, and feeds pile up in the channel rather than in memory-hungry buffers.

---

## 3. Epoch-based flush & cancellation

**Problem:** `flush()` must cancel in-flight synthesis *and* discard already-queued
feeds — but must **not** discard feeds the app submits *after* `flush()` returns.
A naive "drain the channel" approach races: a feed sent microseconds after flush
could be thrown away.

**Solution:** a monotonic `AtomicU64` epoch shared between the API and the worker.

- Each `feed()` stamps its message with the epoch value *at submission time*.
- `flush()` does `epoch.fetch_add(1)` then sends `Flush`.
- The worker, when it dequeues a `Feed`, compares the stamped epoch to the
  current epoch. Older → silently skipped (it predates a flush).
- During generation/decoding, the cancellation predicate is simply
  `feed_epoch != epoch.load()`. It's checked between ONNX calls and between
  sink writes, so a bumped epoch cancels the current chunk promptly.

This makes "cancel everything before now, keep everything after now" a single
atomic compare, with no channel draining and no race. `Drop` also bumps the epoch
so in-flight work cancels at its next check before the thread joins.

> **Caveat:** cancellation is only observed *between* ONNX ops and *between* sink
> writes. A sink that blocks forever inside `write()` will stall flush and
> `Drop`. The `AudioSink` contract (§9) requires `write()` to eventually return.

---

## 4. Text normalization & chunking

**Normalization** (`normalizer.rs`, a port of the Python `text_normalizer.py`)
expands numbers, dates, currencies, and abbreviations (`mr.` → `mister`, dotted
initials like `u.s.`) into spoken words *before* chunking.

**Chunking** (`chunker.rs`) splits normalized text into sentence-like units, each
wrapped in the model's control tokens `[STOP][TEXT]…[START]`. Each chunk is
generated independently. The rules:

- **Strong boundaries** (`.`/`!`/`?`) split unless the text ends in a surviving
  abbreviation (`prof.`, `sr.`, `vs.` — the only ones the normalizer doesn't
  already expand).
- **Soft boundaries** (`;`/`:`/`,`) split only once the current chunk has reached
  `TARGET_CHUNK_CHARS` (140), so long run-on clauses get broken up.
- A final `merge_short_chunks` pass coalesces neighbors when either is below
  `MIN_CHUNK_CHARS` (12) and the merge stays under `MAX_MERGED_CHUNK_CHARS` (220).

Chunk size is a **latency/quality tradeoff**: smaller chunks reach first-audio
sooner but give the model less context per utterance. (Historically chunk size
also drove decoder redundancy — see §7.)

---

## 5. Backbone autoregression & the zero-copy KV cache

`backbone.rs` runs a standard autoregressive loop: feed tokens, get logits +
hidden state, sample the next token, repeat until the stop token or
`MAX_NEW_TOKENS` (512). Two details are subtle.

### Prefill vs. decode steps

Step 0 processes the whole prompt at once (the **prefill**, ~0.1–0.2s for a
typical chunk); every subsequent step feeds a **single** token with the KV cache
carrying all prior context (~31 ms/token on a mid-range Android CPU). The vast
majority of wall-clock time is the per-token decode steps, which scale linearly
with output length — this is the fact that motivates §8.

### KV cache without per-step copies

The KV cache is 17 layers × 2 (key+value) = 34 tensors that **grow every step**.
Copying them in and out each iteration would be quadratic. Instead:

- The cache is held as `Vec<Option<DynValue>>` — **option slots**.
- Each step `.take()`s the values out of their slots and moves them into the
  session inputs (`past_key_values.{i}.{key,value}`).
- After `run()`, the owned `present.{i}.{key,value}` outputs are moved *back*
  into the slots via `outputs.remove(name)` — no clone of the growing cache.

So the cache is transferred by ownership, not copied. The KV input/output names
are formatted **once** before the loop, not per step.

> The non-streaming `generate()` / `generate_cancellable()` (used by tests and
> comparison tooling) and the streaming path share **one** generation loop,
> `generate_core`, which takes a per-token callback. See §8.

---

## 6. Token sampling

`sampler.rs` implements the usual stack, applied per step to the last position's
logits:

1. **Repetition penalty** (default 1.2) — divides/multiplies scores of
   already-seen tokens (tracked in a `seen_tokens` bitmask seeded from the prompt).
2. **Temperature** (default 0.0 = **greedy**; greedy short-circuits the rest).
3. **Top-k** (default 0 = off) then **top-p** nucleus (default 0.95).

Greedy is the default because it matches the Python reference bit-for-bit, which
is what the `compare_python` test asserts.

---

## 7. Windowed decoder — lossless streaming

The Vocos decoder turns hidden states into PCM. Decoding the whole sequence at
once would mean no audio until generation finishes and a large allocation. So the
decoder slides a window over the hidden states.

The key correctness property: **a window must carry `RECEPTIVE_FIELD` (4) tokens
of context on _both_ sides** of the frames it emits. With context on each side,
the emitted frames are bit-for-bit identical to decoding the whole sequence at
once — so there are **no seams to crossfade**. (An earlier left-context-only
design produced audible seams.)

Mechanics (`StreamingDecode` in `decoder.rs`):

- A window of `W` tokens emits `W − 1` audio frames of `SAMPLES_PER_TOKEN` (2048)
  each; a lone token yields no audio.
- To emit frames `[offset, chunk_end)` (chunks of `CHUNK_SIZE` = 8 frames), it
  decodes tokens `[offset − 4, chunk_end + 4)` and slices out the middle.

The lossless guarantee is locked down by `decoder_streaming.rs`, which asserts
streamed output matches `decode_all` within 1 LSB.

> Decoder *redundancy* = `(chunk + 2·RECEPTIVE_FIELD) / chunk`. Smaller emit
> chunks re-decode proportionally more context — the historical perf lever.

---

## 8. Interleaved generation + decoding  ⭐

This is the most important design in the engine, so it gets the most detail.

### The problem

Originally the worker ran generation to **completion** for a chunk, then handed
the full hidden-state buffer to the decoder. Because per-token decode dominates
(§5), time-to-first-audio was essentially *the entire generation time*, and it
**scaled with sentence length**:

| Sentence | tokens | old first-byte |
|---|---|---|
| short | 68 | ~2.2 s |
| long | 152 | ~5.0 s |

Yet the decoder's first window only needs ~12 tokens. The remaining hundreds of
milliseconds were the decoder sitting idle while finished hidden states piled up.

### The fix

Generation and decoding are **interleaved on the worker thread**. `backbone.rs`
exposes `generate_streaming_cancellable`, which invokes a callback with each
token's hidden state *as it is produced*. The callback pushes that state into a
`StreamingDecode`, which flushes any window that has gained enough right context.
Audio therefore starts after roughly a fixed warm-up, **independent of total
sentence length**:

| Sentence | tokens | new first-byte |
|---|---|---|
| short | 68 | ~0.9 s |
| long | 152 | ~1.1 s |

Measured on a Galaxy Tab A8 (SM-X230), CPU EP, f16 models.

```
generate_core (one loop, shared by both paths)
   │  per token: sample → extract hidden → on_step(token, &hidden)
   ├──────────────────────────────► generate_cancellable: on_step pushes into a
   │                                  Vec, returns BackboneOutput (tests/tooling)
   └──────────────────────────────► generate_streaming_cancellable: on_step feeds
                                      StreamingDecode.push(), audio streams live
```

### Why a holdback margin (the subtle part)

Eager streaming collides with **hallucination detection** (§9): the detector only
fires *after* the model has already emitted ~17 degenerate tokens, and the
decoder lags the generation frontier by only `RECEPTIVE_FIELD` (4). So a naive
interleave would emit ~13 frames (~0.8 s) of garbage to the sink before detection
trips — audio the old path discarded entirely.

`StreamingDecode` therefore keeps a **holdback margin** of
`STREAM_HOLDBACK_FRAMES` (= `HALLUCINATION_MAX_CONSECUTIVE` + 2 = 18) frames
un-emitted behind the frontier:

- **Mid-stream** (`push`): only flush frames that clear *both* the right-context
  requirement *and* the holdback margin.
- **Clean end** (`finish`): drain everything, ignoring the margin.
- **Hallucination/cancel:** the coordinator in `tts.rs` simply does **not** call
  `finish()`, so the held-back tail — which contains the entire degenerate run —
  is dropped and never reaches the sink.

This preserves the original invariant ("no hallucinated audio is ever played")
while still emitting the **good prefix** of the chunk. The margin is buffered
audio, not lost audio: on a clean finish it all drains. The cost is ~18 tokens of
warm-up before first audio (the ~0.9 s above) — constant, not length-dependent.

The holdback constant is deliberately tied to the detector's window in
`session.rs` so the two cannot silently drift apart.

> **Tradeoff to know:** because both ONNX sessions now share the single worker
> thread, per-token backbone steps slow slightly (~31→35 ms) and RTF drops
> (~1.5→1.43×) — still comfortably faster than real-time. Moving the decoder to a
> producer thread is a possible future optimization.

---

## 9. Hallucination detection

The model can get "stuck," emitting near-identical hidden states forever.
`HallucinationDetector` (`backbone.rs`) catches this with a consecutive-similarity
counter, run once per generated token:

1. Compute the **L1 distance** between this token's hidden vector and the previous
   one: `Σ |prevᵢ − hiddenᵢ|` over all `HIDDEN_DIM` (512) dims.
2. If `diff < threshold` (300.0) → increment `consecutive_similar`.
3. If `diff ≥ threshold` → **reset** the counter to 0 (any dissimilar token breaks
   the run).
4. Fire when `consecutive_similar > HALLUCINATION_MAX_CONSECUTIVE` (16), i.e. after
   17+ uninterrupted near-identical tokens.

On firing, generation stops and the chunk is aborted: `on_error(tag, "hallucination
detected")`, and (in the streaming path) the held-back tail is dropped (§8).

This is a **heuristic**, not a semantic check. Things to keep in mind if tuning:

- The `300.0` / `16` constants are empirical and tied to the model's hidden-state
  scale; the L1 sum is unnormalized, so the threshold is implicitly coupled to
  `HIDDEN_DIM`.
- It can produce a **false positive** on genuinely sustained, slowly-varying audio
  (a long held vowel, near-silence), cutting the utterance short.
- There is **no minimum-length guard** — it can fire as early as token 17, so an
  input that degenerates immediately yields little audio plus an `on_error`.

---

## 10. The `AudioSink` contract & backpressure

`AudioSink` (`audio/sink.rs`) is the app's output buffer. The engine relies on it
for **backpressure**: `write()` MUST block when full (or return an error). Because
the worker is single-threaded, a blocking `write()` throttles the whole pipeline,
which is the intended flow-control mechanism — the engine never builds unbounded
internal PCM buffers.

Rules the engine depends on:

- `write()` returning `Ok(0)` for non-empty input is treated as a failure (no
  progress).
- `write()` returning `Err(SinkError::Closed)` aborts the current feed **like a
  cancellation** — partial sample count is kept, decoding stops (running inference
  against a dead sink wastes work and drifts sample offsets).
- A blocked `write()` **must eventually return**; otherwise flush and `Drop` hang
  (§3).
- Callbacks fire in this order per feed: zero or more `write(tag, …)` → on
  failure `on_error` (then no more audio for that feed). A feed that synthesizes
  nothing produces no `write` at all, only `on_error`.

---

## 11. Feed tags & playback tracking

`feed(text, tag)` takes an opaque app-defined `tag`. The engine never interprets
it — it **attaches the tag to every `write(tag, samples)`** carrying that feed's
audio. The engine synthesizes one feed at a time, so every sample in a single
`write` belongs to exactly one tag (a write never straddles two feeds).

This lets an app map playback position back to which feed is currently audible
(e.g. for word/sentence highlighting) directly from the write stream — there is
no separate boundary callback to correlate against a sample count, drop, or
collide. A feed that produces no audio (e.g. a hallucination run whose held-back
frames are dropped) simply writes nothing for its tag and is reported via
`on_error`, so a sentence is never silently lost. Offset bookkeeping (including
resetting on `flush()`) is the host's: it tracks position from the bytes it
enqueues and clears that on flush. `e2e_inference.rs` and `decoder_streaming.rs`
pin this down (every fed tag is audible-or-errored; writes carry the tag; a
zero-output stream writes nothing).

> Note the streaming change (§8) slightly altered observable timing: on a
> hallucinating feed the sink now receives the good-prefix audio *before*
> `on_error`, whereas the old path emitted no audio for such a feed. The
> "no audio after `on_error`" rule still holds.

---

## 12. Profiling

All pipeline stages emit `[profile]` timing lines on stderr when `SOPRANO_PROFILE`
is set (any value). Implemented in `profile.rs` as a `OnceLock`-cached env check,
so it's a single bool read when disabled — zero measurable overhead in production.

```bash
SOPRANO_PROFILE=1 cargo run -p soprano-bench -- --model models/ --ep cpu -n 1
```

Emits per-stage timings: normalize+chunk, backbone prefill, per-token decode
average, each decoder window, **first-audio timestamp**, and per-chunk totals.
This is the instrumentation used to attribute the latency win in §8.

---

## 13. Key constants (`inference/session.rs`)

| Constant | Value | Meaning |
|---|---|---|
| `NUM_LAYERS` | 17 | Backbone transformer layers (KV cache depth) |
| `HIDDEN_DIM` | 512 | Hidden-state width |
| `VOCAB_SIZE` | 8192 | Token vocabulary |
| `MAX_NEW_TOKENS` | 512 | Generation cap per chunk |
| `SAMPLE_RATE` | 32000 | Output PCM sample rate (Hz) |
| `SAMPLES_PER_TOKEN` | 2048 | Audio frames the decoder emits per token |
| `RECEPTIVE_FIELD` | 4 | Decoder context tokens on each side of a window |
| `CHUNK_SIZE` | 8 | Frames emitted per decoder window |
| `HALLUCINATION_MAX_CONSECUTIVE` | 16 | Similar-token run length that trips detection |
| `STREAM_HOLDBACK_FRAMES` | 18 | Frames held back so a hallucination run never reaches the sink |

---

## 14. Tests that pin these designs

| Test | Guards |
|---|---|
| `decoder_streaming::streaming_decode_matches_whole_decode` | Windowed decode is lossless vs. `decode_all` (§7) |
| `decoder_streaming::interleaved_decode_with_holdback_matches_whole_decode` | Holdback path drains losslessly on clean finish (§8) |
| `e2e_inference::*` | Full `feed` path: synthesis, length limits, flush/drain, per-feed tagging, every-fed-tag-audible-or-errored, closed-sink abort, chunk-error abort |
| `compare_python` | Token + audio parity with the Python reference (§5–6) |

Integration tests require model files in `models/` and skip gracefully if absent.
