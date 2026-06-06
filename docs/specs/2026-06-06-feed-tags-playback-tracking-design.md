# Feed Tags & Playback-Timeline Tracking — Design

**Date:** 2026-06-06
**Status:** Approved, pending implementation plan

## Goal

Let the application layer:

1. **Highlight the sentence currently being heard**, in sync with playback.
2. **Recover a resume point when stopped** (which sentence was playing).

The app implements the audio sink and owns the playback clock. The engine's
job is to tell the app *where each sentence's audio begins on the continuous
PCM stream*, tagged with an app-defined identifier.

## Core idea

The app attaches an **opaque `u64` tag** to each `feed` call. The engine carries
the tag through untouched and echoes it back at the exact sample offset where
that feed's audio begins, via a new `on_sentence_start(tag, sample_offset)`
callback. The engine never interprets the tag; the app maps it to its own
sentence (array index, db id, handle — its choice).

The PCM byte path is unchanged. All new information travels as a side-channel
callback carrying integers only.

### Why `u64`, not a string or character range

- Crosses UniFFI with zero allocation per feed (a `String` copies).
- Highlighting and resume need only an *identity*; the app already holds the
  text. No payload required.
- Avoids the alternative of making the regex-based normalizer offset-preserving
  so the engine could report character ranges into the original text — a large,
  fragile change to a sensitive part of the codebase. The app owns sentence
  boundaries by feeding one highlightable unit per `feed`.

## API changes

### Core — `crates/soprano-core/src/tts.rs`

- `feed(&self, text: &str)` → `feed(&self, text: &str, tag: u64)`
- `WorkerMsg::Feed { text }` → `WorkerMsg::Feed { text, tag }`

### Core — `crates/soprano-core/src/audio/sink.rs`

- **Remove** `on_sentence_complete(&mut self, sentence_index: usize)`.
- **Add** `on_sentence_start(&mut self, tag: u64, sample_offset: u64)`.

End-of-sentence is no longer signalled explicitly: the end of sentence K is
implied by the start of sentence K+1, and the end of the whole stream is
`on_drain_complete()`.

Resulting trait:

```rust
pub trait AudioSink: Send {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError>;
    fn available(&self) -> usize;
    fn on_sentence_start(&mut self, tag: u64, sample_offset: u64);
    fn on_drain_complete(&mut self);
    fn on_error(&mut self, error: String);
}
```

### FFI — `crates/soprano-ffi/src/lib.rs`

- `feed(text: String) -> Result<(), FfiError>` → `feed(text: String, tag: u64) -> Result<(), FfiError>`
- `FfiAudioSink`: **remove** `on_sentence_complete(&self, sentence_index: u32)`,
  **add** `on_sentence_start(&self, tag: u64, sample_offset: u64)`.
- `SinkAdapter`: drop the `on_sentence_complete` forward, add an
  `on_sentence_start` forward (straight pass-through; no narrowing — `u64`
  end-to-end).

## Behavior & data flow

The worker tracks a running `total_samples_written: u64`, summing the sample
count the decoder reports for each chunk (`decode_streaming_cancellable`'s
return value, currently ignored).

For each `Feed { text, tag }`, **before decoding its first chunk**, the worker
calls:

```rust
sink.on_sentence_start(tag, total_samples_written);
```

Because writes are strictly sequential and ordered, and no other feed
interleaves, `total_samples_written` at that instant is exactly the sample index
where this feed's audio begins. The next `write_pcm` belongs to this feed.

The existing chunker may split one feed into several internal chunks — this is
invisible to the app. `on_sentence_start` fires **once per feed**, at its first
sample; the remaining chunks continue writing audio under the same tag.

`sample_offset` units: cumulative **i16 mono samples** at 32 kHz. Sample `N` is
heard at `N / 32000` seconds.

### Reset semantics

`total_samples_written` resets to `0` on `flush()` (alongside the existing
cancel-flag reset). This matches the app discarding its playback buffer on stop:
after a flush, both sides count from the start of a fresh stream.

### Edge cases

- **Feed produces no audio** (all chunks hallucinate or error): the marker still
  fires with the current offset; the next feed's marker shares the same offset.
  The app's "largest offset ≤ cursor" lookup naturally skips the zero-length
  sentence (ties resolve to the last-appended marker). `on_error` still reports
  the failure.
- **Cancelled mid-feed**: the marker may already have fired; the app ignores
  markers beyond where audio actually stopped.

## How the app uses it

The callback fires at **write time**, which runs *ahead* of playback by the
buffer depth. The app must **not** highlight directly on the callback.

- **Build a timeline:** each `on_sentence_start(tag, offset)` appends
  `(offset, tag)` to a sorted list.
- **Highlight:** from the audio player's *played-sample count* (cursor), pick the
  marker with the largest `offset ≤ cursor` (last one on ties). That tag is
  audible now → highlight it.
- **Resume after stop:** on `flush`, read the cursor, find the current tag —
  that's the resume point. Re-feed from that tag onward (app owns the text).
  Resume granularity is per-sentence; audio cannot resume mid-sentence.

## Out of scope

- Character-level / word-level alignment within a sentence.
- Offset-preserving normalization or character-range reporting.
- Mid-sentence resume.
- Changes to the PCM byte format or transport.
