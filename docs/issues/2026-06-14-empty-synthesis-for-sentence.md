# Engine emits a sentence boundary but zero audio for some inputs

**Status:** Resolved — superseded the lazy-emit `on_sentence_start` fix by
removing the boundary callback entirely (see *Resolution* below).
**Found:** 2026-06-14
**Reported from:** Hearken-Android (consumer of `soprano-ffi`)
**Severity:** High — a sentence is silently dropped from playback (no audio, and the
host app cannot highlight it either).

## Summary

For certain sentence inputs the engine fires the `onSentenceStart` boundary
callback for the sentence's tag but then **produces no PCM samples at all**
before firing the *next* sentence's boundary. The sentence is effectively
skipped: the listener hears the previous sentence run straight into the
following one.

This is an engine-level defect. The host fed the text correctly with the right
tag, and the engine acknowledged the tag — it just synthesized nothing for it.

## Reproduction

A real book passage (Agatha Christie, *Partners in Crime*) reliably drops the
sentence **"A very profound statement, Tuppence."**

The host feeds six consecutive sentences from one paragraph pair. Tags are the
host's sentence indices:

```
tag 0: "“So Tommy and Tuppence were married,” she chanted, ..."
tag 1: "And six years later they were still living together happily ever afterwards. "
tag 2: "It is extraordinary,” she said, “how different everything always is from what you think it is going to be."
tag 3: "“A very profound statement, Tuppence. "      <-- dropped
tag 4: "But not original. "
tag 5: "Eminent poets and still more eminent divines have said it before—and, ..."
```

The drop is also reproducible feeding sentences in isolation (no host playback
stack involved), so it is below the FFI boundary.

## Evidence

Instrumentation in the host's `FfiAudioSink` implementation records, per
engine callback, the PCM bytes written since the previous boundary and the
sample offset reported by the engine:

```
BOUNDARY tag=2 sampleOffset=266240 bytesSincePrev=241664   <- tag 2 synthesized audio
BOUNDARY tag=3 sampleOffset=425984 bytesSincePrev=319488   <- end of tag 2's audio
BOUNDARY tag=4 sampleOffset=425984 bytesSincePrev=0        <- tag 3 produced ZERO bytes
BOUNDARY tag=5 sampleOffset=466944 bytesSincePrev=81920    <- tag 4 synthesized audio
```

Key facts:

- Between `onSentenceStart(tag=3)` and `onSentenceStart(tag=4)`, **`writePcm`
  was never called** (`bytesSincePrev=0`).
- The two boundaries report the **identical** input sample offset (`425984`),
  i.e. tag 3 spans zero samples.
- The very next sentence (tag 4, "But not original.") synthesized normally, so
  the engine recovers — the empty output is specific to tag 3's text.

## Why this is bad for consumers

A zero-length sentence is unrepresentable in a position/tag timeline: two
boundaries at the same sample offset collapse, so a host that maps the audio
playback head back to a tag can never observe tag 3. The consequence in
Hearken-Android is that the sentence has **no audio and no text highlight** —
it vanishes from the UI as well as the audio.

Even an ideal host cannot recover the audio (there are no samples to play), so
the fix belongs in the engine.

## Suspected area

The text reaches the engine as `"“A very profound statement, Tuppence. "`.
After normalization and chunking (`crates/soprano-core/src/text/`), something
about this input yields an empty synthesis pass:

- `chunk_normalized` (`chunker.rs`) could be producing a chunk that the model
  turns into zero frames, or dropping content. Note the sentence ends in
  "Tuppence." — a normal word ending, **not** an abbreviation or single-letter
  initial, so it should be a plain strong boundary. The leading curly quote
  `“` (U+201C) is a candidate: worth checking how the normalizer treats a
  chunk that is (after stripping) only punctuation/quote + short content.
- Worth confirming whether normalization reduces some chunk to empty/whitespace
  or to tokens the model emits EOS on immediately.

Suggested next step: log the normalized text and the chunk list
(`chunk_normalized` output) for this exact input, and the per-chunk token/frame
counts, to see which stage collapses tag 3 to zero output.

## Suggested engine-side guarantees

1. A non-empty input that is acknowledged with an `onSentenceStart` for its tag
   must produce **at least one** PCM frame (or the engine should not emit the
   boundary at all / should emit an `onError` for that tag instead of silently
   skipping).
2. Two consecutive boundaries should never share the same sample offset.

## Repro assets in the consumer repo

Failing regression tests (track the sentence tag via `onSentenceStart`, no audio
assertion needed) in Hearken-Android:

- `cc.hearken.app.engine.SopranoEngineTest#feeding_consecutive_short_sentences_reports_every_tag`
- `cc.hearken.app.integration.PlaybackIntegrationTest#first_sentence_of_paragraph_is_not_skipped`

## Resolution

The root weakness was the **separate `on_sentence_start(tag, sample_offset)`
boundary callback**: the host had to correlate a separately-delivered offset
against its own running sample count, and that correlation could drop, collide
at a shared offset (this bug), or alias against the host's poller. Two fixes
were considered:

1. *Lazy emit* (commit `7faa5c8`): only fire the boundary in the same breath as
   the feed's first real sample write, so a zero-output sentence fires no
   boundary at all. This closed the colliding-offset leak but kept the fragile
   two-channel design.
2. *Tagged buffers* (this change, **adopted**): delete `on_sentence_start` and
   carry the feed's `tag` on every `AudioSink::write` / `FfiAudioSink.write_pcm`.
   The engine synthesizes one feed at a time, so every sample in a write belongs
   to exactly one tag — the host attributes audio to a sentence directly, with
   nothing to correlate. A zero-output sentence writes no tagged audio and is
   reported via `on_error`; a sentence with audio is unambiguously tagged. The
   desync is now **unrepresentable** rather than guarded against.

This is a breaking FFI change: the Kotlin host must read the tag off `write_pcm`
and build its position→tag map from the writes, dropping the `on_sentence_start`
correlation (and the offset-reset/poller logic that went with it).

Regression coverage (engine repo):

- `crates/soprano-core/tests/decoder_streaming.rs` —
  `zero_output_stream_writes_no_tagged_audio` and `tagged_writes_carry_the_streams_tag`:
  deterministic, model-light (synthetic hidden states, no backbone) guards that a
  zero-output stream writes nothing and that every write carries the stream's tag.
- `crates/soprano-core/tests/e2e_inference.rs` —
  `test_e2e_consecutive_short_sentences_each_produce_audio` (the original passage)
  now asserts every fed tag is either audible or reported via `on_error`, never
  silently absent from both.
