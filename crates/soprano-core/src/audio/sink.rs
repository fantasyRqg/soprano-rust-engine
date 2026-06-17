use thiserror::Error;

#[derive(Debug, Error)]
pub enum SinkError {
    #[error("sink closed")]
    Closed,
    #[error("write failed: {0}")]
    WriteFailed(String),
}

/// App-provided output buffer. Engine writes PCM i16 data into it.
/// Implementations control memory allocation and backpressure.
pub trait AudioSink: Send {
    /// Write PCM i16 samples into the sink. `tag` is the app-defined value
    /// passed to `feed` that this audio belongs to: the engine synthesizes one
    /// feed at a time, so every sample in a single `write` belongs to exactly
    /// one `tag`, and the host attributes audio to a sentence directly — no
    /// separate boundary callback to correlate against a sample count. A feed
    /// that synthesizes no audio at all (e.g. a chunk the backbone collapses
    /// into a hallucination run) produces no `write` for its tag and is
    /// reported via [`on_error`](Self::on_error) instead.
    ///
    /// MUST block if buffer is full (provides backpressure) or return an error.
    /// Returning `Ok(0)` for non-empty input is treated as a write failure.
    ///
    /// A blocked `write` MUST eventually return (e.g. by returning
    /// `Err(SinkError::Closed)` when playback stops for good). Cancellation
    /// and engine shutdown are only checked between calls, so a `write` that
    /// never returns hangs `flush` semantics and blocks `SopranoTTS`'s `Drop`
    /// (which joins the worker thread) forever.
    fn write(&mut self, tag: u64, samples: &[i16]) -> Result<usize, SinkError>;

    /// Available space in samples (not bytes). This is advisory; the engine
    /// relies on `write()` for backpressure.
    fn available(&self) -> usize;

    /// Called when all queued sentences are done.
    fn on_drain_complete(&mut self);

    /// Called when a feed fails during normalization, tokenization, inference,
    /// or decoding. `tag` is the value passed to `feed` for the failed feed.
    /// The rest of that feed is aborted — no further audio for it follows.
    fn on_error(&mut self, tag: u64, error: String);
}
