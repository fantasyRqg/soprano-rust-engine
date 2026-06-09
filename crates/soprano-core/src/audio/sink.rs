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
    /// Write PCM i16 samples into the sink.
    /// MUST block if buffer is full (provides backpressure) or return an error.
    /// Returning `Ok(0)` for non-empty input is treated as a write failure.
    ///
    /// A blocked `write` MUST eventually return (e.g. by returning
    /// `Err(SinkError::Closed)` when playback stops for good). Cancellation
    /// and engine shutdown are only checked between calls, so a `write` that
    /// never returns hangs `flush` semantics and blocks `SopranoTTS`'s `Drop`
    /// (which joins the worker thread) forever.
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError>;

    /// Available space in samples (not bytes). This is advisory; the engine
    /// relies on `write()` for backpressure.
    fn available(&self) -> usize;

    /// Called once per `feed`, just before that feed's audio begins streaming
    /// to the sink. `tag` is the app-defined value passed to `feed`.
    /// `sample_offset` is the cumulative i16 sample count (mono, 32kHz) written
    /// to this sink since the start of the current stream (resets on flush) —
    /// i.e. the sample index at which this feed's audio begins.
    fn on_sentence_start(&mut self, tag: u64, sample_offset: u64);

    /// Called when all queued sentences are done.
    fn on_drain_complete(&mut self);

    /// Called when a feed fails during normalization, tokenization, inference,
    /// or decoding. `tag` is the value passed to `feed` for the failed feed.
    /// The rest of that feed is aborted — no further audio for it follows.
    fn on_error(&mut self, tag: u64, error: String);
}
