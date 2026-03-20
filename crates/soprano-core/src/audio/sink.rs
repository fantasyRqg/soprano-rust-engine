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
    /// MUST block if buffer is full (provides backpressure).
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError>;

    /// Available space in samples (not bytes).
    fn available(&self) -> usize;

    /// Called when a sentence finishes.
    fn on_sentence_complete(&mut self, sentence_index: usize);

    /// Called when all queued sentences are done.
    fn on_drain_complete(&mut self);

    /// Called on error during inference.
    fn on_error(&mut self, error: String);
}
