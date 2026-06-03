pub mod audio;
pub mod inference;
pub mod text;
pub mod tts;

pub use audio::sink::{AudioSink, SinkError};
pub use inference::session::SAMPLE_RATE;
pub use text::chunker::chunk_normalized;
pub use text::normalizer::normalize;
pub use tts::{EstimateResult, ExecutionProvider, SopranoConfig, SopranoError, SopranoTTS};
