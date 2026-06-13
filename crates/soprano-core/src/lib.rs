//! Core Soprano TTS engine.
//!
//! The primary API is `SopranoTTS`, `SopranoConfig`, `AudioSink`, and
//! `ExecutionProvider`. The lower-level modules are exposed for tests,
//! comparison tooling, and experiments, but the facade is the intended
//! application boundary.

pub mod audio;
pub mod inference;
pub(crate) mod profile;
pub mod text;
pub mod tts;

pub use audio::sink::{AudioSink, SinkError};
pub use inference::session::SAMPLE_RATE;
pub use text::chunker::chunk_normalized;
pub use text::normalizer::normalize;
pub use tts::{EstimateResult, ExecutionProvider, SopranoConfig, SopranoError, SopranoTTS};
