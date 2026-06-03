use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::Parser;
use hound::{WavSpec, WavWriter};
use soprano_core::{
    chunk_normalized, normalize, AudioSink, ExecutionProvider, SinkError, SopranoConfig, SopranoTTS,
};

#[derive(Parser)]
#[command(
    name = "soprano-tts",
    about = "Convert text to speech using Soprano TTS"
)]
struct Cli {
    /// Text to synthesize
    text: Option<String>,

    /// Read text to synthesize from a file
    #[arg(long, conflicts_with = "text")]
    input_file: Option<PathBuf>,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Path to model directory (containing backbone, decoder, tokenizer)
    #[arg(short, long)]
    model: PathBuf,

    /// Sampling temperature (0.0 = greedy)
    #[arg(long, default_value = "0.0")]
    temperature: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value = "0")]
    top_k: usize,

    /// Top-p nucleus sampling
    #[arg(long, default_value = "0.95")]
    top_p: f32,

    /// Repetition penalty
    #[arg(long, default_value = "1.2")]
    repetition_penalty: f32,

    /// Print normalized text and sentence-style chunks before synthesis
    #[arg(long)]
    print_chunks: bool,
}

struct WavSink {
    samples: Arc<Mutex<Vec<i16>>>,
}

impl WavSink {
    fn new() -> Self {
        Self {
            samples: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn samples(&self) -> Arc<Mutex<Vec<i16>>> {
        self.samples.clone()
    }
}

impl AudioSink for WavSink {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        self.samples.lock().unwrap().extend_from_slice(samples);
        Ok(samples.len())
    }

    fn available(&self) -> usize {
        usize::MAX
    }

    fn on_sentence_complete(&mut self, _sentence_index: usize) {}
    fn on_drain_complete(&mut self) {}
    fn on_error(&mut self, error: String) {
        eprintln!("Error: {}", error);
    }
}

fn main() {
    let cli = Cli::parse();
    let text = match (&cli.text, &cli.input_file) {
        (Some(text), None) => text.clone(),
        (None, Some(path)) => fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Failed to read input file {:?}: {}", path, e);
            std::process::exit(1);
        }),
        (Some(_), Some(_)) => {
            eprintln!("Provide either inline text or --input-file, not both.");
            std::process::exit(2);
        }
        (None, None) => {
            eprintln!("Provide text or --input-file.");
            std::process::exit(2);
        }
    };

    if cli.print_chunks {
        let normalized = normalize(&text);
        eprintln!("Normalized text:");
        eprintln!("{}", normalized);

        let chunks = chunk_normalized(&normalized);
        eprintln!("Sentence-style chunks ({}):", chunks.len());
        for (idx, chunk) in chunks.iter().enumerate() {
            eprintln!("  {}. {}", idx + 1, chunk);
        }
    }

    let sink = WavSink::new();
    let samples_ref = sink.samples();

    let config = SopranoConfig {
        model_path: cli.model.to_string_lossy().to_string(),
        temperature: cli.temperature,
        top_k: cli.top_k,
        top_p: cli.top_p,
        repetition_penalty: cli.repetition_penalty,
        execution_provider: ExecutionProvider::Cpu,
    };

    eprintln!("Loading models from {:?}...", cli.model);
    let t0 = Instant::now();
    let engine = SopranoTTS::new(config, Box::new(sink)).unwrap_or_else(|e| {
        eprintln!("Failed to load models: {}", e);
        std::process::exit(1);
    });
    eprintln!("Models loaded in {:.2}s", t0.elapsed().as_secs_f64());

    eprintln!("Synthesizing: \"{}\"", text);
    let t0 = Instant::now();
    engine.feed(&text).unwrap_or_else(|e| {
        eprintln!("Feed failed: {}", e);
        std::process::exit(1);
    });
    engine.drain();
    let elapsed = t0.elapsed().as_secs_f64();

    let samples = samples_ref.lock().unwrap();
    let duration = samples.len() as f64 / 32000.0;
    eprintln!(
        "Generated {} samples ({:.2}s audio) in {:.2}s (RTF: {:.1}x)",
        samples.len(),
        duration,
        elapsed,
        duration / elapsed
    );

    if samples.is_empty() {
        eprintln!("No audio generated.");
        std::process::exit(1);
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate: 32000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(&cli.output, spec).unwrap_or_else(|e| {
        eprintln!("Failed to create WAV file: {}", e);
        std::process::exit(1);
    });
    for &s in samples.iter() {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
    eprintln!("Saved to {:?}", cli.output);
}
