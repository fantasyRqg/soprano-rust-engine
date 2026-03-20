use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use soprano_core::{AudioSink, ExecutionProvider, SinkError, SopranoConfig, SopranoTTS, SAMPLE_RATE};

#[derive(Clone, ValueEnum)]
enum Ep {
    Cpu,
    Nnapi,
    Xnnpack,
}

#[derive(Parser)]
#[command(name = "soprano-bench", about = "Benchmark Soprano TTS inference on-device")]
struct Cli {
    /// Path to model directory
    #[arg(short, long)]
    model: String,

    /// Execution provider
    #[arg(long, default_value = "cpu")]
    ep: Ep,

    /// Number of iterations
    #[arg(short = 'n', long, default_value = "3")]
    iterations: usize,

    /// Text to synthesize
    #[arg(short, long, default_value = "Hello, this is a benchmark of the Soprano text to speech engine running on device.")]
    text: String,
}

struct CountingSink {
    sample_count: Arc<Mutex<usize>>,
    first_write: Arc<Mutex<Option<Instant>>>,
}

impl CountingSink {
    fn new() -> Self {
        Self {
            sample_count: Arc::new(Mutex::new(0)),
            first_write: Arc::new(Mutex::new(None)),
        }
    }
    fn sample_count(&self) -> Arc<Mutex<usize>> {
        self.sample_count.clone()
    }
    fn first_write(&self) -> Arc<Mutex<Option<Instant>>> {
        self.first_write.clone()
    }
}

impl AudioSink for CountingSink {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        let mut fw = self.first_write.lock().unwrap();
        if fw.is_none() {
            *fw = Some(Instant::now());
        }
        drop(fw);
        *self.sample_count.lock().unwrap() += samples.len();
        Ok(samples.len())
    }
    fn available(&self) -> usize { usize::MAX }
    fn on_sentence_complete(&mut self, _: usize) {}
    fn on_drain_complete(&mut self) {}
    fn on_error(&mut self, error: String) {
        eprintln!("  ERROR: {}", error);
    }
}

fn main() {
    let cli = Cli::parse();

    let ep = match cli.ep {
        Ep::Cpu => ExecutionProvider::Cpu,
        Ep::Nnapi => {
            #[cfg(not(feature = "nnapi"))]
            eprintln!("WARNING: nnapi feature not compiled in, will fall back to CPU");
            ExecutionProvider::Nnapi
        }
        Ep::Xnnpack => {
            #[cfg(not(feature = "xnnpack"))]
            eprintln!("WARNING: xnnpack feature not compiled in, will fall back to CPU");
            ExecutionProvider::Xnnpack
        }
    };

    eprintln!("=== soprano-bench ===");
    eprintln!("EP:         {:?}", ep);
    eprintln!("Model:      {}", cli.model);
    eprintln!("Iterations: {}", cli.iterations);
    eprintln!("Text:       \"{}\"", cli.text);
    eprintln!();

    let mut rtfs = Vec::new();

    for i in 0..cli.iterations {
        let config = SopranoConfig {
            model_path: cli.model.clone(),
            execution_provider: ep,
            ..Default::default()
        };

        // Load
        let t_load = Instant::now();
        let sink = CountingSink::new();
        let sample_count_ref = sink.sample_count();
        let first_write_ref = sink.first_write();

        let engine = match SopranoTTS::new(config, Box::new(sink)) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[iter {}] Load failed: {}", i + 1, e);
                continue;
            }
        };
        let load_ms = t_load.elapsed().as_millis();

        // Synthesize
        let t_synth = Instant::now();
        if let Err(e) = engine.feed(&cli.text) {
            eprintln!("[iter {}] Feed failed: {}", i + 1, e);
            continue;
        }
        engine.drain();
        let synth_ms = t_synth.elapsed().as_millis();

        // Drop engine before reading metrics to ensure worker thread is fully stopped
        drop(engine);

        let samples = *sample_count_ref.lock().unwrap();
        let audio_sec = samples as f64 / SAMPLE_RATE as f64;
        let synth_sec = synth_ms as f64 / 1000.0;
        let rtf = if synth_sec > 0.0 { audio_sec / synth_sec } else { 0.0 };

        let first_byte_ms = first_write_ref.lock().unwrap()
            .map(|t| t.duration_since(t_synth).as_millis())
            .unwrap_or(0);

        eprintln!(
            "[iter {}] load={}ms  first_byte={}ms  synth={}ms  audio={:.2}s  samples={}  RTF={:.2}x",
            i + 1, load_ms, first_byte_ms, synth_ms, audio_sec, samples, rtf
        );
        rtfs.push(rtf);
    }

    if !rtfs.is_empty() {
        let min = rtfs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = rtfs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg = rtfs.iter().sum::<f64>() / rtfs.len() as f64;
        eprintln!();
        eprintln!("RTF summary: min={:.2}x  max={:.2}x  avg={:.2}x  (n={})", min, max, avg, rtfs.len());
    }
}
