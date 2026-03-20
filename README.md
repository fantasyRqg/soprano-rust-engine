# Soprano Rust Engine

A cross-platform real-time text-to-speech engine written in Rust, wrapping the Soprano neural TTS model via ONNX Runtime. Ships with native Android (Kotlin) and iOS (Swift) example apps demonstrating streaming audio synthesis on-device.

## Features

- **Real-time streaming synthesis** — audio plays back as tokens are generated, no waiting for full output
- **Cross-platform** — single Rust core with auto-generated bindings for Android and iOS via UniFFI
- **Backpressure-aware audio pipeline** — worker thread streams PCM to platform audio APIs without unbounded memory growth
- **Comprehensive text normalization** — numbers, dates, currencies, abbreviations, and special characters handled automatically
- **Hallucination detection** — stops generation when hidden states converge, preventing runaway output
- **Multiple execution providers** — CPU (default), NNAPI (Android), XNNPACK (optimized CPU)

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Application                    │
│  (SwiftUI / Jetpack Compose / CLI)              │
├─────────────────────────────────────────────────┤
│              soprano-ffi (UniFFI)               │
│  Auto-generated Kotlin / Swift bindings         │
├─────────────────────────────────────────────────┤
│              soprano-core (Rust)                │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Normalizer│→ │Tokenizer │→ │  Backbone    │  │
│  │ (text)    │  │(HF BPE)  │  │ (autoregress)│  │
│  └───────────┘  └──────────┘  └──────┬───────┘  │
│                                      ↓          │
│                               ┌──────────────┐  │
│                               │   Decoder    │  │
│                               │ (Vocos, PCM) │  │
│                               └──────┬───────┘  │
│                                      ↓          │
│                               ┌───────────────┐ │
│                               │  AudioSink    │ │
│                               │ (app-provided)│ │
│                               └───────────────┘ │
└─────────────────────────────────────────────────┘
```

## Workspace Crates

| Crate | Description |
|-------|-------------|
| `soprano-core` | Core TTS library — inference pipeline, text normalization, audio conversion |
| `soprano-ffi` | UniFFI bindings generating Kotlin and Swift interfaces |
| `soprano-cli` | Command-line tool for offline WAV synthesis |
| `soprano-bench` | On-device benchmarking binary |

## Audio Output

| Property | Value |
|----------|-------|
| Sample rate | 32,000 Hz |
| Format | PCM 16-bit signed (i16), little-endian |
| Channels | Mono |
| Samples per token | 2,048 |
| Max tokens | 512 |
| Buffer alignment | 128 KB (cross-platform) |

## Quick Start

### Prerequisites

- Rust toolchain (stable)
- ONNX model files placed in `models/`:
  - `soprano_backbone_kv_f16.onnx` (or `soprano_backbone_kv.onnx`)
  - `soprano_decoder_f16.onnx` (or `soprano_decoder.onnx`)
  - `tokenizer.json`

### CLI Synthesis

```bash
cargo run -p soprano-cli -- "Hello, world!" --model models/ --output output.wav
```

Options:

```
--temperature <f32>          Sampling temperature (default: 0.0 = greedy)
--top-k <usize>              Top-k filtering (default: 0 = disabled)
--top-p <f32>                Nucleus sampling (default: 0.95)
--repetition-penalty <f32>   Repetition penalty (default: 1.2)
```

### Benchmarking

```bash
cargo run -p soprano-bench -- --model models/ --ep cpu -n 5
```

Outputs per-iteration metrics: model load time, first-byte latency, synthesis time, RTF (real-time factor).

## Android Example

The Android app uses Jetpack Compose with an `AudioTrack`-backed audio sink.

### Build

```bash
# Requires ANDROID_NDK_HOME and aarch64-linux-android target
cd examples/android
./build-rust.sh

# Open in Android Studio and build, or use Gradle
./gradlew assembleDebug
```

Push models to device:

```bash
adb shell mkdir -p /data/local/tmp/soprano-models
adb push models/*.onnx models/*.data models/tokenizer.json /data/local/tmp/soprano-models/
```

See [`examples/android/README.md`](examples/android/README.md) for full setup instructions.

## iOS Example

The iOS app uses SwiftUI (@Observable, iOS 17+) with `AVAudioEngine` for playback.

### Build

```bash
# Requires aarch64-apple-ios and aarch64-apple-ios-sim Rust targets
cd examples/ios
./build-rust.sh
```

Then:
1. Copy model files into `SopranoDemo/Models/`
2. Open `SopranoDemo/SopranoDemo.xcodeproj` in Xcode
3. Select a physical device (or simulator), build and run

See [`examples/ios/README.md`](examples/ios/README.md) for full setup instructions.

## Core API

```rust
use soprano_core::{SopranoTTS, SopranoConfig, AudioSink, ExecutionProvider};

let config = SopranoConfig {
    model_path: "models/".into(),
    temperature: 0.0,
    top_k: 0,
    top_p: 0.95,
    repetition_penalty: 1.2,
    execution_provider: ExecutionProvider::Cpu,
};

let mut tts = SopranoTTS::new(config, Box::new(my_sink))?;
tts.feed("Hello, world!")?;
tts.drain(); // blocks until synthesis completes
```

The `AudioSink` trait lets you provide your own audio output:

```rust
pub trait AudioSink: Send {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError>;
    fn available(&self) -> usize;
    fn on_sentence_complete(&mut self, sentence_index: usize);
    fn on_drain_complete(&mut self);
    fn on_error(&mut self, error: String);
}
```

## Tests

```bash
cargo test
```

Integration tests in `crates/soprano-core/tests/` require model files in `models/` and are skipped gracefully if not present.

## Performance

On-device benchmarks (Samsung Galaxy S24, CPU execution provider):

| Metric | Value |
|--------|-------|
| Model load | ~730 ms |
| First-byte latency | ~700 ms |
| Real-time factor | ~4.3x faster than real-time |

