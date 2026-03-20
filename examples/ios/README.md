# Soprano iOS Demo

SwiftUI demo app for the Soprano TTS engine on iOS.

## Prerequisites

- Xcode 15+ with iOS 17 SDK
- Rust toolchain: `rustup target add aarch64-apple-ios aarch64-apple-ios-sim`

## Setup

### 1. Build Rust Library

```bash
cd examples/ios
./build-rust.sh
```

This cross-compiles `soprano-ffi` for iOS (device + simulator), regenerates Swift bindings, and assembles the XCFramework.

The `ort` crate automatically downloads a prebuilt ONNX Runtime during the Rust build (via its `download-binaries` feature). If auto-download fails or you need a custom ORT build, provide one manually:

```
examples/ios/onnxruntime/
  lib/libonnxruntime.a
  include/onnxruntime_c_api.h
```

Or set `ORT_LIB_LOCATION=/path/to/your/onnxruntime`.

### 2. Copy Models

Copy your Soprano ONNX models into the app bundle:

```bash
mkdir -p SopranoDemo/Models
cp ../../models/soprano_backbone_kv.onnx SopranoDemo/Models/
cp ../../models/soprano_decoder.onnx SopranoDemo/Models/
cp ../../models/soprano_decoder.onnx.data SopranoDemo/Models/
cp ../../models/tokenizer.json SopranoDemo/Models/
```

### 3. Build & Run

Open `SopranoDemo/SopranoDemo.xcodeproj` in Xcode, select your iOS device, and run.

## Architecture

```
SopranoEngine (Swift Package)
  +-- SopranoFFI.xcframework (Rust static lib + C headers)
  +-- soprano_ffi.swift (UniFFI generated bindings)

SopranoDemo (SwiftUI App)
  +-- ContentView -> TtsViewModel -> SopranoTts (FFI) -> AudioEngineSink (AVAudioEngine)
```

Audio flows: Rust inference -> PCM bytes -> FfiAudioSink callback -> AVAudioPCMBuffer -> AVAudioPlayerNode -> speaker.
