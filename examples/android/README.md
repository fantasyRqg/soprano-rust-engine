# Soprano TTS — Android Example

Minimal Android app demonstrating real-time text-to-speech using the Soprano engine via UniFFI Kotlin bindings.

## Prerequisites

- Rust toolchain with `aarch64-linux-android` target
- Android NDK (r25+) — install via Android Studio SDK Manager
- `cargo-ndk`
- Android Studio (for building/running the app)

```bash
rustup target add aarch64-linux-android
cargo install cargo-ndk
```

## Build

### 1. Build the native library

```bash
export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/<version>
./build-rust.sh
```

This cross-compiles `soprano-ffi` and places `libsoprano_ffi.so` into `app/src/main/jniLibs/arm64-v8a/`.

### 2. Build the Android app

Open `examples/android/` in Android Studio and build, or:

```bash
./gradlew assembleDebug
```

### 3. Push model files to device

```bash
adb push ../../models/ /sdcard/Android/data/com.example.soprano/files/soprano-models/
```

### 4. Install and run

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

In the app, the model path is pre-filled. Tap **Load Model**, enter text, and tap **Speak**.

## On-Device Benchmark

A standalone `soprano-bench` binary can be pushed to the device and run via `adb shell` to measure inference performance without app overhead.

### Build and push

```bash
export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/<version>
../../scripts/build-bench-android.sh
```

### Copy models (if not already accessible)

```bash
adb shell mkdir -p /data/local/tmp/soprano-models
for f in soprano_backbone_kv_f16.onnx soprano_decoder_f16.onnx tokenizer.json; do
  adb shell "run-as com.example.soprano cat files/soprano-models/$f" > /tmp/$f
  adb push /tmp/$f /data/local/tmp/soprano-models/
done
```

### Run

```bash
DIR=/data/local/tmp/soprano-bench
MODELS=/data/local/tmp/soprano-models
adb shell "LD_LIBRARY_PATH=$DIR $DIR/soprano-bench --model $MODELS --ep cpu -n 3"
```

### Results — Samsung Galaxy S24 (SM-S931U, Android 16)

Input: *"Hello, this is a benchmark of the Soprano text to speech engine running on device."*

| EP | RTF (avg) | First byte | Synth time | Audio |
|----|-----------|------------|------------|-------|
| CPU | 4.34x | ~700ms | ~870ms | 3.78s |
| NNAPI | 4.44x | ~680ms | ~850ms | 3.78s |
| XNNPACK | 4.22x | ~715ms | ~894ms | 3.78s |

- **RTF** = real-time factor (audio duration / synthesis time). Higher is faster.
- Model load: ~730ms per iteration.
- All three EPs show similar performance because the prebuilt ORT binary does not include NNAPI/XNNPACK support — both fall back to CPU. Building ORT from source with EP support (`ort-sys/compile`) is required for actual hardware acceleration.

## Limitations

- **arm64 only** — ONNX Runtime does not provide x86_64-android prebuilts, so x86_64 emulators are not supported. Use a physical arm64 device or an arm64 emulator image.
- Models must be on the device filesystem (not bundled in the APK).

## Architecture

```
AudioTrackSink  — implements FfiAudioSink, feeds PCM to Android AudioTrack
TtsViewModel    — manages engine lifecycle, runs synthesis on background thread
MainActivity    — Jetpack Compose UI with text input and playback controls
```

Audio flows from the Rust inference thread → JNA callback → `AudioTrack.write()` (blocking when buffer full = natural backpressure).
