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
