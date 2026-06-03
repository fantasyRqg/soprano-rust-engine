#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    echo "Error: ANDROID_NDK_HOME is not set."
    echo "Set it to your NDK installation, e.g.:"
    echo "  export ANDROID_NDK_HOME=\$HOME/Library/Android/sdk/ndk/27.0.12077973"
    exit 1
fi

command -v cargo-ndk >/dev/null 2>&1 || {
    echo "Error: cargo-ndk not found. Install it with:"
    echo "  cargo install cargo-ndk"
    exit 1
}

echo "Building soprano-ffi for arm64-v8a..."
cargo ndk \
    -t arm64-v8a \
    -o "$SCRIPT_DIR/app/src/main/jniLibs" \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    build --release -p soprano-ffi --features soprano-ffi/nnapi,soprano-ffi/xnnpack

# --- Regenerate Kotlin bindings ---
# UniFFI bindings embed FFI checksums and must come from the same source as the
# .so above, or the app panics at load with a checksum mismatch. Generated on
# demand into the app sources; not checked in (see .gitignore).
echo "Generating Kotlin bindings..."
cargo run --release -q \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    -p soprano-ffi \
    --bin uniffi-bindgen \
    -- generate \
    --library "$REPO_ROOT/target/aarch64-linux-android/release/libsoprano_ffi.so" \
    --language kotlin \
    --out-dir "$SCRIPT_DIR/app/src/main/java"

# Copy libc++_shared.so from NDK (needed by ONNX Runtime)
LIBCXX="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ ! -f "$LIBCXX" ]; then
    # Try linux host path
    LIBCXX="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
fi
cp "$LIBCXX" "$SCRIPT_DIR/app/src/main/jniLibs/arm64-v8a/"

echo "Done! Outputs:"
echo "  Native lib: app/src/main/jniLibs/arm64-v8a/libsoprano_ffi.so"
echo "  Bindings:   app/src/main/java/uniffi/soprano_ffi/soprano_ffi.kt"
