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

# Copy libc++_shared.so from NDK (needed by ONNX Runtime)
LIBCXX="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ ! -f "$LIBCXX" ]; then
    # Try linux host path
    LIBCXX="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
fi
cp "$LIBCXX" "$SCRIPT_DIR/app/src/main/jniLibs/arm64-v8a/"

echo "Done! Output: app/src/main/jniLibs/arm64-v8a/libsoprano_ffi.so"
