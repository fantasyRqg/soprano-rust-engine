#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    echo "Error: ANDROID_NDK_HOME is not set."
    echo "  export ANDROID_NDK_HOME=\$HOME/Library/Android/sdk/ndk/<version>"
    exit 1
fi

command -v cargo-ndk >/dev/null 2>&1 || {
    echo "Error: cargo-ndk not found. Install with: cargo install cargo-ndk"
    exit 1
}

DEVICE_DIR="/data/local/tmp/soprano-bench"

echo "Building soprano-bench for arm64-v8a (release)..."
cargo ndk \
    -t arm64-v8a \
    -P 24 \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    build --release -p soprano-bench

BINARY="$REPO_ROOT/target/aarch64-linux-android/release/soprano-bench"
if [ ! -f "$BINARY" ]; then
    echo "Error: binary not found at $BINARY"
    exit 1
fi

# Copy libc++_shared.so from NDK (needed by ONNX Runtime)
LIBCXX="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ ! -f "$LIBCXX" ]; then
    LIBCXX="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
fi

echo "Pushing binary and libs to device at $DEVICE_DIR..."
adb shell "mkdir -p $DEVICE_DIR"
adb push "$BINARY" "$DEVICE_DIR/soprano-bench"
adb push "$LIBCXX" "$DEVICE_DIR/"
adb shell "chmod +x $DEVICE_DIR/soprano-bench"

echo ""
echo "Done! Run on device with:"
echo "  adb shell \"LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/soprano-bench --model /path/to/models --ep cpu -n 3\""
echo ""
echo "Example comparing all EPs:"
echo "  for ep in cpu nnapi xnnpack; do"
echo "    adb shell \"LD_LIBRARY_PATH=$DEVICE_DIR $DEVICE_DIR/soprano-bench --model /sdcard/soprano-models --ep \$ep -n 3\""
echo "  done"
