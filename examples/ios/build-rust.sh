#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FRAMEWORK_DIR="$SCRIPT_DIR/SopranoEngine/SopranoFFI.xcframework"
BINDINGS_OUT="$SCRIPT_DIR/SopranoEngine/Sources/SopranoEngine"

# --- Check prerequisites ---

for TARGET in aarch64-apple-ios aarch64-apple-ios-sim; do
    if ! rustup target list --installed | grep -q "$TARGET"; then
        echo "Error: $TARGET target not installed."
        echo "Install with: rustup target add $TARGET"
        exit 1
    fi
done

# ORT_LIB_LOCATION must point to a directory containing libonnxruntime.a and headers.
if [ -z "${ORT_LIB_LOCATION:-}" ]; then
    if [ -d "$SCRIPT_DIR/onnxruntime" ]; then
        export ORT_LIB_LOCATION="$SCRIPT_DIR/onnxruntime"
    else
        echo "Error: ORT_LIB_LOCATION not set and examples/ios/onnxruntime/ not found."
        echo ""
        echo "Provide a prebuilt ONNX Runtime iOS static library:"
        echo "  1. Download or build onnxruntime for iOS (aarch64, static)"
        echo "  2. Place at examples/ios/onnxruntime/ with structure:"
        echo "       onnxruntime/"
        echo "         lib/libonnxruntime.a"
        echo "         include/onnxruntime_c_api.h"
        echo "  3. Or set ORT_LIB_LOCATION=/path/to/onnxruntime"
        exit 1
    fi
fi

echo "ORT_LIB_LOCATION=$ORT_LIB_LOCATION"

# --- Build Rust static library for iOS (device + simulator) ---

# Match Xcode deployment target so C dependencies don't emit version mismatch warnings.
export IPHONEOS_DEPLOYMENT_TARGET=17.0

for TARGET in aarch64-apple-ios aarch64-apple-ios-sim; do
    echo "Building soprano-ffi for $TARGET..."
    cargo build \
        --release \
        --target "$TARGET" \
        --manifest-path "$REPO_ROOT/Cargo.toml" \
        -p soprano-ffi
done

DEVICE_LIB="$REPO_ROOT/target/aarch64-apple-ios/release/libsoprano_ffi.a"
SIM_LIB="$REPO_ROOT/target/aarch64-apple-ios-sim/release/libsoprano_ffi.a"
for LIB in "$DEVICE_LIB" "$SIM_LIB"; do
    if [ ! -f "$LIB" ]; then
        echo "Error: static lib not found at $LIB"
        exit 1
    fi
done

# --- Regenerate Swift bindings ---

echo "Generating Swift bindings..."
cargo run \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    -p soprano-ffi \
    --bin uniffi-bindgen \
    -- generate \
    --library "$DEVICE_LIB" \
    --language swift \
    --out-dir "$SCRIPT_DIR/.bindings-tmp"

cp "$SCRIPT_DIR/.bindings-tmp/soprano_ffi.swift" "$BINDINGS_OUT/soprano_ffi.swift"

# --- Assemble XCFramework ---

echo "Assembling XCFramework..."
rm -rf "$FRAMEWORK_DIR"

HEADER_DIR="$SCRIPT_DIR/.bindings-tmp/headers"
mkdir -p "$HEADER_DIR"
cp "$SCRIPT_DIR/.bindings-tmp/soprano_ffiFFI.h" "$HEADER_DIR/"
cp "$SCRIPT_DIR/.bindings-tmp/soprano_ffiFFI.modulemap" "$HEADER_DIR/module.modulemap"

xcodebuild -create-xcframework \
    -library "$DEVICE_LIB" \
    -headers "$HEADER_DIR" \
    -library "$SIM_LIB" \
    -headers "$HEADER_DIR" \
    -output "$FRAMEWORK_DIR"

# --- Cleanup ---
rm -rf "$SCRIPT_DIR/.bindings-tmp"

echo ""
echo "Done! Outputs:"
echo "  XCFramework: SopranoEngine/SopranoFFI.xcframework/"
echo "  Bindings:    SopranoEngine/Sources/SopranoEngine/soprano_ffi.swift"
