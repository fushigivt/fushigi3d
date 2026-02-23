#!/usr/bin/env bash
# Download a free VRM model and create placeholder sprites so fushigi3d
# can run out of the box.
#
# Usage: ./scripts/setup.sh
#
# Model:
#   Constraint_Twist_Sample.vrm - by pixiv Inc.
#                    VRM 1.0, long brown hair, 12 spring chains
#                    VRM Public License 1.0
#
# You can also place your own VRM/GLB files in assets/default/models/.

set -euo pipefail

ASSET_DIR="assets/default"
MODEL_DIR="${ASSET_DIR}/models"
EXPR_DIR="${ASSET_DIR}/expressions"

mkdir -p "$MODEL_DIR" "$EXPR_DIR"

# ── Download VRM model ──
DEST="${MODEL_DIR}/Constraint_Twist_Sample.vrm"
URL="https://raw.githubusercontent.com/vrm-c/vrm-specification/master/samples/VRM1_Constraint_Twist_Sample/vrm/VRM1_Constraint_Twist_Sample.vrm"

if [ -f "$DEST" ]; then
    echo "Already exists: $DEST"
else
    echo "Downloading VRM1_Constraint_Twist_Sample (long hair, VRM 1.0)..."
    if command -v curl &>/dev/null; then
        curl -fSL -o "$DEST" "$URL"
    elif command -v wget &>/dev/null; then
        wget -q -O "$DEST" "$URL"
    else
        echo "Error: neither curl nor wget found" >&2
        exit 1
    fi
    echo "Downloaded: $DEST ($(du -h "$DEST" | cut -f1))"
fi

# ── Generate placeholder PNGTuber sprites ──
# Minimal 64x64 solid-color PNGs via Python (stdlib only, no pip deps).
# These are just placeholders for 2D mode — the 3D VRM mode doesn't need them.
generate_png() {
    local file="$1" r="$2" g="$3" b="$4"
    if [ -f "$file" ]; then
        return
    fi
    python3 -c "
import struct, zlib
W, H = 64, 64
def chunk(ctype, data):
    c = ctype + data
    return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
raw = b''
for _ in range(H):
    raw += b'\x00' + bytes([$r, $g, $b, 255]) * W
img = (b'\x89PNG\r\n\x1a\n'
    + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 6, 0, 0, 0))
    + chunk(b'IDAT', zlib.compress(raw))
    + chunk(b'IEND', b''))
open('$file', 'wb').write(img)
" 2>/dev/null && echo "Created: $file" || echo "Skipped: $file (python3 not available)"
}

generate_png "${ASSET_DIR}/idle.png"       100 140 200
generate_png "${ASSET_DIR}/speaking.png"   120 200 140
generate_png "${EXPR_DIR}/happy.png"       240 220 100
generate_png "${EXPR_DIR}/sad.png"         100 120 180
generate_png "${EXPR_DIR}/surprised.png"   240 180 100
generate_png "${EXPR_DIR}/angry.png"       220 100 100

echo ""
echo "Setup complete! Run with:"
echo "  cargo run --release"
echo ""
echo "For face tracking:"
echo "  pip install mediapipe opencv-python"
echo "  python3 scripts/mp_tracker.py"
