#!/bin/bash
# Quantize + compile the split streaming graphs for AX650/NPU3.
#
# Requires Docker and the Pulsar2 image. Run from model_convert/ after
# export_onnx.py and generate_calibration.py + pack_calib.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${PULSAR2_IMAGE:-docker-registry.aitsw.axera-tech.com/pulsar2:20260810-temp-09cadfa9}"

[ -f export/preencode.onnx ] || { echo "missing export/preencode.onnx (run export_onnx.py)"; exit 1; }
[ -f export/encoder.onnx ]   || { echo "missing export/encoder.onnx (run export_onnx.py)"; exit 1; }
[ -f calib_data/chunk.tar.gz ] || { echo "missing calib_data/*.tar.gz (run generate_calibration.py + pack_calib.sh)"; exit 1; }

for name in preencode encoder; do
    echo "=== pulsar2 build $name ==="
    docker run --rm -v "$ROOT":/workspace -w /workspace "$IMAGE" \
        pulsar2 build --config "/workspace/model_convert/pulsar2_${name}.json"
    echo "built: $ROOT/compile/${name}.axmodel"
done
