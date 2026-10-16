#!/bin/bash
# Pack the per-input calibration sample directories into the tar.gz files that
# pulsar2 expects (one archive per graph input).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CALIB_DIR="${NEMOTRON_CALIB_DIR:-calib_data}"
[ -d "$CALIB_DIR" ] || { echo "run generate_calibration.py first"; exit 1; }
rm -f "$CALIB_DIR"/*.tar.gz
for d in "$CALIB_DIR"/*/; do
    name=$(basename "$d")
    tar -czf "$CALIB_DIR/$name.tar.gz" -C "$d" . 2>/dev/null || true
done
echo "packed $(ls "$CALIB_DIR"/*.tar.gz | wc -l) calibration archives -> $CALIB_DIR"
