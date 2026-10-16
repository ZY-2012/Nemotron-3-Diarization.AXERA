#!/bin/bash
# Cross-compile nemotron_ax650 for AX650N (aarch64).
#
# Override the two paths if your toolchain lives elsewhere:
#   TOOLCHAIN_ROOT=/path/to/gcc-arm-...-aarch64-none-linux-gnu \
#   BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out \
#   bash build_ax650.sh
set -euo pipefail
cd "$(dirname "$0")"

TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-$PWD/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu}"
BSP_MSP_DIR="${BSP_MSP_DIR:-$PWD/toolchains/ax650n_bsp_sdk/msp/out}"

CC="$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-gcc"
CXX="$TOOLCHAIN_ROOT/bin/aarch64-none-linux-gnu-g++"

for f in "$CC" "$CXX"; do
    [ -x "$f" ] || { echo "missing cross compiler: $f"; echo "run: bash download_toolchains.sh (or set TOOLCHAIN_ROOT)"; exit 1; }
done
[ -d "$BSP_MSP_DIR/include" ] || { echo "missing BSP: $BSP_MSP_DIR"; echo "set BSP_MSP_DIR"; exit 1; }

# Pass the compilers directly instead of a toolchain file: CMake's try-compile
# then picks up the cross gcc for the C target too (a host gcc produces
# "Relocations in generic ELF (EM: 62)" when linking the kissfft archive).
cmake -S . -B build_ax650 \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DBSP_MSP_DIR="$BSP_MSP_DIR" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build_ax650 -j"$(nproc)"

mkdir -p bin && cp build_ax650/nemotron_ax650 build_ax650/bench_encoder bin/
echo "built: $PWD/bin/nemotron_ax650 $PWD/bin/bench_encoder"
