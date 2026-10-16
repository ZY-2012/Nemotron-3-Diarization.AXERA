#!/bin/bash
# Fetch the aarch64 cross compiler and the AX650N BSP SDK into cpp/toolchains/.
# Neither is committed to this repository.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p toolchains && cd toolchains

GCC_DIR=gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
GCC_TAR=$GCC_DIR.tar.xz
GCC_URL=https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/$GCC_TAR

if [ ! -d "$GCC_DIR" ]; then
    echo "==> downloading $GCC_TAR"
    [ -f "$GCC_TAR" ] || wget -q --show-progress "$GCC_URL"
    tar -xf "$GCC_TAR"
    echo "==> extracted $GCC_DIR"
fi

if [ ! -d ax650n_bsp_sdk ]; then
    echo "==> cloning AX650N BSP SDK"
    git clone --depth=1 https://github.com/AXERA-TECH/ax650n_bsp_sdk.git
fi

echo
echo "toolchain : $PWD/$GCC_DIR"
echo "BSP       : $PWD/ax650n_bsp_sdk/msp/out"
echo "now run   : bash build_ax650.sh  (或 TOOLCHAIN_ROOT/BSP_MSP_DIR 指到本地已有工具链)"
