#!/usr/bin/env python3
"""Generate Pulsar2 calibration samples from real meeting streaming states.

Runs the FP32 ONNX streaming loop over several long meetings and dumps the
graph inputs sampled evenly across each recording (see ``host_stream.py``).
State tensors are cumulative, so sampling the whole recording matters.

    python generate_calibration.py [--samples 16] [--meetings EN2002c TS3003c]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import paths

DEFAULT_MEETINGS = ["EN2002c", "TS3003c", "ES2004b", "IS1009b", "EN2002d"]


def cuda_library_path() -> str:
    import site

    dirs = []
    for site_dir in site.getsitepackages():
        nvidia = Path(site_dir) / "nvidia"
        if nvidia.is_dir():
            dirs.extend(str(lib) for lib in nvidia.glob("*/lib"))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return ":".join(dirs + ([existing] if existing else []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meetings", nargs="*", default=DEFAULT_MEETINGS)
    parser.add_argument("--samples", type=int, default=16, help="samples per meeting and tensor")
    parser.add_argument("--provider", default="cuda")
    parser.add_argument("--mel-frames", type=int, default=0)
    parser.add_argument("--preset", choices=sorted(paths.STREAMING_PRESETS), default=None)
    parser.add_argument("--fifo-len", type=int, default=None)
    parser.add_argument("--spkcache-update-period", type=int, default=None)
    parser.add_argument("--preencode", default=str(paths.EXPORT_DIR / "preencode.onnx"))
    parser.add_argument("--encoder", default=str(paths.EXPORT_DIR / "encoder.onnx"))
    args = parser.parse_args()

    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = cuda_library_path()
    for name in ("chunk", "seq", "total_lengths"):
        directory = paths.CALIB_DIR / name
        if directory.exists():
            for old in directory.glob("*.npy"):
                old.unlink()
    command = [
        sys.executable,
        str(Path(__file__).parent / "host_stream.py"),
        *args.meetings,
        "--provider",
        args.provider,
        "--calib-dir",
        str(paths.CALIB_DIR),
        "--calib-samples",
        str(args.samples),
        "--preencode",
        args.preencode,
        "--encoder",
        args.encoder,
    ]
    if args.preset:
        command += ["--preset", args.preset]
    if args.fifo_len is not None:
        command += ["--fifo-len", str(args.fifo_len)]
    if args.spkcache_update_period is not None:
        command += ["--spkcache-update-period", str(args.spkcache_update_period)]
    if args.mel_frames:
        command += ["--mel-frames", str(args.mel_frames)]
    subprocess.run(command, check=True, env=environment, cwd=str(Path(__file__).parent))

    for name in ("chunk", "seq", "total_lengths"):
        count = len(list((paths.CALIB_DIR / name).glob("*.npy")))
        print(f"{name}: {count} samples")


if __name__ == "__main__":
    main()
