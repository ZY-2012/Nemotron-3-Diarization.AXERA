#!/usr/bin/env python3
"""Nemotron-3-Diarization streaming diarization on AX650N: 16 kHz wav -> RTTM.

    python3 example.py meeting.wav out.rttm
    python3 example.py meeting.wav out.rttm --chunk-len 6 --chunk-right-context 4 \
        --fifo-len 264 --spkcache-update-period 222   # 1.04 s low latency
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nemotron_sdk import AxengineGraphPair, SortformerConfig, StreamingDiarizer, load_learnable_sil_emb  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument("rttm", nargs="?", default=None)
    parser.add_argument("--preencode", default=str(MODELS / "preencode.axmodel"))
    parser.add_argument("--encoder", default=str(MODELS / "encoder.axmodel"))
    parser.add_argument("--sil-emb", default=str(MODELS / "learnable_sil_emb.npy"))
    parser.add_argument("--assets", default=str(MODELS), help="dir with feature_window.npy / mel_filterbank.npy")
    parser.add_argument("--chunk-len", type=int, default=340)
    parser.add_argument("--chunk-left-context", type=int, default=0)
    parser.add_argument("--chunk-right-context", type=int, default=40)
    parser.add_argument("--fifo-len", type=int, default=40)
    parser.add_argument("--spkcache-len", type=int, default=264)
    parser.add_argument("--spkcache-update-period", type=int, default=300)
    parser.add_argument("--threads", type=int, default=8, help="reserved (numpy front-end is single-process)")
    args = parser.parse_args()

    config = SortformerConfig(
        chunk_len=args.chunk_len,
        chunk_left_context=args.chunk_left_context,
        chunk_right_context=args.chunk_right_context,
        fifo_len=args.fifo_len,
        spkcache_len=args.spkcache_len,
        spkcache_update_period=args.spkcache_update_period,
    )
    if not Path(args.sil_emb).exists():
        raise FileNotFoundError(f"learnable silence embedding not found: {args.sil_emb}")
    load_learnable_sil_emb(config, args.sil_emb)

    waveform, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        raise SystemExit(f"expected 16 kHz wav, got {sample_rate}")

    diarizer = StreamingDiarizer(
        AxengineGraphPair(args.preencode, args.encoder), config, feature_assets=args.assets
    )
    start = time.perf_counter()
    preds = diarizer.process_wav(waveform, sample_rate)
    elapsed = time.perf_counter() - start
    duration = waveform.shape[0] / sample_rate

    uri = Path(args.wav).stem
    lines = diarizer.rttm_lines(preds, uri)
    if args.rttm:
        Path(args.rttm).write_text("\n".join(lines) + "\n")

    print(
        f"{uri}: {len(preds)} frames, {len(lines)} segments, "
        f"{elapsed:.1f}s / {duration:.1f}s audio (RTF {elapsed / duration:.4f})"
    )
    if args.rttm:
        print(f"rttm -> {args.rttm}")


if __name__ == "__main__":
    main()
