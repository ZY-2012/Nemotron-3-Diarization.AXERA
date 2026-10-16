#!/usr/bin/env python3
"""Board-side end-to-end check: axengine graphs + numpy SDK on a wav file.

    python board_check.py --preencode-axmodel preencode.axmodel \
        --encoder-axmodel encoder.axmodel --wav meeting.wav --out-dir out \
        --sil-emb learnable_sil_emb.npy --assets models
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nemotron_sdk import AxengineGraphPair, SortformerConfig, StreamingDiarizer, load_learnable_sil_emb  # noqa: E402


class TimedPair:
    def __init__(self, pair):
        self.pair = pair
        self.pre_s = 0.0
        self.enc_s = 0.0
        self.steps = 0

    def pre_encode(self, chunk):
        start = time.perf_counter()
        out = self.pair.pre_encode(chunk)
        self.pre_s += time.perf_counter() - start
        return out

    def encode(self, seq, total_lengths):
        start = time.perf_counter()
        out = self.pair.encode(seq, total_lengths)
        self.enc_s += time.perf_counter() - start
        self.steps += 1
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preencode-axmodel", required=True)
    parser.add_argument("--encoder-axmodel", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sil-emb", default=None)
    parser.add_argument("--assets", default=None, help="dir with feature_window.npy / mel_filterbank.npy")
    parser.add_argument("--chunk-len", type=int, default=340)
    parser.add_argument("--chunk-left-context", type=int, default=0)
    parser.add_argument("--chunk-right-context", type=int, default=40)
    parser.add_argument("--fifo-len", type=int, default=40)
    parser.add_argument("--spkcache-len", type=int, default=264)
    parser.add_argument("--spkcache-update-period", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = TimedPair(AxengineGraphPair(args.preencode_axmodel, args.encoder_axmodel))
    config = SortformerConfig(
        chunk_len=args.chunk_len,
        chunk_left_context=args.chunk_left_context,
        chunk_right_context=args.chunk_right_context,
        fifo_len=args.fifo_len,
        spkcache_len=args.spkcache_len,
        spkcache_update_period=args.spkcache_update_period,
    )
    if args.sil_emb:
        if not Path(args.sil_emb).exists():
            raise FileNotFoundError(f"learnable silence embedding not found: {args.sil_emb}")
        load_learnable_sil_emb(config, args.sil_emb)
    diarizer = StreamingDiarizer(graphs, config, feature_assets=args.assets)

    waveform, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    duration = waveform.shape[0] / sample_rate

    start = time.perf_counter()
    preds = diarizer.process_wav(waveform, sample_rate)
    elapsed = time.perf_counter() - start

    uri = Path(args.wav).stem
    np.save(out_dir / f"{uri}.npy", preds)
    lines = diarizer.rttm_lines(preds, uri)
    (out_dir / f"{uri}.rttm").write_text("\n".join(lines) + "\n")

    steps = graphs.steps
    report = {
        "wav": args.wav,
        "duration_s": duration,
        "frames": int(len(preds)),
        "steps": steps,
        "elapsed_s": elapsed,
        "rtf": elapsed / duration,
        "ms_per_step": 1000.0 * elapsed / max(steps, 1),
        "preencode_s": graphs.pre_s,
        "encoder_s": graphs.enc_s,
        "segments": len(lines),
    }
    (out_dir / f"{uri}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
