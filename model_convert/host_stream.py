#!/usr/bin/env python3
"""Host-side FP32 streaming loop: split ONNX graphs + numpy streaming state machine.

Consumes the mel features dumped by ``baseline_nemo.py`` and reproduces NeMo's
``forward_streaming`` chunk loop (host-side packing), then compares the per-frame
predictions with the PyTorch golden reference.

    python host_stream.py ES2004a IS1009a
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import paths
from baseline_nemo import apply_preset
from nemotron_sdk.diarize import OnnxGraphPair, StreamingDiarizer, load_learnable_sil_emb
from nemotron_sdk.state import SortformerConfig


def float_to_config(args) -> SortformerConfig:
    config = SortformerConfig(
        num_speakers=paths.N_SPK,
        fc_d_model=paths.FC_D_MODEL,
        subsampling_factor=paths.SUBSAMPLING,
        upsample_factor=paths.UPSAMPLE_FACTOR,
        chunk_len=args.chunk_len,
        chunk_left_context=args.chunk_left_context,
        chunk_right_context=args.chunk_right_context,
        fifo_len=args.fifo_len,
        spkcache_len=args.spkcache_len,
        spkcache_update_period=args.spkcache_update_period,
    )
    sil_emb = Path(args.sil_emb)
    if not sil_emb.exists():
        raise FileNotFoundError(
            f"learnable silence embedding not found: {sil_emb} (run dump_assets.py; the AOSC compression "
            "requires the learnable silence embedding)"
        )
    load_learnable_sil_emb(config, sil_emb)
    return config


def meeting_ids(args) -> list:
    if args.meetings:
        return args.meetings
    feats_dir = paths.OUT / "baseline" / "feats"
    return sorted(path.stem for path in feats_dir.glob("*.npy"))


def run_features(graphs, features: np.ndarray, cfg: SortformerConfig, calib_dir: Path = None, calib_samples: int = 64, calib_tag: str = ""):
    diarizer = StreamingDiarizer(graphs, cfg)
    num_chunks = int(np.ceil(features.shape[0] / (cfg.chunk_len * cfg.subsampling_factor)))
    sample_steps = set()
    if calib_dir is not None and num_chunks and calib_samples > 0:
        sample_steps = set(np.linspace(0, num_chunks - 1, min(calib_samples, num_chunks)).round().astype(int).tolist())
        for name in ("chunk", "seq", "total_lengths"):
            (calib_dir / name).mkdir(parents=True, exist_ok=True)

    def on_step(step, chunk_input, seq, total_lengths):
        if calib_dir is not None and step in sample_steps:
            np.save(calib_dir / "chunk" / f"{calib_tag}{step:06d}.npy", chunk_input)
            np.save(calib_dir / "seq" / f"{calib_tag}{step:06d}.npy", seq)
            np.save(calib_dir / "total_lengths" / f"{calib_tag}{step:06d}.npy", np.array([total_lengths], dtype=np.float32))

    start = time.perf_counter()
    preds = diarizer.process_features(features, on_step=on_step)
    elapsed = time.perf_counter() - start
    return preds, num_chunks, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meetings", nargs="*", default=None)
    parser.add_argument("--preencode", default=str(paths.EXPORT_DIR / "preencode.onnx"))
    parser.add_argument("--encoder", default=str(paths.EXPORT_DIR / "encoder.onnx"))
    parser.add_argument("--sil-emb", default=str(paths.EXPORT_DIR / "learnable_sil_emb.npy"))
    parser.add_argument("--preset", choices=sorted(paths.STREAMING_PRESETS), default=None)
    parser.add_argument("--chunk-len", type=int, default=None)
    parser.add_argument("--chunk-left-context", type=int, default=None)
    parser.add_argument("--chunk-right-context", type=int, default=None)
    parser.add_argument("--fifo-len", type=int, default=None)
    parser.add_argument("--spkcache-len", type=int, default=None)
    parser.add_argument("--spkcache-update-period", type=int, default=None)
    parser.add_argument("--mel-frames", type=int, default=0, help="only use the first N mel frames (quick parity)")
    parser.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda", "CPUExecutionProvider", "CUDAExecutionProvider"])
    parser.add_argument("--calib-dir", default=None, help="dump sampled graph inputs for quantization calibration")
    parser.add_argument("--calib-samples", type=int, default=64, help="number of samples per meeting")
    args = parser.parse_args()
    apply_preset(args)

    cfg = float_to_config(args)
    graphs = OnnxGraphPair(args.preencode, args.encoder, provider=args.provider)
    base_dir = paths.OUT / "baseline"
    out_dir = base_dir / "onnx_preds"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    calib_dir = Path(args.calib_dir) if args.calib_dir else None
    for mid in meeting_ids(args):
        features = np.load(base_dir / "feats" / f"{mid}.npy")
        if args.mel_frames:
            features = features[: args.mel_frames]
        preds, steps, elapsed = run_features(
            graphs,
            features,
            cfg,
            calib_dir=calib_dir,
            calib_samples=args.calib_samples,
            calib_tag=f"{mid}_",
        )
        np.save(out_dir / f"{mid}.npy", preds)

        golden = np.load(base_dir / "preds" / f"{mid}.npy")
        n = min(len(preds), len(golden))
        cosine = float(
            np.dot(preds[:n].reshape(-1), golden[:n].reshape(-1))
            / (np.linalg.norm(preds[:n]) * np.linalg.norm(golden[:n]) + 1e-12)
        )
        max_diff = float(np.max(np.abs(preds[:n] - golden[:n]))) if n else float("nan")
        duration = features.shape[0] / 100.0  # 10 ms hop
        print(
            f"{mid}: steps={steps:5d} frames={len(preds):6d}/{len(golden):6d} "
            f"cosine={cosine:.8f} maxdiff={max_diff:.3e} rtf={elapsed / duration:.4f}"
        )
        summary.append(
            {
                "meeting": mid,
                "steps": steps,
                "frames": int(len(preds)),
                "golden_frames": int(len(golden)),
                "cosine": cosine,
                "max_diff": max_diff,
                "rtf": elapsed / duration if duration else None,
            }
        )

    report_path = base_dir / "onnx_stream_report.json"
    with report_path.open("w") as f:
        json.dump({"config": vars(args), "runs": summary}, f, indent=2)
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
