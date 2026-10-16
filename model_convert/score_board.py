#!/usr/bin/env python3
"""Aggregate golden / host-ONNX / AX650-board diarization results.

    python score_board.py ES2004a IS1009a EN2002d TS3003b EN2002a
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import paths  # noqa: E402
from nemotron_sdk.postprocess import predlist_to_timestamps  # noqa: E402


def load_rttm(path: Path, label_field: int = 7):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=path.stem)
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        start, duration = float(fields[3]), float(fields[4])
        if duration > 0:
            annotation[Segment(start, start + duration)] = fields[label_field]
    return annotation


def preds_to_annotation(preds: np.ndarray):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri="hyp")
    for spk, intervals in enumerate(predlist_to_timestamps(preds)):
        for start, end in intervals:
            if end > start:
                annotation[Segment(start, end)] = f"speaker_{spk}"
    return annotation


def score(ref, hyp, collar: float = 0.0):
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=collar)
    detail = metric(ref, hyp, detailed=True)
    total = detail["total"]
    return {
        "der": (detail["confusion"] + detail["false alarm"] + detail["missed detection"]) / total,
        "miss": detail["missed detection"] / total,
        "fa": detail["false alarm"] / total,
        "conf": detail["confusion"] / total,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a = a[:n].reshape(-1)
    b = b[:n].reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def permutation_invariant_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Best-cosine speaker-permutation match via Hungarian assignment (8! brute force is too slow)."""
    from scipy.optimize import linear_sum_assignment

    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    cross = a.T @ b  # (n_spk_a, n_spk_b); maximizing the assignment maximizes the flattened cosine
    rows, cols = linear_sum_assignment(-cross)
    x = a[:, cols[rows]].reshape(-1)
    y = b.reshape(-1)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meetings", nargs="*", default=["ES2004a", "IS1009a", "EN2002d", "TS3003b", "EN2002a"])
    parser.add_argument("--board-dir", default=str(paths.OUT / "out"))
    parser.add_argument("--ref-dir", default=None, help="reference RTTM directory (default: AMI)")
    parser.add_argument("--board-rttm-pattern", default="{dir}/{mid}/{mid}.rttm")
    parser.add_argument("--output", default=str(paths.OUT / "board_report.json"))
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir) if args.ref_dir else paths.AMI_REF_DIR

    rows = []
    for mid in args.meetings:
        ref = load_rttm(ref_dir / f"{mid}.rttm")
        golden_path = paths.OUT / "baseline" / "preds" / f"{mid}.npy"
        onnx_path = paths.OUT / "baseline" / "onnx_preds" / f"{mid}.npy"
        golden = np.load(golden_path) if golden_path.exists() else None
        onnx = np.load(onnx_path) if onnx_path.exists() else None
        board_candidates = [
            Path(args.board_rttm_pattern.format(dir=args.board_dir, mid=mid)),
        ]
        board_rttm = next((path for path in board_candidates if path.exists()), None)
        if board_rttm is None:
            # board_check names outputs after the full wav stem (e.g. ES2004a.Mix-Headset)
            globbed = sorted(Path(args.board_dir).glob(f"{mid}/*{mid}*.rttm"))
            board_rttm = globbed[0] if globbed else None
        if board_rttm is None:
            print(f"skip {mid}: no board rttm")
            continue
        board = load_rttm(board_rttm)

        record = {"meeting": mid, "board": score(ref, board)}
        if golden is not None:
            record["golden"] = score(ref, preds_to_annotation(golden))
        if onnx is not None:
            record["onnx"] = score(ref, preds_to_annotation(onnx))
        board_preds_path = board_rttm.with_suffix(".npy")
        if not board_preds_path.exists():
            board_preds_path = board_rttm.with_suffix(".bin")
        if board_preds_path.exists() and golden is not None:
            board_preds = (
                np.load(board_preds_path)
                if board_preds_path.suffix == ".npy"
                else np.fromfile(board_preds_path, dtype=np.float32).reshape(-1, paths.N_SPK)
            )
            record["board_vs_golden_cosine"] = cosine(board_preds, golden)
            record["board_vs_golden_cosine_perm_invariant"] = permutation_invariant_cosine(board_preds, golden)
            if onnx is not None:
                record["board_vs_onnx_cosine"] = cosine(board_preds, onnx)
                record["board_vs_onnx_cosine_perm_invariant"] = permutation_invariant_cosine(board_preds, onnx)
        rows.append(record)
        parts = [f"{mid:22s}"]
        for key in ("golden", "onnx", "board"):
            if key in record:
                parts.append(f"{key} {record[key]['der']:.4%}")
        print("  ".join(parts))

    report = {"meetings": rows}
    if rows:
        report["summary"] = {}
        for key in ("golden", "onnx", "board"):
            if not all(key in row for row in rows):
                continue
            report["summary"][key] = {
                metric: float(np.mean([row[key][metric] for row in rows])) for metric in ("der", "miss", "fa", "conf")
            }
        print("mean over meetings:")
        for key, value in report["summary"].items():
            print(f"  {key:7s} DER {value['der']:.4%}  miss {value['miss']:.4%}  fa {value['fa']:.4%}  conf {value['conf']:.4%}")

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"report -> {args.output}")


if __name__ == "__main__":
    main()
