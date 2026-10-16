#!/usr/bin/env python3
"""DER scoring for Sortformer predictions, independent of NeMo.

Uses the numpy post-processing port and pyannote.metrics against the AMI
reference RTTMs (same protocol as the AXERA LS-EEND / pyannote benchmarks).

    python score_der.py --preds baseline/preds --collar 0.0
    python score_der.py --rttm-dir board/hyp --collar 0.0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import paths  # noqa: E402
from sortformer_sdk.postprocess import PostProcessingParams, predlist_to_timestamps, timestamps_to_rttm_lines  # noqa: E402


def load_ref(path: Path, uri: str):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        start, duration = float(fields[3]), float(fields[4])
        if duration > 0:
            annotation[Segment(start, start + duration)] = fields[7]
    return annotation


def load_hyp(path: Path, uri: str):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        start, duration = float(fields[3]), float(fields[4])
        if duration > 0:
            annotation[Segment(start, start + duration)] = fields[7]
    return annotation


def preds_to_annotation(preds: np.ndarray, uri: str, params: PostProcessingParams, bypass: bool):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    timestamps = predlist_to_timestamps(preds, params=params, bypass_postprocessing=bypass)
    for spk_idx, intervals in enumerate(timestamps):
        for start, end in intervals:
            if end > start:
                annotation[Segment(start, end)] = f"speaker_{spk_idx}"
    return annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meetings", nargs="*", default=None)
    parser.add_argument("--preds", default=str(paths.OUT / "baseline" / "preds"))
    parser.add_argument("--rttm-dir", default=None, help="score RTTM files instead of npy predictions")
    parser.add_argument("--rttm-pattern", default="{mid}.rttm", help="pattern under --rttm-dir")
    parser.add_argument("--ref-dir", default=None, help="reference RTTM directory (default: AMI)")
    parser.add_argument("--collar", type=float, default=0.0)
    parser.add_argument("--ignore-overlap", action="store_true")
    parser.add_argument("--bypass-postproc", action="store_true")
    parser.add_argument("--onset", type=float, default=None)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--pad-onset", type=float, default=None)
    parser.add_argument("--pad-offset", type=float, default=None)
    parser.add_argument("--min-duration-on", type=float, default=None)
    parser.add_argument("--min-duration-off", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from pyannote.metrics.diarization import DiarizationErrorRate

    ref_dir = Path(args.ref_dir) if args.ref_dir else paths.AMI_REF_DIR

    params = PostProcessingParams()
    for field in ("onset", "offset", "pad_onset", "pad_offset", "min_duration_on", "min_duration_off"):
        value = getattr(args, field)
        if value is not None:
            setattr(params, field, value)

    meetings = args.meetings
    if not meetings:
        if args.rttm_dir:
            meetings = sorted(path.stem for path in Path(args.rttm_dir).glob("*.rttm"))
        else:
            meetings = sorted(path.stem for path in Path(args.preds).glob("*.npy"))

    aggregate = {"num": 0.0, "den": 0.0, "miss": 0.0, "fa": 0.0, "conf": 0.0}
    report = {"collar": args.collar, "params": vars(params), "meetings": []}
    for mid in meetings:
        ref_path = ref_dir / f"{mid}.rttm"
        if not ref_path.exists():
            print(f"skip {mid}: no reference")
            continue
        ref = load_ref(ref_path, mid)
        if args.rttm_dir:
            hyp_path = Path(args.rttm_dir) / args.rttm_pattern.format(mid=mid)
            if not hyp_path.exists():
                print(f"skip {mid}: no hypothesis")
                continue
            hyp = load_hyp(hyp_path, mid)
        else:
            preds_path = Path(args.preds) / f"{mid}.npy"
            if not preds_path.exists():
                continue
            hyp = preds_to_annotation(np.load(preds_path), mid, params, args.bypass_postproc)

        metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.ignore_overlap)
        detail = metric(ref, hyp, detailed=True)
        total = detail["total"]
        record = {
            "meeting": mid,
            "total": total,
            "der": (detail["confusion"] + detail["false alarm"] + detail["missed detection"]) / total,
            "miss": detail["missed detection"] / total,
            "fa": detail["false alarm"] / total,
            "conf": detail["confusion"] / total,
        }
        report["meetings"].append(record)
        print(
            f"{mid:10s} DER {record['der']:.4%}  miss {record['miss']:.4%}  "
            f"fa {record['fa']:.4%}  conf {record['conf']:.4%}"
        )
        aggregate["den"] += total
        aggregate["num"] += total * record["der"]
        aggregate["miss"] += total * record["miss"]
        aggregate["fa"] += total * record["fa"]
        aggregate["conf"] += total * record["conf"]

    if aggregate["den"]:
        report["weighted"] = {
            "der": aggregate["num"] / aggregate["den"],
            "miss": aggregate["miss"] / aggregate["den"],
            "fa": aggregate["fa"] / aggregate["den"],
            "conf": aggregate["conf"] / aggregate["den"],
            "meetings": len(report["meetings"]),
        }
        weighted = report["weighted"]
        print(
            f"WEIGHTED ({weighted['meetings']}): DER {weighted['der']:.4%}  miss {weighted['miss']:.4%}  "
            f"fa {weighted['fa']:.4%}  conf {weighted['conf']:.4%}"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"report -> {args.output}")


if __name__ == "__main__":
    main()
