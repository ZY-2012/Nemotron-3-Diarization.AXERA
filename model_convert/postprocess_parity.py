#!/usr/bin/env python3
"""Parity of the numpy post-processing port against NeMo's implementation."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import paths  # noqa: E402
from nemotron_sdk.postprocess import predlist_to_timestamps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting", nargs="?", default="EN2002a")
    parser.add_argument("--preds-subdir", default="preds")
    args = parser.parse_args()

    from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
    from nemo.collections.asr.parts.utils.vad_utils import load_postprocessing_from_yaml
    from nemo.collections.asr.parts.utils.vad_utils import predlist_to_timestamps as nemo_timestamps

    preds = np.load(paths.OUT / "baseline" / args.preds_subdir / f"{args.meeting}.npy")
    audio_rttm_map = {args.meeting: {"offset": 0.0}}
    cfg = load_postprocessing_from_yaml(None)

    nemo_out = nemo_timestamps(
        batch_preds_list=[torch.from_numpy(preds)],
        audio_rttm_map_dict=audio_rttm_map,
        cfg_vad_params=cfg,
        unit_10ms_frame_count=paths.OUTPUT_SUBSAMPLING_FACTOR,
        bypass_postprocessing=False,
    )[0]
    ours = predlist_to_timestamps(preds)

    max_diff = 0.0
    for spk in range(preds.shape[1]):
        a = np.asarray(nemo_out[spk], dtype=np.float64).reshape(-1, 2)
        b = np.asarray(ours[spk], dtype=np.float64).reshape(-1, 2)
        if a.shape != b.shape:
            print(f"spk {spk}: shape mismatch {a.shape} vs {b.shape}")
            continue
        if a.size:
            max_diff = max(max_diff, float(np.abs(a - b).max()))
    print(f"{args.meeting}: segment timestamp maxdiff vs NeMo = {max_diff:.3e}")

    nemo_lines = generate_diarization_output_lines(nemo_out, preds.shape[1])
    from nemotron_sdk.postprocess import generate_diarization_output_lines as our_lines

    our_lines_out = our_lines(ours, preds.shape[1])
    print(f"nemo lines {len(nemo_lines)} ours {len(our_lines_out)}")
    if nemo_lines:
        print("first nemo:", nemo_lines[0])
        print("first ours:", our_lines_out[0] if our_lines_out else None)
    same = set(nemo_lines) == set(our_lines_out)
    print("line sets equal:", same)
    if not same:
        only_nemo = sorted(set(nemo_lines) - set(our_lines_out))[:3]
        only_ours = sorted(set(our_lines_out) - set(nemo_lines))[:3]
        print("only nemo:", only_nemo)
        print("only ours:", only_ours)


if __name__ == "__main__":
    main()
