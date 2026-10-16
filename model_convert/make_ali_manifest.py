#!/usr/bin/env python3
"""Build a NeMo diarization manifest for the AliMeeting eval set."""

import json
import os
import sys
from pathlib import Path

import soundfile as sf

import paths

ALI_DIR = Path(os.environ.get("NEMOTRON_ALI_DIR", paths.ROOT.parent / "alimeeting"))
AUDIO_DIR = ALI_DIR / "audio_mono"
REF_DIR = ALI_DIR / "ref"
MEETINGS = ["R8001_M8004_MS801", "R8003_M8001_MS801", "R8007_M8010_MS803", "R8007_M8011_MS806"]
MANIFEST = paths.ROOT / "manifests" / "ali_eval.json"


def main() -> None:
    records = []
    wanted = set(sys.argv[1:]) or set(MEETINGS)
    for mid in MEETINGS:
        if mid not in wanted:
            continue
        wav = AUDIO_DIR / f"{mid}.wav"
        ref = REF_DIR / f"{mid}.rttm"
        if not wav.exists() or not ref.exists():
            print(f"skip {mid}")
            continue
        duration = float(sf.info(str(wav)).duration)
        records.append(
            {
                "audio_filepath": str(wav),
                "offset": 0.0,
                "duration": duration,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": str(ref),
                "uem_filepath": None,
            }
        )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} meetings -> {MANIFEST}")


if __name__ == "__main__":
    main()
