#!/usr/bin/env python3
"""Build a NeMo diarization manifest for the AMI Mix-Headset evaluation set.

Reuses the local AMI eval assets (16 kHz Mix-Headset audio + reconstructed
NXT reference RTTMs) that the LS-EEND / pyannote / Sortformer AXERA projects
evaluate on.

    python make_ami_manifest.py            # all complete meetings
    python make_ami_manifest.py ES2004a IS1009a
"""

import json
import sys
from pathlib import Path

import soundfile as sf

import paths

# AMI dev/test meetings used by the AXERA benchmarks.
MEETINGS = [
    "EN2002a", "EN2002b", "EN2002c", "EN2002d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a", "IS1009b", "IS1009c", "IS1009d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d",
]


def ref_span(rttm: Path) -> float:
    span = 0.0
    for line in rttm.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        span = max(span, float(fields[3]) + float(fields[4]))
    return span


def main() -> None:
    wanted = set(sys.argv[1:]) or set(MEETINGS)
    records = []
    for mid in MEETINGS:
        if mid not in wanted:
            continue
        wav = paths.AMI_AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        ref = paths.AMI_REF_DIR / f"{mid}.rttm"
        if not wav.exists() or not ref.exists():
            print(f"skip {mid}: missing audio or reference")
            continue
        duration = float(sf.info(str(wav)).duration)
        span = ref_span(ref)
        if duration < span * 0.98:
            print(f"skip {mid}: truncated download ({duration:.1f}s < {span:.1f}s)")
            continue
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

    paths.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with paths.MANIFEST.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} meetings -> {paths.MANIFEST}")


if __name__ == "__main__":
    main()
