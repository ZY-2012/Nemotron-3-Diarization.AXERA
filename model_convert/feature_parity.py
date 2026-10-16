#!/usr/bin/env python3
"""Parity check for the numpy log-mel front-end against NeMo-dumped features."""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import paths  # noqa: E402
from nemotron_sdk.feature import load_feature_assets, log_mel_spectrogram, mel_filterbank  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting", nargs="?", default="EN2002a")
    args = parser.parse_args()

    wav_path = paths.AMI_AUDIO_DIR / f"{args.meeting}.Mix-Headset.wav"
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)

    reference = np.load(paths.OUT / "baseline" / "feats" / f"{args.meeting}.npy")
    window, mel_filter = load_feature_assets(paths.EXPORT_DIR)
    if mel_filter is None:
        mel_filter = mel_filterbank()
        print("warning: checkpoint mel filterbank not found, using recomputed librosa bank")
    if window is None:
        print("warning: checkpoint window not found, using recomputed hann window")
    ours = log_mel_spectrogram(wav, sr, mel_filter=mel_filter, window=window)

    n = min(len(reference), len(ours))
    diff = np.abs(reference[:n] - ours[:n])
    print(f"reference {reference.shape} ours {ours.shape} compared {n}")
    print(f"maxdiff {diff.max():.3e} mean {diff.mean():.3e}")
    cosine = float(
        np.dot(reference[:n].reshape(-1), ours[:n].reshape(-1))
        / (np.linalg.norm(reference[:n]) * np.linalg.norm(ours[:n]) + 1e-12)
    )
    print(f"cosine {cosine:.8f}")


if __name__ == "__main__":
    main()
