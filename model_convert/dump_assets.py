#!/usr/bin/env python3
"""Dump numpy assets of the NeMo model needed by the host / board SDK.

- learnable_sil_emb: AOSC speaker-cache compression silence embedding.
- feature_window + mel_filterbank: the exact front-end buffers stored in the
  .nemo checkpoint. The released checkpoint buffers are quantized (they differ
  from a freshly computed librosa hann/mel by ~1e-3), so the numpy front-end
  must reuse them to match NeMo inference.

    python dump_assets.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import paths
from baseline_nemo import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(paths.EXPORT_DIR / "learnable_sil_emb.npy"))
    args = parser.parse_args()

    model_path = args.model or (str(paths.LOCAL_NEMO) if paths.LOCAL_NEMO.exists() else paths.MODEL)
    device = torch.device(args.device)
    model = load_model(model_path, device)

    modules = model.sortformer_modules
    print(f"use_learnable_sil_emb={modules.use_learnable_sil_emb}")
    emb = modules.learnable_sil_emb.detach().cpu().numpy().astype(np.float32).reshape(-1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    print(f"learnable_sil_emb: shape={emb.shape} -> {out}")

    featurizer = model.preprocessor.featurizer
    window = featurizer.window.detach().cpu().numpy().astype(np.float32)
    fb = featurizer.fb[0].detach().cpu().numpy().astype(np.float32)
    np.save(out.parent / "feature_window.npy", window)
    np.save(out.parent / "mel_filterbank.npy", fb)
    print(f"feature_window: shape={window.shape}, mel_filterbank: shape={fb.shape} -> {out.parent}")

    meta = {
        "num_spks": modules.n_spk,
        "fc_d_model": modules.fc_d_model,
        "tf_d_model": modules.tf_d_model,
        "upsample_factor": model.upsample_factor,
        "output_subsampling_factor": model.output_subsampling_factor,
        "high_resolution": model.high_resolution,
        "spkcache_sil_frames_per_spk": modules.spkcache_sil_frames_per_spk,
        "use_learnable_sil_emb": bool(modules.use_learnable_sil_emb),
    }
    print(meta)


if __name__ == "__main__":
    main()
