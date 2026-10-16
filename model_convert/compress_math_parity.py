#!/usr/bin/env python3
"""C++ vs numpy _compress_spkcache parity on the first compression inputs.

    python compress_math_parity.py
"""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import paths  # noqa: E402
import nemotron_sdk.state as nps  # noqa: E402
from nemotron_sdk.state import SortformerConfig, init_state, streaming_update  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CPP_TOOL = ROOT / "cpp" / "build_host" / "test_compress"
SIL_EMB = paths.EXPORT_DIR / "learnable_sil_emb.npy"


def synthetic_config() -> SortformerConfig:
    config = SortformerConfig(
        chunk_len=6,
        chunk_left_context=0,
        chunk_right_context=4,
        fifo_len=264,
        spkcache_len=264,
        spkcache_update_period=222,
    )
    config.learnable_sil_emb = np.load(SIL_EMB).astype(np.float32)
    config.use_learnable_sil_emb = True
    return config


def capture_first_compression():
    cfg = synthetic_config()
    rng = np.random.default_rng(0)
    state = init_state(cfg)
    captured = {}
    original = nps._compress_spkcache

    def wrapper(cfg_, emb, preds, mean_sil):
        out = original(cfg_, emb, preds, mean_sil)
        if not captured:
            captured["emb"] = emb.copy()
            captured["preds"] = preds.copy()
            captured["mean"] = mean_sil.copy()
            captured["numpy_out"] = (out[0].copy(), out[1].copy())
        return out

    nps._compress_spkcache = wrapper
    for step in range(400):
        preds_all = np.clip(rng.normal(0.25, 0.35, size=(cfg.spkcache_len + cfg.fifo_len + 10, 8)), 0, 1).astype(np.float32)
        preds = preds_all[: state.spkcache.shape[0] + state.fifo.shape[0] + 10]
        chunk = rng.standard_normal((10, cfg.fc_d_model)).astype(np.float32)
        state, _ = streaming_update(cfg, state, chunk, preds, 0, 4)
        if captured:
            return cfg, captured, step
    raise RuntimeError("no compression happened")


def numpy_selection(cfg, preds):
    n_skp = cfg.num_speakers
    per_spk = cfg.spkcache_len // n_skp - cfg.spkcache_sil_frames_per_spk
    strong = int(per_spk * cfg.strong_boost_rate)
    weak = int(per_spk * cfg.weak_boost_rate)
    min_pos = int(per_spk * cfg.min_pos_scores_rate)
    scores = nps._get_log_pred_scores(cfg, preds)
    scores = nps._disable_low_scores(cfg, preds, scores, min_pos)
    if cfg.scores_boost_latest > 0:
        scores[cfg.spkcache_len :, :] += cfg.scores_boost_latest
    scores = nps._boost_topk_scores(cfg, scores, strong, scale_factor=2)
    scores = nps._boost_topk_scores(cfg, scores, weak, scale_factor=1)
    pad = np.full((cfg.spkcache_sil_frames_per_spk, n_skp), np.inf, dtype=scores.dtype)
    padded = np.concatenate([scores, pad], axis=0)
    indices, disabled = nps._get_topk_indices(cfg, padded)
    return padded, indices, disabled


def numpy_selection_from_scores(cfg, padded):
    indices, disabled = nps._get_topk_indices(cfg, padded)
    return padded, indices, disabled


def main() -> None:
    cfg, captured, step = capture_first_compression()
    emb, preds, mean = captured["emb"], captured["preds"], captured["mean"]
    print(f"first compression at step {step}: emb {emb.shape}, preds {preds.shape}")
    padded, np_idx, np_disabled = numpy_selection(cfg, preds)
    np_cache, np_preds = captured["numpy_out"]

    tmp_dir = ROOT / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_in = tmp_dir / "compress_in.bin"
    tmp_out = tmp_dir / "compress_out.bin"
    with open(tmp_in, "wb") as f:
        f.write(b"SFC1")
        f.write(
            struct.pack(
                "<6i",
                cfg.num_speakers,
                cfg.fc_d_model,
                preds.shape[0],
                cfg.spkcache_len,
                cfg.spkcache_sil_frames_per_spk,
                1 if cfg.use_learnable_sil_emb else 0,
            )
        )
        f.write(emb.astype(np.float32).tobytes())
        f.write(preds.astype(np.float32).tobytes())
        f.write(mean.astype(np.float32).tobytes())
    subprocess.run([str(CPP_TOOL), str(tmp_in), str(tmp_out), str(SIL_EMB)], check=True, capture_output=True)

    with open(tmp_out, "rb") as f:
        selected, padded_frames, n_spk, dim = struct.unpack("<4i", f.read(16))
        scores = np.frombuffer(f.read(padded_frames * n_spk * 4), dtype=np.float32).reshape(padded_frames, n_spk)
        cache = np.frombuffer(f.read(selected * dim * 4), dtype=np.float32).reshape(selected, dim)
        cache_preds = np.frombuffer(f.read(selected * n_spk * 4), dtype=np.float32).reshape(selected, n_spk)
        cpp_idx = np.frombuffer(f.read(selected * 4), dtype=np.int32).reshape(selected)
        cpp_disabled = np.frombuffer(f.read(selected), dtype=np.int8).reshape(selected)

    finite = np.isfinite(padded) & np.isfinite(scores)
    print("finite counts numpy/cpp:", int(np.isfinite(padded).sum()), int(np.isfinite(scores).sum()))
    if finite.any():
        print("finite scores maxdiff:", float(np.abs(padded[finite] - scores[finite]).max()))

    same_indices = np.array_equal(np.sort(np_idx), np.sort(cpp_idx))
    print("index sets equal:", same_indices)
    print("disabled equal:", np.array_equal(np_disabled, cpp_disabled))
    print("cache maxdiff:", float(np.abs(np_cache - cache).max()))
    print("cache_preds maxdiff:", float(np.abs(np_preds - cache_preds).max()))


if __name__ == "__main__":
    main()
