#!/usr/bin/env python3
"""C++ vs numpy streaming_update parity on synthetic data.

Serializes the state machine step to the binary format consumed by
cpp/build_host/test_state and compares the returned state/fields. Uses the
low-latency profile so pops/compressions occur quickly.

    python state_math_parity.py
"""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import paths  # noqa: E402
from nemotron_sdk.state import SortformerConfig, init_state, streaming_update  # noqa: E402

CPP_TOOL = Path(__file__).resolve().parents[1] / "cpp" / "build_host" / "test_state"
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
    if not SIL_EMB.exists():
        raise FileNotFoundError(f"missing {SIL_EMB} (run dump_assets.py)")
    config.learnable_sil_emb = np.load(SIL_EMB).astype(np.float32)
    config.use_learnable_sil_emb = True
    return config


def write_input(path, config, state, chunk, preds, lc, rc):
    with open(path, "wb") as f:
        f.write(b"SFS1")
        f.write(
            struct.pack(
                "<6i", 8, 512, chunk.shape[0], state.spkcache.shape[0], state.fifo.shape[0], state.n_sil_frames
            )
        )
        f.write(
            struct.pack(
                "<5i",
                config.fifo_len,
                config.spkcache_len,
                config.spkcache_update_period,
                config.spkcache_sil_frames_per_spk,
                1 if config.use_learnable_sil_emb else 0,
            )
        )
        for array in (state.spkcache, state.spkcache_preds, state.fifo, state.fifo_preds):
            f.write(np.ascontiguousarray(array, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(state.mean_sil_emb, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(chunk, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(preds, dtype=np.float32).tobytes())
        f.write(struct.pack("<3i", lc, rc, 1 if state.spkcache_compressed else 0))


def read_output(path):
    with open(path, "rb") as f:
        header = struct.unpack("<7i", f.read(28))
        n_spk, dim, chunk_len, t_spk, t_fifo, n_sil, compressed = header

        def array(rows, cols):
            return np.frombuffer(f.read(rows * cols * 4), dtype=np.float32).reshape(rows, cols)

        chunk_preds = array(chunk_len, n_spk)
        spkcache = array(t_spk, dim)
        spkcache_preds = array(t_spk, n_spk)
        fifo = array(t_fifo, dim)
        fifo_preds = array(t_fifo, n_spk)
        mean_sil = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return chunk_preds, spkcache, spkcache_preds, fifo, fifo_preds, mean_sil, n_sil, bool(compressed)


def main() -> None:
    cfg = synthetic_config()
    rng = np.random.default_rng(0)
    state = init_state(cfg)
    tmp_dir = Path(__file__).resolve().parents[1] / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_in = tmp_dir / "state_in.bin"
    tmp_out = tmp_dir / "state_out.bin"
    chunk_frames = cfg.chunk_left_context + cfg.chunk_len + cfg.chunk_right_context

    for step in range(200):
        # Sigmoid-like predictions with plenty of saturation (exercises ties).
        preds_all = np.clip(rng.normal(0.25, 0.35, size=(cfg.spkcache_len + cfg.fifo_len + chunk_frames, 8)), 0, 1).astype(np.float32)
        preds = preds_all[: state.spkcache.shape[0] + state.fifo.shape[0] + chunk_frames]
        chunk = rng.standard_normal((chunk_frames, cfg.fc_d_model)).astype(np.float32)
        lc, rc = (0, 4)

        # Serialize the PRE-step state first (streaming_update mutates in place).
        write_input(tmp_in, cfg, state, chunk, preds, lc, rc)
        np_state, np_chunk_preds = streaming_update(cfg, state, chunk, preds, lc, rc)

        subprocess.run([str(CPP_TOOL), str(tmp_in), str(tmp_out), str(SIL_EMB)], check=True, capture_output=True)
        c_chunk_preds, c_spkcache, c_spkcache_preds, c_fifo, c_fifo_preds, c_mean_sil, c_n_sil, c_compressed = read_output(tmp_out)

        mismatches = []

        def compare(name, a, b):
            if a.shape != b.shape:
                mismatches.append(f"{name}: shape {a.shape} vs {b.shape}")
                return
            diff = float(np.abs(a - b).max()) if a.size else 0.0
            if diff > 1e-4:
                mismatches.append(f"{name}: maxdiff {diff:.3e}")

        compare("chunk_preds", np_chunk_preds, c_chunk_preds)
        compare("spkcache", np_state.spkcache, c_spkcache)
        compare("spkcache_preds", np_state.spkcache_preds, c_spkcache_preds)
        compare("fifo", np_state.fifo, c_fifo)
        compare("fifo_preds", np_state.fifo_preds, c_fifo_preds)
        compare("mean_sil", np_state.mean_sil_emb, c_mean_sil)
        if np_state.n_sil_frames != c_n_sil:
            mismatches.append(f"n_sil {np_state.n_sil_frames} vs {c_n_sil}")
        if np_state.spkcache_compressed != c_compressed:
            mismatches.append(f"compressed {np_state.spkcache_compressed} vs {c_compressed}")
        if mismatches:
            print(f"step {step}: DIVERGED")
            for message in mismatches[:5]:
                print("  ", message)
            return
        state = np_state
        if step % 20 == 0:
            print(f"step {step}: ok (cache {state.spkcache.shape[0]}, fifo {state.fifo.shape[0]})")

    print("all steps matched")


if __name__ == "__main__":
    main()
