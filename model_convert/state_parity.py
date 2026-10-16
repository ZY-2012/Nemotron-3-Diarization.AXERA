#!/usr/bin/env python3
"""Step-by-step parity of the numpy streaming state machine against NeMo (torch).

Feeds identical encoder outputs (ONNX high-res preds -> numpy 80 ms downsample)
into both implementations and reports the first step/field where the states
diverge.

    python state_parity.py EN2002a [--max-steps 600]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import paths  # noqa: E402
from baseline_nemo import apply_preset  # noqa: E402
from baseline_nemo import apply_streaming_config, load_model  # noqa: E402
from host_stream import float_to_config  # noqa: E402
from nemotron_sdk.diarize import OnnxGraphPair  # noqa: E402
from nemotron_sdk.state import downsample_preds, init_state, iter_chunks, pre_encode_length, streaming_update  # noqa: E402


def compare(step, label, torch_value, numpy_value):
    torch_array = torch_value.detach().cpu().numpy() if torch.is_tensor(torch_value) else np.asarray(torch_value)
    numpy_array = np.asarray(numpy_value)
    if torch_array.shape != numpy_array.shape:
        print(f"step {step}: {label} shape {torch_array.shape} vs {numpy_array.shape}")
        return False
    diff = np.abs(torch_array.astype(np.float64) - numpy_array.astype(np.float64))
    max_diff = float(diff.max()) if diff.size else 0.0
    if max_diff > 1e-5:
        print(f"step {step}: {label} maxdiff={max_diff:.3e} (n={diff.size})")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting")
    parser.add_argument("--model", default=None)
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
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    apply_preset(args)

    cfg = float_to_config(args)
    graphs = OnnxGraphPair(args.preencode, args.encoder, provider="cpu")

    model_path = args.model or (str(paths.LOCAL_NEMO) if paths.LOCAL_NEMO.exists() else paths.MODEL)
    model = load_model(model_path, torch.device("cpu"))
    apply_streaming_config(model, args)
    modules = model.sortformer_modules
    torch_state = modules.init_streaming_state(batch_size=1, async_streaming=False, device=torch.device("cpu"))
    numpy_state = init_state(cfg)

    features = np.load(paths.OUT / "baseline" / "feats" / f"{args.meeting}.npy")
    chunk_width = (cfg.chunk_left_context + cfg.chunk_len + cfg.chunk_right_context) * cfg.subsampling_factor
    state_len = cfg.spkcache_len + cfg.fifo_len
    step = -1
    failures = 0

    for step, (chunk, left_offset, right_offset) in enumerate(iter_chunks(cfg, features)):
        if args.max_steps and step >= args.max_steps:
            break
        chunk_input = np.zeros((1, chunk_width, features.shape[1]), dtype=np.float32)
        chunk_input[0, : chunk.shape[0]] = chunk
        chunk_embs_full = graphs.pre_encode(chunk_input)[0]
        chunk_embs_length = pre_encode_length(chunk.shape[0])
        chunk_embs = chunk_embs_full[:chunk_embs_length]

        spkcache_len = numpy_state.spkcache.shape[0]
        fifo_len = numpy_state.fifo.shape[0]
        total_lengths = spkcache_len + fifo_len + chunk_embs_length
        seq = np.zeros((1, state_len + chunk_embs_full.shape[0], cfg.fc_d_model), dtype=np.float32)
        seq[0, :spkcache_len] = numpy_state.spkcache
        seq[0, spkcache_len : spkcache_len + fifo_len] = numpy_state.fifo
        seq[0, spkcache_len + fifo_len : total_lengths] = chunk_embs
        high_res = graphs.encode(seq, total_lengths)[0]
        preds = downsample_preds(high_res, cfg.upsample_factor, total_lengths)

        lc_enc = round(left_offset / cfg.subsampling_factor)
        rc_enc = int(np.ceil(right_offset / cfg.subsampling_factor))

        torch_state, torch_chunk_preds = modules.streaming_update(
            torch_state,
            chunk=torch.from_numpy(chunk_embs).unsqueeze(0),
            preds=torch.from_numpy(preds).unsqueeze(0),
            lc=lc_enc,
            rc=rc_enc,
        )
        numpy_state, numpy_chunk_preds = streaming_update(cfg, numpy_state, chunk_embs, preds, lc_enc, rc_enc)

        ok = True
        ok &= compare(step, "spkcache", torch_state.spkcache[0], numpy_state.spkcache)
        ok &= compare(step, "fifo", torch_state.fifo[0], numpy_state.fifo)
        ok &= compare(step, "chunk_preds", torch_chunk_preds[0], numpy_chunk_preds)
        if torch_state.spkcache_preds is not None:
            ok &= compare(step, "spkcache_preds", torch_state.spkcache_preds[0], numpy_state.spkcache_preds)
        if torch_state.fifo_preds is not None:
            ok &= compare(step, "fifo_preds", torch_state.fifo_preds[0], numpy_state.fifo_preds)
        if not ok:
            failures += 1
            if failures >= 3:
                print(f"stopping after {failures} divergent steps")
                break

    print("steps compared:", step + 1, f"divergent steps: {failures}")


if __name__ == "__main__":
    main()
