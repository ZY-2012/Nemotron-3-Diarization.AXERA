#!/usr/bin/env python3
"""Export the Nemotron-3-Diarization streaming step as two scatter-free ONNX graphs.

Nemotron-3-Diarization runs its encoder stack through PyTorch FlexAttention,
which does not export to ONNX. This script therefore re-implements the encoder
forward with explicit matmul/softmax attention, sharing the *same weights* as
the NeMo model (the manual path is verified against the upstream FlexAttention
path on CPU before export).

Two graphs, with all packing done on the host (no Scatter/Where index chains,
no dynamic shapes, everything folds under onnxsim):

    preencode.onnx
        chunk              [1, 3040, 128]  mel frames (lc+chunk+rc)*8, time-major
      -> chunk_pre_encode_embs [1, 380, 512]

    encoder.onnx
        seq                [1, 684, 512]   packed spkcache | fifo | chunk_embs | zeros
        total_lengths      [1] float32     valid frames in seq
      -> high_res_preds    [1, 5472, 8]    10 ms speaker probabilities (sigmoid, masked)

The host loop (``host_stream.py`` / ``nemotron_sdk``) runs preencode, packs
``seq``, runs encoder, downsamples the high-res predictions back to 80 ms for
the AOSC state update, then applies NeMo's ``streaming_update``.

    python export_onnx.py [--skip-parity] [--preset offline]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import paths
from baseline_nemo import apply_streaming_config
from baseline_nemo import load_model as load_nemo_model


class PreEncodeStep(torch.nn.Module):
    """FeatureStacking + projection: (B, T_mel, 128) -> (B, T_mel/8, 512)."""

    def __init__(self, encoder):
        super().__init__()
        self.pre_encode = encoder.pre_encode

    def forward(self, chunk):
        lengths = torch.full((chunk.shape[0],), chunk.shape[1], dtype=torch.int64, device=chunk.device)
        embs, _ = self.pre_encode(chunk.transpose(1, 2), lengths)
        return embs


class ExportAttention(torch.nn.Module):
    """Manual multi-head attention equivalent to the FlexAttention path.

    RoPE is applied through the shared ``RotaryPositionalEncoding`` module;
    padding keys are masked with -10000 before the softmax and zeroed after,
    matching the pattern Pulsar2 already handles for the Conformer encoders.
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, x, key_mask):
        """
        Args:
            x: (B, T, d_model) layer-normed input.
            key_mask: (B, 1, 1, T) bool, True for valid key positions.
        """
        batch_size, num_frames, _ = x.shape
        n_heads, head_dim = self.attn.n_heads, self.attn.head_dim
        qkv = self.attn.w_qkv(x).view(batch_size, num_frames, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.attn.rope(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
        scores = scores.masked_fill(~key_mask, -10000.0)
        probs = torch.softmax(scores, dim=-1)
        probs = probs.masked_fill(~key_mask, 0.0)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(batch_size, num_frames, self.attn.d_model)
        return self.attn.out_proj(out)


class ExportEncoderStep(torch.nn.Module):
    """Transformer stack + encoder_proj + subpixel upsample + speaker head.

    Consumes pre-encoded embeddings (bypass_pre_encode semantics) exactly like
    ``SortformerEncLabelModel.forward_streaming_step`` does, but with explicit
    attention so the whole graph exports to ONNX.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        modules = model.sortformer_modules
        encoder = model.encoder
        self.embed_norm = encoder.embed_norm
        self.norm1s = nn.ModuleList([layer.norm1 for layer in encoder.layers])
        self.attentions = nn.ModuleList([ExportAttention(layer.attn) for layer in encoder.layers])
        self.norm2s = nn.ModuleList([layer.norm2 for layer in encoder.layers])
        self.ffns = nn.ModuleList([layer.ffn for layer in encoder.layers])
        self.final_norm = encoder.final_norm
        self.encoder_proj = modules.encoder_proj
        self.subpixel_upsample = modules.subpixel_upsample
        self.first_hidden_to_hidden = modules.first_hidden_to_hidden
        self.single_hidden_to_spks = modules.single_hidden_to_spks
        self.upsample_factor = model.upsample_factor

    def forward(self, seq, total_lengths):
        batch_size, num_frames, _ = seq.shape
        lengths = total_lengths.to(torch.int64)
        positions = torch.arange(num_frames, device=seq.device).view(1, -1)
        key_mask = (positions < lengths.view(-1, 1)).view(batch_size, 1, 1, num_frames)

        x = self.embed_norm(seq)
        for norm1, attention, norm2, ffn in zip(self.norm1s, self.attentions, self.norm2s, self.ffns):
            x = x + attention(norm1(x), key_mask)
            x = x + ffn(norm2(x))
        x = self.final_norm(x)
        if self.encoder_proj is not None:
            x = self.encoder_proj(x)

        projected = self.subpixel_upsample(x.transpose(1, 2)).transpose(1, 2)
        hidden_size = projected.shape[-1] // self.upsample_factor
        upsampled = projected.reshape(batch_size, num_frames, self.upsample_factor, hidden_size)
        upsampled = upsampled.reshape(batch_size, num_frames * self.upsample_factor, hidden_size)

        out = torch.relu(upsampled)
        out = self.first_hidden_to_hidden(out)
        out = torch.relu(out)
        logits = self.single_hidden_to_spks(out)

        out_positions = torch.arange(num_frames * self.upsample_factor, device=seq.device, dtype=torch.float32)
        out_mask = out_positions.view(1, -1) < lengths.view(-1, 1).to(torch.float32) * self.upsample_factor
        preds = torch.sigmoid(logits) * out_mask.unsqueeze(-1).to(logits.dtype)
        return preds


def export_graph(module, inputs, output: Path, input_names, output_names):
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"exporting {output.name} ...")
    with torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            str(output),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            dynamic_axes={},
            do_constant_folding=True,
            dynamo=False,
        )
    import onnx
    import onnxsim

    onnx_model = onnx.load(str(output))
    simplified, ok = onnxsim.simplify(onnx_model, perform_optimization=True)
    if not ok:
        raise RuntimeError(f"onnxsim failed for {output.name}")
    onnx.save(simplified, str(output))
    print(f"  nodes: {len(onnx_model.graph.node)} -> {len(simplified.graph.node)}")
    return simplified


def parity(module, onnx_path: Path, cases, input_names):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    min_cosine = 1.0
    max_absdiff = 0.0
    for case_inputs in cases:
        with torch.no_grad():
            expected = module(*case_inputs)
        feed = {name: value.detach().cpu().numpy() for name, value in zip(input_names, case_inputs)}
        actual = session.run(None, feed)
        expected_list = expected if isinstance(expected, tuple) else (expected,)
        for index, (a, e) in enumerate(zip(actual, expected_list)):
            a = np.asarray(a).reshape(-1)
            e = e.detach().cpu().numpy().reshape(-1)
            cosine = float(np.dot(a, e) / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-12))
            diff = float(np.abs(a - e).max())
            min_cosine = min(min_cosine, cosine)
            max_absdiff = max(max_absdiff, diff)
            print(f"  out[{index}] cosine={cosine:.8f} maxdiff={diff:.3e}")
    return min_cosine, max_absdiff


def upstream_parity(model, module: "ExportEncoderStep", seq, lengths):
    """Verify the manual attention against the upstream NeMo forward path (CPU)."""
    with torch.no_grad():
        emb, emb_lengths = model.frontend_encoder(
            processed_signal=seq, processed_signal_length=lengths.to(torch.int64), bypass_pre_encode=True
        )
        expected = model.forward_infer(emb, emb_lengths)
        actual = module(seq, lengths)
    diff = (actual - expected).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(actual.reshape(1, -1), expected.reshape(1, -1)).item()
    print(f"upstream flex-attention parity: maxdiff={diff:.3e} cosine={cosine:.8f}")
    if diff > 1e-3:
        raise RuntimeError("manual attention path deviates from the upstream model")
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(paths.EXPORT_DIR))
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", choices=sorted(paths.STREAMING_PRESETS), default=None)
    parser.add_argument("--chunk-len", type=int, default=None)
    parser.add_argument("--chunk-left-context", type=int, default=None)
    parser.add_argument("--chunk-right-context", type=int, default=None)
    parser.add_argument("--fifo-len", type=int, default=None)
    parser.add_argument("--spkcache-len", type=int, default=None)
    parser.add_argument("--spkcache-update-period", type=int, default=None)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--skip-upstream-parity", action="store_true")
    args = parser.parse_args()

    from baseline_nemo import apply_preset

    apply_preset(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model or (str(paths.LOCAL_NEMO) if paths.LOCAL_NEMO.exists() else paths.MODEL)
    device = torch.device(args.device)
    model = load_nemo_model(model_path, device)
    apply_streaming_config(model, args)
    model = model.to(device)

    modules = model.sortformer_modules
    chunk_mel_frames = (modules.chunk_left_context + modules.chunk_len + modules.chunk_right_context) * 8
    emb_dim = modules.fc_d_model
    chunk_embs_frames = chunk_mel_frames // 8
    state_len = modules.spkcache_len + modules.fifo_len
    seq_frames = state_len + chunk_embs_frames

    pre = PreEncodeStep(model.encoder).eval().to(device)
    encoder = ExportEncoderStep(model).eval().to(device)

    chunk = torch.randn(1, chunk_mel_frames, model.encoder._feat_in, device=device)
    pre_out = output_dir / "preencode.onnx"
    export_graph(pre, (chunk,), pre_out, ["chunk"], ["chunk_pre_encode_embs"])

    seq = torch.randn(1, seq_frames, emb_dim, device=device)
    total_lengths = torch.full((1,), seq_frames, dtype=torch.float32, device=device)
    enc_out = output_dir / "encoder.onnx"
    export_graph(encoder, (seq, total_lengths), enc_out, ["seq", "total_lengths"], ["high_res_preds"])

    meta = {
        "model": model_path,
        "contract": "split-preencode-encoder-highres",
        "simplified": True,
        "config": {
            "preset": args.preset,
            "chunk_len": args.chunk_len,
            "chunk_left_context": args.chunk_left_context,
            "chunk_right_context": args.chunk_right_context,
            "fifo_len": args.fifo_len,
            "spkcache_len": args.spkcache_len,
            "spkcache_update_period": args.spkcache_update_period,
        },
        "preencode": {
            "file": pre_out.name,
            "inputs": {"chunk": [1, chunk_mel_frames, model.encoder._feat_in]},
            "outputs": {"chunk_pre_encode_embs": [1, chunk_embs_frames, emb_dim]},
        },
        "encoder": {
            "file": enc_out.name,
            "inputs": {"seq": [1, seq_frames, emb_dim], "total_lengths": [1]},
            "outputs": {"high_res_preds": [1, seq_frames * model.upsample_factor, modules.n_spk]},
        },
        "state_len": state_len,
        "chunk_mel_frames": chunk_mel_frames,
        "chunk_embs_frames": chunk_embs_frames,
        "upsample_factor": model.upsample_factor,
        "output_subsampling_factor": model.output_subsampling_factor,
        "frame_duration_s": paths.FRAME_DURATION,
        "out_frame_duration_s": paths.OUT_FRAME_DURATION,
        "sample_rate": paths.SAMPLE_RATE,
        "mel_features": paths.MEL_FEATURES,
        "n_spk": modules.n_spk,
    }

    if not args.skip_upstream_parity and device.type == "cpu":
        print("upstream FlexAttention <-> manual attention parity (packed seq):")
        upstream_cases = [
            (seq, torch.full((1,), seq_frames, dtype=torch.float32)),
            (seq, torch.full((1,), state_len + chunk_embs_frames // 2, dtype=torch.float32)),
            (seq, torch.full((1,), 1.0, dtype=torch.float32)),
        ]
        meta["upstream_parity_maxdiff"] = max(upstream_parity(model, encoder, *case) for case in upstream_cases)

    if not args.skip_parity:
        print("preencode torch <-> ONNX parity:")
        pre_cases = [
            (chunk,),
            (torch.randn(1, chunk_mel_frames, model.encoder._feat_in),),
        ]
        meta["preencode"]["parity_min_cosine"], meta["preencode"]["parity_maxdiff"] = parity(
            pre, pre_out, pre_cases, ["chunk"]
        )
        print("encoder torch <-> ONNX parity:")
        enc_cases = [
            (seq, torch.full((1,), seq_frames, dtype=torch.float32)),
            (seq, torch.full((1,), state_len + chunk_embs_frames // 2, dtype=torch.float32)),
            (seq, torch.full((1,), 1.0, dtype=torch.float32)),
        ]
        meta["encoder"]["parity_min_cosine"], meta["encoder"]["parity_maxdiff"] = parity(
            encoder, enc_out, enc_cases, ["seq", "total_lengths"]
        )

    with (output_dir / "model_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta -> {output_dir / 'model_meta.json'}")


if __name__ == "__main__":
    main()
