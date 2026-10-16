#!/usr/bin/env python3
"""NeMo Nemotron-3-Diarization FP32 streaming baseline.

Runs the upstream PyTorch model in streaming mode (default: the 30.4 s
"offline" configuration from the model card), saves per-frame speaker
probabilities at 10 ms resolution as the golden reference for the ONNX /
AX650 ports, and reports DER with the same scorer the NeMo example uses.

    python baseline_nemo.py --meetings ES2004a IS1009a --collar 0.0
    python baseline_nemo.py --meetings ES2004a --preset low_latency
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import paths


def load_model(model_id: str, device: torch.device):
    from nemo.collections.asr.models import SortformerEncLabelModel

    if str(model_id).endswith(".nemo"):
        model = SortformerEncLabelModel.restore_from(restore_path=str(model_id), map_location=device)
    else:
        model = SortformerEncLabelModel.from_pretrained(model_name=str(model_id), map_location=device)
    return model.eval()


def apply_streaming_config(model, args) -> None:
    modules = model.sortformer_modules
    modules.chunk_len = args.chunk_len
    modules.chunk_left_context = args.chunk_left_context
    modules.chunk_right_context = args.chunk_right_context
    modules.fifo_len = args.fifo_len
    modules.spkcache_len = args.spkcache_len
    modules.spkcache_update_period = args.spkcache_update_period
    model.async_streaming = False
    model.async_pad_to_max = False
    modules.async_desync_updates = False
    model._check_streaming_parameters()


def apply_preset(args) -> None:
    """Fill streaming parameters from a preset (or module defaults) unless overridden on the CLI."""
    if args.preset is not None:
        cfg = paths.preset(args.preset)
    else:
        cfg = {
            "chunk_len": paths.CHUNK_LEN,
            "chunk_left_context": paths.CHUNK_LEFT_CONTEXT,
            "chunk_right_context": paths.CHUNK_RIGHT_CONTEXT,
            "fifo_len": paths.FIFO_LEN,
            "spkcache_len": paths.SPKCACHE_LEN,
            "spkcache_update_period": paths.SPKCACHE_UPDATE_PERIOD,
        }
    for key, value in cfg.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def meeting_list(args) -> list:
    audio_dir, pattern, _, meetings = dataset_paths(args)
    if args.meetings:
        return args.meetings
    return [mid for mid in meetings if (audio_dir / pattern.format(mid=mid)).exists()]


def dataset_paths(args):
    if args.dataset == "ali":
        import make_ali_manifest

        return make_ali_manifest.AUDIO_DIR, "{mid}.wav", make_ali_manifest.MANIFEST, make_ali_manifest.MEETINGS
    import make_ami_manifest

    return paths.AMI_AUDIO_DIR, "{mid}.Mix-Headset.wav", paths.MANIFEST, make_ami_manifest.MEETINGS


def run_meeting(model, mid: str, device: torch.device, args):
    audio_dir, pattern, _, _ = dataset_paths(args)
    wav_path = audio_dir / pattern.format(mid=mid)
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if sample_rate != paths.SAMPLE_RATE:
        raise ValueError(f"{mid}: expected {paths.SAMPLE_RATE} Hz, got {sample_rate}")
    audio = audio.mean(axis=1)
    signal = torch.from_numpy(audio).unsqueeze(0).to(device)
    length = torch.tensor([audio.shape[0]], dtype=torch.long, device=device)

    def infer():
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"
        ):
            return model(signal, length)

    for _ in range(args.warmup):
        infer()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    preds = infer()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    with torch.inference_mode():
        features, feature_length = model.process_signal(audio_signal=signal, audio_signal_length=length)
    features = features[:, :, : int(feature_length.max())]

    preds = preds.float().cpu().numpy()[0]
    features = features.float().cpu().numpy()[0].T  # (mel_frames, mel_features)
    out_dir = paths.OUT / "baseline"
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    (out_dir / "feats").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "preds" / f"{mid}.npy", preds)
    np.save(out_dir / "feats" / f"{mid}.npy", features)

    duration = audio.shape[0] / sample_rate
    frame_count = preds.shape[0]
    return {
        "meeting": mid,
        "duration_s": duration,
        "frames": int(frame_count),
        "frame_rate": frame_count / duration if duration else 0.0,
        "elapsed_s": elapsed,
        "rtf": elapsed / duration if duration else 0.0,
    }


def evaluate(args) -> dict:
    import make_ali_manifest
    import make_ami_manifest
    from nemo.collections.asr.metrics.der import score_labels
    from nemo.collections.asr.parts.utils.diarization_utils import convert_pred_mat_to_segments
    from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map

    _, _, manifest, _ = dataset_paths(args)
    if not manifest.exists():
        make_ali_manifest.main() if args.dataset == "ali" else make_ami_manifest.main()
    audio_rttm_dict = audio_rttm_map(str(manifest))

    out_dir = paths.OUT / "baseline"
    preds_dir = out_dir / args.preds_subdir
    keys, preds_list = [], []
    for key in audio_rttm_dict:
        for candidate in (key, key.split(".")[0]):
            path = preds_dir / f"{candidate}.npy"
            if path.exists():
                keys.append(key)
                preds_list.append(torch.from_numpy(np.load(path)))
                break
    if not keys:
        raise FileNotFoundError(f"no predictions under {preds_dir}")
    audio_rttm_dict = {key: audio_rttm_dict[key] for key in keys}

    postprocessing_cfg = None
    if not args.bypass_postproc:
        from nemo.collections.asr.parts.utils.vad_utils import load_postprocessing_from_yaml

        postprocessing_cfg = load_postprocessing_from_yaml(args.postprocessing_yaml)

    rttm_dir = out_dir / "hyp"
    rttm_dir.mkdir(parents=True, exist_ok=True)
    all_hyps, all_refs, all_uems = convert_pred_mat_to_segments(
        audio_rttm_dict,
        postprocessing_cfg=postprocessing_cfg,
        batch_preds_list=preds_list,
        unit_10ms_frame_count=args.unit_10ms_frame_count,
        bypass_postprocessing=args.bypass_postproc,
        out_rttm_dir=str(rttm_dir),
    )
    metric, _, _ = score_labels(
        AUDIO_RTTM_MAP=audio_rttm_dict,
        all_reference=all_refs,
        all_hypothesis=all_hyps,
        all_uem=all_uems,
        collar=args.collar,
        ignore_overlap=args.ignore_overlap,
    )
    return {
        "collar": args.collar,
        "ignore_overlap": args.ignore_overlap,
        "bypass_postprocessing": args.bypass_postproc,
        "postprocessing_yaml": args.postprocessing_yaml,
        "der": abs(metric),
        "miss": float(metric["missed detection"]),
        "false_alarm": float(metric["false alarm"]),
        "confusion": float(metric["confusion"]),
        "total": float(metric["total"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meetings", nargs="*", default=None)
    parser.add_argument("--dataset", choices=["ami", "ali"], default="ami")
    parser.add_argument("--model", default=None, help="HF id or local .nemo (default: local .nemo when present)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true", help="run encoder in bf16 autocast (GPU only)")
    parser.add_argument("--preset", choices=sorted(paths.STREAMING_PRESETS), default=None, help="streaming preset")
    parser.add_argument("--chunk-len", type=int, default=None)
    parser.add_argument("--chunk-left-context", type=int, default=None)
    parser.add_argument("--chunk-right-context", type=int, default=None)
    parser.add_argument("--fifo-len", type=int, default=None)
    parser.add_argument("--spkcache-len", type=int, default=None)
    parser.add_argument("--spkcache-update-period", type=int, default=None)
    parser.add_argument("--collar", type=float, default=0.0)
    parser.add_argument("--ignore-overlap", action="store_true")
    parser.add_argument("--bypass-postproc", action="store_true")
    parser.add_argument("--postprocessing-yaml", default=None)
    parser.add_argument("--unit-10ms-frame-count", type=int, default=paths.OUTPUT_SUBSAMPLING_FACTOR)
    parser.add_argument("--skip-der", action="store_true")
    parser.add_argument("--preds-subdir", default="preds", help="subdirectory under baseline/ to score")
    parser.add_argument("--eval-only", action="store_true", help="skip inference and only score existing predictions")
    parser.add_argument("--warmup", type=int, default=1, help="warmup inference runs before timing")
    args = parser.parse_args()

    if args.eval_only:
        result = evaluate(args)
        print(f"DER (collar={args.collar:.2f}): {result['der']:.4%}")
        out_dir = paths.OUT / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_path = out_dir / f"eval_{args.preds_subdir}.json"
        with eval_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"eval -> {eval_path}")
        return

    apply_preset(args)
    model_path = args.model or (str(paths.LOCAL_NEMO) if paths.LOCAL_NEMO.exists() else paths.MODEL)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    apply_streaming_config(model, args)

    print(
        "streaming config:",
        f"chunk_len={args.chunk_len}",
        f"lc={args.chunk_left_context}",
        f"rc={args.chunk_right_context}",
        f"fifo={args.fifo_len}",
        f"spkcache={args.spkcache_len}",
        f"update_period={args.spkcache_update_period}",
        f"preset={args.preset}",
    )

    runs = []
    for mid in meeting_list(args):
        result = run_meeting(model, mid, device, args)
        print(
            f"{mid}: {result['duration_s']:8.1f}s  {result['frames']:6d} frames "
            f"({result['frame_rate']:.2f}/s)  {result['rtf']:.4f} RTF"
        )
        runs.append(result)

    report = {
        "model": model_path,
        "config": {
            "preset": args.preset,
            "chunk_len": args.chunk_len,
            "chunk_left_context": args.chunk_left_context,
            "chunk_right_context": args.chunk_right_context,
            "fifo_len": args.fifo_len,
            "spkcache_len": args.spkcache_len,
            "spkcache_update_period": args.spkcache_update_period,
        },
        "runs": runs,
    }
    if not args.skip_der:
        report["eval"] = evaluate(args)
        print(f"DER (collar={args.collar:.2f}): {report['eval']['der']:.4%}")

    out_dir = paths.OUT / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
