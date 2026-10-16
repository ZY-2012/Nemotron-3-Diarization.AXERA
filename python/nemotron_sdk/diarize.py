"""End-to-end streaming diarizer: waveform -> RTTM.

Two scatter-free graphs are executed per step (pre-encode + encoder); the
speaker-cache packing, streaming state update, log-mel front-end and
post-processing are all numpy and shared by host and AX650.

The encoder graph returns high-resolution (10 ms) speaker probabilities; the
host extracts the chunk output for the transcript and mean-pools the packed
sequence back to the 80 ms encoder grid for the AOSC state update — exactly
like ``SortformerEncLabelModel.forward_streaming_step`` does.
"""

import argparse
from pathlib import Path
from typing import Dict, Protocol

import numpy as np

from .feature import SAMPLE_RATE, load_feature_assets, log_mel_spectrogram, mel_filterbank, true_num_frames
from .postprocess import PostProcessingParams, predlist_to_timestamps, timestamps_to_rttm_lines
from .state import SortformerConfig, StreamingState, downsample_preds, init_state, iter_chunks, pre_encode_length, streaming_update


class GraphPair(Protocol):
    def pre_encode(self, chunk: np.ndarray) -> np.ndarray:
        """chunk [1, chunk_mel, 128] -> chunk embeddings [1, chunk_embs, 512]."""

    def encode(self, seq: np.ndarray, total_lengths: int) -> np.ndarray:
        """packed seq [1, state_len + chunk_embs, 512] -> high-res preds [1, T*8, num_speakers]."""


def _provider_alias(ort, provider: str) -> str:
    aliases = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
    if provider == "auto":
        available = ort.get_available_providers()
        return "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider"
    return aliases.get(provider, provider)


class OnnxGraphPair:
    """ONNX Runtime implementation of the split graphs (host verification)."""

    def __init__(self, preencode_path: str, encoder_path: str, provider: str = "auto"):
        import onnxruntime as ort

        provider = _provider_alias(ort, provider)
        self.pre_session = ort.InferenceSession(str(preencode_path), providers=[provider])
        self.encoder_session = ort.InferenceSession(str(encoder_path), providers=[provider])
        encoder_inputs = {node.name: node for node in self.encoder_session.get_inputs()}
        self.seq_width = int(encoder_inputs["seq"].shape[1])

    def pre_encode(self, chunk: np.ndarray) -> np.ndarray:
        return self.pre_session.run(None, {"chunk": chunk.astype(np.float32)})[0]

    def encode(self, seq: np.ndarray, total_lengths: int) -> np.ndarray:
        feed = {"seq": seq.astype(np.float32), "total_lengths": np.array([total_lengths], dtype=np.float32)}
        return self.encoder_session.run(None, feed)[0]


class AxengineGraphPair:
    """axengine implementation of the split graphs (AX650 board)."""

    def __init__(self, preencode_path: str, encoder_path: str):
        import axengine

        self.pre_session = axengine.InferenceSession(str(preencode_path))
        self.encoder_session = axengine.InferenceSession(str(encoder_path))

    def pre_encode(self, chunk: np.ndarray) -> np.ndarray:
        return self.pre_session.run(None, {"chunk": chunk.astype(np.float32)})[0]

    def encode(self, seq: np.ndarray, total_lengths: int) -> np.ndarray:
        feed = {"seq": seq.astype(np.float32), "total_lengths": np.array([total_lengths], dtype=np.float32)}
        return self.encoder_session.run(None, feed)[0]


class StreamingDiarizer:
    """Nemotron-3-Diarization streaming diarizer over the split graphs."""

    def __init__(self, graphs: GraphPair, config: SortformerConfig = None, feature_assets=None):
        self.graphs = graphs
        self.config = config or SortformerConfig()
        self.chunk_width = (
            self.config.chunk_left_context + self.config.chunk_len + self.config.chunk_right_context
        ) * self.config.subsampling_factor
        self.state_len = self.config.spkcache_len + self.config.fifo_len
        if feature_assets is not None:
            self.feature_window, self.mel_filter = load_feature_assets(feature_assets)
        else:
            self.feature_window, self.mel_filter = None, None

    def process_features(self, features: np.ndarray, on_step=None) -> np.ndarray:
        cfg = self.config
        state = init_state(cfg)
        mel_dim = features.shape[1]
        upsample = cfg.upsample_factor
        predictions = []
        for step, (chunk, left_offset, right_offset) in enumerate(iter_chunks(cfg, features)):
            chunk_valid = chunk.shape[0]
            chunk_input = np.zeros((1, self.chunk_width, mel_dim), dtype=np.float32)
            chunk_input[0, :chunk_valid] = chunk

            chunk_embs_full = self.graphs.pre_encode(chunk_input)[0]
            chunk_embs_length = pre_encode_length(chunk_valid)
            chunk_embs = chunk_embs_full[:chunk_embs_length]

            spkcache_len = state.spkcache.shape[0]
            fifo_len = state.fifo.shape[0]
            total_lengths = spkcache_len + fifo_len + chunk_embs_length
            seq_width = getattr(self.graphs, "seq_width", self.state_len + chunk_embs_full.shape[0])
            seq = np.zeros((1, seq_width, cfg.fc_d_model), dtype=np.float32)
            seq[0, :spkcache_len] = state.spkcache
            seq[0, spkcache_len : spkcache_len + fifo_len] = state.fifo
            seq[0, spkcache_len + fifo_len : total_lengths] = chunk_embs

            if on_step is not None:
                on_step(step, chunk_input, seq, total_lengths)

            high_res = self.graphs.encode(seq, total_lengths)[0]  # (seq_width * upsample, n_spk)

            lc_enc = round(left_offset / cfg.subsampling_factor)
            rc_enc = int(np.ceil(right_offset / cfg.subsampling_factor))
            chunk_len_enc = chunk_embs_length - lc_enc - rc_enc

            # 10 ms chunk output for the transcript (uses the *pre-update* state lengths).
            start = (spkcache_len + fifo_len + lc_enc) * upsample
            chunk_preds = high_res[start : start + chunk_len_enc * upsample]
            predictions.append(chunk_preds)

            # 80 ms predictions over the packed sequence feed the AOSC state update.
            preds_enc = downsample_preds(high_res, upsample, total_lengths)
            state, _ = streaming_update(cfg, state, chunk_embs, preds_enc, lc_enc, rc_enc)
        if not predictions:
            return np.zeros((0, cfg.num_speakers), dtype=np.float32)
        # NeMo truncates the accumulated predictions to the preprocessor frame count.
        return np.concatenate(predictions, axis=0)[: features.shape[0]]

    def process_wav(self, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE, on_step=None) -> np.ndarray:
        features = log_mel_spectrogram(
            waveform,
            sample_rate,
            mel_filter=self.mel_filter if self.mel_filter is not None else mel_filterbank(),
            window=self.feature_window,
        )
        features = features[: true_num_frames(np.asarray(waveform).reshape(-1).shape[0])]
        return self.process_features(features, on_step=on_step)

    def rttm_lines(self, preds: np.ndarray, uri: str, params: PostProcessingParams = None, bypass: bool = False):
        timestamps = predlist_to_timestamps(preds, params=params, bypass_postprocessing=bypass)
        num_speakers = preds.shape[1] if preds.ndim == 2 else self.config.num_speakers
        return timestamps_to_rttm_lines(timestamps, uri, num_speakers)


def load_learnable_sil_emb(config: SortformerConfig, path) -> None:
    """Attach the learned silence embedding (npy, shape [512]) to the config."""
    emb = np.load(str(path)).astype(np.float32).reshape(-1)
    if emb.shape[0] != config.fc_d_model:
        raise ValueError(f"expected {config.fc_d_model} silence embedding values, got {emb.shape[0]}")
    config.learnable_sil_emb = emb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preencode", default=None, help="preencode.onnx (host mode)")
    parser.add_argument("--encoder", default=None, help="encoder.onnx (host mode)")
    parser.add_argument("--preencode-axmodel", default=None, help="preencode axmodel (board mode)")
    parser.add_argument("--encoder-axmodel", default=None, help="encoder axmodel (board mode)")
    parser.add_argument("--sil-emb", default=None, help="learnable silence embedding (.npy)")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--rttm", default=None)
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--chunk-len", type=int, default=340)
    parser.add_argument("--chunk-left-context", type=int, default=0)
    parser.add_argument("--chunk-right-context", type=int, default=40)
    parser.add_argument("--fifo-len", type=int, default=40)
    parser.add_argument("--spkcache-len", type=int, default=264)
    parser.add_argument("--spkcache-update-period", type=int, default=300)
    parser.add_argument("--bypass-postproc", action="store_true")
    args = parser.parse_args()

    if args.preencode and args.encoder:
        graphs = OnnxGraphPair(args.preencode, args.encoder, provider=args.provider)
    elif args.preencode_axmodel and args.encoder_axmodel:
        graphs = AxengineGraphPair(args.preencode_axmodel, args.encoder_axmodel)
    else:
        raise SystemExit("pass either --preencode/--encoder or the axmodel pair")

    config = SortformerConfig(
        chunk_len=args.chunk_len,
        chunk_left_context=args.chunk_left_context,
        chunk_right_context=args.chunk_right_context,
        fifo_len=args.fifo_len,
        spkcache_len=args.spkcache_len,
        spkcache_update_period=args.spkcache_update_period,
    )
    if args.sil_emb:
        load_learnable_sil_emb(config, args.sil_emb)
    diarizer = StreamingDiarizer(graphs, config)

    import soundfile as sf

    waveform, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    preds = diarizer.process_wav(waveform, sample_rate)
    uri = Path(args.wav).stem
    lines = diarizer.rttm_lines(preds, uri, bypass=args.bypass_postproc)
    print(f"{uri}: {len(preds)} frames, {len(lines)} segments")
    if args.rttm:
        Path(args.rttm).write_text("\n".join(lines) + "\n")
        print(f"rttm -> {args.rttm}")


if __name__ == "__main__":
    main()
