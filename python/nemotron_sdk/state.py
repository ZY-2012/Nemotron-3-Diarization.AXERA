"""Numpy port of NeMo's synchronous Sortformer streaming state machine.

Mirrors ``SortformerModules.streaming_update`` (eval, batch size 1, no speaker
permutation) from NeMo Speech 3.x for the Nemotron-3-Diarization 8-speaker
model (learnable silence embedding enabled):

    nemo/collections/asr/modules/sortformer_modules.py

The module is dependency-free so it can run on the host (base env) and be
reused by the AX650 board SDK.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NEG_INF = float("-inf")


@dataclass
class SortformerConfig:
    num_speakers: int = 8
    fc_d_model: int = 512
    subsampling_factor: int = 8
    upsample_factor: int = 8
    chunk_len: int = 340
    chunk_left_context: int = 0
    chunk_right_context: int = 40
    fifo_len: int = 40
    spkcache_len: int = 264
    spkcache_update_period: int = 300
    spkcache_sil_frames_per_spk: int = 1
    pred_score_threshold: float = 0.25
    max_index: int = 99999
    scores_boost_latest: float = 0.05
    sil_threshold: float = 0.2
    strong_boost_rate: float = 0.75
    weak_boost_rate: float = 1.5
    min_pos_scores_rate: float = 0.5
    use_learnable_sil_emb: bool = True
    learnable_sil_emb: Optional[np.ndarray] = None


@dataclass
class StreamingState:
    spkcache: np.ndarray = field(default_factory=lambda: np.zeros((0, 512), dtype=np.float32))
    spkcache_preds: np.ndarray = field(default_factory=lambda: np.zeros((0, 8), dtype=np.float32))
    spkcache_compressed: bool = False
    fifo: np.ndarray = field(default_factory=lambda: np.zeros((0, 512), dtype=np.float32))
    fifo_preds: np.ndarray = field(default_factory=lambda: np.zeros((0, 8), dtype=np.float32))
    mean_sil_emb: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    n_sil_frames: int = 0


def init_state(cfg: SortformerConfig) -> StreamingState:
    state = StreamingState()
    state.mean_sil_emb = np.zeros(cfg.fc_d_model, dtype=np.float32)
    return state


def streaming_update(cfg: SortformerConfig, state: StreamingState, chunk: np.ndarray, preds: np.ndarray, lc: int, rc: int):
    """Update speaker cache / FIFO with one chunk; returns the chunk predictions.

    ``chunk`` has shape (lc + chunk_len + rc, emb_dim); ``preds`` has shape
    (spkcache_len + fifo_len + chunk.shape[0], num_speakers) and covers the
    speaker cache, FIFO and chunk regions.
    """
    spkcache_len = state.spkcache.shape[0]
    fifo_len = state.fifo.shape[0]
    chunk_len = chunk.shape[0] - lc - rc

    state.fifo_preds = preds[spkcache_len : spkcache_len + fifo_len]
    chunk_body = chunk[lc : chunk_len + lc]
    chunk_preds = preds[spkcache_len + fifo_len + lc : spkcache_len + fifo_len + chunk_len + lc]

    state.fifo = np.concatenate([state.fifo, chunk_body], axis=0)
    state.fifo_preds = np.concatenate([state.fifo_preds, chunk_preds], axis=0)

    if fifo_len + chunk_len > cfg.fifo_len:
        pop_out_len = cfg.spkcache_update_period
        pop_out_len = max(pop_out_len, chunk_len - cfg.fifo_len + fifo_len)
        pop_out_len = min(pop_out_len, fifo_len + chunk_len)

        pop_out_embs = state.fifo[:pop_out_len]
        pop_out_preds = state.fifo_preds[:pop_out_len]
        if not cfg.use_learnable_sil_emb:
            state.mean_sil_emb, state.n_sil_frames = _get_silence_profile(
                cfg, state.mean_sil_emb, state.n_sil_frames, pop_out_embs, pop_out_preds
            )
        state.fifo = state.fifo[pop_out_len:]
        state.fifo_preds = state.fifo_preds[pop_out_len:]

        state.spkcache = np.concatenate([state.spkcache, pop_out_embs], axis=0)
        if state.spkcache_compressed:
            state.spkcache_preds = np.concatenate([state.spkcache_preds, pop_out_preds], axis=0)
        else:
            state.spkcache_preds = np.concatenate([preds[:spkcache_len], pop_out_preds], axis=0)
        if state.spkcache.shape[0] > cfg.spkcache_len:
            state.spkcache, state.spkcache_preds = _compress_spkcache(
                cfg, state.spkcache, state.spkcache_preds, state.mean_sil_emb
            )
            state.spkcache_compressed = True

    return state, chunk_preds


def downsample_preds(preds: np.ndarray, upsample_factor: int, valid_encoder_frames: Optional[int] = None) -> np.ndarray:
    """Mean-pool high-resolution (T*up, S) probabilities back to encoder frames.

    Mirrors ``SortformerModules.downsample_preds`` (avg_pool1d, stride=kernel,
    ceil_mode=True, count_include_pad=False) for the aligned full windows the
    streaming loop produces; positions past ``valid_encoder_frames`` are zeroed
    like ``apply_mask_to_preds`` does.
    """
    if upsample_factor == 1:
        return preds
    frames, n_spk = preds.shape
    pooled = preds.reshape(frames // upsample_factor, upsample_factor, n_spk).mean(axis=1, dtype=np.float32)
    if valid_encoder_frames is not None and valid_encoder_frames < pooled.shape[0]:
        pooled[valid_encoder_frames:] = 0.0
    return pooled


def _get_silence_profile(cfg, mean_sil_emb, n_sil_frames, emb_seq, preds):
    is_sil = preds.sum(axis=1) < cfg.sil_threshold
    sil_count = int(is_sil.sum())
    if sil_count == 0:
        return mean_sil_emb, n_sil_frames
    sil_emb_sum = (emb_seq * is_sil[:, None]).sum(axis=0)
    upd_n_sil_frames = n_sil_frames + sil_count
    total_sil_sum = mean_sil_emb * n_sil_frames + sil_emb_sum
    upd_mean_sil_emb = total_sil_sum / max(upd_n_sil_frames, 1)
    return upd_mean_sil_emb.astype(np.float32), upd_n_sil_frames


def _get_log_pred_scores(cfg, preds):
    log_probs = np.log(np.clip(preds, cfg.pred_score_threshold, None))
    log_1_probs = np.log(np.clip(1.0 - preds, cfg.pred_score_threshold, None))
    log_1_probs_sum = log_1_probs.sum(axis=1, keepdims=True)
    return log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)


def _disable_low_scores(cfg, preds, scores, min_pos_scores_per_spk):
    is_speech = preds > 0.5
    scores = np.where(is_speech, scores, NEG_INF)
    is_pos = scores > 0
    # NeMo sums over the frame dimension (torch: is_pos.sum(dim=1)); batch-free here -> axis=0.
    is_nonpos_replace = (~is_pos) & is_speech & (is_pos.sum(axis=0, keepdims=True) >= min_pos_scores_per_spk)
    return np.where(is_nonpos_replace, NEG_INF, scores)


def _boost_topk_scores(cfg, scores, n_boost_per_spk, scale_factor=1.0, offset=0.5):
    n_frames, n_spk = scores.shape
    n_boost_per_spk = min(n_boost_per_spk, n_frames)
    if n_boost_per_spk <= 0:
        return scores
    for spk in range(n_spk):
        column = scores[:, spk]
        # Stable descending order: ties keep the smaller frame index (matches C++).
        order = np.argsort(-column, kind="stable")[:n_boost_per_spk]
        scores[order, spk] -= scale_factor * math.log(offset)
    return scores


def _get_topk_indices(cfg, scores):
    n_frames, n_spk = scores.shape
    n_frames_no_sil = n_frames - cfg.spkcache_sil_frames_per_spk
    scores_flatten = scores.T.reshape(-1)  # speaker-major, matches permute(0, 2, 1).reshape()
    # Stable descending order: ties keep the smaller flat index (matches C++).
    order = np.argsort(-scores_flatten, kind="stable")
    k = min(cfg.spkcache_len, scores_flatten.shape[0])
    topk_indices = order[:k]
    values = scores_flatten[topk_indices]
    topk_indices = np.where(values != NEG_INF, topk_indices, cfg.max_index)
    topk_indices_sorted = np.sort(topk_indices)
    is_disabled = topk_indices_sorted == cfg.max_index
    topk_indices_sorted = np.remainder(topk_indices_sorted, n_frames)
    is_disabled = is_disabled | (topk_indices_sorted >= n_frames_no_sil)
    topk_indices_sorted = np.where(is_disabled, 0, topk_indices_sorted)
    return topk_indices_sorted, is_disabled


def _compress_spkcache(cfg, emb_seq, preds, mean_sil_emb):
    n_frames, n_spk = preds.shape
    if cfg.use_learnable_sil_emb:
        mean_sil_emb = np.asarray(cfg.learnable_sil_emb, dtype=np.float32)
    spkcache_len_per_spk = cfg.spkcache_len // n_spk - cfg.spkcache_sil_frames_per_spk
    strong_boost_per_spk = math.floor(spkcache_len_per_spk * cfg.strong_boost_rate)
    weak_boost_per_spk = math.floor(spkcache_len_per_spk * cfg.weak_boost_rate)
    min_pos_scores_per_spk = math.floor(spkcache_len_per_spk * cfg.min_pos_scores_rate)

    scores = _get_log_pred_scores(cfg, preds)
    scores = _disable_low_scores(cfg, preds, scores, min_pos_scores_per_spk)
    if cfg.scores_boost_latest > 0:
        scores[cfg.spkcache_len :, :] += cfg.scores_boost_latest
    scores = _boost_topk_scores(cfg, scores, strong_boost_per_spk, scale_factor=2)
    scores = _boost_topk_scores(cfg, scores, weak_boost_per_spk, scale_factor=1)

    if cfg.spkcache_sil_frames_per_spk > 0:
        pad = np.full((cfg.spkcache_sil_frames_per_spk, n_spk), np.inf, dtype=scores.dtype)
        scores = np.concatenate([scores, pad], axis=0)

    topk_indices, is_disabled = _get_topk_indices(cfg, scores)
    spkcache = emb_seq[topk_indices]
    spkcache = np.where(is_disabled[:, None], mean_sil_emb[None, :], spkcache)
    spkcache_preds = preds[topk_indices]
    spkcache_preds = np.where(is_disabled[:, None], 0.0, spkcache_preds)
    return spkcache.astype(np.float32), spkcache_preds.astype(np.float32)


def iter_chunks(cfg: SortformerConfig, features: np.ndarray):
    """Yield (chunk_mel, left_offset, right_offset) following NeMo's ``streaming_feat_loader``."""
    feat_len = features.shape[0]
    start = 0
    while start < feat_len:
        left_offset = min(cfg.chunk_left_context * cfg.subsampling_factor, start)
        end = min(start + cfg.chunk_len * cfg.subsampling_factor, feat_len)
        right_offset = min(cfg.chunk_right_context * cfg.subsampling_factor, feat_len - end)
        chunk = features[start - left_offset : end + right_offset]
        yield chunk, left_offset, right_offset
        start = end


def pre_encode_length(mel_frames: int) -> int:
    """Number of encoder frames after NeMo's FeatureStacking pre-encode (ceil division)."""
    return int(math.ceil(mel_frames / 8))
