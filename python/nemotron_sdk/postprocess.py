"""Numpy port of NeMo's Sortformer/VAD post-processing (preds -> RTTM).

Mirrors ``predlist_to_timestamps`` / ``binarization_vectorized`` / ``filtering``
from ``nemo/collections/asr/parts/utils/vad_utils.py`` and
``generate_diarization_output_lines`` from ``speaker_utils.py`` (NeMo Speech 3.x).

Nemotron-3-Diarization emits 10 ms frames, so the default
``frame_length_in_sec`` is 0.01.
"""

from dataclasses import dataclass

import numpy as np

FRAME_LENGTH_IN_SEC = 0.01


@dataclass
class PostProcessingParams:
    onset: float = 0.5
    offset: float = 0.5
    pad_onset: float = 0.0
    pad_offset: float = 0.0
    min_duration_on: float = 0.0
    min_duration_off: float = 0.0
    filter_speech_first: float = 1.0
    frame_length_in_sec: float = FRAME_LENGTH_IN_SEC


def merge_overlap_segment(segments: np.ndarray) -> np.ndarray:
    if segments.shape[0] <= 1:
        return segments
    segments = segments[np.argsort(segments[:, 0])]
    merge_boundary = segments[:-1, 1] >= segments[1:, 0]
    head_padded = np.concatenate([[False], merge_boundary])
    tail_padded = np.concatenate([merge_boundary, [False]])
    head = segments[~head_padded, 0]
    tail = segments[~tail_padded, 1]
    return np.stack([head, tail], axis=1)


def filter_short_segments(segments: np.ndarray, threshold: float) -> np.ndarray:
    return segments[segments[:, 1] - segments[:, 0] >= threshold]


def get_gap_segments(segments: np.ndarray) -> np.ndarray:
    segments = segments[np.argsort(segments[:, 0])]
    return np.column_stack((segments[:-1, 1], segments[1:, 0]))


def remove_segments(original_segments: np.ndarray, to_be_removed: np.ndarray) -> np.ndarray:
    keep = np.ones(original_segments.shape[0], dtype=bool)
    for segment in to_be_removed:
        keep &= ~(original_segments == segment).all(axis=1)
    return original_segments[keep]


def filtering(segments: np.ndarray, params: PostProcessingParams) -> np.ndarray:
    if segments.shape[0] == 0:
        return segments

    def filter_speech():
        if params.min_duration_on > 0:
            return filter_short_segments(segments, params.min_duration_on)
        return segments

    def restore_short_gaps(current: np.ndarray) -> np.ndarray:
        if params.min_duration_off <= 0 or current.shape[0] == 0:
            return current
        non_speech = get_gap_segments(current)
        short_gaps = remove_segments(non_speech, filter_short_segments(non_speech, params.min_duration_off))
        if short_gaps.shape[0] == 0:
            return current
        return merge_overlap_segment(np.concatenate([current, short_gaps], axis=0))

    if params.filter_speech_first == 1.0:
        segments = filter_speech()
        segments = restore_short_gaps(segments)
    else:
        segments = restore_short_gaps(segments)
        segments = filter_speech()
    return segments


def binarization_vectorized(sequence: np.ndarray, params: PostProcessingParams) -> np.ndarray:
    empty = np.empty((0, 2), dtype=np.float32)
    num_frames = sequence.shape[0]
    if num_frames == 0:
        return empty

    positions = np.arange(1, num_frames + 1)
    onset, offset = params.onset, params.offset

    if onset >= offset:
        force_on = sequence > onset
        force_off = sequence < offset
        has_event = force_on | force_off
        event_positions = np.where(has_event, positions, 0)
        last_event_positions = np.maximum.accumulate(event_positions)
        event_states = np.concatenate([[False], force_on])
        above = event_states[last_event_positions]
    else:
        force_on = sequence >= offset
        force_off = sequence <= onset
        toggle = (sequence > onset) & (sequence < offset)

        has_reset = force_on | force_off
        reset_positions = np.where(has_reset, positions, 0)
        last_reset_positions = np.maximum.accumulate(reset_positions)

        reset_states = np.concatenate([[0], force_on.astype(np.int64)])
        base_state = reset_states[last_reset_positions]
        toggle_prefix = np.concatenate([[0], np.cumsum(toggle.astype(np.int64))])
        toggles_since_reset = toggle_prefix[positions] - toggle_prefix[last_reset_positions]
        above = np.logical_xor(base_state.astype(bool), (toggles_since_reset % 2).astype(bool))

    padded = np.pad(above.astype(np.float32), (1, 1))
    diff = padded[1:] - padded[:-1]
    starts = np.where(diff > 0.5)[0]
    ends = np.where(diff < -0.5)[0]
    if starts.shape[0] == 0:
        return empty

    frame_length_in_sec = params.frame_length_in_sec
    start_times = np.clip(starts.astype(np.float32) * frame_length_in_sec - params.pad_onset, 0.0, None)
    end_times = ends.astype(np.float32) * frame_length_in_sec + params.pad_offset
    valid = end_times > start_times
    if not valid.any():
        return empty
    segments = np.stack([start_times[valid], end_times[valid]], axis=1)
    if params.pad_onset > 0 or params.pad_offset > 0:
        segments = merge_overlap_segment(segments)
    return segments


def predlist_to_timestamps(
    preds: np.ndarray,
    offset: float = 0.0,
    params: PostProcessingParams = None,
    bypass_postprocessing: bool = False,
    precision: int = 2,
):
    """Convert (num_frames, num_speakers) probabilities to per-speaker timestamps."""
    if params is None:
        params = PostProcessingParams()
    if bypass_postprocessing:
        params = PostProcessingParams(
            onset=0.5, offset=0.5, frame_length_in_sec=params.frame_length_in_sec
        )

    timestamps = []
    for spk in range(preds.shape[1]):
        segments = binarization_vectorized(preds[:, spk], params)
        if not bypass_postprocessing:
            segments = filtering(segments, params)
        if segments.shape[0] == 0:
            timestamps.append([])
            continue
        segments = segments + offset
        timestamps.append([[round(float(start), precision), round(float(end), precision)] for start, end in segments])
    return timestamps


def generate_diarization_output_lines(timestamps, model_spk_num: int):
    lines = []
    for spk_idx in range(model_spk_num):
        if not timestamps[spk_idx]:
            continue
        intervals = np.asarray(timestamps[spk_idx], dtype=np.float32).reshape(-1, 2)
        for start, end in merge_overlap_segment(intervals):
            lines.append(f"{start:.3f} {end:.3f} speaker_{int(spk_idx)}")
    return lines


def timestamps_to_rttm_lines(timestamps, uri: str, model_spk_num: int):
    lines = []
    for spk_idx in range(model_spk_num):
        intervals = timestamps[spk_idx]
        if not intervals:
            continue
        merged = merge_overlap_segment(np.asarray(intervals, dtype=np.float32).reshape(-1, 2))
        for start, end in merged:
            duration = float(end) - float(start)
            if duration > 0:
                lines.append(
                    f"SPEAKER {uri} 1 {float(start):.3f} {duration:.3f} <NA> <NA> speaker_{int(spk_idx)} <NA>"
                )
    return lines
