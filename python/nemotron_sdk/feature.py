"""NeMo-compatible log-mel front-end (numpy only, no torch/librosa at runtime).

Mirrors ``FilterbankFeatures`` (eval mode) from NeMo Speech 3.x with the
Nemotron-3-Diarization streaming configuration:

    sample_rate=16000, n_fft=512, win_length=400 (hann, periodic=False),
    hop_length=160, n_mels=128, preemph=0.97, mag_power=2.0,
    log(x + 2**-24), normalize="NA", pad_to=16

``log_mel_spectrogram`` returns ``(num_frames, 128)`` float32 where the first
``floor(num_samples / 160)`` frames are the model's true features; the tail is
padding to a multiple of 16 (cropped by ``true_num_frames`` before streaming).
"""

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = 2.0 ** -24
PAD_TO = 16

_MEL_CACHE = {}


def true_num_frames(num_samples: int) -> int:
    """NeMo ``FilterbankFeatures.get_seq_len`` for the 16 kHz / hop-160 setup."""
    return int(num_samples) // HOP_LENGTH


def _hz_to_mel(freq):
    # Slaney mel scale (librosa default, htk=False)
    f_sp = 200.0 / 3.0
    log_step = np.log(6.4) / 27.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    freq = np.asarray(freq, dtype=np.float64)
    mel_linear = freq / f_sp
    mel_log = min_log_mel + np.log(np.maximum(freq, min_log_hz) / min_log_hz) / log_step
    return np.where(freq >= min_log_hz, mel_log, mel_linear)


def _mel_to_hz(mel):
    f_sp = 200.0 / 3.0
    log_step = np.log(6.4) / 27.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    mel = np.asarray(mel, dtype=np.float64)
    hz_linear = f_sp * mel
    hz_log = min_log_hz * np.exp(log_step * (mel - min_log_mel))
    return np.where(mel >= min_log_mel, hz_log, hz_linear)


def mel_filterbank(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS):
    """librosa-compatible Slaney-normalized Slaney-scale mel filterbank."""
    key = (sample_rate, n_fft, n_mels)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]

    fmin, fmax = 0.0, sample_rate / 2.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    freqs = np.linspace(0.0, sample_rate / 2.0, 1 + n_fft // 2)

    fdiff = np.diff(hz)
    ramps = np.subtract.outer(hz, freqs)
    lower = -ramps[np.arange(n_mels), :] / fdiff[np.arange(n_mels)][:, None]
    upper = ramps[np.arange(2, n_mels + 2), :] / fdiff[np.arange(1, n_mels + 1)][:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / (hz[2 : n_mels + 2] - hz[:n_mels])
    weights *= enorm[:, None]
    result = weights.astype(np.float64)
    _MEL_CACHE[key] = result
    return result


def _hann_window(length: int):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / (length - 1))


def load_feature_assets(directory):
    """Load the checkpoint-stored window / mel filterbank from ``directory``.

    The released .nemo buffers are quantized, so they are preferred over the
    analytically recomputed librosa-equivalent values to match NeMo inference.
    Missing files fall back to ``None`` (recomputed values are used).
    """
    from pathlib import Path

    directory = Path(directory)
    window = mel_filter = None
    window_path = directory / "feature_window.npy"
    filter_path = directory / "mel_filterbank.npy"
    if window_path.exists():
        window = np.load(str(window_path)).astype(np.float32)
    if filter_path.exists():
        mel_filter = np.load(str(filter_path)).astype(np.float32)
    return window, mel_filter


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    preemph: float = PREEMPH,
    pad_to: int = PAD_TO,
    mel_filter: np.ndarray = None,
    window: np.ndarray = None,
) -> np.ndarray:
    """Compute the NeMo Nemotron-3-Diarization log-mel features for a mono waveform."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz input, got {sample_rate}")
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)

    if preemph is not None and waveform.size > 0:
        emphasized = np.empty_like(waveform)
        emphasized[0] = waveform[0]
        emphasized[1:] = waveform[1:] - preemph * waveform[:-1]
        waveform = emphasized

    padded = np.pad(waveform, (N_FFT // 2, N_FFT // 2), mode="constant")
    num_samples = waveform.shape[0]
    num_frames = num_samples // HOP_LENGTH
    if num_frames == 0:
        return np.zeros((0, N_MELS), dtype=np.float32)

    if window is None:
        window = _hann_window(WIN_LENGTH).astype(np.float32)
    window = np.asarray(window, dtype=np.float32).reshape(-1)
    if window.shape[0] != WIN_LENGTH:
        raise ValueError(f"expected a {WIN_LENGTH}-sample window, got {window.shape[0]}")
    left_pad = (N_FFT - WIN_LENGTH) // 2
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(num_frames, N_FFT),
        strides=(padded.strides[0] * HOP_LENGTH, padded.strides[0]),
        writeable=False,
    ).copy()
    frames[:, :left_pad] = 0.0
    frames[:, left_pad : left_pad + WIN_LENGTH] *= window
    frames[:, left_pad + WIN_LENGTH :] = 0.0

    spectrum = np.fft.rfft(frames, n=N_FFT, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32) ** 2.0

    if mel_filter is None:
        mel_filter = mel_filterbank()
    mel = magnitude @ mel_filter.T.astype(np.float32)
    log_mel = np.log(mel + LOG_ZERO_GUARD, dtype=np.float32)

    if pad_to and log_mel.shape[0] % pad_to:
        pad = pad_to - log_mel.shape[0] % pad_to
        log_mel = np.pad(log_mel, ((0, pad), (0, 0)), mode="constant")
    return log_mel.astype(np.float32)
