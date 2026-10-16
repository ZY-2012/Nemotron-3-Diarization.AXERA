"""Central path configuration for Nemotron-3-Diarization.AXERA.

All scripts resolve inputs/outputs through environment variables so that no
machine-specific path is hard-coded. Defaults mirror the layout on the
development host.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# NeMo Speech checkout that provides SortformerEncLabelModel.
NEMO_REPO = Path(os.environ.get("NEMOTRON_NEMO_REPO", ROOT.parent / "nvidia-audio" / "Speech"))

# HuggingFace model id or local .nemo path.
MODEL = os.environ.get("NEMOTRON_MODEL", "nvidia/Nemotron-3-Diarization")

# Local .nemo copy (downloaded from HuggingFace), preferred when present.
LOCAL_NEMO = Path(os.environ.get("NEMOTRON_NEMO_FILE", ROOT / "models" / "Nemotron-3-Diarization.nemo"))

# Output root for exports, calibration data and compile artifacts.
OUT = Path(os.environ.get("NEMOTRON_OUT", ROOT))
EXPORT_DIR = OUT / "export"
CALIB_DIR = Path(os.environ.get("NEMOTRON_CALIB_DIR", OUT / "calib_data"))
COMPILE_DIR = OUT / "compile"

# Streaming configuration: "very high latency (offline)" 30.4 s input buffer
# (recommended by the model card quick start; best DER and lowest NPU cost/step).
CHUNK_LEN = int(os.environ.get("NEMOTRON_CHUNK_LEN", "340"))
CHUNK_LEFT_CONTEXT = int(os.environ.get("NEMOTRON_CHUNK_LEFT_CONTEXT", "0"))
CHUNK_RIGHT_CONTEXT = int(os.environ.get("NEMOTRON_CHUNK_RIGHT_CONTEXT", "40"))
FIFO_LEN = int(os.environ.get("NEMOTRON_FIFO_LEN", "40"))
SPKCACHE_LEN = int(os.environ.get("NEMOTRON_SPKCACHE_LEN", "264"))
SPKCACHE_UPDATE_PERIOD = int(os.environ.get("NEMOTRON_SPKCACHE_UPDATE_PERIOD", "300"))

# Named streaming presets from the Nemotron-3-Diarization model card.
# Latency = (chunk_len + chunk_right_context) * 80 ms.
STREAMING_PRESETS = {
    # 30.4 s input buffer (offline-style; quick start default).
    "offline": {"spkcache_len": 264, "fifo_len": 40, "chunk_len": 340, "chunk_left_context": 0, "chunk_right_context": 40, "spkcache_update_period": 300},
    # 1.04 s low latency.
    "low_latency": {"spkcache_len": 264, "fifo_len": 264, "chunk_len": 6, "chunk_left_context": 0, "chunk_right_context": 4, "spkcache_update_period": 222},
    # 0.64 s very low latency.
    "very_low_latency": {"spkcache_len": 264, "fifo_len": 264, "chunk_len": 6, "chunk_left_context": 0, "chunk_right_context": 2, "spkcache_update_period": 222},
    # 0.32 s ultra low latency.
    "ultra_low_latency": {"spkcache_len": 264, "fifo_len": 264, "chunk_len": 3, "chunk_left_context": 0, "chunk_right_context": 1, "spkcache_update_period": 222},
}

# Evaluation assets (AMI Mix-Headset + reference RTTMs).
AMI_DIR = Path(os.environ.get("NEMOTRON_AMI_DIR", ROOT.parent / "ami_eval"))
AMI_AUDIO_DIR = AMI_DIR / "audio"
AMI_REF_DIR = AMI_DIR / "ref"
MANIFEST = Path(os.environ.get("NEMOTRON_MANIFEST", ROOT / "manifests" / "ami_dev.json"))

# Calibration / full-stream verification audio (16 kHz wav).
CALIB_WAV = Path(
    os.environ.get(
        "NEMOTRON_CALIB_WAV",
        AMI_AUDIO_DIR / "ES2004a.Mix-Headset.wav",
    )
)

SUBSAMPLING = 8
SAMPLE_RATE = 16000
MEL_FEATURES = 128
# Encoder (transformer) frame duration.
FRAME_DURATION = 0.08
# Returned output frame duration (high_resolution model, output_subsampling_factor=1).
OUT_FRAME_DURATION = 0.01
# Subpixel upsampling factor inside SortformerModules.
UPSAMPLE_FACTOR = 8
# Frames of 10 ms represented by one returned output step.
OUTPUT_SUBSAMPLING_FACTOR = 1
N_SPK = 8
FC_D_MODEL = 512
TF_D_MODEL = 192


def preset(name: str) -> dict:
    """Return a named streaming preset (merging env overrides is left to callers)."""
    return dict(STREAMING_PRESETS[name])


def chunk_mel_frames(chunk_left_context: int = None, chunk_len: int = None, chunk_right_context: int = None) -> int:
    """Number of mel frames the preencode graph expects per streaming step."""
    lc = CHUNK_LEFT_CONTEXT if chunk_left_context is None else chunk_left_context
    cl = CHUNK_LEN if chunk_len is None else chunk_len
    rc = CHUNK_RIGHT_CONTEXT if chunk_right_context is None else chunk_right_context
    return (lc + cl + rc) * SUBSAMPLING


def state_len(spkcache_len: int = None, fifo_len: int = None) -> int:
    """Packed encoder-frame capacity of the speaker cache + FIFO."""
    sc = SPKCACHE_LEN if spkcache_len is None else spkcache_len
    ff = FIFO_LEN if fifo_len is None else fifo_len
    return sc + ff
