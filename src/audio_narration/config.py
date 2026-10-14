# SPDX-License-Identifier: AGPL-3.0-only
import os
from dataclasses import dataclass
from pathlib import Path

QWEN = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
SMOL = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
SMOL_REVISION = "067788b187b95ebe7b2e040b3e4299e342e5b8fd"

KOKORO = "hexgrad/Kokoro-82M"
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"


def cache_dir() -> Path:
    if os.environ.get("NARRATION_CACHE"):
        return Path(os.environ["NARRATION_CACHE"]).resolve()
    if Path("models").is_dir():
        return Path("models").resolve()
    base = Path(
        os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    )
    return (base / "RealTimeAudioNarration" / "models").resolve()


@dataclass(frozen=True)
class Settings:
    camera: int = 0
    mode: str = "hybrid"
    detail: str = "brief"
    device: str = "auto"
    vlm_backend: str = "smol"
    quantize: bool = False
    detection_hz: float = 5.0
    confidence: float = 0.35
    min_interval: float = 1.0
    refresh_interval: float = 2.0
    summary_interval: float = 60.0
    max_age: float | None = None  # Default expiry: 5 seconds on every backend.
    audio: bool = True
    tts_backend: str = "kokoro"
    voice: str = "af_heart"
    speech_rate: float = 1.0
    volume: float = 0.8
    frame_count: int = 2
    frame_interval: float = 0.5
    image_size: int = 320
    max_tokens: int = 96

    def __post_init__(self):
        if self.vlm_backend not in {"smol", "qwen"}:
            raise ValueError("Scene model must be smol or qwen")
        if self.mode not in {"hybrid", "baseline"}:
            raise ValueError("Mode must be hybrid or baseline")
        if self.detail not in {"brief", "standard", "detailed"}:
            raise ValueError("Unknown detail level")
        if not (0 < self.confidence <= 1 and 0 <= self.volume <= 1):
            raise ValueError("Invalid confidence or volume")
        if (
            min(
                self.detection_hz,
                self.min_interval,
                self.refresh_interval,
                self.summary_interval,
                self.frame_interval,
            )
            <= 0
        ):
            raise ValueError("Intervals and frequencies must be positive")
        if self.max_age is not None and self.max_age <= 0:
            raise ValueError("Maximum observation age must be positive")
        if not 1 <= self.frame_count <= 8:
            raise ValueError("Use 1–8 sampled frames")
        if self.tts_backend not in {"kokoro", "system"}:
            raise ValueError("Unknown speech backend")


def select_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable. Install a matching PyTorch CUDA wheel/driver, or select CPU."
            )
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS unavailable; select CPU.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
