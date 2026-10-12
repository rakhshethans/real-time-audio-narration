# SPDX-License-Identifier: AGPL-3.0-only
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Frame:
    sequence: int
    timestamp: float  # monotonic time, shared across the process
    image: Any  # owned BGR numpy array; never mutate after publishing


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # normalized xyxy
    track_id: int | None


@dataclass(frozen=True)
class DetectionBatch:
    frame: Frame
    objects: tuple[Detection, ...]


@dataclass(frozen=True)
class Observation:
    timestamp: float
    description: str
    facts: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    changes: tuple[str, ...] = ()
    uncertain: bool = True


@dataclass(frozen=True)
class Narration:
    text: str
    timestamp: float
    priority: int = 1
    key: str = ""
    summary: bool = False


@dataclass
class Metrics:
    values: dict[str, float] = field(default_factory=dict)
