# SPDX-License-Identifier: AGPL-3.0-only
from collections import deque
from threading import Lock

from audio_narration.domain import Frame


class FrameBuffer:
    """One preview frame and a fixed-sized, time-sampled clip; no inference backlog."""

    def __init__(self, count: int = 4, interval: float = 0.7):
        self._lock = Lock()
        self._latest: Frame | None = None
        self._samples: deque[Frame] = deque(maxlen=count)
        self.interval = interval

    def publish(self, frame: Frame) -> None:
        with self._lock:
            self._latest = frame
            if not self._samples or frame.timestamp - self._samples[-1].timestamp >= self.interval:
                self._samples.append(frame)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def clip(self) -> tuple[Frame, ...]:
        with self._lock:
            frames = list(self._samples)
            if self._latest and (not frames or frames[-1].sequence != self._latest.sequence):
                frames = frames[1:] + [self._latest] if frames else [self._latest]
            return tuple(frames)

    def clear(self) -> None:
        with self._lock:
            self._latest = None
            self._samples.clear()
