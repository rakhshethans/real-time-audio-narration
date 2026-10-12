# SPDX-License-Identifier: AGPL-3.0-only
import math
from collections import Counter

import numpy as np


class ChangeDetector:
    """Tracking/appearance signals trigger inspection, never assert a human action."""

    def __init__(self, lost_after: float = 1.5):
        self.tracks = {}
        self.labels = Counter()
        self.signature = None
        self.lost_after = lost_after

    def update(self, batch) -> set[str]:
        reasons = set()
        now = batch.frame.timestamp
        current = Counter(d.label for d in batch.objects)
        # Count changes work even before ByteTrack assigns IDs.
        if current != self.labels:
            reasons.add("object counts changed")
        self.labels = current
        for obj in batch.objects:
            if obj.track_id is None:
                continue
            center = ((obj.box[0] + obj.box[2]) / 2, (obj.box[1] + obj.box[3]) / 2)
            old = self.tracks.get(obj.track_id)
            if old is None:
                reasons.add("track appeared")
                self.tracks[obj.track_id] = (center, now)
            else:
                if math.dist(old[0], center) > 0.08:
                    reasons.add("track moved")
                    self.tracks[obj.track_id] = (center, now)
                else:
                    self.tracks[obj.track_id] = (old[0], now)
        for track_id, (_, seen) in list(self.tracks.items()):
            if now - seen > self.lost_after:
                del self.tracks[track_id]
                reasons.add("track disappeared")
        # Cheap grayscale grid independent of detector classes; movement can indicate actions.
        image = batch.frame.image
        grid = (
            image[:: max(1, image.shape[0] // 24), :: max(1, image.shape[1] // 32)]
            .mean(axis=2)
            .astype(np.float32)
        )
        if self.signature is not None and grid.shape == self.signature.shape:
            if np.abs(grid - self.signature).mean() / 255 > 0.06:
                reasons.add("visual change")
        self.signature = grid
        return reasons


class AnalysisGate:
    def __init__(self, mode: str, minimum: float, refresh: float):
        self.mode, self.minimum, self.refresh = mode, minimum, refresh
        self.last = float("-inf")
        self.dirty = True

    def signal(self, reasons) -> None:
        self.dirty |= bool(reasons)

    def due(self, now: float) -> bool:
        elapsed = now - self.last
        return elapsed >= self.minimum and (
            self.mode == "baseline" or self.dirty or elapsed >= self.refresh
        )

    def consume(self, now: float) -> None:
        self.last, self.dirty = now, False
