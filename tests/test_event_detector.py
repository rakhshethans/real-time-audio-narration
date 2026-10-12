# SPDX-License-Identifier: AGPL-3.0-only
import numpy as np

from audio_narration.domain import Detection, DetectionBatch, Frame
from audio_narration.memory.event_detector import AnalysisGate, ChangeDetector


def batch(t, x=0, visible=True, brightness=0):
    objects = (Detection("person", 0.9, (x, 0, x + 0.1, 0.2), 1),) if visible else ()
    return DetectionBatch(Frame(int(t * 10), t, np.full((48, 64, 3), brightness)), objects)


def test_track_appearance_movement_loss():
    detector = ChangeDetector()
    assert "track appeared" in detector.update(batch(0))
    assert not detector.update(batch(0.5))
    assert "track moved" in detector.update(batch(1, 0.3))
    assert "track disappeared" not in detector.update(batch(1.2, visible=False))
    assert "track disappeared" in detector.update(batch(3, visible=False))


def test_visual_change_independent_of_objects():
    detector = ChangeDetector()
    detector.update(batch(0, visible=False))
    assert "visual change" in detector.update(batch(1, visible=False, brightness=100))


def test_hybrid_refresh_and_rate_limit():
    gate = AnalysisGate("hybrid", 5, 15)
    assert gate.due(0)
    gate.consume(0)
    assert not gate.due(5)
    gate.signal({"movement"})
    assert not gate.due(4)
    assert gate.due(5)
    gate.consume(5)
    assert gate.due(20)


def test_baseline_always_uses_interval():
    gate = AnalysisGate("baseline", 5, 15)
    gate.consume(0)
    assert gate.due(5)


def test_static_hybrid_reduces_calls():
    counts = []
    for mode in ["baseline", "hybrid"]:
        gate = AnalysisGate(mode, 5, 15)
        count = 0
        for t in range(60):
            if gate.due(t):
                gate.consume(t)
                count += 1
        counts.append(count)
    assert counts == [12, 4]  # Scheduling simulation, not an inference benchmark.
