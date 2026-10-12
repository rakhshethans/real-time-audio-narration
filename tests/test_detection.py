# SPDX-License-Identifier: AGPL-3.0-only
from types import SimpleNamespace

import numpy as np
import pytest

from audio_narration.vision.detector import parse_result


class Tensor:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def tolist(self):
        return self.data


class Boxes:
    xyxy = Tensor([[10, 20, 110, 80]])
    cls = Tensor([0])
    conf = Tensor([0.9])
    id = Tensor([42])

    def __len__(self):
        return 1


def test_detection_preserves_track_and_normalizes():
    result = SimpleNamespace(boxes=Boxes(), names={0: "person"})
    (detection,) = parse_result(result, 200, 100)
    assert detection.box == pytest.approx((0.05, 0.2, 0.55, 0.8))
    assert detection.track_id == 42
    assert detection.label == "person"
    assert detection.confidence == 0.9


def test_detection_without_track_id():
    boxes = Boxes()
    boxes.id = None
    (detection,) = parse_result(SimpleNamespace(boxes=boxes, names={0: "person"}), 200, 100)
    assert detection.track_id is None


def test_no_detections():
    assert parse_result(SimpleNamespace(boxes=None), 200, 100) == ()


def test_frame_buffer_bounded_latest_and_ordered():
    from audio_narration.camera.frame_buffer import FrameBuffer
    from audio_narration.domain import Frame

    buffer = FrameBuffer(4, 0.5)
    for sequence in range(1000):
        buffer.publish(Frame(sequence, sequence / 30, np.zeros((1, 1, 3))))
    assert buffer.latest().sequence == 999
    assert len(buffer.clip()) == 4
    assert list(buffer.clip()) == sorted(buffer.clip(), key=lambda f: f.timestamp)
    assert buffer.clip()[-1].sequence == 999
