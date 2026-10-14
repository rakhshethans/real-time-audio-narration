# SPDX-License-Identifier: AGPL-3.0-only
import time
from threading import Event

import numpy as np

from audio_narration.config import Settings
from audio_narration.domain import Frame, Observation
from audio_narration.pipeline.orchestrator import Pipeline


def fake_camera(index, buffer, stop, emit, metric):
    sequence = 0
    while not stop.wait(0.01):
        sequence += 1
        buffer.publish(Frame(sequence, time.monotonic(), np.zeros((16, 16, 3), dtype=np.uint8)))


class FakeModel:
    def __init__(self, settings):
        pass

    def analyse(self, frames, objects, previous, cancel):
        return Observation(
            frames[-1].timestamp, "A person sits.", ("A person sits.",), uncertain=False
        )


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Worker did not reach expected state")
        time.sleep(0.01)


def test_pipeline_delivers_and_shuts_down():
    messages = []
    pipeline = Pipeline(
        Settings(mode="baseline", audio=False, min_interval=0.05),
        lambda *args: messages.append(args),
        fake_camera,
        vlm_factory=FakeModel,
    )
    pipeline.start()
    try:
        wait_for(lambda: any(kind == "description" for kind, _ in messages))
        pipeline.set_paused(True)
        assert pipeline.paused.is_set()
        pipeline.set_paused(False)
        pipeline.set_narration(False)
        assert not pipeline.enabled
    finally:
        pipeline.request_stop()
        assert pipeline.join() == []


def test_model_initialization_failure_reported():
    messages = []

    def fail(settings):
        raise RuntimeError("weights missing")

    pipeline = Pipeline(
        Settings(mode="baseline", audio=False),
        lambda *a: messages.append(a),
        fake_camera,
        vlm_factory=fail,
    )
    pipeline.start()
    try:
        wait_for(lambda: any(k == "error" and "weights missing" in t for k, t in messages))
    finally:
        pipeline.request_stop()
        assert pipeline.join() == []


def test_camera_failure_stops_everything():
    def fail(*args):
        raise RuntimeError("disconnected")

    pipeline = Pipeline(
        Settings(mode="baseline", audio=False), camera_fn=fail, vlm_factory=FakeModel
    )
    pipeline.start()
    wait_for(pipeline.stop_event.is_set)
    assert pipeline.join() == []


def test_pause_resume_cannot_revive_cancelled_inference():
    entered, release = Event(), Event()
    messages = []

    class Slow(FakeModel):
        def analyse(self, frames, objects, previous, cancel):
            entered.set()
            release.wait(2)
            assert cancel.is_set()
            raise InterruptedError()

    pipeline = Pipeline(
        Settings(mode="baseline", audio=False),
        lambda *a: messages.append(a),
        fake_camera,
        vlm_factory=Slow,
    )
    pipeline.start()
    try:
        assert entered.wait(2)
        pipeline.set_paused(True)
        pipeline.set_paused(False)
        release.set()
        time.sleep(0.05)
        assert not any(k == "description" for k, _ in messages)
    finally:
        release.set()
        pipeline.request_stop()
        assert pipeline.join() == []


def test_capture_releases_camera_on_failure():
    import pytest

    from audio_narration.camera.capture import capture
    from audio_narration.camera.frame_buffer import FrameBuffer

    class Camera:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    camera = Camera()
    with pytest.raises(RuntimeError):
        capture(
            0,
            FrameBuffer(),
            Event(),
            lambda *a: None,
            lambda *a: None,
            factory=lambda index: camera,
        )
    assert camera.released


def test_speech_is_warmed_before_scene_requests():
    ready, entered = Event(), Event()
    messages = []

    class Speech:
        def __init__(self, settings):
            pass

        def warm_up(self):
            entered.set()
            ready.wait(2)

        def speak(self, text, cancel, first_audio, metric, **kwargs):
            first_audio(0.01)

    pipeline = Pipeline(
        Settings(mode="baseline"),
        lambda *a: messages.append(a),
        fake_camera,
        vlm_factory=FakeModel,
        tts_factory=Speech,
    )
    pipeline.start()
    try:
        assert entered.wait(2)
        assert not any(k == "description" for k, _ in messages)
        ready.set()
        wait_for(lambda: any(k == "spoken" for k, _ in messages))
    finally:
        ready.set()
        pipeline.request_stop()
        assert pipeline.join() == []


def test_deadline_cancels_slow_scene_without_speaking():
    expired = Event()
    messages = []

    class Slow(FakeModel):
        def analyse(self, frames, objects, previous, cancel):
            assert cancel.wait(1)
            expired.set()
            raise InterruptedError()

    pipeline = Pipeline(
        Settings(mode="baseline", audio=False, max_age=0.05),
        lambda *a: messages.append(a),
        fake_camera,
        vlm_factory=Slow,
    )
    pipeline.start()
    try:
        assert expired.wait(2)
        assert not any(k in {"spoken", "description"} for k, _ in messages)
    finally:
        pipeline.request_stop()
        assert pipeline.join() == []


def test_audio_load_failure_keeps_text_descriptions_available():
    messages = []

    def fail(settings):
        raise RuntimeError("No speech backend")

    pipeline = Pipeline(
        Settings(mode="baseline"),
        lambda *a: messages.append(a),
        fake_camera,
        vlm_factory=FakeModel,
        tts_factory=fail,
    )
    pipeline.start()
    try:
        wait_for(lambda: any(k == "description" for k, _ in messages))
        assert any(k == "error" and "No speech backend" in t for k, t in messages)
    finally:
        pipeline.request_stop()
        assert pipeline.join() == []
