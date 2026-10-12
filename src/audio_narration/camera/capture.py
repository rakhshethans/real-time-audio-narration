# SPDX-License-Identifier: AGPL-3.0-only
import sys
import time
from threading import Event

from audio_narration.domain import Frame


def open_camera(index: int):
    import cv2

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def discover_cameras(limit: int = 5) -> list[int]:
    available = []
    for index in range(limit):
        camera = open_camera(index)
        try:
            if camera.isOpened() and camera.read()[0]:
                available.append(index)
        finally:
            camera.release()
    return available


def capture(index: int, buffer, stop: Event, emit, metric, factory=open_camera) -> None:
    camera = factory(index)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Camera {index} unavailable or permission denied")
        emit("camera", "Camera active")
        sequence = 0
        started = time.monotonic()
        failures = 0
        while not stop.is_set():
            ok, image = camera.read()
            if not ok:
                failures += 1
                if failures >= 10:
                    raise RuntimeError("Camera disconnected or stopped supplying frames")
                stop.wait(0.1)
                continue
            failures = 0
            sequence += 1
            now = time.monotonic()
            buffer.publish(Frame(sequence, now, image))
            metric("webcam_fps", sequence / max(now - started, 0.01))
    finally:
        camera.release()
        buffer.clear()
        emit("camera", "Camera off")
