# SPDX-License-Identifier: AGPL-3.0-only
import time
from threading import Event, Lock, Thread
from typing import Callable

from audio_narration.camera.capture import capture
from audio_narration.camera.frame_buffer import FrameBuffer
from audio_narration.config import Settings
from audio_narration.domain import DetectionBatch
from audio_narration.memory.event_detector import AnalysisGate, ChangeDetector
from audio_narration.memory.temporal_memory import TemporalMemory
from audio_narration.narration.scheduler import NarrationScheduler
from audio_narration.narration.tts import speech_factory
from audio_narration.pipeline.workers import worker
from audio_narration.vision.detector import Detector
from audio_narration.vision.vlm import VisionLanguageModel


class Pipeline:
    """One session per start. UI calls are thread-safe; callbacks must queue UI work."""

    def __init__(
        self,
        settings: Settings,
        emit: Callable[[str, str], None] = lambda *a: None,
        camera_fn=capture,
        detector_factory=Detector,
        vlm_factory=VisionLanguageModel,
        tts_factory=speech_factory,
    ):
        self.settings, self.emit = settings, emit
        self.camera_fn, self.detector_factory = camera_fn, detector_factory
        self.vlm_factory, self.tts_factory = vlm_factory, tts_factory
        self.frames = FrameBuffer(settings.frame_count, settings.frame_interval)
        self.stop_event, self.paused, self.analysis_cancel = Event(), Event(), Event()
        self.lock = Lock()
        self.metrics: dict[str, float] = {}
        self.detections: DetectionBatch | None = None
        self.memory = TemporalMemory(summary_interval=settings.summary_interval)
        self.gate = AnalysisGate(settings.mode, settings.min_interval, settings.refresh_interval)
        self.max_age = settings.max_age or 30.0
        self.scheduler = NarrationScheduler(self.max_age)
        self.threads: list[Thread] = []
        self.enabled = True
        self.audio_enabled = settings.audio
        self.epoch = 0
        self.calls = 0
        self.started = 0.0

    def metric(self, key: str, value: float) -> None:
        with self.lock:
            self.metrics[key] = value

    def snapshot(self):
        with self.lock:
            return self.frames.latest(), self.detections, dict(self.metrics)

    def start(self) -> None:
        if self.threads:
            raise RuntimeError("Create a new pipeline for each camera session")
        self.started = time.monotonic()
        self.threads.append(
            worker(
                "camera",
                lambda: self.camera_fn(
                    self.settings.camera, self.frames, self.stop_event, self.emit, self.metric
                ),
                self.error,
            )
        )
        if self.settings.mode == "hybrid":
            self.threads.append(worker("detection", self._detect, self.error))
        self.threads.append(worker("scene", self._analyse, self.error))
        self.threads.append(worker("audio", self._audio, self.error))

    def error(self, name: str, message: str) -> None:
        self.emit(name, f"Failed: {message}")
        self.emit("error", f"{name}: {message}")
        if name == "camera":
            self.request_stop()

    def set_paused(self, value: bool) -> None:
        with self.lock:
            self.epoch += 1
            if value:
                self.paused.set()
                self.analysis_cancel.set()
            else:
                self.paused.clear()
                self.gate.signal({"resume"})
        self.scheduler.flush()

    def set_narration(self, value: bool) -> None:
        with self.lock:
            self.enabled = value
            self.epoch += 1
            if not value:
                self.analysis_cancel.set()
            else:
                self.gate.signal({"narration enabled"})
        self.scheduler.flush()

    def set_audio(self, value: bool) -> None:
        with self.lock:
            self.audio_enabled = value
        if not value:
            self.scheduler.flush()

    def request_stop(self) -> None:
        with self.lock:
            self.stop_event.set()
            self.analysis_cancel.set()
        self.scheduler.close()

    def join(self, timeout: float = 5.0) -> list[str]:
        deadline = time.monotonic() + timeout
        for thread in self.threads:
            thread.join(max(0, deadline - time.monotonic()))
        return [t.name for t in self.threads if t.is_alive()]

    def _detect(self):
        if self.stop_event.is_set():
            return
        self.emit("detection", "Loading YOLO26 / ByteTrack")
        detector = self.detector_factory(self.settings)
        changes = ChangeDetector()
        sequence = -1
        while not self.stop_event.is_set():
            frame = self.frames.latest()
            if frame is not None and frame.sequence != sequence and not self.paused.is_set():
                sequence = frame.sequence
                start = time.monotonic()
                batch = detector.detect(frame)
                self.metric("detection_ms", (time.monotonic() - start) * 1000)
                reasons = changes.update(batch)
                with self.lock:
                    self.detections = batch
                    self.gate.signal(reasons)
                self.emit("detection", "Tracking objects")
            self.stop_event.wait(1 / self.settings.detection_hz)

    def _analyse(self):
        while not self.enabled or self.paused.is_set():
            if self.stop_event.wait(0.1):
                return
        if self.stop_event.is_set():
            return
        self.emit("scene", "Loading Qwen3-VL locally")
        model = self.vlm_factory(self.settings)
        if getattr(model, "device", None) == "cpu" and self.settings.max_age is None:
            self.max_age = self.scheduler.max_age = 180.0
        self.emit(
            "hardware",
            f"{getattr(model, 'device', 'injected test backend')}; "
            f"observation expiry {self.max_age:.0f} s",
        )
        self.emit("scene", "Ready")
        while not self.stop_event.wait(0.1):
            clip = self.frames.clip()
            now = time.monotonic()
            with self.lock:
                allowed = self.enabled and not self.paused.is_set() and not self.stop_event.is_set()
                due = self.gate.due(now)
                if not allowed or not due or not clip or now - clip[-1].timestamp > 1.5:
                    continue
                epoch = self.epoch
                batch = self.detections
                objects = batch.objects if batch and now - batch.frame.timestamp < 2 else ()
                self.gate.consume(now)
                # Each inference gets its own token; a fast pause/resume cannot revive it.
                token = Event()
                self.analysis_cancel = token
            self.emit("scene", "Interpreting a sequence of frames")
            start = time.monotonic()
            self.calls += 1
            try:
                observation = model.analyse(clip, objects, self.memory.latest, token)
            except InterruptedError:
                continue
            except Exception as exc:
                self.error("scene", str(exc))
                with self.lock:
                    self.gate.signal({"retry"})
                continue
            finally:
                self.metric("vlm_ms", (time.monotonic() - start) * 1000)
                self.metric(
                    "vlm_calls_per_min",
                    self.calls * 60 / max(0.01, time.monotonic() - self.started),
                )
            with self.lock:
                if epoch != self.epoch or token.is_set() or self.stop_event.is_set():
                    continue
                if time.monotonic() - observation.timestamp > self.max_age:
                    self.emit(
                        "scene",
                        "Discarded outdated description. Lower detail or use "
                        "acceleration; --max-age changes the expiry limit.",
                    )
                    continue
                item = self.memory.update(observation)
                if item and self.audio_enabled:
                    self.scheduler.submit(item)
            self.emit("description", observation.description)
            self.emit("scene", "Ready")
            if item:
                self.emit("event", item.text)

    def _audio(self):
        speaker = None
        while not self.stop_event.is_set():
            entry = self.scheduler.take()
            if entry is None:
                continue
            item, cancel = entry
            try:
                if speaker is None:
                    self.emit("audio", f"Loading {self.settings.tts_backend} speech")
                    speaker = self.tts_factory(self.settings)
                if (
                    cancel.is_set()
                    or self.stop_event.is_set()
                    or time.monotonic() - item.timestamp > self.max_age
                ):
                    continue
                self.emit("audio", "Speaking")

                def first_audio(seconds):
                    self.metric("time_to_first_audio_ms", seconds * 1000)
                    self.metric("event_to_audio_ms", (time.monotonic() - item.timestamp) * 1000)
                    self.emit("spoken", item.text)

                speaker.speak(
                    item.text,
                    cancel,
                    first_audio,
                    lambda seconds: self.metric("synthesis_ms", seconds * 1000),
                    deadline=item.timestamp + self.max_age,
                )
            except Exception as exc:
                self.error("audio", str(exc))
            finally:
                self.scheduler.finished()
                self.emit("audio", "Idle")
