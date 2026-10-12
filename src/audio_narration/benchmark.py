# SPDX-License-Identifier: AGPL-3.0-only
"""Real-model comparison on a user-supplied video. No synthetic performance claims."""

import argparse
import json
import statistics
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2
import psutil

from audio_narration.config import Settings
from audio_narration.domain import Frame
from audio_narration.memory.event_detector import AnalysisGate, ChangeDetector
from audio_narration.memory.temporal_memory import TemporalMemory
from audio_narration.vision.detector import Detector
from audio_narration.vision.vlm import VisionLanguageModel


def summarize(samples):
    return (
        {
            "count": len(samples),
            "mean_ms": statistics.mean(samples) * 1000,
            "max_ms": max(samples) * 1000,
        }
        if samples
        else {"count": 0}
    )


def compare(path: Path, settings: Settings, limit=60):
    model = VisionLanguageModel(settings)
    results = {}
    process = psutil.Process()
    for mode in ("baseline", "hybrid"):
        detector = Detector(settings) if mode == "hybrid" else None
        video = cv2.VideoCapture(str(path))
        if not video.isOpened():
            raise RuntimeError(f"Cannot open video {path}")
        fps = video.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            video.release()
            raise RuntimeError("Video must have a valid frame rate")
        gate = AnalysisGate(mode, settings.min_interval, settings.refresh_interval)
        changes, memory = ChangeDetector(), TemporalMemory()
        clip = deque(maxlen=settings.frame_count)
        vlm_times, detection_times = [], []
        last_sample, last_detection = -100.0, -100.0
        objects, errors, candidates = (), 0, 0
        started, cpu_started = time.monotonic(), sum(process.cpu_times()[:2])
        sequence, stamp, peak = 0, 0.0, 0.0
        try:
            while True:
                ok, image = video.read()
                if not ok:
                    break
                stamp = sequence / fps
                if stamp >= limit:
                    break
                sequence += 1
                frame = Frame(sequence, stamp, image)
                if stamp - last_sample >= settings.frame_interval:
                    clip.append(frame)
                    last_sample = stamp
                if detector and stamp - last_detection >= 1 / settings.detection_hz:
                    start = time.monotonic()
                    batch = detector.detect(frame)
                    detection_times.append(time.monotonic() - start)
                    gate.signal(changes.update(batch))
                    objects, last_detection = batch.objects, stamp
                if clip and gate.due(stamp):
                    gate.consume(stamp)
                    start = time.monotonic()
                    try:
                        observation = model.analyse(tuple(clip), objects, memory.latest)
                        candidates += memory.update(observation) is not None
                    except Exception as exc:
                        print(f"{mode}: scene response failed: {exc}")
                        errors += 1
                    vlm_times.append(time.monotonic() - start)
                peak = max(peak, process.memory_info().rss / 1024**2)
        finally:
            video.release()
        wall = time.monotonic() - started
        results[mode] = {
            "source_seconds": stamp,
            "wall_seconds": wall,
            "vlm": summarize(vlm_times),
            "detection": summarize(detection_times),
            "vlm_calls_per_source_minute": len(vlm_times) * 60 / max(stamp, 0.01),
            "narration_candidates": candidates,
            "model_errors": errors,
            "sampled_peak_rss_mb": peak,
            "process_cpu_percent": (sum(process.cpu_times()[:2]) - cpu_started) / wall * 100,
        }
    return {
        "kind": "sequential offline replay, not live webcam FPS",
        "device": model.device,
        "settings": replace(settings).__dict__,
        "modes": results,
        "notes": "Baseline runs first with a shared loaded VLM; warm-up order affects latency. "
        "Speech, GPU utilization, event-to-speech, and live preview are not measured.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    results = compare(args.video, Settings(device=args.device, audio=False), args.seconds)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
