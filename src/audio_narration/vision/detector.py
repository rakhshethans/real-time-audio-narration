# SPDX-License-Identifier: AGPL-3.0-only
from audio_narration.config import cache_dir, select_device
from audio_narration.domain import Detection, DetectionBatch, Frame


def parse_result(result, width: int, height: int) -> tuple[Detection, ...]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()
    coordinates = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(coordinates)
    return tuple(
        Detection(
            result.names[int(label)],
            float(score),
            tuple(
                max(0.0, min(1.0, v / dimension))
                for v, dimension in zip(box, (width, height, width, height))
            ),
            int(track_id) if track_id is not None else None,
        )
        for box, label, score, track_id in zip(coordinates, classes, confidences, ids)
    )


class Detector:
    """Ultralytics maintains ByteTrack state when persist=True on one worker."""

    def __init__(self, settings):
        from ultralytics import YOLO

        path = cache_dir() / "yolo26n.pt"
        if not path.exists():
            raise RuntimeError("YOLO weights missing. Run audio-narration-setup first.")
        self.model = YOLO(str(path))
        self.device = select_device(settings.device)
        self.confidence = settings.confidence

    def detect(self, frame: Frame) -> DetectionBatch:
        result = self.model.track(
            frame.image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]
        height, width = frame.image.shape[:2]
        return DetectionBatch(frame, parse_result(result, width, height))
