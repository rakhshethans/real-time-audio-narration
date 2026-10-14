# SPDX-License-Identifier: AGPL-3.0-only
import argparse
import json
import logging
import sys

from audio_narration.config import Settings


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(
            {
                "time": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "worker": getattr(record, "worker", None),
                "exception": self.formatException(record.exc_info) if record.exc_info else None,
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Local webcam scene narration")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", choices=["hybrid", "baseline"], default="hybrid")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--vlm", choices=["smol", "qwen"], default="smol")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--tts-backend", choices=["kokoro", "system"], default="kokoro")
    parser.add_argument("--max-age", type=float, default=None)
    args = parser.parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    from PySide6.QtWidgets import QApplication

    from audio_narration.ui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Real-Time Audio Narration")
    window = MainWindow(
        Settings(
            camera=args.camera,
            mode=args.mode,
            device=args.device,
            vlm_backend=args.vlm,
            quantize=args.quantize,
            audio=not args.no_audio,
            tts_backend=args.tts_backend,
            max_age=args.max_age,
        )
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
