# SPDX-License-Identifier: AGPL-3.0-only
import time
from dataclasses import replace
from threading import Thread

import cv2
import psutil
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from audio_narration.camera.capture import discover_cameras
from audio_narration.pipeline.orchestrator import Pipeline


class Bridge(QObject):
    message = Signal(str, str)
    cameras = Signal(list)


class MainWindow(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.settings, self.pipeline = settings, None
        self.closing, self.stopping, self.scanning = False, False, False
        self.last_sequence = -1
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.last_resource = 0.0
        self.bridge = Bridge()
        self.bridge.message.connect(self.on_message)
        self.bridge.cameras.connect(self.on_cameras)
        self.setWindowTitle("Real-Time Audio Narration")
        self.resize(1200, 830)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("Real-Time Audio Narration")
        title.setStyleSheet("font-size:26px; font-weight:600; color:#91dfdc")
        layout.addWidget(title)
        layout.addWidget(QLabel("Local scene understanding · No footage uploaded or recorded"))
        toolbar = QHBoxLayout()
        self.start = QPushButton("&Start camera")
        self.stop = QPushButton("S&top all")
        self.stop.setEnabled(False)
        self.pause = QPushButton("&Pause interpretation")
        self.pause.setCheckable(True)
        self.pause.setEnabled(False)
        self.narration = QCheckBox("&Narration enabled")
        self.narration.setChecked(True)
        self.audio = QCheckBox("&Audio enabled")
        self.audio.setChecked(settings.audio)
        self.overlays = QCheckBox("Object &overlays")
        self.overlays.setChecked(True)
        for widget in (
            self.start,
            self.stop,
            self.pause,
            self.narration,
            self.audio,
            self.overlays,
        ):
            toolbar.addWidget(widget)
        layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.preview = QLabel("CAMERA OFF\n\nSelect a camera and press Start camera.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(500, 360)
        self.preview.setStyleSheet(
            "background:#0a111b; border:1px solid #344254; border-radius:8px"
        )
        self.preview.setAccessibleName("Live webcam preview")
        left_layout.addWidget(self.preview, 1)
        self.camera_status = QLabel("Camera off")
        left_layout.addWidget(self.camera_status)
        left_layout.addWidget(QLabel("Current scene"))
        self.description = QPlainTextEdit()
        self.description.setReadOnly(True)
        self.description.setMaximumHeight(120)
        self.description.setPlaceholderText("A description appears after local model inference.")
        self.description.setAccessibleName("Current scene description")
        left_layout.addWidget(self.description)
        splitter.addWidget(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        config_group = QGroupBox("Session settings")
        form = QFormLayout(config_group)
        self.camera = QComboBox()
        self.camera.setEditable(True)
        self.camera.addItem(str(settings.camera))
        self.scan = QPushButton("Find cameras")
        camera_row = QWidget()
        row = QHBoxLayout(camera_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.camera)
        row.addWidget(self.scan)
        form.addRow("&Camera index", camera_row)
        self.mode = QComboBox()
        self.mode.addItems(["hybrid", "baseline"])
        self.mode.setCurrentText(settings.mode)
        form.addRow("&Inference mode", self.mode)
        self.scene_model = QComboBox()
        self.scene_model.addItems(["smol", "qwen"])
        self.scene_model.setCurrentText(settings.vlm_backend)
        form.addRow("Scene model (smol = fast)", self.scene_model)
        self.detail = QComboBox()
        self.detail.addItems(["brief", "standard", "detailed"])
        self.detail.setCurrentText(settings.detail)
        form.addRow("&Description detail", self.detail)
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.25, 120)
        self.interval.setValue(settings.min_interval)
        self.interval.setSuffix(" s minimum")
        form.addRow("Analysis spacing", self.interval)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.95)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(settings.confidence)
        form.addRow("Detection confidence", self.threshold)
        self.rate = QDoubleSpinBox()
        self.rate.setRange(0.5, 2)
        self.rate.setValue(settings.speech_rate)
        form.addRow("Speech speed", self.rate)
        self.volume = QDoubleSpinBox()
        self.volume.setRange(0, 1)
        self.volume.setValue(settings.volume)
        form.addRow("Volume", self.volume)
        self.backend = QComboBox()
        self.backend.addItems(["kokoro", "system"])
        self.backend.setCurrentText(settings.tts_backend)
        form.addRow("Speech backend", self.backend)
        right_layout.addWidget(config_group)
        self.session_controls = [
            self.camera,
            self.scan,
            self.mode,
            self.scene_model,
            self.detail,
            self.interval,
            self.threshold,
            self.rate,
            self.volume,
            self.backend,
        ]
        self.status = QLabel("Scene: idle\nAudio: idle")
        self.status.setWordWrap(True)
        right_layout.addWidget(self.status)
        self.status_values = {"scene": "idle", "audio": "idle"}
        self.diagnostics = QLabel("Performance metrics appear during capture.")
        self.diagnostics.setWordWrap(True)
        right_layout.addWidget(self.diagnostics)
        right_layout.addWidget(QLabel("Recent events and errors"))
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setMaximumBlockCount(100)
        self.history.setAccessibleName("Recent narration history and errors")
        right_layout.addWidget(self.history, 1)
        splitter.addWidget(right)
        splitter.setSizes([750, 450])
        footer = QLabel(
            "Descriptions may be inaccurate. This prototype is not a navigation "
            "or hazard-detection aid. Escape stops all processing."
        )
        footer.setWordWrap(True)
        layout.addWidget(footer)
        self.setStyleSheet("""
            QWidget { background:#16202e; color:#edf2f7; font-size:14px; }
            QPushButton, QComboBox, QDoubleSpinBox { padding:7px; background:#29384b;
                border:1px solid #596d83; border-radius:5px; }
            QPushButton:focus, QComboBox:focus, QDoubleSpinBox:focus { border:2px solid #91dfdc; }
            QPushButton:disabled { color:#8191a3; }
            QPlainTextEdit { background:#0e1825; border:1px solid #344254; padding:7px; }
            QGroupBox { border:1px solid #344254; margin-top:14px; padding-top:14px; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; }
        """)
        self.start.clicked.connect(self.start_session)
        self.stop.clicked.connect(self.stop_session)
        self.scan.clicked.connect(self.scan_cameras)
        self.pause.toggled.connect(self.pause_session)
        self.narration.toggled.connect(
            lambda v: self.pipeline.set_narration(v) if self.pipeline else None
        )
        self.audio.toggled.connect(lambda v: self.pipeline.set_audio(v) if self.pipeline else None)
        self.escape = QShortcut(QKeySequence("Esc"), self)
        self.escape.activated.connect(self.stop_session)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

    def scan_cameras(self):
        self.scan.setEnabled(False)
        self.start.setEnabled(False)
        self.scanning = True
        self.camera_status.setText("Checking camera availability…")

        def scan():
            try:
                cameras = discover_cameras()
            except Exception as exc:
                self.bridge.message.emit("error", str(exc))
                cameras = []
            self.bridge.cameras.emit(cameras)

        Thread(target=scan, daemon=True).start()

    def on_cameras(self, cameras):
        self.scanning = False
        self.camera.clear()
        self.camera.addItems([str(i) for i in cameras] or ["0"])
        self.scan.setEnabled(True)
        self.start.setEnabled(True)
        self.camera_status.setText(
            "Camera off — "
            + (f"found {cameras}" if cameras else "none found; enter an index to retry")
        )

    def start_session(self):
        if self.pipeline:
            return
        try:
            settings = replace(
                self.settings,
                camera=int(self.camera.currentText()),
                mode=self.mode.currentText(),
                vlm_backend=self.scene_model.currentText(),
                detail=self.detail.currentText(),
                min_interval=self.interval.value(),
                refresh_interval=max(self.interval.value(), 2),
                confidence=self.threshold.value(),
                audio=self.audio.isChecked(),
                speech_rate=self.rate.value(),
                volume=self.volume.value(),
                tts_backend=self.backend.currentText(),
            )
            self.pipeline = Pipeline(settings, self.bridge.message.emit)
            self.pipeline.set_narration(self.narration.isChecked())
            self.pipeline.start()
        except Exception as exc:
            self.on_message("error", str(exc))
            self.pipeline = None
            return
        self.stopping = False
        self.start.setEnabled(False)
        self.stop.setEnabled(True)
        self.pause.setEnabled(True)
        for control in self.session_controls:
            control.setEnabled(False)
        self.camera_status.setText("Opening camera…")

    def stop_session(self):
        if self.pipeline:
            self.pipeline.request_stop()
            self.stopping = True
            self.stop.setEnabled(False)
            self.pause.setEnabled(False)
            self.camera_status.setText("Stopping workers and releasing camera…")

    def pause_session(self, value):
        if self.pipeline:
            self.pipeline.set_paused(value)
        self.pause.setText("&Resume interpretation" if value else "&Pause interpretation")

    def on_message(self, kind, text):
        if kind == "description":
            self.description.setPlainText(text)
        elif kind == "camera":
            self.camera_status.setText(text)
        elif kind in {"scene", "audio", "hardware", "detection"}:
            self.status_values[kind] = text
            self.status.setText(
                "\n".join(f"{k.title()}: {v}" for k, v in self.status_values.items())
            )
        elif kind in {"event", "error", "spoken"}:
            self.history.appendPlainText(f"{time.strftime('%H:%M:%S')} [{kind}] {text}")
            if kind == "error":
                self.status_values["last error"] = text
                self.status.setText(
                    "\n".join(f"{k.title()}: {v}" for k, v in self.status_values.items())
                )

    def tick(self):
        if not self.pipeline:
            return
        if self.pipeline.stop_event.is_set():
            self.stopping = True
        if self.stopping:
            if all(not thread.is_alive() for thread in self.pipeline.threads):
                self.pipeline.join(0)
                self.pipeline = None
                self.last_sequence = -1
                self.preview.clear()
                self.preview.setText("CAMERA OFF")
                self.camera_status.setText("Camera off — all workers stopped")
                self.start.setEnabled(True)
                self.stop.setEnabled(False)
                self.pause.setChecked(False)
                for control in self.session_controls:
                    control.setEnabled(True)
                if self.closing:
                    self.close()
            return
        frame, batch, metrics = self.pipeline.snapshot()
        if frame and frame.sequence != self.last_sequence:
            self.last_sequence = frame.sequence
            image = frame.image.copy()
            if batch and self.overlays.isChecked() and frame.timestamp - batch.frame.timestamp < 1:
                h, w = image.shape[:2]
                for obj in batch.objects:
                    x1, y1, x2, y2 = [int(v * d) for v, d in zip(obj.box, (w, h, w, h))]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (130, 230, 190), 2)
                    cv2.putText(
                        image,
                        f"{obj.label} #{obj.track_id} {obj.confidence:.0%}",
                        (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (130, 230, 190),
                        1,
                    )
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(
                rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
            ).copy()
            self.preview.setPixmap(
                QPixmap.fromImage(qimage).scaled(
                    self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        if time.monotonic() - self.last_resource >= 1:
            self.last_resource = time.monotonic()
            metrics["process_cpu_percent"] = self.process.cpu_percent()
            metrics["memory_mb"] = self.process.memory_info().rss / 1024**2
            text = "\n".join(
                f"{key.replace('_', ' ')}: {value:.1f}" for key, value in metrics.items()
            )
            objects = ", ".join(sorted({o.label for o in batch.objects})) if batch else "—"
            self.diagnostics.setText(text + "\nObjects: " + objects)

    def closeEvent(self, event):
        if self.pipeline:
            self.closing = True
            self.stop_session()
            event.ignore()
        else:
            event.accept()
