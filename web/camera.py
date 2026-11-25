
import cv2
import time
import threading
import sys
import torch
from pathlib import Path
from urllib.parse import urlparse

from ultralytics import YOLO
from cv2_enumerate_cameras import enumerate_cameras


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT / "models" / "best.pt"
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

if sys.platform == "darwin":
    CAMERA_BACKEND = cv2.CAP_AVFOUNDATION
elif sys.platform == "win32":
    CAMERA_BACKEND = cv2.CAP_DSHOW
else:
    CAMERA_BACKEND = cv2.CAP_V4L2

print(f"AegisVision inference device: {DEVICE}")


class CameraWorker:
    def __init__(self, slot, manager):
        self.slot = slot
        self.manager = manager
        self.capture = None
        self.thread = None
        self.running = False
        self.source_key = None
        self.source_name = None
        self.error = None
        self.frame = None
        self.faces = 0
        self.fps = 0.0
        self.lock = threading.Lock()

    def start(self, source_type, source):
        if self.running or (
            self.thread and self.thread.is_alive()
        ):
            raise RuntimeError("Camera slot is busy")

        if source_type == "local":
            index = int(source)
            devices = self.manager.list_cameras()
            selected = next(
                (d for d in devices if d["index"] == index),
                None
            )
            if selected is None:
                raise ValueError("Camera device not found")

            key = f"local:{index}"
            name = selected["name"]
            capture = cv2.VideoCapture(
                index, CAMERA_BACKEND
            )

        elif source_type == "network":
            if not isinstance(source, str):
                raise ValueError("Invalid stream URL")

            source = source.strip()
            parsed = urlparse(source)
            if (
                parsed.scheme.lower()
                not in ("http", "https", "rtsp", "rtsps")
                or not parsed.hostname
            ):
                raise ValueError("Invalid network camera URL")

            key = f"network:{source}"
            name = "Network Camera"
            capture = cv2.VideoCapture(source)

        else:
            raise ValueError("Unknown source type")

        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Unable to open camera")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        with self.lock:
            self.capture = capture
            self.source_key = key
            self.source_name = name
            self.frame = None
            self.faces = 0
            self.fps = 0.0
            self.error = None
            self.running = True

        self.thread = threading.Thread(
            target=self._process,
            daemon=True
        )
        self.thread.start()

    def _process(self):
        previous = time.perf_counter()

        try:
            while self.running:
                ok, image = self.capture.read()
                if not ok:
                    raise RuntimeError(
                        "Camera stream disconnected"
                    )

                # Satu model dipakai bersama.
                # Inferensi tidak dijalankan bersamaan.
                with self.manager.inference_lock:
                    prediction = self.manager.model.predict(
                        source=image,
                        imgsz=416,
                        conf=0.25,
                        device=DEVICE,
                        verbose=False
                    )

                annotated = prediction[0].plot()
                face_count = len(prediction[0].boxes)

                now = time.perf_counter()
                fps = 1 / max(now - previous, 0.000001)
                previous = now

                ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if not ok:
                    continue

                with self.lock:
                    self.frame = encoded.tobytes()
                    self.faces = face_count
                    self.fps = round(fps, 1)

        except Exception as exc:
            with self.lock:
                self.error = str(exc)

        finally:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

            with self.lock:
                self.running = False
                self.frame = None
                self.faces = 0
                self.fps = 0.0

    def stop(self):
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        if self.thread and self.thread.is_alive():
            raise RuntimeError(
                "Camera has not stopped yet"
            )

        with self.lock:
            self.frame = None
            self.faces = 0
            self.fps = 0.0
            self.source_key = None
            self.source_name = None

    def get_frame(self):
        with self.lock:
            return self.frame

    def status(self):
        with self.lock:
            return {
                "slot": self.slot,
                "online": self.running,
                "source": self.source_name,
                "faces": self.faces,
                "fps": self.fps,
                "error": self.error
            }


class CameraManager:
    def __init__(self):
        self.model = None
        self.inference_lock = threading.Lock()
        self.control_lock = threading.RLock()
        self.cameras = {
            1: CameraWorker(1, self),
            2: CameraWorker(2, self)
        }

    def list_cameras(self):
        return [
            {
                "index": info.index,
                "name": info.name
            }
            for info in enumerate_cameras(
                CAMERA_BACKEND
            )
        ]

    def get(self, slot):
        if slot not in self.cameras:
            raise ValueError("Invalid camera slot")
        return self.cameras[slot]

    def start(self, slot, source_type, source):
        with self.control_lock:
            worker = self.get(slot)

            key = (
                f"local:{int(source)}"
                if source_type == "local"
                else f"network:{str(source).strip()}"
            )

            for other_slot, other in self.cameras.items():
                if (
                    other_slot != slot
                    and other.running
                    and other.source_key == key
                ):
                    raise ValueError(
                        "Source already used by another camera"
                    )

            if self.model is None:
                self.model = YOLO(str(MODEL_PATH))

            worker.start(source_type, source)
            return worker.status()

    def stop(self, slot):
        with self.control_lock:
            worker = self.get(slot)
            worker.stop()
            return worker.status()

    def stop_all(self):
        for slot in self.cameras:
            try:
                self.stop(slot)
            except RuntimeError:
                pass

    def statuses(self):
        return [
            worker.status()
            for worker in self.cameras.values()
        ]


camera_manager = CameraManager()
