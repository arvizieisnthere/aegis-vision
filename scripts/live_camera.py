
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# Lokasi model
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / (
    "runs/detect/runs/aegisvision/"
    "training_main/weights/best.pt"
)

model = YOLO(str(MODEL_PATH))

# Mengaktifkan kamera MacBook
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Kamera tidak dapat diakses.")

print("AegisVision AI - Camera Online")
print("Tekan Q untuk menghentikan kamera.")

try:
    while True:
        start_time = time.perf_counter()

        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca kamera.")
            break

        # Deteksi wajah
        results = model.predict(
            source=frame,
            imgsz=416,
            conf=0.25,
            device="mps",
            verbose=False
        )

        # Bounding box dan jumlah wajah
        annotated_frame = results[0].plot()
        face_count = len(results[0].boxes)

        # Menghitung FPS pemrosesan
        elapsed = time.perf_counter() - start_time
        fps = 1 / max(elapsed, 0.000001)

        # Informasi kamera
        cv2.putText(
            annotated_frame,
            "AEGISVISION AI | LIVE",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            annotated_frame,
            f"Faces: {face_count}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.imshow("AegisVision AI", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("AegisVision AI - Camera Offline")
