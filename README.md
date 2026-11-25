# AegisVision

**Real-Time Multi-Camera Face Detection with YOLOv8**

AegisVision is a web-based surveillance prototype that detects human faces from live camera feeds using a custom-trained YOLOv8n model. It combines **Ultralytics YOLOv8**, **OpenCV**, and **FastAPI** in a lightweight monitoring dashboard.

The model was trained using the **UFDD (Unconstrained Face Detection Dataset)** to explore face detection under challenging visual conditions, including rain, snow, haze, low or uneven illumination, and blur.

> **Project status:** Working prototype for local live monitoring. Camera recording, time-range playback, and MP4 export are planned features and are **not yet implemented**.

## Features

- Live face detection with bounding boxes and prediction confidence.
- Two independent camera slots (**CAM 01** and **CAM 02**), with individual Start/Stop controls.
- Select available local cameras, such as a built-in webcam, a USB webcam, or an iPhone connected through macOS Continuity Camera.
- Connect a compatible network camera via a reachable RTSP or HTTP video-stream URL.
- Display detected face count, approximate processing FPS, and camera connection status for each slot.
- Automatic inference-device selection: **CUDA**, **Apple MPS**, or **CPU**, depending on the installed PyTorch build and hardware.
- Run locally in a browser using FastAPI.

**Note:** This is **face detection**, not face recognition. It does not identify people. The two camera feeds share one YOLO model; actual FPS depends on the hardware, source resolution, and number of active feeds.

## Technology

| Component | Technology |
| --- | --- |
| Face detection | YOLOv8n / Ultralytics |
| Deep learning | PyTorch |
| Video processing | OpenCV |
| Camera discovery | `cv2-enumerate-cameras` |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Model weights | `models/best.pt` |

## Repository structure

```text
AegisVision/
├── models/
│   └── best.pt                 # Trained face detection weights
├── web/
│   ├── camera.py               # Camera manager and inference
│   ├── main.py                 # FastAPI endpoints
│   └── templates/
│       └── index.html          # Monitoring dashboard
├── scripts/
│   ├── convert_ufdd.py        # Dataset conversion (development)
│   └── live_camera.py         # Earlier local camera test
├── check_empty_labels.py      # Dataset audit utility
├── requirements.txt
└── .gitignore
```

The original UFDD dataset, generated YOLO dataset, training runs, and local recordings are not included in this repository. **The provided `models/best.pt` file is sufficient for inference.** The scripts under `scripts/` are development utilities; use the web application for the portable demo.

## Requirements

- Python **3.12** (recommended for this prototype)
- Git
- An accessible webcam or compatible network camera
- Internet connection for the **initial package installation**; the local webcam demo does not require a cloud inference service

> On Windows without a supported NVIDIA CUDA setup, the application can run with CPU inference, but FPS may be lower. PyTorch installation behavior varies by operating system and available hardware.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/arvizieisnthere/aegis-vision.git
cd aegis-vision
```

### 2A. Set up on macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.12` is unavailable, install Python 3.12 or use the matching Python interpreter on your machine.

### 2B. Set up on Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using `.venv\Scripts\python.exe` directly avoids PowerShell activation-policy issues.

### 3. Start the web application

On **macOS / Linux** (with the virtual environment activated):

```bash
python -m uvicorn web.main:app --host 127.0.0.1 --port 8000
```

On **Windows PowerShell**:

```powershell
.\.venv\Scripts\python.exe -m uvicorn web.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in a browser.

**Important:** Run the command from the repository root (the directory containing `web/` and `models/`). Do not launch two instances that attempt to use the same camera or port.

## How to use

1. Open the dashboard and select **Refresh Devices** to detect local cameras.
2. For **CAM 01**, choose **Local** and select a webcam, then click **Start Camera**.
3. If available, choose a **different** webcam for **CAM 02** and start it separately.
4. Alternatively, select **Network** and enter the *actual* URL exposed by a compatible IP camera or phone streaming application (for example, an RTSP or HTTP/MJPEG stream).
5. Observe face bounding boxes, face counts, FPS, and online/offline status.
6. Click **Stop** to release a camera slot before selecting a different source.

Only one camera can use a given local source at a time. Both slots can run simultaneously when separate compatible sources are available.

### Using a phone as a camera

- **iPhone + Mac:** macOS Continuity Camera may expose the iPhone as a local webcam. Select it from the detected device list.
- **Android or iPhone over Wi-Fi:** use a camera streaming application capable of publishing a compatible **RTSP** or **HTTP/MJPEG** stream, and enter its reachable stream URL under **Network**. The phone and computer must have network connectivity to one another.

A normal phone camera app does not automatically provide a stream URL. Do not use an example URL unless it matches the actual streaming application and network setup.

## Model and evaluation

- **Architecture:** YOLOv8n, one detection class (`face`)
- **Training data:** UFDD, with a custom 70/20/10 train/validation/test split
- **Training configuration:** 20 epochs, image size 416, batch size 2, random seed 42
- **Test set:** 644 images, 1,088 annotated face instances

| Test metric | Result |
| --- | ---: |
| Precision | 72.3% |
| Recall | 52.4% |
| mAP@0.50 | 57.6% |
| mAP@0.50:0.95 | 25.9% |

The split was generated from the available source annotations; further source-level leakage and duplicate checks are needed before treating these results as independent benchmark performance. The model can miss faces, especially in difficult conditions. **Do not use this prototype as the sole basis for security, identity verification, access control, or emergency decisions.** It has not been validated for detecting faces hidden by dense smoke.

## Roadmap

- [x] UFDD dataset preparation and YOLO training
- [x] Static-image and recorded-video detection tests
- [x] Local webcam inference
- [x] Two-camera web dashboard
- [x] Local and compatible network camera source selection
- [ ] Recording and file storage per camera
- [ ] Recording metadata in SQLite
- [ ] Search recordings by date and exact time range
- [ ] Browser playback of archived footage
- [ ] Export selected recording ranges to MP4
- [ ] Authentication and access controls for deployment

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Install `requirements.txt` in the active/new virtual environment. |
| `models/best.pt` not found | Confirm the file exists and run the server from the repository root. |
| Camera missing | Grant camera permissions, connect the webcam, close apps using it, and click **Refresh Devices**. |
| iPhone opens instead of Mac webcam | Select the desired device name instead of assuming camera index `0` is the built-in webcam. |
| Network camera not connecting | Verify the actual stream URL, protocol, credentials, network access, and OpenCV codec support. |
| FPS is low | Try one camera first; CPU inference and network streams may be slower. |
| Port 8000 already in use | Stop the earlier server instance, or run on another local port. |

## Privacy and scope

AegisVision processes video locally in the current prototype. Only use cameras and footage you are authorized to access. Network-stream URLs may contain credentials, so do not publish them in your repository or logs. The existing API has no user authentication; **keep the demo bound to `127.0.0.1` and do not expose it directly to the public internet**.

## Repository

[github.com/arvizieisnthere/aegis-vision](https://github.com/arvizieisnthere/aegis-vision)
