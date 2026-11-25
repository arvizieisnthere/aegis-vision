
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    StreamingResponse
)
from pydantic import BaseModel

from web.camera import camera_manager


ROOT = Path(__file__).resolve().parent


class StartRequest(BaseModel):
    source_type: str
    source: str | int


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    camera_manager.stop_all()


app = FastAPI(
    title="AegisVision Multi-Camera",
    lifespan=lifespan
)


@app.get("/")
def homepage():
    return FileResponse(
        ROOT / "templates/index.html"
    )


@app.get("/api/cameras")
def list_devices():
    return {
        "devices": camera_manager.list_cameras()
    }


@app.get("/api/status")
def status():
    return {
        "cameras": camera_manager.statuses()
    }


@app.post("/api/cameras/{slot}/start")
def start_camera(slot: int, request: StartRequest):
    try:
        return camera_manager.start(
            slot,
            request.source_type,
            request.source
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


@app.post("/api/cameras/{slot}/stop")
def stop_camera(slot: int):
    try:
        return camera_manager.stop(slot)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


async def stream_frames(slot: int):
    worker = camera_manager.get(slot)

    while worker.running:
        frame = worker.get_frame()

        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame +
                b"\r\n"
            )

        await asyncio.sleep(0.04)


@app.get("/api/cameras/{slot}/stream")
async def video_stream(slot: int):
    try:
        camera_manager.get(slot)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Invalid camera slot"
        )

    return StreamingResponse(
        stream_frames(slot),
        media_type=(
            "multipart/x-mixed-replace; boundary=frame"
        )
    )
