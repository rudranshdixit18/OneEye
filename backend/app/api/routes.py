from __future__ import annotations

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from backend.app.schemas import CameraAction, SettingsUpdate
from backend.app.services.inference import decode_image, draw_predictions

router = APIRouter(prefix="/api/v1")
EVENT_TYPES = {"fight", "garbage", "fallen_person", "mobile_phone", "accident"}


def _public_event(event: dict) -> dict:
    value = event.copy()
    snapshot = value.pop("snapshot_path", None)
    clip = value.pop("clip_path", None)
    value["snapshot_url"] = f"/snapshots/{snapshot}" if snapshot else None
    value["clip_url"] = f"/clips/{clip}" if clip else None
    return value


@router.get("/health")
def health(request: Request) -> dict:
    services = request.app.state.services
    model_status = services.inference.statuses()
    loaded = sum(1 for item in model_status.values() if item.get("loaded"))
    return {
        "status": "ready" if loaded == len(model_status) else "degraded",
        "models_loaded": loaded,
        "models_expected": len(model_status),
        "models": model_status,
        "cameras": services.cameras.statuses(),
        "pipeline": services.pipeline.stats(),
        "version": "1.0.0",
    }


@router.get("/models")
def models(request: Request) -> dict:
    return request.app.state.services.inference.statuses()


@router.post("/models/reload")
def reload_models(request: Request) -> dict:
    request.app.state.services.inference.load()
    return request.app.state.services.inference.statuses()


@router.get("/cameras")
def cameras(request: Request) -> list[dict]:
    return request.app.state.services.cameras.statuses()


@router.post("/cameras/{camera_id}/start")
def start_camera(camera_id: str, request: Request, action: CameraAction | None = None) -> dict:
    try:
        request.app.state.services.pipeline.start(camera_id, action.source if action else None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return request.app.state.services.cameras.get(camera_id).status()


@router.post("/cameras/{camera_id}/stop")
def stop_camera(camera_id: str, request: Request) -> dict:
    request.app.state.services.pipeline.stop(camera_id)
    return {"camera_id": camera_id, "stopped": True}


@router.post("/cameras/{camera_id}/recording/start")
def start_recording(camera_id: str, request: Request) -> dict:
    worker = request.app.state.services.cameras.get(camera_id)
    if not worker:
        raise HTTPException(status_code=409, detail="Camera is not started")
    worker.start_recording()
    return worker.status()


@router.post("/cameras/{camera_id}/recording/stop")
def stop_recording(camera_id: str, request: Request) -> dict:
    worker = request.app.state.services.cameras.get(camera_id)
    if not worker:
        raise HTTPException(status_code=409, detail="Camera is not started")
    path = worker.stop_recording()
    return {"camera_id": camera_id, "recording": False, "saved_path": path}


@router.get("/cameras/{camera_id}/stream")
def camera_stream(camera_id: str, request: Request) -> StreamingResponse:
    if not request.app.state.services.cameras.get(camera_id):
        raise HTTPException(status_code=409, detail="Camera is not started")
    return StreamingResponse(
        request.app.state.services.pipeline.stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/cameras/{camera_id}/snapshot")
def camera_snapshot(camera_id: str, request: Request) -> Response:
    worker = request.app.state.services.cameras.get(camera_id)
    if not worker:
        raise HTTPException(status_code=409, detail="Camera is not started")
    snapshot = worker.snapshot()
    if snapshot.frame is None:
        raise HTTPException(status_code=503, detail="No frame is available")
    annotated = draw_predictions(snapshot.frame, request.app.state.services.pipeline.latest_predictions(camera_id))
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Snapshot encoding failed")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@router.get("/events")
def list_events(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    event_type: str | None = None,
    search: str | None = None,
    hours: int | None = Query(default=None, ge=1, le=24 * 365),
) -> list[dict]:
    if event_type and event_type not in EVENT_TYPES:
        raise HTTPException(status_code=422, detail=f"Unknown event type: {event_type}")
    rows = request.app.state.services.database.list_events(limit, offset, event_type, search, hours)
    return [_public_event(row) for row in rows]


@router.patch("/events/{event_id}/acknowledge")
def acknowledge_event(event_id: str, request: Request) -> dict:
    if not request.app.state.services.database.acknowledge_event(event_id):
        raise HTTPException(status_code=404, detail="Event not found")
    return {"id": event_id, "acknowledged": True}


@router.get("/analytics/summary")
def analytics(request: Request, hours: int = Query(default=24, ge=1, le=24 * 365)) -> dict:
    return request.app.state.services.database.analytics(hours)


@router.get("/settings")
def get_settings(request: Request) -> dict:
    services = request.app.state.services
    value = services.database.get_settings()
    value.update(
        {
            "confirmation_frames": services.pipeline.confirmation_frames,
            "event_cooldown_seconds": services.events.cooldown_seconds,
            "inference_fps": services.pipeline.inference_fps,
            "stream_fps": services.pipeline.stream_fps,
            "stream_width": services.pipeline.stream_size[0],
            "stream_height": services.pipeline.stream_size[1],
            "jpeg_quality": services.pipeline.jpeg_quality,
        }
    )
    return value


@router.put("/settings")
def update_settings(update: SettingsUpdate, request: Request) -> dict:
    services = request.app.state.services
    if update.thresholds:
        for event_type, threshold in update.thresholds.items():
            if not 0.91 <= threshold <= 0.99:
                raise HTTPException(
                    status_code=422,
                    detail=f"Threshold for {event_type} must be between 0.91 and 0.99",
                )
            services.database.update_setting(f"threshold.{event_type}", threshold)
        services.inference.set_thresholds(update.thresholds)
    if update.confirmation_frames is not None:
        services.pipeline.confirmation_frames = update.confirmation_frames
        services.database.update_setting("confirmation_frames", update.confirmation_frames)
    if update.event_cooldown_seconds is not None:
        services.events.set_cooldown(update.event_cooldown_seconds)
        services.database.update_setting("event_cooldown_seconds", update.event_cooldown_seconds)
    if update.inference_fps is not None:
        services.pipeline.inference_fps = update.inference_fps
        services.database.update_setting("inference_fps", update.inference_fps)
    stream_width = update.stream_width or services.pipeline.stream_size[0]
    stream_height = update.stream_height or services.pipeline.stream_size[1]
    if update.stream_width is not None or update.stream_height is not None:
        services.pipeline.stream_size = (stream_width, stream_height)
        services.database.update_setting("stream_width", stream_width)
        services.database.update_setting("stream_height", stream_height)
    if update.stream_fps is not None:
        services.pipeline.stream_fps = update.stream_fps
        services.database.update_setting("stream_fps", update.stream_fps)
    if update.jpeg_quality is not None:
        services.pipeline.jpeg_quality = update.jpeg_quality
        services.database.update_setting("jpeg_quality", update.jpeg_quality)
    return get_settings(request)


@router.post("/infer")
async def infer_image(
    request: Request,
    file: UploadFile = File(...),
    event_types: str | None = Form(default=None),
) -> dict:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB limit")
    try:
        frame = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    selected = None
    if event_types:
        selected = {value.strip() for value in event_types.split(",") if value.strip()}
        unknown = selected - EVENT_TYPES
        if unknown:
            raise HTTPException(status_code=422, detail=f"Unknown event types: {sorted(unknown)}")
    predictions = request.app.state.services.inference.predict(frame, selected)
    return {"filename": file.filename, "predictions": predictions}
