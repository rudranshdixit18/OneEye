from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.config import settings
from backend.app.core.database import DEFAULT_THRESHOLDS, Database
from backend.app.services.camera import CameraManager
from backend.app.services.events import EventService
from backend.app.services.inference import InferenceEngine
from backend.app.services.pipeline import SurveillancePipeline

settings.ensure_directories()


@dataclass
class Services:
    database: Database
    cameras: CameraManager
    inference: InferenceEngine
    events: EventService
    pipeline: SurveillancePipeline


def build_services() -> Services:
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    persisted = database.get_settings()
    saved_thresholds = persisted.get("thresholds", {})
    thresholds = {
        event_type: max(default, float(saved_thresholds.get(event_type, default)))
        for event_type, default in DEFAULT_THRESHOLDS.items()
    }
    for event_type, threshold in thresholds.items():
        database.update_setting(f"threshold.{event_type}", threshold)
    cameras = CameraManager(
        settings.storage_dir / "clips",
        settings.camera_sources,
        capture_size=(settings.camera_width, settings.camera_height),
        capture_fps=settings.camera_fps,
    )
    inference = InferenceEngine(settings.models_dir, thresholds)
    inference.load()
    events = EventService(database, settings.storage_dir / "snapshots", settings.event_cooldown_seconds)
    pipeline = SurveillancePipeline(
        cameras,
        inference,
        events,
        inference_fps=float(persisted.get("inference_fps", settings.inference_fps)),
        confirmation_frames=int(persisted.get("confirmation_frames", settings.confirmation_frames)),
        stream_size=(
            int(persisted.get("stream_width", settings.stream_width)),
            int(persisted.get("stream_height", settings.stream_height)),
        ),
        stream_fps=float(persisted.get("stream_fps", settings.stream_fps)),
        jpeg_quality=int(persisted.get("jpeg_quality", settings.jpeg_quality)),
    )
    events.set_cooldown(int(persisted.get("event_cooldown_seconds", settings.event_cooldown_seconds)))
    return Services(database, cameras, inference, events, pipeline)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = build_services()
    yield
    app.state.services.pipeline.stop_all()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.storage_dir / "logs" / "oneeye.log", encoding="utf-8"),
    ],
)

app = FastAPI(
    title="OneEye Sentinel API",
    version="1.0.0",
    description="Five-model surveillance inference, event persistence, camera streaming, and analytics.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(router)
app.mount("/snapshots", StaticFiles(directory=settings.storage_dir / "snapshots"), name="snapshots")
app.mount("/clips", StaticFiles(directory=settings.storage_dir / "clips"), name="clips")
app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
