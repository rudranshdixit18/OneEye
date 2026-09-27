from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal["fight", "garbage", "fallen_person", "mobile_phone", "accident"]


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Prediction(BaseModel):
    event_type: EventType
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox | None = None
    model_version: str


class EventRecord(BaseModel):
    id: str
    camera_id: str
    event_type: EventType
    label: str
    confidence: float
    started_at: datetime
    ended_at: datetime | None = None
    bbox: BoundingBox | None = None
    snapshot_url: str | None = None
    clip_url: str | None = None
    model_version: str
    acknowledged: bool = False


class CameraAction(BaseModel):
    source: str | int | None = None


class SettingsUpdate(BaseModel):
    thresholds: dict[EventType, float] | None = None
    confirmation_frames: int | None = Field(default=None, ge=1, le=30)
    event_cooldown_seconds: int | None = Field(default=None, ge=0, le=3600)
    inference_fps: float | None = Field(default=None, gt=0, le=30)
    stream_fps: float | None = Field(default=None, ge=1, le=60)
    stream_width: int | None = Field(default=None, ge=320, le=3840)
    stream_height: int | None = Field(default=None, ge=240, le=2160)
    jpeg_quality: int | None = Field(default=None, ge=60, le=100)
