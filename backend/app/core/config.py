from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _csv_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _camera_sources() -> dict[str, str]:
    raw = os.getenv("ONEEYE_CAMERA_SOURCES", "")
    if raw:
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                return {str(k): str(v) for k, v in value.items()}
        except json.JSONDecodeError:
            pass
    source = os.getenv("CAMERA_STREAM_URL", "0")
    return {"cam1": source, "cam2": "", "cam3": "", "cam4": ""}


@dataclass(slots=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    frontend_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "FrontEnd")
    storage_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    database_path: Path = field(default_factory=lambda: PROJECT_ROOT / "storage" / "database" / "oneeye.db")
    host: str = field(default_factory=lambda: os.getenv("ONEEYE_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("ONEEYE_PORT", "8000")))
    cors_origins: list[str] = field(
        default_factory=lambda: _csv_env(
            "ONEEYE_CORS_ORIGINS",
            "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5500,http://localhost:5500",
        )
    )
    camera_sources: dict[str, str] = field(default_factory=_camera_sources)
    camera_width: int = field(default_factory=lambda: int(os.getenv("ONEEYE_CAMERA_WIDTH", "1920")))
    camera_height: int = field(default_factory=lambda: int(os.getenv("ONEEYE_CAMERA_HEIGHT", "1080")))
    camera_fps: float = field(default_factory=lambda: float(os.getenv("ONEEYE_CAMERA_FPS", "60")))
    # Four analyzed frames per second keeps the four-tile temporal window close
    # to the 0.75-second span used by fight/accident training while the live
    # camera stream remains independently capable of 60 FPS.
    inference_fps: float = field(default_factory=lambda: float(os.getenv("ONEEYE_INFERENCE_FPS", "4")))
    stream_fps: float = field(default_factory=lambda: float(os.getenv("ONEEYE_STREAM_FPS", "60")))
    stream_width: int = field(default_factory=lambda: int(os.getenv("ONEEYE_STREAM_WIDTH", "1920")))
    stream_height: int = field(default_factory=lambda: int(os.getenv("ONEEYE_STREAM_HEIGHT", "1080")))
    jpeg_quality: int = field(default_factory=lambda: int(os.getenv("ONEEYE_JPEG_QUALITY", "92")))
    event_cooldown_seconds: int = field(default_factory=lambda: int(os.getenv("ONEEYE_EVENT_COOLDOWN_SECONDS", "20")))
    confirmation_frames: int = field(default_factory=lambda: int(os.getenv("ONEEYE_CONFIRMATION_FRAMES", "2")))

    def ensure_directories(self) -> None:
        for path in (
            self.storage_dir / "database",
            self.storage_dir / "events",
            self.storage_dir / "snapshots",
            self.storage_dir / "clips",
            self.storage_dir / "logs",
            self.models_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
