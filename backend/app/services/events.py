from __future__ import annotations

import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.core.database import Database


class EventService:
    def __init__(self, database: Database, snapshots_dir: Path, cooldown_seconds: int):
        self.database = database
        self.snapshots_dir = snapshots_dir
        self.cooldown_seconds = cooldown_seconds
        self._last_emitted: dict[tuple[str, str], float] = {}
        self._lock = threading.RLock()

    def set_cooldown(self, seconds: int) -> None:
        self.cooldown_seconds = seconds

    def create(self, camera_id: str, prediction: dict[str, Any], frame: np.ndarray) -> dict[str, Any] | None:
        key = (camera_id, prediction["event_type"])
        now_monotonic = time.monotonic()
        with self._lock:
            if now_monotonic - self._last_emitted.get(key, -1e9) < self.cooldown_seconds:
                return None
            self._last_emitted[key] = now_monotonic

        event_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC)
        day_dir = self.snapshots_dir / timestamp.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        snapshot = day_dir / f"{timestamp.strftime('%H%M%S_%f')}_{camera_id}_{prediction['event_type']}.jpg"
        cv2.imwrite(str(snapshot), frame)
        relative_snapshot = snapshot.relative_to(self.snapshots_dir).as_posix()
        record = {
            "id": event_id,
            "camera_id": camera_id,
            "event_type": prediction["event_type"],
            "label": prediction["label"],
            "confidence": float(prediction["confidence"]),
            "started_at": timestamp.isoformat(),
            "ended_at": None,
            "bbox": prediction.get("bbox"),
            "snapshot_path": relative_snapshot,
            "clip_path": None,
            "model_version": prediction["model_version"],
            "acknowledged": False,
        }
        self.database.insert_event(record)
        return record
