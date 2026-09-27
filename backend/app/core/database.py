from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLDS = {
    "fight": 0.91,
    "garbage": 0.91,
    "fallen_person": 0.91,
    "mobile_phone": 0.91,
    "accident": 0.91,
}


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        with self._write_lock, self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN (
                        'fight', 'garbage', 'fallen_person', 'mobile_phone', 'accident'
                    )),
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    bbox_json TEXT,
                    snapshot_path TEXT,
                    clip_path TEXT,
                    model_version TEXT NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0 CHECK(acknowledged IN (0, 1)),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_started_at ON events(started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events(camera_id, started_at DESC);

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            for event_type, threshold in DEFAULT_THRESHOLDS.items():
                connection.execute(
                    "INSERT OR IGNORE INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)",
                    (f"threshold.{event_type}", json.dumps(threshold), self._now()),
                )
            connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def insert_event(self, event: dict[str, Any]) -> None:
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO events(
                    id, camera_id, event_type, label, confidence, started_at, ended_at,
                    bbox_json, snapshot_path, clip_path, model_version, acknowledged, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["id"],
                    event["camera_id"],
                    event["event_type"],
                    event["label"],
                    event["confidence"],
                    event["started_at"],
                    event.get("ended_at"),
                    json.dumps(event.get("bbox")) if event.get("bbox") else None,
                    event.get("snapshot_path"),
                    event.get("clip_path"),
                    event["model_version"],
                    int(event.get("acknowledged", False)),
                    self._now(),
                ),
            )
            connection.commit()

    @staticmethod
    def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        bbox_json = value.pop("bbox_json", None)
        value["bbox"] = json.loads(bbox_json) if bbox_json else None
        value["acknowledged"] = bool(value["acknowledged"])
        value.pop("created_at", None)
        return value

    def list_events(
        self,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        search: str | None = None,
        hours: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if event_type:
            clauses.append("event_type = ?")
            values.append(event_type)
        if search:
            clauses.append("(LOWER(label) LIKE ? OR LOWER(camera_id) LIKE ? OR LOWER(event_type) LIKE ?)")
            needle = f"%{search.lower()}%"
            values.extend([needle, needle, needle])
        if hours:
            cutoff = datetime.now(UTC) - timedelta(hours=hours)
            clauses.append("started_at >= ?")
            values.append(cutoff.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.extend([min(max(limit, 1), 500), max(offset, 0)])
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM events {where} ORDER BY started_at DESC LIMIT ? OFFSET ?",
                values,
            ).fetchall()
        return [self._event_dict(row) for row in rows]

    def acknowledge_event(self, event_id: str) -> bool:
        with self._write_lock, self.connect() as connection:
            result = connection.execute("UPDATE events SET acknowledged = 1 WHERE id = ?", (event_id,))
            connection.commit()
            return result.rowcount > 0

    def get_settings(self) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute("SELECT key, value_json FROM settings").fetchall()
        nested: dict[str, Any] = {"thresholds": {}}
        for row in rows:
            value = json.loads(row["value_json"])
            if row["key"].startswith("threshold."):
                nested["thresholds"][row["key"].split(".", 1)[1]] = value
            else:
                nested[row["key"]] = value
        return nested

    def update_setting(self, key: str, value: Any) -> None:
        with self._write_lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                (key, json.dumps(value), self._now()),
            )
            connection.commit()

    def analytics(self, hours: int = 24) -> dict[str, Any]:
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        with self.connect() as connection:
            totals = connection.execute(
                """
                SELECT event_type, COUNT(*) AS count, AVG(confidence) AS average_confidence
                FROM events WHERE started_at >= ? GROUP BY event_type
                """,
                (cutoff.isoformat(),),
            ).fetchall()
            timeline = connection.execute(
                """
                SELECT substr(started_at, 1, 13) || ':00:00+00:00' AS hour,
                       event_type, COUNT(*) AS count
                FROM events WHERE started_at >= ? GROUP BY hour, event_type ORDER BY hour
                """,
                (cutoff.isoformat(),),
            ).fetchall()
        return {
            "hours": hours,
            "totals": {
                row["event_type"]: {"count": row["count"], "average_confidence": row["average_confidence"]}
                for row in totals
            },
            "timeline": [dict(row) for row in timeline],
        }
