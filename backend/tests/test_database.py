from pathlib import Path

from backend.app.core.database import Database


def test_event_round_trip_and_analytics(tmp_path: Path):
    database = Database(tmp_path / "oneeye.db")
    database.initialize()
    database.insert_event(
        {
            "id": "evt-1",
            "camera_id": "cam1",
            "event_type": "fight",
            "label": "fight",
            "confidence": 0.91,
            "started_at": "2026-09-07T00:00:00+00:00",
            "bbox": {"x1": 1, "y1": 2, "x2": 10, "y2": 20},
            "model_version": "test",
        }
    )
    rows = database.list_events(event_type="fight")
    assert rows[0]["id"] == "evt-1"
    assert rows[0]["bbox"]["x2"] == 10
    assert database.acknowledge_event("evt-1") is True


def test_settings_round_trip(tmp_path: Path):
    database = Database(tmp_path / "oneeye.db")
    database.initialize()
    database.update_setting("threshold.fight", 0.88)
    assert database.get_settings()["thresholds"]["fight"] == 0.88
