from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

LOGGER = logging.getLogger("oneeye.inference")
MINIMUM_ALERT_CONFIDENCE = 0.91


@dataclass(frozen=True)
class ModelSpec:
    event_type: str
    preferred_path: Path
    fallback_path: Path | None
    label_hints: tuple[str, ...]


class InferenceEngine:
    def __init__(self, models_dir: Path, thresholds: dict[str, float]):
        self.models_dir = models_dir
        self.thresholds = {key: max(MINIMUM_ALERT_CONFIDENCE, value) for key, value in thresholds.items()}
        self._models: dict[str, Any] = {}
        self._status: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        legacy = models_dir / "legacy"
        self.specs = {
            "fight": ModelSpec(
                "fight", models_dir / "fight" / "best.pt", legacy / "fight_Light.pt", ("fight", "violence")
            ),
            "garbage": ModelSpec(
                "garbage", models_dir / "garbage" / "best.pt", None, ("garbage", "trash", "litter", "waste")
            ),
            "fallen_person": ModelSpec(
                "fallen_person",
                models_dir / "fallen_person" / "best.pt",
                legacy / "accident_fallen_model.pt",
                ("fall", "fallen", "lying", "person down"),
            ),
            "mobile_phone": ModelSpec(
                "mobile_phone",
                models_dir / "mobile_phone" / "best.pt",
                legacy / "mobile_phone.pt",
                ("phone", "mobile", "cell"),
            ),
            "accident": ModelSpec(
                "accident", models_dir / "accident" / "best.pt", None, ("accident", "crash", "collision")
            ),
        }

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            for event_type in self.specs:
                self._status[event_type] = {"loaded": False, "error": f"Ultralytics unavailable: {exc}"}
            return

        with self._lock:
            self._models.clear()
            for event_type, spec in self.specs.items():
                path = spec.preferred_path if spec.preferred_path.exists() else spec.fallback_path
                if not path or not path.exists():
                    self._status[event_type] = {
                        "loaded": False,
                        "path": str(spec.preferred_path),
                        "error": "trained checkpoint not found",
                    }
                    continue
                try:
                    model = YOLO(str(path))
                    self._models[event_type] = model
                    model_config_path = path.parent / "model_config.yaml"
                    model_config = (
                        yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
                        if model_config_path.exists()
                        else {}
                    )
                    self._status[event_type] = {
                        "loaded": True,
                        "path": str(path),
                        "task": getattr(model, "task", "unknown"),
                        "classes": list(getattr(model, "names", {}).values()),
                        "version": path.stem,
                        "temporal_tiles": int(model_config.get("temporal_tiles", 1)),
                        "error": None,
                    }
                except Exception as exc:  # corrupted or incompatible model must not stop the API
                    LOGGER.exception("Failed to load %s model", event_type)
                    self._status[event_type] = {"loaded": False, "path": str(path), "error": str(exc)}

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        self.thresholds.update(
            {key: max(MINIMUM_ALERT_CONFIDENCE, value) for key, value in thresholds.items()}
        )

    def statuses(self) -> dict[str, dict[str, Any]]:
        return {key: value.copy() for key, value in self._status.items()}

    def temporal_event_types(self) -> set[str]:
        return {
            event_type
            for event_type, status in self._status.items()
            if status.get("loaded") and status.get("temporal_tiles") == 4
        }

    def predict(self, frame: np.ndarray, only: set[str] | None = None) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        with self._lock:
            selected = [(key, model) for key, model in self._models.items() if only is None or key in only]
            for event_type, model in selected:
                predictions.extend(self._predict_model(event_type, model, frame))
        return predictions

    def _predict_model(self, event_type: str, model: Any, frame: np.ndarray) -> list[dict[str, Any]]:
        threshold = float(self.thresholds.get(event_type, 0.5))
        status = self._status[event_type]
        results = model.predict(frame, conf=threshold, verbose=False)
        output: list[dict[str, Any]] = []
        for result in results:
            if getattr(result, "boxes", None) is not None:
                names = getattr(result, "names", getattr(model, "names", {}))
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = str(names[class_id])
                    if not self._label_matches(event_type, label, names):
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    output.append(
                        {
                            "event_type": event_type,
                            "label": label,
                            "confidence": confidence,
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "model_version": status["version"],
                        }
                    )
            probs = getattr(result, "probs", None)
            if probs is not None and probs.top1conf is not None:
                confidence = float(probs.top1conf)
                class_id = int(probs.top1)
                names = getattr(result, "names", getattr(model, "names", {}))
                label = str(names[class_id])
                if confidence >= threshold and self._positive_class(event_type, label, names):
                    output.append(
                        {
                            "event_type": event_type,
                            "label": label,
                            "confidence": confidence,
                            "bbox": None,
                            "model_version": status["version"],
                        }
                    )
        return output

    def _label_matches(self, event_type: str, label: str, names: dict | list) -> bool:
        labels = [str(value).lower() for value in (names.values() if isinstance(names, dict) else names)]
        hints = self.specs[event_type].label_hints
        # Dedicated single-class checkpoints represent their configured event even if the label is generic.
        if len(labels) == 1:
            return True
        value = label.lower()
        return any(hint in value for hint in hints)

    def _positive_class(self, event_type: str, label: str, names: dict | list) -> bool:
        value = label.lower()
        negative_tokens = (
            "normal",
            "nonfight",
            "non-fight",
            "no_fight",
            "unfight",
            "un_fight",
            "not_fight",
            "no accident",
            "no_accident",
            "adl",
        )
        if any(token in value for token in negative_tokens):
            return False
        labels = [str(item).lower() for item in (names.values() if isinstance(names, dict) else names)]
        if len(labels) == 2:
            return any(hint in value for hint in self.specs[event_type].label_hints)
        return True


EVENT_COLORS = {
    "fight": (94, 63, 244),
    "garbage": (11, 158, 245),
    "fallen_person": (255, 140, 0),
    "mobile_phone": (248, 189, 56),
    "accident": (50, 50, 255),
}


def draw_predictions(frame: np.ndarray, predictions: list[dict[str, Any]]) -> np.ndarray:
    annotated = frame.copy()
    for prediction in predictions:
        bbox = prediction.get("bbox")
        if not bbox:
            continue
        color = EVENT_COLORS.get(prediction["event_type"], (16, 185, 129))
        cv2.rectangle(annotated, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), color, 2)
        caption = f"{prediction['event_type']} {prediction['confidence']:.0%}"
        cv2.putText(
            annotated,
            caption,
            (bbox["x1"], max(24, bbox["y1"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image")
    return image
