from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from typing import Any

import cv2

from backend.app.services.camera import CameraManager
from backend.app.services.events import EventService
from backend.app.services.inference import InferenceEngine, draw_predictions


class SurveillancePipeline:
    def __init__(
        self,
        cameras: CameraManager,
        inference: InferenceEngine,
        events: EventService,
        inference_fps: float,
        confirmation_frames: int,
        stream_size: tuple[int, int],
        stream_fps: float,
        jpeg_quality: int,
    ):
        self.cameras = cameras
        self.inference = inference
        self.events = events
        self.inference_fps = inference_fps
        self.confirmation_frames = confirmation_frames
        self.stream_size = stream_size
        self.stream_fps = stream_fps
        self.jpeg_quality = jpeg_quality
        self._threads: dict[str, threading.Thread] = {}
        self._stops: dict[str, threading.Event] = {}
        self._predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._frame_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=4))
        self._stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"frames_analyzed": 0, "detections": 0, "last_inference_ms": None, "last_analyzed_at": None}
        )
        self._lock = threading.RLock()

    def start(self, camera_id: str, source: str | int | None = None) -> None:
        self.cameras.start(camera_id, source)
        if camera_id in self._threads and self._threads[camera_id].is_alive():
            return
        stop = threading.Event()
        self._stops[camera_id] = stop
        thread = threading.Thread(
            target=self._analyze, args=(camera_id, stop), name=f"pipeline-{camera_id}", daemon=True
        )
        self._threads[camera_id] = thread
        thread.start()

    def stop(self, camera_id: str) -> None:
        stop = self._stops.get(camera_id)
        if stop:
            stop.set()
        thread = self._threads.get(camera_id)
        if thread and thread.is_alive():
            thread.join(timeout=3)
        self.cameras.stop(camera_id)
        self._frame_history.pop(camera_id, None)

    def _analyze(self, camera_id: str, stop: threading.Event) -> None:
        last_sequence = -1
        streaks: dict[str, int] = defaultdict(int)
        while not stop.is_set():
            worker = self.cameras.get(camera_id)
            if worker is None:
                stop.wait(0.2)
                continue
            snapshot = worker.snapshot()
            if snapshot.frame is None or snapshot.sequence == last_sequence:
                stop.wait(0.02)
                continue
            last_sequence = snapshot.sequence
            started = time.perf_counter()
            temporal_events = self.inference.temporal_event_types()
            regular_events = set(self.inference.specs) - temporal_events
            predictions = self.inference.predict(snapshot.frame, only=regular_events)
            history = self._frame_history[camera_id]
            history.append(snapshot.frame.copy())
            if temporal_events and len(history) == 4:
                temporal_frame = self._compose_temporal_mosaic(list(history))
                predictions.extend(self.inference.predict(temporal_frame, only=temporal_events))
            elapsed_ms = (time.perf_counter() - started) * 1000
            detected_types = {item["event_type"] for item in predictions}
            for event_type in self.inference.specs:
                streaks[event_type] = streaks[event_type] + 1 if event_type in detected_types else 0
            with self._lock:
                self._predictions[camera_id] = predictions
                stats = self._stats[camera_id]
                stats["frames_analyzed"] += 1
                stats["detections"] += len(predictions)
                stats["last_inference_ms"] = round(elapsed_ms, 1)
                stats["last_analyzed_at"] = time.time()
            annotated = draw_predictions(snapshot.frame, predictions)
            for prediction in predictions:
                if streaks[prediction["event_type"]] >= self.confirmation_frames:
                    self.events.create(camera_id, prediction, annotated)
            # Read the setting each cycle so a live settings update takes effect
            # without restarting the camera worker.
            interval = 1.0 / max(self.inference_fps, 0.1)
            stop.wait(max(0, interval - (time.perf_counter() - started)))

    def stream(self, camera_id: str) -> Iterator[bytes]:
        last_sequence = -1
        while True:
            started = time.perf_counter()
            worker = self.cameras.get(camera_id)
            if worker is None:
                return
            snapshot = worker.snapshot()
            if snapshot.frame is None:
                time.sleep(0.01)
                continue
            if snapshot.sequence == last_sequence:
                time.sleep(0.001)
                continue
            last_sequence = snapshot.sequence
            frame = (
                snapshot.frame
                if (snapshot.frame.shape[1], snapshot.frame.shape[0]) == self.stream_size
                else cv2.resize(snapshot.frame, self.stream_size, interpolation=cv2.INTER_LINEAR)
            )
            with self._lock:
                predictions = list(self._predictions.get(camera_id, []))
            annotated = draw_predictions(frame, self._scale_predictions(predictions, snapshot.frame.shape, frame.shape))
            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            time.sleep(max(0, 1 / max(self.stream_fps, 1) - (time.perf_counter() - started)))

    @staticmethod
    def _compose_temporal_mosaic(frames: list) -> Any:
        height = max(2, frames[0].shape[0] // 2)
        width = max(2, frames[0].shape[1] // 2)
        tiles = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]
        return cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])])

    @staticmethod
    def _scale_predictions(predictions: list[dict[str, Any]], original_shape, new_shape) -> list[dict[str, Any]]:
        sx = new_shape[1] / original_shape[1]
        sy = new_shape[0] / original_shape[0]
        scaled = []
        for prediction in predictions:
            item = prediction.copy()
            if prediction.get("bbox"):
                box = prediction["bbox"]
                item["bbox"] = {
                    "x1": int(box["x1"] * sx),
                    "y1": int(box["y1"] * sy),
                    "x2": int(box["x2"] * sx),
                    "y2": int(box["y2"] * sy),
                }
            scaled.append(item)
        return scaled

    def latest_predictions(self, camera_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._predictions.get(camera_id, []))

    def stats(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {key: value.copy() for key, value in self._stats.items()}

    def stop_all(self) -> None:
        for camera_id in list(self._threads):
            self.stop(camera_id)
