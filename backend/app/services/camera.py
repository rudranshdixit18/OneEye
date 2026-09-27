from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("oneeye.camera")


def normalize_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    value = str(source).strip()
    if value.isdigit() and len(value) <= 2:
        return int(value)
    return value


@dataclass
class CameraSnapshot:
    frame: np.ndarray | None
    sequence: int
    captured_at: float


class CameraWorker:
    def __init__(
        self,
        camera_id: str,
        source: str | int,
        clips_dir: Path,
        capture_size: tuple[int, int] = (1920, 1080),
        capture_fps: float = 60,
    ):
        self.camera_id = camera_id
        self.source = normalize_source(source)
        self.clips_dir = clips_dir
        self.capture_size = capture_size
        self.capture_fps = capture_fps
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._captured_at = 0.0
        self._online = False
        self._error: str | None = None
        self._recording = False
        self._writer: cv2.VideoWriter | None = None
        self._recording_path: Path | None = None
        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps = 0.0
        self._fps_window_started = 0.0
        self._fps_window_frames = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._release()

    def _open(self) -> bool:
        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            self._error = f"Unable to open camera source: {self.source}"
            self._online = False
            self._capture.release()
            self._capture = None
            return False
        if isinstance(self.source, int):
            width, height = self.capture_size
            self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._capture.set(cv2.CAP_PROP_FPS, self.capture_fps)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        reported_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0)
        self._actual_fps = reported_fps if 1 <= reported_fps <= 240 else 0.0
        self._fps_window_started = time.perf_counter()
        self._fps_window_frames = 0
        self._online = True
        self._error = None
        return True

    def _run(self) -> None:
        retry_delay = 1.0
        while not self._stop.is_set():
            if self._capture is None and not self._open():
                self._stop.wait(retry_delay)
                retry_delay = min(retry_delay * 1.7, 10)
                continue
            ok, frame = self._capture.read()
            if not ok or frame is None:
                self._online = False
                self._error = "Camera read failed; reconnecting"
                self._release_capture()
                self._stop.wait(retry_delay)
                continue
            retry_delay = 1.0
            self._fps_window_frames += 1
            fps_elapsed = time.perf_counter() - self._fps_window_started
            if fps_elapsed >= 1.0:
                self._actual_fps = self._fps_window_frames / fps_elapsed
                self._fps_window_started = time.perf_counter()
                self._fps_window_frames = 0
            with self._lock:
                self._frame = frame
                self._sequence += 1
                self._captured_at = time.time()
                self._online = True
                if self._recording:
                    self._write_recording_frame(frame)
        self._release()

    def snapshot(self) -> CameraSnapshot:
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            return CameraSnapshot(frame, self._sequence, self._captured_at)

    def start_recording(self) -> str:
        with self._lock:
            self._recording = True
            self._recording_path = None
            return "recording scheduled"

    def stop_recording(self) -> str | None:
        with self._lock:
            self._recording = False
            if self._writer:
                self._writer.release()
                self._writer = None
            return str(self._recording_path) if self._recording_path else None

    def _write_recording_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self.clips_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%S")
            self._recording_path = self.clips_dir / f"{self.camera_id}_{stamp}.mp4"
            height, width = frame.shape[:2]
            fps = self._capture.get(cv2.CAP_PROP_FPS) if self._capture else 20
            if not fps or fps <= 1 or fps > 120:
                fps = 20
            self._writer = cv2.VideoWriter(
                str(self._recording_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
        self._writer.write(frame)

    def status(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "source": str(self.source),
            "online": self._online,
            "recording": self._recording,
            "sequence": self._sequence,
            "last_frame_at": self._captured_at or None,
            "error": self._error,
            "requested_width": self.capture_size[0],
            "requested_height": self.capture_size[1],
            "requested_fps": self.capture_fps,
            "actual_width": self._actual_width,
            "actual_height": self._actual_height,
            "actual_fps": round(self._actual_fps, 1),
        }

    def _release_capture(self) -> None:
        if self._capture:
            self._capture.release()
            self._capture = None

    def _release(self) -> None:
        self._release_capture()
        if self._writer:
            self._writer.release()
            self._writer = None
        self._online = False


class CameraManager:
    def __init__(
        self,
        clips_dir: Path,
        configured_sources: dict[str, str],
        capture_size: tuple[int, int] = (1920, 1080),
        capture_fps: float = 60,
    ):
        self.clips_dir = clips_dir
        self.configured_sources = configured_sources
        self.capture_size = capture_size
        self.capture_fps = capture_fps
        self._workers: dict[str, CameraWorker] = {}
        self._lock = threading.RLock()

    def start(self, camera_id: str, source: str | int | None = None) -> CameraWorker:
        with self._lock:
            resolved = source if source is not None else self.configured_sources.get(camera_id, "")
            if resolved == "":
                raise ValueError(f"No source configured for {camera_id}")
            current = self._workers.get(camera_id)
            if current and str(current.source) != str(normalize_source(resolved)):
                current.stop()
                current = None
            if current is None:
                current = CameraWorker(
                    camera_id,
                    resolved,
                    self.clips_dir,
                    capture_size=self.capture_size,
                    capture_fps=self.capture_fps,
                )
                self._workers[camera_id] = current
            current.start()
            return current

    def stop(self, camera_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(camera_id)
            if not worker:
                return False
            worker.stop()
            return True

    def get(self, camera_id: str) -> CameraWorker | None:
        return self._workers.get(camera_id)

    def statuses(self) -> list[dict]:
        known = set(self.configured_sources) | set(self._workers)
        result = []
        for camera_id in sorted(known):
            worker = self._workers.get(camera_id)
            if worker:
                result.append(worker.status())
            else:
                result.append(
                    {
                        "camera_id": camera_id,
                        "source": self.configured_sources.get(camera_id, ""),
                        "online": False,
                        "recording": False,
                        "sequence": 0,
                        "last_frame_at": None,
                        "error": None,
                        "requested_width": self.capture_size[0],
                        "requested_height": self.capture_size[1],
                        "requested_fps": self.capture_fps,
                        "actual_width": 0,
                        "actual_height": 0,
                        "actual_fps": 0.0,
                    }
                )
        return result

    def stop_all(self) -> None:
        for worker in list(self._workers.values()):
            worker.stop()
