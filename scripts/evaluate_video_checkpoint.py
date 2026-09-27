from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.pipeline import stratified_group_split  # noqa: E402


def video_records(raw_root: Path) -> list[tuple[Path, str, str]]:
    with (raw_root / "selected_videos.csv").open(encoding="utf-8", newline="") as handle:
        selected = list(csv.DictReader(handle))
    labels = {"fight": "fight", "normal": "normal"}
    records = []
    groups = {}
    for row in selected:
        label = labels[row["local_label"].lower()]
        filename = row["filename"]
        stem_parts = Path(filename).stem.rsplit("_", 1)
        source_stem = stem_parts[0] if len(stem_parts) == 2 and stem_parts[1].isdigit() else Path(filename).stem
        group = f"{label}/{source_stem}"
        path = raw_root / row["local_label"] / filename
        records.append((path, group, label))
        groups[group] = label
    split_map = stratified_group_split(groups)
    return [record for record in records if split_map[record[1]] == "test"]


def sample_frames(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = {round(value) for value in np.linspace(0, max(0, frame_count - 1), count)}
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in targets:
            frames.append(frame)
        index += 1
    capture.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO classifier by held-out source video")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = YOLO(str(args.checkpoint))
    records = video_records(PROJECT_ROOT / "machine_learning/datasets/raw/rwf_2000_hf")
    tp = tn = fp = fn = 0
    per_video = []
    for path, _, label in records:
        frames = sample_frames(path, args.frames)
        predictions = model.predict(frames, batch=args.frames, device=args.device, verbose=False)
        positive_scores = []
        for prediction in predictions:
            positive_scores.append(
                sum(
                    float(prediction.probs.data[class_id])
                    for class_id, class_name in prediction.names.items()
                    if "fight" in str(class_name).lower() and "un_fight" not in str(class_name).lower()
                )
            )
        score = float(np.mean(positive_scores)) if positive_scores else 0.0
        actual_positive = label == "fight"
        predicted_positive = score >= 0.5
        if actual_positive and predicted_positive:
            tp += 1
        elif actual_positive:
            fn += 1
        elif predicted_positive:
            fp += 1
        else:
            tn += 1
        per_video.append({"video": path.name, "label": label, "fight_score": score})
    total = tp + tn + fp + fn
    report = {
        "checkpoint": str(args.checkpoint),
        "videos": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "per_video": per_video,
    }
    output = args.output or PROJECT_ROOT / "machine_learning/reports/fight_legacy_video_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
