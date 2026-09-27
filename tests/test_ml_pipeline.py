import csv
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from machine_learning import pipeline
from machine_learning.pipeline import (
    _select_diverse_videos,
    compose_temporal_mosaic,
    deterministic_split,
    load_registry,
    stratified_group_split,
)


def test_dataset_registry_covers_all_five_models():
    covered = {item["model"] for item in load_registry().values()}
    assert {"fight", "garbage", "fallen_person", "mobile_phone", "accident"} <= covered


def test_training_status_updates_are_atomic_and_preserve_models(tmp_path, monkeypatch):
    status_path = tmp_path / "training_status.json"
    monkeypatch.setattr(pipeline, "STATUS_PATH", status_path)
    monkeypatch.setattr(pipeline, "STATUS_LOCK_PATH", tmp_path / "training_status.lock")

    pipeline.update_status("garbage", "train", "in_progress", {"epoch": 1})
    pipeline.update_status("accident", "acquire", "in_progress", {"files": 10})

    state = json.loads(status_path.read_text(encoding="utf-8"))
    assert state["models"]["garbage"]["details"] == {"epoch": 1}
    assert state["models"]["accident"]["details"] == {"files": 10}
    assert not status_path.with_suffix(".tmp").exists()


def test_epoch_status_callback_records_restart_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "STATUS_PATH", tmp_path / "training_status.json")
    monkeypatch.setattr(pipeline, "STATUS_LOCK_PATH", tmp_path / "training_status.lock")
    callback = pipeline.epoch_status_callback("mobile_phone", 80)

    callback(
        SimpleNamespace(
            epoch=4,
            epochs=80,
            save_dir=tmp_path / "runs" / "mobile_phone",
            metrics={"metrics/mAP50(B)": np.float32(0.75), "ignored": object()},
        )
    )

    state = json.loads(pipeline.STATUS_PATH.read_text(encoding="utf-8"))["models"]["mobile_phone"]
    assert state["details"]["epoch_completed"] == 5
    assert state["details"]["metrics"] == {"metrics/mAP50(B)": 0.75}
    assert state["details"]["last_checkpoint"].endswith("weights\\last.pt")


def test_deterministic_split_is_reproducible_and_disjoint():
    keys = [f"video-{index}" for index in range(20)]
    first = deterministic_split(keys)
    second = deterministic_split(reversed(keys))
    assert first == second
    assert set(first) == set(keys)
    assert set(first.values()) == {"train", "val", "test"}


def test_stratified_split_places_each_class_in_each_split():
    groups = {f"fight-{index}": "fight" for index in range(10)}
    groups.update({f"normal-{index}": "normal" for index in range(10)})
    result = stratified_group_split(groups)
    for label in {"fight", "normal"}:
        assert {result[key] for key, value in groups.items() if value == label} == {"train", "val", "test"}


def test_video_selection_prefers_distinct_source_families():
    files = [
        {"path": f"train/Fight/source-{source}_{clip}.avi"}
        for source in range(5)
        for clip in range(3)
    ]
    selected = _select_diverse_videos(files, 5)
    families = {item["path"].rsplit("_", 1)[0] for item in selected}
    assert len(families) == 5


def test_temporal_mosaic_preserves_frame_order():
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (10, 20, 30, 40)]
    mosaic = compose_temporal_mosaic(frames, 16)
    assert mosaic.shape == (16, 16, 3)
    assert [mosaic[4, 4, 0], mosaic[4, 12, 0], mosaic[12, 4, 0], mosaic[12, 12, 0]] == [
        10,
        20,
        30,
        40,
    ]


def test_compact_timestamp_sampling_preserves_requested_order(tmp_path):
    video_path = tmp_path / "sequence.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 16)
    )
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((16, 16, 3), index * 10, dtype=np.uint8))
    writer.release()

    capture = cv2.VideoCapture(str(video_path))
    frames = pipeline.sample_frames_at_timestamps(capture, 10.0, 20, [0.2, 0.4, 0.6, 0.8])
    capture.release()

    assert frames is not None
    assert [round(float(frame.mean()) / 10) for frame in frames] == [2, 4, 6, 8]


def test_accident_manifest_requires_two_video_disjoint_mosaics(tmp_path):
    rows = []
    for split in pipeline.SPLITS:
        for label in ("accident", "normal"):
            video = f"{split}_{label}.mp4"
            target = tmp_path / split / label
            target.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                (target / f"{Path(video).stem}_mosaic_{index:04d}.jpg").write_bytes(b"image")
                rows.append(
                    {
                        "video": video,
                        "split": split,
                        "label": label,
                        "sample_times_seconds": "1;1.25;1.5;1.75",
                        "reference_event_time_seconds": "1.25",
                        "offsets_seconds": "-0.25;0;0.25;0.5",
                    }
                )
    manifest = tmp_path / "frame_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    assert pipeline.validate_accident_manifest(tmp_path, len(rows)) == []
    rows[1]["split"] = "test"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    assert any("crosses split" in error for error in pipeline.validate_accident_manifest(tmp_path, len(rows)))


def test_ur_fall_preparation_rebuilds_and_uses_frame_posture_labels(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "ur_fall"
    raw.mkdir(parents=True)
    encoded, buffer = cv2.imencode(".png", np.full((8, 8, 3), 127, dtype=np.uint8))
    assert encoded

    rows = {"falls": [], "adls": []}
    for prefix, table in (("fall", "falls"), ("adl", "adls")):
        for number in range(1, 4):
            sequence = f"{prefix}-{number:02d}"
            archive = raw / f"{sequence}-cam0-rgb.zip"
            with zipfile.ZipFile(archive, "w") as package:
                for frame in (2, 4, 6):
                    package.writestr(
                        f"{sequence}-cam0-rgb/{sequence}-cam0-rgb-{frame:03d}.png",
                        buffer.tobytes(),
                    )
            rows[table].extend(
                [f"{sequence},2,0", f"{sequence},4,1", f"{sequence},6,-1"]
            )
    for table, values in rows.items():
        (raw / f"urfall-cam0-{table}.csv").write_text("\n".join(values), encoding="utf-8")

    stale = tmp_path / "processed" / "fallen_person" / "train" / "normal" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(buffer.tobytes())
    monkeypatch.setattr(pipeline, "DATASETS", tmp_path)

    pipeline.prepare_ur_fall()

    assert not stale.exists()
    output = tmp_path / "processed" / "fallen_person"
    assert len(list(output.rglob("fallen_person/*.png"))) == 6
    assert len(list(output.rglob("normal/*.png"))) == 6
    assert not list(output.rglob("*_00002.png"))


def test_mju_waste_augments_only_garbage_training_split(tmp_path, monkeypatch):
    taco = tmp_path / "raw" / "taco"
    (taco / "images").mkdir(parents=True)
    encoded, jpeg = cv2.imencode(".jpg", np.full((16, 16, 3), 90, dtype=np.uint8))
    assert encoded
    images = []
    annotations = []
    for image_id in range(10):
        filename = f"{image_id}.jpg"
        (taco / "images" / filename).write_bytes(jpeg.tobytes())
        images.append({"id": image_id, "file_name": filename, "width": 16, "height": 16})
        annotations.append({"id": image_id, "image_id": image_id, "bbox": [2, 2, 8, 8]})
    (taco / "annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations}), encoding="utf-8"
    )

    mju = tmp_path / "raw" / "mju_waste"
    mju.mkdir(parents=True)
    (mju / "example_color.png").write_bytes(jpeg.tobytes())
    (mju / "unannotated_color.png").write_bytes(jpeg.tobytes())
    (mju / "train.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "example_color.png", "width": 16, "height": 16},
                    {"id": 2, "file_name": "unannotated_color.png", "width": 16, "height": 16},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "bbox": [12, 12, 20, 20],
                        "segmentation": [1, 2, 11, 2, 11, 12, 1, 12],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "DATASETS", tmp_path)

    pipeline.prepare_garbage()

    output = tmp_path / "processed" / "garbage"
    assert (output / "images" / "train" / "mju_example_color.jpg").exists() or (
        output / "images" / "train" / "mju_example_color.png"
    ).exists()
    assert (output / "labels" / "train" / "mju_example_color.txt").exists()
    label = (output / "labels" / "train" / "mju_example_color.txt").read_text(encoding="utf-8")
    assert label == "0 0.375000 0.437500 0.625000 0.625000\n"
    assert not list((output / "images" / "train").glob("mju_unannotated_*"))
    assert not list((output / "images" / "val").glob("mju_*"))
    assert not list((output / "images" / "test").glob("mju_*"))
