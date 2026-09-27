from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
import urllib.parse
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path

import numpy as np
import requests
import yaml
from filelock import FileLock
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = ROOT / "machine_learning"
DATASETS = ML_ROOT / "datasets"
REGISTRY_PATH = ML_ROOT / "dataset_registry.yaml"
CONFIGS = ML_ROOT / "configs"
MODEL_NAMES = ("fight", "garbage", "fallen_person", "mobile_phone", "accident")
SPLITS = ("train", "val", "test")
STATUS_PATH = ML_ROOT / "training_status.json"
STATUS_LOCK_PATH = ML_ROOT / "training_status.lock"
MINIMUM_HELD_OUT_SCORE = 0.95


def load_registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))["datasets"]


def load_config(model_name: str) -> dict:
    path = CONFIGS / f"{model_name}.yaml"
    if not path.exists():
        raise SystemExit(f"Unknown model: {model_name}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def update_status(model_name: str, stage: str, status: str, details: dict | None = None) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(STATUS_LOCK_PATH), timeout=30):
        state = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.exists() else {"models": {}}
        model_state = state["models"].setdefault(model_name, {})
        model_state.update(
            {
                "stage": stage,
                "status": status,
                "updated_at": datetime.now(UTC).isoformat(),
                "details": details or {},
            }
        )
        temporary = STATUS_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(STATUS_PATH)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_download(
    url: str,
    destination: Path,
    show_progress: bool = True,
    max_attempts: int = 6,
    read_timeout: int = 120,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(max_attempts):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(url, stream=True, timeout=(20, read_timeout), headers=headers) as response:
                response.raise_for_status()
                resumed = existing > 0 and response.status_code == 206
                if existing and not resumed:
                    existing = 0
                total = existing + int(response.headers.get("content-length", 0))
                mode = "ab" if resumed else "wb"
                with (
                    partial.open(mode) as handle,
                    tqdm(
                        total=total,
                        initial=existing,
                        unit="B",
                        unit_scale=True,
                        desc=destination.name,
                        disable=not show_progress,
                    ) as bar,
                ):
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            bar.update(len(chunk))
            partial.replace(destination)
            return
        except requests.RequestException:
            if attempt + 1 >= max_attempts:
                raise
            time.sleep(min(2**attempt, 16))


def write_manifest(dataset_name: str, root: Path, source: dict) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "dataset_manifest.json":
            files.append(
                {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    manifest = {
        "dataset": dataset_name,
        "source": source,
        "acquired_at": datetime.now(UTC).isoformat(),
        "files": files,
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def acquire_taco(destination: Path, max_images: int | None) -> None:
    archive = destination / "taco-master.zip"
    if not archive.exists():
        stream_download("https://github.com/pedropro/TACO/archive/refs/heads/master.zip", archive)
    with zipfile.ZipFile(archive) as package:
        package.extractall(destination)
    repo = destination / "TACO-master"
    annotations = repo / "data" / "annotations.json"
    if not annotations.exists():
        candidates = list(repo.rglob("annotations*.json"))
        if not candidates:
            raise RuntimeError("TACO annotations were not found in the official archive")
        annotations = candidates[0]
    data = json.loads(annotations.read_text(encoding="utf-8"))
    images = data["images"][:max_images] if max_images else data["images"]
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    def download_image(item: dict) -> bool:
        target = image_dir / item["file_name"]
        if target.exists():
            return True
        partial = target.with_suffix(target.suffix + ".part")
        # The official 640 px derivative preserves aspect ratio and therefore the
        # normalized COCO boxes while avoiding multi-megabyte originals. Existing
        # partial originals retain their original URL so byte-range resume is valid.
        if partial.exists():
            url = item.get("flickr_url") or item.get("flickr_640_url") or item.get("coco_url")
        else:
            url = item.get("flickr_640_url") or item.get("flickr_url") or item.get("coco_url")
        if not url:
            return False
        try:
            stream_download(url, target, show_progress=False, max_attempts=2, read_timeout=20)
        except requests.RequestException:
            return False
        return True

    with ThreadPoolExecutor(max_workers=min(24, max(1, len(images)))) as executor:
        futures = [executor.submit(download_image, item) for item in images]
        for future in tqdm(as_completed(futures), total=len(futures), desc="TACO images"):
            future.result()
    shutil.copy2(annotations, destination / "annotations.json")


def _open_images_urls(split: str) -> tuple[str, str]:
    annotations = {
        "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
        "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    }[split]
    image_prefix = {"train": "train", "validation": "validation", "test": "test"}[split]
    return annotations, image_prefix


def acquire_open_images(destination: Path, max_images: int | None) -> None:
    class_file = destination / "class-descriptions-boxable.csv"
    stream_download(
        "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
        class_file,
    )
    mobile_mid = None
    with class_file.open(encoding="utf-8", newline="") as handle:
        for mid, name in csv.reader(handle):
            if name.strip().lower() in {"mobile phone", "cell phone"}:
                mobile_mid = mid
                break
    if not mobile_mid:
        raise RuntimeError("Mobile phone class was not found in Open Images class descriptions")
    records = []
    negative_records = []
    # The cap covers a 4:1 positive-to-background set across the three official
    # splits. This teaches false-positive suppression without creating batches
    # dominated by empty labels.
    positive_per_split = max(1, max_images * 4 // 15) if max_images else None
    negative_per_split = max(1, max_images // 15) if max_images else None
    for split in ("train", "validation", "test"):
        annotation_url, image_prefix = _open_images_urls(split)
        selected = []
        selected_image_ids: set[str] = set()
        selected_negative_ids: list[str] = []
        # Stream the official multi-gigabyte annotation table and stop once the
        # requested class subset is complete instead of storing unrelated rows.
        with requests.get(annotation_url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            response.encoding = "utf-8"
            reader = csv.DictReader(response.iter_lines(decode_unicode=True))
            for image_id, image_rows in groupby(reader, key=lambda row: row["ImageID"]):
                group_rows = list(image_rows)
                mobile_rows = [row for row in group_rows if row["LabelName"] == mobile_mid]
                if mobile_rows and (
                    positive_per_split is None or len(selected_image_ids) < positive_per_split
                ):
                    selected.extend(mobile_rows)
                    selected_image_ids.add(image_id)
                elif not mobile_rows and (
                    negative_per_split is None or len(selected_negative_ids) < negative_per_split
                ):
                    selected_negative_ids.append(image_id)
                positive_complete = (
                    positive_per_split is not None and len(selected_image_ids) >= positive_per_split
                )
                negative_complete = (
                    negative_per_split is not None and len(selected_negative_ids) >= negative_per_split
                )
                if positive_complete and negative_complete:
                    break
        image_ids = sorted(selected_image_ids | set(selected_negative_ids))
        image_dir = destination / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        def download_image(
            image_id: str, image_dir: Path = image_dir, image_prefix: str = image_prefix
        ) -> str | None:
            target = image_dir / f"{image_id}.jpg"
            if not target.exists():
                url = f"https://open-images-dataset.s3.amazonaws.com/{image_prefix}/{image_id}.jpg"
                try:
                    stream_download(url, target, show_progress=False)
                except requests.RequestException:
                    return None
            return image_id

        worker_count = min(12, max(1, len(image_ids)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(tqdm(executor.map(download_image, image_ids), total=len(image_ids), desc=f"Open Images {split}"))
        records.extend({**row, "Split": split} for row in selected if (image_dir / f"{row['ImageID']}.jpg").exists())
        negative_records.extend(
            {"ImageID": image_id, "Split": split}
            for image_id in selected_negative_ids
            if (image_dir / f"{image_id}.jpg").exists()
        )
    with (destination / "mobile_phone_boxes.csv").open("w", encoding="utf-8", newline="") as handle:
        if records:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    with (destination / "negative_images.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageID", "Split"])
        writer.writeheader()
        writer.writerows(negative_records)


def acquire_ur_fall(destination: Path, max_images: int | None) -> None:
    page = requests.get("https://fenix.ur.edu.pl/~mkepski/ds/uf.html", timeout=(20, 60))
    page.raise_for_status()
    import re

    links = sorted(set(re.findall(r'href\s*=\s*["\']([^"\']+-cam0-rgb\.zip)["\']', page.text, re.IGNORECASE)))
    if not links:
        raise RuntimeError("Official UR Fall RGB archive links were not found")
    if max_images:
        # max-images acts as an acquisition cap; one sequence archive is the minimum unit.
        sequence_cap = max(6, max_images // 100)
        normal_links = [link for link in links if "/adl-" in link.lower()]
        fall_links = [link for link in links if "/fall-" in link.lower()]
        per_class = max(3, sequence_cap // 2)
        links = normal_links[:per_class] + fall_links[:per_class]
    for link in links:
        url = urllib.parse.urljoin(page.url, link)
        target = destination / Path(urllib.parse.urlparse(url).path).name
        if not target.exists():
            stream_download(url, target)
    # The official extracted-feature tables contain the authoritative posture
    # label for each frame: -1 not lying, 0 transitional, and 1 lying down.
    feature_links = sorted(
        set(
            re.findall(
                r'href\s*=\s*["\']([^"\']*urfall-cam0-(?:falls|adls)\.csv)["\']',
                page.text,
                re.IGNORECASE,
            )
        )
    )
    if len(feature_links) != 2:
        raise RuntimeError("Official UR Fall frame-label tables were not found")
    for link in feature_links:
        url = urllib.parse.urljoin(page.url, link)
        target = destination / Path(urllib.parse.urlparse(url).path).name
        if not target.exists():
            stream_download(url, target)


def acquire_google_drive_zip(destination: Path, source: dict) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive dataset acquisition requires `pip install -r machine_learning/requirements.txt`"
        ) from exc

    archive = destination / source["archive_name"]
    downloaded = gdown.download(
        id=source["google_drive_id"],
        output=str(archive),
        quiet=False,
        resume=True,
    )
    if not downloaded or not archive.exists() or not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Google Drive did not return a valid ZIP archive: {archive}")
    annotation_base = source["annotations_base_url"].rstrip("/")
    for split in ("train", "val", "test"):
        stream_download(f"{annotation_base}/{split}.json", destination / f"{split}.json")
    marker = destination / ".extracted"
    if not marker.exists():
        destination_root = destination.resolve()
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                target = (destination / member.filename).resolve()
                if destination_root not in target.parents and target != destination_root:
                    raise RuntimeError(f"Unsafe path in {archive.name}: {member.filename}")
            package.extractall(destination)
        marker.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")


def _huggingface_files(repo_id: str, folder: str) -> list[dict]:
    encoded_folder = urllib.parse.quote(folder, safe="/")
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{encoded_folder}"
    response = requests.get(url, params={"recursive": "true", "expand": "false", "limit": 1000}, timeout=(20, 120))
    response.raise_for_status()
    return [
        item
        for item in response.json()
        if item.get("type") == "file" and Path(item["path"]).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    ]


def _huggingface_dataset_rows(repo_id: str, split: str = "train") -> list[dict]:
    """Read lightweight row metadata without downloading converted video payloads."""
    endpoint = "https://datasets-server.huggingface.co/rows"
    rows: list[dict] = []
    offset = 0
    page_size = 100
    total = 1
    while offset < total:
        response = requests.get(
            endpoint,
            params={
                "dataset": repo_id,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": page_size,
            },
            timeout=(20, 120),
        )
        response.raise_for_status()
        payload = response.json()
        page = [item["row"] for item in payload["rows"]]
        rows.extend(page)
        total = int(payload["num_rows_total"])
        if not page:
            break
        offset += len(page)
    return rows


def _write_huggingface_video_metadata(destination: Path, repo_id: str, selected_paths: set[str]) -> None:
    records = []
    for row in _huggingface_dataset_rows(repo_id):
        source_url = row["video"]["src"]
        marker = "/train/"
        if marker not in source_url:
            continue
        source_path = "train/" + urllib.parse.unquote(source_url.split(marker, 1)[1])
        if source_path not in selected_paths:
            continue
        source_label = Path(source_path).parent.name
        records.append(
            {
                "source_path": source_path,
                "local_label": "accident" if source_label == "positive" else "normal",
                "filename": Path(source_path).name,
                "time_of_event": row.get("time_of_event"),
                "time_of_alert": row.get("time_of_alert"),
                "light_conditions": row.get("light_conditions"),
                "weather": row.get("weather"),
                "scene": row.get("scene"),
            }
        )
    if len(records) != len(selected_paths):
        found = {record["source_path"] for record in records}
        missing = sorted(selected_paths - found)
        raise RuntimeError(f"Hugging Face metadata is missing {len(missing)} selected videos: {missing[:5]}")
    target = destination / "video_metadata.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda item: item["source_path"]))


def _select_diverse_videos(files: list[dict], limit: int | None) -> list[dict]:
    if limit is None or len(files) <= limit:
        return sorted(files, key=lambda item: item["path"])
    families: dict[str, list[dict]] = defaultdict(list)
    for item in sorted(files, key=lambda value: value["path"]):
        stem = Path(item["path"]).stem
        parts = stem.rsplit("_", 1)
        family = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        families[family].append(item)
    family_names = sorted(families)
    random.Random(20260907).shuffle(family_names)
    selected = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for family in family_names:
            if round_index < len(families[family]):
                selected.append(families[family][round_index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        round_index += 1
    return sorted(selected, key=lambda item: item["path"])


def acquire_huggingface_binary_video(destination: Path, source: dict, max_images: int | None) -> None:
    repo_id = source["repo_id"]
    folders = source["folders"]
    per_class_limit = max(1, max_images // len(folders)) if max_images else None
    selected_paths: set[str] = set()
    selected_records = []
    for label, folder in folders.items():
        files = _select_diverse_videos(_huggingface_files(repo_id, folder), per_class_limit)
        selected_paths.update(item["path"] for item in files)
        selected_records.extend(
            {"local_label": label, "source_path": item["path"], "filename": Path(item["path"]).name}
            for item in files
        )

        def download_video(item: dict, label: str = label) -> str:
            filename = Path(item["path"]).name
            target = destination / label / filename
            expected_size = item.get("size")
            if target.exists():
                if expected_size is None or target.stat().st_size == expected_size:
                    return filename
                raise RuntimeError(f"Existing Hugging Face file has unexpected size: {target}")
            encoded_path = urllib.parse.quote(item["path"], safe="/")
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{encoded_path}"
            stream_download(url, target, show_progress=False)
            return filename

        worker_count = min(4, max(1, len(files)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(
                tqdm(
                    executor.map(download_video, files),
                    total=len(files),
                    desc=f"{source['model']} {label} videos",
                )
            )
    with (destination / "selected_videos.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("local_label", "source_path", "filename"))
        writer.writeheader()
        writer.writerows(sorted(selected_records, key=lambda item: item["source_path"]))
    if source.get("row_metadata") == "nexar_collision":
        _write_huggingface_video_metadata(destination, repo_id, selected_paths)


def acquire_zenodo_7z(destination: Path, source: dict) -> None:
    import multivolumefile
    import py7zr

    record_id = source["zenodo_record"]
    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=(20, 120))
    response.raise_for_status()
    files = sorted(response.json()["files"], key=lambda item: item["key"])
    for item in tqdm(files, desc="RWF-2000 archive volumes"):
        target = destination / item["key"]
        stream_download(item["links"]["self"], target)
        algorithm, expected = item["checksum"].split(":", 1)
        actual = file_digest(target, algorithm)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {target.name}: expected {expected}, received {actual}")
    first_volume = destination / files[0]["key"]
    archive_root = first_volume.with_suffix("")
    extracted_marker = destination / ".extracted"
    if not extracted_marker.exists():
        with multivolumefile.open(archive_root, mode="rb") as multi_archive:
            with py7zr.SevenZipFile(multi_archive, mode="r") as package:
                package.extractall(path=destination)
        extracted_marker.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")


def acquire(dataset_name: str, accept_license: bool, max_images: int | None) -> None:
    registry = load_registry()
    if dataset_name not in registry:
        raise SystemExit(f"Unknown dataset: {dataset_name}")
    source = registry[dataset_name]
    if not accept_license:
        raise SystemExit("Review dataset_registry.yaml and rerun with --accept-license")
    destination = DATASETS / "raw" / dataset_name
    destination.mkdir(parents=True, exist_ok=True)
    update_status(source["model"], "acquire", "in_progress", {"dataset": dataset_name})
    method = source["acquisition"]
    if method == "manual":
        raise SystemExit(source["notes"])
    if method == "taco":
        acquire_taco(destination, max_images)
    elif method == "open_images":
        acquire_open_images(destination, max_images)
    elif method == "ur_fall":
        acquire_ur_fall(destination, max_images)
    elif method == "google_drive_zip":
        acquire_google_drive_zip(destination, source)
    elif method == "huggingface_binary_video":
        acquire_huggingface_binary_video(destination, source, max_images)
    elif method == "zenodo_7z":
        acquire_zenodo_7z(destination, source)
    else:
        raise SystemExit(f"Unsupported acquisition method: {method}")
    write_manifest(dataset_name, destination, source)
    update_status(source["model"], "acquire", "complete", {"dataset": dataset_name})


def deterministic_split(keys: Iterable[str], seed: int = 20260907) -> dict[str, str]:
    values = sorted(set(keys))
    random.Random(seed).shuffle(values)
    count = len(values)
    train_end = max(1, int(count * 0.70))
    val_end = max(train_end + 1, int(count * 0.85)) if count > 2 else train_end
    return {
        key: "train" if index < train_end else "val" if index < val_end else "test" for index, key in enumerate(values)
    }


def stratified_group_split(group_labels: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for label in sorted(set(group_labels.values())):
        keys = [key for key, value in group_labels.items() if value == label]
        result.update(deterministic_split(keys))
    return result


def coco_annotation_box(annotation: dict, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Return a clipped COCO xywh box, preferring polygon geometry when present.

    MJU-Waste's exported ``bbox`` values are inconsistent with some polygons
    and can extend beyond the image. The segmentation points are the source
    geometry used for its masks, so derive the box from them when possible.
    """
    segmentation = annotation.get("segmentation")
    coordinates: list[float] = []
    if isinstance(segmentation, list):
        polygons = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
        for polygon in polygons:
            if isinstance(polygon, list):
                coordinates.extend(float(value) for value in polygon)

    if len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            return None
        x1, y1, box_width, box_height = (float(value) for value in bbox)
        x2, y2 = x1 + box_width, y1 + box_height

    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def prepare_garbage() -> None:
    raw = DATASETS / "raw" / "taco"
    annotations = json.loads((raw / "annotations.json").read_text(encoding="utf-8"))
    output = DATASETS / "processed" / "garbage"
    if output.exists():
        shutil.rmtree(output)
    images_by_id = {item["id"]: item for item in annotations["images"]}
    boxes = defaultdict(list)
    for annotation in annotations["annotations"]:
        boxes[annotation["image_id"]].append(annotation["bbox"])
    split_map = deterministic_split([str(image_id) for image_id in images_by_id if boxes[image_id]])
    for image_id, image in images_by_id.items():
        source = raw / "images" / image["file_name"]
        if not source.exists() or not boxes[image_id]:
            continue
        split = split_map[str(image_id)]
        image_target = output / "images" / split / f"{image_id}.jpg"
        label_target = output / "labels" / split / f"{image_id}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_target)
        width, height = image["width"], image["height"]
        lines = []
        for x, y, w, h in boxes[image_id]:
            lines.append(f"0 {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}")
        label_target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # When MJU-Waste has been acquired, its official training split augments
    # TACO training only. TACO validation and test stay untouched, preserving a
    # stable held-out benchmark across the TACO-only and combined candidates.
    mju_raw = DATASETS / "raw" / "mju_waste"
    mju_annotations_path = mju_raw / "train.json"
    if mju_annotations_path.exists():
        mju_annotations = json.loads(mju_annotations_path.read_text(encoding="utf-8"))
        mju_images = {
            path.name: path
            for path in mju_raw.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        mju_annotations_by_image = defaultdict(list)
        for annotation in mju_annotations["annotations"]:
            mju_annotations_by_image[annotation["image_id"]].append(annotation)
        for image in mju_annotations["images"]:
            source_image = mju_images.get(image["file_name"])
            if source_image is None:
                raise RuntimeError(f"MJU-Waste image is missing: {image['file_name']}")
            width, height = image["width"], image["height"]
            image_boxes = [
                box
                for annotation in mju_annotations_by_image[image["id"]]
                if (box := coco_annotation_box(annotation, width, height)) is not None
            ]
            # An unannotated waste image is not a trustworthy negative. Exclude
            # it instead of creating an empty label that would teach a false
            # background example.
            if not image_boxes:
                continue
            prefix = f"mju_{Path(image['file_name']).stem}"
            image_target = output / "images" / "train" / f"{prefix}{source_image.suffix.lower()}"
            label_target = output / "labels" / "train" / f"{prefix}.txt"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, image_target)
            lines = []
            for x, y, w, h in image_boxes:
                lines.append(
                    f"0 {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
                    f"{w / width:.6f} {h / height:.6f}"
                )
            label_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "garbage"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def prepare_mobile_phone() -> None:
    raw = DATASETS / "raw" / "open_images_mobile_phone"
    output = DATASETS / "processed" / "mobile_phone"
    if output.exists():
        shutil.rmtree(output)
    rows = list(csv.DictReader((raw / "mobile_phone_boxes.csv").open(encoding="utf-8", newline="")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["Split"], row["ImageID"])].append(row)
    split_names = {"train": "train", "validation": "val", "test": "test"}
    for (source_split, image_id), records in grouped.items():
        split = split_names[source_split]
        source = raw / "images" / source_split / f"{image_id}.jpg"
        if not source.exists():
            continue
        target = output / "images" / split / source.name
        label = output / "labels" / split / f"{image_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        lines = []
        for row in records:
            xmin, xmax = float(row["XMin"]), float(row["XMax"])
            ymin, ymax = float(row["YMin"]), float(row["YMax"])
            lines.append(f"0 {(xmin + xmax) / 2:.6f} {(ymin + ymax) / 2:.6f} {xmax - xmin:.6f} {ymax - ymin:.6f}")
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    negative_path = raw / "negative_images.csv"
    if negative_path.exists():
        for row in csv.DictReader(negative_path.open(encoding="utf-8", newline="")):
            source_split, image_id = row["Split"], row["ImageID"]
            split = split_names[source_split]
            source = raw / "images" / source_split / f"{image_id}.jpg"
            if not source.exists():
                continue
            target = output / "images" / split / source.name
            label = output / "labels" / split / f"{image_id}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            label.write_text("", encoding="utf-8")
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "mobile_phone"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def prepare_ur_fall() -> None:
    raw = DATASETS / "raw" / "ur_fall"
    output = DATASETS / "processed" / "fallen_person"
    archives = sorted(raw.glob("*-cam0-rgb.zip"))
    # A larger acquisition can assign an existing sequence to a different
    # deterministic split. Rebuild from source so stale frames cannot remain in
    # two splits and inflate held-out accuracy through sequence leakage.
    if output.exists():
        shutil.rmtree(output)
    frame_labels: dict[tuple[str, int], int] = {}
    label_tables = sorted(raw.glob("urfall-cam0-*.csv"))
    if len(label_tables) != 2:
        raise RuntimeError("UR Fall preparation requires both official frame-label tables")
    for table in label_tables:
        with table.open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 3:
                    continue
                frame_labels[(row[0], int(row[1]))] = int(row[2])

    group_labels = {}
    for archive in archives:
        sequence = archive.stem.split("-cam0")[0]
        group_labels[sequence] = "fallen_person" if sequence.startswith("fall-") else "normal"
    split_map = stratified_group_split(group_labels)
    for archive in archives:
        sequence = archive.stem.split("-cam0")[0]
        split = split_map[sequence]
        with zipfile.ZipFile(archive) as package:
            members = [member for member in package.namelist() if member.lower().endswith((".png", ".jpg", ".jpeg"))]
            for member in members:
                frame_number = int(Path(member).stem.rsplit("-", 1)[-1])
                posture = frame_labels.get((sequence, frame_number))
                # Transitional and unlabelled frames are neither a stable
                # fallen nor normal pose and are deliberately excluded.
                if posture not in {-1, 1}:
                    continue
                label = "fallen_person" if posture == 1 else "normal"
                # Positive poses are rarer, so retain them at about 15 fps
                # while sampling normal poses at about 5 fps.
                stride = 2 if posture == 1 else 6
                if frame_number % stride:
                    continue
                target = output / split / label / f"{sequence}_{frame_number:05d}{Path(member).suffix.lower()}"
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)


def compose_temporal_mosaic(frames: list[np.ndarray], image_size: int) -> np.ndarray:
    """Place four ordered frames into a 2x2 image without changing their sequence."""
    import cv2

    if len(frames) != 4:
        raise ValueError("A temporal mosaic requires exactly four frames")
    tile_width = image_size // 2
    tile_height = image_size // 2
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, 2)
        y1, x1 = row * tile_height, column * tile_width
        canvas[y1 : y1 + tile_height, x1 : x1 + tile_width] = tile
    return canvas


def sample_temporal_mosaics(
    capture,
    frame_count: int,
    image_size: int,
    samples_per_video: int,
) -> list[np.ndarray]:
    """Sample overlapping ordered windows with one sequential video decode."""
    if frame_count < 4:
        return []
    window_span = 0.30
    starts = np.linspace(0.0, 1.0 - window_span, samples_per_video)
    windows = [
        [round(position * (frame_count - 1)) for position in np.linspace(start, start + window_span, 4)]
        for start in starts
    ]
    required_indices = {index for window in windows for index in window}
    captured: dict[int, np.ndarray] = {}
    frame_index = 0
    while frame_index <= max(required_indices):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in required_indices:
            captured[frame_index] = frame.copy()
        frame_index += 1
    return [
        compose_temporal_mosaic([captured[index] for index in window], image_size)
        for window in windows
        if all(index in captured for index in window)
    ]


def sample_frames_at_timestamps(
    capture,
    fps: float,
    frame_count: int,
    timestamps: list[float],
) -> list[np.ndarray] | None:
    """Decode a compact timestamp range with one seek instead of one seek per frame."""
    import cv2

    if frame_count < 1 or fps <= 0 or not timestamps:
        return None
    target_indices = [
        min(max(0, round(timestamp * fps)), frame_count - 1) for timestamp in timestamps
    ]
    required = set(target_indices)
    first_index, last_index = min(required), max(required)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)
    current_index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    captured: dict[int, np.ndarray] = {}
    while current_index <= last_index:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        if decoded_index in required:
            captured[decoded_index] = frame.copy()
        current_index = decoded_index + 1
    if not all(index in captured for index in target_indices):
        return None
    return [captured[index] for index in target_indices]


def prepare_video_frames(model_name: str, raw_name: str, positive_name: str) -> None:
    raw = DATASETS / "raw" / raw_name
    output = DATASETS / "processed" / model_name
    classes = {
        positive_name: positive_name,
        "positive": positive_name,
        "normal": "normal",
        "negative": "normal",
        "nonviolent": "normal",
        "non-violent": "normal",
        "nonfight": "normal",
        "non-fight": "normal",
    }
    selected_index = raw / "selected_videos.csv"
    selected = None
    if selected_index.exists():
        with selected_index.open(encoding="utf-8", newline="") as handle:
            selected = {(row["local_label"], row["filename"]) for row in csv.DictReader(handle)}
    videos = [
        path
        for path in raw.rglob("*")
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        and (selected is None or (path.parent.name, path.name) in selected)
    ]
    records = []
    for video in videos:
        parent_label = video.parent.name.lower().replace(" ", "_")
        label = classes.get(parent_label)
        if label is None:
            label = positive_name if positive_name in parent_label else "normal"
        stem_parts = video.stem.rsplit("_", 1)
        source_stem = stem_parts[0] if len(stem_parts) == 2 and stem_parts[1].isdigit() else video.stem
        group = label + "/" + source_stem
        records.append((video, group, label))
    split_map = stratified_group_split({group: label for _, group, label in records})
    import cv2

    config = load_config(model_name)
    temporal_tiles = int(config.get("temporal_tiles", 1))
    if temporal_tiles not in {1, 4}:
        raise ValueError("temporal_tiles must be either 1 or 4")
    if output.exists():
        shutil.rmtree(output)
    for video, group, label in records:
        capture = cv2.VideoCapture(str(video))
        if temporal_tiles == 4:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            mosaics = sample_temporal_mosaics(
                capture,
                frame_count,
                int(config["image_size"]),
                int(config.get("samples_per_video", 4)),
            )
            for saved, mosaic in enumerate(mosaics):
                target = output / split_map[group] / label / f"{video.stem}_mosaic_{saved:04d}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), mosaic)
            capture.release()
            continue
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        stride = max(1, int(fps / 2))
        index = 0
        saved = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                target = output / split_map[group] / label / f"{video.stem}_{saved:05d}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), frame)
                saved += 1
            index += 1
        capture.release()


def prepare_nexar_accident() -> None:
    raw = DATASETS / "raw" / "nexar_collision"
    output = DATASETS / "processed" / "accident"
    metadata_path = raw / "video_metadata.csv"
    if not metadata_path.exists():
        raise RuntimeError("Missing Nexar video_metadata.csv; rerun dataset acquisition")
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        metadata = {
            (row["local_label"], row["filename"]): row for row in csv.DictReader(handle)
        }
    videos = sorted(
        path
        for path in raw.rglob("*.mp4")
        if path.parent.name in {"accident", "normal"} and (path.parent.name, path.name) in metadata
    )
    groups = [path.parent.name + "/" + path.stem for path in videos]
    split_map = stratified_group_split({group: video.parent.name for video, group in zip(videos, groups, strict=True)})
    import cv2

    if output.exists():
        shutil.rmtree(output)

    accident_videos = [video for video in videos if video.parent.name == "accident"]
    normal_videos = [video for video in videos if video.parent.name == "normal"]
    accident_event_times = {
        video.name: float(metadata[("accident", video.name)]["time_of_event"])
        for video in accident_videos
    }
    ordered_event_times = [accident_event_times[video.name] for video in accident_videos]
    if not ordered_event_times:
        raise RuntimeError("No positive Nexar event timestamps were found")
    # Pair normal clips with the positive timestamp distribution. This prevents the
    # classifier from exploiting video position instead of learning collision content.
    normal_reference_times = {
        video.name: ordered_event_times[index % len(ordered_event_times)]
        for index, video in enumerate(normal_videos)
    }
    frame_records = []
    sample_windows = ((-0.25, 0.0, 0.25, 0.5), (0.5, 0.75, 1.0, 1.25))
    image_size = int(load_config("accident")["image_size"])
    for video, group in tqdm(list(zip(videos, groups, strict=True)), desc="Accident frames"):
        label = video.parent.name
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        event_time = (
            accident_event_times[video.name]
            if label == "accident"
            else normal_reference_times[video.name]
        )
        timestamps = [
            min(max(0.0, event_time + offset), max(0.0, duration - 1 / fps))
            for offsets in sample_windows
            for offset in offsets
        ]
        sampled_frames = sample_frames_at_timestamps(capture, fps, frame_count, timestamps)
        if sampled_frames is None:
            capture.release()
            continue
        for saved, offsets in enumerate(sample_windows):
            start = saved * 4
            frames = sampled_frames[start : start + 4]
            window_timestamps = timestamps[start : start + 4]
            if len(frames) != 4:
                continue
            mosaic = compose_temporal_mosaic(frames, image_size)
            target = output / split_map[group] / label / f"{video.stem}_mosaic_{saved:04d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), mosaic)
            frame_records.append(
                {
                    "video": video.name,
                    "split": split_map[group],
                    "label": label,
                    "sample_times_seconds": ";".join(map(str, (round(value, 3) for value in window_timestamps))),
                    "reference_event_time_seconds": event_time,
                    "offsets_seconds": ";".join(map(str, offsets)),
                }
            )
        capture.release()
    with (output / "frame_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=frame_records[0].keys())
        writer.writeheader()
        writer.writerows(frame_records)


def prepare(model_name: str) -> None:
    update_status(model_name, "prepare", "in_progress")
    if model_name == "garbage":
        prepare_garbage()
    elif model_name == "mobile_phone":
        prepare_mobile_phone()
    elif model_name == "fallen_person":
        prepare_ur_fall()
    elif model_name == "fight":
        mirror_root = DATASETS / "raw" / "rwf_2000_hf"
        raw_name = "rwf_2000_hf" if any(mirror_root.rglob("*.avi")) else "rwf_2000"
        prepare_video_frames("fight", raw_name, "fight")
    elif model_name == "accident":
        prepare_nexar_accident()
    update_status(model_name, "prepare", "complete")


def validate_accident_manifest(root: Path, image_count: int) -> list[str]:
    """Validate temporal-sample cardinality and source-video split isolation."""
    manifest_path = root / "frame_manifest.csv"
    if not manifest_path.exists():
        return ["Missing accident frame_manifest.csv"]
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    if len(rows) != image_count:
        errors.append(f"Manifest/image count mismatch: {len(rows)} rows for {image_count} images")
    by_video = defaultdict(list)
    for row in rows:
        by_video[row.get("video", "")].append(row)
    for video, video_rows in by_video.items():
        splits = {row.get("split") for row in video_rows}
        labels = {row.get("label") for row in video_rows}
        if not video or len(video_rows) != 2:
            errors.append(f"Expected two temporal mosaics for video {video!r}, found {len(video_rows)}")
        if len(splits) != 1 or len(labels) != 1:
            errors.append(f"Video crosses split or class boundary: {video!r}")
        for sample_index, row in enumerate(video_rows):
            timestamps = row.get("sample_times_seconds", "").split(";")
            offsets = row.get("offsets_seconds", "").split(";")
            if len(timestamps) != 4 or len(offsets) != 4:
                errors.append(f"Video {video!r} sample {sample_index} does not contain four frames")
            expected = (
                root
                / row.get("split", "")
                / row.get("label", "")
                / f"{Path(video).stem}_mosaic_{sample_index:04d}.jpg"
            )
            if not expected.exists():
                errors.append(f"Manifest image is missing: {expected}")
    label_counts = {
        label: sum(1 for row in rows if row.get("label") == label)
        for label in {row.get("label") for row in rows}
    }
    if label_counts.get("accident") != label_counts.get("normal"):
        errors.append(f"Accident classes are imbalanced: {label_counts}")
    return errors


def validate(model_name: str) -> dict:
    config = load_config(model_name)
    root = ROOT / config["data"]
    if config["task"] == "detect":
        data = yaml.safe_load(root.read_text(encoding="utf-8"))
        dataset_root = Path(data["path"])
        counts = {}
        errors = []
        for split in SPLITS:
            images = list((dataset_root / "images" / split).glob("*"))
            labels = dataset_root / "labels" / split
            counts[split] = len(images)
            for image_path in images:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, SyntaxError, ValueError) as exc:
                    errors.append(f"{image_path}: {exc}")
                label_path = labels / f"{image_path.stem}.txt"
                if not label_path.exists():
                    errors.append(f"Missing label: {image_path.name}")
                    continue
                for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                    parts = line.split()
                    try:
                        class_id = int(parts[0])
                        coordinates = [float(value) for value in parts[1:]]
                    except (IndexError, ValueError):
                        errors.append(f"Invalid label syntax: {label_path}:{line_number}")
                        continue
                    if len(parts) != 5 or class_id != 0:
                        errors.append(f"Invalid label shape/class: {label_path}:{line_number}")
                    elif (
                        any(value < 0 or value > 1 for value in coordinates)
                        or coordinates[2] <= 0
                        or coordinates[3] <= 0
                    ):
                        errors.append(f"Invalid normalized box: {label_path}:{line_number}")
    else:
        errors = []
        counts = {}
        classes = set()
        split_classes = {}
        for split in SPLITS:
            files = [
                path
                for path in (root / split).rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
            counts[split] = len(files)
            split_classes[split] = {path.parent.name for path in files}
            classes.update(split_classes[split])
            for image_path in files:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, SyntaxError, ValueError) as exc:
                    errors.append(f"{image_path}: {exc}")
        if len(classes) < 2:
            errors.append(f"Expected at least two classes, found {sorted(classes)}")
        for split in SPLITS:
            missing = classes - split_classes[split]
            if missing:
                errors.append(f"Split {split} is missing classes: {sorted(missing)}")
        if model_name == "accident":
            errors.extend(validate_accident_manifest(root, sum(counts.values())))
    report = {"model": model_name, "task": config["task"], "counts": counts, "errors": errors}
    report_dir = DATASETS / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{model_name}_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors or not counts.get("train") or not counts.get("val"):
        update_status(model_name, "validate", "failed", report)
        raise SystemExit(json.dumps(report, indent=2))
    update_status(model_name, "validate", "complete", report)
    return report


def epoch_status_callback(model_name: str, requested_epochs: int):
    """Build an Ultralytics callback that records a durable epoch checkpoint."""

    def record_epoch(trainer) -> None:
        raw_metrics = getattr(trainer, "metrics", {}) or {}
        metrics = {}
        for name, value in raw_metrics.items():
            try:
                metrics[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        save_dir = Path(getattr(trainer, "save_dir", ML_ROOT / "runs" / model_name))
        update_status(
            model_name,
            "train",
            "in_progress",
            {
                "epoch_completed": int(getattr(trainer, "epoch", -1)) + 1,
                "epochs": int(getattr(trainer, "epochs", requested_epochs)),
                "last_checkpoint": str(save_dir / "weights" / "last.pt"),
                "metrics": metrics,
            },
        )

    return record_epoch


def train(model_name: str, epochs: int | None, device: str, resume: bool = False) -> Path:
    from ultralytics import YOLO

    config = load_config(model_name)
    validate(model_name)
    requested_epochs = epochs or config["epochs"]
    last_checkpoint = ML_ROOT / "runs" / model_name / "weights" / "last.pt"
    if resume:
        if not last_checkpoint.exists():
            raise SystemExit(f"Cannot resume; missing checkpoint: {last_checkpoint}")
        model = YOLO(str(last_checkpoint))
        update_status(model_name, "train", "in_progress", {"resume_from": str(last_checkpoint)})
    else:
        model = YOLO(config["base_model"])
        update_status(model_name, "train", "in_progress", {"epochs": requested_epochs, "device": device})
    # Fit-end runs after validation; train-end fires too early and exposes zeroed
    # validation metrics even though the epoch checkpoint itself is valid.
    model.add_callback("on_fit_epoch_end", epoch_status_callback(model_name, requested_epochs))
    if resume:
        result = model.train(resume=True, device=device)
    else:
        result = model.train(
            data=str((ROOT / config["data"]).resolve()),
            epochs=requested_epochs,
            imgsz=config["image_size"],
            batch=config["batch"],
            patience=config["patience"],
            optimizer=config.get("optimizer", "auto"),
            lr0=config.get("learning_rate", 0.01),
            dropout=config.get("dropout", 0.0),
            auto_augment=config.get("auto_augment", "randaugment"),
            erasing=config.get("erasing", 0.4),
            fliplr=config.get("fliplr", 0.5),
            scale=config.get("scale", 0.5),
            device=device,
            project=str((ML_ROOT / "runs").resolve()),
            name=model_name,
            exist_ok=True,
            seed=20260907,
            deterministic=True,
            workers=2,
        )
    best_checkpoint = Path(result.save_dir) / "weights" / "best.pt"
    update_status(model_name, "train", "complete", {"checkpoint": str(best_checkpoint)})
    return best_checkpoint


def evaluate(model_name: str, device: str) -> dict:
    from ultralytics import YOLO

    checkpoint = ML_ROOT / "runs" / model_name / "weights" / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    config = load_config(model_name)
    model = YOLO(str(checkpoint))
    metrics = model.val(data=str((ROOT / config["data"]).resolve()), split="test", device=device)
    results = metrics.results_dict
    if config["task"] == "classify":
        test_root = ROOT / config["data"] / "test"
        image_paths = sorted(
            path
            for path in test_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        predictions = model.predict(
            [str(path) for path in image_paths],
            batch=config["batch"],
            device=device,
            verbose=False,
        )
        positive_hints = [value.lower() for value in config["positive_classes"]]
        tp = tn = fp = fn = 0
        video_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        for image_path, prediction in zip(image_paths, predictions, strict=True):
            actual_positive = any(hint in image_path.parent.name.lower() for hint in positive_hints)
            predicted_label = str(prediction.names[int(prediction.probs.top1)]).lower()
            predicted_positive = any(hint in predicted_label for hint in positive_hints)
            positive_score = sum(
                float(prediction.probs.data[class_id])
                for class_id, class_name in prediction.names.items()
                if any(hint in str(class_name).lower() for hint in positive_hints)
            )
            source_stem = image_path.stem.rsplit("_", 1)[0]
            video_scores[(image_path.parent.name, source_stem)].append(positive_score)
            if actual_positive and predicted_positive:
                tp += 1
            elif actual_positive:
                fn += 1
            elif predicted_positive:
                fp += 1
            else:
                tn += 1
        video_tp = video_tn = video_fp = video_fn = 0
        for (label, _), scores in video_scores.items():
            actual_positive = any(hint in label.lower() for hint in positive_hints)
            predicted_positive = sum(scores) / len(scores) >= 0.5
            if actual_positive and predicted_positive:
                video_tp += 1
            elif actual_positive:
                video_fn += 1
            elif predicted_positive:
                video_fp += 1
            else:
                video_tn += 1
        video_total = video_tp + video_tn + video_fp + video_fn
        results = {
            **results,
            "metrics/precision": tp / (tp + fp) if tp + fp else 0.0,
            "metrics/recall": tp / (tp + fn) if tp + fn else 0.0,
            "metrics/specificity": tn / (tn + fp) if tn + fp else 0.0,
            "test/tp": tp,
            "test/tn": tn,
            "test/fp": fp,
            "test/fn": fn,
            "video/accuracy": (video_tp + video_tn) / video_total if video_total else 0.0,
            "video/precision": video_tp / (video_tp + video_fp) if video_tp + video_fp else 0.0,
            "video/recall": video_tp / (video_tp + video_fn) if video_tp + video_fn else 0.0,
            "video/specificity": video_tn / (video_tn + video_fp) if video_tn + video_fp else 0.0,
            "video/tp": video_tp,
            "video/tn": video_tn,
            "video/fp": video_fp,
            "video/fn": video_fn,
        }
        gate_metrics = ("metrics/accuracy_top1", "metrics/precision", "metrics/recall")
    else:
        gate_metrics = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)")
    gate_scores = {name: float(results.get(name, 0.0)) for name in gate_metrics}
    quality_gate = {
        "minimum_exclusive": MINIMUM_HELD_OUT_SCORE,
        "scores": gate_scores,
        "passed": all(score > MINIMUM_HELD_OUT_SCORE for score in gate_scores.values()),
    }
    output = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "results": results,
        "quality_gate": quality_gate,
    }
    report_dir = ML_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{model_name}_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    update_status(
        model_name,
        "evaluate",
        "complete" if quality_gate["passed"] else "quality_gate_failed",
        output,
    )
    return output


def export(model_name: str) -> Path:
    checkpoint = ML_ROOT / "runs" / model_name / "weights" / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    report_path = ML_ROOT / "reports" / f"{model_name}_metrics.json"
    if not report_path.exists():
        raise SystemExit(f"Evaluate before export; missing report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("quality_gate", {}).get("passed"):
        raise SystemExit(
            f"Export blocked: {model_name} has not exceeded the {MINIMUM_HELD_OUT_SCORE:.0%} held-out quality gate"
        )
    destination = ROOT / "models" / model_name / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)
    config = load_config(model_name)
    (destination.parent / "model_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    update_status(model_name, "export", "complete", {"checkpoint": str(destination)})
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="OneEye five-model ML pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sources")
    acquire_parser = sub.add_parser("acquire")
    acquire_parser.add_argument("--dataset", required=True, choices=load_registry().keys())
    acquire_parser.add_argument("--accept-license", action="store_true")
    acquire_parser.add_argument("--max-images", type=int)
    for command in ("prepare", "validate", "train", "evaluate", "export"):
        item = sub.add_parser(command)
        item.add_argument("--model", required=True, choices=MODEL_NAMES)
        if command == "train":
            item.add_argument("--epochs", type=int)
            item.add_argument("--resume", action="store_true")
        if command in {"train", "evaluate"}:
            item.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.command == "sources":
        print(yaml.safe_dump(load_registry(), sort_keys=False))
    elif args.command == "acquire":
        acquire(args.dataset, args.accept_license, args.max_images)
    elif args.command == "prepare":
        prepare(args.model)
    elif args.command == "validate":
        print(json.dumps(validate(args.model), indent=2))
    elif args.command == "train":
        print(train(args.model, args.epochs, args.device, args.resume))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.model, args.device), indent=2))
    elif args.command == "export":
        print(export(args.model))


if __name__ == "__main__":
    main()
