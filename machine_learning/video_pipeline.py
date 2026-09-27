from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R3D_18_Weights, r3d_18
from tqdm import tqdm

from machine_learning.pipeline import (
    DATASETS,
    MINIMUM_HELD_OUT_SCORE,
    ML_ROOT,
    ROOT,
    update_status,
)

SEED = 20260908
KINETICS_MEAN = torch.tensor((0.43216, 0.394666, 0.37645)).view(3, 1, 1, 1)
KINETICS_STD = torch.tensor((0.22803, 0.22145, 0.216989)).view(3, 1, 1, 1)


@dataclass(frozen=True)
class VideoRecord:
    path: Path
    group: str
    label: int
    official_test: bool


def fight_records() -> dict[str, list[VideoRecord]]:
    raw = DATASETS / "raw" / "rwf_2000_hf"
    with (raw / "selected_videos.csv").open(encoding="utf-8", newline="") as handle:
        selected = list(csv.DictReader(handle))
    records = []
    development_groups: dict[int, list[str]] = {0: [], 1: []}
    for row in selected:
        label_name = row["local_label"].lower()
        label = 1 if label_name.startswith("fight") else 0
        stem_parts = Path(row["filename"]).stem.rsplit("_", 1)
        source_stem = stem_parts[0] if len(stem_parts) == 2 and stem_parts[1].isdigit() else Path(row["filename"]).stem
        class_name = "fight" if label else "normal"
        group = f"{class_name}/{source_stem}"
        official_test = "/val/" in row["source_path"].replace("\\", "/")
        records.append(VideoRecord(raw / row["local_label"] / row["filename"], group, label, official_test))
        if not official_test and group not in development_groups[label]:
            development_groups[label].append(group)
    validation_groups = set()
    for label, groups in development_groups.items():
        random.Random(SEED + label).shuffle(groups)
        validation_groups.update(groups[: max(1, round(len(groups) * 0.15))])
    return {
        "train": [record for record in records if not record.official_test and record.group not in validation_groups],
        "val": [record for record in records if not record.official_test and record.group in validation_groups],
        "test": [record for record in records if record.official_test],
    }


def _center_crop_resize(frame: np.ndarray, image_size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    y1 = (height - side) // 2
    x1 = (width - side) // 2
    crop = frame[y1 : y1 + side, x1 : x1 + side]
    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)


def decode_windows(
    path: Path,
    frames_per_clip: int,
    image_size: int,
    clips_per_video: int,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < frames_per_clip:
        capture.release()
        return []
    window_size = min(frame_count, max(frames_per_clip, round(frame_count / clips_per_video)))
    starts = [round(value) for value in np.linspace(0, frame_count - window_size, clips_per_video)]
    window_targets = [
        [round(value) for value in np.linspace(start, start + window_size - 1, frames_per_clip)] for start in starts
    ]
    targets_set = {target for targets in window_targets for target in targets}
    decoded = {}
    index = 0
    while index <= max(targets_set):
        ok, frame = capture.read()
        if not ok:
            break
        if index in targets_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded[index] = _center_crop_resize(rgb, image_size)
        index += 1
    capture.release()
    return [
        np.stack([decoded[target] for target in targets])
        for targets in window_targets
        if all(target in decoded for target in targets)
    ]


def prepare_fight(frames_per_clip: int = 16, image_size: int = 112, clips_per_video: int = 4) -> dict:
    output = DATASETS / "processed" / "fight_video"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    split_records = fight_records()
    report = {
        "model": "fight",
        "frames_per_clip": frames_per_clip,
        "clips_per_video": clips_per_video,
        "sampling": "short_window_sliding",
        "image_size": image_size,
        "splits": {},
    }
    update_status("fight", "video_prepare", "in_progress", report)
    for split, records in split_records.items():
        clips = np.lib.format.open_memmap(
            output / f"{split}_clips.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(records) * clips_per_video, frames_per_clip, image_size, image_size, 3),
        )
        labels = np.empty(len(records) * clips_per_video, dtype=np.uint8)
        index_rows = []
        saved = 0
        for record in tqdm(records, desc=f"fight video {split}"):
            windows = decode_windows(record.path, frames_per_clip, image_size, clips_per_video)
            for window_index, clip in enumerate(windows):
                clips[saved] = clip
                labels[saved] = record.label
                index_rows.append(
                    {
                        "row": saved,
                        "video": record.path.name,
                        "group": record.group,
                        "label": record.label,
                        "window": window_index,
                    }
                )
                saved += 1
        clips.flush()
        del clips
        if saved != len(records):
            source = np.load(output / f"{split}_clips.npy", mmap_mode="r")
            compact = np.lib.format.open_memmap(
                output / f"{split}_clips.compact.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(saved, frames_per_clip, image_size, image_size, 3),
            )
            compact[:] = source[:saved]
            compact.flush()
            del source, compact
            (output / f"{split}_clips.npy").unlink()
            (output / f"{split}_clips.compact.npy").replace(output / f"{split}_clips.npy")
        np.save(output / f"{split}_labels.npy", labels[:saved])
        with (output / f"{split}_index.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("row", "video", "group", "label", "window"))
            writer.writeheader()
            writer.writerows(index_rows)
        report["splits"][split] = {
            "requested_videos": len(records),
            "requested_clips": len(records) * clips_per_video,
            "saved_clips": saved,
        }
    (output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    update_status("fight", "video_prepare", "complete", report)
    return report


class ClipDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool):
        self.clips = np.load(root / f"{split}_clips.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels.npy")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip = torch.from_numpy(np.array(self.clips[index], copy=True)).permute(3, 0, 1, 2).float() / 255.0
        if self.augment:
            # Use one transform for every frame so the original motion stays coherent.
            if torch.rand(()) < 0.5:
                clip = torch.flip(clip, dims=(3,))
            crop_size = int(torch.randint(96, 113, ()).item())
            top = int(torch.randint(0, 113 - crop_size, ()).item())
            left = int(torch.randint(0, 113 - crop_size, ()).item())
            clip = functional.interpolate(
                clip[:, :, top : top + crop_size, left : left + crop_size].permute(1, 0, 2, 3),
                size=(112, 112),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3)
            brightness = 0.85 + float(torch.rand(())) * 0.30
            contrast = 0.85 + float(torch.rand(())) * 0.30
            channel_mean = clip.mean(dim=(1, 2, 3), keepdim=True)
            clip = ((clip - channel_mean) * contrast + channel_mean) * brightness
            clip = clip.clamp(0.0, 1.0)
        clip = (clip - KINETICS_MEAN) / KINETICS_STD
        return clip, torch.tensor(int(self.labels[index]), dtype=torch.long)


class VideoClipDataset(ClipDataset):
    def __init__(self, root: Path, split: str, augment: bool):
        super().__init__(root, split, augment)
        with (root / f"{split}_index.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        grouped: dict[str, dict] = {}
        for row in rows:
            key = f"{row['label']}/{row['video']}"
            item = grouped.setdefault(key, {"label": int(row["label"]), "rows": []})
            item["rows"].append(int(row["row"]))
        self.videos = list(grouped.values())

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        video = self.videos[index]
        clips = [super(VideoClipDataset, self).__getitem__(row)[0] for row in video["rows"]]
        return torch.stack(clips), torch.tensor(video["label"], dtype=torch.long)


def _loader(root: Path, split: str, batch_size: int, workers: int, augment: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        ClipDataset(root, split, augment),
        batch_size=batch_size,
        shuffle=augment,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


def _video_loader(root: Path, split: str, batch_size: int, workers: int, augment: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        VideoClipDataset(root, split, augment),
        batch_size=batch_size,
        shuffle=augment,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


def _aggregate_window_logits(logits: torch.Tensor, windows: int) -> torch.Tensor:
    margins = (logits[:, 1] - logits[:, 0]).reshape(-1, windows)
    temperature = 5.0
    bag_margin = (torch.logsumexp(margins * temperature, dim=1) - np.log(windows)) / temperature
    return torch.stack((-bag_margin / 2.0, bag_margin / 2.0), dim=1)


def _video_epoch(model, loader, device, criterion, optimizer=None, scaler=None) -> tuple[float, float, dict]:
    training = optimizer is not None
    model.train(training)
    if training:
        for stage_name in getattr(model, "_frozen_stage_names", ()):
            getattr(model, stage_name).eval()
    total_loss = correct = count = tp = tn = fp = fn = 0
    confidences = []
    for clips, labels in loader:
        batch, windows, channels, frames, height, width = clips.shape
        clips = clips.reshape(batch * windows, channels, frames, height, width).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = _aggregate_window_logits(model(clips), windows)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        probabilities = torch.softmax(logits.detach(), dim=1)
        predicted = probabilities.argmax(1)
        confidences.extend(probabilities.max(1).values.cpu().tolist())
        total_loss += float(loss.detach()) * len(labels)
        correct += int((predicted == labels).sum())
        count += len(labels)
        for actual, guess in zip(labels.cpu().tolist(), predicted.cpu().tolist(), strict=True):
            if actual == 1 and guess == 1:
                tp += 1
            elif actual == 1:
                fn += 1
            elif guess == 1:
                fp += 1
            else:
                tn += 1
    scores = {
        "accuracy": (tp + tn) / count if count else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }
    details = {
        "scores": scores,
        "mean_top1_confidence": float(np.mean(confidences)),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "videos": count,
    }
    return total_loss / count, correct / count, details


def _epoch(model, loader, device, criterion, optimizer=None, scaler=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        for stage_name in getattr(model, "_frozen_stage_names", ()):
            getattr(model, stage_name).eval()
    total_loss = correct = count = 0
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(clips)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.detach()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        count += len(labels)
    return total_loss / count, correct / count


def _video_scores(model: nn.Module, loader: DataLoader, index_path: Path, device: torch.device) -> dict:
    window_probabilities = []
    model.eval()
    with torch.inference_mode():
        for clips, _labels in loader:
            probabilities = torch.softmax(model(clips.to(device, non_blocking=True)), dim=1)
            window_probabilities.extend(probabilities[:, 1].cpu().tolist())
    with index_path.open(encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(index_rows) != len(window_probabilities):
        raise RuntimeError("Video index and predicted window counts do not match")
    videos: dict[str, dict] = {}
    for row, fight_probability in zip(index_rows, window_probabilities, strict=True):
        video_key = f"{row['label']}/{row['video']}"
        video = videos.setdefault(video_key, {"label": int(row["label"]), "probabilities": []})
        video["probabilities"].append(fight_probability)
    tp = tn = fp = fn = 0
    confidences = []
    for video in videos.values():
        fight_probability = float(np.mean(video["probabilities"]))
        predicted = int(fight_probability >= 0.5)
        actual = video["label"]
        confidences.append(max(fight_probability, 1.0 - fight_probability))
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 1:
            fn += 1
        elif predicted == 1:
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    return {
        "scores": {
            "accuracy": (tp + tn) / total if total else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
        },
        "mean_top1_confidence": float(np.mean(confidences)),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "videos": total,
    }


def build_model(pretrained: bool) -> nn.Module:
    model = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 2))
    return model


def train_fight(epochs: int, batch_size: int, workers: int, device_name: str) -> Path:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(f"cuda:{device_name}" if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    root = DATASETS / "processed" / "fight_video"
    train_loader = _video_loader(root, "train", batch_size, workers, True)
    val_loader = _video_loader(root, "val", batch_size, workers, False)
    model = build_model(pretrained=True).to(device)
    # General Kinetics motion features are reusable. Freezing early blocks reduces
    # memorisation of RWF camera/background details on this modest-sized corpus.
    for module in (model.stem, model.layer1, model.layer2):
        for parameter in module.parameters():
            parameter.requires_grad = False
    model._frozen_stage_names = ("stem", "layer1", "layer2")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer3.parameters(), "lr": 1e-5},
            {"params": model.layer4.parameters(), "lr": 3e-5},
            {"params": model.fc.parameters(), "lr": 2e-4},
        ],
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_dir = ML_ROOT / "runs" / "fight_video"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.pt"
    history = []
    best_accuracy = 0.0
    stale_epochs = 0
    update_status("fight", "video_train", "in_progress", {"epochs": epochs, "device": str(device)})
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy, _train_details = _video_epoch(
            model, train_loader, device, criterion, optimizer, scaler
        )
        with torch.inference_mode():
            val_loss, val_accuracy, validation = _video_epoch(model, val_loader, device, criterion)
        val_video_accuracy = validation["scores"]["accuracy"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_video_accuracy": val_video_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_video_accuracy > best_accuracy:
            best_accuracy = val_video_accuracy
            stale_epochs = 0
            torch.save(
                {
                    "architecture": "r3d_18",
                    "model_state": model.state_dict(),
                    "classes": ["normal", "fight"],
                    "frames_per_clip": 16,
                    "image_size": 112,
                    "val_accuracy": val_video_accuracy,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        scheduler.step()
        if stale_epochs >= 7:
            break
    update_status("fight", "video_train", "complete", {"checkpoint": str(checkpoint), "best_accuracy": best_accuracy})
    return checkpoint


def evaluate_fight(batch_size: int, workers: int, device_name: str) -> dict:
    device = torch.device(f"cuda:{device_name}" if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    checkpoint = ML_ROOT / "runs" / "fight_video" / "best.pt"
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(saved["model_state"])
    loader = _video_loader(DATASETS / "processed" / "fight_video", "test", batch_size, workers, False)
    criterion = nn.CrossEntropyLoss()
    _loss, _accuracy, aggregated = _video_epoch(model, loader, device, criterion)
    scores = aggregated["scores"]
    report = {
        "model": "fight",
        "checkpoint": str(checkpoint),
        "scores": scores,
        "mean_top1_confidence": aggregated["mean_top1_confidence"],
        "confusion": aggregated["confusion"],
        "videos": aggregated["videos"],
        "aggregation": "multiple_instance_log_mean_exp_across_short_windows",
        "quality_gate": {
            "minimum_exclusive": MINIMUM_HELD_OUT_SCORE,
            "passed": all(scores[name] > MINIMUM_HELD_OUT_SCORE for name in ("accuracy", "precision", "recall")),
        },
    }
    report_path = ML_ROOT / "reports" / "fight_video_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    update_status(
        "fight",
        "video_evaluate",
        "complete" if report["quality_gate"]["passed"] else "quality_gate_failed",
        report,
    )
    return report


def export_fight() -> Path:
    report = json.loads((ML_ROOT / "reports" / "fight_video_metrics.json").read_text(encoding="utf-8"))
    if not report["quality_gate"]["passed"]:
        raise SystemExit("Export blocked: fight video model has not exceeded the 95% held-out quality gate")
    source = ML_ROOT / "runs" / "fight_video" / "best.pt"
    destination = ROOT / "models" / "fight" / "video_best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    config = {
        "model_name": "fight",
        "runtime": "torchvision_video",
        "architecture": "r3d_18",
        "frames_per_clip": 16,
        "image_size": 112,
        "classes": ["normal", "fight"],
    }
    import yaml

    (destination.parent / "video_model_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    update_status("fight", "video_export", "complete", {"checkpoint": str(destination)})
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="OneEye temporal video training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=30)
    for command in (train_parser,):
        command.add_argument("--batch", type=int, default=8)
        command.add_argument("--workers", type=int, default=2)
        command.add_argument("--device", default="0")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--batch", type=int, default=8)
    evaluate_parser.add_argument("--workers", type=int, default=2)
    evaluate_parser.add_argument("--device", default="0")
    sub.add_parser("export")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_fight(), indent=2))
    elif args.command == "train":
        print(train_fight(args.epochs, args.batch, args.workers, args.device))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_fight(args.batch, args.workers, args.device), indent=2))
    elif args.command == "export":
        print(export_fight())


if __name__ == "__main__":
    main()
