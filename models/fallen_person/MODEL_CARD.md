# OneEye Fallen-Person Classifier Model Card

## Purpose

This Ultralytics YOLO11n image classifier identifies frames that contain a fallen person. It is loaded by the OneEye inference service as the `fallen_person` event model.

## Training data

- Source: UR Fall Detection Dataset
- Source licence: CC BY-NC-SA 4.0
- Scope: five normal activity-of-daily-living sequences and five fall sequences from RGB camera 0
- Prepared frames: 264
- Split: 159 train, 56 validation, 49 held-out test images
- Classes: `fallen_person`, `normal`
- Split policy: sequence-disjoint and stratified, so frames from one source sequence never cross splits

## Training configuration

- Architecture: YOLO11n classification
- Input size: 224 pixels
- Batch size: 16
- Maximum epochs: 40
- Completed epoch: 11, stopped by the configured patience rule
- Hardware: NVIDIA GeForce RTX 5050 Laptop GPU

## Held-out test metrics

| Metric | Result |
|---|---:|
| Top-1 accuracy | 1.0000 |
| Top-5 accuracy | 1.0000 |

The test set contains frames from held-out sequences, but it is small and comes from one controlled dataset. This score must not be represented as production performance or as proof of generalization to arbitrary camera angles.

## Runtime contract

- Checkpoint: `models/fallen_person/best.pt`
- Ultralytics task: `classify`
- Class mapping: `{0: fallen_person, 1: normal}`
- Output: one `fallen_person` event prediction when the positive class exceeds the configured threshold
- Negative behavior: a `normal` top prediction is discarded by the inference adapter

## Limitations

The source environment, subjects, viewpoints, and acted falls are limited. Similar body postures, occlusion, crowded scenes, and camera viewpoints unlike UR Fall may cause errors. Representative local-camera testing, temporal confirmation, and threshold calibration remain required before operational use.
