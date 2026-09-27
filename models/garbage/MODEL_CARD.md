# OneEye Garbage Detector Model Card

## Purpose

This Ultralytics YOLO11n object detector identifies visible garbage and litter in surveillance frames. It is loaded by the OneEye inference service as the `garbage` event model.

## Training data

- Source: Trash Annotations in Context (TACO)
- Source licence: CC BY 4.0
- Prepared subset: 300 images
- Split: 209 train, 47 validation, 44 held-out test images
- Label transformation: every valid TACO litter category is mapped to the single `garbage` class
- Split policy: deterministic and grouped to prevent the same source image from crossing splits

## Training configuration

- Architecture: YOLO11n detection
- Input size: 640 pixels
- Batch size: 8
- Maximum epochs: 60
- Completed epoch: 50, stopped by the configured patience rule
- Hardware: NVIDIA GeForce RTX 5050 Laptop GPU

## Held-out test metrics

| Metric | Result |
|---|---:|
| Precision | 0.6392 |
| Recall | 0.3913 |
| mAP@50 | 0.4522 |
| mAP@50-95 | 0.2912 |

These metrics describe the selected 300-image development subset and must not be represented as full-dataset or production performance.

## Runtime contract

- Checkpoint: `models/garbage/best.pt`
- Ultralytics task: `detect`
- Class mapping: `{0: garbage}`
- Output: confidence and pixel bounding box for each accepted detection
- Default threshold: controlled by the backend settings API and dashboard

## Limitations

The subset is small and visually diverse, so recall is currently limited. Small, occluded, distant, or unusual waste objects may be missed. Deployment acceptance requires representative local-camera validation and threshold calibration.
