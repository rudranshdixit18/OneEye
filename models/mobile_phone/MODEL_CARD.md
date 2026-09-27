# OneEye Mobile-Phone Detector Model Card

## Purpose

This detector identifies visible mobile phones in surveillance frames. It is loaded by the OneEye inference service as the `mobile_phone` event model.

## Training data

- Source: Open Images V7/V6 bounding-box distribution
- Licence: per-image licence metadata distributed by Open Images; review is required before redistribution
- Prepared subset: 300 images across the official train, validation, and test partitions
- Split composition: 80 phone-positive and 20 controlled background images in each split
- Held-out test annotations: 102 phone bounding boxes across 80 positive images
- Label transformation: the Open Images `Mobile phone` class is mapped to `mobile_phone`; background images receive empty YOLO label files

## Training configuration

- Architecture: repository-supplied one-class YOLO detector, fine-tuned on the prepared Open Images subset
- Input size: 640 pixels
- Batch size: 8
- Epochs: 20
- Optimizer: AdamW
- Initial learning rate: 0.0001
- Hardware: NVIDIA GeForce RTX 5050 Laptop GPU

## Held-out test metrics

| Metric | Result |
|---|---:|
| Precision | 0.8454 |
| Recall | 0.7451 |
| mAP@50 | 0.8379 |
| mAP@50-95 | 0.6690 |

The metrics describe this controlled 300-image development subset and must not be represented as universal surveillance-camera performance.

## Runtime contract

- Checkpoint: `models/mobile_phone/best.pt`
- Ultralytics task: `detect`
- Class mapping: `{0: phone}`
- Output: confidence and pixel bounding box for each accepted phone detection
- Default threshold: controlled by the backend settings API and dashboard

## Threshold calibration

On the 80 positive and 20 background test images, an 0.85 confidence threshold produced an image-level hit rate of 58.8% with one background false positive (5%). At 0.90, the measured background false-positive rate fell to 0% but image-level recall fell to 40%. OneEye therefore defaults to 0.85 and adds two-frame temporal confirmation; operators can adjust the threshold from the dashboard for their camera environment.

## Limitations

Very small, occluded, distant, screen-only, or unusually shaped phones may be missed. Objects with a similar rectangular appearance may cause false positives. Camera-specific threshold calibration and representative field testing remain required.
