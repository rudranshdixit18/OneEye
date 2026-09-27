# OneEye Implementation Progress

Last updated: 13 September 2026; guarded single-GPU training sequence complete

| Area | Status | Evidence |
|---|---|---|
| Frontend audit | DONE | `documentation/FRONTEND_AUDIT.md` |
| Project folder separation | DONE | `backend`, `FrontEnd`, `machine_learning`, `models`, `storage`, `scripts`, `tests`, `documentation` |
| Five-event API contract | DONE | `backend/app/api/routes.py` |
| Single-reader camera pipeline | DONE | `backend/app/services/camera.py` and `pipeline.py` |
| Detector/classifier adapter | DONE | `backend/app/services/inference.py` |
| SQLite event/settings storage | DONE | `backend/app/core/database.py` |
| Existing frontend integration | DONE | Browser-validated dashboard, navigation, live API state, event empty states, analytics, and persisted thresholds |
| Dataset source registry | DONE | `machine_learning/dataset_registry.yaml` |
| Acquisition/preparation pipeline | DONE | Full TACO, MJU-Waste, RWF-2000, Open Images phone, UR Fall, and all 1,500 Nexar videos are acquired, prepared, and validated |
| Model training | DONE | All five candidates completed guarded held-out evaluation; only expanded fallen-person passed and was exported, while fight, garbage, expanded mobile-phone, and accident were preserved as research artifacts |
| End-to-end verification | DONE | CUDA, Ruff, 18 tests, JavaScript syntax, five dataset validators, live APIs, five-model loading, upload inference, browser navigation, physical camera lifecycle, 1080p snapshot, and MJPEG streaming all passed |

## Verified environment

- Python 3.11 virtual environment: `.venv`
- PyTorch: `2.14.0+cu130`
- CUDA runtime visible to PyTorch: `13.0`
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU
- Automated verification: Ruff passes, JavaScript syntax passes, and all 18 Pytest tests pass
- Dataset verification: all five validators pass with zero reported structural, corruption, label, cardinality, or split-boundary errors
- Live API verification: health, model status, cameras, events, analytics, settings, frontend delivery, upload inference, snapshot, and MJPEG routes passed; invalid event names return 422 and unsupported media returns 415
- Runtime checkpoint availability: all five checkpoints load without error; the expanded fallen-person model is the only newly trained candidate that passed the strict held-out gate, while the other runtime checkpoints are retained functional baselines
- Physical camera verification: OpenCV opened camera 0 at 1920×1080. Although 60 FPS is requested and reported by the driver, measured capture was about 30 FPS without the full workload and 18.5–20.0 FPS during simultaneous five-model inference and MJPEG streaming; the UI reports the measured value rather than claiming 60 FPS
- Five-model live inference: a warmed API camera pass completed in 110.5 ms on the RTX 5050 Laptop GPU; the displayed measurement can vary with streaming and system load
- Browser verification: Dashboard, Live Feed, Detections, Analytics, and Settings rendered correctly; Start/Stop controlled the physical camera; saved thresholds displayed correctly; no console warnings or errors were emitted
- Portable startup verification: `scripts/setup.ps1` discovers an available Python installation without a user-specific path, and `scripts/start.ps1` loads `.env` before starting the configured host and port
- Default analysis cadence: 4 FPS. The pipeline supports a four-frame temporal mosaic whenever a future gate-passing exported model declares `temporal_tiles: 4`; retained single-frame runtime baselines correctly declare 1

## Garbage detector status

- Historical baseline dataset: deterministic 300-image TACO subset with 209 train, 47 validation, and 44 held-out test images
- Historical baseline training: YOLO11n detector, early-stopped at epoch 50 from a configured maximum of 60 epochs
- Test precision: 0.6392
- Test recall: 0.3913
- Test mAP@50: 0.4522
- Test mAP@50-95: 0.2912
- Candidate checkpoint: `models/garbage/best.pt`
- Runtime check: checkpoint loads as a one-class `garbage` detector and produces no detection on a blank image
- Expanded TACO-only dataset: all 1,500 official TACO records; 1,050 train, 225 validation, and 225 untouched test images
- TACO-only YOLO11s result: early-stopped at epoch 94, restoring best epoch 74
- TACO-only untouched-test result: precision 0.7324, recall 0.4240, mAP@50 0.5003, and mAP@50-95 0.3459
- Promotion result: blocked because the TACO-only candidate does not meet the greater-than-95% held-out gate
- Preserved research artifact: `machine_learning/runs/garbage_taco_only_20260909/`
- Additional source: complete official MJU-Waste v1.0 archive (1,485 train, 248 validation, 742 test RGB images) and COCO annotations acquired with a SHA-256 manifest
- Combined-candidate rule: 1,477 annotated MJU training images augment detector training; eight unannotated training images are excluded, and the original 225-image TACO validation and 225-image TACO test sets stay unchanged
- MJU label rule: polygon segmentation geometry is preferred over inconsistent exported boxes and is clipped to real image bounds
- Combined validation: 2,527 train, 225 validation, and 225 test image/label pairs pass with zero integrity errors
- Combined training: YOLO11s on the combined data, 768px, batch 8; all 100 configured epochs completed
- Best individual validation figures: precision 0.7720, recall 0.4774, mAP@50 0.5121, and mAP@50-95 0.3811; these peaks occurred across different epochs and are not test results
- Combined untouched-test result: precision 0.6754, recall 0.4449, mAP@50 0.4904, and mAP@50-95 0.3540
- Promotion result: blocked because the combined candidate does not meet the greater-than-95% held-out gate; the previous runtime checkpoint remains unchanged

## Fallen-person dataset gate

- Historical subset: five UR Fall ADL sequences and five UR Fall fall sequences, producing 264 frames
- Expanded source: twenty ADL and twenty fall RGB sequences plus both official frame-posture label tables
- Prepared classification frames: 937 train, 185 validation, and 163 untouched test images
- Class counts: train 372 fallen/565 normal, validation 59 fallen/126 normal, test 45 fallen/118 normal
- Labelling rule: official posture `1` is `fallen_person`, `-1` is `normal`, and ambiguous transition `0` frames are excluded
- Class coverage: both `fallen_person` and `normal` are present in every split
- Validation result: zero missing, corrupt, or structurally invalid images
- Executed training configuration: YOLO11s classifier, 320px, batch 16, maximum 60 epochs, early stopping patience 15

## Fallen-person classifier result

- Expanded training: YOLO11s classifier, early-stopped at epoch 16 from a configured maximum of 60 epochs
- Untouched frame-level test result across 163 frames: 1.0000 accuracy, 1.0000 precision, 1.0000 recall, and 1.0000 specificity (45 TP, 118 TN, 0 FP, 0 FN)
- Untouched sequence-level result across nine held-out videos: 1.0000 accuracy, 1.0000 precision, 1.0000 recall, and 1.0000 specificity (3 TP, 6 TN, 0 FP, 0 FN)
- Promotion result: passed the strict greater-than-95% held-out gate and exported
- Exported checkpoint: `models/fallen_person/best.pt`
- Runtime check: a held-out fall frame emits `fallen_person`; a held-out normal frame emits no event
- Generalization caveat: the held-out sequences are from one controlled dataset, so local-camera and cross-dataset testing remain required

## Mobile-phone detector result

- Historical baseline dataset: 300 Open Images examples with 80 phone-positive and 20 controlled background images per split
- Training: conservative 20-epoch fine-tune from the compatible supplied phone detector
- Test precision: 0.8454
- Test recall: 0.7451
- Test mAP@50: 0.8379
- Test mAP@50-95: 0.6690
- Existing runtime checkpoint retained at `models/mobile_phone/best.pt`; it predates the current strict promotion gate
- Quality gate: a from-scratch one-class head was rejected after its precision collapsed; the final transfer-learning run outperformed the supplied baseline on the same held-out set
- Expanded dataset: 3,497 Open Images V7 examples preserving source boundaries: 2,166 train, 538 validation, and 793 untouched test images
- Expanded-data validation: all 3,497 image/label pairs pass integrity checks with no missing labels or corrupt images
- Expanded training: YOLO11s, 768px, batch 8, early-stopped at epoch 40 after restoring best epoch 25
- Recovery status: the first process was externally interrupted during epoch 16 on 12 September without a training error; on 13 September the run was resumed from `machine_learning/runs/mobile_phone/weights/last.pt`, preserving all 15 completed epochs
- Best individual validation figures: precision 0.8688, recall 0.8442, mAP@50 0.8819, and mAP@50-95 0.7889; these peaks occurred across different epochs and are not untouched-test results. Epoch 25 was the strongest single checkpoint, with precision 0.8587, recall 0.7853, mAP@50 0.8819, and mAP@50-95 0.7889
- Expanded untouched-test result across 793 images: precision 0.8731, recall 0.8510, mAP@50 0.8922, and mAP@50-95 0.7652
- Promotion result: blocked because precision, recall, and mAP@50 did not all exceed the strict 0.95 threshold; the existing runtime checkpoint remains unchanged

## Fight detector research results

- Dataset: all 2,000 RWF-2000 videos with the official 1,600-development/400-test boundary preserved
- Internal development split: 1,342 training and 258 validation videos, grouped to prevent source-family leakage
- R3D-18 full-video baseline on the 400 official test videos: 83.25% accuracy, 88.44% precision, 76.50% recall, and 90.00% specificity
- Four-window multiple-instance model on the same official test set: 83.75% accuracy, 91.41% precision, 74.50% recall, and 93.00% specificity
- Promotion result: blocked because accuracy, precision, and recall did not all exceed 95%
- Research checkpoints and reports remain in ignored `machine_learning/runs` and `machine_learning/reports` directories for reproducibility; no failed candidate was promoted

## Accident detector status

- Historical baseline: 30 collision and 30 normal Nexar videos sampled into 480 individual frames
- Historical held-out top-1 accuracy: 0.5250; this candidate fails the promotion gate
- Expanded source: the complete balanced Nexar training set of 750 collision and 750 normal videos (about 24 GB)
- Acquisition: complete; 1,500 selected videos, official event metadata, and a SHA-256 manifest are present
- Preparation rule: two ordered four-frame mosaics per video, centred on the official collision timestamp; normal videos use the same reference-time distribution
- Preparation implementation: one keyframe seek followed by compact sequential decoding captures all eight requested frames per video, preserving the sampling times while avoiding eight expensive random seeks
- Preparation result: all 1,500 videos produced exactly 3,000 mosaics in 15 minutes 27 seconds: 2,100 train, 448 validation, and 452 untouched test images
- Validation gate: verifies every image, exact two-mosaic-per-video cardinality, four ordered timestamps per mosaic, class balance, manifest/image agreement, and that no video crosses a split or class boundary
- Validation result: zero corrupt images, missing samples, cardinality errors, class imbalance, or split-boundary violations
- Executed training configuration: YOLO11s classifier, 320px, batch 16, maximum 60 epochs, early stopping patience 15
- Expanded training result: early-stopped after epoch 32 and restored best epoch 17, whose validation top-1 accuracy was 0.8125
- Untouched image-level test result across 452 mosaics: accuracy 0.7942, precision 0.7904, recall 0.8009, and specificity 0.7876 (181 TP, 178 TN, 48 FP, 45 FN)
- Untouched source-video result across 226 held-out videos: accuracy 0.8009, precision 0.7931, recall 0.8142, and specificity 0.7876 (92 TP, 89 TN, 24 FP, 21 FN)
- Promotion result: blocked because accuracy, precision, and recall did not all exceed the strict 0.95 threshold; the previous runtime checkpoint remains unchanged
- Research integration: the unpromoted expanded candidate declares `temporal_tiles: 4`, making the backend compose four ordered live-camera frames if that candidate later passes and is exported. The retained baseline runtime model correctly remains single-frame with `temporal_tiles: 1`

## Git checkpoints

- `a4e7a0c` — backend, frontend integration, and initial ML pipelines
- `934772a` — CUDA verification, runtime fixes, API smoke tests, and frontend browser QA
- `142a628` — garbage baseline and fall-dataset preparation
- `7a7de9c` — fall and phone model training
- `b0245e9` — event-aligned accident pipeline
- `5a89dbf` — strict quality gates and 1080p60-capable streaming
- `44b074b` — temporal video training pipeline
- `929a4a4` — full-RWF temporal benchmark
- `0504ee2` — physical-camera validation and fight multiple-instance learning
- `4b74df0` — full-TACO acquisition pipeline and resumable garbage training
- `25214dc` — expanded garbage split validation record
- `5901d56` — expanded 1,865-image phone dataset and validation record
- `3f79130` — authoritative frame-level UR Fall labelling and leakage regression test
- `a870c4a` — expanded 40-sequence fall dataset and stronger classifier configuration
- `265877a` — complete MJU-Waste acquisition and integrity validation
- `2c63462` — expanded 3,497-image mobile-phone dataset and validation record
- `fe04d45` — corrected MJU polygon conversion and zero-error combined garbage validation
- `c3f4d4e` — optimized full Nexar temporal preparation
- `9b34fd0` — atomic per-epoch training status checkpoints
- `06b4b15` — temporal-manifest and video-boundary validation gate
- `1548275` — complete 3,000-mosaic Nexar dataset validation
- `6438604` — complete garbage evaluation and promote the gate-passing fallen-person model
- `e1fa358` — preserve the recovered mobile-phone run through epoch 15
- `394fee7` — preserve the recovered mobile-phone run through epoch 20
- `1dbdb55` — preserve the recovered mobile-phone run through epoch 25
- `823294a` — preserve the recovered mobile-phone run through epoch 30
- `e3e7d16` — preserve the recovered mobile-phone run through epoch 35
- `c29e96d` — complete expanded mobile-phone evaluation and start accident training
- `74d6448` — preserve accident training through epoch 5
- `2181563` — preserve accident training through epoch 10
- `f8a61d4` — preserve accident training through epoch 15
- `adeaeb9` — preserve accident training through epoch 20
- `036718e` — preserve accident training through epoch 25
- `6d84f42` — preserve accident training through epoch 30
- `31f55d6` — complete the guarded five-model training and evaluation sequence

## Completed training and recovery record

The TACO-only garbage run is complete and preserved under `machine_learning/runs/garbage_taco_only_20260909/`. The combined run was interrupted during epoch 19, resumed from epoch 18, completed all 100 epochs, and failed the untouched-test promotion gate. Inspect it with:

```powershell
Set-Location "C:\Users\Rudransh Dixit\OneDrive\Desktop\One Eye"
Get-Content .\machine_learning\runs\garbage-combined.out.log -Tail 30
Get-Content .\machine_learning\runs\garbage-combined.err.log -Tail 30
Get-Content .\machine_learning\runs\garbage-resume-20260911-143003.err.log -Tail 30
Get-Content .\machine_learning\runs\garbage-resume-20260912-013358.out.log -Tail 30
Get-Content .\machine_learning\runs\garbage\results.csv -Tail 5
```

Windows Smart App Control was turned off by the user on 11 September 2026. OpenCV 4.14, PyTorch 2.14.0+cu130, CUDA, and the RTX 5050 were reverified before the sequence resumed. The sequential runner finished normally on 13 September 2026, restored permanent Windows sleep behaviour, and left no trainer running. Garbage, fallen-person, expanded mobile-phone, and accident training are complete and must not be restarted as though they were interrupted.

The completed recovery logs are `machine_learning/runs/mobile-phone-resume-20260913-082122.out.log`, `mobile-phone-resume-20260913-082122.err.log`, `remaining-sequence-20260913-082122-retry.out.log`, and `remaining-sequence-20260913-082122-retry.err.log`. The final queue log ends with `The configured OneEye training sequence completed.`

The runner does not infer success from a detached Windows process exit code. It requires the pipeline's atomic `training_status.json` marker to say `stage: train` and `status: complete` before it will evaluate the initial model, preventing a killed or crashed trainer from being treated as successful.

The epoch status callback is attached to Ultralytics `on_fit_epoch_end`, after validation, so the resumed garbage run and all subsequent runs persist real validation metrics alongside each `last.pt` path. `results.csv` remains the authoritative per-epoch record.

Full Nexar acquisition and temporal-mosaic preparation are complete. Inspect the validated result with:

```powershell
(Get-ChildItem .\machine_learning\datasets\processed\accident -Recurse -Filter *.jpg).Count
Get-Content .\machine_learning\datasets\reports\accident_validation.json
```

If the processed data is intentionally rebuilt, run:

```powershell
.\.venv\Scripts\python.exe -m machine_learning.pipeline prepare --model accident
```

Do not reinstall the environment or recreate the project. Review this file and `git log --oneline` before resuming.
