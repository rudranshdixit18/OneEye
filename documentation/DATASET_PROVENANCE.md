# OneEye Dataset Provenance and Usage Register

Last reviewed: 10 September 2026

This register is the authoritative source list for OneEye model development. Raw media stays under `machine_learning/datasets/raw/`, is excluded from Git, and is never redistributed with the application. Each completed acquisition produces a SHA-256 manifest beside the raw data.

| OneEye model | Primary dataset | Official source | Usage terms | Pipeline status |
|---|---|---|---|---|
| Fight detection | RWF-2000 | https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection and https://zenodo.org/records/15687512 | Research/non-commercial; original dataset conditions apply | All 2,000 clips are acquired. The official 1,600-development/400-test boundary is preserved; R3D-18 full-video and four-window MIL candidates have been evaluated without passing the promotion gate. |
| Garbage detection | TACO plus MJU-Waste v1.0 | https://github.com/pedropro/TACO and https://github.com/realwecan/mju-waste | TACO: CC BY 4.0; MJU-Waste repository: MIT; both require source attribution | All 1,500 TACO records and all 2,475 MJU-Waste RGB images are acquired. MJU contributes 1,477 annotated official-training images; eight unannotated training images are excluded. TACO’s 225 validation and 225 test images remain unchanged. |
| Fallen-person detection | UR Fall Detection Dataset | https://fenix.ur.edu.pl/~mkepski/ds/uf.html | CC BY-NC-SA 4.0; non-commercial and share-alike | Twenty ADL and twenty fall RGB sequences are acquired and prepared into 1,285 posture-labelled frames: 937 train, 185 validation, and 163 untouched test. Official per-frame labels distinguish lying (`1`) from normal (`-1`); ambiguous transition (`0`) frames are excluded. |
| Mobile-phone detection | Open Images V7 `Mobile phone` subset | https://storage.googleapis.com/openimages/web/download_v7.html | Per-image licences must be retained and reviewed | A 3,497-image subset is acquired, prepared, and validated: 2,166 official-train, 538 official-validation, and 793 untouched official-test images. Human-verified phone boxes and controlled empty-label backgrounds retain their source split and licence metadata. |
| Accident detection | Nexar Collision Prediction Dataset | https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction | Nexar Open Data License; attribution required, no resale, ethical-use restrictions | The complete balanced source—750 collision and 750 normal dashcam videos—is acquired with official metadata and a SHA-256 manifest. All videos produce two event-aligned mosaics: 2,100 train, 448 validation, and 452 untouched test images in deterministic video-disjoint splits. |

## Processing rules

- Dataset splits are deterministic and video/sequence-disjoint to prevent leakage.
- TACO and Open Images annotations are transformed into Ultralytics YOLO bounding-box labels. Open Images background examples receive explicit empty label files.
- MJU-Waste polygon segmentations are converted to clipped one-class `garbage` boxes and added only to training. Polygon geometry is authoritative because some exported COCO `bbox` fields exceed image bounds. Its official validation/test images remain available for separate cross-dataset evaluation and never enter the TACO held-out splits.
- UR Fall, RWF-2000, and Nexar media are sampled without modifying source files.
- UR Fall uses the official `urfall-cam0-falls.csv` and `urfall-cam0-adls.csv` posture labels. Positive poses are sampled more densely than normal poses to reduce class imbalance, while sequence-level split boundaries prevent adjacent-frame leakage.
- RWF keeps the official test partition untouched. Development video families remain split-disjoint, and four coherent short windows are grouped into one multiple-instance training example.
- Nexar positive clips use the official event timestamp metadata to centre temporal samples around the collision. Negative clips use matched relative positions, reducing clip-position shortcuts.
- Training is blocked when train/validation data is missing, a class is absent, an image is corrupt, or a detection label is missing.
- Dataset terms and citations must accompany any public report or model release.

## Required attribution

- TACO: Proença, Pedro F. and Simões, Pedro. *TACO: Trash Annotations in Context for Litter Detection*.
- UR Fall: Kępski, Michal and Kwolek, Bogdan. *Fall Detection Using Ceiling-Mounted 3D Depth Camera*.
- RWF-2000: Cheng, Ming; Cai, Kunjing; and Li, Ming. *RWF-2000: An Open Large Scale Video Database for Violence Detection*.
- Nexar: Moura, Daniel C. and Zvitia, Orly. *Nexar Collision Dataset*, Nexar Inc., 2025.
- Open Images: Kuznetsova et al. *The Open Images Dataset V4* and the licence metadata distributed with each selected image.
