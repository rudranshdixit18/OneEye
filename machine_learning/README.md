# OneEye machine-learning workspace

This directory contains the reproducible acquisition, preparation, validation, training, and export pipeline for the five OneEye event models.

Raw third-party data is stored under `datasets/raw/` and is never committed. Every source must have a manifest entry and an accepted licence before acquisition. Prepared datasets are written to `datasets/processed/<model>/` without changing the raw source.

## Pipeline

```powershell
python -m machine_learning.pipeline sources
python -m machine_learning.pipeline acquire --dataset taco --accept-license
python -m machine_learning.pipeline prepare --model garbage
python -m machine_learning.pipeline validate --model garbage
python -m machine_learning.pipeline train --model garbage
python -m machine_learning.pipeline evaluate --model garbage
python -m machine_learning.pipeline export --model garbage
```

Run the same sequence for `fight`, `fallen_person`, `mobile_phone`, and `accident`. Acquisition intentionally stops when an official dataset requires an account, manual agreement, or owner approval.

Garbage training can combine two independently licensed official sources. Acquire both before `prepare`; MJU-Waste contributes only its annotated training images, while TACO validation and test stay fixed:

```powershell
python -m machine_learning.pipeline acquire --dataset taco --accept-license
python -m machine_learning.pipeline acquire --dataset mju_waste --accept-license
python -m machine_learning.pipeline prepare --model garbage
```

The expanded phone candidate uses a larger Open Images training selection while preserving the official validation/test boundaries:

```powershell
python -m machine_learning.pipeline acquire --dataset open_images_mobile_phone --accept-license --max-images 6500
python -m machine_learning.pipeline prepare --model mobile_phone
```

## Resume after an interruption

`training_status.json` records the last completed stage for every model and, for newly started training runs, the completed epoch, current validation metrics, and `last.pt` restart path. Writes use an inter-process lock and atomic replacement so concurrent jobs cannot corrupt it. Dataset downloads use `.part` files and keep already completed assets. Ultralytics writes `last.pt` after each epoch; resume an interrupted training run with:

```powershell
python -m machine_learning.pipeline train --model garbage --device 0 --resume
```

For accident classification, the pipeline uses the official Nexar positive/negative training folders and official event metadata. Positive temporal mosaics are centred on the collision timestamp; negative videos use matched relative offsets so clip position is not a class shortcut.

For long background runs, `scripts/watch_training_completion.ps1` can wait for a known trainer PID, temporarily suppress idle system sleep, evaluate the untouched test split only after a successful exit, and export only after the strict gate passes. It restores the normal sleep policy when it exits and deliberately does not start another training job.

`scripts/run_remaining_training_sequence.ps1` is the guarded full queue. It waits for a specified active trainer, evaluates and conditionally exports that model, then runs the configured remaining models sequentially. Before a new model run it moves any older run to a timestamped `*_previous_*` folder inside `machine_learning/runs`; runtime models are untouched unless a new candidate passes the quality gate. The queue also suppresses idle sleep and restores the normal policy when it exits.

Fight detection also has a true temporal pipeline that preserves the official RWF test boundary and groups four short windows per video for multiple-instance learning:

```powershell
python -m machine_learning.video_pipeline prepare
python -m machine_learning.video_pipeline train --device 0
python -m machine_learning.video_pipeline evaluate --device 0
python -m machine_learning.video_pipeline export
```

The export commands enforce the project quality gate. A research checkpoint that fails held-out testing stays under `machine_learning/runs/` and is not promoted to `models/`.
