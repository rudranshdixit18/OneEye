# OneEye Sentinel AI

OneEye is a five-event surveillance intelligence prototype with the Sentinel AI frontend, a FastAPI backend, persistent events, shared camera workers, reproducible dataset preparation, and model training pipelines.

The hosted Vercel site is a **frontend preview only**. Live camera capture, model inference, alert persistence, and settings require the Windows-hosted FastAPI backend. The public repository excludes raw training datasets, newly trained checkpoints, and legacy checkpoints whose redistribution terms have not been confirmed. See the local project and `documentation/` for evaluation evidence. Do not present the hosted preview as a live five-model cloud service.

## Event models

- Fight detection
- Garbage detection
- Fallen-person detection
- Mobile-phone detection
- Accident detection

The three supplied checkpoints are retained in the local project but are not redistributed here. The guarded pipeline exports a newly trained checkpoint to `models/<event>/best.pt` only after every required held-out score is strictly greater than 95%. In the completed training sequence, the expanded fallen-person classifier passed and was exported locally; fight, garbage, expanded mobile-phone, and accident candidates were preserved as research artifacts after failing the strict gate. The backend supports both Ultralytics detection and classification checkpoints through one normalized event contract.

The presentation-ready evidence report is `documentation/OneEye_Model_Evaluation_and_System_Verification.docx`. Exact machine-readable evaluation outcomes and promotion decisions are recorded in `documentation/MODEL_EVALUATION_RESULTS.json`; `machine_learning/training_status.json` records the latest completed pipeline stage for each model.

## First run on Windows

```powershell
Set-Location "<path-to-your-OneEye-checkout>"
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
Copy-Item .env.example .env
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

Open `http://127.0.0.1:8000`. The backend serves the existing frontend, so no separate web server is required.

By default, `CAMERA_STREAM_URL=0` selects the first local webcam. An IP Webcam or RTSP address can be placed in `.env`. Four sources can be supplied through `ONEEYE_CAMERA_SOURCES` as shown in `.env.example`.

The software requests 1920×1080 at 60 FPS and reports the actual measured capture rate. The tested built-in camera delivered about 30 FPS without the full workload and 18.5–20 FPS during simultaneous five-model inference and MJPEG streaming. Sustained 60 unique frames per second requires a camera and capture interface that can physically deliver 1080p60.

## API

- Interactive documentation: `http://127.0.0.1:8000/docs`
- Health and model status: `GET /api/v1/health`
- Start/stop a camera: `POST /api/v1/cameras/{camera_id}/start|stop`
- Annotated MJPEG stream: `GET /api/v1/cameras/{camera_id}/stream`
- Events: `GET /api/v1/events`
- Analytics: `GET /api/v1/analytics/summary`
- Thresholds: `GET|PUT /api/v1/settings`
- Image test: `POST /api/v1/infer`

## Train models

The source and license registry is `machine_learning/dataset_registry.yaml`. See `machine_learning/README.md` for acquisition, preparation, validation, training, evaluation, and export commands. Never combine raw third-party data with source code or redistribute it without checking its terms.

## Important behavior

- One camera worker owns each source; browser viewers never read the camera independently.
- Inference is performed by one analysis pipeline per camera.
- Consecutive-frame confirmation and cooldowns reduce alert flooding.
- Events and settings use SQLite WAL storage.
- Snapshots, clips, models, datasets, and logs have separate directories.
- Missing models produce a degraded health state without crashing the dashboard.
