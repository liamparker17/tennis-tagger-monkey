# Tennis Tagger

Automated tennis match analysis. Video in → Dartfish-compatible point-by-point CSV out.

**Status:** Mid-rewrite from ball-tracker-centric pipeline to a per-point clip model over pose + audio + court + (sparse) ball. See [`docs/superpowers/2026-04-20-tagger-roadmap.md`](docs/superpowers/2026-04-20-tagger-roadmap.md).

## Roadmap

| Plan | Status | Description |
|---|---|---|
| 1 | done | Clip + multi-label extractor — cuts per-point clips from match video + Dartfish CSV |
| 2A | done | Per-frame feature extractor — pose, court homography, audio log-mel, sparse YOLO-ball → `.npz` per clip |
| 2B | done | Stdlib HTTP contact-frame labeler — browser UI for hand-labeling shot contacts |
| 3 | pending | Multi-task model (6 heads) + training |
| 4 | pending | Go pipeline rewrite + CSV writer for model output |

## Pipeline (rewrite target)

```
Match video + Dartfish CSV
  ↓  (Plan 1)  python -m ml.dartfish_to_clips
Per-point clips  +  labels.json
  ↓  (Plan 2A)  python -m ml.feature_extractor
.npz per clip (pose, court, audio, ball, clip meta)
  ↓                         ↓  (Plan 2B)  python -m ml.contact_labeler.server
  ↓                         contact_labels.json  (hand-labeled contact frames)
  ↓  (Plan 3 — pending)  multi-task transformer
PointPrediction
  ↓  (Plan 4 — pending)  Go inference + CSV writer
Dartfish CSV
```

## Current pipeline (`./tagger.exe video.mp4`)

The existing Go + Python pipeline still runs while Plans 3/4 are built out.

- **Go** (`cmd/tagger/`, `internal/`) — CLI, ffmpeg I/O, 3-stage concurrent pipeline, shot segmentation, point recognition, scoring, Dartfish CSV export.
- **Python** (`ml/`) — ML inference via JSON-RPC subprocess bridge. YOLO player+ball detection, trajectory fitting, stroke classification, court detection.

### Package layout

```
cmd/tagger/                CLI entry point
internal/
  app/                     Top-level orchestration
  bridge/                  Go↔Python bridge (process / mock backends)
  config/                  YAML config
  export/                  Dartfish CSV writer
  pipeline/                3-stage concurrent pipeline
  point/                   Shot segmentation, point recognition, scoring
  tracker/                 Kalman + Hungarian tracker
  video/                   ffmpeg frame reader
  tactics/                 Pattern / tendency analysis
  corrections/             Human correction storage
ml/
  bridge_server.py         JSON-RPC server (legacy pipeline)
  detector.py              YOLO player + ball detection
  analyzer.py              Court detection + homography
  trajectory.py            Trajectory fitting + bounce detection
  trainer.py               Fine-tuning support
  dartfish_to_clips/       Plan 1 — CSV → per-point clips
  feature_extractor/       Plan 2A — clip → .npz feature cache
  contact_labeler/         Plan 2B — browser labeling UI
  tests/                   pytest suite
```

## Prerequisites

- Go 1.22+
- Python 3.11+ (`ultralytics`, `opencv-python`, `scipy`, `filterpy`, `easyocr`, `librosa`)
- ffmpeg / ffprobe on PATH
- Pre-trained weights in `models/` — `yolov8s.pt`, `yolov8s-pose.pt`

## Build & run

```bash
go build -o tagger.exe ./cmd/tagger

./tagger.exe video.mp4                   # full pipeline
./tagger.exe --mock video.mp4            # no Python / GPU
./tagger.exe --court-corners x1,y1,...   # manual court corners
```

Writes `<video>_output.csv` — Dartfish-compatible point-by-point tagging.

## Plan 1 — clip extractor

Cuts per-point clips from `files/data/training_pairs/<match>/*.mp4` + matching Dartfish CSV.

```bash
python -m ml.dartfish_to_clips files/data/training_pairs --out files/data/clips
```

Output: `files/data/clips/<match>/p_NNNN.mp4` + one `labels.json` per match covering every Dartfish supervision column (server, returner, contact/placement XYs, stroke types, winner/error, score state, …).

## Plan 2A — feature extractor

Walks the clips from Plan 1 → writes `.npz` feature cache per clip.

```bash
python -m ml.feature_extractor files/data/clips --out files/data/features
python -m ml.feature_extractor.inspect files/data/features/<match>/p_0001.npz
```

Each `.npz` packs per-clip tensors:

- `pose_px (T,2,17,2)` — pixel-space keypoints, near/far slot
- `pose_conf (T,2,17)` — per-keypoint confidence
- `pose_court (T,2,17,2)` — keypoints in court coords via homography
- `pose_valid (T,2)` — slot has any keypoint above threshold
- `court_h (3,3)` — homography, pixel → normalised court
- `ball (T,3)` — `(x, y, conf)`, zeros where no detection
- `audio_mel (64, F)` — log-mel spectrogram
- `clip_meta (4,)` — `duration_s, w, h, native_fps`

## Plan 2B — contact-frame labeler

Browser UI for hand-labeling per-shot contact frames. Feeds Plan 3's contact-event head.

```bash
python -m ml.contact_labeler.server --clips files/data/clips
# open http://127.0.0.1:8765
```

Keys: `,` `.` step frame · Space play/pause · `n` near-side hit · `f` far-side hit · `u` undo · `s` save · `]` next clip.

Labels persist to `<match>/contact_labels.json` under `--clips`.

## Tests

```bash
go test ./internal/... -v
python -m pytest ml/tests/ -v
```

Plan 2 suites: `test_feat_*.py`, `test_contact_server.py`. Tests that need sample video or model weights skip when unavailable.

## Docs

- Roadmap — `docs/superpowers/2026-04-20-tagger-roadmap.md`
- Plan 1 — `docs/superpowers/plans/2026-04-20-clip-multilabel-extractor.md`
- Plan 2 — `docs/superpowers/plans/2026-04-20-per-frame-feature-extractor.md`
- Plan 3 — `docs/superpowers/plans/2026-04-20-multi-task-model-training.md`
- Plan 4 — `docs/superpowers/plans/2026-04-20-go-pipeline-rewrite.md`
- Agent / contributor guidance — `CLAUDE.md`
