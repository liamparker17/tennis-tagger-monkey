# Tennis Tagger

Automated tennis match analysis. Video in → Dartfish-compatible point-by-point CSV out.

**Status:** Plans 1–3 done. Plan 4 (Go rewrite + CSV writer driven by the multi-task model) in progress. The full workflow is now driven from a single Tkinter launcher — no CLI required end-to-end.

See [`docs/superpowers/2026-04-20-tagger-roadmap.md`](docs/superpowers/2026-04-20-tagger-roadmap.md).

## Roadmap

| Plan | Status | Description |
|---|---|---|
| 1 | done | Clip + multi-label extractor — per-point clips from match video + Dartfish CSV |
| 2A | done | Per-frame feature extractor — pose, court, audio log-mel, sparse ball → `.npz` per clip |
| 2B | done | Stdlib HTTP contact-frame labeler — browser UI for hand-labeling shot contacts |
| 2C | done | Preflight calibration — per-match court corners, ball color, player colors, ball anchors |
| 2D | done | Fine-tuned ball YOLO — trained per-project from preflight ball labels |
| 3  | done | Multi-task point model + training |
| 3B | done | Bayesian fusion inference server (Python) + Go JSON-RPC client |
| 4  | wip  | Go pipeline rewrite + CSV writer driven by the point model |

## Tkinter launcher (the main workflow)

```bash
pythonw tagger_ui.py
```

All stages are exposed as buttons. The launcher chains the required subprocesses in order and skips any stage whose output already exists, so users can resume a partial run:

1. **Preflight** (`preflight.py`) — for each match folder under `files/data/training_pairs/<match>/`, the user calibrates once: clicks court corners on a reference frame, picks 1–2 ball color swatches, picks primary/secondary clothing colors for near/far players, then labels a handful of ball positions on scattered frames. Output: `setup.json` + `ball_labels.json` per match.
2. **Ball YOLO fine-tune** — if `files/models/yolo_ball/best.pt` is missing, the launcher runs `ml.ball_labels_to_yolo` (preflight ball labels → YOLO dataset) then `ml.train_yolo` (fine-tunes `models/yolov8s.pt` at `imgsz=640`, single class). One-shot per project.
3. **Clip extractor** (`ml.dartfish_to_clips`) — cuts per-point clips from match video + Dartfish CSV, writes `labels.json` per match.
4. **Feature extractor** (`ml.feature_extractor`) — pose, court homography, audio log-mel, sparse ball → `.npz` per clip.
5. **Contact labeler** (`ml.contact_labeler.server`) — optional browser UI for human contact-frame labeling.
6. **Point model train / infer** — multi-task head trained on the `.npz` cache, then served over JSON-RPC for Go.

The launcher is idempotent — existing outputs are detected and skipped, so editing one stage doesn't re-run the whole project.

## Pipeline

```
files/data/training_pairs/<match>/{video.mp4, *.csv}
  ↓  preflight.py                         →  setup.json, ball_labels.json  (per match)
  ↓  ml.ball_labels_to_yolo + train_yolo  →  files/models/yolo_ball/best.pt  (project-wide)
  ↓  ml.dartfish_to_clips                 →  files/data/clips/<match>/p_NNNN.mp4 + labels.json
  ↓  ml.feature_extractor                 →  files/data/features/<match>/p_NNNN.npz
  ↓  ml.contact_labeler.server (optional) →  contact_labels.json
  ↓  ml.point_model                       →  files/models/point_model/*.pt
  ↓  ml.point_model.serve (JSON-RPC)      →  PointPrediction for the Go client
  ↓  Go inference + CSV writer            →  Dartfish CSV  (Plan 4, wip)
```

## Prerequisites

- Go 1.22+
- Python 3.11+ (`ultralytics`, `torch`, `opencv-python`, `numpy`, `scipy`, `filterpy`, `easyocr`, `librosa`)
- ffmpeg / ffprobe on PATH
- Pre-trained weights in `models/` — `yolov8s.pt`, `yolov8s-pose.pt`
- NVIDIA GPU strongly recommended for ball-YOLO fine-tune (≈4 GB VRAM at `imgsz=640 batch=4`)

## Package layout

```
tagger_ui.py                 Tkinter launcher (button-driven workflow)
preflight.py                 Per-match calibration (court + colors + ball anchors)
cmd/tagger/                  Legacy Go CLI (Plan 4 rewrite target)
internal/                    Legacy Go pipeline (bridge, tracker, point, export …)
ml/
  dartfish_to_clips/         Plan 1 — CSV → per-point clips
  feature_extractor/         Plan 2A — clip → .npz feature cache
    ball.py                  YOLO + color fused ball detector
    ball_color.py            α-projection CIELab ball color model
    pose.py                  YOLO-pose + color-driven player selection
    player_color.py          α-projection CIELab player color model, pick_best()
    court.py                 Preflight homography (with estimated fallback)
    audio.py                 Log-mel spectrogram
    process.py               Orchestrator — streams frames, writes .npz
  contact_labeler/           Plan 2B — browser contact-frame UI
  point_model/               Plan 3 — multi-task model + training + inference
  ball_labels_to_yolo.py     Preflight ball labels → YOLO dataset
  labels_to_yolo.py          (legacy/general-purpose label converter)
  train_yolo.py              Ultralytics fine-tune wrapper
  tests/                     pytest suite
models/                      Stock pre-trained weights (yolov8s-pose.pt, yolov8s.pt, …)
files/
  data/training_pairs/       Raw inputs (match video + Dartfish CSV)  — gitignored
  data/clips/                Plan 1 output                              — gitignored
  data/features/             Plan 2A output                             — gitignored
  data/yolo_ball/            Ball-YOLO dataset (generated)
  models/yolo_ball/          Fine-tuned ball weights (project-local)
  models/point_model/        Trained point model weights
```

## Feature extractor — design notes

### Ball detection

Fused detector (`ml/feature_extractor/ball.py`): each YOLO-ball candidate is scored as
`0.6·yolo_conf + 0.4·color_score`, then accepted if *any* of the following hold:
fused ≥ 0.10, yolo_conf ≥ 0.30, or color_score ≥ 0.30 (disjunctive accept — either
strong channel rescues a weak counterpart). When no YOLO box clears the bar, the color
model does a stride-4 frame scan and can recover the ball on its own at a down-weighted
confidence.

The fine-tuned ball weights (class id = 0) are auto-detected at
`files/models/yolo_ball/best.pt`; otherwise the detector falls back to stock COCO
`yolov8s.pt` (class id = 32 = "sports ball").

**Anchor-window gating.** The detector is only *run* in the ±15-frame window around each
human-clicked Dartfish ball anchor — those are the only frames with ground truth to
validate against, so asking the detector for frames outside those windows just produces
unsupervised noise. Detection rates are reported within the gated windows only; whole-clip
ratios are not a useful metric here.

### Pose

YOLO-pose at `conf=0.10` returns every plausible person box (including stands/umpire/
ballkids at broadcast framing). The per-match player color model
(`player_color.py::pick_best`) scores every box against the preflight-tagged near/far
clothing signatures and picks the best match for each player identity. This uses the
preflight labels actively — *finding* the far player amongst background clutter — rather
than just disambiguating two already-picked top-2 boxes. Fallback when no color model is
present: top-2 detection confidence, sorted by y-coordinate (works at point start but is
wrong after side swaps).

### Color models

Both `ball_color.py` and `player_color.py` use the same α-projection in CIELab:
for each pixel, residual-of-projection onto the anchor-vs-background axis is Gaussian-
scored, weighted by the clipped projection scalar α, then averaged. Background-invariant
and cheap. Players get two anchors each (shirt + shorts) to survive partial occlusion or
a single body region matching the opponent.

### `.npz` schema (per clip)

- `pose_px (T,2,17,2)` — pixel-space keypoints, `[player_a, player_b]` slots (identity, not side — survives side swaps)
- `pose_conf (T,2,17)` — per-keypoint confidence
- `pose_court (T,2,17,2)` — keypoints projected to normalised court via homography
- `pose_valid (T,2)` — per-slot keypoint-above-threshold flag
- `court_h (3,3)` — homography, image pixels → normalised court
- `ball (T,3)` — `(x, y, conf)`, zeros outside anchor windows or on misses
- `audio_mel (64, F)` — log-mel spectrogram
- `clip_meta (4,)` — `duration_s, w, h, native_fps`

Inspect: `python -m ml.feature_extractor.inspect files/data/features/<match>/p_0001.npz`

## Manual entry points

The Tkinter launcher drives everything, but each stage is runnable standalone:

```bash
python preflight.py <match_folder>
python -m ml.dartfish_to_clips files/data/training_pairs --out files/data/clips
python -m ml.ball_labels_to_yolo --pairs-dir files/data/training_pairs \
    --output-dir files/data/yolo_ball
python -m ml.train_yolo --data files/data/yolo_ball/_shared/data.yaml \
    --base models/yolov8s.pt --out files/models/yolo_ball --imgsz 640 --batch 4
python -m ml.feature_extractor files/data/clips --out files/data/features
python -m ml.contact_labeler.server --clips files/data/clips     # http://127.0.0.1:8765
python -m ml.point_model train   files/data/features --out files/models/point_model
python -m ml.point_model serve   files/models/point_model
```

## Legacy Go CLI (`./tagger.exe video.mp4`)

The pre-rewrite pipeline still builds and runs while Plan 4 is finished.

```bash
go build -o tagger.exe ./cmd/tagger
./tagger.exe video.mp4                   # full pipeline
./tagger.exe --mock video.mp4            # no Python / GPU
./tagger.exe --court-corners x1,y1,...   # manual court corners
```

Writes `<video>_output.csv`. Replacement path: the Plan 3B JSON-RPC server + a new Go
client driven by the multi-task point model.

## Tests

```bash
go test ./internal/... -v
python -m pytest ml/tests/ -v
```

Suites: `test_extract.py`, `test_parse.py`, `test_feat_*.py`, `test_contact_server.py`,
`test_pm_*.py`. Tests that need sample video or pretrained weights skip when those
aren't present.

## Docs

- Roadmap — `docs/superpowers/2026-04-20-tagger-roadmap.md`
- Plan 1 — `docs/superpowers/plans/2026-04-20-clip-multilabel-extractor.md`
- Plan 2 — `docs/superpowers/plans/2026-04-20-per-frame-feature-extractor.md`
- Plan 3 — `docs/superpowers/plans/2026-04-20-multi-task-model-training.md`
- Plan 4 — `docs/superpowers/plans/2026-04-20-go-pipeline-rewrite.md`
- Contributor guidance — `CLAUDE.md`
