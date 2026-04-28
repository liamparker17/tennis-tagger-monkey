# Tennis Tagger

Automated tennis match analysis. Match video in → Dartfish-compatible point-by-point CSV out.

The whole workflow is driven from a single launcher — no CLI required end-to-end. Per-match calibration (court corners, player/ball colors, ball anchors) takes a couple of minutes; everything after that runs as a chain of skippable, resumable stages.

## Install (Windows)

**[➡ Download the latest installer](https://github.com/liamparker17/tennis-tagger-monkey/releases/latest)**

1. On the page above, click `TennisTagger-Setup-<version>.exe` to download.
2. Run the file. Windows will show a blue **"Windows protected your PC"** screen because the installer isn't code-signed (this is expected for hobby releases — the file is safe). Click **More info → Run anyway**.
3. Follow the wizard. You can pick where to install; defaults are fine.
4. When the installer finishes, leave **"Launch Tennis Tagger"** ticked and click **Finish**. The app opens.

That's it — no Python, no `pip install`, no PATH setup. Everything is bundled.

**Requirements:** Windows 10 or 11, x64. About 1 GB free disk space. NVIDIA GPU optional (the default installer ships CPU-only PyTorch; a GPU build is available from the Releases page if you need it).

**Sharing trained models with a friend:** the launcher has a **"Share with a friend (USB)"** option in the home screen's Advanced row. Plug a USB stick in, click **Send my model**, hand over the drive. Your friend plugs it in, clicks **Use my friend's model**, and picks **Merge** to average their model with yours, or **Replace** to adopt yours wholesale. Their previous model is auto-backed-up either way.

## Quick start (from source)

```bash
pythonw tagger_ui.py
```

Drop match folders into `files/data/training_pairs/<match>/` (each containing the match `.mp4` and the Dartfish `.csv`), then click through:

1. **Preflight** — one-time per match: click the 4 court corners, pick 1–2 ball-color swatches, pick shirt + shorts colors for near/far players, label a handful of ball positions on scattered frames. Writes `setup.json` + `ball_labels.json` next to the video.
2. **Train ball detector** — if `files/models/yolo_ball/best.pt` is missing, the launcher converts every project's `ball_labels.json` into a YOLO dataset and fine-tunes `yolov8s.pt` on it (single class, `imgsz=640`, ≈4 GB VRAM). One-shot per project.
3. **Cut clips** — slices the match video into one `.mp4` per point plus a `labels.json` covering every Dartfish supervision column.
4. **Extract features** — per-clip pose + court homography + audio log-mel + sparse ball → `.npz` under `files/data/features/<match>/`.
5. *(optional)* **Label contact frames** — browser UI at `http://127.0.0.1:8765` for hand-labeling per-shot ball-contact frames.
6. **Train / serve point model** — multi-task head trained on the `.npz` cache, then served over JSON-RPC for the Go CSV writer.

Each stage skips itself when its output already exists, so editing one stage doesn't re-run the whole project.

## Pipeline

```
files/data/training_pairs/<match>/{video.mp4, *.csv}
  ↓  preflight.py                         setup.json + ball_labels.json  (per match)
  ↓  ml.ball_labels_to_yolo + train_yolo  files/models/yolo_ball/best.pt (project-wide)
  ↓  ml.dartfish_to_clips                 files/data/clips/<match>/p_NNNN.mp4 + labels.json
  ↓  ml.feature_extractor                 files/data/features/<match>/p_NNNN.npz
  ↓  ml.contact_labeler.server (optional) contact_labels.json
  ↓  ml.point_model                       files/models/point_model/*.pt
  ↓  ml.point_model.serve (JSON-RPC)      PointPrediction for the Go client
  ↓  Go inference + CSV writer            Dartfish CSV
```

## Prerequisites

- Go 1.22+
- Python 3.11+ (`ultralytics`, `torch`, `opencv-python`, `numpy`, `scipy`, `filterpy`, `easyocr`, `librosa`)
- ffmpeg / ffprobe on PATH
- Stock weights in `models/` — `yolov8s.pt`, `yolov8s-pose.pt`
- NVIDIA GPU strongly recommended for the ball-YOLO fine-tune

## Package layout

```
tagger_ui.py                 Tkinter launcher (button-driven workflow)
preflight.py                 Per-match calibration
cmd/tagger/                  Go CLI (bridges to the point-model server)
internal/                    Go pipeline (bridge, tracker, point, export, …)
ml/
  dartfish_to_clips/         CSV → per-point clips
  feature_extractor/
    ball.py                  YOLO + color fused ball detector
    ball_color.py            α-projection CIELab ball color model
    pose.py                  YOLO-pose + color-driven player selection
    player_color.py          α-projection CIELab player color model, pick_best()
    court.py                 Preflight homography (with estimated fallback)
    audio.py                 Log-mel spectrogram
    process.py               Orchestrator — streams frames, writes .npz
  contact_labeler/           Browser contact-frame labeling UI
  point_model/               Multi-task model + training + JSON-RPC inference
  ball_labels_to_yolo.py     Preflight ball labels → YOLO dataset
  labels_to_yolo.py          General-purpose label converter
  train_yolo.py              Ultralytics fine-tune wrapper
  tests/                     pytest suite
models/                      Stock pre-trained weights
files/
  data/training_pairs/       Raw inputs (gitignored)
  data/clips/                Clip output (gitignored)
  data/features/             Feature output (gitignored)
  data/yolo_ball/            Generated ball-YOLO dataset
  models/yolo_ball/          Fine-tuned ball weights
  models/point_model/        Trained point-model weights
```

## Feature extractor — design notes

### Ball detection

Fused detector (`ml/feature_extractor/ball.py`): each YOLO-ball candidate is scored as
`0.6·yolo_conf + 0.4·color_score` and accepted if *any* of the following hold — fused ≥ 0.10,
yolo_conf ≥ 0.30, or color_score ≥ 0.30 (disjunctive accept: either strong channel rescues
a weak counterpart). When no YOLO box clears the bar, the color model does a stride-4 frame
scan and can recover the ball on its own at a down-weighted confidence.

The fine-tuned ball weights (class id = 0) are auto-detected at `files/models/yolo_ball/best.pt`;
otherwise the detector falls back to stock COCO `yolov8s.pt` (class id = 32 = "sports ball").

**Anchor-window gating.** The detector is only *run* in the ±15-frame window around each
human-clicked Dartfish ball anchor — those are the only frames with ground truth, so asking
the detector for frames outside those windows just produces unsupervised noise. Detection
rates are reported within the gated windows only; whole-clip ratios are not a useful metric.

### Pose

YOLO-pose at `conf=0.10` returns every plausible person box — including stands, umpire, and
ball kids at broadcast framing. The per-match player color model
(`player_color.py::pick_best`) scores every box against the preflight-tagged near/far
clothing signatures and picks the best match for each player identity. This uses the
preflight labels *actively* — finding the far player amongst background clutter — rather
than just disambiguating two already-picked top-2 boxes. Fallback when no color model is
present: top-2 detection confidence, sorted by y-coordinate (works at point start but is
wrong after side swaps).

### Color models

Both `ball_color.py` and `player_color.py` use the same α-projection in CIELab: for each
pixel, residual-of-projection onto the anchor-vs-background axis is Gaussian-scored,
weighted by the clipped projection scalar α, then averaged. Background-invariant and cheap.
Players get two anchors each (shirt + shorts) to survive partial occlusion or a single body
region matching the opponent.

### `.npz` schema (per clip)

- `pose_px (T,2,17,2)` — pixel-space keypoints, `[player_a, player_b]` slots (identity, not side)
- `pose_conf (T,2,17)` — per-keypoint confidence
- `pose_court (T,2,17,2)` — keypoints projected to normalised court via homography
- `pose_valid (T,2)` — per-slot keypoint-above-threshold flag
- `court_h (3,3)` — homography, image pixels → normalised court
- `ball (T,3)` — `(x, y, conf)`; zeros outside anchor windows or on misses
- `audio_mel (64, F)` — log-mel spectrogram
- `clip_meta (4,)` — `duration_s, w, h, native_fps`

Inspect: `python -m ml.feature_extractor.inspect files/data/features/<match>/p_0001.npz`

## Training notes — how `epochs`, data, and val loss actually interact

One epoch = one pass through every training clip. With 2 matches (~700 clips total,
~385 train clips after the match-level val split) and batch size 8 that's ~48 gradient
updates per epoch. Match *duration* is irrelevant; only *clip count* matters.

Empirically, the overfit inflection point moves with dataset size:

| Training clips    | Val bottoms out around | Sensible epoch count |
|-------------------|------------------------|----------------------|
| ~385 (1 match)    | ep 5–8                 | 8                    |
| ~700 (2 matches)  | ep 5–7                 | 8–15                 |
| 2000+ (6+ matches)| ep 15–25               | 20–40                |

**The default of 20 epochs is intentionally overshooting** — `best.pt` is saved at the
lowest val epoch regardless of how long the run goes, so you get the best model whether
the real peak is at ep 5 or ep 25. The wasted compute after the overfit point is
cosmetic; the shipped model is identical.

**Data variety dominates data volume.** Ten Wimbledon R1 matches all on centre court
with the same broadcaster will help less than five matches across different tournaments,
surfaces, and broadcast styles. Player pairs, lighting, and camera angles are the
dimensions the model generalizes over.

### Single-match val fallback

When `files/data/features/` only contains one match, match-level split
(`split_matches`) leaves val empty. The trainer then falls back to a clip-level split
(`split_clips`, hash-deterministic ~20% of clips held out from the one training match)
so you still get a real val signal. Output looks like:

```
matches: 1 total, 1 train, 0 val
single-match fallback: holding out 84/385 clips from <match> as val (~22%)
```

That val number is an honest benchmark for *single-match* generalization, but it's
easier than cross-match val because the held-out clips share everything except their
exact moment in the match. Real cross-match generalization only shows up at ≥2 matches.

### Known failure modes

- **`train=nan` mid-run** with no recovery → gradient explosion; LR or a too-small
  batch is to blame. `best.pt` is unaffected if saved before the blowup.
- **Val flat at 0.0** every epoch → no val data. Either empty `files/data/features/`
  or the clip-level fallback didn't fire. Check the log header.
- **Val climbing while train drops** → textbook overfit. Expected on small data; the
  fix is more matches, not regularization.

## Manual entry points

The launcher drives everything, but each stage is runnable standalone:

```bash
python preflight.py <match_folder>
python -m ml.dartfish_to_clips files/data/training_pairs --out files/data/clips
python -m ml.ball_labels_to_yolo --pairs-dir files/data/training_pairs \
    --output-dir files/data/yolo_ball
python -m ml.train_yolo --data files/data/yolo_ball/_shared/data.yaml \
    --base models/yolov8s.pt --out files/models/yolo_ball --imgsz 640 --batch 4
python -m ml.feature_extractor files/data/clips --out files/data/features
python -m ml.contact_labeler.server --clips files/data/clips     # http://127.0.0.1:8765
python -m ml.point_model train \
    --clips files/data/clips --features files/data/features \
    --out files/models/point_model/current --epochs 20
python -m ml.point_model eval \
    --ckpt files/models/point_model/current/best.pt \
    --clips files/data/clips --features files/data/features
```

## Go CLI

```bash
go build -o tagger.exe ./cmd/tagger
./tagger.exe video.mp4                   # full pipeline
./tagger.exe --mock video.mp4            # no Python / GPU
./tagger.exe --court-corners x1,y1,...   # manual court corners
```

Writes `<video>_output.csv` — Dartfish-compatible tagging.

## Sharing trained models with a collaborator

Two taggers training the point model on different matches can pool their
results so both machines converge to the same unified model faster — both
people training in parallel on different footage, weights merged after
every round.

### Recommended: shared cloud folder

Pick any folder service both machines already sync (Dropbox, Google Drive,
OneDrive). Each machine publishes its trained model into the shared folder;
each machine syncs from it.

```bash
# Person A finishes training:
./tagger.exe model publish --author alice "D:/Dropbox/TennisTagger/shared"

# Person B picks up everyone else's progress:
./tagger.exe model sync "D:/Dropbox/TennisTagger/shared"
```

Layout under the shared folder:

```
shared/
  146ef67e.../              <- machine A's publishes (each machine has a stable id)
    2026-04-28T204700Z/     <- one bundle per training round
      manifest.json
      weights.pt
    2026-04-29T193000Z/
      ...
  92ff8c12.../              <- machine B's publishes
    ...
```

`sync` walks the folder, ignores your own machine's publishes, finds bundles
newer than the last one you merged from each remote machine (state stored at
`files/models/point_model/current/.sync-state.json`), and merges each in
chronological order. Re-running `sync` is safe — already-merged bundles are
skipped. Add `--dry-run` to list new bundles without merging.

A typical round: both machines start from the same `best.pt`, each labels
+ trains on a different match for a day or two, both `model publish`, both
`model sync`, both restart training from the merged weights. Repeat.

### USB fallback

When taggers don't share a cloud folder, the same bundle format works on
USB. The launcher's "Share with a friend (USB)" screen walks you through
it. CLI equivalents:

```bash
# Send: write your trained model + a manifest into an empty folder
./tagger.exe model export --author alice /e/usb/week17

# Receive (option A): replace your model with theirs
./tagger.exe model import /e/usb/friend-week17

# Receive (option B): average theirs into yours
./tagger.exe model merge /e/usb/friend-week17
```

A bundle folder contains `manifest.json` (author, date, SHA256, optional
notes) and `weights.pt`. Both partners must train from the same base
architecture; otherwise the merge errors out with a layer mismatch.

The launcher always backs up your current `best.pt` before replacing or
merging — look for `best.before-merge-<timestamp>.pt` next to the active
checkpoint if you need to roll back.

## Tests

```bash
go test ./internal/... -v
python -m pytest ml/tests/ -v
```

Tests that need sample video or pretrained weights skip when those aren't present.

## Crash reporting (Sentry)

When a tagger out in the field hits a crash, it gets reported back to a
Sentry project with the stack trace, OS, and version — no manual bug report
needed. Both the Go pipeline and every Python entry point (UI, Preflight,
bridge, training, inference) wire the same DSN.

**For end users** — nothing to set up; reports are sent automatically when
the bundle ships with a `sentry.dsn` file. Anyone can opt out by setting
`TENNIS_TAGGER_TELEMETRY=off` in their environment before launching.

**For maintainers** — to enable reporting in your builds:

1. Create a Sentry project at sentry.io (free tier covers a small team).
2. Copy the DSN.
3. Build the bundle with the DSN baked in:

   ```powershell
   .\packaging\build_bundle.ps1 -SentryDsn 'https://abc@o123.ingest.sentry.io/456'
   ```

   Or set `TENNIS_TAGGER_SENTRY_DSN` in the build environment. CI:
   add the DSN as a repo secret and pass it through to `build_bundle.ps1`.

The DSN ends up at `<install-dir>\sentry.dsn`. Username and home-directory
paths are scrubbed from every event before send (player names appear in
match filenames, so this matters).
