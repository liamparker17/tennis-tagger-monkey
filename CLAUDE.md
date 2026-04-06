# Tennis Tagger

Automated tennis match analysis. Processes video footage → structured shot-by-shot CSV compatible with Dartfish tagging format. Goal: replace human taggers with ML detection and analysis.

## Architecture

```
Video → [Go Pipeline] → [Python ML Bridge] → [Go Point Logic] → CSV
```

**Go** (`cmd/tagger/`, `internal/`) — CLI, video I/O (ffmpeg), 3-stage concurrent pipeline, shot segmentation, point recognition, scoring, CSV export.

**Python** (`ml/`) — ML inference via JSON-RPC subprocess bridge. TrackNet ball detection, YOLO player detection, court detection, trajectory fitting, stroke classification.

**Frontend** (`frontend/`) — Svelte UI (secondary, not the main workflow).

### Data Flow

```
Video frames
  → BackgroundSubtractor (median of 30 sampled frames)
  → TrackNet (ball heatmap → peak detection, threshold=0.3)
  → YOLO (player bounding boxes)
  → Court detection (homography, broadcast fallback if detection fails)
  → mergeBallPositions() (combine TrackNet + YOLO ball detections)
  → deduplicate_detections() (keep highest confidence per frame)
  → segment_detections() (adaptive gap-aware splitting into shot segments)
  → TrajectoryFitter.fit() per segment (linear fit, bounce detection, speed clamping 5-300 km/h)
  → SegmentShots() (bounce-based hitter ID, velocity fallback for bounceless)
  → groupShotsIntoPoints() (3-second gap = new point)
  → RecognizePoint() (ace, double fault, winner, error classification)
  → MatchState scoring (game/set/match state machine)
  → Dartfish CSV export (80+ columns)
```

### Go Packages

| Package | Purpose |
|---------|---------|
| `cmd/tagger` | CLI entry point, flag parsing |
| `internal/pipeline` | 3-stage concurrent pipeline: frame extraction → ML inference → tracking/assembly |
| `internal/bridge` | Go↔Python JSON-RPC bridge, shared memory for frames, `MockBridge` for tests |
| `internal/point` | `Shot` struct, `SegmentShots()` from trajectories, `RecognizePoint()`, `MatchState` scoring |
| `internal/video` | ffmpeg-based video reader, frame extraction |
| `internal/export` | Dartfish-compatible CSV export |
| `internal/tracker` | Multi-object tracker (Kalman filter + Hungarian assignment) |
| `internal/tactics` | Pattern analysis, player tendencies |
| `internal/config` | YAML config loading |
| `internal/corrections` | Human correction storage for retraining |
| `internal/models` | Model download and checksum verification |

### Python Modules

| Module | Purpose |
|--------|---------|
| `ml/bridge_server.py` | JSON-RPC server, dispatches to ML modules |
| `ml/tracknet.py` | BallTrackerNet ball detection (yastrebksv weights at 640x360), background subtraction, peak detection |
| `ml/trajectory.py` | Trajectory fitting, bounce detection, in/out calls, speed estimation, `segment_detections()` with adaptive gap tolerance |
| `ml/detector.py` | YOLO player detection |
| `ml/analyzer.py` | Court detection and homography estimation |
| `ml/classifier.py` | Stroke type classification |
| `ml/trainer.py` | Model fine-tuning from corrections |

## Development

### Prerequisites

- Go 1.26.1+
- Python 3.11+ with PyTorch, OpenCV, scipy, filterpy, easyocr
- ffmpeg/ffprobe on PATH
- Pre-trained models in `models/` (yastrebksv_tracknet.pt, yolov8s.pt)

### Commands

```bash
# Build
go build -o tagger.exe ./cmd/tagger

# Process a video
./tagger.exe <video.mp4>
./tagger.exe --mock <video.mp4>          # no Python/GPU needed
./tagger.exe --court-corners x1,y1,...   # manual court corners

# Run all Go tests
go test ./internal/... -v

# Run all Python tests
python -m pytest ml/tests/ -v

# Run specific test suites
python -m pytest ml/tests/test_trajectory.py -v   # segmentation, fitting, speed
python -m pytest ml/tests/test_tracknet.py -v      # peak detection, homography
go test ./internal/point/... -v                     # shot/point recognition
```

### Test Data

Test videos in `testdata/`: `sample_a.mp4` through `sample_e.mp4`, `tennis_10s.mp4`, `tennis_60s.mp4`, `tennis_mid_10s.mp4`, `tennis_mid2_60s.mp4`. CSV outputs are `<video>_output.csv`.

## Current State

### Detection Rate

~15-20% ball detection rate on test footage. The BallTrackerNet weights (trained at 640x360) struggle with this specific camera angle. Scaling up resolution does not help — the pretrained weights are resolution-specific and would need fine-tuning at higher resolution.

### Segmentation (recently improved)

`segment_detections()` uses **adaptive gap tolerance** scaled by detection density:
- Dense detection (every frame): 15-frame fallback gap (0.5s at 30fps)
- Sparse detection (~21% rate): scales up to 45-frame gap (1.5s)
- `_MIN_DETECTIONS` = 2 (previously 3), so short points with few detections are no longer dropped

This fixed two counting bugs:
- Shot over-counting (aggressive splitting of sparse detections into too many segments)
- Point under-counting (short points dropped due to minimum detection threshold)

### Known Limitations

- **Hitter ID**: All bounces map to far-player side due to broadcast fallback homography producing negative court-Y values. Proper court detection or manual `--court-corners` would fix this.
- **Speed estimates**: Many shots show 0.0 km/h because 2-detection segments can't produce reliable speed fits.
- **Court detection**: Falls back to approximate broadcast homography when line detection fails. This affects bounce in/out classification.

### Key Parameters

| Parameter | Location | Value | Purpose |
|-----------|----------|-------|---------|
| `_MIN_GAP_FRAMES` | `ml/trajectory.py` | 5 | Gaps ≤ this never trigger split checks |
| `_FALLBACK_GAP_FRAMES_BASE` | `ml/trajectory.py` | 15 | Base gap tolerance (scaled by density) |
| `_FALLBACK_GAP_FRAMES_MAX` | `ml/trajectory.py` | 45 | Max adaptive gap tolerance |
| `_MAX_MERGE_GAP_FRAMES` | `ml/trajectory.py` | 75 | Hard cap — never merge across this |
| `_MIN_DETECTIONS` | `ml/trajectory.py` | 2 | Min detections to fit a trajectory |
| `_PEAK_THRESHOLD` | `ml/tracknet.py` | 0.5 (0.3 for classifier) | Ball detection confidence threshold |
| `groupShotsIntoPoints` gap | `internal/pipeline/concurrent.go` | 3 seconds | Gap between shots that starts a new point |

## Conventions

- Go: standard library style, `slog` for structured logging, table-driven tests
- Python: type hints, dataclasses, pytest, snake_case internally with camelCase JSON for Go interop
- Commits: conventional commits (`feat:`, `fix:`, `test:`, `docs:`)
- Plans/specs in `docs/superpowers/`
