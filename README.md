# Tennis Tagger v2

Automated tennis match analysis tool. Processes video footage to detect players, track the ball, classify strokes, and export point-by-point data in Dartfish-compatible CSV format.

## Architecture

Go + Python hybrid design:

- **Go** handles the core pipeline: video I/O (via ffmpeg), multi-object tracking, checkpoint/resume, CSV export, and CLI orchestration.
- **Python** (via bridge interface) runs ML inference: YOLOv8 detection, pose estimation, stroke classification, court detection, rally segmentation, and score reading.
- **Bridge layer** (`internal/bridge/`) abstracts the Go-Python boundary with pluggable backends (embedded CPython, subprocess, or mock for testing).

### Package Layout

```
cmd/tagger/          CLI entry point
internal/
  app/               Top-level app (ProcessVideo, ExportCSV)
  bridge/            Go-Python bridge (interface + backends)
  config/            YAML configuration
  export/            Dartfish CSV exporter (62 columns)
  models/            ML model manifest and checker
  pipeline/          Frame processing pipeline with checkpoint/resume
  tracker/           Multi-object tracker (Kalman + Hungarian)
  video/             Video reader (ffmpeg-based frame extraction)
ml/                  Python ML modules
  detector.py        YOLOv8 player + ball detection
  pose.py            Pose estimation
  classifier.py      Stroke classification (X3D)
  analyzer.py        Placement analysis
  score.py           Scoreboard OCR
  trainer.py         Fine-tuning support
models/              ML model files + manifest.json
testdata/            Test fixtures (sample video)
```

## Prerequisites

- Go 1.22+
- Python 3.11+
- ffmpeg on PATH

## Build

```bash
go build ./cmd/tagger/
```

## Run

```bash
./tagger <video.mp4>
```

This processes the video and writes a `<video.mp4>_output.csv` file with 62-column Dartfish-compatible data.

## Live Mode

Process frames from a webcam or RTSP stream in real time:

```bash
# Webcam (index 0)
./tagger --live 0

# RTSP stream
./tagger --live rtsp://192.168.1.100:554/stream

# With mock bridge (no Python)
./tagger --mock --live 0
```

Live mode prints detection counts per batch as frames are processed.

## QC & Retraining

Corrections accumulate as you fix ML mistakes in the UI. When enough corrections are stored, retrain the model:

```bash
./tagger --retrain
```

The CLI will also suggest retraining after video processing when the correction threshold is reached.

## Test

```bash
go test ./internal/... -v
```
