# Phase 1: Core Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working Go+Python hybrid desktop app that processes tennis videos and produces Dartfish-compatible CSVs identical to the current Python-only app.

**Architecture:** Wails (Go + Svelte) desktop app with embedded Python (go-python3) for ML inference. Go handles video I/O (ffmpeg), tracking (Kalman+Hungarian via gonum), orchestration (3-stage concurrent pipeline), CSV export, and UI. Python handles detection, classification, and analysis only.

**Tech Stack:** Go 1.22+, Wails v2, Svelte 4, go-python3, gonum, ffmpeg (external binary), Python 3.11, PyTorch 2.4, Ultralytics 8.3, numpy 1.26

**Spec:** `docs/superpowers/specs/2026-03-20-tennis-tagger-rewrite-design.md`

**Existing code reference:** `C:\Users\liamp\Downloads\Tennis Tagger Monkey\files\`

---

## File Structure

```
tennis-tagger/                          # New project root
├── cmd/tagger/main.go                  # Wails entry point
├── internal/
│   ├── app/app.go                      # Wails bindings (Go <-> JS)
│   ├── video/
│   │   ├── reader.go                   # VideoReader: ffmpeg frame extraction
│   │   ├── reader_test.go
│   │   └── frame.go                    # Frame type definitions
│   ├── tracker/
│   │   ├── kalman.go                   # Kalman filter (predict/update)
│   │   ├── kalman_test.go
│   │   ├── hungarian.go                # IoU + greedy assignment
│   │   ├── hungarian_test.go
│   │   ├── tracker.go                  # MultiObjectTracker
│   │   └── tracker_test.go
│   ├── bridge/
│   │   ├── types.go                    # BridgeBackend interface + result types
│   │   ├── embedded.go                 # EmbeddedBridge (go-python3)
│   │   ├── process.go                  # ProcessBridge fallback
│   │   ├── worker.go                   # bridgeWorker (dedicated OS thread)
│   │   └── mock.go                     # MockBridge for testing
│   ├── pipeline/
│   │   ├── pipeline.go                 # Pipeline orchestrator
│   │   ├── pipeline_test.go
│   │   ├── checkpoint.go              # Checkpoint save/resume
│   │   └── checkpoint_test.go
│   ├── export/
│   │   ├── dartfish.go                 # DartfishExporter
│   │   ├── dartfish_test.go
│   │   └── columns.go                 # 62 column definitions
│   ├── config/
│   │   ├── config.go                   # Config struct + YAML load/save
│   │   └── config_test.go
│   └── models/
│       ├── manifest.go                 # Manifest types + checker
│       ├── download.go                 # HTTP download + checksum verification
│       └── download_test.go
├── ml/                                 # Python ML core (embedded)
│   ├── __init__.py
│   ├── detector.py                     # Unified YOLO detection
│   ├── pose.py                         # Pose estimation + serve detection
│   ├── classifier.py                   # Stroke classification (3D CNN)
│   ├── analyzer.py                     # Court + placement + rally
│   ├── score.py                        # Score tracking (EasyOCR)
│   ├── trainer.py                      # Training stub (Phase 5)
│   └── requirements.txt
├── models/
│   └── manifest.json                   # Model registry
├── frontend/                           # Svelte UI (Wails)
│   └── src/
│       ├── App.svelte
│       └── lib/
│           ├── ProcessView.svelte
│           ├── ResultsView.svelte
│           └── ProgressBar.svelte
├── testdata/
│   └── sample_5s.mp4
├── go.mod
├── Makefile
└── README.md
```

---

## Task Groups

| Group | Name | Depends On | Tasks |
|-------|------|------------|-------|
| 1 | Project scaffolding + config | — | 1-3 |
| 2 | Video I/O | Group 1 | 4-5 |
| 3 | Tracker (Go) | Group 1 | 6-9 |
| 4 | Python ML core | — (independent) | 10-15 |
| 5 | Bridge | Group 1 | 16-19 |
| 6 | Export (Dartfish CSV) | Group 1 | 20-22 |
| 7 | Pipeline | Groups 2, 3, 5, 6 | 23-26 |
| 8 | Model download | Group 1 | 27-28 |
| 9 | Frontend UI + integration | Groups 7, 8 | 29-33 |

Groups 2, 3, 4, 5, 6 can run in parallel. Group 5 only needs Group 1 (skeletons don't call Python). Real Python integration happens in Task 32 (Group 9).

---

## Group 1: Project Scaffolding + Config

### Task 1: Initialize Go + Wails project

**Files:**
- Create: `tennis-tagger/go.mod`
- Create: `tennis-tagger/cmd/tagger/main.go`
- Create: `tennis-tagger/Makefile`

- [ ] **Step 1: Create project and init Go module**

```bash
mkdir -p ~/tennis-tagger && cd ~/tennis-tagger
go mod init github.com/liamp/tennis-tagger
```

- [ ] **Step 2: Install Wails CLI**

```bash
go install github.com/wailsapp/wails/v2/cmd/wails@latest
```

- [ ] **Step 3: Init Wails with Svelte template**

```bash
wails init -n tagger -t svelte
```

- [ ] **Step 4: Reorganize to target structure**

Move `main.go` → `cmd/tagger/main.go`, `app.go` → `internal/app/app.go`. Update imports.

- [ ] **Step 5: Create Makefile**

```makefile
.PHONY: dev build test clean
dev:
	wails dev
build:
	wails build
test:
	go test ./internal/... -v
clean:
	rm -rf build/
```

- [ ] **Step 6: Verify `make dev` opens window**

- [ ] **Step 7: Commit**

```bash
git init && git add -A && git commit -m "feat: initialize Wails + Svelte project"
```

---

### Task 2: Config system

**Files:**
- Create: `internal/config/config.go`
- Create: `internal/config/config_test.go`

- [ ] **Step 1: Write failing tests** — Load defaults, load from file, save round-trip

```go
// internal/config/config_test.go
package config

import (
    "os"
    "path/filepath"
    "testing"
)

func TestLoadConfig_Defaults(t *testing.T) {
    cfg, err := Load("")
    if err != nil { t.Fatalf("unexpected error: %v", err) }
    if cfg.ConfigVersion != 1 { t.Errorf("expected 1, got %d", cfg.ConfigVersion) }
    if cfg.Device != "auto" { t.Errorf("expected 'auto', got %s", cfg.Device) }
    if cfg.Pipeline.BatchSize != 32 { t.Errorf("expected 32, got %d", cfg.Pipeline.BatchSize) }
}

func TestLoadConfig_FromFile(t *testing.T) {
    dir := t.TempDir()
    path := filepath.Join(dir, "config.yaml")
    os.WriteFile(path, []byte("configVersion: 1\ndevice: cpu\n"), 0644)
    cfg, _ := Load(path)
    if cfg.Device != "cpu" { t.Errorf("expected 'cpu', got %s", cfg.Device) }
}

func TestSaveConfig(t *testing.T) {
    dir := t.TempDir()
    path := filepath.Join(dir, "config.yaml")
    cfg := Default()
    cfg.Device = "cuda"
    Save(cfg, path)
    loaded, _ := Load(path)
    if loaded.Device != "cuda" { t.Errorf("expected 'cuda', got %s", loaded.Device) }
}
```

- [ ] **Step 2: Run tests — expect FAIL**
- [ ] **Step 3: Implement config.go**

```go
package config

import (
    "os"
    "gopkg.in/yaml.v3"
)

type Config struct {
    ConfigVersion int            `yaml:"configVersion"`
    ModelsDir     string         `yaml:"modelsDir"`
    Device        string         `yaml:"device"`
    Pipeline      PipelineConfig `yaml:"pipeline"`
    Export        ExportConfig   `yaml:"export"`
    Training      TrainingConfig `yaml:"training"`
    ActiveModel   string         `yaml:"activeModel"`
}

type PipelineConfig struct {
    BatchSize       int  `yaml:"batchSize"`
    FrameSkip       int  `yaml:"frameSkip"`
    EnablePose      bool `yaml:"enablePose"`
    EnableScore     bool `yaml:"enableScore"`
    CheckpointEvery int  `yaml:"checkpointEvery"`
}

type ExportConfig struct {
    Format string `yaml:"format"`
}

type TrainingConfig struct {
    DefaultEpochs    int `yaml:"defaultEpochs"`
    DefaultBatchSize int `yaml:"defaultBatchSize"`
}

func Default() *Config {
    return &Config{
        ConfigVersion: 1, ModelsDir: "models", Device: "auto", ActiveModel: "v1",
        Pipeline: PipelineConfig{BatchSize: 32, FrameSkip: 1, EnablePose: true, CheckpointEvery: 1000},
        Export:   ExportConfig{Format: "dartfish"},
    }
}

func Load(path string) (*Config, error) {
    cfg := Default()
    if path == "" { return cfg, nil }
    data, err := os.ReadFile(path)
    if err != nil { return nil, err }
    if err := yaml.Unmarshal(data, cfg); err != nil { return nil, err }
    return cfg, nil
}

func Save(cfg *Config, path string) error {
    data, err := yaml.Marshal(cfg)
    if err != nil { return err }
    return os.WriteFile(path, data, 0644)
}
```

- [ ] **Step 4: `go get gopkg.in/yaml.v3` and run tests — expect PASS**
- [ ] **Step 5: Commit**

---

### Task 3: Shared types + BridgeBackend interface

**Files:**
- Create: `internal/bridge/types.go`

- [ ] **Step 1: Define types** — BBox, Frame, FrameClip, DetectionResult, StrokeResult, PlacementResult, CourtData, RallyResult, BridgeConfig, BridgeBackend interface

```go
package bridge

type BBox struct {
    X1, Y1, X2, Y2, Confidence float64
}
func (b BBox) Center() (float64, float64) { return (b.X1+b.X2)/2, (b.Y1+b.Y2)/2 }
func (b BBox) Width() float64  { return b.X2 - b.X1 }
func (b BBox) Height() float64 { return b.Y2 - b.Y1 }

type Frame struct { Data []byte; Width, Height int }
type FrameClip struct { Frames []Frame; Center int }

type DetectionResult struct {
    FrameIndex int      `json:"frame_index"`
    Players    []BBox   `json:"players"`
    Ball       *BBox    `json:"ball"`
    Poses      []PoseData `json:"poses"`
}

type PoseKeypoint struct { X, Y, Confidence float64 }
type PoseData struct { PlayerBBox BBox; Keypoints []PoseKeypoint }
type StrokeResult struct { Type string; Confidence float64; Frame, PlayerID int }
type PlacementResult struct { Zone, Depth string; Angle float64 }
type CourtData struct { Corners [4][2]float64; Homography [3][3]float64; Method string; Confidence float64 }
type RallyResult struct { StartFrame, EndFrame, NumStrokes, DurationFrames int }
type BridgeConfig struct { ModelsDir, Device string }
type TrainingPair struct { VideoPath, CSVPath string }
type TrainingConfig struct { Task string; Epochs, BatchSize int; Device string }

type BridgeBackend interface {
    Init(config BridgeConfig) error
    DetectBatch(frames []Frame) ([]DetectionResult, error)
    ClassifyStrokes(clips []FrameClip) ([]StrokeResult, error)
    AnalyzePlacements(detections []DetectionResult, court CourtData) ([]PlacementResult, error)
    SegmentRallies(detections []DetectionResult, fps float64) ([]RallyResult, error)
    DetectCourt(frame Frame) (CourtData, error)
    TrainModel(pairs []TrainingPair, config TrainingConfig) error
    Close()
}
```

- [ ] **Step 2: `go build ./internal/bridge/` — verify compiles**
- [ ] **Step 3: Commit**

---

## Group 2: Video I/O

### Task 4: VideoReader with ffmpeg

**Files:**
- Create: `internal/video/frame.go`, `internal/video/reader.go`, `internal/video/reader_test.go`

- [ ] **Step 1: Write tests** — Open invalid path (error), Open real video (metadata correct), ExtractBatch (correct frame count + size)
- [ ] **Step 2: Run tests — expect FAIL**
- [ ] **Step 3: Implement** — `probe()` via ffprobe JSON, `ExtractBatch()` via `ffmpeg -f rawvideo -pix_fmt rgb24 pipe:1`
- [ ] **Step 4: Run tests — expect PASS**
- [ ] **Step 5: Commit**

### Task 5: Generate test video fixture

- [ ] **Step 1: `ffmpeg -f lavfi -i testsrc=duration=5:size=640x480:rate=30 -c:v libx264 testdata/sample_5s.mp4`**
- [ ] **Step 2: Add integration tests using real video**
- [ ] **Step 3: Commit**

---

## Group 3: Tracker (Pure Go)

### Task 6: Kalman filter

**Files:**
- Create: `internal/tracker/kalman.go`, `internal/tracker/kalman_test.go`

State: `[x, y, w, h, vx, vy, vw]` (7D). Measurement: `[x, y, w, h]` (4D). Matches existing Python tracker's noise matrices.

- [ ] **Step 1: Write tests** — Predict (position stable at v=0), Update (moves toward measurement), Velocity estimation (10 frames moving right)
- [ ] **Step 2: Run tests — expect FAIL**
- [ ] **Step 3: Implement using gonum/mat** — `NewKalmanFilter(bbox)`, `Predict()`, `Update(measurement)`, `State()`, `BBox()`
- [ ] **Step 4: `go get gonum.org/v1/gonum` and run tests — expect PASS**
- [ ] **Step 5: Commit**

### Task 7: IoU + greedy assignment

**Files:**
- Create: `internal/tracker/hungarian.go`, `internal/tracker/hungarian_test.go`

- [ ] **Step 1: Write tests** — ComputeIoU (identical=1, non-overlapping=0, partial), Assign (2-2 match, unmatched detection)
- [ ] **Step 2: Implement** — `ComputeIoU(a, b [4]float64) float64`, `Assign(tracks, dets [][4]float64, threshold float64) ([]Match, []int, []int)`. Greedy assignment (sufficient for 2-4 tennis players).
- [ ] **Step 3: Run tests — expect PASS**
- [ ] **Step 4: Commit**

### Task 8: MultiObjectTracker

**Files:**
- Create: `internal/tracker/tracker.go`, `internal/tracker/tracker_test.go`

- [ ] **Step 1: Write tests** — Single object tracked, persistent ID across frames, track deletion after maxAge, new detection creates new track
- [ ] **Step 2: Implement** — `NewTracker(maxAge, minHits, iouThreshold)`, `Update(detections []BBox) []TrackedObject`, `Reset()`
- [ ] **Step 3: Run tests — expect PASS**
- [ ] **Step 4: Commit**

### Task 9: Tracker benchmark

- [ ] **Step 1: Write benchmark** — `BenchmarkTrackerUpdate` with 2 objects
- [ ] **Step 2: Run `go test -bench=. -benchmem` — expect sub-microsecond per call**
- [ ] **Step 3: Commit**

---

## Group 4: Python ML Core

### Task 10: Package setup

**Files:**
- Create: `ml/__init__.py`, `ml/requirements.txt`

- [ ] **Step 1: Create __init__.py** — `torch.set_num_threads(1)` + configure `logging.basicConfig(format="%(levelname)s %(name)s %(message)s")` so Python log output is parseable by Go's slog
- [ ] **Step 2: Create requirements.txt** — torch>=2.4, torchvision>=0.19, ultralytics>=8.3, opencv-python>=4.10, onnxruntime-gpu>=1.19, mediapipe>=0.10.14, numpy>=1.26,<2.0, scipy>=1.14, easyocr>=1.7

**Note:** All ML module constructors use explicit parameters (`model_path`, `device`) instead of the existing `config: dict` pattern. This is intentional — Go passes specific values, not raw dicts.
- [ ] **Step 3: Commit**

### Task 11: Transplant detector

**Files:**
- Create: `ml/detector.py`

**Source:** `detection/unified_detector.py` (257 LOC)

- [ ] **Step 1: Create detector.py** — `Detector(model_path, device)` with `detect_batch(frames: np.ndarray) -> list[dict]`. Single YOLO inference, split by CLASS_PERSON=0 / CLASS_SPORTS_BALL=32. Ball size filter 5-100px. Player confidence 0.5, ball 0.3. Ball history deque(maxlen=30).
- [ ] **Step 2: Verify import** — `python -c "from ml.detector import Detector; print('OK')"`
- [ ] **Step 3: Commit**

### Task 12: Transplant pose + serve detection

**Files:**
- Create: `ml/pose.py`

**Source:** `detection/pose_estimator.py` (255 LOC) + `detection/serve_detector.py` (91 LOC)

- [ ] **Step 1: Create pose.py** — `PoseEstimator(device)` with `estimate_batch()` and `detect_serves()`. YOLOv8n-pose primary, MediaPipe fallback. Serve detection: arm angle at elbow > 120 degrees + wrist above shoulder. Dedup within 30 frames.
- [ ] **Step 2: Verify import**
- [ ] **Step 3: Commit**

### Task 13: Transplant classifier

**Files:**
- Create: `ml/classifier.py`

**Source:** X3D 3D CNN from existing codebase

- [ ] **Step 1: Create classifier.py** — `Simple3DCNN(nn.Module)` with Conv3d layers (3→64→128→256), BatchNorm, AdaptiveAvgPool3d. `StrokeClassifier(model_path, device)` with `classify(clips: np.ndarray) -> list[dict]`. 8 classes: forehand, backhand, forehand_volley, backhand_volley, serve, smash, drop_shot, lob. Clip: 16 frames, 224x224. Confidence threshold 0.7.
- [ ] **Step 2: Verify import**
- [ ] **Step 3: Commit**

### Task 14: Transplant analyzer (court + placement + rally)

**Files:**
- Create: `ml/analyzer.py`

**Source:** `court_detector.py` (142 LOC) + `placement_analyzer.py` (207 LOC) + `rally_analyzer.py` (55 LOC)

- [ ] **Step 1: Create analyzer.py** — `Analyzer()` with `detect_court(frame)` (Canny + HoughLinesP + findHomography, cached), `analyze_placements(detections, court)` (6 zones: deuce/ad × wide/body/t, 3 depths: baseline/mid/net), `segment_rallies(detections, fps)` (max_gap=90 frames, min_length=2 strokes).
- [ ] **Step 2: Verify import**
- [ ] **Step 3: Commit**

### Task 15: Transplant score tracker + trainer stub

**Files:**
- Create: `ml/score.py`, `ml/trainer.py`

- [ ] **Step 1: Create score.py** — `ScoreTracker(device)` with `read_score(frame)`. EasyOCR lazy-loaded. OCR region: top-right 30%×15%. Score state machine (game: 4pts win by 2, set: 6 games win by 2, tiebreak at 6-6).
- [ ] **Step 2: Create trainer.py stub** — `Trainer(models_dir, device)` with `train()`, `fine_tune()`, `get_versions()`. Train/fine_tune raise NotImplementedError (Phase 5).
- [ ] **Step 3: Verify imports**
- [ ] **Step 4: Commit**

---

## Group 5: Bridge

### Task 16: Mock bridge

**Files:**
- Create: `internal/bridge/mock.go`

- [ ] **Step 1: Implement MockBridge** — Returns canned detections (2 players + 1 ball per frame), canned strokes (forehand), canned placements (deuce_wide), identity court. Counts calls for test assertions.
- [ ] **Step 2: Verify compiles**
- [ ] **Step 3: Commit**

### Task 17: Bridge worker

**Files:**
- Create: `internal/bridge/worker.go`

- [ ] **Step 1: Implement Worker** — `runtime.LockOSThread()`, request/response channels, `Call(method, payload)` blocks until response, `Stop()` closes channel. `StubCaller` for testing.
- [ ] **Step 2: Verify compiles**
- [ ] **Step 3: Commit**

### Task 18: Embedded bridge skeleton

**Files:**
- Create: `internal/bridge/embedded.go` (build tag: `//go:build cgo && python3`)

- [ ] **Step 1: Create skeleton** — `EmbeddedBridge` wraps Worker + embeddedCaller. All methods delegate to worker. embeddedCaller methods return "not yet implemented" errors. Comments explain go-python3 requirements.
- [ ] **Step 2: Verify compiles with tag** — `go build -tags python3 ./internal/bridge/`
- [ ] **Step 3: Commit**

### Task 19: Process bridge skeleton

**Files:**
- Create: `internal/bridge/process.go` (build tag: `//go:build !python3`)

- [ ] **Step 1: Create skeleton** — `ProcessBridge` wraps Worker + processCaller. Named pipes transport. All methods return "not yet implemented" errors.
- [ ] **Step 2: Verify compiles (default, no tag)** — `go build ./internal/bridge/`
- [ ] **Step 3: Commit**

---

## Group 6: Export (Dartfish CSV)

### Task 20: Column definitions

**Files:**
- Create: `internal/export/columns.go`

- [ ] **Step 1: Define all 62 Dartfish columns** — Copy exact string names from `files/analysis/csv_generator.py` COLUMNS list. Do NOT abbreviate or rename. The column names must be character-for-character identical to the existing Python output (e.g., `"0 - Point Level"` not `"0-Point Level"`, `"U1: Confidence Score"` not `"U1: Confidence"`, `"X1: Player 1 Name"` not `"X1: Player Names"`). Use a `[]string` slice (not fixed-size array) for extensibility.
- [ ] **Step 2: Verify compiles**
- [ ] **Step 3: Commit**

### Task 21: Dartfish exporter

**Files:**
- Create: `internal/export/dartfish.go`, `internal/export/dartfish_test.go`

- [ ] **Step 1: Write tests** — Header row has 62 columns with correct names, empty results produce header-only CSV
- [ ] **Step 2: Run tests — expect FAIL**
- [ ] **Step 3: Implement** — `DartfishExporter` with `Export(rows []ResultRow, w io.Writer)`, `ExportFile(rows, path)`. ResultRow has `Fields [62]string`.
- [ ] **Step 4: Run tests — expect PASS**
- [ ] **Step 5: Commit**

### Task 22: Timestamp formatter

- [ ] **Step 1: Write test** — 0s→"00:00.000", 65.5s→"01:05.500", 3661.123s→"61:01.123"
- [ ] **Step 2: Implement `FormatTimestamp(seconds float64) string`**
- [ ] **Step 3: Run tests — expect PASS**
- [ ] **Step 4: Commit**

---

## Group 7: Pipeline

### Task 23: Checkpoint system

**Files:**
- Create: `internal/pipeline/checkpoint.go`, `internal/pipeline/checkpoint_test.go`

- [ ] **Step 1: Write tests** — Save+Load round-trip, reject incompatible version
- [ ] **Step 2: Implement** — `Checkpoint{Version, VideoPath, ProcessedFrames, TotalFrames}`, `SaveCheckpoint()`, `LoadCheckpoint()` with version validation
- [ ] **Step 3: Run tests — expect PASS**
- [ ] **Step 4: Commit**

### Task 24: Pipeline orchestrator

**Files:**
- Create: `internal/pipeline/pipeline.go`, `internal/pipeline/pipeline_test.go`

- [ ] **Step 1: Write test** — Process test video with MockBridge, verify non-zero frames processed
- [ ] **Step 2: Implement Pipeline** — `New(bridge, config)`, `Process(videoPath) (*Result, error)`, `Progress() ProgressInfo`. Stages: open video → detect court (first frame) → batch loop (extract → detect → track → assemble) → post-process (rallies, placements) → done. Checkpoint every N frames. Channel buffer sizes: Stage 1→2 = 2 batches (backpressure), Stage 2→3 = 2 batches. **Detection validation:** after each DetectBatch call, drop BBoxes with NaN coordinates or confidence > 1.0, log warning. **CUDA OOM retry:** if DetectBatch returns a CUDA OOM error, retry with halved batch size. If retry fails, fall back to CPU with user notification via Wails event. **ffmpeg error handling:** capture stderr, return in error message on non-zero exit.
- [ ] **Step 3: Run tests — expect PASS**
- [ ] **Step 4: Commit**

### Task 25: App bindings

**Files:**
- Create/Modify: `internal/app/app.go`

- [ ] **Step 1: Implement App struct** — `NewApp(cfg, bridge)`, `Startup(ctx)`, `SelectVideo()` (native dialog), `ProcessVideo(path)` (goroutine + events), `GetProgress()`, `ExportCSV()` (save dialog), `GetDeviceInfo()`
- [ ] **Step 2: Verify compiles**
- [ ] **Step 3: Commit**

### Task 26: Svelte UI views

**Files:**
- Create: `frontend/src/lib/ProgressBar.svelte`, `ProcessView.svelte`, `ResultsView.svelte`
- Modify: `frontend/src/App.svelte`

- [ ] **Step 1: Create ProgressBar** — Animated bar with percent, stage, frame count
- [ ] **Step 2: Create ProcessView** — File picker button, video path display, process button, progress bar. Polls GetProgress every 500ms.
- [ ] **Step 3: Create ResultsView stub** — "Processing complete" message + Export CSV button
- [ ] **Step 4: Wire App.svelte** — Route between views, listen for processing-complete event
- [ ] **Step 5: `make dev` — verify UI renders**
- [ ] **Step 6: Commit**

---

## Group 8: Model Download

### Task 27: Model manifest + checker

**Files:**
- Create: `models/manifest.json`, `internal/models/manifest.go`, `internal/models/download_test.go`

- [ ] **Step 1: Create manifest.json** — detector (yolov8x.pt, 136MB, required), stroke (stroke_x3d.pt, 20MB, required), pose (yolov8n-pose.pt, 12MB, required)
- [ ] **Step 2: Implement** — `LoadManifest(path)`, `CheckModels(manifest, dir) []Model` (returns missing required models)
- [ ] **Step 3: Write tests** — All missing returns 1, none missing returns 0
- [ ] **Step 4: Run tests — expect PASS**
- [ ] **Step 5: Commit**

### Task 27b: Model downloader

**Files:**
- Create: `internal/models/download.go`

- [ ] **Step 1: Implement** — `DownloadModel(model Model, destDir string, progressFn func(downloaded, total int64)) error`. HTTP GET with streaming to temp file, SHA256 checksum verification, atomic rename to final path. `DownloadMissing(manifest, dir, progressFn) error` downloads all missing required models.
- [ ] **Step 2: Write test** — Mock HTTP server, verify download + checksum. Verify corrupt download (wrong checksum) returns error.
- [ ] **Step 3: Run tests — expect PASS**
- [ ] **Step 4: Commit**

### Task 28: Wire main.go entry point

**Files:**
- Modify: `cmd/tagger/main.go`

- [ ] **Step 1: Wire everything** — slog JSON logger, load config, create MockBridge (swap to real bridge later), create App, start Wails with title "Tennis Tagger", 1400x900, embed frontend dist
- [ ] **Step 2: `wails build` — verify binary produced**
- [ ] **Step 3: Commit**

---

## Group 9: Integration + Polish

### Task 29: End-to-end smoke test

- [ ] **Step 1: `make dev`, select test video, verify MockBridge processing runs, progress bar updates**
- [ ] **Step 2: Verify CSV export button works (produces file with 62-column header)**

### Task 30: Generate ground-truth CSV

- [ ] **Step 1: Run current Python app on sample_5s.mp4, save output as `testdata/expected_output.csv`**

### Task 31: CSV comparison integration test

- [ ] **Step 1: Write test** — Run pipeline → export → diff against expected_output.csv (column names match, row structure matches)

### Task 32: Wire real bridge

- [ ] **Step 1: Implement go-python3 embeddedCaller (or processCaller if cgo fails)**
- [ ] **Step 2: Verify with actual ML inference on test video**

### Task 33: Final polish

- [ ] **Step 1: Update README.md** — Prerequisites: Go 1.22+, Python 3.11, ffmpeg on PATH, Wails CLI, CGo toolchain (gcc/MinGW on Windows for go-python3). Build/run instructions.
- [ ] **Step 2: Final commit**

---

## Summary

| Group | Tasks | Deliverable |
|-------|-------|-------------|
| 1. Scaffolding | 1-3 | Go project, config, shared types |
| 2. Video I/O | 4-5 | ffmpeg frame extraction |
| 3. Tracker | 6-9 | Kalman + IoU in pure Go |
| 4. ML Core | 10-15 | 6 Python modules transplanted |
| 5. Bridge | 16-19 | Mock + embedded + process skeletons |
| 6. Export | 20-22 | 62-column Dartfish CSV in Go |
| 7. Pipeline | 23-26 | 3-stage orchestrator + UI |
| 8. Models | 27-27b-28 | Manifest + downloader + checker |
| 9. Integration | 29-33 | End-to-end verification |
