# Tennis Tagger v2 — Go + Python Hybrid Rewrite

**Date:** 2026-03-20
**Status:** Approved
**Approach:** Surgical Transplant — extract Python ML core, wrap in Go

---

## 1. Overview

Rewrite Tennis Tagger as a native desktop app using Go (Wails + Svelte) for UI, video I/O, orchestration, and export, with an embedded Python runtime for ML inference. The current Python codebase (~26 files) is stripped to 6 focused ML modules (tracking moved to Go). Everything else is rewritten in Go.

**Goals:**
- Fast, single-binary desktop app (no Python install required for users)
- Preserve working ML pipeline (85-98% accuracy)
- Upgrade to cutting-edge models (YOLOv11, RT-DETR, Video Swin Transformer)
- Add real-time live tagging, tactical analysis, and self-improving feedback loop
- Ship incrementally across 5 phases

---

## 2. Architecture

```
tennis-tagger/
  cmd/
    tagger/
      main.go                  # Wails app entry point
  internal/
    app/                       # Wails app bindings (Go <-> JS)
    video/                     # Video I/O (ffmpeg, frame extraction)
    pipeline/                  # Orchestration (batching, checkpoints, concurrency)
    bridge/                    # go-python3 embedding layer
    tracker/                   # Multi-object tracking (Kalman + Hungarian, pure Go)
    export/                    # CSV / Dartfish output generation
    config/                    # App configuration (YAML, versioned schema)
  frontend/                    # Wails UI (Svelte)
  ml/                          # Python ML core (embedded)
    detector.py                # YOLOv8/v11/RT-DETR unified detection
    pose.py                    # Pose estimation + serve detection
    classifier.py              # Stroke classification (3D CNN / Video Swin)
    analyzer.py                # Placement + rally + court detection
    score.py                   # Score tracking (EasyOCR, optional)
    trainer.py                 # Model training / fine-tuning / incremental learning
    requirements.txt           # Minimal Python deps
  models/                      # Pre-trained model weights (versioned, downloaded on first run)
    manifest.json              # Model registry with checksums
  docs/                        # Consolidated documentation
  go.mod
  Makefile
```

### Layer Responsibilities

| Layer | Language | Responsibility |
|-------|----------|----------------|
| UI | JS (Svelte) via Wails | Video player, controls, CSV preview, progress, settings, QC corrections |
| Orchestration | Go | Frame extraction (ffmpeg), batching, checkpoint/resume, file management, config, concurrent pipeline |
| Tracking | Go | Kalman filter + Hungarian algorithm (gonum). No ML, no Python dependency |
| Bridge | Go (cgo) | go-python3 embedding. Passes frame buffers to Python, receives structured detection results. Fallback: subprocess + shared memory (named pipes + mmap) |
| ML Core | Python | Pure inference: detection, classification, analysis, training. No I/O, no GUI, no tracking |

### Data Flow

```
Video file (user selects via native dialog)
  -> Go: ffmpeg frame extraction (goroutines, async)
    -> Go: batch frames (32-64) into C-heap buffers (C.malloc, pinned)
      -> Bridge: pass C pointer + dimensions to Python
        -> Python: numpy.frombuffer(ptr) -> detect -> classify -> analyze
        -> Python: return structured results (JSON)
      -> Go: receive detections
    -> Go: tracker.Update(detections) — Kalman + Hungarian in Go
    -> Go: checkpoint to disk
  -> Go: generate Dartfish CSV (62 columns)
  -> UI: display results table + download
```

### 3-Stage Concurrent Pipeline (Go)

```
Stage 1 (goroutine)       Stage 2 (dedicated OS thread)  Stage 3 (goroutine)
Frame extraction      ->  ML inference (Python)        ->  Result assembly
  - ffmpeg decode           - detect batch                   - tracking (Go)
  - C.malloc buffers        - classify                       - build CSV rows
  - frame skipping          - analyze                        - checkpoint
                            - runtime.LockOSThread()         - update UI progress
                            - all Python calls serialized
```

Channels connect stages with natural backpressure. Stage 2 runs on a locked OS thread (see Bridge Threading Model below).

---

## 3. Go <-> Python Bridge

### Interface

```go
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

type EmbeddedBridge struct { ... }  // go-python3 (primary)
type ProcessBridge struct { ... }   // subprocess + named pipes + mmap (fallback)
```

`Init()` creates persistent Python objects (Detector, Classifier, Analyzer, etc.) that live for the session. Each call to `DetectBatch`, `ClassifyStrokes`, etc. invokes methods on these persistent instances.

### Bridge Threading Model

**This is the highest-risk component. These rules are non-negotiable:**

1. **All Python calls run on a single dedicated goroutine** (the "bridge worker"). This goroutine calls `runtime.LockOSThread()` at startup and never unlocks. This guarantees CUDA context thread affinity.

2. **The bridge worker owns a request/response channel pair.** Other goroutines send work items to the request channel and block on a per-request response channel. Python calls are serialized.

3. **Concurrency comes from overlapping stages, not parallel Python calls.** Stage 1 (frame extraction) and Stage 3 (tracking + result assembly) run on separate goroutines and overlap with Stage 2 (Python inference). The GIL is never contested from the Go side.

4. **PyTorch's internal ATen thread pool is limited** to avoid conflict with Go's scheduler: `torch.set_num_threads(1)` at init. GPU operations release the GIL naturally; CPU-bound Python operations are serialized by design.

```go
type bridgeWorker struct {
    requests  chan bridgeRequest
    pyModules map[string]*python3.PyObject
}

type bridgeRequest struct {
    method   string
    args     []interface{}
    response chan bridgeResponse
}

func (w *bridgeWorker) run() {
    runtime.LockOSThread()
    // Initialize Python, load modules, enter loop
    for req := range w.requests {
        result, err := w.callPython(req.method, req.args)
        req.response <- bridgeResponse{result, err}
    }
}
```

### Frame Data Transfer (Minimal-Copy)

True zero-copy is not achievable due to GPU upload requirements. The actual memory model:

1. **Go allocates frames on the C heap** (`C.malloc`) — pinned memory, not subject to Go GC relocation
2. **Python wraps C pointer as numpy array** via `numpy.frombuffer` — no copy on the CPU side
3. **First real copy happens during model preprocessing** — resize, normalize, `torch.Tensor.to(device)` uploads to GPU
4. **The GPU upload is the true latency bottleneck**, not CPU-side memory transfer. Profiling should focus here.
5. **Go frees C buffers** after Python returns results for that batch

### Result Types

```go
type DetectionResult struct {
    Players  []BBox      `json:"players"`
    Ball     *BBox       `json:"ball"`
    Poses    []PoseData  `json:"poses"`
}

type StrokeResult struct {
    Type       string  `json:"type"`
    Confidence float64 `json:"confidence"`
    Frame      int     `json:"frame"`
}

type PlacementResult struct {
    Zone  string  `json:"zone"`
    Depth string  `json:"depth"`
    Angle float64 `json:"angle"`
}
```

Note: `TrackIDs` removed from `DetectionResult` — tracking is now handled in Go after detections are returned.

### Fallback Strategy

If go-python3 + CUDA proves unstable during Phase 1 development, swap to `ProcessBridge`:

- **Transport:** Named pipes (Windows) or Unix domain sockets (Linux/Mac) for commands + JSON results. Memory-mapped files for frame buffers (avoids serializing raw pixels).
- **Trigger:** Any of: segfault in cgo, CUDA context errors, GIL deadlocks, or ATen thread pool conflicts.
- **Switch:** Requires rebuild with `ProcessBridge` as the `BridgeBackend` implementation. Not a runtime toggle — the decision is made during Phase 1 development and baked in.

### Failure Modes

| Failure | Behavior |
|---------|----------|
| ffmpeg fails mid-extraction (corrupt video, missing codec) | Abort with user-facing error message. Partial results up to failure point are preserved via checkpoint. |
| Python model fails to load (missing weights) | `Init()` returns error. UI shows "Download models" prompt, triggers model download from manifest. |
| CUDA out of memory | Python catches `torch.cuda.OutOfMemoryError`, returns error. Bridge retries with halved batch size. If still fails, falls back to CPU with user notification. |
| Detection returns garbage (NaN bbox, confidence > 1.0) | Go-side validation after each `DetectBatch` call. Invalid detections are dropped with warning log. |
| Checkpoint schema mismatch (version upgrade) | Config has `checkpointVersion` field. Incompatible checkpoints are discarded with user notification; processing restarts from beginning. |
| Python process crash (ProcessBridge only) | Go detects pipe close, respawns Python process, reinitializes models, resumes from last checkpoint. |

---

## 4. Python ML Core

6 modules stripped from current 26-file codebase. Pure inference, no I/O, no GUI. Tracking moved to Go.

### Module APIs

```python
# ml/detector.py
class Detector:
    def __init__(self, model_path: str, device: str): ...
    def detect_batch(self, frames: np.ndarray) -> list[dict]: ...

# ml/pose.py
class PoseEstimator:
    def __init__(self, device: str): ...
    def estimate_batch(self, frames: np.ndarray, players: list[list[dict]]) -> list[list[dict]]: ...
    def detect_serves(self, poses: list[dict]) -> list[dict]: ...

# ml/classifier.py
class StrokeClassifier:
    def __init__(self, model_path: str, device: str): ...
    def classify(self, clips: np.ndarray) -> list[dict]: ...

# ml/analyzer.py
class Analyzer:
    def __init__(self): ...
    def detect_court(self, frame: np.ndarray) -> dict: ...
    def analyze_placements(self, detections: list[dict], court: dict) -> list[dict]: ...
    def segment_rallies(self, detections: list[dict], fps: float) -> list[dict]: ...

# ml/score.py
class ScoreTracker:
    def __init__(self, device: str): ...
    def read_score(self, frame: np.ndarray) -> dict | None: ...

# ml/trainer.py
class Trainer:
    def __init__(self, models_dir: str, device: str): ...
    def train(self, pairs: list[dict], config: dict) -> dict: ...
    def fine_tune(self, corrections: list[dict], config: dict) -> dict: ...
    def get_versions(self) -> list[dict]: ...
```

All classes are instantiated once during `BridgeBackend.Init()` and persist for the session.

**Note on constructor simplification:** The existing Python code uses `config: dict` constructors throughout. The new modules use explicit parameters (`model_path`, `device`) instead. This is intentional — Go passes specific values from its config, not a raw dict. EasyOCR is lazy-loaded in `ScoreTracker` (not at `__init__` time) to avoid heavy imports when score tracking is disabled.

### What's Kept vs Thrown Away

**Kept (transplanted to ml/):**
- Unified detection pipeline (merged from 3 detector files)
- Pose estimation (YOLOv8-pose + MediaPipe fallback)
- Serve detection (arm angle analysis)
- 3D CNN stroke classification (X3D architecture)
- Court detection (Canny + Hough + homography via cv2)
- Placement analysis (6-zone grid + depth)
- Rally segmentation (ball gap detection)
- Score tracking (EasyOCR, lazy-loaded)
- Training pipelines (merged from 3 training files)

**Moved to Go:**
- Multi-object tracking (Kalman filter + Hungarian algorithm) — no ML involved, pure linear algebra, better as native Go with gonum

**Thrown away:**
- All GUI code: app.py, gui.py, app_streamlined.py, tagging_desktop.py, launcher.py
- Orchestration: main.py (replaced by Go pipeline/)
- CSV export: csv_generator.py (replaced by Go export/)
- Batch processing: batch_process.py (replaced by Go)
- All 37 documentation files (consolidated)
- mnt/ directory (dead Docker artifact)
- All diagnostic/setup scripts (pywebview, webview2, .bat files)
- setup.py, setup_complete.py, setup_complete.ps1

### Dependencies (audited against actual imports in transplanted modules)

```
torch>=2.4
torchvision>=0.19
ultralytics>=8.3           # YOLOv11 support, includes ByteTrack
opencv-python>=4.10        # cv2.Canny, cv2.HoughLines, cv2.findHomography (court detection)
onnxruntime-gpu>=1.19
mediapipe>=0.10.14         # pose fallback
numpy>=1.26,<2.0          # stay on 1.x — numpy 2.0 has breaking changes (np.bool, np.int removed, copy semantics changed). Migrate to 2.x in Phase 2.
scipy>=1.14                # scipy.optimize.linear_sum_assignment used in analyzer placement matching
easyocr>=1.7               # optional, lazy-loaded in score.py
```

**Removed from Python deps (now in Go):**
- `filterpy` — Kalman filters reimplemented in Go with gonum
- `lap` — Hungarian algorithm via gonum

**Note on numpy 2.0:** The existing ML code uses `np.bool`, `np.int`, and relies on pre-2.0 copy semantics. Migrating to numpy 2.0 during Phase 1 adds unnecessary risk. Pin to 1.x and upgrade in Phase 2 when the ML core is stable.

---

## 5. Go-Side Components

### internal/video/

```go
type VideoReader struct {
    path   string
    fps    float64
    width  int
    height int
    total  int
}

func Open(path string) (*VideoReader, error)
func (v *VideoReader) ExtractBatch(start, count int) ([]Frame, error)
func (v *VideoReader) Metadata() VideoMeta
func (v *VideoReader) Close()
```

Uses `ffmpeg -f rawvideo` piped to stdout. Goroutines prefetch frames while ML processes previous batch.

**Failure handling:** If ffmpeg exits non-zero or stdout closes unexpectedly, `ExtractBatch` returns an error with the ffmpeg stderr output. The pipeline preserves partial results via checkpoint.

### internal/tracker/

```go
type MultiObjectTracker struct {
    tracks    map[int]*Track
    nextID    int
    maxAge    int  // frames before track deletion
}

type Track struct {
    ID            int
    State         KalmanState  // [x, y, w, h, vx, vy, vw] (7D, matching existing Python tracker)
    HitCount      int
    TimeSinceUpdate int
}

func NewTracker(maxAge int) *MultiObjectTracker
func (t *MultiObjectTracker) Update(detections []BBox) []TrackedObject
func (t *MultiObjectTracker) Reset()
```

Kalman filter predict/update cycle using `gonum/mat`. Hungarian algorithm for detection-to-track assignment using `gonum/optimize`. This eliminates two Python dependencies and one bridge round-trip per frame batch.

### internal/pipeline/

```go
type Pipeline struct {
    bridge     BridgeBackend
    tracker    *tracker.MultiObjectTracker
    video      *VideoReader
    config     PipelineConfig
    checkpoint *Checkpoint
}

type PipelineConfig struct {
    BatchSize       int
    FrameSkip       int
    EnablePose      bool
    EnableScore     bool
    CheckpointEvery int
    Device          string
}

func (p *Pipeline) Process(videoPath string, opts ...Option) (*Result, error)
func (p *Pipeline) Resume(checkpointPath string) (*Result, error)
func (p *Pipeline) ProcessLive(stream io.Reader) (<-chan DetectionResult, error)
```

### internal/export/

```go
type DartfishExporter struct {
    columns []ColumnDef
}

type ColumnDef struct {
    Name      string
    Group     string  // A-Z hierarchical grouping
    Format    func(row *ResultRow) string
    Required  bool
}

func (e *DartfishExporter) Export(results *Result, w io.Writer) error
func (e *DartfishExporter) ExportFile(results *Result, path string) error
```

62-column Dartfish-compatible CSV generation in pure Go. Column definitions are extracted from the existing `csv_generator.py` as a reference specification during Phase 1 implementation. The existing Python output serves as the ground-truth test fixture — Phase 1 exit criteria requires identical CSV output.

### internal/config/

```go
type Config struct {
    ConfigVersion int            `yaml:"configVersion"`  // schema version for migration
    ModelsDir     string         `yaml:"modelsDir"`
    Device        string         `yaml:"device"`
    Pipeline      PipelineConfig `yaml:"pipeline"`
    Export        ExportConfig   `yaml:"export"`
    Training      TrainingConfig `yaml:"training"`
    ActiveModel   string         `yaml:"activeModel"`
}
```

YAML config, stored in user app data directory. `ConfigVersion` field enables schema migration between app versions. Tracks active model version.

---

## 6. Model Distribution

### Problem

Model weights are large: YOLOv8x (~130MB), X3D stroke model (~20MB), EasyOCR (~100MB). Total ~300-500MB. Bundling in the binary defeats the "single small binary" goal.

### Solution: Download on First Run

```
models/
  manifest.json              # shipped with binary
  v1/                        # downloaded on first run
    yolov8x.pt
    stroke_x3d.pt
    yolov8n-pose.pt
  v2/                        # created by training
    stroke_x3d_v2.pt
```

**manifest.json:**
```json
{
  "version": "1.0",
  "baseUrl": "https://models.tennistagger.dev/",
  "models": {
    "detector": {
      "file": "yolov8x.pt",
      "size": 136000000,
      "sha256": "abc123...",
      "required": true
    },
    "stroke": {
      "file": "stroke_x3d.pt",
      "size": 20000000,
      "sha256": "def456...",
      "required": true
    },
    "pose": {
      "file": "yolov8n-pose.pt",
      "size": 12000000,
      "sha256": "ghi789...",
      "required": true
    },
    "score": {
      "file": "easyocr_en.zip",
      "size": 100000000,
      "sha256": "jkl012...",
      "required": false
    }
  }
}
```

**Behavior:**
1. On first launch, app checks `models/` against manifest
2. Missing or corrupt models (checksum mismatch) trigger download prompt in UI
3. Download shows progress bar per model
4. User-trained model versions (v2, v3...) are stored alongside and never overwritten
5. Garbage collection: UI shows disk usage per version, user can delete old versions

---

## 7. Frontend UI (Wails + Svelte)

### 5 Views

**Process View (main screen):**
- Native file picker, video preview thumbnail
- Device selector (GPU/CPU auto-detected)
- Processing settings (batch size, frame skip, enable score tracking)
- Process button, real-time progress bar with ETA
- Live stats during processing

**Results View:**
- Virtualized sortable/filterable data table (62 columns)
- Video playback synced to CSV rows (click row -> jump to frame)
- Export button (Dartfish CSV)
- Summary stats panel (rallies, stroke distribution, placement heatmap)

**QC / Corrections View:**
- Video player with frame-by-frame controls
- Detection overlay (bboxes, labels, zones)
- Click to correct: change stroke type, adjust placement, mark false positives
- Corrections saved to file
- Retrain button when threshold reached

**Training View:**
- Drag & drop video + CSV pairs
- Training progress (loss curve, accuracy, epoch)
- Model version history with accuracy comparison
- Set Active button to switch model version

**Settings:**
- Model directory, GPU/CPU preference, default export format
- Checkpoint behavior, live input source (Phase 3)

### Wails Bindings (Go -> JS)

```go
type App struct {
    pipeline *pipeline.Pipeline
    bridge   bridge.BridgeBackend
    config   *config.Config
}

func (a *App) ProcessVideo(path string) error
func (a *App) GetProgress() ProgressInfo
func (a *App) GetResults() *Result
func (a *App) SaveCorrection(correction Correction) error
func (a *App) StartTraining(pairs []TrainingPair) error
func (a *App) GetTrainingProgress() TrainingProgress
func (a *App) ListModelVersions() []ModelVersion
func (a *App) SetActiveModel(version string) error
func (a *App) GetDeviceInfo() DeviceInfo
func (a *App) DownloadModels() error
func (a *App) GetModelDownloadProgress() DownloadProgress
```

Wails auto-generates TypeScript bindings from Go methods.

---

## 8. Training & Tagging Flows

### Training Flow

```
User provides: video file + corrected CSV (ground truth tags)
  -> Go: extract frames, pair with CSV labels
    -> Bridge: pass frames + labels to Python
      -> Python trainer.py:
        - Fine-tune detector (YOLOv11) on player/ball positions
        - Train stroke classifier (3D CNN) on labeled stroke clips
        - Train placement model on labeled shot zones
        - Save versioned weights (models/v1/ -> models/v2/ -> ...)
      -> Go: update config to point to new model version
```

### Tagging Flow

```
User provides: raw video file (no labels)
  -> Go: extract frames
    -> Bridge: pass frames to Python
      -> Python: detect -> classify -> analyze -> score
    -> Go: tracker.Update(detections) — tracking in Go
    -> Go: receive structured results
  -> Go: generate Dartfish CSV (62 columns)
  -> UI: display results, allow QC corrections
```

### Self-Improving Loop (Phase 5)

```
Tag video -> User corrects mistakes in UI -> Corrections saved
  -> Accumulate corrections until threshold (e.g. 100 corrected events)
    -> Auto-retrain on corrections (incremental, low LR)
      -> New model version deployed
        -> Next video is more accurate
```

Training modes:
- **Initial training** from video + CSV pairs
- **Fine-tuning** from QC corrections
- **Incremental learning** without catastrophic forgetting (low LR + replay buffer)

---

## 9. Phased Delivery

### Phase 1: Core Rewrite
- Go project scaffolding (Wails + Svelte)
- internal/video/ — ffmpeg frame extraction with goroutine pipeline
- internal/bridge/ — go-python3 embedding with fallback interface
- internal/tracker/ — Kalman + Hungarian in pure Go (gonum)
- ml/ — stripped-down Python ML core (6 modules from current codebase)
- internal/pipeline/ — 3-stage concurrent processing
- internal/export/ — Dartfish CSV generation in Go (column spec extracted from existing csv_generator.py)
- Model download system (manifest + first-run download)
- Process View + Results View in UI
- Checkpoint/resume in Go
- Structured logging (Go: slog, Python: logging to stdout captured by Go)
- **Exit criteria (correctness):** process a tennis video end-to-end, produce identical CSV to current Python app
- **Exit criteria (performance):** process a 60-minute video faster than the current Python app on same hardware

### Phase 2: Upgraded ML
- Swap YOLOv8 -> YOLOv11 in ml/detector.py
- Add RT-DETR as alternative detector (config toggle)
- Upgrade stroke classifier: Video Swin Transformer alongside 3D CNN
- Improve court detection (handle more camera angles)
- Migrate numpy to 2.x (fix deprecated APIs in ML code)
- Benchmark accuracy improvements against Phase 1 baseline
- **Exit criteria:** measurable accuracy improvement on test set of 5+ videos

### Phase 3: Real-Time Live Tagging
- Pipeline.ProcessLive() — streaming frame input via RTSP/camera
- Go-side: live ffmpeg capture from camera/stream URL
- UI: live video feed with detection overlays in real-time
- Reduced latency mode (smaller batches, skip non-critical analysis)
- Settings view for stream configuration
- **Exit criteria:** connect to webcam/RTSP stream, display live detections with <500ms latency

### Phase 4: Match Context & Tactical Analysis
- Game state machine in Go (points -> games -> sets -> match)
- Pattern detection ("Player A targets backhand on break points")
- Tactical summary generation (per-set, per-match)
- New UI panel: match insights / tactical report
- Historical comparison across matches
- **Exit criteria:** generates tactical summary a coach would find useful

### Phase 5: Self-Improving Feedback Loop
- QC Corrections View fully wired
- Correction accumulation + threshold trigger
- ml/trainer.py incremental retraining from corrections
- Model versioning with rollback
- Accuracy tracking per version
- Training View UI complete
- **Exit criteria:** corrections from 3+ videos measurably improve detection on subsequent videos

Each phase is independently shippable.

---

## 10. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| App framework | Wails (Go + Svelte) | Single binary, native dialogs, no Electron bloat, web tech for rich UI |
| ML integration | Embedded Python (go-python3) | Tightest integration, single process, full access to torch/CUDA ecosystem |
| Bridge fallback | Subprocess + named pipes + mmap | If cgo + CUDA is unstable, same API surface, different transport. Decision made during Phase 1, not runtime |
| Bridge threading | Single dedicated OS thread, serialized Python calls | Avoids GIL contention, guarantees CUDA thread affinity, concurrency from stage overlap |
| Frame transfer | Minimal-copy (C.malloc + numpy.frombuffer) | C-heap allocation avoids GC relocation. First real copy at GPU upload |
| Tracking | Pure Go (gonum) | No ML involved, eliminates filterpy/lap deps, reduces bridge round-trips |
| UI framework | Svelte | Lighter than React, reactive model fits desktop app, smaller bundle |
| Video I/O | Go + ffmpeg (os/exec pipe) | No cgo dependency for video, goroutines for async prefetch |
| CSV export | Pure Go | No reason for Python here, type-safe, fast. Column spec from existing csv_generator.py |
| Model format | PyTorch native + ONNX export path | PyTorch for training flexibility, ONNX as future optimization path |
| Model distribution | Download on first run with manifest + checksums | Keeps binary small (~15MB), models downloaded once (~300-500MB) |
| Config | Versioned YAML in user app data dir | configVersion field enables schema migration between app versions |
| Checkpoint | Versioned JSON on disk (Go-managed) | Resume long videos, survive crashes. Incompatible versions discarded cleanly |
| numpy version | Pin to 1.x in Phase 1, upgrade in Phase 2 | Avoids breaking API changes during the riskiest phase |
| Logging | Go: slog. Python: logging to stdout, captured by Go | Unified structured logging, Python output visible in Go logs |
| Court detection | Run once per video, cache result | Static homography for fixed camera. Re-detect only if camera angle changes (Phase 3) |
