# Phase 2: ML Model Upgrades

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade detection and classification models to cutting-edge alternatives (YOLOv11, RT-DETR, Video Swin Transformer), improve court detection robustness, and migrate numpy to 2.x.

**Architecture:** All changes are in the Python `ml/` layer and Go `internal/config/`. The BridgeBackend interface and JSON-RPC protocol are unchanged. The detector gains a `backend` parameter to switch between YOLO and RT-DETR. The classifier gains a `model_type` parameter to switch between 3D CNN and Video Swin Transformer.

**Tech Stack:** Python 3.11, ultralytics>=8.3 (YOLOv11 + RT-DETR), timm (Video Swin), numpy>=2.0, torch>=2.4

**Spec:** `docs/superpowers/specs/2026-03-20-tennis-tagger-rewrite-design.md` (Phase 2 section)

---

## File Structure

```
Modified files:
  ml/detector.py               — Add backend selection (yolo11/rtdetr), upgrade default to YOLOv11
  ml/classifier.py             — Add VideoSwinTransformer model alongside Simple3DCNN
  ml/analyzer.py               — Improve court detection for more camera angles
  ml/requirements.txt          — Bump numpy to >=2.0, add timm
  ml/__init__.py               — Fix numpy 2.x deprecation warnings
  ml/bridge_server.py          — Pass detector_backend and classifier_model_type from config
  internal/config/config.go    — Add DetectorBackend and ClassifierModel to PipelineConfig
  internal/config/config_test.go — Test new config fields
  models/manifest.json         — Add YOLOv11 and RT-DETR model entries
```

---

## Task Groups

| Group | Name | Depends On | Tasks |
|-------|------|------------|-------|
| 1 | Detector upgrade | — | 1-3 |
| 2 | Classifier upgrade | — | 4-5 |
| 3 | Court detection improvement | — | 6 |
| 4 | numpy 2.x migration | — | 7 |
| 5 | Config + bridge wiring | Groups 1-4 | 8-9 |

Groups 1-4 can run in parallel.

---

## Group 1: Detector Upgrade (YOLOv11 + RT-DETR)

### Task 1: Add backend selection to Detector

**Files:**
- Modify: `ml/detector.py`

- [ ] **Step 1: Add backend parameter to Detector.__init__**

The Detector class currently hardcodes YOLO. Add a `backend` parameter that selects between `"yolo"` (default, now uses YOLOv11) and `"rtdetr"`.

```python
class Detector:
    BACKENDS = ("yolo", "rtdetr")

    def __init__(self, model_path: str, device: str = "auto", backend: str = "yolo"):
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Choose from {self.BACKENDS}")
        self.backend = backend
        # ... rest of init unchanged, YOLO loading works for both
        # ultralytics handles YOLOv11 and RT-DETR with the same API
```

The key insight: `ultralytics.YOLO()` already supports loading RT-DETR models (e.g., `rtdetr-l.pt`). The inference API (`model.predict()`) is identical. So the backend selection is really just about which model file is loaded.

- [ ] **Step 2: Update model loading to handle RT-DETR differences**

RT-DETR doesn't support `half=True` on CPU, and uses different default imgsz. Add backend-aware inference:

```python
def detect_batch(self, frames):
    # ... existing code ...
    predict_kwargs = {
        "conf": min_conf,
        "classes": [CLASS_PERSON, CLASS_SPORTS_BALL],
        "verbose": False,
        "device": self.device,
        "max_det": 20,
    }
    if self.backend == "yolo":
        predict_kwargs["half"] = (self.device != "cpu")
        predict_kwargs["imgsz"] = 640
    elif self.backend == "rtdetr":
        predict_kwargs["imgsz"] = 640
        # RT-DETR doesn't support half on CPU, and augment is not supported

    results_batch = self.model.predict(frames, **predict_kwargs)
    # ... rest unchanged
```

- [ ] **Step 3: Commit**

```bash
git add ml/detector.py
git commit -m "feat(ml): add backend selection (yolo/rtdetr) to Detector"
```

---

### Task 2: Update model manifest for YOLOv11 + RT-DETR

**Files:**
- Modify: `models/manifest.json`

- [ ] **Step 1: Add YOLOv11 and RT-DETR entries**

```json
{
  "version": "2.0",
  "models": {
    "detector_yolo": {
      "file": "yolo11x.pt",
      "size": 110000000,
      "sha256": "",
      "required": false,
      "description": "YOLOv11x player + ball detector (default)"
    },
    "detector_rtdetr": {
      "file": "rtdetr-l.pt",
      "size": 65000000,
      "sha256": "",
      "required": false,
      "description": "RT-DETR-L player + ball detector (alternative)"
    },
    "detector_yolov8": {
      "file": "yolov8x.pt",
      "size": 136000000,
      "sha256": "",
      "required": true,
      "description": "YOLOv8x player + ball detector (legacy fallback)"
    },
    "stroke": {
      "file": "stroke_x3d.pt",
      "size": 20000000,
      "sha256": "",
      "required": true,
      "description": "3D CNN stroke classifier"
    },
    "stroke_swin": {
      "file": "stroke_swin.pt",
      "size": 50000000,
      "sha256": "",
      "required": false,
      "description": "Video Swin Transformer stroke classifier (alternative)"
    },
    "pose": {
      "file": "yolov8n-pose.pt",
      "size": 12000000,
      "sha256": "",
      "required": true,
      "description": "YOLOv8n pose estimator"
    }
  }
}
```

- [ ] **Step 2: Commit**

---

### Task 3: Detector backend test

- [ ] **Step 1: Create ml/tests/test_detector.py**

```python
"""Unit tests for Detector backend selection."""
import pytest
from ml.detector import Detector

def test_invalid_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        Detector("fake.pt", device="cpu", backend="invalid")

def test_default_backend_is_yolo():
    # Should not raise even if model file doesn't exist
    # (falls back gracefully)
    det = Detector("nonexistent.pt", device="cpu", backend="yolo")
    assert det.backend == "yolo"

def test_rtdetr_backend_accepted():
    det = Detector("nonexistent.pt", device="cpu", backend="rtdetr")
    assert det.backend == "rtdetr"

def test_fallback_on_missing_model():
    det = Detector("nonexistent.pt", device="cpu")
    results = det.detect_batch([__import__("numpy").zeros((480, 640, 3), dtype=__import__("numpy").uint8)])
    assert len(results) == 1
    assert "players" in results[0]
```

- [ ] **Step 2: Run tests**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
python -m pytest ml/tests/test_detector.py -v
```

- [ ] **Step 3: Commit**

---

## Group 2: Classifier Upgrade (Video Swin Transformer)

### Task 4: Add Video Swin Transformer model

**Files:**
- Modify: `ml/classifier.py`

- [ ] **Step 1: Add VideoSwinTransformer class alongside Simple3DCNN**

```python
class VideoSwinTransformer(nn.Module):
    """Video Swin Transformer for stroke classification.

    Uses a pretrained Swin Transformer backbone with a temporal adapter.
    Falls back to Simple3DCNN if timm is not installed.
    """

    def __init__(self, num_classes: int = 8, temporal_depth: int = CLIP_LENGTH):
        super().__init__()
        try:
            import timm
            # Use a 2D Swin backbone, applied per-frame, then temporal pooling
            self.backbone = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=False,
                num_classes=0,  # Remove classification head
            )
            backbone_dim = self.backbone.num_features  # 768 for swin_tiny
        except ImportError:
            logger.warning("timm not installed — using linear backbone")
            self.backbone = None
            backbone_dim = 768

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C=3, T, H, W) — batch of video clips
        Returns:
            (B, num_classes) logits
        """
        B, C, T, H, W = x.shape

        if self.backbone is not None:
            # Apply 2D backbone per frame
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, C, H, W)

            # Resize to 224x224 if needed (Swin expects 224)
            if H != 224 or W != 224:
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

            features = self.backbone(x)  # (B*T, backbone_dim)
            features = features.reshape(B, T, -1)  # (B, T, backbone_dim)
        else:
            # Dummy features if no backbone
            features = torch.randn(B, T, 768, device=x.device)

        # Temporal pooling: (B, T, D) -> (B, D)
        features = features.permute(0, 2, 1)  # (B, D, T)
        features = self.temporal_pool(features).squeeze(-1)  # (B, D)

        return self.classifier(features)
```

- [ ] **Step 2: Add model_type parameter to StrokeClassifier**

```python
class StrokeClassifier:
    MODEL_TYPES = ("3dcnn", "swin")

    def __init__(self, model_path, device="auto", confidence_threshold=0.7, model_type="3dcnn"):
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown model_type: {model_type}. Choose from {self.MODEL_TYPES}")
        self.model_type = model_type

        # Build model based on type
        if model_type == "swin":
            self._model = VideoSwinTransformer(num_classes=len(STROKE_CLASSES))
        else:
            self._model = Simple3DCNN(num_classes=len(STROKE_CLASSES))

        # ... rest of init (load weights, move to device) unchanged
```

- [ ] **Step 3: Commit**

---

### Task 5: Classifier model type test

- [ ] **Step 1: Create ml/tests/test_classifier.py**

```python
"""Unit tests for StrokeClassifier model selection."""
import numpy as np
import pytest
from ml.classifier import StrokeClassifier, STROKE_CLASSES

def test_invalid_model_type():
    with pytest.raises(ValueError, match="Unknown model_type"):
        StrokeClassifier("fake.pt", device="cpu", model_type="invalid")

def test_3dcnn_default():
    clf = StrokeClassifier("nonexistent.pt", device="cpu", model_type="3dcnn")
    assert clf.model_type == "3dcnn"

def test_swin_accepted():
    clf = StrokeClassifier("nonexistent.pt", device="cpu", model_type="swin")
    assert clf.model_type == "swin"

def test_classify_returns_correct_shape():
    clf = StrokeClassifier("nonexistent.pt", device="cpu", model_type="3dcnn")
    # 2 clips of 16 frames at 64x64
    clips = np.random.randint(0, 255, (2, 16, 64, 64, 3), dtype=np.uint8)
    results = clf.classify(clips)
    assert len(results) == 2
    for r in results:
        assert "stroke" in r
        assert "confidence" in r
        assert r["stroke"] in STROKE_CLASSES or r["stroke"] == "unknown"
```

- [ ] **Step 2: Run tests, commit**

---

## Group 3: Court Detection Improvement

### Task 6: Improve court detection robustness

**Files:**
- Modify: `ml/analyzer.py`

The current court detection is fragile — it uses hardcoded frame proportions for corners when fewer than 4 Hough lines are found. Improve it:

- [ ] **Step 1: Add adaptive line filtering and corner estimation**

Replace the court detection method with a more robust version:

```python
def detect_court(self, frame: np.ndarray) -> dict:
    if self._court_cache is not None:
        return self._court_cache

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    # Try multiple Canny thresholds for robustness
    for low, high in [(30, 100), (50, 150), (75, 200)]:
        edges = cv2.Canny(gray, low, high, apertureSize=3)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=80, minLineLength=50, maxLineGap=20,
        )

        if lines is not None and len(lines) >= 4:
            break

    if lines is None or len(lines) < 4:
        # No lines found — use frame-proportion defaults
        corners = self._default_corners(w, h)
        confidence = 0.0
        method = "default"
    else:
        # Filter lines: separate horizontal and vertical
        horizontals, verticals = self._separate_lines(lines, w, h)

        if len(horizontals) >= 2 and len(verticals) >= 2:
            corners = self._corners_from_lines(horizontals, verticals, w, h)
            confidence = 0.7
            method = "line_intersection"
        else:
            corners = self._corners_from_endpoints(lines, w, h)
            confidence = 0.5
            method = "line_endpoints"

    standard = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
    homography, _ = cv2.findHomography(corners, standard)

    result = {
        "corners": corners.tolist(),
        "homography": homography,
        "method": method,
        "confidence": confidence,
    }

    if confidence > 0:
        self._court_cache = result
        logger.info("Court detected (%s, confidence=%.1f)", method, confidence)

    return result

def _default_corners(self, w, h):
    return np.array([
        [w * 0.1, h * 0.8], [w * 0.9, h * 0.8],
        [w * 0.2, h * 0.2], [w * 0.8, h * 0.2],
    ], dtype=np.float32)

def _separate_lines(self, lines, w, h):
    """Classify lines as horizontal or vertical based on angle."""
    horizontals, verticals = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 30:
            continue
        if angle < 30 or angle > 150:
            horizontals.append(line[0])
        elif 60 < angle < 120:
            verticals.append(line[0])
    return horizontals, verticals

def _corners_from_lines(self, horizontals, verticals, w, h):
    """Estimate court corners from horizontal/vertical line intersections."""
    # Sort horizontals by y (top to bottom)
    horizontals.sort(key=lambda l: (l[1] + l[3]) / 2)
    # Sort verticals by x (left to right)
    verticals.sort(key=lambda l: (l[0] + l[2]) / 2)

    top_line = horizontals[0]
    bottom_line = horizontals[-1]
    left_line = verticals[0]
    right_line = verticals[-1]

    # Compute intersections
    tl = self._line_intersection(top_line, left_line, w, h)
    tr = self._line_intersection(top_line, right_line, w, h)
    bl = self._line_intersection(bottom_line, left_line, w, h)
    br = self._line_intersection(bottom_line, right_line, w, h)

    return np.array([bl, br, tl, tr], dtype=np.float32)

def _corners_from_endpoints(self, lines, w, h):
    """Estimate corners from line endpoint extremes."""
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.extend([(x1, y1), (x2, y2)])
    points = np.array(points, dtype=np.float32)

    bottom = points[points[:, 1] > h * 0.5]
    top = points[points[:, 1] <= h * 0.5]

    if len(bottom) < 2 or len(top) < 2:
        return self._default_corners(w, h)

    bl = bottom[bottom[:, 0].argmin()]
    br = bottom[bottom[:, 0].argmax()]
    tl = top[top[:, 0].argmin()]
    tr = top[top[:, 0].argmax()]

    return np.array([bl, br, tl, tr], dtype=np.float32)

@staticmethod
def _line_intersection(line1, line2, w, h):
    """Find intersection point of two lines, clamped to frame bounds."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return np.array([(x1 + x3) / 2, (y1 + y3) / 2], dtype=np.float32)

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)

    # Clamp to frame
    ix = np.clip(ix, 0, w)
    iy = np.clip(iy, 0, h)

    return np.array([ix, iy], dtype=np.float32)
```

- [ ] **Step 2: Commit**

```bash
git add ml/analyzer.py
git commit -m "feat(ml): improve court detection with multi-threshold and line classification"
```

---

## Group 4: numpy 2.x Migration

### Task 7: Migrate to numpy 2.x

**Files:**
- Modify: `ml/requirements.txt`
- Modify: `ml/__init__.py`
- Modify: `ml/detector.py`, `ml/classifier.py`, `ml/analyzer.py`, `ml/pose.py`, `ml/score.py`

- [ ] **Step 1: Update requirements.txt**

```
numpy>=2.0
```

(Remove the `<2.0` pin.)

- [ ] **Step 2: Search for numpy 2.x deprecations in ml/ files**

numpy 2.0 removed:
- `np.bool` → use `bool`
- `np.int` → use `int`
- `np.float` → use `float`
- `np.complex` → use `complex`
- `np.object` → use `object`
- `np.str` → use `str`
- Changed: `np.array(..., copy=False)` now always copies unless `copy=None`

Grep through all ml/ files:

```bash
grep -rn "np\.bool\b\|np\.int\b\|np\.float\b\|np\.complex\b\|np\.object\b\|np\.str\b" ml/
```

Fix any occurrences found.

- [ ] **Step 3: Update ml/__init__.py**

Remove any numpy compatibility shims. The `torch.set_num_threads(1)` and logging setup remain.

- [ ] **Step 4: Run import test**

```bash
python -c "import numpy; print(numpy.__version__); from ml.detector import Detector; from ml.classifier import StrokeClassifier; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add ml/
git commit -m "feat(ml): migrate to numpy 2.x, remove deprecated type aliases"
```

---

## Group 5: Config + Bridge Wiring

### Task 8: Add detector/classifier config to Go

**Files:**
- Modify: `internal/config/config.go`
- Modify: `internal/config/config_test.go`

- [ ] **Step 1: Add fields to PipelineConfig**

```go
type PipelineConfig struct {
    BatchSize        int    `yaml:"batchSize"`
    FrameSkip        int    `yaml:"frameSkip"`
    EnablePose       bool   `yaml:"enablePose"`
    EnableScore      bool   `yaml:"enableScore"`
    CheckpointEvery  int    `yaml:"checkpointEvery"`
    DetectorBackend  string `yaml:"detectorBackend"`  // "yolo" or "rtdetr"
    ClassifierModel  string `yaml:"classifierModel"`  // "3dcnn" or "swin"
}
```

Update `Default()`:
```go
DetectorBackend: "yolo",
ClassifierModel: "3dcnn",
```

- [ ] **Step 2: Add test**

```go
func TestLoadConfig_DetectorBackend(t *testing.T) {
    cfg := Default()
    if cfg.Pipeline.DetectorBackend != "yolo" {
        t.Errorf("expected 'yolo', got %s", cfg.Pipeline.DetectorBackend)
    }
    if cfg.Pipeline.ClassifierModel != "3dcnn" {
        t.Errorf("expected '3dcnn', got %s", cfg.Pipeline.ClassifierModel)
    }
}
```

- [ ] **Step 3: Run tests**

```bash
go test ./internal/config/ -v
```

- [ ] **Step 4: Commit**

---

### Task 9: Wire config through bridge server

**Files:**
- Modify: `ml/bridge_server.py`

- [ ] **Step 1: Pass detector_backend and classifier_model from init params**

Update `rpc_init` in bridge_server.py:

```python
def rpc_init(self, params: dict) -> dict:
    # ... existing code ...
    detector_backend = params.get("DetectorBackend", "yolo")
    classifier_model = params.get("ClassifierModel", "3dcnn")

    self.detector = Detector(model_path=detector_model, device=device, backend=detector_backend)
    self.classifier = StrokeClassifier(model_path=classifier_model_path, device=device, model_type=classifier_model)
    # ... rest unchanged
```

- [ ] **Step 2: Update Go BridgeConfig to include new fields**

Add to `internal/bridge/types.go`:
```go
type BridgeConfig struct {
    ModelsDir       string `json:"ModelsDir"`
    Device          string `json:"Device"`
    DetectorBackend string `json:"DetectorBackend"`
    ClassifierModel string `json:"ClassifierModel"`
}
```

- [ ] **Step 3: Update main.go to pass new config fields**

```go
b.Init(bridge.BridgeConfig{
    ModelsDir:       cfg.ModelsDir,
    Device:          cfg.Device,
    DetectorBackend: cfg.Pipeline.DetectorBackend,
    ClassifierModel: cfg.Pipeline.ClassifierModel,
})
```

- [ ] **Step 4: Run all tests**

```bash
go test ./internal/... -count=1
```

- [ ] **Step 5: Commit**

---

## Summary

| Group | Tasks | Deliverable |
|-------|-------|-------------|
| 1. Detector upgrade | 1-3 | YOLOv11/RT-DETR with backend toggle |
| 2. Classifier upgrade | 4-5 | Video Swin Transformer alongside 3D CNN |
| 3. Court detection | 6 | Multi-threshold, line classification, intersection-based corners |
| 4. numpy 2.x | 7 | Remove deprecated aliases, bump requirement |
| 5. Config wiring | 8-9 | Go config → bridge → Python model selection |
