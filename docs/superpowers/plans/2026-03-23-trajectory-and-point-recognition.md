# Trajectory Prediction & Point Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 5-layer system that transforms sparse ball detections into full shot-level tennis analytics with trajectory prediction, in/out calls, point outcomes, and score tracking.

**Architecture:** TrackNet v2 provides dense ball detection (~85% of frames), merged with existing YOLO detections. A physics-based trajectory fitter predicts ball arcs and bounce points. Go-side logic segments shots, determines point outcomes, and tracks the score using tennis rules.

**Tech Stack:** Python (PyTorch, NumPy, OpenCV) for TrackNet + trajectory fitting. Go for shot segmentation, point recognition, score tracking, and pipeline integration.

**Spec:** `docs/superpowers/specs/2026-03-23-trajectory-and-point-recognition-design.md`

---

## File Map

### New Python Files

| File | Responsibility |
|------|---------------|
| `ml/tracknet.py` | TrackNet v2 model definition, background subtraction, inference wrapper, peak detection |
| `ml/trajectory.py` | Trajectory fitting (least-squares), bounce detection, in/out classification, height estimation |

### New Go Files

| File | Responsibility |
|------|---------------|
| `internal/point/rules.go` | Tennis court geometry constants (dimensions, service boxes, line positions) and in/out boundary checks |
| `internal/point/rules_test.go` | Tests for court boundary geometry |
| `internal/point/shot.go` | Shot struct, shot segmentation from trajectory data (direction changes, bounces) |
| `internal/point/shot_test.go` | Tests for shot segmentation logic |
| `internal/point/point.go` | Point recognition state machine — determines how each point ended |
| `internal/point/point_test.go` | Tests for point recognition |
| `internal/point/score.go` | MatchState, tennis scoring rules (points, games, sets, tiebreaks) |
| `internal/point/score_test.go` | Tests for scoring (deuce, tiebreak, set transitions, server rotation) |

### Modified Files

| File | Changes |
|------|---------|
| `ml/bridge_server.py` | Add `rpc_tracknet_batch` and `rpc_fit_trajectories` RPCs, init TrackNet in `rpc_init` |
| `internal/bridge/types.go` | Add `BallPosition`, `TrajectoryResult`, `BounceResult`, `ShotResult`, `PointResult`, `MatchScore` types |
| `internal/bridge/process.go` | Add `TrackNetBatch()` and `FitTrajectories()` bridge methods |
| `internal/pipeline/concurrent.go` | Add post-processing stages: trajectories → shots → points → score |
| `internal/app/app.go` | Update CSV export with shot/point/score columns |

---

## Task 1: Tennis Court Geometry (Go)

Pure geometry — no ML, no bridge. The foundation everything else builds on.

**Files:**
- Create: `internal/point/rules.go`
- Create: `internal/point/rules_test.go`

- [ ] **Step 1: Write failing tests for court boundary checks**

```go
// internal/point/rules_test.go
package point

import "testing"

func TestIsInSingles(t *testing.T) {
	tests := []struct {
		name string
		cx, cy float64
		want bool
	}{
		{"center court", 5.485, 11.885, true},
		{"near baseline inside", 4.0, 1.0, true},
		{"far baseline inside", 4.0, 22.0, true},
		{"outside singles left", -0.5, 11.0, false},
		{"outside singles right", 8.5, 11.0, false},
		{"behind near baseline", 4.0, -0.5, false},
		{"behind far baseline", 4.0, 24.0, false},
		{"on the line", 0.0, 0.0, true},           // lines are in
		{"just outside", -0.01, 11.0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsInSingles(tt.cx, tt.cy)
			if got != tt.want {
				t.Errorf("IsInSingles(%v, %v) = %v, want %v", tt.cx, tt.cy, got, tt.want)
			}
		})
	}
}

func TestIsInServiceBox(t *testing.T) {
	tests := []struct {
		name   string
		cx, cy float64
		side   string // "deuce" or "ad", "near" or "far"
		want   bool
	}{
		{"near deuce center", 6.0, 9.0, "near_deuce", true},
		{"near ad center", 2.0, 9.0, "near_ad", true},
		{"far deuce center", 6.0, 15.0, "far_deuce", true},
		{"too deep", 4.0, 4.0, "near_deuce", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsInServiceBox(tt.cx, tt.cy, tt.side)
			if got != tt.want {
				t.Errorf("IsInServiceBox(%v, %v, %q) = %v, want %v", tt.cx, tt.cy, tt.side, got, tt.want)
			}
		})
	}
}

func TestClassifyLanding(t *testing.T) {
	tests := []struct {
		name string
		cx, cy float64
		want string
	}{
		{"center court", 4.0, 11.0, "in"},
		{"way out", 15.0, 11.0, "out"},
		{"close to line", 8.22, 11.0, "close_call"},   // within 5cm of singles line at 8.23m
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ClassifyLanding(tt.cx, tt.cy)
			if got != tt.want {
				t.Errorf("ClassifyLanding(%v, %v) = %q, want %q", tt.cx, tt.cy, got, tt.want)
			}
		})
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && go test ./internal/point/ -v -run TestIsIn`
Expected: compilation error — package and functions don't exist yet.

- [ ] **Step 3: Implement court geometry**

```go
// internal/point/rules.go
package point

// Tennis court dimensions in meters (ITF standard).
const (
	CourtLength     = 23.77  // baseline to baseline
	SinglesWidth    = 8.23   // singles sideline to sideline
	DoublesWidth    = 10.97  // doubles sideline to sideline
	ServiceBoxDepth = 6.40   // net to service line
	NetY            = 11.885 // net position (halfway)
	CenterLineX     = 4.115 // center of singles court
	CloseCallMargin = 0.05  // 5cm margin for "close call"
)

// IsInSingles reports whether (cx, cy) in meters is inside the singles court.
func IsInSingles(cx, cy float64) bool {
	return cx >= 0 && cx <= SinglesWidth && cy >= 0 && cy <= CourtLength
}

// IsInDoubles reports whether (cx, cy) in meters is inside the doubles court.
func IsInDoubles(cx, cy float64) bool {
	return cx >= 0 && cx <= DoublesWidth && cy >= 0 && cy <= CourtLength
}

// IsInServiceBox checks whether (cx, cy) lands inside the specified service box.
// side is one of: "near_deuce", "near_ad", "far_deuce", "far_ad".
func IsInServiceBox(cx, cy float64, side string) bool {
	switch side {
	case "near_deuce":
		return cx >= CenterLineX && cx <= SinglesWidth &&
			cy >= (NetY-ServiceBoxDepth) && cy <= NetY
	case "near_ad":
		return cx >= 0 && cx <= CenterLineX &&
			cy >= (NetY-ServiceBoxDepth) && cy <= NetY
	case "far_deuce":
		return cx >= 0 && cx <= CenterLineX &&
			cy >= NetY && cy <= (NetY+ServiceBoxDepth)
	case "far_ad":
		return cx >= CenterLineX && cx <= SinglesWidth &&
			cy >= NetY && cy <= (NetY+ServiceBoxDepth)
	}
	return false
}

// ClassifyLanding returns "in", "out", or "close_call" for a bounce at (cx, cy).
func ClassifyLanding(cx, cy float64) string {
	if IsInSingles(cx, cy) {
		return "in"
	}
	// Check if within close-call margin of any boundary
	if cx >= -CloseCallMargin && cx <= SinglesWidth+CloseCallMargin &&
		cy >= -CloseCallMargin && cy <= CourtLength+CloseCallMargin {
		return "close_call"
	}
	return "out"
}
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && go test ./internal/point/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add internal/point/rules.go internal/point/rules_test.go
git commit -m "feat(point): add tennis court geometry and boundary checks"
```

---

## Task 2: Score Tracking State Machine (Go)

Pure logic — tennis scoring rules. No dependencies on ML or detection.

**Files:**
- Create: `internal/point/score.go`
- Create: `internal/point/score_test.go`

- [ ] **Step 1: Write failing tests for score state machine**

```go
// internal/point/score_test.go
package point

import "testing"

func TestScoreBasicGame(t *testing.T) {
	m := NewMatchState(0) // player 0 serves first
	// Player 0 wins 4 points in a row = holds serve
	for i := 0; i < 4; i++ {
		m.AwardPoint(0)
	}
	if m.GameScore[0] != 1 || m.GameScore[1] != 0 {
		t.Errorf("game score = %v, want [1 0]", m.GameScore)
	}
	if m.Server != 1 { // server switches after game
		t.Errorf("server = %d, want 1", m.Server)
	}
}

func TestScoreDeuce(t *testing.T) {
	m := NewMatchState(0)
	// Go to 40-40
	for i := 0; i < 3; i++ { m.AwardPoint(0) }
	for i := 0; i < 3; i++ { m.AwardPoint(1) }
	if !m.IsDeuce {
		t.Error("should be deuce at 40-40")
	}
	// Player 0 gets advantage
	m.AwardPoint(0)
	if m.PointScore[0] != 41 { // 41 = advantage
		t.Errorf("point score = %v, want [41 40]", m.PointScore)
	}
	// Player 1 wins next point -> back to deuce
	m.AwardPoint(1)
	if m.PointScore[0] != 40 || m.PointScore[1] != 40 {
		t.Errorf("should be back to deuce, got %v", m.PointScore)
	}
	// Player 0 gets ad + wins
	m.AwardPoint(0)
	m.AwardPoint(0)
	if m.GameScore[0] != 1 {
		t.Errorf("game score = %v, want [1 0]", m.GameScore)
	}
}

func TestScoreTiebreak(t *testing.T) {
	m := NewMatchState(0)
	// Get to 6-6
	for g := 0; g < 6; g++ {
		for i := 0; i < 4; i++ { m.AwardPoint(0) }
	}
	for g := 0; g < 6; g++ {
		for i := 0; i < 4; i++ { m.AwardPoint(1) }
	}
	if !m.IsTiebreak {
		t.Error("should be tiebreak at 6-6")
	}
	// Player 0 wins tiebreak 7-0
	for i := 0; i < 7; i++ {
		m.AwardPoint(0)
	}
	if m.SetScore[0] != 1 {
		t.Errorf("set score = %v, want [1 0]", m.SetScore)
	}
}

func TestServeSideAlternates(t *testing.T) {
	m := NewMatchState(0)
	if m.ServeSide != "deuce" {
		t.Errorf("first point should be deuce, got %q", m.ServeSide)
	}
	m.AwardPoint(0) // 15-0
	if m.ServeSide != "ad" {
		t.Errorf("second point should be ad, got %q", m.ServeSide)
	}
	m.AwardPoint(0) // 30-0
	if m.ServeSide != "deuce" {
		t.Errorf("third point should be deuce, got %q", m.ServeSide)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && go test ./internal/point/ -v -run TestScore`
Expected: compilation error — `NewMatchState`, `AwardPoint` don't exist.

- [ ] **Step 3: Implement score tracking**

```go
// internal/point/score.go
package point

// MatchState tracks the full state of a tennis match.
type MatchState struct {
	Server      int      // 0 or 1
	ServeSide   string   // "deuce" or "ad"
	PointScore  [2]int   // 0, 15, 30, 40 (41 = advantage)
	GameScore   [2]int   // games in current set
	SetScore    [2]int   // sets won
	Sets        [][2]int // completed set scores
	IsDeuce     bool
	IsTiebreak  bool
	PointNumber int
}

// NewMatchState creates a new match with the given initial server (0 or 1).
func NewMatchState(firstServer int) *MatchState {
	return &MatchState{
		Server:    firstServer,
		ServeSide: "deuce",
	}
}

// AwardPoint awards a point to the given player (0 or 1) and updates all state.
func (m *MatchState) AwardPoint(winner int) {
	m.PointNumber++

	if m.IsTiebreak {
		m.awardTiebreakPoint(winner)
	} else {
		m.awardGamePoint(winner)
	}

	// Alternate serve side
	if m.ServeSide == "deuce" {
		m.ServeSide = "ad"
	} else {
		m.ServeSide = "deuce"
	}
}

func (m *MatchState) awardGamePoint(winner int) {
	loser := 1 - winner

	switch m.PointScore[winner] {
	case 0:
		m.PointScore[winner] = 15
	case 15:
		m.PointScore[winner] = 30
	case 30:
		m.PointScore[winner] = 40
	case 40:
		if m.PointScore[loser] < 40 {
			m.winGame(winner)
		} else if m.PointScore[loser] == 40 {
			// Deuce -> advantage
			m.PointScore[winner] = 41
			m.IsDeuce = false
		} else {
			// Loser had advantage -> back to deuce
			// This shouldn't happen; if loser is at 41, winner is at 40
		}
	case 41:
		// Winner had advantage -> wins game
		m.winGame(winner)
	}

	// If winner was at 40 and loser had advantage (41)
	if m.PointScore[loser] == 41 {
		m.PointScore[0] = 40
		m.PointScore[1] = 40
		m.IsDeuce = true
	}

	// Check deuce state
	if m.PointScore[0] == 40 && m.PointScore[1] == 40 {
		m.IsDeuce = true
	}
}

func (m *MatchState) awardTiebreakPoint(winner int) {
	m.PointScore[winner]++
	p0, p1 := m.PointScore[0], m.PointScore[1]

	if p0 >= 7 && p0-p1 >= 2 {
		m.winSet(0)
	} else if p1 >= 7 && p1-p0 >= 2 {
		m.winSet(1)
	}

	// Server alternates every 2 points in tiebreak (after the first point)
	totalTBPoints := p0 + p1
	if totalTBPoints == 1 || (totalTBPoints > 1 && (totalTBPoints-1)%2 == 0) {
		m.Server = 1 - m.Server
	}
}

func (m *MatchState) winGame(winner int) {
	m.GameScore[winner]++
	m.PointScore = [2]int{0, 0}
	m.IsDeuce = false
	m.ServeSide = "deuce"

	g0, g1 := m.GameScore[0], m.GameScore[1]

	// Check for set win
	if g0 >= 6 && g0-g1 >= 2 {
		m.winSet(0)
	} else if g1 >= 6 && g1-g0 >= 2 {
		m.winSet(1)
	} else if g0 == 6 && g1 == 6 {
		m.IsTiebreak = true
	}

	// Rotate server (only if not entering tiebreak — tiebreak has its own rotation)
	if !m.IsTiebreak {
		m.Server = 1 - m.Server
	}
}

func (m *MatchState) winSet(winner int) {
	m.Sets = append(m.Sets, m.GameScore)
	m.SetScore[winner]++
	m.GameScore = [2]int{0, 0}
	m.PointScore = [2]int{0, 0}
	m.IsTiebreak = false
	m.IsDeuce = false
	m.ServeSide = "deuce"
	// Server continues rotating from the pre-set state
	// (the last game of the set already set the correct server)
}
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && go test ./internal/point/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add internal/point/score.go internal/point/score_test.go
git commit -m "feat(point): add tennis scoring state machine with deuce and tiebreak"
```

---

## Task 3: TrackNet v2 Model (Python)

The core ML model for dense ball detection.

**Files:**
- Create: `ml/tracknet.py`

- [ ] **Step 1: Implement TrackNet v2 model architecture**

The model is a U-Net style encoder-decoder. Input: 9 channels (3 BGR diff frames). Output: 1 channel heatmap. Include background subtraction helper and peak detection.

Reference architecture: TrackNet v2 paper (encoder: VGG-style conv blocks with max pooling, decoder: upsample + conv blocks with skip connections from encoder).

```python
# ml/tracknet.py
"""
TrackNet v2 — Dense ball detection via heatmap regression.

Encoder-decoder (U-Net style) that takes 3 background-subtracted frames
(every-other-frame spacing) and outputs a probability heatmap of the ball position.
"""

import logging
from typing import Optional

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

logger = logging.getLogger(__name__)


def _conv_block(in_ch: int, out_ch: int, num_convs: int = 2) -> nn.Sequential:
    """VGG-style convolution block: N x (Conv3x3 + BN + ReLU)."""
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class TrackNetV2(nn.Module):
    """TrackNet v2 encoder-decoder for ball detection."""

    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.enc1 = _conv_block(9, 64, 2)    # 3 diff frames x 3 channels = 9
        self.enc2 = _conv_block(64, 128, 2)
        self.enc3 = _conv_block(128, 256, 3)
        self.enc4 = _conv_block(256, 512, 3)  # bottleneck

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _conv_block(512, 256, 3)  # 256 skip + 256 up = 512 in
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _conv_block(256, 128, 2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _conv_block(128, 64, 2)

        self.out_conv = nn.Conv2d(64, 1, 1)   # 1-channel heatmap

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Encoder
        e1 = self.enc1(x)              # -> (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # -> (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # -> (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # -> (B, 512, H/8, W/8)

        # Decoder with skip connections
        d3 = self.up3(e4)
        d3 = self._pad_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out_conv(d1))  # -> (B, 1, H, W)

    @staticmethod
    def _pad_and_cat(upsampled: "torch.Tensor", skip: "torch.Tensor") -> "torch.Tensor":
        """Handle size mismatch between upsampled and skip connection."""
        dh = skip.shape[2] - upsampled.shape[2]
        dw = skip.shape[3] - upsampled.shape[3]
        upsampled = F.pad(upsampled, [0, dw, 0, dh])
        return torch.cat([upsampled, skip], dim=1)


class BackgroundSubtractor:
    """Builds a static background reference and produces diff frames."""

    def __init__(self) -> None:
        self.reference: Optional[np.ndarray] = None

    def build_reference(self, frames: list[np.ndarray]) -> None:
        """Build background reference from a list of BGR frames (median)."""
        stacked = np.stack(frames, axis=0)
        self.reference = np.median(stacked, axis=0).astype(np.uint8)
        logger.info("Background reference built from %d frames", len(frames))

    def subtract(self, frame: np.ndarray) -> np.ndarray:
        """Return |frame - reference|. Falls back to raw frame if no reference."""
        if self.reference is None:
            return frame
        return cv2.absdiff(frame, self.reference)


class TrackNetDetector:
    """Inference wrapper for TrackNet v2."""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.model: Optional[TrackNetV2] = None
        self.device = "cpu"
        self.bg = BackgroundSubtractor()
        self._frame_buffer: list[np.ndarray] = []  # last 5 raw frames

        if not _TORCH_OK:
            logger.warning("torch not installed — TrackNet disabled")
            return

        # Resolve device
        if device == "auto":
            import torch as _torch
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        elif device != "cpu":
            self.device = device

        self.model = TrackNetV2()

        # Load weights if provided
        if model_path is not None:
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                logger.info("TrackNet weights loaded from %s", model_path)
            except Exception:
                logger.warning("Failed to load TrackNet weights, using random init", exc_info=True)

        self.model.to(self.device)
        self.model.eval()

        # Warmup
        dummy = torch.zeros(1, 9, 360, 640, device=self.device)
        with torch.no_grad():
            self.model(dummy)
        logger.info("TrackNet ready on %s", self.device)

    def feed_frame(self, frame: np.ndarray) -> Optional[dict]:
        """Feed a single BGR frame. Returns ball detection when enough frames buffered.

        Uses every-other-frame spacing: frames [N-4, N-2, N].
        Returns a detection for frame N, or None if not enough frames yet.
        """
        self._frame_buffer.append(frame)

        # Need at least 5 frames to get [N-4, N-2, N]
        if len(self._frame_buffer) < 5:
            return None

        # Keep only the last 5 frames
        if len(self._frame_buffer) > 5:
            self._frame_buffer = self._frame_buffer[-5:]

        # Select every-other-frame: indices 0, 2, 4 from buffer
        f0 = self._frame_buffer[0]
        f1 = self._frame_buffer[2]
        f2 = self._frame_buffer[4]

        # Background subtract
        d0 = self.bg.subtract(f0)
        d1 = self.bg.subtract(f1)
        d2 = self.bg.subtract(f2)

        # Run inference
        return self._detect(d0, d1, d2)

    def detect_batch(self, diff_triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[Optional[dict]]:
        """Batch detection on pre-built diff triplets.

        Args:
            diff_triplets: List of (diff_N-4, diff_N-2, diff_N) tuples.

        Returns:
            List of {x, y, confidence} dicts or None for each triplet.
        """
        if self.model is None or len(diff_triplets) == 0:
            return [None] * len(diff_triplets)

        # Stack into batch tensor
        batch = []
        for d0, d1, d2 in diff_triplets:
            stacked = np.concatenate([d0, d1, d2], axis=2)  # (H, W, 9)
            tensor = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0
            batch.append(tensor)

        batch_tensor = torch.stack(batch).to(self.device)

        with torch.no_grad():
            heatmaps = self.model(batch_tensor)  # (B, 1, H, W)

        results = []
        for i in range(len(diff_triplets)):
            hm = heatmaps[i, 0].cpu().numpy()
            results.append(self._peak_detect(hm))

        return results

    def _detect(self, d0: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> Optional[dict]:
        """Run inference on a single triplet of diff frames."""
        result = self.detect_batch([(d0, d1, d2)])
        return result[0]

    @staticmethod
    def _peak_detect(heatmap: np.ndarray, threshold: float = 0.5) -> Optional[dict]:
        """Extract ball position from heatmap via weighted centroid."""
        mask = (heatmap > threshold).astype(np.uint8)
        if mask.sum() == 0:
            return None

        # Find connected components, keep the brightest
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return None

        # Skip label 0 (background), find component with highest mean heatmap value
        best_label = -1
        best_val = 0.0
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            mean_val = heatmap[component_mask].mean()
            if mean_val > best_val:
                best_val = mean_val
                best_label = label_id

        if best_label < 0:
            return None

        # Weighted centroid of the best component for sub-pixel accuracy
        component_mask = (labels == best_label).astype(np.float32)
        weighted = heatmap * component_mask
        total = weighted.sum()
        if total < 1e-6:
            return None

        ys, xs = np.mgrid[:heatmap.shape[0], :heatmap.shape[1]]
        cx = (xs * weighted).sum() / total
        cy = (ys * weighted).sum() / total
        confidence = float(best_val)

        return {"x": float(cx), "y": float(cy), "confidence": confidence}
```

- [ ] **Step 2: Verify model instantiates and runs**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -c "from ml.tracknet import TrackNetDetector; d = TrackNetDetector(); print('TrackNet OK, params:', sum(p.numel() for p in d.model.parameters()))"`

Expected: `TrackNet OK, params: ~2500000` (approximately)

- [ ] **Step 3: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/tracknet.py
git commit -m "feat(ml): add TrackNet v2 model with background subtraction and peak detection"
```

---

## Task 4: TrackNet Bridge Integration (Python + Go)

Wire TrackNet into the bridge so Go can call it.

**Files:**
- Modify: `ml/bridge_server.py`
- Modify: `internal/bridge/types.go`
- Modify: `internal/bridge/process.go`

- [ ] **Step 1: Add BallPosition type to Go bridge types**

```go
// Add to internal/bridge/types.go after existing types

// BallPosition holds a ball detection with sub-pixel position and source.
type BallPosition struct {
	X          float64 `json:"x"`
	Y          float64 `json:"y"`
	Confidence float64 `json:"confidence"`
	FrameIndex int     `json:"frameIndex"`
	Source     string  `json:"source"` // "tracknet", "yolo", "merged"
}
```

- [ ] **Step 2: Add TrackNet RPC to bridge_server.py**

Add `rpc_tracknet_batch` method to `BridgeServer` class in `ml/bridge_server.py`. This takes a batch of frames, builds diff triplets (every-other-frame), runs TrackNet, and returns ball positions.

Also update `rpc_init` to instantiate `TrackNetDetector` alongside the existing `Detector`.

Add `"tracknet_batch": self.rpc_tracknet_batch` to the RPC dispatch table.

```python
# In rpc_init, after existing detector init:
from ml.tracknet import TrackNetDetector
tracknet_model = os.path.join(models_dir, "tracknet_v2.pt")
self.tracknet = TrackNetDetector(
    model_path=tracknet_model if os.path.exists(tracknet_model) else None,
    device=device,
)

# New RPC method:
def rpc_tracknet_batch(self, params: dict) -> Any:
    """Run TrackNet on a batch of frames with background subtraction."""
    self._require_init()

    shm_path = params.get("shm_path")
    frames_meta = params.get("frames", [])
    if not frames_meta:
        return []

    # Read frames from shared memory
    frames = []
    with open(shm_path, "rb") as f:
        for meta in frames_meta:
            f.seek(meta["offset"])
            raw = f.read(meta["size"])
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                (meta["height"], meta["width"], 3)
            )
            frames.append(arr)

    # Build background reference from first batch if not done
    if self.tracknet.bg.reference is None and len(frames) >= 10:
        sample_indices = list(range(0, len(frames), max(1, len(frames) // 30)))
        self.tracknet.bg.build_reference([frames[i] for i in sample_indices[:30]])

    # Build every-other-frame diff triplets
    results = []
    for i in range(4, len(frames), 2):
        d0 = self.tracknet.bg.subtract(frames[i - 4])
        d1 = self.tracknet.bg.subtract(frames[i - 2])
        d2 = self.tracknet.bg.subtract(frames[i])
        det = self.tracknet._detect(d0, d1, d2)
        results.append({
            "frame_index": i,
            "ball": det,
        })

    return results
```

- [ ] **Step 3: Add Go bridge method**

Add `TrackNetBatch` method to `ProcessBridge` in `internal/bridge/process.go`, following the same shared-memory pattern as `DetectBatch`. Returns `[]BallPosition`.

- [ ] **Step 4: Build and verify compilation**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && go build ./...`
Expected: clean build.

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/bridge_server.py internal/bridge/types.go internal/bridge/process.go
git commit -m "feat(bridge): wire TrackNet into Go-Python bridge"
```

---

## Task 5: Trajectory Fitting (Python)

Physics-based trajectory fitter with bounce detection and in/out calls.

**Files:**
- Create: `ml/trajectory.py`

- [ ] **Step 1: Implement trajectory fitter**

The module takes a sequence of ball detections + court homography and produces fitted trajectories with bounce points. Key classes:

- `TrajectoryFitter`: takes ball positions, fits arcs using least-squares
- `fit_trajectory(positions, fps, homography)`: fits cx(t), cy(t), h(t)
- `find_bounces(trajectory)`: finds where h(t) = 0
- `classify_landing(cx, cy)`: in/out determination using court geometry constants

The fitter uses `numpy.linalg.lstsq` to solve for trajectory parameters. Height is estimated from pixel-y vs expected court surface y using the homography.

Court dimensions are defined as constants matching the Go `rules.go` (23.77m x 8.23m singles, net at 11.885m, service box 6.40m deep).

The output is a list of trajectory segments, each with start/end frame, fitted parameters, bounce point (if any), and in/out classification.

- [ ] **Step 2: Add trajectory RPC to bridge_server.py**

Add `rpc_fit_trajectories` to `BridgeServer`. Takes ball positions + court data, returns trajectory results. Add to RPC dispatch table.

- [ ] **Step 3: Test with synthetic data**

Run a quick sanity check: create 10 ball positions along a known parabolic path, fit, verify bounce point matches expected.

```python
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && python -c "
from ml.trajectory import TrajectoryFitter
import numpy as np

# Simulate a ball at (4, 2) moving forward at 20m/s, height 1.5m, g=9.8
fitter = TrajectoryFitter()
positions = []
fps = 30
for i in range(15):
    t = i / fps
    cx = 4.0
    cy = 2.0 + 20.0 * t
    h = 1.5 + 5.0 * t - 0.5 * 9.8 * t * t
    if h < 0:
        break
    positions.append({'cx': cx, 'cy': cy, 'h': h, 't': t, 'confidence': 0.9})

traj = fitter.fit(positions)
print(f'Bounce at cy={traj.bounce_cy:.1f}m (expected ~12.2m)')
print(f'g_eff={traj.g_eff:.1f} (expected ~9.8)')
"
```

- [ ] **Step 4: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add ml/trajectory.py ml/bridge_server.py
git commit -m "feat(ml): add physics-based trajectory fitter with bounce detection"
```

---

## Task 6: Shot Segmentation (Go)

Breaks continuous ball trajectory data into individual shots.

**Files:**
- Create: `internal/point/shot.go`
- Create: `internal/point/shot_test.go`
- Modify: `internal/bridge/types.go` (add trajectory/shot result types)

- [ ] **Step 1: Add trajectory types to bridge**

Add to `internal/bridge/types.go`:

```go
// TrajectoryResult holds a fitted ball trajectory segment.
type TrajectoryResult struct {
	StartFrame int     `json:"startFrame"`
	EndFrame   int     `json:"endFrame"`
	BounceCX   float64 `json:"bounceCx"` // court x in meters
	BounceCY   float64 `json:"bounceCy"` // court y in meters
	BounceH    float64 `json:"bounceH"`  // should be ~0
	SpeedKPH   float64 `json:"speedKph"`
	GEff       float64 `json:"gEff"`
	InOut      string  `json:"inOut"` // "in", "out", "close_call"
	Confidence float64 `json:"confidence"`
}
```

- [ ] **Step 2: Write failing shot segmentation tests**

Test that direction changes and bounces correctly split trajectories into shots, with correct player assignment based on ball direction.

- [ ] **Step 3: Implement shot segmentation**

```go
// internal/point/shot.go
package point

import "github.com/liamp/tennis-tagger/internal/bridge"

// Shot represents a single hit in a rally.
type Shot struct {
	Index      int                    // 1-based shot index in the rally
	Hitter     int                    // 0 = near player, 1 = far player
	Trajectory bridge.TrajectoryResult
	IsServe    bool
}

// SegmentShots splits a sequence of trajectory results into individual shots.
// Direction changes (ball reverses cy direction) indicate a new shot.
// Near player (0) hits when ball starts moving away (cy increasing).
// Far player (1) hits when ball starts moving toward camera (cy decreasing).
func SegmentShots(trajectories []bridge.TrajectoryResult) []Shot {
    // Implementation: iterate trajectories, detect cy direction changes,
    // assign hitter based on which side of the net the direction changed.
    // ...
}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && go test ./internal/point/ -v -run TestSegment`

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add internal/point/shot.go internal/point/shot_test.go internal/bridge/types.go
git commit -m "feat(point): add shot segmentation from trajectory data"
```

---

## Task 7: Point Recognition (Go)

Determines how each point ended from the shot sequence.

**Files:**
- Create: `internal/point/point.go`
- Create: `internal/point/point_test.go`

- [ ] **Step 1: Write failing point recognition tests**

Test each point-end condition: ball out, double bounce, ace, double fault, unreturned winner, ball into net.

- [ ] **Step 2: Implement point recognition state machine**

```go
// internal/point/point.go
package point

// PointOutcome describes how a point ended.
type PointOutcome struct {
	Winner     int    // 0 or 1
	Category   string // "winner", "unforced_error", "ace", "double_fault", "error"
	Confidence float64
}

// Point holds all data for a single point in the match.
type Point struct {
	Number     int
	Shots      []Shot
	Outcome    PointOutcome
	Server     int
	ServeSide  string
}

// RecognizePoint analyzes a sequence of shots and determines the point outcome.
func RecognizePoint(shots []Shot, server int, serveSide string) Point {
    // Check conditions in order:
    // 1. Last shot bounce is out -> error by hitter
    // 2. Two bounces on same side -> winner for hitter
    // 3. Serve in box + no return -> ace
    // 4. Two serves out of box -> double fault
    // 5. Bounce in + no return + past baseline -> winner
    // 6. Trajectory doesn't cross net -> error
    // ...
}
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && go test ./internal/point/ -v -run TestRecognize`

- [ ] **Step 4: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add internal/point/point.go internal/point/point_test.go
git commit -m "feat(point): add point recognition with 6 end conditions"
```

---

## Task 8: Pipeline Integration

Wire everything into the concurrent processing pipeline.

**Files:**
- Modify: `internal/pipeline/concurrent.go`
- Modify: `internal/bridge/process.go`
- Modify: `internal/app/app.go`

- [ ] **Step 1: Add TrackNet stage to pipeline**

In `concurrent.go`, after YOLO detection (stage 2), add a TrackNet stage that runs on the same frame batches. Merge TrackNet ball positions with YOLO ball detections.

- [ ] **Step 2: Add post-processing stages**

After existing rally segmentation (step 4) and placement analysis (step 5), add:

```go
// 6. Fit trajectories (Python)
trajectories, err := p.bridge.FitTrajectories(mergedBallPositions, result.Court)

// 7. Segment shots (Go - pure logic)
shots := point.SegmentShots(trajectories)

// 8. Recognize points (Go - pure logic)
points := point.RecognizePoints(shots)

// 9. Track score (Go - pure logic)
score := point.TrackScore(points, initialServer)
```

- [ ] **Step 3: Update CSV export**

Update `internal/app/app.go` to include shot-level data (hitter, bounce position, in/out, speed), point outcome (winner/error), and running score in the Dartfish CSV output.

- [ ] **Step 4: Build and test end-to-end**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && go build -o tagger.exe ./cmd/tagger/ && time ./tagger.exe testdata/tennis_mid2_60s.mp4`

Verify: CSV output contains trajectory, shot, point, and score data.

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add internal/pipeline/concurrent.go internal/bridge/process.go internal/app/app.go
git commit -m "feat(pipeline): integrate TrackNet + trajectory + point recognition into pipeline"
```

---

## Task 9: End-to-End Validation

Run on real footage, verify results make sense.

- [ ] **Step 1: Process the 60-second test clip**

Run: `cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey" && ./tagger.exe testdata/tennis_mid2_60s.mp4`

- [ ] **Step 2: Analyze output**

Check the CSV for:
- Ball detected in significantly more frames than before (target: 50%+ vs previous 1-2%)
- Trajectories fitted for rally segments
- Bounce points with in/out calls
- Shots attributed to near/far player
- Points with outcomes (winner/error)
- Running score that follows tennis rules

- [ ] **Step 3: Compare ball detection rates**

```bash
python -c "
import csv
rows = list(csv.DictReader(open('testdata/tennis_mid2_60s.mp4_output.csv')))
# Count frames with ball detections from different sources
# Compare TrackNet vs YOLO vs merged detection rates
"
```

- [ ] **Step 4: Commit any fixes**

```bash
cd "C:/Users/liamp/Downloads/Tennis Tagger Monkey"
git add -A
git commit -m "fix: end-to-end validation fixes for trajectory and point recognition"
```
