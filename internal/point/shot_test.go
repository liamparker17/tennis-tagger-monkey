package point

import (
	"testing"

	"github.com/liamp/tennis-tagger/internal/bridge"
)

// helper to build a TrajectoryResult with a single bounce.
func makeTraj(startFrame, endFrame int, bounces []bridge.Bounce, speedKPH, confidence float64) bridge.TrajectoryResult {
	return bridge.TrajectoryResult{
		StartFrame: startFrame,
		EndFrame:   endFrame,
		Bounces:    bounces,
		SpeedKPH:   speedKPH,
		Confidence: confidence,
	}
}

func bounce(frameIndex int, cx, cy float64) bridge.Bounce {
	return bridge.Bounce{FrameIndex: frameIndex, CX: cx, CY: cy, InOut: "in"}
}

// TestSegmentShotsEmpty: no trajectories → no shots.
func TestSegmentShotsEmpty(t *testing.T) {
	shots := SegmentShots(nil)
	if len(shots) != 0 {
		t.Fatalf("expected 0 shots, got %d", len(shots))
	}

	shots = SegmentShots([]bridge.TrajectoryResult{})
	if len(shots) != 0 {
		t.Fatalf("expected 0 shots for empty slice, got %d", len(shots))
	}
}

// TestSegmentShotsSingleBounce: one bounce → one shot that is a serve.
func TestSegmentShotsSingleBounce(t *testing.T) {
	trajs := []bridge.TrajectoryResult{
		makeTraj(0, 30, []bridge.Bounce{bounce(25, 2.0, 5.0)}, 120.0, 0.9),
	}
	shots := SegmentShots(trajs)
	if len(shots) != 1 {
		t.Fatalf("expected 1 shot, got %d", len(shots))
	}
	if !shots[0].IsServe {
		t.Error("expected first shot to be a serve")
	}
	if shots[0].Index != 1 {
		t.Errorf("expected Index=1, got %d", shots[0].Index)
	}
}

// TestSegmentShotsRally: 4 bounces alternating sides of net → 4 shots with alternating hitters.
func TestSegmentShotsRally(t *testing.T) {
	// NetY = 11.885
	// cy < NetY  → ball landed on near side  → far player (1) hit it
	// cy >= NetY → ball landed on far side   → near player (0) hit it
	trajs := []bridge.TrajectoryResult{
		makeTraj(0, 120, []bridge.Bounce{
			bounce(20, 2.0, 5.0),  // near side (cy=5 < 11.885), hitter=1
			bounce(45, 2.0, 18.0), // far side  (cy=18 >= 11.885), hitter=0
			bounce(70, 2.0, 4.0),  // near side, hitter=1
			bounce(95, 2.0, 20.0), // far side, hitter=0
		}, 100.0, 0.85),
	}

	shots := SegmentShots(trajs)
	if len(shots) != 4 {
		t.Fatalf("expected 4 shots, got %d", len(shots))
	}

	expectedHitters := []int{1, 0, 1, 0}
	for i, s := range shots {
		if s.Hitter != expectedHitters[i] {
			t.Errorf("shot %d: expected hitter=%d, got %d", i+1, expectedHitters[i], s.Hitter)
		}
		if s.Index != i+1 {
			t.Errorf("shot %d: expected Index=%d, got %d", i+1, i+1, s.Index)
		}
	}
}

// TestSegmentShotsHitterAssignment: verify hitter logic explicitly.
// Near player (0) hits when ball bounces on far side (cy >= NetY).
// Far player (1) hits when ball bounces on near side (cy < NetY).
func TestSegmentShotsHitterAssignment(t *testing.T) {
	tests := []struct {
		name           string
		cy             float64
		expectedHitter int
	}{
		{"near side bounce → far player hit it", 3.0, 1},
		{"far side bounce → near player hit it", 20.0, 0},
		{"exactly at net boundary → near player", NetY, 0},
		{"just below net → far player", NetY - 0.001, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trajs := []bridge.TrajectoryResult{
				makeTraj(0, 30, []bridge.Bounce{bounce(20, 2.0, tt.cy)}, 100.0, 0.9),
			}
			shots := SegmentShots(trajs)
			if len(shots) != 1 {
				t.Fatalf("expected 1 shot, got %d", len(shots))
			}
			if shots[0].Hitter != tt.expectedHitter {
				t.Errorf("cy=%.3f: expected hitter=%d, got %d", tt.cy, tt.expectedHitter, shots[0].Hitter)
			}
		})
	}
}

// TestSegmentShotsServeFlag: first shot is always a serve regardless of bounce side.
func TestSegmentShotsServeFlag(t *testing.T) {
	tests := []struct {
		name   string
		bouncy []bridge.Bounce
	}{
		{
			"serve lands near side",
			[]bridge.Bounce{bounce(10, 2.0, 5.0), bounce(30, 2.0, 18.0)},
		},
		{
			"serve lands far side",
			[]bridge.Bounce{bounce(10, 2.0, 18.0), bounce(30, 2.0, 4.0)},
		},
		{
			"single serve no return",
			[]bridge.Bounce{bounce(10, 2.0, 14.0)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trajs := []bridge.TrajectoryResult{
				makeTraj(0, 60, tt.bouncy, 180.0, 0.95),
			}
			shots := SegmentShots(trajs)
			if len(shots) == 0 {
				t.Fatal("expected at least 1 shot")
			}
			if !shots[0].IsServe {
				t.Errorf("expected shots[0].IsServe=true, got false")
			}
			for i := 1; i < len(shots); i++ {
				if shots[i].IsServe {
					t.Errorf("expected shots[%d].IsServe=false, got true", i)
				}
			}
		})
	}
}

// TestSegmentShotsSpeedAndConfidence: speed and confidence propagate from trajectory.
func TestSegmentShotsSpeedAndConfidence(t *testing.T) {
	trajs := []bridge.TrajectoryResult{
		makeTraj(0, 50, []bridge.Bounce{bounce(20, 2.0, 5.0), bounce(40, 2.0, 18.0)}, 145.5, 0.77),
	}
	shots := SegmentShots(trajs)
	for i, s := range shots {
		if s.SpeedKPH != 145.5 {
			t.Errorf("shot %d: expected SpeedKPH=145.5, got %f", i, s.SpeedKPH)
		}
		if s.Confidence != 0.77 {
			t.Errorf("shot %d: expected Confidence=0.77, got %f", i, s.Confidence)
		}
	}
}

// TestSegmentShotsBouncePointer: each shot's Bounce pointer references the correct bounce.
func TestSegmentShotsBouncePointer(t *testing.T) {
	b1 := bounce(20, 1.0, 5.0)
	b2 := bounce(45, 2.0, 18.0)
	trajs := []bridge.TrajectoryResult{
		makeTraj(0, 60, []bridge.Bounce{b1, b2}, 100.0, 0.9),
	}
	shots := SegmentShots(trajs)
	if len(shots) != 2 {
		t.Fatalf("expected 2 shots, got %d", len(shots))
	}
	if shots[0].Bounce == nil || shots[0].Bounce.FrameIndex != b1.FrameIndex {
		t.Errorf("shot 0: wrong bounce pointer, got %+v", shots[0].Bounce)
	}
	if shots[1].Bounce == nil || shots[1].Bounce.FrameIndex != b2.FrameIndex {
		t.Errorf("shot 1: wrong bounce pointer, got %+v", shots[1].Bounce)
	}
}

// TestSegmentShotsMultiTrajectory: bounces across multiple trajectories are merged and sorted.
func TestSegmentShotsMultiTrajectory(t *testing.T) {
	// Two trajectories contributing two bounces each; second traj has earlier frames
	// to ensure sorting by frame index, not traj order.
	trajs := []bridge.TrajectoryResult{
		makeTraj(60, 120, []bridge.Bounce{
			bounce(70, 2.0, 18.0), // far side
			bounce(95, 2.0, 5.0),  // near side
		}, 110.0, 0.8),
		makeTraj(0, 60, []bridge.Bounce{
			bounce(15, 2.0, 5.0),  // near side
			bounce(40, 2.0, 18.0), // far side
		}, 130.0, 0.9),
	}

	shots := SegmentShots(trajs)
	if len(shots) != 4 {
		t.Fatalf("expected 4 shots, got %d", len(shots))
	}

	// After sorting by frame: 15, 40, 70, 95
	expectedFrames := []int{15, 40, 70, 95}
	for i, s := range shots {
		if s.Bounce == nil {
			t.Errorf("shot %d: expected non-nil Bounce", i)
			continue
		}
		if s.Bounce.FrameIndex != expectedFrames[i] {
			t.Errorf("shot %d: expected FrameIndex=%d, got %d", i, expectedFrames[i], s.Bounce.FrameIndex)
		}
	}
}
