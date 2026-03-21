package pipeline

import (
	"math"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/config"
)

func testVideoPath(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot determine test file path")
	}
	return filepath.Join(filepath.Dir(thisFile), "..", "..", "testdata", "sample_5s.mp4")
}

func TestProcessWithMockBridge(t *testing.T) {
	cfg := config.Default()
	mock := bridge.NewMockBridge()
	p := New(mock, cfg)

	result, err := p.Process(testVideoPath(t))
	if err != nil {
		t.Fatalf("Process: %v", err)
	}

	if result.TotalFrames == 0 {
		t.Error("TotalFrames should be non-zero")
	}
	if result.ProcessedFrames == 0 {
		t.Error("ProcessedFrames should be non-zero")
	}
	if result.FPS == 0 {
		t.Error("FPS should be non-zero")
	}
	if len(result.Detections) == 0 {
		t.Error("Detections should be non-empty")
	}
	if len(result.Tracks) == 0 {
		t.Error("Tracks should be non-empty")
	}

	progress := p.Progress()
	if progress.Stage != "complete" {
		t.Errorf("Stage = %q, want %q", progress.Stage, "complete")
	}
	if progress.Percent != 100 {
		t.Errorf("Percent = %f, want 100", progress.Percent)
	}
}

func TestFilterBBoxes(t *testing.T) {
	boxes := []bridge.BBox{
		{X1: 0, Y1: 0, X2: 10, Y2: 10, Confidence: 0.9},               // valid
		{X1: math.NaN(), Y1: 0, X2: 10, Y2: 10, Confidence: 0.9},       // NaN X1
		{X1: 0, Y1: 0, X2: 10, Y2: 10, Confidence: 1.5},                // confidence > 1.0
		{X1: 0, Y1: 0, X2: 10, Y2: math.NaN(), Confidence: 0.8},        // NaN Y2
		{X1: 5, Y1: 5, X2: 15, Y2: 15, Confidence: 1.0},                // valid (confidence == 1.0)
	}

	filtered := filterBBoxes(boxes)
	if len(filtered) != 2 {
		t.Errorf("filterBBoxes returned %d boxes, want 2", len(filtered))
	}
}
