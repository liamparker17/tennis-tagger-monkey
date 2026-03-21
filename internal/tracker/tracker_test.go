package tracker

import (
	"testing"

	"github.com/liamp/tennis-tagger/internal/bridge"
)

func TestTracker_SingleObject(t *testing.T) {
	tracker := NewTracker(3, 1, 0.3)

	detections := []bridge.BBox{
		{X1: 100, Y1: 100, X2: 150, Y2: 150, Confidence: 0.9},
	}

	result := tracker.Update(detections)
	if len(result) != 1 {
		t.Fatalf("Expected 1 tracked object, got %d", len(result))
	}
	if result[0].ID <= 0 {
		t.Errorf("Expected positive ID, got %d", result[0].ID)
	}
}

func TestTracker_PersistentID(t *testing.T) {
	tracker := NewTracker(3, 1, 0.3)

	// Frame 1
	det1 := []bridge.BBox{
		{X1: 100, Y1: 100, X2: 150, Y2: 150, Confidence: 0.9},
	}
	result1 := tracker.Update(det1)
	if len(result1) != 1 {
		t.Fatalf("Frame 1: expected 1 tracked object, got %d", len(result1))
	}
	id := result1[0].ID

	// Frame 2: slight movement
	det2 := []bridge.BBox{
		{X1: 103, Y1: 102, X2: 153, Y2: 152, Confidence: 0.9},
	}
	result2 := tracker.Update(det2)
	if len(result2) != 1 {
		t.Fatalf("Frame 2: expected 1 tracked object, got %d", len(result2))
	}
	if result2[0].ID != id {
		t.Errorf("Expected persistent ID %d, got %d", id, result2[0].ID)
	}

	// Frame 3: another slight movement
	det3 := []bridge.BBox{
		{X1: 106, Y1: 104, X2: 156, Y2: 154, Confidence: 0.9},
	}
	result3 := tracker.Update(det3)
	if len(result3) != 1 {
		t.Fatalf("Frame 3: expected 1 tracked object, got %d", len(result3))
	}
	if result3[0].ID != id {
		t.Errorf("Expected persistent ID %d across frames, got %d", id, result3[0].ID)
	}
}

func TestTracker_DeleteAfterMaxAge(t *testing.T) {
	maxAge := 2
	tracker := NewTracker(maxAge, 1, 0.3)

	// Frame 1: detection present
	det := []bridge.BBox{
		{X1: 100, Y1: 100, X2: 150, Y2: 150, Confidence: 0.9},
	}
	tracker.Update(det)

	// Frames 2 through maxAge+2: no detections
	for i := 0; i < maxAge+2; i++ {
		result := tracker.Update(nil)
		_ = result
	}

	// After maxAge+2 empty frames, track should be gone
	result := tracker.Update(nil)
	if len(result) != 0 {
		t.Errorf("Expected 0 tracked objects after maxAge exceeded, got %d", len(result))
	}
}

func TestTracker_NewFarDetectionCreatesNewTrack(t *testing.T) {
	tracker := NewTracker(3, 1, 0.3)

	// Frame 1: one object
	det1 := []bridge.BBox{
		{X1: 100, Y1: 100, X2: 150, Y2: 150, Confidence: 0.9},
	}
	result1 := tracker.Update(det1)
	if len(result1) != 1 {
		t.Fatalf("Frame 1: expected 1 tracked object, got %d", len(result1))
	}
	firstID := result1[0].ID

	// Frame 2: same object + new far-away object
	det2 := []bridge.BBox{
		{X1: 103, Y1: 102, X2: 153, Y2: 152, Confidence: 0.9},
		{X1: 500, Y1: 500, X2: 550, Y2: 550, Confidence: 0.8},
	}
	result2 := tracker.Update(det2)
	if len(result2) != 2 {
		t.Fatalf("Frame 2: expected 2 tracked objects, got %d", len(result2))
	}

	// Verify we have the original ID and a new one
	ids := map[int]bool{}
	for _, r := range result2 {
		ids[r.ID] = true
	}
	if !ids[firstID] {
		t.Errorf("Original track ID %d should still be present", firstID)
	}
	if len(ids) != 2 {
		t.Errorf("Expected 2 distinct IDs, got %d", len(ids))
	}
}
