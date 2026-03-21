package tracker

import (
	"math"
	"testing"
)

func TestComputeIoU_Identical(t *testing.T) {
	a := [4]float64{0, 0, 100, 100}
	b := [4]float64{0, 0, 100, 100}
	iou := ComputeIoU(a, b)
	if math.Abs(iou-1.0) > 1e-9 {
		t.Errorf("Expected IoU=1.0 for identical boxes, got %f", iou)
	}
}

func TestComputeIoU_NonOverlapping(t *testing.T) {
	a := [4]float64{0, 0, 50, 50}
	b := [4]float64{100, 100, 200, 200}
	iou := ComputeIoU(a, b)
	if iou != 0 {
		t.Errorf("Expected IoU=0 for non-overlapping boxes, got %f", iou)
	}
}

func TestComputeIoU_HalfOverlap(t *testing.T) {
	// Box a: [0,0,100,100] area=10000
	// Box b: [50,0,150,100] area=10000
	// Intersection: [50,0,100,100] area=5000
	// Union: 10000+10000-5000=15000
	// IoU: 5000/15000 = 1/3
	a := [4]float64{0, 0, 100, 100}
	b := [4]float64{50, 0, 150, 100}
	iou := ComputeIoU(a, b)
	expected := 1.0 / 3.0
	if math.Abs(iou-expected) > 1e-9 {
		t.Errorf("Expected IoU=%f, got %f", expected, iou)
	}
}

func TestAssign_TwoTracksAndTwoDetections(t *testing.T) {
	tracks := [][4]float64{
		{0, 0, 50, 50},
		{100, 100, 150, 150},
	}
	detections := [][4]float64{
		{2, 2, 52, 52},     // close to track 0
		{102, 102, 152, 152}, // close to track 1
	}

	matches, unmatchedDets, unmatchedTracks := Assign(tracks, detections, 0.3)

	if len(matches) != 2 {
		t.Fatalf("Expected 2 matches, got %d", len(matches))
	}
	if len(unmatchedDets) != 0 {
		t.Errorf("Expected 0 unmatched detections, got %d", len(unmatchedDets))
	}
	if len(unmatchedTracks) != 0 {
		t.Errorf("Expected 0 unmatched tracks, got %d", len(unmatchedTracks))
	}

	// Verify correct assignment
	for _, m := range matches {
		if m.TrackIdx == 0 && m.DetectionIdx != 0 {
			t.Errorf("Track 0 should match detection 0, got detection %d", m.DetectionIdx)
		}
		if m.TrackIdx == 1 && m.DetectionIdx != 1 {
			t.Errorf("Track 1 should match detection 1, got detection %d", m.DetectionIdx)
		}
	}
}

func TestAssign_OneTrackTwoDetections(t *testing.T) {
	tracks := [][4]float64{
		{0, 0, 50, 50},
	}
	detections := [][4]float64{
		{2, 2, 52, 52},       // close to track 0
		{200, 200, 250, 250}, // far away
	}

	matches, unmatchedDets, unmatchedTracks := Assign(tracks, detections, 0.3)

	if len(matches) != 1 {
		t.Fatalf("Expected 1 match, got %d", len(matches))
	}
	if len(unmatchedDets) != 1 {
		t.Fatalf("Expected 1 unmatched detection, got %d", len(unmatchedDets))
	}
	if unmatchedDets[0] != 1 {
		t.Errorf("Expected unmatched detection index 1, got %d", unmatchedDets[0])
	}
	if len(unmatchedTracks) != 0 {
		t.Errorf("Expected 0 unmatched tracks, got %d", len(unmatchedTracks))
	}
}
