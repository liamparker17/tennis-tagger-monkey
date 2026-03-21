package tracker

import (
	"testing"

	"github.com/liamp/tennis-tagger/internal/bridge"
)

func BenchmarkTrackerUpdate(b *testing.B) {
	tracker := NewTracker(5, 1, 0.3)

	// Two objects with slight frame-to-frame movement
	detections := []bridge.BBox{
		{X1: 100, Y1: 100, X2: 200, Y2: 300, Confidence: 0.95},
		{X1: 400, Y1: 200, X2: 500, Y2: 400, Confidence: 0.90},
	}

	// Warm up: seed the tracker with initial detections
	tracker.Update(detections)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simulate slight movement each frame
		offset := float64(i%10) * 0.5
		dets := []bridge.BBox{
			{X1: 100 + offset, Y1: 100 + offset, X2: 200 + offset, Y2: 300 + offset, Confidence: 0.95},
			{X1: 400 - offset, Y1: 200 + offset, X2: 500 - offset, Y2: 400 + offset, Confidence: 0.90},
		}
		tracker.Update(dets)
	}
}
