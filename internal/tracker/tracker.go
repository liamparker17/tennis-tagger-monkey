package tracker

import (
	"github.com/liamp/tennis-tagger/internal/bridge"
)

// TrackedObject represents a tracked object with its current state.
type TrackedObject struct {
	ID              int
	BBox            bridge.BBox
	HitCount        int
	TimeSinceUpdate int
	Age             int
}

// track is the internal representation of a tracked object.
type track struct {
	id              int
	kf              *KalmanFilter
	hitCount        int
	timeSinceUpdate int
	age             int
	lastConfidence  float64
}

// MultiObjectTracker tracks multiple objects across frames using Kalman filters and IoU assignment.
type MultiObjectTracker struct {
	maxAge       int
	minHits      int
	iouThreshold float64
	tracks       []*track
	nextID       int
}

// NewTracker creates a new MultiObjectTracker with the given parameters.
// maxAge: number of frames to keep a track alive without detections.
// minHits: minimum number of hits before a track is returned as active.
// iouThreshold: minimum IoU for a detection to be matched to a track.
func NewTracker(maxAge, minHits int, iouThreshold float64) *MultiObjectTracker {
	return &MultiObjectTracker{
		maxAge:       maxAge,
		minHits:      minHits,
		iouThreshold: iouThreshold,
		nextID:       1,
	}
}

// bboxToXYWH converts a bridge.BBox [x1,y1,x2,y2] to [cx, cy, w, h].
func bboxToXYWH(b bridge.BBox) [4]float64 {
	cx := (b.X1 + b.X2) / 2
	cy := (b.Y1 + b.Y2) / 2
	w := b.X2 - b.X1
	h := b.Y2 - b.Y1
	return [4]float64{cx, cy, w, h}
}

// xywhToX1Y1X2Y2 converts [cx, cy, w, h] to [x1, y1, x2, y2].
func xywhToX1Y1X2Y2(xywh [4]float64) [4]float64 {
	return [4]float64{
		xywh[0] - xywh[2]/2,
		xywh[1] - xywh[3]/2,
		xywh[0] + xywh[2]/2,
		xywh[1] + xywh[3]/2,
	}
}

// Update processes a new set of detections and returns the list of active tracked objects.
func (t *MultiObjectTracker) Update(detections []bridge.BBox) []TrackedObject {
	// 1. Predict all existing tracks
	for _, tr := range t.tracks {
		tr.kf.Predict()
	}

	// 2. Build bbox arrays in [x1,y1,x2,y2] format for assignment
	trackBoxes := make([][4]float64, len(t.tracks))
	for i, tr := range t.tracks {
		kfBBox := tr.kf.BBox() // [cx, cy, w, h]
		trackBoxes[i] = xywhToX1Y1X2Y2(kfBBox)
	}

	detBoxes := make([][4]float64, len(detections))
	for i, d := range detections {
		detBoxes[i] = [4]float64{d.X1, d.Y1, d.X2, d.Y2}
	}

	// 3. Assign detections to tracks
	matches, unmatchedDets, _ := Assign(trackBoxes, detBoxes, t.iouThreshold)

	// 4. Update matched tracks
	for _, m := range matches {
		tr := t.tracks[m.TrackIdx]
		det := detections[m.DetectionIdx]
		measurement := bboxToXYWH(det)
		tr.kf.Update(measurement)
		tr.hitCount++
		tr.timeSinceUpdate = 0
		tr.age++
		tr.lastConfidence = det.Confidence
	}

	// 5. Mark unmatched tracks
	matchedTrackSet := make(map[int]bool)
	for _, m := range matches {
		matchedTrackSet[m.TrackIdx] = true
	}
	for i, tr := range t.tracks {
		if !matchedTrackSet[i] {
			tr.timeSinceUpdate++
			tr.age++
		}
	}

	// 6. Create new tracks for unmatched detections
	for _, di := range unmatchedDets {
		det := detections[di]
		measurement := bboxToXYWH(det)
		kf := NewKalmanFilter(measurement)
		tr := &track{
			id:              t.nextID,
			kf:              kf,
			hitCount:        1,
			timeSinceUpdate: 0,
			age:             1,
			lastConfidence:  det.Confidence,
		}
		t.nextID++
		t.tracks = append(t.tracks, tr)
	}

	// 7. Remove dead tracks (timeSinceUpdate > maxAge)
	alive := make([]*track, 0, len(t.tracks))
	for _, tr := range t.tracks {
		if tr.timeSinceUpdate <= t.maxAge {
			alive = append(alive, tr)
		}
	}
	t.tracks = alive

	// 8. Return active tracks (hitCount >= minHits)
	var result []TrackedObject
	for _, tr := range t.tracks {
		if tr.hitCount >= t.minHits {
			kfBBox := tr.kf.BBox() // [cx, cy, w, h]
			xyxy := xywhToX1Y1X2Y2(kfBBox)
			result = append(result, TrackedObject{
				ID: tr.id,
				BBox: bridge.BBox{
					X1:         xyxy[0],
					Y1:         xyxy[1],
					X2:         xyxy[2],
					Y2:         xyxy[3],
					Confidence: tr.lastConfidence,
				},
				HitCount:        tr.hitCount,
				TimeSinceUpdate: tr.timeSinceUpdate,
				Age:             tr.age,
			})
		}
	}

	return result
}

// Reset clears all tracks and resets the ID counter.
func (t *MultiObjectTracker) Reset() {
	t.tracks = nil
	t.nextID = 1
}
