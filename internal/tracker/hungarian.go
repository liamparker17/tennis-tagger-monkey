package tracker

import "math"

// Match represents a matched pair of track and detection indices with their IoU score.
type Match struct {
	TrackIdx     int
	DetectionIdx int
	IoU          float64
}

// ComputeIoU calculates the Intersection over Union for two bounding boxes
// in [x1, y1, x2, y2] format.
func ComputeIoU(a, b [4]float64) float64 {
	// Intersection coordinates
	x1 := math.Max(a[0], b[0])
	y1 := math.Max(a[1], b[1])
	x2 := math.Min(a[2], b[2])
	y2 := math.Min(a[3], b[3])

	// Intersection area
	interW := math.Max(0, x2-x1)
	interH := math.Max(0, y2-y1)
	interArea := interW * interH

	if interArea == 0 {
		return 0
	}

	// Union area
	areaA := (a[2] - a[0]) * (a[3] - a[1])
	areaB := (b[2] - b[0]) * (b[3] - b[1])
	unionArea := areaA + areaB - interArea

	if unionArea <= 0 {
		return 0
	}

	return interArea / unionArea
}

// Assign performs greedy assignment of detections to tracks using an IoU cost matrix.
// It returns matched pairs, indices of unmatched detections, and indices of unmatched tracks.
func Assign(tracks, detections [][4]float64, iouThreshold float64) ([]Match, []int, []int) {
	numTracks := len(tracks)
	numDets := len(detections)

	if numTracks == 0 && numDets == 0 {
		return nil, nil, nil
	}
	if numTracks == 0 {
		unmatchedDets := make([]int, numDets)
		for i := range unmatchedDets {
			unmatchedDets[i] = i
		}
		return nil, unmatchedDets, nil
	}
	if numDets == 0 {
		unmatchedTracks := make([]int, numTracks)
		for i := range unmatchedTracks {
			unmatchedTracks[i] = i
		}
		return nil, nil, unmatchedTracks
	}

	// Build IoU cost matrix
	iouMatrix := make([][]float64, numTracks)
	for i := range iouMatrix {
		iouMatrix[i] = make([]float64, numDets)
		for j := 0; j < numDets; j++ {
			iouMatrix[i][j] = ComputeIoU(tracks[i], detections[j])
		}
	}

	// Greedy assignment: pick the highest IoU pair iteratively
	trackUsed := make([]bool, numTracks)
	detUsed := make([]bool, numDets)
	var matches []Match

	for {
		bestIoU := -1.0
		bestT, bestD := -1, -1

		for i := 0; i < numTracks; i++ {
			if trackUsed[i] {
				continue
			}
			for j := 0; j < numDets; j++ {
				if detUsed[j] {
					continue
				}
				if iouMatrix[i][j] > bestIoU {
					bestIoU = iouMatrix[i][j]
					bestT = i
					bestD = j
				}
			}
		}

		if bestT < 0 || bestIoU < iouThreshold {
			break
		}

		matches = append(matches, Match{
			TrackIdx:     bestT,
			DetectionIdx: bestD,
			IoU:          bestIoU,
		})
		trackUsed[bestT] = true
		detUsed[bestD] = true
	}

	// Collect unmatched
	var unmatchedDets []int
	for j := 0; j < numDets; j++ {
		if !detUsed[j] {
			unmatchedDets = append(unmatchedDets, j)
		}
	}
	var unmatchedTracks []int
	for i := 0; i < numTracks; i++ {
		if !trackUsed[i] {
			unmatchedTracks = append(unmatchedTracks, i)
		}
	}

	return matches, unmatchedDets, unmatchedTracks
}
