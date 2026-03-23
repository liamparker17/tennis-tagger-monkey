package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/point"
	"github.com/liamp/tennis-tagger/internal/video"
)

// frameBatch carries extracted frames and their starting index through the pipeline.
type frameBatch struct {
	Frames     []video.Frame
	StartFrame int
}

// detectionBatch carries detection results, ball positions, and their starting frame index through the pipeline.
type detectionBatch struct {
	Detections    []bridge.DetectionResult
	BallPositions []bridge.BallPosition
	StartFrame    int
}

// ProcessConcurrent runs the full analysis pipeline using a 3-stage concurrent architecture.
// Stage 1: Frame extraction (goroutine) -> frameCh
// Stage 2: ML inference (goroutine) -> detCh
// Stage 3: Tracking + assembly (main goroutine)
func (p *Pipeline) ProcessConcurrent(videoPath string) (*Result, error) {
	// 1. Open video
	p.progress = ProgressInfo{Stage: "opening"}
	vr, err := video.Open(videoPath)
	if err != nil {
		return nil, fmt.Errorf("open video: %w", err)
	}
	defer vr.Close()

	meta := vr.Metadata()
	p.progress.TotalFrames = meta.TotalFrames

	result := &Result{
		VideoPath:   videoPath,
		TotalFrames: meta.TotalFrames,
		FPS:         meta.FPS,
	}

	// 2. Detect court from first frame at native resolution
	// Native res gives clearer court lines for detection.
	// The bridge scales the resulting polygon to match detection frame size.
	p.progress.Stage = "court_detection"
	nativeFrame, err := vr.ExtractNative(0)
	if err != nil {
		slog.Warn("Native frame extraction failed, using scaled frame", "error", err)
		// Fallback to scaled frame
		firstFrames, err2 := vr.ExtractBatch(0, 1)
		if err2 == nil && len(firstFrames) > 0 {
			nativeFrame = firstFrames[0]
		}
	}
	if nativeFrame.Data != nil {
		bridgeFrame := videoFrameToBridgeFrame(nativeFrame)
		court, err := p.bridge.DetectCourt(bridgeFrame)
		if err != nil {
			return nil, fmt.Errorf("detect court: %w", err)
		}
		result.Court = court
	}

	// 2b. Sample reference frames for TrackNet background subtraction.
	// Extract 30 frames evenly spaced across the video.
	{
		refCount := 30
		if meta.TotalFrames < refCount {
			refCount = meta.TotalFrames
		}
		step := meta.TotalFrames / refCount
		if step < 1 {
			step = 1
		}
		var refFrames []bridge.Frame
		for i := 0; i < meta.TotalFrames && len(refFrames) < refCount; i += step {
			frames, err := vr.ExtractBatch(i, 1)
			if err == nil && len(frames) > 0 {
				refFrames = append(refFrames, videoFrameToBridgeFrame(frames[0]))
			}
		}
		if len(refFrames) > 0 {
			if err := p.bridge.SetBackgroundReference(refFrames); err != nil {
				slog.Warn("Failed to set background reference", "error", err)
			} else {
				slog.Info("Background reference set", "frames", len(refFrames))
			}
		}
	}

	// 3. Set up pipeline parameters
	p.progress.Stage = "detection"
	batchSize := p.config.Pipeline.BatchSize
	if batchSize <= 0 {
		batchSize = 32
	}
	checkpointEvery := p.config.Pipeline.CheckpointEvery
	if checkpointEvery <= 0 {
		checkpointEvery = 1000
	}
	cpPath := checkpointPath(videoPath)
	p.tracker.Reset()

	// Buffered channels between stages
	frameCh := make(chan frameBatch, 2)
	detCh := make(chan detectionBatch, 2)

	var wg sync.WaitGroup
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var stage1Err, stage2Err error

	// --- Stage 1: Frame extraction (goroutine) ---
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(frameCh)

		for start := 0; start < meta.TotalFrames; start += batchSize {
			count := batchSize
			if start+count > meta.TotalFrames {
				count = meta.TotalFrames - start
			}

			frames, err := vr.ExtractBatch(start, count)
			if err != nil {
				stage1Err = fmt.Errorf("extract batch at frame %d: %w", start, err)
				cancel()
				return
			}
			if len(frames) == 0 {
				return
			}

			select {
			case frameCh <- frameBatch{Frames: frames, StartFrame: start}:
			case <-ctx.Done():
				return
			}
		}
	}()

	// --- Stage 2: ML inference (goroutine) ---
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(detCh)

		for fb := range frameCh {
			bFrames := make([]bridge.Frame, len(fb.Frames))
			for i, f := range fb.Frames {
				bFrames[i] = videoFrameToBridgeFrame(f)
			}

			detections, err := p.bridge.DetectBatch(bFrames)
			if err != nil {
				stage2Err = fmt.Errorf("detect batch at frame %d: %w", fb.StartFrame, err)
				cancel()
				// Drain frameCh to unblock stage 1
				for range frameCh {
				}
				return
			}

			// Set FrameIndex and filter invalid BBoxes
			for i := range detections {
				detections[i].FrameIndex = fb.StartFrame + i
				before := len(detections[i].Players)
				detections[i].Players = filterBBoxes(detections[i].Players)
				if dropped := before - len(detections[i].Players); dropped > 0 {
					slog.Warn("Dropped invalid player bboxes", "frame", fb.StartFrame+i, "count", dropped)
				}
				if detections[i].Ball != nil && !isValidBBox(*detections[i].Ball) {
					slog.Warn("Dropped invalid ball bbox", "frame", fb.StartFrame+i)
					detections[i].Ball = nil
				}
			}

			// Run TrackNet on the same frames for dedicated ball detection.
			trackNetPositions, trackErr := p.bridge.TrackNetBatch(bFrames)
			if trackErr != nil {
				slog.Warn("TrackNet batch failed, using YOLO ball only", "frame", fb.StartFrame, "error", trackErr)
			}

			// Merge TrackNet + YOLO ball positions for this batch.
			ballPositions := mergeBallPositions(detections, trackNetPositions, fb.StartFrame)

			select {
			case detCh <- detectionBatch{Detections: detections, BallPositions: ballPositions, StartFrame: fb.StartFrame}:
			case <-ctx.Done():
				return
			}
		}
	}()

	// --- Stage 3: Tracking + assembly (main goroutine) ---
	for db := range detCh {
		result.Detections = append(result.Detections, db.Detections...)
		result.BallPositions = append(result.BallPositions, db.BallPositions...)

		for _, det := range db.Detections {
			tracked := p.tracker.Update(det.Players)
			result.Tracks = append(result.Tracks, tracked)
		}

		result.ProcessedFrames = db.StartFrame + len(db.Detections)
		pct := 0.0
		if meta.TotalFrames > 0 {
			pct = float64(result.ProcessedFrames) / float64(meta.TotalFrames) * 100
		}
		p.setProgress(ProgressInfo{
			ProcessedFrames: result.ProcessedFrames,
			TotalFrames:     meta.TotalFrames,
			Percent:         pct,
			Stage:           "processing",
		})

		// Save checkpoint periodically
		if result.ProcessedFrames%checkpointEvery < batchSize {
			cp := &Checkpoint{
				VideoPath:       videoPath,
				ProcessedFrames: result.ProcessedFrames,
				TotalFrames:     meta.TotalFrames,
			}
			_ = SaveCheckpoint(cp, cpPath) // best-effort
		}
	}

	// Wait for stages 1 and 2 to finish (they should already be done since detCh is closed)
	wg.Wait()

	// Check for errors from stages 1 and 2
	if stage1Err != nil {
		return nil, stage1Err
	}
	if stage2Err != nil {
		return nil, stage2Err
	}

	// 4. Post-processing: segment rallies
	p.progress.Stage = "post_processing"
	rallies, err := p.bridge.SegmentRallies(result.Detections, meta.FPS)
	if err != nil {
		return nil, fmt.Errorf("segment rallies: %w", err)
	}
	result.Rallies = rallies

	// 5. Post-processing: analyze placements
	placements, err := p.bridge.AnalyzePlacements(result.Detections, result.Court)
	if err != nil {
		return nil, fmt.Errorf("analyze placements: %w", err)
	}
	result.Placements = placements

	// 6. Fit trajectories
	p.progress.Stage = "trajectory_fitting"
	slog.Info("Ball detection summary", "tracknet_and_yolo", len(result.BallPositions), "total_frames", len(result.Detections))
	if len(result.BallPositions) > 0 {
		trajectories, err := p.bridge.FitTrajectories(result.BallPositions, result.Court, meta.FPS)
		if err != nil {
			slog.Warn("trajectory fitting failed", "error", err)
		} else {
			result.Trajectories = trajectories
		}
	}

	// 7. Segment shots (pure Go)
	if len(result.Trajectories) > 0 {
		result.Shots = point.SegmentShots(result.Trajectories)
	}

	// 8. Recognize points + track score (pure Go)
	if len(result.Shots) > 0 {
		shotGroups := groupShotsIntoPoints(result.Shots, meta.FPS)
		points, score := point.RecognizePoints(shotGroups, 0) // 0 = near player serves first (default)
		result.Points = points
		result.MatchScore = score
	}

	slog.Info("Analysis complete", "trajectories", len(result.Trajectories), "shots", len(result.Shots), "points", len(result.Points))

	p.progress.Stage = "complete"
	p.progress.Percent = 100

	return result, nil
}

// mergeBallPositions combines TrackNet ball positions with YOLO ball detections.
// For each frame, if YOLO detected a ball, a BallPosition is created from it.
// TrackNet positions are included as-is. When both sources detect a ball on
// the same frame, both are kept (downstream trajectory fitting can weight them).
func mergeBallPositions(detections []bridge.DetectionResult, trackNet []bridge.BallPosition, startFrame int) []bridge.BallPosition {
	var merged []bridge.BallPosition

	// Add TrackNet positions.
	merged = append(merged, trackNet...)

	// Add YOLO ball detections as BallPositions.
	for _, det := range detections {
		if det.Ball != nil {
			cx := (det.Ball.X1 + det.Ball.X2) / 2
			cy := (det.Ball.Y1 + det.Ball.Y2) / 2
			merged = append(merged, bridge.BallPosition{
				X:          cx,
				Y:          cy,
				Confidence: det.Ball.Confidence,
				FrameIndex: det.FrameIndex,
				Source:     "yolo",
			})
		}
	}

	return merged
}

// groupShotsIntoPoints splits a sequence of shots into groups, where each
// group represents one point. A gap of more than 3 seconds (measured in
// frames) between consecutive shots starts a new group.
func groupShotsIntoPoints(shots []point.Shot, fps float64) [][]point.Shot {
	if len(shots) == 0 {
		return nil
	}
	gapFrames := int(fps * 3) // 3-second gap = new point
	if gapFrames <= 0 {
		gapFrames = 90 // fallback for unknown FPS
	}

	var groups [][]point.Shot
	current := []point.Shot{shots[0]}

	for i := 1; i < len(shots); i++ {
		if shots[i].StartFrame-shots[i-1].EndFrame > gapFrames {
			groups = append(groups, current)
			current = []point.Shot{shots[i]}
		} else {
			current = append(current, shots[i])
		}
	}
	groups = append(groups, current)

	return groups
}
