package pipeline

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/video"
)

// frameBatch carries extracted frames and their starting index through the pipeline.
type frameBatch struct {
	Frames     []video.Frame
	StartFrame int
}

// detectionBatch carries detection results and their starting frame index through the pipeline.
type detectionBatch struct {
	Detections []bridge.DetectionResult
	StartFrame int
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

	// 2. Detect court from first frame
	p.progress.Stage = "court_detection"
	firstFrames, err := vr.ExtractBatch(0, 1)
	if err != nil {
		return nil, fmt.Errorf("extract first frame: %w", err)
	}
	if len(firstFrames) > 0 {
		bridgeFrame := videoFrameToBridgeFrame(firstFrames[0])
		court, err := p.bridge.DetectCourt(bridgeFrame)
		if err != nil {
			return nil, fmt.Errorf("detect court: %w", err)
		}
		result.Court = court
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

			select {
			case detCh <- detectionBatch{Detections: detections, StartFrame: fb.StartFrame}:
			case <-ctx.Done():
				return
			}
		}
	}()

	// --- Stage 3: Tracking + assembly (main goroutine) ---
	for db := range detCh {
		result.Detections = append(result.Detections, db.Detections...)

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

	p.progress.Stage = "complete"
	p.progress.Percent = 100

	return result, nil
}
