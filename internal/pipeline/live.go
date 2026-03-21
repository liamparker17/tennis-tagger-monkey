package pipeline

import (
	"fmt"

	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/video"
)

// LiveResult holds the output of a single live processing batch.
type LiveResult struct {
	Detections []bridge.DetectionResult
	FrameCount int
	Error      error
}

// ProcessLive starts live capture from source and processes frames continuously.
// It returns a channel that emits LiveResult for each processed batch.
// The channel is closed when the stream ends or StopLive is called.
func (p *Pipeline) ProcessLive(source string, width, height int, fps float64) (<-chan LiveResult, error) {
	lr, err := video.OpenLive(source, width, height, fps)
	if err != nil {
		return nil, fmt.Errorf("open live source: %w", err)
	}

	p.liveMu.Lock()
	p.liveReader = lr
	p.liveRunning = true
	p.liveMu.Unlock()

	batchSize := p.config.Pipeline.LiveBatchSize
	if batchSize <= 0 {
		batchSize = 4
	}

	ch := make(chan LiveResult, 2)

	go func() {
		defer close(ch)
		defer func() {
			p.liveMu.Lock()
			p.liveRunning = false
			p.liveMu.Unlock()
		}()

		frameCount := 0

		// Detect court from the first batch of frames.
		firstBatch, err := lr.ReadBatch(batchSize)
		if err != nil || len(firstBatch) == 0 {
			ch <- LiveResult{Error: fmt.Errorf("read first batch: %w", err)}
			return
		}

		if len(firstBatch) > 0 {
			bridgeFrame := videoFrameToBridgeFrame(firstBatch[0])
			_, courtErr := p.bridge.DetectCourt(bridgeFrame)
			if courtErr != nil {
				// Court detection failure is non-fatal for live mode; continue processing.
			}
		}

		// Process the first batch through detection.
		firstResult := p.processLiveBatch(firstBatch, frameCount)
		frameCount += len(firstBatch)
		ch <- firstResult

		// Main processing loop.
		for {
			p.liveMu.Lock()
			running := p.liveRunning
			p.liveMu.Unlock()
			if !running {
				break
			}

			frames, err := lr.ReadBatch(batchSize)
			if err != nil || len(frames) == 0 {
				break
			}

			result := p.processLiveBatch(frames, frameCount)
			frameCount += len(frames)
			ch <- result

			if result.Error != nil {
				break
			}
		}
	}()

	return ch, nil
}

// processLiveBatch runs detection and tracking on a batch of live frames.
func (p *Pipeline) processLiveBatch(frames []video.Frame, startIndex int) LiveResult {
	bridgeFrames := make([]bridge.Frame, len(frames))
	for i, f := range frames {
		bridgeFrames[i] = videoFrameToBridgeFrame(f)
	}

	detections, err := p.bridge.DetectBatch(bridgeFrames)
	if err != nil {
		return LiveResult{Error: fmt.Errorf("detect batch: %w", err)}
	}

	for i := range detections {
		detections[i].FrameIndex = startIndex + i
		detections[i].Players = filterBBoxes(detections[i].Players)
		if detections[i].Ball != nil && !isValidBBox(*detections[i].Ball) {
			detections[i].Ball = nil
		}
	}

	// Update tracker with each frame's detections.
	for _, det := range detections {
		p.tracker.Update(det.Players)
	}

	return LiveResult{
		Detections: detections,
		FrameCount: len(frames),
	}
}

// StopLive stops the live capture and processing.
func (p *Pipeline) StopLive() {
	p.liveMu.Lock()
	defer p.liveMu.Unlock()

	p.liveRunning = false
	if p.liveReader != nil {
		p.liveReader.Stop()
		p.liveReader = nil
	}
}

// IsLive returns whether live processing is currently active.
func (p *Pipeline) IsLive() bool {
	p.liveMu.Lock()
	defer p.liveMu.Unlock()
	return p.liveRunning
}
