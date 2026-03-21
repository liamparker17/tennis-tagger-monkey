package pipeline

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"sync"

	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/config"
	"github.com/liamp/tennis-tagger/internal/tracker"
	"github.com/liamp/tennis-tagger/internal/video"
)

// Result holds the complete output of a pipeline run.
type Result struct {
	VideoPath       string
	TotalFrames     int
	ProcessedFrames int
	FPS             float64
	Detections      []bridge.DetectionResult
	Tracks          [][]tracker.TrackedObject
	Court           bridge.CourtData
	Rallies         []bridge.RallyResult
	Placements      []bridge.PlacementResult
	Strokes         []bridge.StrokeResult
}

// ProgressInfo reports the current state of pipeline processing.
type ProgressInfo struct {
	ProcessedFrames int
	TotalFrames     int
	Percent         float64
	Stage           string
}

// Pipeline orchestrates video analysis through detection, tracking, and post-processing.
type Pipeline struct {
	bridge      bridge.BridgeBackend
	config      *config.Config
	tracker     *tracker.MultiObjectTracker
	progress    ProgressInfo
	liveReader  *video.LiveReader // nil when not in live mode
	liveRunning bool
	liveMu      sync.Mutex
}

// New creates a new Pipeline with the given backend and configuration.
func New(b bridge.BridgeBackend, cfg *config.Config) *Pipeline {
	return &Pipeline{
		bridge:  b,
		config:  cfg,
		tracker: tracker.NewTracker(5, 3, 0.3),
	}
}

// Progress returns the current processing progress.
func (p *Pipeline) Progress() ProgressInfo {
	return p.progress
}

// Bridge returns the underlying BridgeBackend used by this pipeline.
func (p *Pipeline) Bridge() bridge.BridgeBackend {
	return p.bridge
}

// Process runs the full analysis pipeline on the video at videoPath.
// It delegates to ProcessConcurrent for improved throughput.
func (p *Pipeline) Process(videoPath string) (*Result, error) {
	return p.ProcessConcurrent(videoPath)
}

// processSequential is the original sequential implementation, kept as a fallback.
func (p *Pipeline) processSequential(videoPath string) (*Result, error) {
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

	// 3. Batch detection loop
	p.progress.Stage = "detection"
	batchSize := p.config.Pipeline.BatchSize
	if batchSize <= 0 {
		batchSize = 32
	}
	checkpointEvery := p.config.Pipeline.CheckpointEvery
	if checkpointEvery <= 0 {
		checkpointEvery = 1000
	}

	// Determine checkpoint path (next to video file)
	cpPath := checkpointPath(videoPath)

	p.tracker.Reset()

	for start := 0; start < meta.TotalFrames; start += batchSize {
		count := batchSize
		if start+count > meta.TotalFrames {
			count = meta.TotalFrames - start
		}

		frames, err := vr.ExtractBatch(start, count)
		if err != nil {
			return nil, fmt.Errorf("extract batch at frame %d: %w", start, err)
		}
		if len(frames) == 0 {
			break
		}

		// Convert video frames to bridge frames
		bridgeFrames := make([]bridge.Frame, len(frames))
		for i, f := range frames {
			bridgeFrames[i] = videoFrameToBridgeFrame(f)
		}

		// Run detection
		detections, err := p.bridge.DetectBatch(bridgeFrames)
		if err != nil {
			return nil, fmt.Errorf("detect batch at frame %d: %w", start, err)
		}

		// Validate and filter detections
		for i := range detections {
			detections[i].FrameIndex = start + i
			detections[i].Players = filterBBoxes(detections[i].Players)
			if detections[i].Ball != nil && !isValidBBox(*detections[i].Ball) {
				detections[i].Ball = nil
			}
		}

		result.Detections = append(result.Detections, detections...)

		// Run tracker on each frame's detections
		for _, det := range detections {
			tracked := p.tracker.Update(det.Players)
			result.Tracks = append(result.Tracks, tracked)
		}

		result.ProcessedFrames = start + len(frames)
		p.progress.ProcessedFrames = result.ProcessedFrames
		if p.progress.TotalFrames > 0 {
			p.progress.Percent = float64(p.progress.ProcessedFrames) / float64(p.progress.TotalFrames) * 100
		}

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

// videoFrameToBridgeFrame converts a video.Frame to a bridge.Frame.
func videoFrameToBridgeFrame(f video.Frame) bridge.Frame {
	return bridge.Frame{
		Data:   f.Data,
		Width:  f.Width,
		Height: f.Height,
	}
}

// isValidBBox returns true if a bounding box has no NaN values and confidence <= 1.0.
func isValidBBox(b bridge.BBox) bool {
	if math.IsNaN(b.X1) || math.IsNaN(b.Y1) || math.IsNaN(b.X2) || math.IsNaN(b.Y2) || math.IsNaN(b.Confidence) {
		return false
	}
	if b.Confidence > 1.0 {
		return false
	}
	return true
}

// filterBBoxes returns only valid bounding boxes from the input slice.
func filterBBoxes(boxes []bridge.BBox) []bridge.BBox {
	var valid []bridge.BBox
	for _, b := range boxes {
		if isValidBBox(b) {
			valid = append(valid, b)
		}
	}
	return valid
}

// checkpointPath returns the checkpoint file path for a given video path.
func checkpointPath(videoPath string) string {
	dir := filepath.Dir(videoPath)
	base := strings.TrimSuffix(filepath.Base(videoPath), filepath.Ext(videoPath))
	return filepath.Join(dir, base+".checkpoint.json")
}
