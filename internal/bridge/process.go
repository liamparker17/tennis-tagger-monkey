//go:build !python3

package bridge

import (
	"encoding/json"
	"errors"
)

// processCaller is a pythonCaller stub for the subprocess-based bridge.
// The real implementation will spawn a Python child process and communicate via JSON-RPC.
type processCaller struct{}

func (p *processCaller) call(method string, payload json.RawMessage) (json.RawMessage, error) {
	return nil, errors.New("process caller: not yet implemented")
}

func (p *processCaller) init(config BridgeConfig) error {
	return errors.New("process caller: not yet implemented")
}

func (p *processCaller) close() {}

// ProcessBridge implements BridgeBackend by communicating with a Python subprocess.
// This build variant is selected when the python3 build tag is NOT set (default).
type ProcessBridge struct {
	worker *Worker
}

// Compile-time interface check.
var _ BridgeBackend = (*ProcessBridge)(nil)

// NewProcessBridge creates a new ProcessBridge with a worker backed by processCaller.
func NewProcessBridge() *ProcessBridge {
	return &ProcessBridge{
		worker: NewWorker(&processCaller{}),
	}
}

// Init initializes the Python subprocess.
func (b *ProcessBridge) Init(config BridgeConfig) error {
	return errors.New("ProcessBridge.Init: not yet implemented")
}

// DetectBatch runs object detection on a batch of frames.
func (b *ProcessBridge) DetectBatch(frames []Frame) ([]DetectionResult, error) {
	return nil, errors.New("ProcessBridge.DetectBatch: not yet implemented")
}

// ClassifyStrokes classifies strokes from frame clips.
func (b *ProcessBridge) ClassifyStrokes(clips []FrameClip) ([]StrokeResult, error) {
	return nil, errors.New("ProcessBridge.ClassifyStrokes: not yet implemented")
}

// AnalyzePlacements analyzes shot placements given detections and court data.
func (b *ProcessBridge) AnalyzePlacements(detections []DetectionResult, court CourtData) ([]PlacementResult, error) {
	return nil, errors.New("ProcessBridge.AnalyzePlacements: not yet implemented")
}

// SegmentRallies segments a sequence of detections into rallies.
func (b *ProcessBridge) SegmentRallies(detections []DetectionResult, fps float64) ([]RallyResult, error) {
	return nil, errors.New("ProcessBridge.SegmentRallies: not yet implemented")
}

// DetectCourt detects the court in a single frame.
func (b *ProcessBridge) DetectCourt(frame Frame) (CourtData, error) {
	return CourtData{}, errors.New("ProcessBridge.DetectCourt: not yet implemented")
}

// TrainModel starts a training run.
func (b *ProcessBridge) TrainModel(pairs []TrainingPair, config TrainingConfig) error {
	return errors.New("ProcessBridge.TrainModel: not yet implemented")
}

// Close releases resources held by the process bridge.
func (b *ProcessBridge) Close() {
	if b.worker != nil {
		b.worker.Stop()
	}
}
