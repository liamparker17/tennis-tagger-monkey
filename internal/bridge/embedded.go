//go:build ignore

package bridge

import (
	"encoding/json"
	"errors"
)

// embeddedCaller is a pythonCaller stub for the embedded CPython bridge.
// The real implementation will use cgo to call into a Python interpreter.
type embeddedCaller struct{}

func (e *embeddedCaller) call(method string, payload json.RawMessage) (json.RawMessage, error) {
	return nil, errors.New("embedded python caller: not yet implemented")
}

func (e *embeddedCaller) init(config BridgeConfig) error {
	return errors.New("embedded python caller: not yet implemented")
}

func (e *embeddedCaller) close() {}

// EmbeddedBridge implements BridgeBackend using an in-process CPython interpreter.
// This file is parked with //go:build ignore until the embedded approach is needed.
type EmbeddedBridge struct {
	worker *Worker
}

// Compile-time interface check.
var _ BridgeBackend = (*EmbeddedBridge)(nil)

// NewEmbeddedBridge creates a new EmbeddedBridge with a worker backed by embeddedCaller.
func NewEmbeddedBridge() *EmbeddedBridge {
	return &EmbeddedBridge{
		worker: NewWorker(&embeddedCaller{}),
	}
}

// Init initializes the embedded Python interpreter.
func (b *EmbeddedBridge) Init(config BridgeConfig) error {
	return errors.New("EmbeddedBridge.Init: not yet implemented")
}

// DetectBatch runs object detection on a batch of frames.
func (b *EmbeddedBridge) DetectBatch(frames []Frame) ([]DetectionResult, error) {
	return nil, errors.New("EmbeddedBridge.DetectBatch: not yet implemented")
}

// ClassifyStrokes classifies strokes from frame clips.
func (b *EmbeddedBridge) ClassifyStrokes(clips []FrameClip) ([]StrokeResult, error) {
	return nil, errors.New("EmbeddedBridge.ClassifyStrokes: not yet implemented")
}

// AnalyzePlacements analyzes shot placements given detections and court data.
func (b *EmbeddedBridge) AnalyzePlacements(detections []DetectionResult, court CourtData) ([]PlacementResult, error) {
	return nil, errors.New("EmbeddedBridge.AnalyzePlacements: not yet implemented")
}

// SegmentRallies segments a sequence of detections into rallies.
func (b *EmbeddedBridge) SegmentRallies(detections []DetectionResult, fps float64) ([]RallyResult, error) {
	return nil, errors.New("EmbeddedBridge.SegmentRallies: not yet implemented")
}

// DetectCourt detects the court in a single frame.
func (b *EmbeddedBridge) DetectCourt(frame Frame) (CourtData, error) {
	return CourtData{}, errors.New("EmbeddedBridge.DetectCourt: not yet implemented")
}

// TrackNetBatch runs TrackNet ball detection on a batch of frames.
func (b *EmbeddedBridge) TrackNetBatch(frames []Frame) ([]BallPosition, error) {
	return nil, errors.New("EmbeddedBridge.TrackNetBatch: not yet implemented")
}

// FitTrajectories sends ball positions to the trajectory fitter.
func (b *EmbeddedBridge) FitTrajectories(positions []BallPosition, court CourtData, fps float64) ([]TrajectoryResult, error) {
	return nil, errors.New("EmbeddedBridge.FitTrajectories: not yet implemented")
}

// TrainModel starts a training run.
func (b *EmbeddedBridge) TrainModel(pairs []TrainingPair, config TrainingConfig) error {
	return errors.New("EmbeddedBridge.TrainModel: not yet implemented")
}

// FineTune sends user corrections to the fine-tuning endpoint.
func (b *EmbeddedBridge) FineTune(corrections []CorrectionData, config TrainingConfig) error {
	return errors.New("EmbeddedBridge.FineTune: not yet implemented")
}

// Close releases resources held by the embedded bridge.
func (b *EmbeddedBridge) Close() {
	if b.worker != nil {
		b.worker.Stop()
	}
}
