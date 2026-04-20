package bridge

// MockBridge is a test double for BridgeBackend that returns canned data.
// It can be configured with custom DetectFunc and CourtFunc for specific test scenarios.
type MockBridge struct {
	initialized bool
	detectCalls int
	DetectFunc  func(frames []Frame) ([]DetectionResult, error)
	CourtFunc   func(frame Frame) (CourtData, error)
}

// Compile-time interface check.
var _ BridgeBackend = (*MockBridge)(nil)

// NewMockBridge creates a new MockBridge instance.
func NewMockBridge() *MockBridge {
	return &MockBridge{}
}

// Init marks the bridge as initialized.
func (m *MockBridge) Init(config BridgeConfig) error {
	m.initialized = true
	return nil
}

// DetectBatch increments detectCalls and returns detections. If DetectFunc is set,
// it delegates to it. Otherwise, it returns canned data: 2 players + 1 ball per frame.
func (m *MockBridge) DetectBatch(frames []Frame) ([]DetectionResult, error) {
	m.detectCalls++

	if m.DetectFunc != nil {
		return m.DetectFunc(frames)
	}

	results := make([]DetectionResult, len(frames))
	for i := range frames {
		results[i] = DetectionResult{
			FrameIndex: i,
			Players: []BBox{
				{X1: 100, Y1: 200, X2: 150, Y2: 280, Confidence: 0.9},
				{X1: 400, Y1: 200, X2: 450, Y2: 280, Confidence: 0.85},
			},
			Ball: &BBox{X1: 300, Y1: 250, X2: 310, Y2: 260, Confidence: 0.7},
		}
	}
	return results, nil
}

// AnalyzePlacements returns "deuce_wide", "baseline", 45.0 for each detection.
func (m *MockBridge) AnalyzePlacements(detections []DetectionResult, court CourtData) ([]PlacementResult, error) {
	results := make([]PlacementResult, len(detections))
	for i := range detections {
		results[i] = PlacementResult{
			Zone:  "deuce_wide",
			Depth: "baseline",
			Angle: 45.0,
		}
	}
	return results, nil
}

// SegmentRallies returns 1 rally spanning all detections if there are >= 10 detections.
func (m *MockBridge) SegmentRallies(detections []DetectionResult, fps float64) ([]RallyResult, error) {
	if len(detections) < 10 {
		return nil, nil
	}

	last := detections[len(detections)-1]
	return []RallyResult{
		{
			StartFrame:     detections[0].FrameIndex,
			EndFrame:       last.FrameIndex,
			NumStrokes:     len(detections),
			DurationFrames: last.FrameIndex - detections[0].FrameIndex,
		},
	}, nil
}

// DetectCourt returns court data. If CourtFunc is set, it delegates to it.
// Otherwise, it returns an identity homography with mock corners.
func (m *MockBridge) DetectCourt(frame Frame) (CourtData, error) {
	if m.CourtFunc != nil {
		return m.CourtFunc(frame)
	}

	return CourtData{
		Corners: [4][2]float64{
			{0, 0},
			{1, 0},
			{1, 1},
			{0, 1},
		},
		Homography: [3][3]float64{
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
		},
		Method:     "mock",
		Confidence: 1.0,
	}, nil
}

// FitTrajectories returns one trajectory spanning all positions with a mock bounce.
func (m *MockBridge) FitTrajectories(positions []BallPosition, court CourtData, fps float64) ([]TrajectoryResult, error) {
	if len(positions) == 0 {
		return nil, nil
	}
	return []TrajectoryResult{
		{
			StartFrame: positions[0].FrameIndex,
			EndFrame:   positions[len(positions)-1].FrameIndex,
			Bounces: []Bounce{
				{
					FrameIndex: positions[len(positions)/2].FrameIndex,
					CX:         0.5,
					CY:         0.3,
					InOut:      "in",
				},
			},
			SpeedKPH:   120.0,
			Confidence: 0.85,
		},
	}, nil
}

// SetManualCourt is a no-op for the mock bridge.
func (m *MockBridge) SetManualCourt(setup MatchSetup) error { return nil }

// TrainModel is a no-op that returns nil.
func (m *MockBridge) TrainModel(pairs []TrainingPair, config TrainingConfig) error {
	return nil
}

// FineTune is a no-op that returns nil.
func (m *MockBridge) FineTune(corrections []CorrectionData, config TrainingConfig) error {
	return nil
}

// Close is a no-op.
func (m *MockBridge) Close() {}

// Initialized returns whether Init has been called.
func (m *MockBridge) Initialized() bool {
	return m.initialized
}

// DetectCalls returns the number of times DetectBatch has been called.
func (m *MockBridge) DetectCalls() int {
	return m.detectCalls
}
