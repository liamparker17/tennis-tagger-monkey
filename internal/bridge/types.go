package bridge

// BBox represents a bounding box with confidence score.
type BBox struct {
	X1         float64 `json:"x1"`
	Y1         float64 `json:"y1"`
	X2         float64 `json:"x2"`
	Y2         float64 `json:"y2"`
	Confidence float64 `json:"confidence"`
}

// Center returns the center point of the bounding box.
func (b BBox) Center() (float64, float64) {
	return (b.X1 + b.X2) / 2, (b.Y1 + b.Y2) / 2
}

// Width returns the width of the bounding box.
func (b BBox) Width() float64 {
	return b.X2 - b.X1
}

// Height returns the height of the bounding box.
func (b BBox) Height() float64 {
	return b.Y2 - b.Y1
}

// Frame represents a single video frame.
type Frame struct {
	Data   []byte `json:"data"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
}

// FrameClip represents a clip of consecutive frames centered on a particular frame.
type FrameClip struct {
	Frames []Frame `json:"frames"`
	Center int     `json:"center"`
}

// PoseKeypoint represents a single keypoint in a pose skeleton.
type PoseKeypoint struct {
	X          float64 `json:"x"`
	Y          float64 `json:"y"`
	Confidence float64 `json:"confidence"`
}

// PoseData represents pose estimation results for a single player.
type PoseData struct {
	PlayerBBox BBox           `json:"playerBBox"`
	Keypoints  []PoseKeypoint `json:"keypoints"`
}

// DetectionResult holds detection outputs for a single frame.
type DetectionResult struct {
	FrameIndex int        `json:"frameIndex"`
	Players    []BBox     `json:"players"`
	Ball       *BBox      `json:"ball,omitempty"`
	Poses      []PoseData `json:"poses"`
}

// StrokeResult holds stroke classification output.
type StrokeResult struct {
	Type       string  `json:"type"`
	Confidence float64 `json:"confidence"`
	Frame      int     `json:"frame"`
	PlayerID   int     `json:"playerId"`
}

// PlacementResult holds shot placement analysis output.
type PlacementResult struct {
	Zone  string  `json:"zone"`
	Depth string  `json:"depth"`
	Angle float64 `json:"angle"`
}

// CourtData holds court detection and homography data.
type CourtData struct {
	Corners    [4][2]float64 `json:"corners"`
	Homography [3][3]float64 `json:"homography"`
	Method     string        `json:"method"`
	Confidence float64       `json:"confidence"`
}

// RallyResult holds rally segmentation output.
type RallyResult struct {
	StartFrame     int `json:"startFrame"`
	EndFrame       int `json:"endFrame"`
	NumStrokes     int `json:"numStrokes"`
	DurationFrames int `json:"durationFrames"`
}

// BallPosition holds a ball detection with sub-pixel position and source.
type BallPosition struct {
	X          float64 `json:"x"`
	Y          float64 `json:"y"`
	Confidence float64 `json:"confidence"`
	FrameIndex int     `json:"frameIndex"`
	Source     string  `json:"source"` // "tracknet", "yolo", "merged"
}

// BridgeConfig holds configuration for initializing a bridge backend.
type BridgeConfig struct {
	ModelsDir       string `json:"ModelsDir"`
	Device          string `json:"Device"`
	DetectorBackend string `json:"DetectorBackend"`
	ClassifierModel string `json:"ClassifierModel"`
}

// TrainingPair represents a video-CSV pair for training.
type TrainingPair struct {
	VideoPath string `json:"videoPath"`
	CSVPath   string `json:"csvPath"`
}

// TrainingConfig holds configuration for a training run.
type TrainingConfig struct {
	Task      string `json:"task"`
	Epochs    int    `json:"epochs"`
	BatchSize int    `json:"batchSize"`
	Device    string `json:"device"`
}

// Bounce holds a single detected ball bounce with court position and in/out call.
type Bounce struct {
	FrameIndex int     `json:"frameIndex"`
	CX         float64 `json:"cx"`
	CY         float64 `json:"cy"`
	InOut      string  `json:"inOut"` // "in", "out", "close_call"
}

// TrajectoryResult holds a fitted ball trajectory segment.
type TrajectoryResult struct {
	StartFrame int       `json:"startFrame"`
	EndFrame   int       `json:"endFrame"`
	Bounces    []Bounce  `json:"bounces"`
	SpeedKPH   float64   `json:"speedKph"`
	Confidence float64   `json:"confidence"`
}

// BridgeBackend defines the interface for ML backend communication.
// Implementations may use subprocess, gRPC, or in-process FFI.
type BridgeBackend interface {
	// Init initializes the backend with the given configuration.
	Init(config BridgeConfig) error

	// DetectBatch runs object detection on a batch of frames.
	DetectBatch(frames []Frame) ([]DetectionResult, error)

	// ClassifyStrokes classifies strokes from frame clips.
	ClassifyStrokes(clips []FrameClip) ([]StrokeResult, error)

	// AnalyzePlacements analyzes shot placements given detections and court data.
	AnalyzePlacements(detections []DetectionResult, court CourtData) ([]PlacementResult, error)

	// SegmentRallies segments a sequence of detections into rallies.
	SegmentRallies(detections []DetectionResult, fps float64) ([]RallyResult, error)

	// DetectCourt detects the court in a single frame.
	DetectCourt(frame Frame) (CourtData, error)

	// TrainModel starts a training run with the given video-CSV pairs and config.
	TrainModel(pairs []TrainingPair, config TrainingConfig) error

	// Close releases all resources held by the backend.
	Close()
}
