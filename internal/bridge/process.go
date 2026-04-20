package bridge

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
)

// processCaller communicates with a Python subprocess via stdin/stdout JSON-RPC.
type processCaller struct {
	pythonPath string
	cmd        *exec.Cmd
	stdin      io.WriteCloser
	scanner    *bufio.Scanner
	mu         sync.Mutex
	nextID     atomic.Int64
}

// init starts the Python subprocess (if not running) and sends the __init__ JSON-RPC request.
func (p *processCaller) init(config BridgeConfig) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.cmd == nil {
		if err := p.startProcess(); err != nil {
			return fmt.Errorf("processCaller.init: failed to start process: %w", err)
		}
	}

	payload, err := json.Marshal(map[string]interface{}{
		"ModelsDir": config.ModelsDir,
		"Device":    config.Device,
	})
	if err != nil {
		return fmt.Errorf("processCaller.init: marshal config: %w", err)
	}

	result, err := p.callLocked("__init__", json.RawMessage(payload))
	if err != nil {
		return fmt.Errorf("processCaller.init: %w", err)
	}
	_ = result
	return nil
}

// startProcess launches the Python bridge_server subprocess.
// Must be called with p.mu held.
func (p *processCaller) startProcess() error {
	p.cmd = exec.Command(p.pythonPath, "-m", "ml.bridge_server")
	// Working directory is the project root (where ml/ lives).
	// exec.Command uses the current process working directory by default,
	// which is correct when run from the project root.

	var err error
	p.stdin, err = p.cmd.StdinPipe()
	if err != nil {
		p.cmd = nil
		return fmt.Errorf("stdin pipe: %w", err)
	}

	stdout, err := p.cmd.StdoutPipe()
	if err != nil {
		p.cmd = nil
		return fmt.Errorf("stdout pipe: %w", err)
	}

	// stderr inherits parent process stderr for Python logging
	p.cmd.Stderr = nil // nil means inherit

	// Scanner with 512 MB buffer for large base64 encoded frames
	p.scanner = bufio.NewScanner(stdout)
	const maxBuf = 512 * 1024 * 1024
	p.scanner.Buffer(make([]byte, 64*1024), maxBuf)

	if err := p.cmd.Start(); err != nil {
		p.cmd = nil
		return fmt.Errorf("start process: %w", err)
	}

	// Wait for the ready signal
	if !p.scanner.Scan() {
		err := p.scanner.Err()
		if err == nil {
			err = fmt.Errorf("subprocess closed stdout before sending ready signal")
		}
		_ = p.cmd.Process.Kill()
		p.cmd = nil
		return fmt.Errorf("waiting for ready: %w", err)
	}

	var ready struct {
		Ready bool `json:"ready"`
	}
	if err := json.Unmarshal(p.scanner.Bytes(), &ready); err != nil || !ready.Ready {
		_ = p.cmd.Process.Kill()
		p.cmd = nil
		return fmt.Errorf("invalid ready signal: %s", p.scanner.Text())
	}

	return nil
}

// call sends a JSON-RPC request and reads the response.
func (p *processCaller) call(method string, payload json.RawMessage) (json.RawMessage, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Lazy start: if the process isn't running yet, start it
	if p.cmd == nil {
		if err := p.startProcess(); err != nil {
			return nil, fmt.Errorf("processCaller.call: failed to start process: %w", err)
		}
	}

	return p.callLocked(method, payload)
}

// callLocked sends a JSON-RPC request and reads the response.
// Must be called with p.mu held.
func (p *processCaller) callLocked(method string, payload json.RawMessage) (json.RawMessage, error) {
	id := p.nextID.Add(1)

	req := map[string]interface{}{
		"method": method,
		"params": payload,
		"id":     id,
	}

	reqBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// Write request as a single line
	reqBytes = append(reqBytes, '\n')
	if _, err := p.stdin.Write(reqBytes); err != nil {
		return nil, fmt.Errorf("write to subprocess: %w", err)
	}

	// Read response line
	if !p.scanner.Scan() {
		err := p.scanner.Err()
		if err == nil {
			err = fmt.Errorf("subprocess closed stdout")
		}
		return nil, fmt.Errorf("read response: %w", err)
	}

	var resp struct {
		Result json.RawMessage `json:"result"`
		Error  *string         `json:"error"`
		ID     int64           `json:"id"`
	}

	if err := json.Unmarshal(p.scanner.Bytes(), &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("python error: %s", *resp.Error)
	}

	return resp.Result, nil
}

// close shuts down the Python subprocess.
func (p *processCaller) close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.cmd == nil {
		return
	}

	_ = p.stdin.Close()
	_ = p.cmd.Process.Kill()
	_ = p.cmd.Wait()
	p.cmd = nil
}

// ProcessBridge implements BridgeBackend by communicating with a Python subprocess.
// Frame data is transferred via shared memory files (no base64 serialization).
type ProcessBridge struct {
	worker *Worker
	shm    *SharedMemBuffer
}

// Compile-time interface check.
var _ BridgeBackend = (*ProcessBridge)(nil)

// NewProcessBridge creates a new ProcessBridge that will spawn a Python subprocess.
func NewProcessBridge(pythonPath string) *ProcessBridge {
	caller := &processCaller{pythonPath: pythonPath}
	return &ProcessBridge{
		worker: NewWorker(caller),
	}
}

// Init initializes the Python subprocess and ML modules.
func (b *ProcessBridge) Init(config BridgeConfig) error {
	payload, err := json.Marshal(config)
	if err != nil {
		return fmt.Errorf("ProcessBridge.Init: marshal config: %w", err)
	}

	_, err = b.worker.Call("__init__", payload)
	return err
}

// DetectBatch runs object detection on a batch of frames.
// Uses shared memory file for frame data transfer (no base64 serialization).
func (b *ProcessBridge) DetectBatch(frames []Frame) ([]DetectionResult, error) {
	// Lazy init shared memory buffer
	if b.shm == nil {
		var err error
		b.shm, err = NewSharedMemBuffer(os.TempDir())
		if err != nil {
			return nil, fmt.Errorf("DetectBatch: create shm: %w", err)
		}
	}

	// Write raw frame bytes to shared memory file
	shmPath, metas, err := b.shm.WriteBatch(frames)
	if err != nil {
		return nil, fmt.Errorf("DetectBatch: write shm: %w", err)
	}

	// Send only metadata (path + offsets) — Python reads pixels from file
	payload, err := json.Marshal(map[string]interface{}{
		"shm_path": shmPath,
		"frames":   metas,
	})
	if err != nil {
		return nil, fmt.Errorf("DetectBatch: marshal: %w", err)
	}

	result, err := b.worker.Call("detect_batch", payload)
	if err != nil {
		return nil, err
	}

	return adaptDetectionResults(result)
}

// TrackNetBatch runs TrackNet ball detection on a batch of frames.
// Uses shared memory file for frame data transfer (no base64 serialization).
// Returns one BallPosition per frame where a ball was detected; frames with
// no detection are omitted from the returned slice.
// AnalyzePlacements analyzes shot placements given detections and court data.
func (b *ProcessBridge) AnalyzePlacements(detections []DetectionResult, court CourtData) ([]PlacementResult, error) {
	payload, err := json.Marshal(map[string]interface{}{
		"detections": detections,
		"court":      court,
	})
	if err != nil {
		return nil, fmt.Errorf("AnalyzePlacements: marshal: %w", err)
	}

	result, err := b.worker.Call("analyze_placements", payload)
	if err != nil {
		return nil, err
	}

	var placements []PlacementResult
	if err := json.Unmarshal(result, &placements); err != nil {
		return nil, fmt.Errorf("AnalyzePlacements: unmarshal: %w", err)
	}
	return placements, nil
}

// SegmentRallies segments a sequence of detections into rallies.
func (b *ProcessBridge) SegmentRallies(detections []DetectionResult, fps float64) ([]RallyResult, error) {
	payload, err := json.Marshal(map[string]interface{}{
		"detections": detections,
		"fps":        fps,
	})
	if err != nil {
		return nil, fmt.Errorf("SegmentRallies: marshal: %w", err)
	}

	result, err := b.worker.Call("segment_rallies", payload)
	if err != nil {
		return nil, err
	}

	var rallies []RallyResult
	if err := json.Unmarshal(result, &rallies); err != nil {
		return nil, fmt.Errorf("SegmentRallies: unmarshal: %w", err)
	}
	return rallies, nil
}

// DetectCourt detects the court in a single frame.
// Uses shared memory for the frame data.
func (b *ProcessBridge) DetectCourt(frame Frame) (CourtData, error) {
	if b.shm == nil {
		var err error
		b.shm, err = NewSharedMemBuffer(os.TempDir())
		if err != nil {
			return CourtData{}, fmt.Errorf("DetectCourt: create shm: %w", err)
		}
	}

	shmPath, metas, err := b.shm.WriteBatch([]Frame{frame})
	if err != nil {
		return CourtData{}, fmt.Errorf("DetectCourt: write shm: %w", err)
	}

	payload, err := json.Marshal(map[string]interface{}{
		"frame": map[string]interface{}{
			"shm_path": shmPath,
			"offset":   metas[0].Offset,
			"width":    metas[0].Width,
			"height":   metas[0].Height,
			"size":     metas[0].Size,
		},
	})
	if err != nil {
		return CourtData{}, fmt.Errorf("DetectCourt: marshal: %w", err)
	}

	result, err := b.worker.Call("detect_court", payload)
	if err != nil {
		return CourtData{}, err
	}

	var court CourtData
	if err := json.Unmarshal(result, &court); err != nil {
		return CourtData{}, fmt.Errorf("DetectCourt: unmarshal: %w", err)
	}
	return court, nil
}

// TrainModel starts a training run with the given video-CSV pairs and config.
func (b *ProcessBridge) TrainModel(pairs []TrainingPair, config TrainingConfig) error {
	payload, err := json.Marshal(map[string]interface{}{
		"pairs":  pairs,
		"config": config,
	})
	if err != nil {
		return fmt.Errorf("TrainModel: marshal: %w", err)
	}

	_, err = b.worker.Call("train", payload)
	return err
}

// FineTune sends user corrections to the Python fine_tune endpoint.
func (b *ProcessBridge) FineTune(corrections []CorrectionData, config TrainingConfig) error {
	payload, err := json.Marshal(map[string]interface{}{
		"corrections": corrections,
		"config":      config,
	})
	if err != nil {
		return fmt.Errorf("FineTune: marshal: %w", err)
	}

	_, err = b.worker.Call("fine_tune", payload)
	return err
}

// FitTrajectories sends a list of ball positions to the Python trajectory fitter
// and returns fitted TrajectoryResult values with bounce detection and in/out calls.
func (b *ProcessBridge) FitTrajectories(positions []BallPosition, court CourtData, fps float64) ([]TrajectoryResult, error) {
	// Convert BallPosition to the shape Python expects (camelCase keys).
	type ballPos struct {
		X          float64 `json:"x"`
		Y          float64 `json:"y"`
		Confidence float64 `json:"confidence"`
		FrameIndex int     `json:"frameIndex"`
	}

	pyPositions := make([]ballPos, len(positions))
	for i, p := range positions {
		pyPositions[i] = ballPos{
			X:          p.X,
			Y:          p.Y,
			Confidence: p.Confidence,
			FrameIndex: p.FrameIndex,
		}
	}

	payload, err := json.Marshal(map[string]interface{}{
		"ball_positions": pyPositions,
		"court":          court,
		"fps":            fps,
	})
	if err != nil {
		return nil, fmt.Errorf("FitTrajectories: marshal: %w", err)
	}

	result, err := b.worker.Call("fit_trajectories", payload)
	if err != nil {
		return nil, err
	}

	var trajectories []TrajectoryResult
	if err := json.Unmarshal(result, &trajectories); err != nil {
		return nil, fmt.Errorf("FitTrajectories: unmarshal: %w", err)
	}
	return trajectories, nil
}

// SetManualCourt sends user-clicked corners to the Python analyzer.
// Skips auto court detection for the rest of the session.
func (b *ProcessBridge) SetManualCourt(setup MatchSetup) error {
	if len(setup.CornersPixel) != 4 {
		return fmt.Errorf("SetManualCourt: expected 4 corners, got %d", len(setup.CornersPixel))
	}
	payload, _ := json.Marshal(map[string]interface{}{
		"corners_pixel": setup.CornersPixel,
		"frame_width":   setup.FrameWidth,
		"frame_height":  setup.FrameHeight,
		"near_player":   setup.NearPlayer,
		"far_player":    setup.FarPlayer,
	})
	_, err := b.worker.Call("set_manual_court", payload)
	return err
}

// Close releases resources held by the process bridge.
// Safe to call multiple times.
func (b *ProcessBridge) Close() {
	if b.shm != nil {
		b.shm.Close()
		b.shm = nil
	}
	if b.worker != nil {
		b.worker.Stop()
		b.worker.backend.close()
	}
}

// toFloat safely converts an interface{} value to float64.
// Handles both float64 and json.Number types from JSON unmarshalling.
func toFloat(v interface{}) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case json.Number:
		f, _ := n.Float64()
		return f
	default:
		return 0
	}
}

// adaptDetectionResults converts Python detector output to []DetectionResult.
// Python returns: [{"players": [{"bbox": [x1,y1,x2,y2], "confidence": ...}], "ball": {"bbox": [...], "confidence": ...} | null}]
func adaptDetectionResults(raw json.RawMessage) ([]DetectionResult, error) {
	var pyResults []map[string]interface{}
	if err := json.Unmarshal(raw, &pyResults); err != nil {
		return nil, fmt.Errorf("adaptDetectionResults: unmarshal: %w", err)
	}

	results := make([]DetectionResult, len(pyResults))
	for i, pr := range pyResults {
		results[i].FrameIndex = i

		// Parse players
		if playersRaw, ok := pr["players"]; ok {
			if players, ok := playersRaw.([]interface{}); ok {
				for _, p := range players {
					pm, ok := p.(map[string]interface{})
					if !ok {
						continue
					}
					bbox := extractBBox(pm)
					results[i].Players = append(results[i].Players, bbox)
				}
			}
		}

		// Parse ball
		if ballRaw, ok := pr["ball"]; ok && ballRaw != nil {
			if bm, ok := ballRaw.(map[string]interface{}); ok {
				bbox := extractBBox(bm)
				results[i].Ball = &bbox
			}
		}
	}

	return results, nil
}

// extractBBox extracts a BBox from a Python detection dict.
// Supports {"bbox": [x1,y1,x2,y2], "confidence": ...} format.
func extractBBox(m map[string]interface{}) BBox {
	var bbox BBox

	if bboxRaw, ok := m["bbox"]; ok {
		if coords, ok := bboxRaw.([]interface{}); ok && len(coords) >= 4 {
			bbox.X1 = toFloat(coords[0])
			bbox.Y1 = toFloat(coords[1])
			bbox.X2 = toFloat(coords[2])
			bbox.Y2 = toFloat(coords[3])
		}
	}

	if conf, ok := m["confidence"]; ok {
		bbox.Confidence = toFloat(conf)
	}

	return bbox
}
