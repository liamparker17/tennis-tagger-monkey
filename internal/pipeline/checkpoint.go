package pipeline

import (
	"encoding/json"
	"fmt"
	"os"
)

const checkpointVersion = 1

// Checkpoint holds the state needed to resume a pipeline run.
type Checkpoint struct {
	Version         int    `json:"version"`
	VideoPath       string `json:"video_path"`
	ProcessedFrames int    `json:"processed_frames"`
	TotalFrames     int    `json:"total_frames"`
}

// SaveCheckpoint marshals a Checkpoint to JSON and writes it to path.
// It always sets the Version field to the current checkpointVersion before saving.
func SaveCheckpoint(cp *Checkpoint, path string) error {
	cp.Version = checkpointVersion

	data, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal checkpoint: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write checkpoint: %w", err)
	}

	return nil
}

// LoadCheckpoint reads a checkpoint file from path, unmarshals it, and
// validates that the version is compatible with the current checkpointVersion.
func LoadCheckpoint(path string) (*Checkpoint, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read checkpoint: %w", err)
	}

	var cp Checkpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return nil, fmt.Errorf("unmarshal checkpoint: %w", err)
	}

	if cp.Version != checkpointVersion {
		return nil, fmt.Errorf("incompatible checkpoint version: got %d, want %d", cp.Version, checkpointVersion)
	}

	return &cp, nil
}
