package pipeline

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCheckpointRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "checkpoint.json")

	original := &Checkpoint{
		VideoPath:       "/path/to/video.mp4",
		ProcessedFrames: 500,
		TotalFrames:     1000,
	}

	if err := SaveCheckpoint(original, path); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}

	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}

	if loaded.Version != checkpointVersion {
		t.Errorf("Version = %d, want %d", loaded.Version, checkpointVersion)
	}
	if loaded.VideoPath != original.VideoPath {
		t.Errorf("VideoPath = %q, want %q", loaded.VideoPath, original.VideoPath)
	}
	if loaded.ProcessedFrames != original.ProcessedFrames {
		t.Errorf("ProcessedFrames = %d, want %d", loaded.ProcessedFrames, original.ProcessedFrames)
	}
	if loaded.TotalFrames != original.TotalFrames {
		t.Errorf("TotalFrames = %d, want %d", loaded.TotalFrames, original.TotalFrames)
	}
}

func TestCheckpointIncompatibleVersion(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "checkpoint.json")

	// Write a checkpoint with a future version directly.
	data := []byte(`{"version":99,"video_path":"test.mp4","processed_frames":10,"total_frames":100}`)
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, err := LoadCheckpoint(path)
	if err == nil {
		t.Fatal("expected error for incompatible version, got nil")
	}
}
