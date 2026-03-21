package models

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCheckModels_AllMissing(t *testing.T) {
	// Create an empty temp directory
	dir := t.TempDir()

	manifest := &Manifest{
		Version: "1.0",
		Models: map[string]Model{
			"detector": {
				File:     "yolov8x.pt",
				Size:     136000000,
				Required: true,
			},
		},
	}

	missing := CheckModels(manifest, dir)
	if len(missing) != 1 {
		t.Fatalf("expected 1 missing model, got %d", len(missing))
	}
	if missing[0].File != "yolov8x.pt" {
		t.Errorf("expected missing file yolov8x.pt, got %s", missing[0].File)
	}
}

func TestCheckModels_NoneMissing(t *testing.T) {
	dir := t.TempDir()

	// Create a fake .pt file
	fakePath := filepath.Join(dir, "yolov8x.pt")
	if err := os.WriteFile(fakePath, []byte("fake model data"), 0644); err != nil {
		t.Fatalf("failed to create fake model file: %v", err)
	}

	manifest := &Manifest{
		Version: "1.0",
		Models: map[string]Model{
			"detector": {
				File:     "yolov8x.pt",
				Size:     136000000,
				Required: true,
			},
		},
	}

	missing := CheckModels(manifest, dir)
	if len(missing) != 0 {
		t.Fatalf("expected 0 missing models, got %d", len(missing))
	}
}
