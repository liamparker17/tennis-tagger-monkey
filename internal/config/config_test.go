package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig_Defaults(t *testing.T) {
	// Loading from a non-existent file should return defaults.
	cfg, err := Load("nonexistent_config_file.yaml")
	if err != nil {
		t.Fatalf("Load returned error for missing file: %v", err)
	}

	if cfg.ConfigVersion != 1 {
		t.Errorf("ConfigVersion = %d, want 1", cfg.ConfigVersion)
	}
	if cfg.ModelsDir != "models" {
		t.Errorf("ModelsDir = %q, want %q", cfg.ModelsDir, "models")
	}
	if cfg.Device != "auto" {
		t.Errorf("Device = %q, want %q", cfg.Device, "auto")
	}
	if cfg.ActiveModel != "v1" {
		t.Errorf("ActiveModel = %q, want %q", cfg.ActiveModel, "v1")
	}
	if cfg.Pipeline.BatchSize != 32 {
		t.Errorf("Pipeline.BatchSize = %d, want 32", cfg.Pipeline.BatchSize)
	}
	if cfg.Pipeline.FrameSkip != 1 {
		t.Errorf("Pipeline.FrameSkip = %d, want 1", cfg.Pipeline.FrameSkip)
	}
	if !cfg.Pipeline.EnablePose {
		t.Error("Pipeline.EnablePose = false, want true")
	}
	if cfg.Pipeline.EnableScore {
		t.Error("Pipeline.EnableScore = true, want false")
	}
	if cfg.Pipeline.CheckpointEvery != 1000 {
		t.Errorf("Pipeline.CheckpointEvery = %d, want 1000", cfg.Pipeline.CheckpointEvery)
	}
	if cfg.Export.Format != "dartfish" {
		t.Errorf("Export.Format = %q, want %q", cfg.Export.Format, "dartfish")
	}
}

func TestLoadConfig_FromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")

	content := []byte(`configVersion: 2
modelsDir: /custom/models
device: cuda
activeModel: v3
pipeline:
  batchSize: 64
  frameSkip: 2
  enablePose: false
  enableScore: true
  checkpointEvery: 500
export:
  format: json
training:
  defaultEpochs: 20
  defaultBatchSize: 32
`)
	if err := os.WriteFile(path, content, 0644); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}

	if cfg.ConfigVersion != 2 {
		t.Errorf("ConfigVersion = %d, want 2", cfg.ConfigVersion)
	}
	if cfg.ModelsDir != "/custom/models" {
		t.Errorf("ModelsDir = %q, want %q", cfg.ModelsDir, "/custom/models")
	}
	if cfg.Device != "cuda" {
		t.Errorf("Device = %q, want %q", cfg.Device, "cuda")
	}
	if cfg.ActiveModel != "v3" {
		t.Errorf("ActiveModel = %q, want %q", cfg.ActiveModel, "v3")
	}
	if cfg.Pipeline.BatchSize != 64 {
		t.Errorf("Pipeline.BatchSize = %d, want 64", cfg.Pipeline.BatchSize)
	}
	if cfg.Pipeline.FrameSkip != 2 {
		t.Errorf("Pipeline.FrameSkip = %d, want 2", cfg.Pipeline.FrameSkip)
	}
	if cfg.Pipeline.EnablePose {
		t.Error("Pipeline.EnablePose = true, want false")
	}
	if !cfg.Pipeline.EnableScore {
		t.Error("Pipeline.EnableScore = false, want true")
	}
	if cfg.Pipeline.CheckpointEvery != 500 {
		t.Errorf("Pipeline.CheckpointEvery = %d, want 500", cfg.Pipeline.CheckpointEvery)
	}
	if cfg.Export.Format != "json" {
		t.Errorf("Export.Format = %q, want %q", cfg.Export.Format, "json")
	}
	if cfg.Training.DefaultEpochs != 20 {
		t.Errorf("Training.DefaultEpochs = %d, want 20", cfg.Training.DefaultEpochs)
	}
	if cfg.Training.DefaultBatchSize != 32 {
		t.Errorf("Training.DefaultBatchSize = %d, want 32", cfg.Training.DefaultBatchSize)
	}
}

func TestLoadConfig_MLDefaults(t *testing.T) {
	cfg, _ := Load("")
	if cfg.Pipeline.DetectorBackend != "yolo" {
		t.Errorf("expected 'yolo', got %s", cfg.Pipeline.DetectorBackend)
	}
	if cfg.Pipeline.ClassifierModel != "3dcnn" {
		t.Errorf("expected '3dcnn', got %s", cfg.Pipeline.ClassifierModel)
	}
}

func TestLoadConfig_LiveDefaults(t *testing.T) {
	cfg, _ := Load("")
	if cfg.Pipeline.LiveBatchSize != 4 {
		t.Errorf("Pipeline.LiveBatchSize = %d, want 4", cfg.Pipeline.LiveBatchSize)
	}
	if cfg.Pipeline.LiveSource != "0" {
		t.Errorf("Pipeline.LiveSource = %q, want %q", cfg.Pipeline.LiveSource, "0")
	}
}

func TestSaveConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")

	original := Default()
	original.Device = "cuda"
	original.Pipeline.BatchSize = 128
	original.Export.Format = "csv"

	if err := Save(original, path); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load returned error after Save: %v", err)
	}

	if loaded.Device != original.Device {
		t.Errorf("Device = %q, want %q", loaded.Device, original.Device)
	}
	if loaded.Pipeline.BatchSize != original.Pipeline.BatchSize {
		t.Errorf("Pipeline.BatchSize = %d, want %d", loaded.Pipeline.BatchSize, original.Pipeline.BatchSize)
	}
	if loaded.Export.Format != original.Export.Format {
		t.Errorf("Export.Format = %q, want %q", loaded.Export.Format, original.Export.Format)
	}
	if loaded.ConfigVersion != original.ConfigVersion {
		t.Errorf("ConfigVersion = %d, want %d", loaded.ConfigVersion, original.ConfigVersion)
	}
	if loaded.ActiveModel != original.ActiveModel {
		t.Errorf("ActiveModel = %q, want %q", loaded.ActiveModel, original.ActiveModel)
	}
	if loaded.Pipeline.EnablePose != original.Pipeline.EnablePose {
		t.Errorf("Pipeline.EnablePose = %v, want %v", loaded.Pipeline.EnablePose, original.Pipeline.EnablePose)
	}
}
