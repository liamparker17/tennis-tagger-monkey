package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

// Config holds all application configuration.
type Config struct {
	ConfigVersion int            `yaml:"configVersion"`
	ModelsDir     string         `yaml:"modelsDir"`
	Device        string         `yaml:"device"`
	Pipeline      PipelineConfig `yaml:"pipeline"`
	Export        ExportConfig   `yaml:"export"`
	Training      TrainingConfig `yaml:"training"`
	ActiveModel   string         `yaml:"activeModel"`
}

// PipelineConfig holds pipeline-specific settings.
type PipelineConfig struct {
	BatchSize       int    `yaml:"batchSize"`
	FrameSkip       int    `yaml:"frameSkip"`
	EnablePose      bool   `yaml:"enablePose"`
	EnableScore     bool   `yaml:"enableScore"`
	CheckpointEvery int    `yaml:"checkpointEvery"`
	DetectorBackend string `yaml:"detectorBackend"`
	ClassifierModel string `yaml:"classifierModel"`
}

// ExportConfig holds export format settings.
type ExportConfig struct {
	Format string `yaml:"format"`
}

// TrainingConfig holds training-specific settings.
type TrainingConfig struct {
	DefaultEpochs    int `yaml:"defaultEpochs"`
	DefaultBatchSize int `yaml:"defaultBatchSize"`
}

// Default returns a Config populated with sensible defaults.
func Default() *Config {
	return &Config{
		ConfigVersion: 1,
		ModelsDir:     "models",
		Device:        "auto",
		ActiveModel:   "v1",
		Pipeline: PipelineConfig{
			BatchSize:       32,
			FrameSkip:       1,
			EnablePose:      true,
			EnableScore:     false,
			CheckpointEvery: 1000,
			DetectorBackend: "yolo",
			ClassifierModel: "3dcnn",
		},
		Export: ExportConfig{
			Format: "dartfish",
		},
		Training: TrainingConfig{
			DefaultEpochs:    10,
			DefaultBatchSize: 16,
		},
	}
}

// Load reads a YAML config file from path. If the file does not exist,
// it returns the default configuration.
func Load(path string) (*Config, error) {
	cfg := Default()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, err
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

// Save writes the config to a YAML file at path.
func Save(cfg *Config, path string) error {
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}
