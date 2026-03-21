package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Manifest describes the set of ML model files required by the application.
type Manifest struct {
	Version string           `json:"version"`
	Models  map[string]Model `json:"models"`
}

// Model describes a single ML model file.
type Model struct {
	File        string `json:"file"`
	Size        int64  `json:"size"`
	SHA256      string `json:"sha256"`
	Required    bool   `json:"required"`
	Description string `json:"description"`
}

// LoadManifest reads and parses a manifest JSON file from path.
func LoadManifest(path string) (*Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading manifest %s: %w", path, err)
	}

	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parsing manifest: %w", err)
	}

	return &m, nil
}

// CheckModels inspects modelsDir and returns a slice of required models
// whose files are missing from the directory.
func CheckModels(manifest *Manifest, modelsDir string) []Model {
	var missing []Model
	for _, model := range manifest.Models {
		if !model.Required {
			continue
		}
		path := filepath.Join(modelsDir, model.File)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			missing = append(missing, model)
		}
	}
	return missing
}
