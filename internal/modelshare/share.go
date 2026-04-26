// Package modelshare packages a trained PyTorch checkpoint into a portable
// folder (manifest + weights) so two collaborators can swap models via USB.
package modelshare

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	CurrentSchema  = 1
	manifestFile   = "manifest.json"
	defaultWeights = "weights.pt"
)

// Manifest is the small JSON file at the root of every shared bundle.
type Manifest struct {
	SchemaVersion   int    `json:"schema_version"`
	Author          string `json:"author"`
	CreatedAt       string `json:"created_at"` // RFC3339
	ModelKind       string `json:"model_kind"` // e.g. "point_model"
	WeightsFilename string `json:"weights_filename"`
	SHA256          string `json:"sha256"`
	SizeBytes       int64  `json:"size_bytes"`
	Notes           string `json:"notes,omitempty"`
}

// ExportModel copies srcPt + a manifest into outDir.
func ExportModel(srcPt, outDir, author, modelKind, notes string) (*Manifest, error) {
	src, err := os.Open(srcPt)
	if err != nil {
		return nil, fmt.Errorf("open weights: %w", err)
	}
	defer src.Close()
	info, err := src.Stat()
	if err != nil {
		return nil, err
	}

	if entries, _ := os.ReadDir(outDir); len(entries) > 0 {
		return nil, fmt.Errorf("output directory %q is not empty", outDir)
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return nil, err
	}

	dst, err := os.Create(filepath.Join(outDir, defaultWeights))
	if err != nil {
		return nil, err
	}
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(dst, h), src); err != nil {
		_ = dst.Close()
		return nil, err
	}
	if err := dst.Close(); err != nil {
		return nil, err
	}

	if author == "" {
		author = "anonymous"
	}
	if modelKind == "" {
		modelKind = "point_model"
	}
	m := &Manifest{
		SchemaVersion:   CurrentSchema,
		Author:          author,
		CreatedAt:       time.Now().UTC().Format(time.RFC3339),
		ModelKind:       modelKind,
		WeightsFilename: defaultWeights,
		SHA256:          hex.EncodeToString(h.Sum(nil)),
		SizeBytes:       info.Size(),
		Notes:           notes,
	}
	data, _ := json.MarshalIndent(m, "", "  ")
	if err := os.WriteFile(filepath.Join(outDir, manifestFile), data, 0o644); err != nil {
		return nil, err
	}
	return m, nil
}

// LoadManifest reads a bundle's manifest.json.
func LoadManifest(bundleDir string) (*Manifest, error) {
	data, err := os.ReadFile(filepath.Join(bundleDir, manifestFile))
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	if m.SchemaVersion > CurrentSchema {
		return nil, fmt.Errorf("bundle is from a newer Tennis Tagger version (schema v%d)", m.SchemaVersion)
	}
	if m.WeightsFilename == "" {
		return nil, fmt.Errorf("manifest missing weights_filename")
	}
	return &m, nil
}

// VerifyChecksum recomputes SHA256 of the bundled weights and compares it
// to the manifest. Returns nil on match.
func VerifyChecksum(bundleDir string, m *Manifest) error {
	if m.SHA256 == "" {
		return nil // older bundles may lack one; skip silently
	}
	f, err := os.Open(filepath.Join(bundleDir, m.WeightsFilename))
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(got, m.SHA256) {
		return fmt.Errorf("checksum mismatch — bundle is corrupted or was modified")
	}
	return nil
}

// CopyWeightsTo extracts the bundle's weights to dstPath, overwriting it.
func CopyWeightsTo(bundleDir string, m *Manifest, dstPath string) error {
	src, err := os.Open(filepath.Join(bundleDir, m.WeightsFilename))
	if err != nil {
		return err
	}
	defer src.Close()
	if err := os.MkdirAll(filepath.Dir(dstPath), 0o755); err != nil {
		return err
	}
	dst, err := os.Create(dstPath)
	if err != nil {
		return err
	}
	defer dst.Close()
	_, err = io.Copy(dst, src)
	return err
}
