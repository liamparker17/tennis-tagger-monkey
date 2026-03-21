package models

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// ProgressFunc is called during download with bytes downloaded and total size.
type ProgressFunc func(downloaded, total int64)

// progressWriter wraps an io.Writer and calls a ProgressFunc on each Write.
type progressWriter struct {
	w          io.Writer
	progress   ProgressFunc
	total      int64
	downloaded int64
}

func (pw *progressWriter) Write(p []byte) (int, error) {
	n, err := pw.w.Write(p)
	pw.downloaded += int64(n)
	if pw.progress != nil {
		pw.progress(pw.downloaded, pw.total)
	}
	return n, err
}

// fileMatchesChecksum returns true if the file at path exists and its SHA256
// matches the expected hex-encoded checksum.
func fileMatchesChecksum(path, expectedSHA256 string) bool {
	if expectedSHA256 == "" {
		return false
	}
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false
	}
	actual := hex.EncodeToString(h.Sum(nil))
	return actual == expectedSHA256
}

// DownloadModel downloads a single model to destDir with checksum verification.
// It skips the download if the file already exists with a matching checksum.
// The file is first written to a temporary path and atomically renamed on success.
func DownloadModel(model Model, baseURL, destDir string, progress ProgressFunc) error {
	destPath := filepath.Join(destDir, model.File)

	// Skip if file already exists with matching checksum.
	if model.SHA256 != "" && fileMatchesChecksum(destPath, model.SHA256) {
		return nil
	}

	url := baseURL + model.File
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("downloading %s: %w", model.File, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("downloading %s: HTTP %d", model.File, resp.StatusCode)
	}

	tmpPath := destPath + ".tmp"
	tmpFile, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("creating temp file %s: %w", tmpPath, err)
	}
	defer func() {
		tmpFile.Close()
		// Clean up temp file on error (best effort).
		os.Remove(tmpPath)
	}()

	hasher := sha256.New()

	var dest io.Writer = io.MultiWriter(tmpFile, hasher)

	if progress != nil {
		dest = &progressWriter{
			w:        dest,
			progress: progress,
			total:    resp.ContentLength,
		}
	}

	if _, err := io.Copy(dest, resp.Body); err != nil {
		return fmt.Errorf("writing %s: %w", model.File, err)
	}

	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("closing temp file %s: %w", tmpPath, err)
	}

	// Verify checksum if expected.
	if model.SHA256 != "" {
		actual := hex.EncodeToString(hasher.Sum(nil))
		if actual != model.SHA256 {
			os.Remove(tmpPath)
			return fmt.Errorf("checksum mismatch for %s: expected %s, got %s", model.File, model.SHA256, actual)
		}
	}

	// Atomic rename.
	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("renaming %s to %s: %w", tmpPath, destPath, err)
	}

	return nil
}

// DownloadMissing downloads all missing required models from baseURL into modelsDir.
func DownloadMissing(manifest *Manifest, baseURL, modelsDir string, progress ProgressFunc) error {
	missing := CheckModels(manifest, modelsDir)
	if len(missing) == 0 {
		return nil
	}

	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		return fmt.Errorf("creating models directory %s: %w", modelsDir, err)
	}

	for _, model := range missing {
		if err := DownloadModel(model, baseURL, modelsDir, progress); err != nil {
			return err
		}
	}

	return nil
}
