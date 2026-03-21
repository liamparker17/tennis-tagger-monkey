package models

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
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

func sha256hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func TestDownloadModel_ChecksumVerification(t *testing.T) {
	content := []byte("known model content for checksum test")
	checksum := sha256hex(content)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(content)
	}))
	defer server.Close()

	destDir := t.TempDir()
	model := Model{
		File:   "test_model.pt",
		SHA256: checksum,
	}

	err := DownloadModel(model, server.URL+"/", destDir, nil)
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}

	// Verify file exists and content matches.
	got, err := os.ReadFile(filepath.Join(destDir, "test_model.pt"))
	if err != nil {
		t.Fatalf("failed to read downloaded file: %v", err)
	}
	if string(got) != string(content) {
		t.Errorf("file content mismatch: got %q, want %q", string(got), string(content))
	}
}

func TestDownloadModel_BadChecksum(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("bad data"))
	}))
	defer server.Close()

	destDir := t.TempDir()
	model := Model{
		File:   "bad_model.pt",
		SHA256: "0000000000000000000000000000000000000000000000000000000000000000",
	}

	err := DownloadModel(model, server.URL+"/", destDir, nil)
	if err == nil {
		t.Fatal("expected error for checksum mismatch, got nil")
	}

	if got := err.Error(); !contains(got, "checksum mismatch") {
		t.Errorf("expected error containing 'checksum mismatch', got: %s", got)
	}

	// Verify temp file was cleaned up.
	tmpPath := filepath.Join(destDir, "bad_model.pt.tmp")
	if _, err := os.Stat(tmpPath); !os.IsNotExist(err) {
		t.Errorf("expected temp file to be cleaned up, but it still exists")
	}

	// Verify final file was not created.
	finalPath := filepath.Join(destDir, "bad_model.pt")
	if _, err := os.Stat(finalPath); !os.IsNotExist(err) {
		t.Errorf("expected final file to not exist, but it does")
	}
}

func TestDownloadModel_ProgressCallback(t *testing.T) {
	// Create 1000 bytes of content.
	content := make([]byte, 1000)
	for i := range content {
		content[i] = byte(i % 256)
	}
	checksum := sha256hex(content)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000")
		w.Write(content)
	}))
	defer server.Close()

	destDir := t.TempDir()
	model := Model{
		File:   "progress_model.pt",
		SHA256: checksum,
	}

	var callCount atomic.Int64
	progress := func(downloaded, total int64) {
		callCount.Add(1)
	}

	err := DownloadModel(model, server.URL+"/", destDir, progress)
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}

	if callCount.Load() < 1 {
		t.Errorf("expected progress func to be called at least once, got %d calls", callCount.Load())
	}
}

func TestDownloadModel_SkipExisting(t *testing.T) {
	content := []byte("existing model data")
	checksum := sha256hex(content)

	var requestCount atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount.Add(1)
		w.Write([]byte("should not be downloaded"))
	}))
	defer server.Close()

	destDir := t.TempDir()

	// Pre-create the file with matching content/checksum.
	destPath := filepath.Join(destDir, "existing_model.pt")
	if err := os.WriteFile(destPath, content, 0644); err != nil {
		t.Fatalf("failed to create existing file: %v", err)
	}

	model := Model{
		File:   "existing_model.pt",
		SHA256: checksum,
	}

	err := DownloadModel(model, server.URL+"/", destDir, nil)
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}

	if requestCount.Load() != 0 {
		t.Errorf("expected no HTTP requests, but server received %d", requestCount.Load())
	}

	// Verify file content is unchanged.
	got, _ := os.ReadFile(destPath)
	if string(got) != string(content) {
		t.Errorf("file content was modified unexpectedly")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchSubstring(s, substr)
}

func searchSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
