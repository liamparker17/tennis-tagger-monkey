package video

import (
	"math"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// testVideoPath returns the absolute path to the test video file.
func testVideoPath(t *testing.T) string {
	t.Helper()
	// Navigate from internal/video up to the project root, then into testdata.
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to determine test file path")
	}
	return filepath.Join(filepath.Dir(thisFile), "..", "..", "testdata", "sample_5s.mp4")
}

// ffmpegAvailable returns true if ffmpeg and ffprobe are on PATH.
func ffmpegAvailable() bool {
	_, errFF := exec.LookPath("ffmpeg")
	_, errFP := exec.LookPath("ffprobe")
	return errFF == nil && errFP == nil
}

func TestFFmpegAvailable(t *testing.T) {
	if !ffmpegAvailable() {
		t.Skip("ffmpeg/ffprobe not found in PATH; skipping")
	}
	t.Log("ffmpeg and ffprobe are available")
}

func TestOpen_InvalidPath(t *testing.T) {
	if !ffmpegAvailable() {
		t.Skip("ffmpeg/ffprobe not found in PATH; skipping")
	}

	_, err := Open("/nonexistent/path/to/video.mp4")
	if err == nil {
		t.Fatal("expected error for invalid path, got nil")
	}
}

func TestOpen_RealVideo(t *testing.T) {
	if !ffmpegAvailable() {
		t.Skip("ffmpeg/ffprobe not found in PATH; skipping")
	}

	videoPath := testVideoPath(t)

	vr, err := Open(videoPath)
	if err != nil {
		t.Fatalf("Open(%q) returned error: %v", videoPath, err)
	}
	defer vr.Close()

	meta := vr.Metadata()

	if meta.Width != 640 {
		t.Errorf("Width = %d, want 640", meta.Width)
	}
	if meta.Height != 480 {
		t.Errorf("Height = %d, want 480", meta.Height)
	}

	// FPS should be approximately 30.
	if math.Abs(meta.FPS-30.0) > 1.0 {
		t.Errorf("FPS = %f, want ~30.0", meta.FPS)
	}

	if meta.TotalFrames <= 0 {
		t.Errorf("TotalFrames = %d, want > 0", meta.TotalFrames)
	}

	if meta.Duration <= 0 {
		t.Errorf("Duration = %f, want > 0", meta.Duration)
	}
}

func TestExtractBatch(t *testing.T) {
	if !ffmpegAvailable() {
		t.Skip("ffmpeg/ffprobe not found in PATH; skipping")
	}

	videoPath := testVideoPath(t)

	vr, err := Open(videoPath)
	if err != nil {
		t.Fatalf("Open(%q) returned error: %v", videoPath, err)
	}
	defer vr.Close()

	frames, err := vr.ExtractBatch(0, 5)
	if err != nil {
		t.Fatalf("ExtractBatch(0, 5) returned error: %v", err)
	}

	if len(frames) != 5 {
		t.Fatalf("ExtractBatch returned %d frames, want 5", len(frames))
	}

	expectedSize := 640 * 480 * 3
	for i, f := range frames {
		if len(f.Data) != expectedSize {
			t.Errorf("frame[%d] Data length = %d, want %d", i, len(f.Data), expectedSize)
		}
		if f.Width != 640 {
			t.Errorf("frame[%d] Width = %d, want 640", i, f.Width)
		}
		if f.Height != 480 {
			t.Errorf("frame[%d] Height = %d, want 480", i, f.Height)
		}
	}
}

func TestExtractBatch_ZeroCount(t *testing.T) {
	if !ffmpegAvailable() {
		t.Skip("ffmpeg/ffprobe not found in PATH; skipping")
	}

	videoPath := testVideoPath(t)

	vr, err := Open(videoPath)
	if err != nil {
		t.Fatalf("Open(%q) returned error: %v", videoPath, err)
	}
	defer vr.Close()

	frames, err := vr.ExtractBatch(0, 0)
	if err != nil {
		t.Fatalf("ExtractBatch(0, 0) returned error: %v", err)
	}

	if len(frames) != 0 {
		t.Errorf("ExtractBatch(0, 0) returned %d frames, want 0", len(frames))
	}
}
