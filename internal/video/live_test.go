package video

import (
	"os/exec"
	"runtime"
	"strings"
	"testing"
)

func TestBuildLiveArgs_RTSP(t *testing.T) {
	args := buildLiveArgs("rtsp://192.168.1.100:554/stream", 1280, 720, 30.0)

	joined := strings.Join(args, " ")

	// Must contain tcp transport for RTSP.
	if !strings.Contains(joined, "tcp") {
		t.Errorf("expected args to contain 'tcp', got: %v", args)
	}

	// Must end with pipe:1.
	last := args[len(args)-1]
	if last != "pipe:1" {
		t.Errorf("expected last arg to be 'pipe:1', got: %s", last)
	}

	// Must contain the RTSP URL.
	if !strings.Contains(joined, "rtsp://192.168.1.100:554/stream") {
		t.Errorf("expected args to contain RTSP URL, got: %v", args)
	}
}

func TestBuildLiveArgs_Webcam(t *testing.T) {
	args := buildLiveArgs("0", 640, 480, 25.0)

	joined := strings.Join(args, " ")

	// Verify platform-specific input format is detected.
	switch runtime.GOOS {
	case "windows":
		if !strings.Contains(joined, "dshow") {
			t.Errorf("expected 'dshow' on windows, got: %v", args)
		}
		if !strings.Contains(joined, "video=0") {
			t.Errorf("expected 'video=0' on windows, got: %v", args)
		}
	case "linux":
		if !strings.Contains(joined, "v4l2") {
			t.Errorf("expected 'v4l2' on linux, got: %v", args)
		}
		if !strings.Contains(joined, "/dev/video0") {
			t.Errorf("expected '/dev/video0' on linux, got: %v", args)
		}
	case "darwin":
		if !strings.Contains(joined, "avfoundation") {
			t.Errorf("expected 'avfoundation' on darwin, got: %v", args)
		}
	}

	// Must contain video_size.
	if !strings.Contains(joined, "640x480") {
		t.Errorf("expected args to contain '640x480', got: %v", args)
	}

	// Must end with pipe:1.
	last := args[len(args)-1]
	if last != "pipe:1" {
		t.Errorf("expected last arg to be 'pipe:1', got: %s", last)
	}
}

func TestOpenLive_InvalidSource(t *testing.T) {
	// Skip if ffmpeg is not available.
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not found, skipping live source test")
	}

	// Try to open an invalid RTSP source. This should either fail to start
	// or fail on the first read. It must not panic.
	lr, err := OpenLive("rtsp://invalid.invalid:554/nonexistent", 320, 240, 15.0)
	if err != nil {
		// Expected — ffmpeg could not connect.
		t.Logf("OpenLive returned error as expected: %v", err)
		return
	}

	// If OpenLive succeeded (ffmpeg started but hasn't failed yet),
	// try reading a frame — it should fail or return EOF.
	defer lr.Stop()

	_, readErr := lr.ReadFrame()
	if readErr != nil {
		t.Logf("ReadFrame returned error as expected: %v", readErr)
		return
	}

	t.Log("OpenLive and ReadFrame did not return errors; ffmpeg may have buffered")
}
