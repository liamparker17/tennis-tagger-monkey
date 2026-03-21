package video

import (
	"fmt"
	"io"
	"os/exec"
	"runtime"
	"sync"
	"sync/atomic"
)

// LiveReader captures frames from a live source (webcam or RTSP) via ffmpeg.
type LiveReader struct {
	cmd           *exec.Cmd
	stdout        io.ReadCloser
	width, height int
	fps           float64
	running       atomic.Bool
	mu            sync.Mutex
}

// OpenLive starts ffmpeg to capture frames from a live source.
// source can be a webcam index (e.g. "0") or an RTSP/URL string.
func OpenLive(source string, width, height int, fps float64) (*LiveReader, error) {
	args := buildLiveArgs(source, width, height, fps)

	cmd := exec.Command("ffmpeg", args...)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start ffmpeg live capture: %w", err)
	}

	lr := &LiveReader{
		cmd:    cmd,
		stdout: stdout,
		width:  width,
		height: height,
		fps:    fps,
	}
	lr.running.Store(true)

	return lr, nil
}

// ReadFrame reads one raw RGB frame from the ffmpeg stdout pipe.
// Returns io.EOF when the stream ends.
func (lr *LiveReader) ReadFrame() (Frame, error) {
	lr.mu.Lock()
	defer lr.mu.Unlock()

	frameSize := lr.width * lr.height * 3
	buf := make([]byte, frameSize)

	n, err := io.ReadFull(lr.stdout, buf)
	if err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			lr.running.Store(false)
			if n == 0 {
				return Frame{}, io.EOF
			}
			// Partial frame at end of stream — discard it.
			return Frame{}, io.EOF
		}
		lr.running.Store(false)
		return Frame{}, fmt.Errorf("failed to read frame: %w", err)
	}

	return Frame{
		Data:   buf,
		Width:  lr.width,
		Height: lr.height,
	}, nil
}

// ReadBatch reads up to n frames from the live source.
// Returns fewer frames if the stream ends before n frames are read.
func (lr *LiveReader) ReadBatch(n int) ([]Frame, error) {
	frames := make([]Frame, 0, n)
	for i := 0; i < n; i++ {
		f, err := lr.ReadFrame()
		if err != nil {
			if err == io.EOF {
				break
			}
			return frames, err
		}
		frames = append(frames, f)
	}
	return frames, nil
}

// Stop kills the ffmpeg process and releases resources.
func (lr *LiveReader) Stop() {
	lr.mu.Lock()
	defer lr.mu.Unlock()

	lr.running.Store(false)
	if lr.cmd != nil && lr.cmd.Process != nil {
		_ = lr.cmd.Process.Kill()
		_ = lr.cmd.Wait()
	}
}

// Running returns whether the live reader is still capturing.
func (lr *LiveReader) Running() bool {
	return lr.running.Load()
}

// Width returns the configured frame width.
func (lr *LiveReader) Width() int {
	return lr.width
}

// Height returns the configured frame height.
func (lr *LiveReader) Height() int {
	return lr.height
}

// FPS returns the configured frames per second.
func (lr *LiveReader) FPS() float64 {
	return lr.fps
}

// buildLiveArgs constructs the ffmpeg argument list for live capture.
// For webcam indices (1-2 character source), platform-specific input
// formats are used. For RTSP/URL sources, TCP transport is used.
func buildLiveArgs(source string, width, height int, fps float64) []string {
	sizeStr := fmt.Sprintf("%dx%d", width, height)
	fpsStr := fmt.Sprintf("%.2f", fps)

	var args []string

	if len(source) <= 2 {
		// Webcam index — use platform-specific input format.
		switch runtime.GOOS {
		case "windows":
			args = []string{
				"-f", "dshow",
				"-video_size", sizeStr,
				"-framerate", fpsStr,
				"-i", "video=" + source,
			}
		case "linux":
			args = []string{
				"-f", "v4l2",
				"-video_size", sizeStr,
				"-framerate", fpsStr,
				"-i", "/dev/video" + source,
			}
		case "darwin":
			args = []string{
				"-f", "avfoundation",
				"-video_size", sizeStr,
				"-framerate", fpsStr,
				"-i", source,
			}
		default:
			// Fallback: treat as direct input.
			args = []string{"-i", source}
		}
	} else {
		// RTSP or URL source.
		args = []string{
			"-rtsp_transport", "tcp",
			"-i", source,
		}
	}

	// Output: raw RGB24 frames to stdout.
	args = append(args,
		"-f", "rawvideo",
		"-pix_fmt", "rgb24",
		"-s", sizeStr,
		"-r", fpsStr,
		"-v", "error",
		"pipe:1",
	)

	return args
}
