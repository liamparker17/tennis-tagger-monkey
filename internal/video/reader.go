package video

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// ffprobeOutput represents the JSON output from ffprobe.
type ffprobeOutput struct {
	Streams []ffprobeStream `json:"streams"`
	Format  ffprobeFormat   `json:"format"`
}

type ffprobeStream struct {
	Width      int    `json:"width"`
	Height     int    `json:"height"`
	RFrameRate string `json:"r_frame_rate"`
	NbFrames   string `json:"nb_frames"`
	CodecType  string `json:"codec_type"`
}

type ffprobeFormat struct {
	Duration string `json:"duration"`
}

// VideoReader reads video frames using ffmpeg/ffprobe.
type VideoReader struct {
	meta VideoMeta
}

// Open probes the video file at path using ffprobe and returns a VideoReader.
func Open(path string) (*VideoReader, error) {
	cmd := exec.Command(
		"ffprobe",
		"-v", "quiet",
		"-print_format", "json",
		"-show_streams",
		"-show_format",
		path,
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffprobe failed: %w: %s", err, stderr.String())
	}

	var probe ffprobeOutput
	if err := json.Unmarshal(stdout.Bytes(), &probe); err != nil {
		return nil, fmt.Errorf("failed to parse ffprobe output: %w", err)
	}

	// Find the video stream.
	var vs *ffprobeStream
	for i := range probe.Streams {
		if probe.Streams[i].CodecType == "video" {
			vs = &probe.Streams[i]
			break
		}
	}
	if vs == nil {
		return nil, fmt.Errorf("no video stream found in %s", path)
	}

	// Parse r_frame_rate (e.g. "30/1" or "30000/1001").
	fps, err := parseFrameRate(vs.RFrameRate)
	if err != nil {
		return nil, fmt.Errorf("failed to parse frame rate %q: %w", vs.RFrameRate, err)
	}

	// Parse nb_frames.
	totalFrames := 0
	if vs.NbFrames != "" && vs.NbFrames != "N/A" {
		totalFrames, _ = strconv.Atoi(vs.NbFrames)
	}

	// Parse duration from format (more reliable than stream duration).
	duration := 0.0
	if probe.Format.Duration != "" {
		duration, _ = strconv.ParseFloat(probe.Format.Duration, 64)
	}

	// If nb_frames was not available, estimate from duration and FPS.
	if totalFrames == 0 && duration > 0 && fps > 0 {
		totalFrames = int(duration * fps)
	}

	// Compute scaled output dimensions (scale=640:-2 in ffmpeg).
	outWidth := 640
	outHeight := vs.Height
	if vs.Width > 0 {
		outHeight = vs.Height * outWidth / vs.Width
		if outHeight%2 != 0 {
			outHeight++ // round to even (ffmpeg -2 flag)
		}
	}

	return &VideoReader{
		meta: VideoMeta{
			Path:         path,
			FPS:          fps,
			Width:        outWidth,
			Height:       outHeight,
			NativeWidth:  vs.Width,
			NativeHeight: vs.Height,
			TotalFrames:  totalFrames,
			Duration:     duration,
		},
	}, nil
}

// Metadata returns the video metadata obtained during Open.
func (vr *VideoReader) Metadata() VideoMeta {
	return vr.meta
}

// ExtractBatch extracts count raw RGB frames starting at frame index start.
// It uses ffmpeg to seek to the appropriate timestamp and decode frames to raw RGB24.
func (vr *VideoReader) ExtractBatch(start, count int) ([]Frame, error) {
	if count <= 0 {
		return nil, nil
	}

	// Calculate seek time from frame index and FPS.
	seekSec := 0.0
	if vr.meta.FPS > 0 {
		seekSec = float64(start) / vr.meta.FPS
	}

	// Scale to 640px width (YOLO resizes to 640 internally anyway).
	// This dramatically reduces memory and data transfer.
	// -2 for height means "maintain aspect ratio, round to even".
	cmd := exec.Command(
		"ffmpeg",
		"-ss", fmt.Sprintf("%.6f", seekSec),
		"-i", vr.meta.Path,
		"-frames:v", strconv.Itoa(count),
		"-vf", "scale=640:-2",
		"-f", "rawvideo",
		"-pix_fmt", "bgr24",
		"-v", "error",
		"pipe:1",
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg extract failed: %w: %s", err, stderr.String())
	}

	frameSize := vr.meta.Width * vr.meta.Height * 3
	if frameSize == 0 {
		return nil, fmt.Errorf("invalid frame dimensions: %dx%d", vr.meta.Width, vr.meta.Height)
	}

	raw := stdout.Bytes()
	numFrames := len(raw) / frameSize
	if numFrames == 0 && len(raw) > 0 {
		return nil, fmt.Errorf("incomplete frame data: got %d bytes, expected at least %d", len(raw), frameSize)
	}

	frames := make([]Frame, numFrames)
	for i := 0; i < numFrames; i++ {
		offset := i * frameSize
		// Copy frame data so the underlying buffer can be GC'd.
		data := make([]byte, frameSize)
		copy(data, raw[offset:offset+frameSize])
		frames[i] = Frame{
			Data:   data,
			Width:  vr.meta.Width,
			Height: vr.meta.Height,
		}
	}

	return frames, nil
}

// ExtractNative extracts a single frame at native resolution (no downscale).
// Used for court detection where higher resolution improves line detection.
func (vr *VideoReader) ExtractNative(frameIdx int) (Frame, error) {
	seekSec := 0.0
	if vr.meta.FPS > 0 {
		seekSec = float64(frameIdx) / vr.meta.FPS
	}

	cmd := exec.Command(
		"ffmpeg",
		"-ss", fmt.Sprintf("%.6f", seekSec),
		"-i", vr.meta.Path,
		"-frames:v", "1",
		"-f", "rawvideo",
		"-pix_fmt", "bgr24",
		"-v", "error",
		"pipe:1",
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return Frame{}, fmt.Errorf("ffmpeg native extract: %w: %s", err, stderr.String())
	}

	frameSize := vr.meta.NativeWidth * vr.meta.NativeHeight * 3
	raw := stdout.Bytes()
	if len(raw) < frameSize {
		return Frame{}, fmt.Errorf("incomplete native frame: got %d, expected %d", len(raw), frameSize)
	}

	data := make([]byte, frameSize)
	copy(data, raw[:frameSize])
	return Frame{
		Data:   data,
		Width:  vr.meta.NativeWidth,
		Height: vr.meta.NativeHeight,
	}, nil
}

// Close releases resources held by the VideoReader.
// Currently a no-op since ffmpeg is invoked per-batch, but included
// to satisfy a resource-management interface pattern.
func (vr *VideoReader) Close() {
	// No persistent resources to release.
}

// parseFrameRate parses an ffprobe r_frame_rate string like "30/1" or "30000/1001".
func parseFrameRate(rate string) (float64, error) {
	parts := strings.SplitN(rate, "/", 2)
	if len(parts) != 2 {
		// Try parsing as a plain number.
		return strconv.ParseFloat(rate, 64)
	}

	num, err := strconv.ParseFloat(parts[0], 64)
	if err != nil {
		return 0, err
	}
	den, err := strconv.ParseFloat(parts[1], 64)
	if err != nil {
		return 0, err
	}
	if den == 0 {
		return 0, fmt.Errorf("frame rate denominator is zero")
	}
	return num / den, nil
}
