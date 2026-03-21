# Phase 3: Real-Time Live Tagging

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process live camera/RTSP streams in real-time, displaying detections with <500ms latency.

**Architecture:** Go captures live frames via ffmpeg RTSP/webcam input, sends small batches (4-8 frames) to the Python bridge for low-latency inference, and streams results back. Non-critical analysis (rally segmentation, placements) is skipped in live mode. A new LiveView in the UI shows the video feed with detection overlays.

**Tech Stack:** Go, ffmpeg (RTSP/webcam capture), existing ProcessBridge, Svelte (LiveView)

**Spec:** Phase 3 section of `docs/superpowers/specs/2026-03-20-tennis-tagger-rewrite-design.md`

---

## File Structure

```
New files:
  internal/video/live.go              — LiveReader: ffmpeg RTSP/webcam capture
  internal/video/live_test.go         — LiveReader tests
  internal/pipeline/live.go           — ProcessLive: streaming pipeline
  internal/pipeline/live_test.go      — Live pipeline tests
  frontend/src/lib/LiveView.svelte    — Live stream UI (placeholder)

Modified files:
  internal/config/config.go           — Add LiveConfig (source URL, batch size)
  internal/config/config_test.go      — Test new fields
  internal/app/app.go                 — Add StartLive/StopLive methods
  cmd/tagger/main.go                  — Add --live flag
```

---

## Task Groups

| Group | Name | Depends On | Tasks |
|-------|------|------------|-------|
| 1 | Live video capture | — | 1-2 |
| 2 | Live pipeline | Group 1 | 3-4 |
| 3 | Config + CLI + UI | Group 2 | 5-6 |

---

## Group 1: Live Video Capture

### Task 1: LiveReader

**Files:**
- Create: `internal/video/live.go`
- Create: `internal/video/live_test.go`

- [ ] **Step 1: Implement LiveReader**

```go
// internal/video/live.go
package video

import (
    "bufio"
    "fmt"
    "io"
    "os/exec"
    "sync"
    "sync/atomic"
)

// LiveReader captures frames from a live source (webcam or RTSP stream) via ffmpeg.
type LiveReader struct {
    cmd     *exec.Cmd
    stdout  io.ReadCloser
    width   int
    height  int
    fps     float64
    running atomic.Bool
    mu      sync.Mutex
}

// OpenLive starts capturing from the given source.
// source can be:
//   - "0" or "1" for webcam index (Windows: uses dshow, Linux: v4l2)
//   - "rtsp://..." for RTSP stream
//   - Any URL ffmpeg accepts as input
func OpenLive(source string, width, height int, fps float64) (*LiveReader, error) {
    args := buildLiveArgs(source, width, height, fps)

    cmd := exec.Command("ffmpeg", args...)
    stdout, err := cmd.StdoutPipe()
    if err != nil {
        return nil, fmt.Errorf("stdout pipe: %w", err)
    }
    cmd.Stderr = nil // inherit parent stderr for diagnostics

    if err := cmd.Start(); err != nil {
        return nil, fmt.Errorf("start ffmpeg: %w", err)
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

// ReadFrame reads a single raw RGB frame. Blocks until frame is available.
// Returns io.EOF when stream ends.
func (lr *LiveReader) ReadFrame() (Frame, error) {
    if !lr.running.Load() {
        return Frame{}, io.EOF
    }

    frameSize := lr.width * lr.height * 3
    buf := make([]byte, frameSize)
    _, err := io.ReadFull(lr.stdout, buf)
    if err != nil {
        lr.running.Store(false)
        return Frame{}, err
    }

    return Frame{Data: buf, Width: lr.width, Height: lr.height}, nil
}

// ReadBatch reads n frames into a batch. Returns fewer frames if stream ends.
func (lr *LiveReader) ReadBatch(n int) ([]Frame, error) {
    frames := make([]Frame, 0, n)
    for i := 0; i < n; i++ {
        f, err := lr.ReadFrame()
        if err != nil {
            break
        }
        frames = append(frames, f)
    }
    if len(frames) == 0 {
        return nil, io.EOF
    }
    return frames, nil
}

// Stop terminates the ffmpeg capture process.
func (lr *LiveReader) Stop() {
    lr.mu.Lock()
    defer lr.mu.Unlock()

    lr.running.Store(false)
    if lr.stdout != nil {
        lr.stdout.Close()
    }
    if lr.cmd != nil && lr.cmd.Process != nil {
        lr.cmd.Process.Kill()
        lr.cmd.Wait()
    }
}

// Running returns whether the live capture is active.
func (lr *LiveReader) Running() bool {
    return lr.running.Load()
}

// Width returns the frame width.
func (lr *LiveReader) Width() int { return lr.width }

// Height returns the frame height.
func (lr *LiveReader) Height() int { return lr.height }

// FPS returns the configured frame rate.
func (lr *LiveReader) FPS() float64 { return lr.fps }

// buildLiveArgs constructs ffmpeg arguments for live capture.
func buildLiveArgs(source, width, height int, fps float64) []string {
    // Detect source type
    isWebcam := len(source) <= 2 // "0", "1", etc.

    var args []string

    if isWebcam {
        // Platform-specific webcam input
        // Windows: -f dshow -i video="Camera Name" (simplified: use index)
        // Linux: -f v4l2 -i /dev/video0
        // For cross-platform, try generic approach
        args = []string{
            "-f", "dshow",
            "-video_size", fmt.Sprintf("%dx%d", width, height),
            "-framerate", fmt.Sprintf("%.0f", fps),
            "-i", fmt.Sprintf("video=%s", source),
        }
    } else {
        // RTSP or URL input
        args = []string{
            "-rtsp_transport", "tcp",
            "-i", source,
        }
    }

    // Output: raw RGB to stdout
    args = append(args,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", fmt.Sprintf("%dx%d", width, height),
        "-r", fmt.Sprintf("%.0f", fps),
        "-v", "error",
        "pipe:1",
    )

    return args
}
```

Note: Fix the `buildLiveArgs` signature — the parameters should be `(source string, width, height int, fps float64)`.

- [ ] **Step 2: Write tests**

```go
// internal/video/live_test.go
package video

import (
    "os/exec"
    "testing"
)

func TestBuildLiveArgs_RTSP(t *testing.T) {
    args := buildLiveArgs("rtsp://192.168.1.100:554/stream", 640, 480, 30)

    // Should contain rtsp_transport tcp
    found := false
    for _, a := range args {
        if a == "tcp" {
            found = true
        }
    }
    if !found {
        t.Error("expected rtsp_transport tcp in args")
    }

    // Should end with pipe:1
    if args[len(args)-1] != "pipe:1" {
        t.Errorf("expected last arg pipe:1, got %s", args[len(args)-1])
    }
}

func TestBuildLiveArgs_Webcam(t *testing.T) {
    args := buildLiveArgs("0", 640, 480, 30)

    // Should contain dshow (Windows)
    found := false
    for _, a := range args {
        if a == "dshow" {
            found = true
        }
    }
    if !found {
        t.Log("dshow not found — may need platform-specific handling")
    }
}

func TestOpenLive_InvalidSource(t *testing.T) {
    _, err := exec.LookPath("ffmpeg")
    if err != nil {
        t.Skip("ffmpeg not in PATH")
    }

    // Invalid RTSP URL should fail quickly
    lr, err := OpenLive("rtsp://invalid:554/nonexistent", 640, 480, 30)
    if err == nil {
        // If it didn't error on start, reading should fail
        _, readErr := lr.ReadFrame()
        if readErr == nil {
            t.Error("expected read error from invalid source")
        }
        lr.Stop()
    }
}
```

- [ ] **Step 3: Run tests, commit**

```bash
go test ./internal/video/ -v -run "TestBuildLive|TestOpenLive_Invalid"
git add internal/video/live.go internal/video/live_test.go
git commit -m "feat: add LiveReader for RTSP/webcam frame capture"
```

---

### Task 2: Fix buildLiveArgs signature and platform detection

- [ ] **Step 1: Use runtime.GOOS for platform detection**

```go
import "runtime"

func buildLiveArgs(source string, width, height int, fps float64) []string {
    isWebcam := len(source) <= 2

    var args []string
    if isWebcam {
        switch runtime.GOOS {
        case "windows":
            args = []string{"-f", "dshow", "-video_size", fmt.Sprintf("%dx%d", width, height),
                "-framerate", fmt.Sprintf("%.0f", fps), "-i", fmt.Sprintf("video=%s", source)}
        case "linux":
            args = []string{"-f", "v4l2", "-video_size", fmt.Sprintf("%dx%d", width, height),
                "-framerate", fmt.Sprintf("%.0f", fps), "-i", fmt.Sprintf("/dev/video%s", source)}
        case "darwin":
            args = []string{"-f", "avfoundation", "-video_size", fmt.Sprintf("%dx%d", width, height),
                "-framerate", fmt.Sprintf("%.0f", fps), "-i", source}
        default:
            args = []string{"-i", source}
        }
    } else {
        args = []string{"-rtsp_transport", "tcp", "-i", source}
    }

    args = append(args, "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", fmt.Sprintf("%dx%d", width, height),
        "-r", fmt.Sprintf("%.0f", fps), "-v", "error", "pipe:1")
    return args
}
```

- [ ] **Step 2: Run tests, commit**

---

## Group 2: Live Pipeline

### Task 3: ProcessLive method

**Files:**
- Create: `internal/pipeline/live.go`
- Create: `internal/pipeline/live_test.go`

- [ ] **Step 1: Implement ProcessLive**

```go
// internal/pipeline/live.go
package pipeline

import (
    "fmt"
    "log/slog"
    "sync"

    "github.com/liamp/tennis-tagger/internal/bridge"
    "github.com/liamp/tennis-tagger/internal/video"
)

// LiveResult is sent for each processed batch during live streaming.
type LiveResult struct {
    Detections []bridge.DetectionResult
    FrameCount int
    Error      error
}

// ProcessLive starts live processing from the given source.
// Results are streamed to the returned channel.
// Call StopLive() to terminate.
func (p *Pipeline) ProcessLive(source string, width, height int, fps float64) (<-chan LiveResult, error) {
    lr, err := video.OpenLive(source, width, height, fps)
    if err != nil {
        return nil, fmt.Errorf("open live source: %w", err)
    }

    p.liveMu.Lock()
    p.liveReader = lr
    p.liveRunning = true
    p.liveMu.Unlock()

    resultCh := make(chan LiveResult, 4)
    p.progress = ProgressInfo{Stage: "live"}
    p.tracker.Reset()

    // Reduced batch size for low latency
    batchSize := p.config.Pipeline.LiveBatchSize
    if batchSize <= 0 {
        batchSize = 4 // Small batches = low latency
    }

    // Court detection from first frame
    go func() {
        defer close(resultCh)
        defer lr.Stop()

        courtDetected := false
        frameCount := 0

        slog.Info("Live pipeline started", "source", source, "batch", batchSize)

        for lr.Running() {
            frames, err := lr.ReadBatch(batchSize)
            if err != nil {
                slog.Info("Live stream ended", "error", err)
                break
            }

            // Detect court from first batch (once)
            if !courtDetected && len(frames) > 0 {
                court, err := p.bridge.DetectCourt(bridge.Frame{
                    Data: frames[0].Data, Width: frames[0].Width, Height: frames[0].Height,
                })
                if err == nil && court.Confidence > 0 {
                    courtDetected = true
                    slog.Info("Court detected in live stream")
                }
                _ = court // stored for later use if needed
            }

            // Convert and detect
            bFrames := make([]bridge.Frame, len(frames))
            for i, f := range frames {
                bFrames[i] = bridge.Frame{Data: f.Data, Width: f.Width, Height: f.Height}
            }

            detections, err := p.bridge.DetectBatch(bFrames)
            if err != nil {
                slog.Error("Live detection failed", "error", err)
                resultCh <- LiveResult{Error: err}
                continue
            }

            // Set frame indices and filter
            for i := range detections {
                detections[i].FrameIndex = frameCount + i
                detections[i].Players = filterBBoxes(detections[i].Players)
            }

            // Track objects
            for _, det := range detections {
                p.tracker.Update(det.Players)
            }

            frameCount += len(frames)
            p.progress.ProcessedFrames = frameCount

            resultCh <- LiveResult{
                Detections: detections,
                FrameCount: frameCount,
            }
        }

        p.liveMu.Lock()
        p.liveRunning = false
        p.liveMu.Unlock()

        p.progress.Stage = "stopped"
        slog.Info("Live pipeline stopped", "frames", frameCount)
    }()

    return resultCh, nil
}

// StopLive terminates a running live stream.
func (p *Pipeline) StopLive() {
    p.liveMu.Lock()
    defer p.liveMu.Unlock()

    if p.liveReader != nil {
        p.liveReader.Stop()
        p.liveRunning = false
    }
}

// IsLive returns whether live processing is active.
func (p *Pipeline) IsLive() bool {
    p.liveMu.Lock()
    defer p.liveMu.Unlock()
    return p.liveRunning
}
```

- [ ] **Step 2: Add live fields to Pipeline struct**

Add to `internal/pipeline/pipeline.go`:

```go
import "sync"

type Pipeline struct {
    bridge      bridge.BridgeBackend
    config      *config.Config
    tracker     *tracker.MultiObjectTracker
    progress    ProgressInfo
    liveReader  *video.LiveReader
    liveRunning bool
    liveMu      sync.Mutex
}
```

Also need to import `video` package in pipeline.go (may already be imported).

- [ ] **Step 3: Write test**

```go
// internal/pipeline/live_test.go
package pipeline

import (
    "testing"

    "github.com/liamp/tennis-tagger/internal/bridge"
    "github.com/liamp/tennis-tagger/internal/config"
)

func TestProcessLive_InvalidSource(t *testing.T) {
    cfg := config.Default()
    mock := bridge.NewMockBridge()
    p := New(mock, cfg)

    _, err := p.ProcessLive("rtsp://invalid:554/nonexistent", 640, 480, 30)
    // May or may not error depending on ffmpeg behavior with invalid RTSP
    // Just verify it doesn't panic
    if err != nil {
        t.Logf("Expected error for invalid source: %v", err)
    } else {
        // If it started, stop it
        p.StopLive()
    }
}

func TestStopLive_WhenNotRunning(t *testing.T) {
    cfg := config.Default()
    mock := bridge.NewMockBridge()
    p := New(mock, cfg)

    // Should not panic
    p.StopLive()

    if p.IsLive() {
        t.Error("expected IsLive() = false")
    }
}
```

- [ ] **Step 4: Run tests, commit**

```bash
go test ./internal/pipeline/ -v -run "TestProcessLive|TestStopLive"
git add internal/pipeline/live.go internal/pipeline/live_test.go internal/pipeline/pipeline.go
git commit -m "feat: add live streaming pipeline with ProcessLive/StopLive"
```

---

### Task 4: Add LiveBatchSize to config

**Files:**
- Modify: `internal/config/config.go`

- [ ] **Step 1: Add LiveBatchSize field**

```go
type PipelineConfig struct {
    // ... existing fields ...
    LiveBatchSize   int    `yaml:"liveBatchSize"`   // Batch size for live mode (default 4)
    LiveSource      string `yaml:"liveSource"`      // Default live source URL or webcam index
}
```

Update `Default()`:
```go
LiveBatchSize: 4,
LiveSource:    "0",  // Default webcam
```

- [ ] **Step 2: Add test**

```go
func TestLoadConfig_LiveDefaults(t *testing.T) {
    cfg, _ := Load("")
    if cfg.Pipeline.LiveBatchSize != 4 {
        t.Errorf("expected LiveBatchSize 4, got %d", cfg.Pipeline.LiveBatchSize)
    }
}
```

- [ ] **Step 3: Run tests, commit**

---

## Group 3: CLI + App Wiring

### Task 5: Add --live flag to CLI

**Files:**
- Modify: `cmd/tagger/main.go`
- Modify: `internal/app/app.go`

- [ ] **Step 1: Add StartLive/StopLive to App**

```go
// StartLive begins live processing from the configured source.
func (a *App) StartLive(source string) (<-chan pipeline.LiveResult, error) {
    width, height := 640, 480
    fps := 30.0
    return a.pipeline.ProcessLive(source, width, height, fps)
}

// StopLive terminates live processing.
func (a *App) StopLive() {
    a.pipeline.StopLive()
}

// IsLive returns whether live processing is active.
func (a *App) IsLive() bool {
    return a.pipeline.IsLive()
}
```

- [ ] **Step 2: Add --live flag to main.go**

```go
liveSource := flag.String("live", "", "Live source (webcam index or RTSP URL)")

// After flag parsing:
if *liveSource != "" {
    fmt.Printf("Starting live capture: %s\n", *liveSource)
    resultCh, err := a.StartLive(*liveSource)
    if err != nil {
        slog.Error("Live start failed", "error", err)
        os.Exit(1)
    }

    // Print detections as they arrive
    for result := range resultCh {
        if result.Error != nil {
            fmt.Printf("Error: %v\n", result.Error)
            continue
        }
        players := 0
        balls := 0
        for _, det := range result.Detections {
            players += len(det.Players)
            if det.Ball != nil {
                balls++
            }
        }
        fmt.Printf("Frame %d: %d players, %d balls detected\n",
            result.FrameCount, players, balls)
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add cmd/tagger/main.go internal/app/app.go
git commit -m "feat: add --live flag for real-time stream processing"
```

---

### Task 6: LiveView placeholder + README update

**Files:**
- Create: `frontend/src/lib/LiveView.svelte`
- Modify: `README.md`

- [ ] **Step 1: Create LiveView placeholder**

```svelte
<!-- frontend/src/lib/LiveView.svelte -->
<div class="live-view">
  <h2>Live Tagging</h2>
  <p>Connect to a camera or RTSP stream for real-time detection.</p>
  <p>CLI usage: <code>tagger --live 0</code> (webcam) or <code>tagger --live rtsp://...</code></p>
</div>

<style>
  .live-view { padding: 2rem; }
  code { background: #1a1a2e; padding: 0.2em 0.5em; border-radius: 4px; }
</style>
```

- [ ] **Step 2: Update README with live mode**

Add to README.md:
```markdown
## Live Mode
tagger --live 0                          # Webcam
tagger --live rtsp://192.168.1.100/stream  # RTSP stream
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/LiveView.svelte README.md
git commit -m "feat: add LiveView placeholder and live mode documentation"
```

---

## Summary

| Group | Tasks | Deliverable |
|-------|-------|-------------|
| 1. Live capture | 1-2 | LiveReader with RTSP/webcam/cross-platform support |
| 2. Live pipeline | 3-4 | ProcessLive with streaming results channel, small batches |
| 3. CLI + App | 5-6 | --live flag, App bindings, LiveView placeholder |
