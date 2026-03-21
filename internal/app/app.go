package app

import (
	"context"
	"fmt"

	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/config"
	"github.com/liamp/tennis-tagger/internal/export"
	"github.com/liamp/tennis-tagger/internal/pipeline"
)

// App is the top-level application struct that orchestrates the pipeline,
// configuration, and export. It serves as the Wails binding target.
type App struct {
	ctx      context.Context // Wails application context, set via Startup()
	pipeline *pipeline.Pipeline
	config   *config.Config
	exporter *export.DartfishExporter
	result   *pipeline.Result
}

// NewApp creates a new App with the given config and bridge backend.
func NewApp(cfg *config.Config, b bridge.BridgeBackend) *App {
	return &App{
		pipeline: pipeline.New(b, cfg),
		config:   cfg,
		exporter: export.NewDartfishExporter(),
	}
}

// Startup is called by Wails at application start. It stores the
// application context for use by Wails runtime calls (dialogs, events).
func (a *App) Startup(ctx context.Context) {
	a.ctx = ctx
}

// SelectVideoPath opens a file dialog to select a video file.
// Returns "" until Wails runtime is wired in.
// TODO: wailsRuntime.OpenFileDialog(a.ctx, wailsRuntime.OpenDialogOptions{...})
func (a *App) SelectVideoPath() string {
	return ""
}

// ProcessVideoAsync starts video processing in a background goroutine.
// TODO: emit wailsRuntime.EventsEmit(a.ctx, "processing-complete") when done
// TODO: emit wailsRuntime.EventsEmit(a.ctx, "processing-error", err.Error()) on failure
func (a *App) ProcessVideoAsync(path string) {
	go func() {
		if err := a.ProcessVideo(path); err != nil {
			// TODO: wailsRuntime.EventsEmit(a.ctx, "processing-error", err.Error())
			return
		}
		// TODO: wailsRuntime.EventsEmit(a.ctx, "processing-complete", nil)
	}()
}

// GetDeviceInfo returns the configured compute device as a string map.
func (a *App) GetDeviceInfo() map[string]string {
	return map[string]string{"device": a.config.Device}
}

// ProcessVideo runs the full analysis pipeline on the video at path
// and stores the result. Runs synchronously.
func (a *App) ProcessVideo(path string) error {
	result, err := a.pipeline.Process(path)
	if err != nil {
		return fmt.Errorf("process video: %w", err)
	}
	a.result = result
	return nil
}

// GetProgress returns the current pipeline processing progress.
func (a *App) GetProgress() pipeline.ProgressInfo {
	return a.pipeline.Progress()
}

// GetResult returns the result of the last pipeline run, or nil if
// no video has been processed yet.
func (a *App) GetResult() *pipeline.Result {
	return a.result
}

// ExportCSV exports the current result to a Dartfish-compatible CSV file
// at outputPath. Returns an error if no result is available.
func (a *App) ExportCSV(outputPath string) error {
	if a.result == nil {
		return fmt.Errorf("no result to export; run ProcessVideo first")
	}

	// Convert pipeline detections to export rows.
	rows := resultToRows(a.result)

	if err := a.exporter.ExportFile(rows, outputPath); err != nil {
		return fmt.Errorf("export CSV: %w", err)
	}

	return nil
}

// StartLive begins live capture and processing from the given source.
// Returns a channel of LiveResult that emits detection results per batch.
func (a *App) StartLive(source string) (<-chan pipeline.LiveResult, error) {
	return a.pipeline.ProcessLive(source, 640, 480, 30.0)
}

// StopLive stops the live capture and processing.
func (a *App) StopLive() {
	a.pipeline.StopLive()
}

// IsLive returns whether live processing is currently active.
func (a *App) IsLive() bool {
	return a.pipeline.IsLive()
}

// resultToRows converts a pipeline Result into Dartfish export rows.
// Each detection frame becomes one row with basic fields populated.
func resultToRows(r *pipeline.Result) []export.ResultRow {
	rows := make([]export.ResultRow, 0, len(r.Detections))
	for _, det := range r.Detections {
		fields := make([]string, len(export.DartfishColumns))

		// T1: Video Timestamp (column index 48)
		if r.FPS > 0 {
			seconds := float64(det.FrameIndex) / r.FPS
			fields[48] = export.FormatTimestamp(seconds)
		}
		// T2: Frame Number (column index 49)
		fields[49] = fmt.Sprintf("%d", det.FrameIndex)

		// U1: Confidence Score (column index 50)
		if det.Ball != nil {
			fields[50] = fmt.Sprintf("%.3f", det.Ball.Confidence)
		}

		// G2: Total Strokes — player count as placeholder
		fields[23] = fmt.Sprintf("%d", len(det.Players))

		rows = append(rows, export.ResultRow{Fields: fields})
	}
	return rows
}
