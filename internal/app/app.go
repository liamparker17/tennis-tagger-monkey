package app

import (
	"fmt"

	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/config"
	"github.com/liamp/tennis-tagger/internal/export"
	"github.com/liamp/tennis-tagger/internal/pipeline"
)

// App is the top-level application struct that orchestrates the pipeline,
// configuration, and export. It will later become the Wails binding target.
type App struct {
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

// resultToRows converts a pipeline Result into Dartfish export rows.
// Each detection frame becomes one row with basic fields populated.
func resultToRows(r *pipeline.Result) []export.ResultRow {
	rows := make([]export.ResultRow, 0, len(r.Detections))
	for _, det := range r.Detections {
		fields := make([]string, 62) // DartfishColumns has 62 columns

		// Populate basic timing fields
		timestamp := ""
		if r.FPS > 0 {
			seconds := float64(det.FrameIndex) / r.FPS
			timestamp = export.FormatTimestamp(seconds)
		}
		fields[0] = timestamp // Time
		fields[1] = fmt.Sprintf("%d", det.FrameIndex)

		// Player count
		fields[2] = fmt.Sprintf("%d", len(det.Players))

		// Ball detected
		if det.Ball != nil {
			fields[3] = "true"
		} else {
			fields[3] = "false"
		}

		rows = append(rows, export.ResultRow{Fields: fields})
	}
	return rows
}
