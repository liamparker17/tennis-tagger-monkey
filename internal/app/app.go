package app

import (
	"context"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"

	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/config"
	"github.com/liamp/tennis-tagger/internal/corrections"
	"github.com/liamp/tennis-tagger/internal/export"
	"github.com/liamp/tennis-tagger/internal/pipeline"
	"github.com/liamp/tennis-tagger/internal/point"
	"github.com/liamp/tennis-tagger/internal/tactics"
)

// App is the top-level application struct that orchestrates the pipeline,
// configuration, and export. It serves as the Wails binding target.
type App struct {
	ctx         context.Context // Wails application context, set via Startup()
	pipeline    *pipeline.Pipeline
	config      *config.Config
	exporter    *export.DartfishExporter
	result      *pipeline.Result
	report      *tactics.TacticalReport
	corrections *corrections.Store
	setup       *bridge.MatchSetup // optional: loaded from <video>.setup.json
}

// SetMatchSetup records the user's pre-flight data so exported rows can
// reference the real player names instead of generic indices.
func (a *App) SetMatchSetup(setup *bridge.MatchSetup) {
	a.setup = setup
}

// NewApp creates a new App with the given config and bridge backend.
func NewApp(cfg *config.Config, b bridge.BridgeBackend) *App {
	return &App{
		pipeline:    pipeline.New(b, cfg),
		config:      cfg,
		exporter:    export.NewDartfishExporter(),
		corrections: corrections.NewStore(filepath.Join(cfg.ModelsDir, "corrections")),
	}
}

// Startup is called by Wails at application start. It stores the
// application context for use by Wails runtime calls (dialogs, events).
func (a *App) Startup(ctx context.Context) {
	a.ctx = ctx
}

// SelectVideoPath is a placeholder for the Wails file dialog.
// Returns an error string until the desktop UI is implemented.
func (a *App) SelectVideoPath() string {
	slog.Warn("SelectVideoPath: desktop UI not yet implemented — use CLI with a file path argument")
	return ""
}

// ProcessVideoAsync is a placeholder for Wails async processing.
// Logs a warning and processes synchronously until event emission is wired.
func (a *App) ProcessVideoAsync(path string) {
	slog.Warn("ProcessVideoAsync: desktop event emission not yet wired — running synchronously")
	if err := a.ProcessVideo(path); err != nil {
		slog.Error("ProcessVideoAsync failed", "error", err)
	}
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
	rows := resultToRows(a.result, a.setup)

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

// GetTacticalReport runs pattern analysis and report generation on the current
// pipeline result. Returns nil if no result is available.
func (a *App) GetTacticalReport() *tactics.TacticalReport {
	if a.result == nil {
		return nil
	}

	patterns := tactics.AnalyzePatterns(
		a.result.Detections,
		a.result.Placements,
		a.result.Strokes,
		a.result.Rallies,
	)

	durationSec := 0.0
	if a.result.FPS > 0 {
		durationSec = float64(a.result.TotalFrames) / a.result.FPS
	}

	// Count frames with a ball detection for the summary.
	ballDetections := 0
	for _, det := range a.result.Detections {
		if det.Ball != nil {
			ballDetections++
		}
	}

	a.report = tactics.GenerateReport(patterns, a.result.Rallies, durationSec, ballDetections)
	return a.report
}

// SaveCorrection persists a user correction for later retraining.
func (a *App) SaveCorrection(c corrections.Correction) error {
	return a.corrections.Save(c)
}

// GetCorrectionCount returns the number of accumulated corrections.
func (a *App) GetCorrectionCount() int {
	return a.corrections.Count()
}

// ShouldRetrain reports whether enough corrections have accumulated to
// justify a retraining run.
func (a *App) ShouldRetrain() bool {
	return a.corrections.ShouldRetrain()
}

// TriggerRetrain flushes accumulated corrections and starts a fine-tuning
// run via the bridge backend.
func (a *App) TriggerRetrain() (map[string]interface{}, error) {
	batch, err := a.corrections.Flush()
	if err != nil {
		return nil, fmt.Errorf("flush corrections: %w", err)
	}

	// Convert corrections to bridge correction data for fine-tuning
	corrData := make([]bridge.CorrectionData, len(batch.Corrections))
	for i, c := range batch.Corrections {
		corrData[i] = bridge.CorrectionData{
			VideoPath:  c.VideoPath,
			FrameIndex: c.FrameIndex,
			Type:       c.Type,
			Original:   c.Original,
			Corrected:  c.Corrected,
			PlayerID:   c.PlayerID,
		}
	}

	cfg := bridge.TrainingConfig{Task: "fine_tune", Epochs: 5, BatchSize: 16}
	if err := a.pipeline.Bridge().FineTune(corrData, cfg); err != nil {
		return nil, fmt.Errorf("retrain: %w", err)
	}

	return map[string]interface{}{
		"status":      "ok",
		"corrections": len(batch.Corrections),
	}, nil
}

// resultToRows converts a pipeline Result into Dartfish export rows.
//
// When the result contains recognised Points, one row is emitted per point
// using the Dartfish column mapping below. Each point row also encodes
// per-shot detail (bounce position, speed, in/out) in V1: Notes.
//
// When no points are available (e.g. early in processing), it falls back to
// one row per detection frame as before.
//
// Column index reference (62 columns total):
//
//	 0  Name            — "Point N"
//	 1  Position        — video timestamp of first shot
//	 2  Duration        — duration of the point (MM:SS.mmm)
//	 3  "0 - Point Level"
//	 4  A1: Server      — 0 or 1
//	 6  A3: Serve Placement — serve side (deuce/ad)
//	16  E1: Last Shot   — last shot hitter (0 or 1)
//	17  E2: Last Shot Winner — "1" when last hitter won
//	18  E3: Last Shot Error  — "1" when last hitter errored
//	19  E4: Last Shot Placement — in/out of last shot
//	20  F1: Point Won   — winner player (0 or 1)
//	21  F2: Point Score — score string after point (e.g. "15-0")
//	22  G1: Rally Length — shot count
//	35  M1: Serve Speed  — first serve speed kph
//	36  M2: Serve Number — "1" = first serve, "2" = second serve
//	37  N1: First Serve In — "1" if first serve landed in
//	48  T1: Video Timestamp — timestamp of first shot frame
//	49  T2: Frame Number    — start frame of first shot
//	50  U1: Confidence Score — minimum shot confidence in point
//	51  V1: Notes        — per-shot detail (CSV encoded inline)
func resultToRows(r *pipeline.Result, setup *bridge.MatchSetup) []export.ResultRow {
	if len(r.Points) > 0 {
		return pointsToRows(r, setup)
	}
	return detectionsToRows(r)
}

// playerName resolves a player index (0 = near / Player A, 1 = far /
// Player B) to the user-supplied name from the pre-flight sidecar, or
// to "Player A" / "Player B" if no setup was provided.
func playerName(setup *bridge.MatchSetup, idx int) string {
	if setup != nil {
		if idx == 0 && setup.NearPlayer != "" {
			return setup.NearPlayer
		}
		if idx == 1 && setup.FarPlayer != "" {
			return setup.FarPlayer
		}
	}
	if idx == 0 {
		return "Player A"
	}
	return "Player B"
}

// xyCoord formats a normalised court position [0,1] as the "x;y" string
// Dartfish expects (using a 0-200 coordinate scale to match the
// human-tagged export range, e.g. "113;74").
func xyCoord(cx, cy float64) string {
	if cx == 0 && cy == 0 {
		return ""
	}
	return fmt.Sprintf("%d;%d", int(cx*200), int(cy*200))
}

// pointsToRows emits one row per recognised Point, mapping our pipeline
// output into the real Dartfish column layout the human tagger uses.
func pointsToRows(r *pipeline.Result, setup *bridge.MatchSetup) []export.ResultRow {
	rows := make([]export.ResultRow, 0, len(r.Points))

	// Build a running match state mirror so we can report score-after-point.
	// We replay AwardPoint in sequence to get the state snapshot before each
	// point is awarded, then use the state after to report the current score.
	ms := point.NewMatchState(0)
	if r.MatchScore != nil {
		// If the pipeline computed a final MatchState, use its initial server.
		ms = point.NewMatchState(r.Points[0].Server)
	}

	// Match-level metadata (same on every row).
	playerA := playerName(setup, 0) // near = Player A
	playerB := playerName(setup, 1) // far  = Player B

	for _, pt := range r.Points {
		fields := make([]string, len(export.DartfishColumns))

		// Name / identity
		fields[export.ColName] = fmt.Sprintf("Point %d", pt.Number)
		fields[export.ColPlayerA] = playerA
		fields[export.ColPlayerB] = playerB

		// Server / Returner (full names)
		serverName := playerName(setup, pt.Server)
		returnerName := playerName(setup, 1-pt.Server)
		fields[export.ColServer] = serverName
		fields[export.ColReturner] = returnerName

		// Serve placement
		fields[export.ColServePlacement] = pt.ServeSide

		// Stroke count (H1) — this is what the human tagger records,
		// NOT a separate rally concept.
		fields[export.ColStrokeCount] = fmt.Sprintf("%d", len(pt.Shots))

		// Point winner (F1) — stored as the winning player's name so it
		// matches the Dartfish convention instead of a numeric index.
		fields[export.ColPointWon] = playerName(setup, pt.Outcome.Winner)

		// Last-shot data + winner/error flags
		if n := len(pt.Shots); n > 0 {
			last := pt.Shots[n-1]
			fields[export.ColLastShot] = playerName(setup, last.Hitter)

			if last.Hitter == pt.Outcome.Winner {
				fields[export.ColLastShotWinner] = playerName(setup, last.Hitter)
			} else {
				fields[export.ColLastShotError] = playerName(setup, last.Hitter)
			}

			if last.Bounce != nil {
				fields[export.ColLastShotPlacement] = last.Bounce.InOut
				fields[export.ColXYLastShot] = xyCoord(last.Bounce.CX, last.Bounce.CY)
			}

			// Serve data + speed + XY
			first := pt.Shots[0]
			if first.IsServe {
				if first.SpeedKPH > 0 {
					fields[export.ColServeData] = fmt.Sprintf("%.1f kph", first.SpeedKPH)
				}
				// Which court was the serve struck from?
				if first.Bounce != nil {
					if strings.EqualFold(pt.ServeSide, "Ad") || strings.HasPrefix(strings.ToLower(pt.ServeSide), "ad") {
						fields[export.ColXYAd] = xyCoord(first.Bounce.CX, first.Bounce.CY)
					} else {
						fields[export.ColXYDeuce] = xyCoord(first.Bounce.CX, first.Bounce.CY)
					}
				}
			}

			// Return shot (the 2nd shot in the rally, if present).
			if n >= 2 {
				ret := pt.Shots[1]
				fields[export.ColReturner] = playerName(setup, ret.Hitter)
				if ret.Bounce != nil {
					fields[export.ColReturnPlacement] = ret.Bounce.InOut
					fields[export.ColXYReturn] = xyCoord(ret.Bounce.CX, ret.Bounce.CY)
				}
			}

			// Serve+1 and Return+1 XY
			if n >= 3 && pt.Shots[2].Bounce != nil {
				fields[export.ColXYSrvPlus1] = xyCoord(pt.Shots[2].Bounce.CX, pt.Shots[2].Bounce.CY)
			}
			if n >= 4 && pt.Shots[3].Bounce != nil {
				fields[export.ColXYRetPlus1] = xyCoord(pt.Shots[3].Bounce.CX, pt.Shots[3].Bounce.CY)
			}

			// Position (start of point) + Duration
			if r.FPS > 0 {
				sec := float64(first.StartFrame) / r.FPS
				fields[export.ColPosition] = export.FormatTimestamp(sec)

				endFrame := last.EndFrame
				if endFrame <= first.StartFrame {
					endFrame = last.StartFrame
				}
				durSec := float64(endFrame-first.StartFrame) / r.FPS
				fields[export.ColDuration] = export.FormatTimestamp(durSec)
			}
		}

		// Advance mirrored match state, snapshot for score columns.
		ms.AwardPoint(pt.Outcome.Winner)
		fields[export.ColPointScore] = formatPointScore(ms)
		fields[export.ColGameScore] = fmt.Sprintf("%d-%d", ms.GameScore[0], ms.GameScore[1])
		fields[export.ColDeuceAd] = ms.ServeSide
		fields[export.ColSetNumber] = fmt.Sprintf("%d", len(ms.Sets)+1)
		fields[export.ColSetScore] = fmt.Sprintf("%d-%d", ms.SetScore[0], ms.SetScore[1])

		rows = append(rows, export.ResultRow{Fields: fields})
	}

	return rows
}

// formatPointScore returns a human-readable point score string such as "15-0"
// or "40-40(D)" based on the current MatchState.
func formatPointScore(ms *point.MatchState) string {
	if ms.IsDeuce {
		return "40-40(D)"
	}
	p0 := ms.PointScore[0]
	p1 := ms.PointScore[1]
	label := func(v int) string {
		switch v {
		case 41:
			return "A"
		default:
			return fmt.Sprintf("%d", v)
		}
	}
	return fmt.Sprintf("%s-%s", label(p0), label(p1))
}

// encodeShotNotes encodes per-shot data as a compact semicolon-separated
// string for the V1: Notes column. Example entry:
//
//	"shot1:hitter=0,cx=3.21,cy=7.45,inout=in,speed=185.3,serve=true"
func encodeShotNotes(shots []point.Shot) string {
	parts := make([]string, 0, len(shots))
	for _, s := range shots {
		cx, cy, inout := 0.0, 0.0, "none"
		if s.Bounce != nil {
			cx = s.Bounce.CX
			cy = s.Bounce.CY
			inout = s.Bounce.InOut
		}
		parts = append(parts, fmt.Sprintf(
			"shot%d:hitter=%d,cx=%.2f,cy=%.2f,inout=%s,speed=%.1f,serve=%v",
			s.Index, s.Hitter, cx, cy, inout, s.SpeedKPH, s.IsServe,
		))
	}
	return strings.Join(parts, ";")
}

// detectionsToRows is the legacy fallback: one row per detection frame.
// Used when no point data is available.
func detectionsToRows(r *pipeline.Result) []export.ResultRow {
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
