package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/liamp/tennis-tagger/internal/app"
	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/config"
)

// loadMatchSetup reads <video>.setup.json (produced by preflight.py) if
// present and converts it to a bridge.MatchSetup. Returns (nil, nil) when
// the sidecar is absent; returns an error only when the file exists but
// is malformed.
func loadMatchSetup(videoPath string) (*bridge.MatchSetup, error) {
	sidecarPath := videoPath + ".setup.json"
	data, err := os.ReadFile(sidecarPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read %s: %w", sidecarPath, err)
	}

	var raw struct {
		FrameWidth       int `json:"frame_width"`
		FrameHeight      int `json:"frame_height"`
		CourtCornersPixel []struct {
			Label string  `json:"label"`
			X     float64 `json:"x"`
			Y     float64 `json:"y"`
		} `json:"court_corners_pixel"`
		Players struct {
			Near string `json:"near"`
			Far  string `json:"far"`
		} `json:"players"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse %s: %w", sidecarPath, err)
	}
	if len(raw.CourtCornersPixel) != 4 {
		return nil, fmt.Errorf("%s: expected 4 corners, got %d", sidecarPath, len(raw.CourtCornersPixel))
	}

	setup := &bridge.MatchSetup{
		FrameWidth:  raw.FrameWidth,
		FrameHeight: raw.FrameHeight,
		NearPlayer:  raw.Players.Near,
		FarPlayer:   raw.Players.Far,
	}
	setup.CornersPixel = make([][2]float64, 4)
	for i, c := range raw.CourtCornersPixel {
		setup.CornersPixel[i] = [2]float64{c.X, c.Y}
	}
	return setup, nil
}

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo})))

	useMock := flag.Bool("mock", false, "Use MockBridge (no Python needed)")
	pythonPath := flag.String("python", "python", "Path to Python executable")
	liveSource := flag.String("live", "", "Live source (webcam index or RTSP URL)")
	retrain := flag.Bool("retrain", false, "Retrain model from accumulated corrections")
	configPath := flag.String("config", "", "Path to YAML config file (default: use built-in defaults)")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

	var b bridge.BridgeBackend

	if *useMock {
		slog.Info("Using MockBridge (--mock flag set)")
		b = bridge.NewMockBridge()
	} else {
		pb := bridge.NewProcessBridge(*pythonPath)
		err := pb.Init(bridge.BridgeConfig{
			ModelsDir:       cfg.ModelsDir,
			Device:          cfg.Device,
			DetectorBackend: cfg.Pipeline.DetectorBackend,
			ClassifierModel: cfg.Pipeline.ClassifierModel,
		})
		if err != nil {
			slog.Error("ProcessBridge init failed — cannot produce real analysis", "error", err)
			fmt.Fprintf(os.Stderr, "Error: Python bridge failed to start: %v\nUse --mock for synthetic test data.\n", err)
			pb.Close()
			os.Exit(1)
		}
		b = pb
	}

	a := app.NewApp(cfg, b)

	if *retrain {
		fmt.Println("Retraining from corrections...")
		result, err := a.TriggerRetrain()
		if err != nil {
			slog.Error("Retrain failed", "error", err)
			os.Exit(1)
		}
		fmt.Printf("Retrain complete: %v\n", result)
		b.Close()
		return
	}

	if *liveSource != "" {
		fmt.Printf("Starting live capture from: %s\n", *liveSource)
		ch, err := a.StartLive(*liveSource)
		if err != nil {
			slog.Error("Failed to start live capture", "error", err)
			os.Exit(1)
		}

		totalDetections := 0
		for result := range ch {
			if result.Error != nil {
				slog.Error("Live batch error", "error", result.Error)
				break
			}
			totalDetections += len(result.Detections)
			fmt.Printf("Live batch: %d frames, %d detections (total: %d)\n",
				result.FrameCount, len(result.Detections), totalDetections)
		}

		a.StopLive()
		fmt.Println("Live capture stopped.")
		return
	}

	args := flag.Args()
	if len(args) > 0 {
		videoPath := args[0]
		fmt.Printf("Processing: %s\n", videoPath)

		if setup, err := loadMatchSetup(videoPath); err != nil {
			slog.Error("Failed to read setup sidecar", "error", err)
			fmt.Fprintf(os.Stderr, "Warning: setup sidecar present but unreadable: %v\n", err)
		} else if setup != nil {
			a.SetMatchSetup(setup)
			if err := b.SetManualCourt(*setup); err != nil {
				slog.Error("SetManualCourt failed — falling back to auto-detect", "error", err)
			} else {
				fmt.Printf("Using pre-flight setup (near=%s, far=%s)\n", setup.NearPlayer, setup.FarPlayer)
			}
		}

		if err := a.ProcessVideo(videoPath); err != nil {
			slog.Error("Processing failed", "error", err)
			os.Exit(1)
		}
		progress := a.GetProgress()
		fmt.Printf("Done: %d/%d frames processed\n", progress.ProcessedFrames, progress.TotalFrames)

		outputPath := videoPath + "_output.csv"
		if err := a.ExportCSV(outputPath); err != nil {
			slog.Error("Export failed", "error", err)
			os.Exit(1)
		}
		fmt.Printf("CSV exported to: %s\n", outputPath)

		if report := a.GetTacticalReport(); report != nil {
			fmt.Println()
			fmt.Print(report.FormatText())
		}

		if a.ShouldRetrain() {
			fmt.Printf("\n%d corrections accumulated. Run with --retrain to improve.\n", a.GetCorrectionCount())
		}
	} else {
		fmt.Println("Tennis Tagger v2")
		fmt.Println("Usage: tagger [--mock] [--python path] [--live source] [--retrain] <video.mp4>")
	}
}
