package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/liamp/tennis-tagger/internal/app"
	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/config"
)

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
