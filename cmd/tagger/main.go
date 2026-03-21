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
	flag.Parse()

	cfg, err := config.Load("")
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

	var b bridge.BridgeBackend

	if *useMock {
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
			slog.Warn("ProcessBridge init failed, falling back to MockBridge", "error", err)
			pb.Close()
			b = bridge.NewMockBridge()
		} else {
			b = pb
		}
	}

	a := app.NewApp(cfg, b)

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
	} else {
		fmt.Println("Tennis Tagger v2")
		fmt.Println("Usage: tagger [--mock] [--python path] <video.mp4>")
	}
}
