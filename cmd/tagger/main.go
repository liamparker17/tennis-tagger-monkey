package main

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/liamp/tennis-tagger/internal/app"
	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/config"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo})))

	cfg, err := config.Load("")
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

	b := bridge.NewMockBridge()
	a := app.NewApp(cfg, b)

	if len(os.Args) > 1 {
		videoPath := os.Args[1]
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
		fmt.Println("Usage: tagger <video.mp4>")
	}
}
