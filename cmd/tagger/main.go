package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/liamp/tennis-tagger/internal/app"
	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/config"
	"github.com/liamp/tennis-tagger/internal/modelshare"
	"github.com/liamp/tennis-tagger/internal/pointmodel"
)


// runModelSubcommand handles `tagger model {export|import|merge}` for the
// point model. Designed for two collaborators swapping weights via USB.
//
//   export <out-dir>     write the local point model + manifest into a folder
//   import <bundle-dir>  replace the local point model with the bundle's
//   merge  <bundle-dir>  average the bundle's weights into the local model
//
// All three operate on files/models/point_model/current/best.pt.
func runModelSubcommand(args []string) {
	const localPt = "files/models/point_model/current/best.pt"

	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: tagger model <export|import|merge> <path>")
		os.Exit(2)
	}
	verb := args[0]

	switch verb {
	case "export":
		fs := flag.NewFlagSet("model export", flag.ExitOnError)
		author := fs.String("author", "", "Your name (shown in the manifest)")
		notes := fs.String("notes", "", "Optional free-text note")
		_ = fs.Parse(args[1:])
		if fs.NArg() != 1 {
			fmt.Fprintln(os.Stderr, "usage: tagger model export [--author NAME] [--notes TEXT] <out-dir>")
			os.Exit(2)
		}
		m, err := modelshare.ExportModel(localPt, fs.Arg(0), *author, "point_model", *notes)
		if err != nil {
			fmt.Fprintf(os.Stderr, "export failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Exported %s (%.1f MB) to %s\n", m.ModelKind, float64(m.SizeBytes)/(1024*1024), fs.Arg(0))

	case "import":
		fs := flag.NewFlagSet("model import", flag.ExitOnError)
		_ = fs.Parse(args[1:])
		if fs.NArg() != 1 {
			fmt.Fprintln(os.Stderr, "usage: tagger model import <bundle-dir>")
			os.Exit(2)
		}
		bundleDir := fs.Arg(0)
		m, err := modelshare.LoadManifest(bundleDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		if err := modelshare.VerifyChecksum(bundleDir, m); err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		if err := modelshare.CopyWeightsTo(bundleDir, m, localPt); err != nil {
			fmt.Fprintf(os.Stderr, "copy: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Imported %s's model (%s) to %s\n", m.Author, m.CreatedAt, localPt)

	case "merge":
		fs := flag.NewFlagSet("model merge", flag.ExitOnError)
		python := fs.String("python", "python", "Python interpreter")
		_ = fs.Parse(args[1:])
		if fs.NArg() != 1 {
			fmt.Fprintln(os.Stderr, "usage: tagger model merge <bundle-dir>")
			os.Exit(2)
		}
		bundleDir := fs.Arg(0)
		m, err := modelshare.LoadManifest(bundleDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		if err := modelshare.VerifyChecksum(bundleDir, m); err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(1)
		}
		cmd := exec.Command(*python, filepath.Join("ml", "merge_models.py"),
			"--out", localPt, localPt, filepath.Join(bundleDir, m.WeightsFilename))
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "merge failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Merged %s's model (%s) into %s\n", m.Author, m.CreatedAt, localPt)

	default:
		fmt.Fprintf(os.Stderr, "unknown verb %q (expected export, import, or merge)\n", verb)
		os.Exit(2)
	}
}

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
	// Subcommand dispatch (kept simple to avoid restructuring the existing flag set).
	if len(os.Args) >= 2 && os.Args[1] == "model" {
		runModelSubcommand(os.Args[2:])
		return
	}

	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo})))

	useMock := flag.Bool("mock", false, "Use MockBridge (no Python needed)")
	pythonPath := flag.String("python", "python", "Path to Python executable")
	liveSource := flag.String("live", "", "Live source (webcam index or RTSP URL)")
	retrain := flag.Bool("retrain", false, "Retrain model from accumulated corrections")
	configPath := flag.String("config", "", "Path to YAML config file (default: use built-in defaults)")
	usePointModel := flag.Bool("use-pointmodel", false, "Use Plan 3 multi-task model + Bayesian fusion for point recognition (requires --pointmodel-ckpt)")
	pointModelCkpt := flag.String("pointmodel-ckpt", "files/models/point_model/run0/best.pt", "Path to the PointModel checkpoint")
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

	if *usePointModel {
		pmc, err := pointmodel.Start(*pythonPath, *pointModelCkpt)
		if err != nil {
			slog.Error("Failed to start PointModel inference server", "error", err, "ckpt", *pointModelCkpt)
			fmt.Fprintf(os.Stderr, "Error: --use-pointmodel but server start failed: %v\n", err)
			b.Close()
			os.Exit(1)
		}
		defer pmc.Close()
		if err := pmc.Ping(); err != nil {
			slog.Error("PointModel ping failed", "error", err)
			b.Close()
			os.Exit(1)
		}
		a.SetPointModelClient(pmc)
		slog.Info("PointModel inference server ready", "ckpt", *pointModelCkpt)
	}

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
