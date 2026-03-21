package tactics

import (
	"strings"
	"testing"

	"github.com/liamp/tennis-tagger/internal/bridge"
)

func TestGenerateReport_Basic(t *testing.T) {
	patterns := []PlayerPattern{
		{
			PlayerIndex: 0,
			Strokes: StrokeStats{
				Total:  5,
				ByType: map[string]int{"forehand": 3, "backhand": 2},
				ByZone: map[string]int{"deuce": 3, "ad": 2},
				ByDepth: map[string]int{"deep": 4, "short": 1},
			},
			Tendencies: []string{"Dominant stroke: forehand (60%)"},
		},
		{
			PlayerIndex: 1,
			Strokes: StrokeStats{
				Total:  3,
				ByType: map[string]int{"forehand": 2, "serve": 1},
				ByZone: map[string]int{"center": 3},
				ByDepth: map[string]int{"deep": 3},
			},
			Tendencies: []string{},
		},
	}

	rallies := []bridge.RallyResult{
		{StartFrame: 0, EndFrame: 100, NumStrokes: 4, DurationFrames: 100},
		{StartFrame: 200, EndFrame: 350, NumStrokes: 3, DurationFrames: 150},
	}

	report := GenerateReport(patterns, rallies, 120.0)

	if report.TotalRallies != 2 {
		t.Errorf("TotalRallies = %d, want 2", report.TotalRallies)
	}
	if report.Duration != "2m 00s" {
		t.Errorf("Duration = %q, want %q", report.Duration, "2m 00s")
	}
	if len(report.Players) != 2 {
		t.Fatalf("expected 2 players, got %d", len(report.Players))
	}
	if report.Players[0].Label != "Player 1" {
		t.Errorf("Players[0].Label = %q, want %q", report.Players[0].Label, "Player 1")
	}
	if report.Players[0].StrokeBreakdown["forehand"] != 3 {
		t.Errorf("Players[0] forehand = %d, want 3", report.Players[0].StrokeBreakdown["forehand"])
	}
	if report.MatchSummary == "" {
		t.Error("MatchSummary should not be empty")
	}
}

func TestFormatText_NotEmpty(t *testing.T) {
	patterns := []PlayerPattern{
		{
			PlayerIndex: 0,
			Strokes: StrokeStats{
				Total:  2,
				ByType: map[string]int{"forehand": 2},
				ByZone: map[string]int{},
				ByDepth: map[string]int{},
			},
			Tendencies: []string{"Dominant stroke: forehand (100%)"},
		},
		{
			PlayerIndex: 1,
			Strokes: StrokeStats{
				Total:  1,
				ByType: map[string]int{"backhand": 1},
				ByZone: map[string]int{},
				ByDepth: map[string]int{},
			},
			Tendencies: []string{},
		},
	}

	report := GenerateReport(patterns, nil, 30.0)
	text := report.FormatText()

	if text == "" {
		t.Fatal("FormatText returned empty string")
	}
	if !strings.Contains(text, "TACTICAL REPORT") {
		t.Error("FormatText missing header")
	}
	if !strings.Contains(text, "Player 1") {
		t.Error("FormatText missing Player 1 label")
	}
	if !strings.Contains(text, "Player 2") {
		t.Error("FormatText missing Player 2 label")
	}
	if !strings.Contains(text, "forehand(2)") {
		t.Error("FormatText missing forehand count")
	}
}
