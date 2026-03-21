package tactics

import (
	"strings"
	"testing"

	"github.com/liamp/tennis-tagger/internal/bridge"
)

func TestAnalyzePatterns_EmptyInput(t *testing.T) {
	patterns := AnalyzePatterns(nil, nil, nil, nil)
	if len(patterns) != 2 {
		t.Fatalf("expected 2 patterns, got %d", len(patterns))
	}
	for i, p := range patterns {
		if p.PlayerIndex != i {
			t.Errorf("pattern[%d].PlayerIndex = %d, want %d", i, p.PlayerIndex, i)
		}
		if p.Strokes.Total != 0 {
			t.Errorf("pattern[%d].Strokes.Total = %d, want 0", i, p.Strokes.Total)
		}
		if len(p.Tendencies) != 0 {
			t.Errorf("pattern[%d].Tendencies should be empty, got %v", i, p.Tendencies)
		}
	}
}

func TestStrokeStats_CountByType(t *testing.T) {
	strokes := []bridge.StrokeResult{
		{Type: "forehand", PlayerID: 0, Frame: 10},
		{Type: "forehand", PlayerID: 0, Frame: 20},
		{Type: "backhand", PlayerID: 0, Frame: 30},
		{Type: "forehand", PlayerID: 1, Frame: 15},
		{Type: "serve", PlayerID: 1, Frame: 5},
	}

	patterns := AnalyzePatterns(nil, nil, strokes, nil)

	// Player 0: 2 forehand, 1 backhand = 3 total
	if patterns[0].Strokes.Total != 3 {
		t.Errorf("player 0 total = %d, want 3", patterns[0].Strokes.Total)
	}
	if patterns[0].Strokes.ByType["forehand"] != 2 {
		t.Errorf("player 0 forehand = %d, want 2", patterns[0].Strokes.ByType["forehand"])
	}
	if patterns[0].Strokes.ByType["backhand"] != 1 {
		t.Errorf("player 0 backhand = %d, want 1", patterns[0].Strokes.ByType["backhand"])
	}

	// Player 1: 1 forehand, 1 serve = 2 total
	if patterns[1].Strokes.Total != 2 {
		t.Errorf("player 1 total = %d, want 2", patterns[1].Strokes.Total)
	}
	if patterns[1].Strokes.ByType["forehand"] != 1 {
		t.Errorf("player 1 forehand = %d, want 1", patterns[1].Strokes.ByType["forehand"])
	}
	if patterns[1].Strokes.ByType["serve"] != 1 {
		t.Errorf("player 1 serve = %d, want 1", patterns[1].Strokes.ByType["serve"])
	}
}

func TestTendencies_Generated(t *testing.T) {
	// 20 forehands out of 20 total = 100% -> should generate dominant stroke tendency
	strokes := make([]bridge.StrokeResult, 20)
	for i := range strokes {
		strokes[i] = bridge.StrokeResult{Type: "forehand", PlayerID: 0, Frame: i * 10}
	}

	patterns := AnalyzePatterns(nil, nil, strokes, nil)

	if len(patterns[0].Tendencies) == 0 {
		t.Fatal("expected at least one tendency for player 0")
	}

	found := false
	for _, tend := range patterns[0].Tendencies {
		if strings.Contains(tend, "Dominant stroke: forehand") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected dominant forehand tendency, got %v", patterns[0].Tendencies)
	}

	// Player 1 should have no tendencies (no strokes)
	if len(patterns[1].Tendencies) != 0 {
		t.Errorf("expected no tendencies for player 1, got %v", patterns[1].Tendencies)
	}
}
