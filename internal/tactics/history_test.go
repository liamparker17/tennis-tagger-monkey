package tactics

import (
	"os"
	"testing"
)

func TestHistory_SaveAndList(t *testing.T) {
	dir, err := os.MkdirTemp("", "tactics-history-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	h := NewMatchHistory(dir)

	rec1 := MatchRecord{
		ID:        "match-001",
		Date:      "2026-03-20",
		VideoPath: "/video/match1.mp4",
		Score:     "6-4 6-3",
		Report: &TacticalReport{
			Duration:     "1h 20m",
			TotalRallies: 10,
			Players: []PlayerReport{
				{Label: "Player 1", StrokeBreakdown: map[string]int{"forehand": 5, "backhand": 3}},
				{Label: "Player 2", StrokeBreakdown: map[string]int{"forehand": 4, "serve": 2}},
			},
		},
	}
	rec2 := MatchRecord{
		ID:        "match-002",
		Date:      "2026-03-22",
		VideoPath: "/video/match2.mp4",
		Score:     "7-5 6-7 6-2",
		Report: &TacticalReport{
			Duration:     "2h 10m",
			TotalRallies: 15,
			Players: []PlayerReport{
				{Label: "Player 1", StrokeBreakdown: map[string]int{"forehand": 8, "backhand": 5}},
				{Label: "Player 2", StrokeBreakdown: map[string]int{"forehand": 7, "serve": 4}},
			},
		},
	}

	if err := h.Save(rec1); err != nil {
		t.Fatalf("Save rec1: %v", err)
	}
	if err := h.Save(rec2); err != nil {
		t.Fatalf("Save rec2: %v", err)
	}

	records := h.List()
	if len(records) != 2 {
		t.Fatalf("expected 2 records, got %d", len(records))
	}
	// Newest first
	if records[0].ID != "match-002" {
		t.Errorf("expected newest first, got %s", records[0].ID)
	}
	if records[1].ID != "match-001" {
		t.Errorf("expected oldest second, got %s", records[1].ID)
	}
}

func TestHistory_Compare(t *testing.T) {
	dir, err := os.MkdirTemp("", "tactics-compare-*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	h := NewMatchHistory(dir)

	rec1 := MatchRecord{
		ID:   "m1",
		Date: "2026-01-01",
		Report: &TacticalReport{
			Players: []PlayerReport{
				{StrokeBreakdown: map[string]int{"forehand": 10, "backhand": 5}},
			},
		},
	}
	rec2 := MatchRecord{
		ID:   "m2",
		Date: "2026-02-01",
		Report: &TacticalReport{
			Players: []PlayerReport{
				{StrokeBreakdown: map[string]int{"forehand": 7, "serve": 3}},
			},
		},
	}

	_ = h.Save(rec1)
	_ = h.Save(rec2)

	diff := h.Compare("m1", "m2")

	if diff["forehand"] != [2]int{10, 7} {
		t.Errorf("forehand diff = %v, want [10 7]", diff["forehand"])
	}
	if diff["backhand"] != [2]int{5, 0} {
		t.Errorf("backhand diff = %v, want [5 0]", diff["backhand"])
	}
	if diff["serve"] != [2]int{0, 3} {
		t.Errorf("serve diff = %v, want [0 3]", diff["serve"])
	}
}
