package point

import (
	"testing"

	"github.com/liamp/tennis-tagger/internal/pointmodel"
)

func TestPointFromPrediction(t *testing.T) {
	pred := &pointmodel.FusedPointPrediction{
		ContactFrames: []int{10, 40, 80},
		Strokes: []pointmodel.FusedStroke{
			{Index: 0, Stroke: "Serve", Hitter: 0, InCourt: true, ContactFrame: 10, Prob: 0.94},
			{Index: 1, Stroke: "Forehand", Hitter: 1, InCourt: true, ContactFrame: 40, Prob: 0.82},
			{Index: 2, Stroke: "Backhand", Hitter: 0, InCourt: true, ContactFrame: 80, Prob: 0.65},
		},
		Outcome: "Winner", OutcomeProb: 0.71,
	}
	pt := PointFromPrediction(pred, 12.0, 30.0)
	if len(pt.Shots) != 3 {
		t.Fatalf("want 3 shots, got %d", len(pt.Shots))
	}
	if pt.Shots[0].StrokeType != "Serve" {
		t.Errorf("shot 0 stroke = %q, want Serve", pt.Shots[0].StrokeType)
	}
	if !pt.Shots[0].IsServe {
		t.Errorf("shot 0 IsServe = false, want true")
	}
	if pt.WinnerOrError != "Winner" {
		t.Errorf("outcome = %q", pt.WinnerOrError)
	}
	wantT := 12.0 + 10.0/30.0
	if pt.Shots[0].TimeS < wantT-1e-6 || pt.Shots[0].TimeS > wantT+1e-6 {
		t.Errorf("TimeS = %v, want %v", pt.Shots[0].TimeS, wantT)
	}
}

func TestPointFromPredictionNil(t *testing.T) {
	pt := PointFromPrediction(nil, 0, 30)
	if pt == nil || len(pt.Shots) != 0 {
		t.Fatalf("nil pred should produce empty Point")
	}
}
