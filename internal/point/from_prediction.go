package point

import (
	"github.com/liamp/tennis-tagger/internal/pointmodel"
)

// PointFromPrediction converts a fused model prediction for a single clip into a Point
// with shots timed in match-absolute seconds. clipStartS is where the clip begins in
// the source video; fps is the clip's frame rate (used to convert ContactFrame → time).
func PointFromPrediction(pred *pointmodel.FusedPointPrediction, clipStartS, fps float64) *Point {
	if pred == nil {
		return &Point{}
	}
	if fps <= 0 {
		fps = 30.0
	}
	shots := make([]Shot, 0, len(pred.Strokes))
	for _, s := range pred.Strokes {
		shots = append(shots, Shot{
			Index:      s.Index + 1, // 1-based
			Hitter:     s.Hitter,
			StartFrame: s.ContactFrame,
			EndFrame:   s.ContactFrame,
			IsServe:    s.Stroke == "Serve",
			Confidence: s.Prob,
			StrokeType: s.Stroke,
			InCourt:    s.InCourt,
			TimeS:      clipStartS + float64(s.ContactFrame)/fps,
		})
	}
	return &Point{
		Shots:         shots,
		WinnerOrError: pred.Outcome,
		Confidence:    pred.OutcomeProb,
		LowConfidence: pred.LowConfidence,
	}
}
