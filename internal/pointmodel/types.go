package pointmodel

type FusedStroke struct {
	Index        int     `json:"index"`
	Stroke       string  `json:"stroke"`
	Hitter       int     `json:"hitter"`
	InCourt      bool    `json:"in_court"`
	Prob         float64 `json:"prob"`
	ContactFrame int     `json:"contact_frame"`
}

type FusedPointPrediction struct {
	ContactFrames []int         `json:"contact_frames"`
	BounceFrames  []int         `json:"bounce_frames"`
	Strokes       []FusedStroke `json:"strokes"`
	Outcome       string        `json:"outcome"`
	OutcomeProb   float64       `json:"outcome_prob"`
	LowConfidence bool          `json:"low_confidence"`
}
