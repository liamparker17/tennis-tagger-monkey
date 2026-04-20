package point

// PointOutcome describes how a point ended.
type PointOutcome struct {
	Winner     int     // 0 = near player, 1 = far player
	Category   string  // "winner", "unforced_error", "ace", "double_fault", "error"
	Confidence float64
}

// Point holds all data for a single point.
type Point struct {
	Number    int
	Shots     []Shot
	Outcome   PointOutcome
	Server    int
	ServeSide string

	// Fields populated by the Plan 3 multi-task model via PointFromPrediction.
	WinnerOrError string  // "Ace","DoubleFault","Winner","ForcedError","UnforcedError"
	Confidence    float64 // fused outcome probability
	LowConfidence bool    // any head below its low-conf threshold
}

// minConfidence returns the minimum Confidence across all shots, or 1.0 if
// there are no shots.
func minConfidence(shots []Shot) float64 {
	if len(shots) == 0 {
		return 1.0
	}
	min := shots[0].Confidence
	for _, s := range shots[1:] {
		if s.Confidence < min {
			min = s.Confidence
		}
	}
	return min
}

// RecognizePoint analyzes a sequence of shots and determines the point outcome.
//
// Conditions are checked in order:
//  1. Double fault  – first two shots are both serves, both bounces out.
//  2. Ace           – single serve shot, bounce inside the correct service box.
//  3. Ball out      – last shot's bounce InOut == "out" → unforced_error by hitter.
//  4. Ball into net – last shot has nil Bounce → error by hitter.
//  5. Winner        – last shot's bounce InOut == "in", no subsequent shot.
//  6. Close call    – last shot's bounce InOut == "close_call" → unforced_error, lower confidence.
//  7. Unknown       – confidence = 0.
func RecognizePoint(shots []Shot, server int, serveSide string) Point {
	p := Point{
		Shots:     shots,
		Server:    server,
		ServeSide: serveSide,
	}

	base := minConfidence(shots)

	// 1. Double fault: first two shots both serves with out bounces.
	if len(shots) >= 2 && shots[0].IsServe && shots[1].IsServe {
		b0 := shots[0].Bounce
		b1 := shots[1].Bounce
		if b0 != nil && b1 != nil && b0.InOut == "out" && b1.InOut == "out" {
			p.Outcome = PointOutcome{
				Winner:     1 - server,
				Category:   "double_fault",
				Confidence: base,
			}
			return p
		}
	}

	// 2. Ace: single serve shot, bounce inside the correct service box.
	if len(shots) == 1 && shots[0].IsServe {
		b := shots[0].Bounce
		if b != nil && b.InOut == "in" {
			if isCorrectServiceBox(b.CX, b.CY, server, serveSide) {
				p.Outcome = PointOutcome{
					Winner:     server,
					Category:   "ace",
					Confidence: base,
				}
				return p
			}
		}
	}

	if len(shots) == 0 {
		p.Outcome = PointOutcome{Confidence: 0}
		return p
	}

	last := shots[len(shots)-1]

	// 4. Ball into net: nil bounce (checked before InOut branches).
	if last.Bounce == nil {
		p.Outcome = PointOutcome{
			Winner:     1 - last.Hitter,
			Category:   "error",
			Confidence: base,
		}
		return p
	}

	// 3. Ball out.
	if last.Bounce.InOut == "out" {
		p.Outcome = PointOutcome{
			Winner:     1 - last.Hitter,
			Category:   "unforced_error",
			Confidence: base,
		}
		return p
	}

	// 5. Winner: bounce in, no subsequent shot → this IS the last shot.
	if last.Bounce.InOut == "in" {
		p.Outcome = PointOutcome{
			Winner:     last.Hitter,
			Category:   "winner",
			Confidence: base,
		}
		return p
	}

	// 6. Close call: like out but reduced confidence.
	if last.Bounce.InOut == "close_call" {
		p.Outcome = PointOutcome{
			Winner:     1 - last.Hitter,
			Category:   "unforced_error",
			Confidence: base * 0.5,
		}
		return p
	}

	// 7. Unknown.
	p.Outcome = PointOutcome{Confidence: 0}
	return p
}

// isCorrectServiceBox returns true when (cx, cy) lands in the service box
// that the server must target given the serve side.
//
// Serve-side to target box mapping:
//   - server=0 (near), side="deuce" → far_deuce
//   - server=0 (near), side="ad"    → far_ad
//   - server=1 (far),  side="deuce" → near_deuce
//   - server=1 (far),  side="ad"    → near_ad
func isCorrectServiceBox(cx, cy float64, server int, serveSide string) bool {
	var target string
	if server == 0 {
		if serveSide == "deuce" {
			target = "far_deuce"
		} else {
			target = "far_ad"
		}
	} else {
		if serveSide == "deuce" {
			target = "near_deuce"
		} else {
			target = "near_ad"
		}
	}
	return IsInServiceBox(cx, cy, target)
}

// RecognizePoints takes groups of shots (each group = one point's shots) and
// processes them sequentially. For each group it calls RecognizePoint, feeds
// the outcome to MatchState.AwardPoint, and returns the list of Points and the
// final MatchState.
func RecognizePoints(shotGroups [][]Shot, server int) ([]Point, *MatchState) {
	ms := NewMatchState(server)
	points := make([]Point, 0, len(shotGroups))

	for i, shots := range shotGroups {
		p := RecognizePoint(shots, ms.Server, ms.ServeSide)
		p.Number = i + 1
		points = append(points, p)
		ms.AwardPoint(p.Outcome.Winner)
	}

	return points, ms
}
