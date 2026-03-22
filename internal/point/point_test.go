package point

import (
	"testing"

	"github.com/liamp/tennis-tagger/internal/bridge"
)

// makeBounce is a helper that constructs a *bridge.Bounce.
func makeBounce(cx, cy float64, inOut string) *bridge.Bounce {
	return &bridge.Bounce{CX: cx, CY: cy, InOut: inOut}
}

// makeShot is a helper that constructs a Shot with the given fields.
func makeShot(index, hitter int, isServe bool, b *bridge.Bounce) Shot {
	return Shot{
		Index:      index,
		Hitter:     hitter,
		IsServe:    isServe,
		Bounce:     b,
		Confidence: 0.9,
	}
}

// -----------------------------------------------------------------------
// TestRecognizeAce: single serve shot, bounce inside service box → ace
// -----------------------------------------------------------------------
func TestRecognizeAce(t *testing.T) {
	// Server is player 0 (near). Far deuce service box: cx 0..CenterLineX, cy NetY..NetY+ServiceBoxDepth
	b := makeBounce(2.0, NetY+1.0, "in") // inside far_deuce box
	shots := []Shot{makeShot(1, 0, true, b)}

	p := RecognizePoint(shots, 0, "deuce")

	if p.Outcome.Category != "ace" {
		t.Errorf("expected category=ace, got %q", p.Outcome.Category)
	}
	if p.Outcome.Winner != 0 {
		t.Errorf("expected Winner=0 (server), got %d", p.Outcome.Winner)
	}
	if p.Outcome.Confidence <= 0 {
		t.Errorf("expected positive confidence, got %f", p.Outcome.Confidence)
	}
}

// -----------------------------------------------------------------------
// TestRecognizeDoubleFault: two serve shots both out → double fault
// -----------------------------------------------------------------------
func TestRecognizeDoubleFault(t *testing.T) {
	// Both bounces are outside the service box (InOut == "out")
	b1 := makeBounce(1.0, NetY+7.0, "out") // deep out
	b2 := makeBounce(1.0, NetY+7.0, "out")
	shots := []Shot{
		makeShot(1, 0, true, b1),
		makeShot(2, 0, true, b2),
	}

	p := RecognizePoint(shots, 0, "deuce")

	if p.Outcome.Category != "double_fault" {
		t.Errorf("expected category=double_fault, got %q", p.Outcome.Category)
	}
	// Point goes to returner = 1 - server(0) = 1
	if p.Outcome.Winner != 1 {
		t.Errorf("expected Winner=1 (returner), got %d", p.Outcome.Winner)
	}
}

// -----------------------------------------------------------------------
// TestRecognizeBallOut: rally of 3 shots, last bounce is out → unforced error
// -----------------------------------------------------------------------
func TestRecognizeBallOut(t *testing.T) {
	shots := []Shot{
		makeShot(1, 0, true, makeBounce(2.0, NetY+1.0, "in")),  // serve in
		makeShot(2, 1, false, makeBounce(3.0, NetY-2.0, "in")), // return in
		makeShot(3, 0, false, makeBounce(9.0, NetY+2.0, "out")), // hitter 0 hits out
	}

	p := RecognizePoint(shots, 0, "deuce")

	if p.Outcome.Category != "unforced_error" {
		t.Errorf("expected category=unforced_error, got %q", p.Outcome.Category)
	}
	// Last hitter was 0, so point goes to 1
	if p.Outcome.Winner != 1 {
		t.Errorf("expected Winner=1 (opponent of hitter 0), got %d", p.Outcome.Winner)
	}
}

// -----------------------------------------------------------------------
// TestRecognizeWinner: rally of 3 shots, last bounce is in, no return → winner
// -----------------------------------------------------------------------
func TestRecognizeWinner(t *testing.T) {
	shots := []Shot{
		makeShot(1, 0, true, makeBounce(2.0, NetY+1.0, "in")),  // serve in
		makeShot(2, 1, false, makeBounce(3.0, NetY-2.0, "in")), // return in
		makeShot(3, 0, false, makeBounce(2.0, NetY+3.0, "in")), // winner by hitter 0
	}

	p := RecognizePoint(shots, 0, "deuce")

	if p.Outcome.Category != "winner" {
		t.Errorf("expected category=winner, got %q", p.Outcome.Category)
	}
	// Last hitter was 0
	if p.Outcome.Winner != 0 {
		t.Errorf("expected Winner=0 (hitter), got %d", p.Outcome.Winner)
	}
}

// -----------------------------------------------------------------------
// TestRecognizeBallIntoNet: shot with nil bounce → ball into net, error by hitter
// -----------------------------------------------------------------------
func TestRecognizeBallIntoNet(t *testing.T) {
	shots := []Shot{
		makeShot(1, 0, true, makeBounce(2.0, NetY+1.0, "in")),
		makeShot(2, 1, false, makeBounce(3.0, NetY-2.0, "in")),
		makeShot(3, 0, false, nil), // no bounce = net
	}

	p := RecognizePoint(shots, 0, "deuce")

	if p.Outcome.Category != "error" {
		t.Errorf("expected category=error, got %q", p.Outcome.Category)
	}
	// Last hitter was 0, point goes to 1
	if p.Outcome.Winner != 1 {
		t.Errorf("expected Winner=1, got %d", p.Outcome.Winner)
	}
}

// -----------------------------------------------------------------------
// TestRecognizeCloseCall: last bounce is "close_call" → like out but lower confidence
// -----------------------------------------------------------------------
func TestRecognizeCloseCall(t *testing.T) {
	shots := []Shot{
		makeShot(1, 0, true, makeBounce(2.0, NetY+1.0, "in")),
		makeShot(2, 1, false, makeBounce(3.0, NetY-2.0, "in")),
		makeShot(3, 0, false, makeBounce(8.24, NetY+2.0, "close_call")), // just outside singles
	}

	p := RecognizePoint(shots, 0, "deuce")

	if p.Outcome.Category != "unforced_error" {
		t.Errorf("expected category=unforced_error for close_call, got %q", p.Outcome.Category)
	}
	// Confidence should be reduced (lower than full shot confidence)
	if p.Outcome.Confidence >= 0.9 {
		t.Errorf("expected reduced confidence for close_call, got %f", p.Outcome.Confidence)
	}
	if p.Outcome.Winner != 1 {
		t.Errorf("expected Winner=1 (opponent of hitter 0), got %d", p.Outcome.Winner)
	}
}

// -----------------------------------------------------------------------
// TestRecognizePointsSequence: process 4 points, verify score updates correctly
// -----------------------------------------------------------------------
func TestRecognizePointsSequence(t *testing.T) {
	// 4 groups of shots; player 0 wins first 3, player 1 wins the 4th.
	shotGroups := [][]Shot{
		// Point 1: ace by server (player 0)
		{makeShot(1, 0, true, makeBounce(2.0, NetY+1.0, "in"))},
		// Point 2: player 0 winner
		{
			makeShot(1, 0, true, makeBounce(2.0, NetY+1.0, "in")),
			makeShot(2, 1, false, makeBounce(3.0, NetY-2.0, "in")),
			makeShot(3, 0, false, makeBounce(2.0, NetY+3.0, "in")),
		},
		// Point 3: player 0 winner
		{
			makeShot(1, 0, true, makeBounce(2.0, NetY+1.0, "in")),
			makeShot(2, 1, false, makeBounce(3.0, NetY-2.0, "in")),
			makeShot(3, 0, false, makeBounce(2.0, NetY+3.0, "in")),
		},
		// Point 4: player 1 wins (player 0 hits out)
		{
			makeShot(1, 0, true, makeBounce(2.0, NetY+1.0, "in")),
			makeShot(2, 1, false, makeBounce(3.0, NetY-2.0, "in")),
			makeShot(3, 0, false, makeBounce(9.0, NetY+2.0, "out")),
		},
	}

	points, ms := RecognizePoints(shotGroups, 0)

	if len(points) != 4 {
		t.Fatalf("expected 4 points, got %d", len(points))
	}

	// Points 1-3 won by player 0, point 4 by player 1.
	expectedWinners := []int{0, 0, 0, 1}
	for i, p := range points {
		if p.Outcome.Winner != expectedWinners[i] {
			t.Errorf("point %d: expected Winner=%d, got %d", i+1, expectedWinners[i], p.Outcome.Winner)
		}
		if p.Number != i+1 {
			t.Errorf("point %d: expected Number=%d, got %d", i+1, i+1, p.Number)
		}
	}

	// After 3 wins for player 0 and 1 win for player 1:
	// Score should be 40-15 (0 wins 3, 1 wins 1 → 40 vs 15).
	if ms == nil {
		t.Fatal("expected non-nil MatchState")
	}
	if ms.PointScore[0] != 40 {
		t.Errorf("expected PointScore[0]=40 after 3 wins, got %d", ms.PointScore[0])
	}
	if ms.PointScore[1] != 15 {
		t.Errorf("expected PointScore[1]=15 after 1 win, got %d", ms.PointScore[1])
	}
	if ms.PointNumber != 4 {
		t.Errorf("expected PointNumber=4, got %d", ms.PointNumber)
	}
}
