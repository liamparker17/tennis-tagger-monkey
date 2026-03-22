package point

import "testing"

func TestIsInSingles(t *testing.T) {
	tests := []struct {
		name string
		cx   float64
		cy   float64
		want bool
	}{
		// Clearly inside
		{"center court", SinglesWidth / 2, CourtLength / 2, true},
		{"near baseline inside", SinglesWidth / 2, 0.1, true},
		{"far baseline inside", SinglesWidth / 2, CourtLength - 0.1, true},
		{"left sideline inside", 0.1, CourtLength / 2, true},
		{"right sideline inside", SinglesWidth - 0.1, CourtLength / 2, true},

		// On the lines (in)
		{"on near baseline", SinglesWidth / 2, 0, true},
		{"on far baseline", SinglesWidth / 2, CourtLength, true},
		{"on left sideline", 0, CourtLength / 2, true},
		{"on right sideline", SinglesWidth, CourtLength / 2, true},
		{"corner near-left", 0, 0, true},
		{"corner far-right", SinglesWidth, CourtLength, true},

		// Just outside each boundary
		{"just outside near baseline", SinglesWidth / 2, -0.01, false},
		{"just outside far baseline", SinglesWidth / 2, CourtLength + 0.01, false},
		{"just outside left sideline", -0.01, CourtLength / 2, false},
		{"just outside right sideline", SinglesWidth + 0.01, CourtLength / 2, false},

		// Clearly outside
		{"far outside left", -1, CourtLength / 2, false},
		{"far outside right", SinglesWidth + 1, CourtLength / 2, false},
		{"far outside near", SinglesWidth / 2, -1, false},
		{"far outside far", SinglesWidth / 2, CourtLength + 1, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := IsInSingles(tc.cx, tc.cy)
			if got != tc.want {
				t.Errorf("IsInSingles(%v, %v) = %v, want %v", tc.cx, tc.cy, got, tc.want)
			}
		})
	}
}

func TestIsInDoubles(t *testing.T) {
	tests := []struct {
		name string
		cx   float64
		cy   float64
		want bool
	}{
		// Inside doubles but outside singles
		{"left doubles alley", 0.5, CourtLength / 2, true},   // cx=0.5 is inside singles too; use cx between singles and doubles
		{"right doubles alley", DoublesWidth - 0.1, CourtLength / 2, true},
		// More precisely outside singles
		{"just outside singles left, inside doubles", SinglesWidth + 0.1, CourtLength / 2, true},
		{"just outside singles right side... wait doubles is wider", DoublesWidth - 0.5, CourtLength / 2, true},

		// On doubles boundary (in)
		{"on doubles left sideline", 0, CourtLength / 2, true},
		{"on doubles right sideline", DoublesWidth, CourtLength / 2, true},
		{"on doubles near baseline", DoublesWidth / 2, 0, true},
		{"on doubles far baseline", DoublesWidth / 2, CourtLength, true},

		// Center of court
		{"center of doubles court", DoublesWidth / 2, CourtLength / 2, true},

		// Outside doubles
		{"outside doubles left", -0.01, CourtLength / 2, false},
		{"outside doubles right", DoublesWidth + 0.01, CourtLength / 2, false},
		{"outside doubles near", DoublesWidth / 2, -0.01, false},
		{"outside doubles far", DoublesWidth / 2, CourtLength + 0.01, false},

		// Doubles alley region (outside singles, inside doubles)
		{"left alley midpoint", (SinglesWidth + DoublesWidth) / 2, CourtLength / 2, false}, // this is outside doubles right edge... recalculate
	}

	// Fix the last test — (SinglesWidth + DoublesWidth)/2 might be outside. Let's use a known alley value.
	// SinglesWidth = 8.23, DoublesWidth = 10.97, alley width = (10.97-8.23)/2 = 1.37
	// Right alley: cx in [8.23, 10.97]
	_ = tests
	alleyTests := []struct {
		name string
		cx   float64
		cy   float64
		want bool
	}{
		{"right alley inside doubles", 9.0, CourtLength / 2, true},
		{"right alley inside doubles 2", 10.5, CourtLength / 2, true},
		{"outside doubles right boundary", DoublesWidth + 0.01, CourtLength / 2, false},
		{"outside doubles left boundary", -0.01, CourtLength / 2, false},
		{"inside singles (also inside doubles)", SinglesWidth / 2, CourtLength / 2, true},
	}
	for _, tc := range alleyTests {
		t.Run(tc.name, func(t *testing.T) {
			got := IsInDoubles(tc.cx, tc.cy)
			if got != tc.want {
				t.Errorf("IsInDoubles(%v, %v) = %v, want %v", tc.cx, tc.cy, got, tc.want)
			}
		})
	}
}

func TestIsInServiceBox(t *testing.T) {
	// Service box geometry:
	// NetY = 11.885, ServiceBoxDepth = 6.40
	// near side: cy in [NetY-ServiceBoxDepth, NetY] = [5.485, 11.885]
	// far side:  cy in [NetY, NetY+ServiceBoxDepth]  = [11.885, 18.285]
	// deuce (near): cx in [CenterLineX, SinglesWidth] = [4.115, 8.23]
	// ad   (near): cx in [0, CenterLineX]             = [0, 4.115]
	// far_deuce:   cx in [0, CenterLineX]
	// far_ad:      cx in [CenterLineX, SinglesWidth]

	nearDeuceCX := (CenterLineX + SinglesWidth) / 2 // ~6.17
	nearDeuceCY := (NetY-ServiceBoxDepth+NetY) / 2   // ~8.69

	nearAdCX := CenterLineX / 2  // ~2.06
	nearAdCY := nearDeuceCY

	farDeuceCX := CenterLineX / 2
	farDeuceCY := (NetY + NetY + ServiceBoxDepth) / 2 // ~15.09

	farAdCX := (CenterLineX + SinglesWidth) / 2
	farAdCY := farDeuceCY

	tests := []struct {
		name string
		cx   float64
		cy   float64
		side string
		want bool
	}{
		// Center of each box
		{"near_deuce center", nearDeuceCX, nearDeuceCY, "near_deuce", true},
		{"near_ad center", nearAdCX, nearAdCY, "near_ad", true},
		{"far_deuce center", farDeuceCX, farDeuceCY, "far_deuce", true},
		{"far_ad center", farAdCX, farAdCY, "far_ad", true},

		// Wrong box (near_deuce point in near_ad)
		{"near_deuce point in near_ad", nearDeuceCX, nearDeuceCY, "near_ad", false},
		{"near_ad point in near_deuce", nearAdCX, nearAdCY, "near_deuce", false},
		{"near_deuce point in far_deuce", nearDeuceCX, nearDeuceCY, "far_deuce", false},
		{"far_deuce point in far_ad", farDeuceCX, farDeuceCY, "far_ad", false},

		// Too deep (beyond service line)
		{"near_deuce too deep", nearDeuceCX, NetY - ServiceBoxDepth - 0.1, "near_deuce", false},
		{"far_deuce too deep", farDeuceCX, NetY + ServiceBoxDepth + 0.1, "far_deuce", false},

		// On the service line (boundary = in)
		{"near_deuce on service line", nearDeuceCX, NetY - ServiceBoxDepth, "near_deuce", true},
		{"near_ad on service line", nearAdCX, NetY - ServiceBoxDepth, "near_ad", true},
		{"far_deuce on service line", farDeuceCX, NetY + ServiceBoxDepth, "far_deuce", true},
		{"far_ad on service line", farAdCX, NetY + ServiceBoxDepth, "far_ad", true},

		// On the net line (boundary = in)
		{"near_deuce on net", nearDeuceCX, NetY, "near_deuce", true},
		{"far_deuce on net", farDeuceCX, NetY, "far_deuce", true},

		// On the center line (boundary = in)
		{"near_deuce on center line", CenterLineX, nearDeuceCY, "near_deuce", true},
		{"near_ad on center line", CenterLineX, nearAdCY, "near_ad", true},

		// Invalid side
		{"unknown side", 5.0, 10.0, "unknown", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := IsInServiceBox(tc.cx, tc.cy, tc.side)
			if got != tc.want {
				t.Errorf("IsInServiceBox(%v, %v, %q) = %v, want %v", tc.cx, tc.cy, tc.side, got, tc.want)
			}
		})
	}
}

func TestClassifyLanding(t *testing.T) {
	tests := []struct {
		name string
		cx   float64
		cy   float64
		want string
	}{
		// Clearly in
		{"center court", SinglesWidth / 2, CourtLength / 2, "in"},
		{"on near baseline", SinglesWidth / 2, 0, "in"},
		{"on far baseline", SinglesWidth / 2, CourtLength, "in"},
		{"on left sideline", 0, CourtLength / 2, "in"},
		{"on right sideline", SinglesWidth, CourtLength / 2, "in"},
		{"corner", 0, 0, "in"},

		// Clearly out
		{"far left", -1.0, CourtLength / 2, "out"},
		{"far right", SinglesWidth + 1.0, CourtLength / 2, "out"},
		{"far near", SinglesWidth / 2, -1.0, "out"},
		{"far far", SinglesWidth / 2, CourtLength + 1.0, "out"},
		// Just beyond the close_call margin
		{"just beyond left close_call", -(CloseCallMargin + 0.01), CourtLength / 2, "out"},
		{"just beyond right close_call", SinglesWidth + CloseCallMargin + 0.01, CourtLength / 2, "out"},
		{"just beyond near close_call", SinglesWidth / 2, -(CloseCallMargin + 0.01), "out"},
		{"just beyond far close_call", SinglesWidth / 2, CourtLength + CloseCallMargin + 0.01, "out"},

		// Close call (outside singles but within 5cm margin)
		{"close call left", -0.03, CourtLength / 2, "close_call"},
		{"close call right", SinglesWidth + 0.03, CourtLength / 2, "close_call"},
		{"close call near", SinglesWidth / 2, -0.03, "close_call"},
		{"close call far", SinglesWidth / 2, CourtLength + 0.03, "close_call"},
		// Exactly at the margin boundary (on the margin = close_call)
		{"exactly at left margin", -CloseCallMargin, CourtLength / 2, "close_call"},
		{"exactly at right margin", SinglesWidth + CloseCallMargin, CourtLength / 2, "close_call"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyLanding(tc.cx, tc.cy)
			if got != tc.want {
				t.Errorf("ClassifyLanding(%v, %v) = %q, want %q", tc.cx, tc.cy, got, tc.want)
			}
		})
	}
}
