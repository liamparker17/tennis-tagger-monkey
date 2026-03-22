package point

// Tennis court dimensions in meters (ITF standard).
const (
	CourtLength     = 23.77  // baseline to baseline
	SinglesWidth    = 8.23   // singles sideline to sideline
	DoublesWidth    = 10.97  // doubles sideline to sideline
	ServiceBoxDepth = 6.40   // net to service line
	NetY            = 11.885 // net position (halfway)
	CenterLineX     = 4.115  // center of singles court
	CloseCallMargin = 0.05   // 5cm margin for "close call"
)

// IsInSingles reports whether (cx, cy) in meters is inside the singles court.
// Lines are considered in.
func IsInSingles(cx, cy float64) bool {
	return cx >= 0 && cx <= SinglesWidth && cy >= 0 && cy <= CourtLength
}

// IsInDoubles reports whether (cx, cy) in meters is inside the doubles court.
func IsInDoubles(cx, cy float64) bool {
	return cx >= 0 && cx <= DoublesWidth && cy >= 0 && cy <= CourtLength
}

// IsInServiceBox checks whether (cx, cy) lands inside the specified service box.
// side is one of: "near_deuce", "near_ad", "far_deuce", "far_ad".
func IsInServiceBox(cx, cy float64, side string) bool {
	switch side {
	case "near_deuce":
		return cx >= CenterLineX && cx <= SinglesWidth &&
			cy >= (NetY-ServiceBoxDepth) && cy <= NetY
	case "near_ad":
		return cx >= 0 && cx <= CenterLineX &&
			cy >= (NetY-ServiceBoxDepth) && cy <= NetY
	case "far_deuce":
		return cx >= 0 && cx <= CenterLineX &&
			cy >= NetY && cy <= (NetY+ServiceBoxDepth)
	case "far_ad":
		return cx >= CenterLineX && cx <= SinglesWidth &&
			cy >= NetY && cy <= (NetY+ServiceBoxDepth)
	}
	return false
}

// ClassifyLanding returns "in", "out", or "close_call" for a bounce at (cx, cy).
func ClassifyLanding(cx, cy float64) string {
	if IsInSingles(cx, cy) {
		return "in"
	}
	if cx >= -CloseCallMargin && cx <= SinglesWidth+CloseCallMargin &&
		cy >= -CloseCallMargin && cy <= CourtLength+CloseCallMargin {
		return "close_call"
	}
	return "out"
}
