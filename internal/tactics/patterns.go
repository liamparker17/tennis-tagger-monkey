package tactics

import (
	"fmt"

	"github.com/liamp/tennis-tagger/internal/bridge"
)

// StrokeStats aggregates stroke counts by type, zone, and depth.
type StrokeStats struct {
	Total   int            `json:"total"`
	ByType  map[string]int `json:"by_type"`
	ByZone  map[string]int `json:"by_zone"`
	ByDepth map[string]int `json:"by_depth"`
}

// PressureStats tracks performance under pressure situations.
type PressureStats struct {
	BreakPointsPlayed int `json:"break_points_played"`
	BreakPointsWon    int `json:"break_points_won"`
	PointsWonOnServe  int `json:"points_won_on_serve"`
	TotalServePoints  int `json:"total_serve_points"`
}

// PlayerPattern holds tactical pattern data for a single player.
type PlayerPattern struct {
	PlayerIndex int           `json:"player_index"`
	Strokes     StrokeStats   `json:"strokes"`
	Pressure    PressureStats `json:"pressure"`
	Tendencies  []string      `json:"tendencies"`
}

// AnalyzePatterns computes per-player stroke statistics and tactical tendencies
// from detection, placement, stroke, and rally data.
// Always returns exactly 2 PlayerPattern entries (player 0 and player 1).
func AnalyzePatterns(
	detections []bridge.DetectionResult,
	placements []bridge.PlacementResult,
	strokes []bridge.StrokeResult,
	rallies []bridge.RallyResult,
) []PlayerPattern {
	patterns := make([]PlayerPattern, 2)
	for i := range patterns {
		patterns[i] = PlayerPattern{
			PlayerIndex: i,
			Strokes: StrokeStats{
				ByType:  make(map[string]int),
				ByZone:  make(map[string]int),
				ByDepth: make(map[string]int),
			},
			Tendencies: []string{},
		}
	}

	// Count strokes by type per player.
	for _, s := range strokes {
		pid := s.PlayerID
		if pid < 0 || pid > 1 {
			continue
		}
		patterns[pid].Strokes.Total++
		patterns[pid].Strokes.ByType[s.Type]++
	}

	// Count placements by zone and depth.
	// Placements are matched to strokes by index when available.
	for i, p := range placements {
		pid := 0
		if i < len(strokes) {
			pid = strokes[i].PlayerID
		}
		if pid < 0 || pid > 1 {
			continue
		}
		if p.Zone != "" {
			patterns[pid].Strokes.ByZone[p.Zone]++
		}
		if p.Depth != "" {
			patterns[pid].Strokes.ByDepth[p.Depth]++
		}
	}

	// Generate tendencies for each player.
	for i := range patterns {
		patterns[i].Tendencies = generateTendencies(&patterns[i])
	}

	return patterns
}

// generateTendencies produces human-readable tendency strings from a player's stats.
func generateTendencies(p *PlayerPattern) []string {
	var tendencies []string

	if p.Strokes.Total == 0 {
		return tendencies
	}

	// Dominant stroke type: >50% of total strokes.
	for strokeType, count := range p.Strokes.ByType {
		pct := float64(count) / float64(p.Strokes.Total) * 100
		if pct > 50 {
			tendencies = append(tendencies, fmt.Sprintf("Dominant stroke: %s (%.0f%%)", strokeType, pct))
		}
	}

	// Targeted zone: >40% of placements.
	totalZone := 0
	for _, count := range p.Strokes.ByZone {
		totalZone += count
	}
	if totalZone > 0 {
		for zone, count := range p.Strokes.ByZone {
			pct := float64(count) / float64(totalZone) * 100
			if pct > 40 {
				tendencies = append(tendencies, fmt.Sprintf("Targets %s (%.0f%%)", zone, pct))
			}
		}
	}

	// Serve point win percentage.
	if p.Pressure.TotalServePoints > 0 {
		pct := float64(p.Pressure.PointsWonOnServe) / float64(p.Pressure.TotalServePoints) * 100
		tendencies = append(tendencies, fmt.Sprintf("Serve point win rate: %.0f%%", pct))
	}

	return tendencies
}
