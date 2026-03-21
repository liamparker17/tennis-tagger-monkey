package tactics

import (
	"fmt"
	"sort"
	"strings"

	"github.com/liamp/tennis-tagger/internal/bridge"
)

// TacticalReport holds the complete tactical analysis for a match.
type TacticalReport struct {
	MatchSummary string         `json:"match_summary"`
	Duration     string         `json:"duration"`
	TotalRallies int            `json:"total_rallies"`
	Players      []PlayerReport `json:"players"`
}

// PlayerReport holds tactical data for a single player within a report.
type PlayerReport struct {
	Label           string         `json:"label"`
	StrokeBreakdown map[string]int `json:"stroke_breakdown"`
	PlacementMap    map[string]int `json:"placement_map"`
	Tendencies      []string       `json:"tendencies"`
}

// GenerateReport creates a TacticalReport from analyzed patterns, rally data, and duration.
func GenerateReport(
	patterns []PlayerPattern,
	rallies []bridge.RallyResult,
	durationSec float64,
) *TacticalReport {
	report := &TacticalReport{
		Duration:     formatDuration(durationSec),
		TotalRallies: len(rallies),
		Players:      make([]PlayerReport, len(patterns)),
	}

	totalStrokes := 0
	for _, p := range patterns {
		totalStrokes += p.Strokes.Total
	}

	report.MatchSummary = fmt.Sprintf("%d rallies, %d total strokes in %s",
		len(rallies), totalStrokes, report.Duration)

	for i, p := range patterns {
		report.Players[i] = PlayerReport{
			Label:           fmt.Sprintf("Player %d", i+1),
			StrokeBreakdown: copyMap(p.Strokes.ByType),
			PlacementMap:    copyMap(p.Strokes.ByZone),
			Tendencies:      p.Tendencies,
		}
	}

	return report
}

// FormatText returns a human-readable multi-line summary of the report.
func (r *TacticalReport) FormatText() string {
	var sb strings.Builder

	sb.WriteString("=== TACTICAL REPORT ===\n")
	sb.WriteString(fmt.Sprintf("Duration: %s | Rallies: %d\n", r.Duration, r.TotalRallies))

	for _, player := range r.Players {
		sb.WriteString(fmt.Sprintf("\n--- %s ---\n", player.Label))

		// Stroke breakdown sorted alphabetically for deterministic output.
		if len(player.StrokeBreakdown) > 0 {
			sb.WriteString("Strokes: ")
			keys := sortedKeys(player.StrokeBreakdown)
			parts := make([]string, len(keys))
			for i, k := range keys {
				parts[i] = fmt.Sprintf("%s(%d)", k, player.StrokeBreakdown[k])
			}
			sb.WriteString(strings.Join(parts, " "))
			sb.WriteString("\n")
		}

		// Placement map.
		if len(player.PlacementMap) > 0 {
			sb.WriteString("Placements: ")
			keys := sortedKeys(player.PlacementMap)
			parts := make([]string, len(keys))
			for i, k := range keys {
				parts[i] = fmt.Sprintf("%s(%d)", k, player.PlacementMap[k])
			}
			sb.WriteString(strings.Join(parts, " "))
			sb.WriteString("\n")
		}

		// Tendencies.
		if len(player.Tendencies) > 0 {
			sb.WriteString("Tendencies:\n")
			for _, tend := range player.Tendencies {
				sb.WriteString(fmt.Sprintf("  - %s\n", tend))
			}
		}
	}

	return sb.String()
}

// formatDuration converts seconds to a human-readable "Xm XXs" string.
func formatDuration(seconds float64) string {
	if seconds < 0 {
		seconds = 0
	}
	totalSec := int(seconds)
	m := totalSec / 60
	s := totalSec % 60
	if m > 0 {
		return fmt.Sprintf("%dm %02ds", m, s)
	}
	return fmt.Sprintf("%ds", s)
}

// copyMap returns a shallow copy of a string->int map.
func copyMap(m map[string]int) map[string]int {
	out := make(map[string]int, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// sortedKeys returns the keys of a map in sorted order.
func sortedKeys(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
