package tactics

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// MatchRecord holds metadata and tactical report for a single match.
type MatchRecord struct {
	ID        string          `json:"id"`
	Date      string          `json:"date"`
	VideoPath string          `json:"video_path"`
	Score     string          `json:"score"`
	Report    *TacticalReport `json:"report"`
}

// MatchHistory manages persistence of match records to a directory.
type MatchHistory struct {
	dir     string
	records []MatchRecord
}

// NewMatchHistory creates a MatchHistory that stores records in the given directory.
func NewMatchHistory(dir string) *MatchHistory {
	return &MatchHistory{dir: dir}
}

// Save writes a MatchRecord to a JSON file named by its ID.
func (h *MatchHistory) Save(record MatchRecord) error {
	if err := os.MkdirAll(h.dir, 0o755); err != nil {
		return fmt.Errorf("create history dir: %w", err)
	}

	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal record: %w", err)
	}

	path := filepath.Join(h.dir, record.ID+".json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write record: %w", err)
	}

	return nil
}

// List loads all match records from the history directory, sorted newest first by date.
func (h *MatchHistory) List() []MatchRecord {
	entries, err := os.ReadDir(h.dir)
	if err != nil {
		return nil
	}

	var records []MatchRecord
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		data, err := os.ReadFile(filepath.Join(h.dir, entry.Name()))
		if err != nil {
			continue
		}

		var rec MatchRecord
		if err := json.Unmarshal(data, &rec); err != nil {
			continue
		}
		records = append(records, rec)
	}

	// Sort newest first by date string (ISO 8601 sorts lexicographically).
	sort.Slice(records, func(i, j int) bool {
		return records[i].Date > records[j].Date
	})

	h.records = records
	return records
}

// Compare returns the stroke type differences between two matches.
// The returned map keys are stroke types; values are [2]int{match1Count, match2Count}.
func (h *MatchHistory) Compare(id1, id2 string) map[string][2]int {
	records := h.List()

	var r1, r2 *MatchRecord
	for i := range records {
		if records[i].ID == id1 {
			r1 = &records[i]
		}
		if records[i].ID == id2 {
			r2 = &records[i]
		}
	}

	result := make(map[string][2]int)

	if r1 == nil || r2 == nil || r1.Report == nil || r2.Report == nil {
		return result
	}

	// Aggregate stroke breakdowns across all players for each match.
	agg1 := aggregateStrokes(r1.Report)
	agg2 := aggregateStrokes(r2.Report)

	// Collect all stroke types.
	allTypes := make(map[string]bool)
	for k := range agg1 {
		allTypes[k] = true
	}
	for k := range agg2 {
		allTypes[k] = true
	}

	for st := range allTypes {
		result[st] = [2]int{agg1[st], agg2[st]}
	}

	return result
}

// aggregateStrokes sums stroke breakdowns from all players in a report.
func aggregateStrokes(report *TacticalReport) map[string]int {
	agg := make(map[string]int)
	for _, p := range report.Players {
		for k, v := range p.StrokeBreakdown {
			agg[k] += v
		}
	}
	return agg
}
