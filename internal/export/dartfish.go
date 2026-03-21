package export

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
)

// ResultRow holds one row of Dartfish export data.
// Fields is indexed by DartfishColumns position (0–61).
type ResultRow struct {
	Fields []string
}

// DartfishExporter writes ResultRows as Dartfish-compatible CSV.
type DartfishExporter struct{}

// NewDartfishExporter creates a new DartfishExporter.
func NewDartfishExporter() *DartfishExporter {
	return &DartfishExporter{}
}

// Export writes a CSV (header + data rows) to w.
func (e *DartfishExporter) Export(rows []ResultRow, w io.Writer) error {
	cw := csv.NewWriter(w)

	// Write header row
	if err := cw.Write(DartfishColumns); err != nil {
		return fmt.Errorf("writing header: %w", err)
	}

	// Write data rows
	for i, row := range rows {
		fields := row.Fields

		// Pad or truncate to match column count
		if len(fields) < len(DartfishColumns) {
			padded := make([]string, len(DartfishColumns))
			copy(padded, fields)
			fields = padded
		} else if len(fields) > len(DartfishColumns) {
			fields = fields[:len(DartfishColumns)]
		}

		if err := cw.Write(fields); err != nil {
			return fmt.Errorf("writing row %d: %w", i, err)
		}
	}

	cw.Flush()
	return cw.Error()
}

// ExportFile writes a CSV to the given file path.
func (e *DartfishExporter) ExportFile(rows []ResultRow, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating file %s: %w", path, err)
	}

	if err := e.Export(rows, f); err != nil {
		f.Close()
		return err
	}

	return f.Close()
}

// FormatTimestamp converts seconds to MM:SS.mmm format.
// Minutes are not capped at 59 — they accumulate (e.g. 3661.123 → "61:01.123").
func FormatTimestamp(seconds float64) string {
	totalMs := math.Round(seconds * 1000)
	totalSec := int(totalMs / 1000)
	ms := int(totalMs) % 1000
	if ms < 0 {
		ms = -ms
	}

	minutes := totalSec / 60
	secs := totalSec % 60

	return fmt.Sprintf("%02d:%02d.%03d", minutes, secs, ms)
}
