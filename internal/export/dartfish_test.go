package export

import (
	"bytes"
	"encoding/csv"
	"strings"
	"testing"
)

func TestDartfishColumns_Count(t *testing.T) {
	if got := len(DartfishColumns); got != 62 {
		t.Fatalf("DartfishColumns: want 62 columns, got %d", got)
	}
}

func TestExport_HeaderRow(t *testing.T) {
	exp := NewDartfishExporter()
	var buf bytes.Buffer

	// Export with one data row to verify header + data
	row := ResultRow{Fields: make([]string, 62)}
	row.Fields[0] = "Point 1"
	if err := exp.Export([]ResultRow{row}, &buf); err != nil {
		t.Fatalf("Export: %v", err)
	}

	r := csv.NewReader(strings.NewReader(buf.String()))
	records, err := r.ReadAll()
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}

	if len(records) != 2 {
		t.Fatalf("want 2 rows (header + data), got %d", len(records))
	}

	header := records[0]
	if len(header) != 62 {
		t.Fatalf("header: want 62 columns, got %d", len(header))
	}

	// Verify every column name matches
	for i, want := range DartfishColumns {
		if header[i] != want {
			t.Errorf("header[%d]: want %q, got %q", i, want, header[i])
		}
	}
}

func TestExport_EmptyResults(t *testing.T) {
	exp := NewDartfishExporter()
	var buf bytes.Buffer

	if err := exp.Export(nil, &buf); err != nil {
		t.Fatalf("Export: %v", err)
	}

	r := csv.NewReader(strings.NewReader(buf.String()))
	records, err := r.ReadAll()
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}

	if len(records) != 1 {
		t.Fatalf("want 1 row (header only), got %d", len(records))
	}

	if len(records[0]) != 62 {
		t.Fatalf("header: want 62 columns, got %d", len(records[0]))
	}
}

func TestExport_PadShortRow(t *testing.T) {
	exp := NewDartfishExporter()
	var buf bytes.Buffer

	// Row with only 3 fields — should be padded to 62
	row := ResultRow{Fields: []string{"Point 1", "00:05.000", "3.2s"}}
	if err := exp.Export([]ResultRow{row}, &buf); err != nil {
		t.Fatalf("Export: %v", err)
	}

	r := csv.NewReader(strings.NewReader(buf.String()))
	records, err := r.ReadAll()
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}

	if len(records) != 2 {
		t.Fatalf("want 2 rows, got %d", len(records))
	}

	if len(records[1]) != 62 {
		t.Fatalf("data row: want 62 columns, got %d", len(records[1]))
	}

	if records[1][0] != "Point 1" {
		t.Errorf("data[0]: want %q, got %q", "Point 1", records[1][0])
	}
}

func TestFormatTimestamp(t *testing.T) {
	tests := []struct {
		input float64
		want  string
	}{
		{0, "00:00.000"},
		{65.5, "01:05.500"},
		{3661.123, "61:01.123"},
		{0.001, "00:00.001"},
		{59.999, "00:59.999"},
		{120.0, "02:00.000"},
	}

	for _, tc := range tests {
		got := FormatTimestamp(tc.input)
		if got != tc.want {
			t.Errorf("FormatTimestamp(%v): want %q, got %q", tc.input, tc.want, got)
		}
	}
}

// TestExport_ColumnNamesMatchPython verifies the three E-columns that
// differ from some documentation — the Python source uses "Last Shot"
// prefix on E2, E3, E4.
func TestExport_ColumnNamesMatchPython(t *testing.T) {
	checks := map[int]string{
		17: "E2: Last Shot Winner",
		18: "E3: Last Shot Error",
		19: "E4: Last Shot Placement",
	}

	for idx, want := range checks {
		if DartfishColumns[idx] != want {
			t.Errorf("DartfishColumns[%d]: want %q, got %q", idx, want, DartfishColumns[idx])
		}
	}
}
