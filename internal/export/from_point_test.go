package export

import (
	"strings"
	"testing"

	"github.com/liamp/tennis-tagger/internal/point"
)

func TestRowFromPointLowConfPrefix(t *testing.T) {
	pt := &point.Point{
		Number:        3,
		Server:        0,
		WinnerOrError: "Winner",
		LowConfidence: true,
		Shots: []point.Shot{
			{StrokeType: "Serve", TimeS: 10.0},
			{StrokeType: "Forehand", TimeS: 12.5},
		},
	}
	row := RowFromPoint(pt)
	if len(row.Fields) != len(DartfishColumns) {
		t.Fatalf("row len = %d, want %d", len(row.Fields), len(DartfishColumns))
	}
	if !strings.HasPrefix(row.Fields[0], "[LOW_CONF]") {
		t.Errorf("name col missing [LOW_CONF] prefix: %q", row.Fields[0])
	}
	if row.Fields[16] != "Winner" {
		t.Errorf("last shot col = %q, want Winner", row.Fields[16])
	}
	if row.Fields[31] != "2" {
		t.Errorf("stroke count = %q, want 2", row.Fields[31])
	}
	if row.Fields[3] != "P1" {
		t.Errorf("server = %q, want P1", row.Fields[3])
	}
}

func TestRowFromPointHighConf(t *testing.T) {
	pt := &point.Point{Number: 1, LowConfidence: false, WinnerOrError: "Ace",
		Shots: []point.Shot{{StrokeType: "Serve", TimeS: 0}}}
	row := RowFromPoint(pt)
	if strings.Contains(row.Fields[0], "[LOW_CONF]") {
		t.Errorf("high-confidence row should not be prefixed: %q", row.Fields[0])
	}
}
