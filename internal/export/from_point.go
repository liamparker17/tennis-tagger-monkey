package export

import (
	"fmt"

	"github.com/liamp/tennis-tagger/internal/point"
)

// RowFromPoint builds a Dartfish ResultRow from a model-produced Point.
// Only the columns we can fill from the fused prediction are populated — everything
// else is left blank so downstream tooling that reads the full 116-col schema still parses.
//
// Low-confidence rows are prefixed with "[LOW_CONF]" in the Name column so humans can
// spot-check the model's weakest predictions quickly.
func RowFromPoint(pt *point.Point) ResultRow {
	fields := make([]string, len(DartfishColumns))
	if pt == nil {
		return ResultRow{Fields: fields}
	}

	namePrefix := ""
	if pt.LowConfidence {
		namePrefix = "[LOW_CONF] "
	}
	fields[0] = fmt.Sprintf("%sPoint %d", namePrefix, pt.Number)

	if len(pt.Shots) > 0 {
		fields[1] = FormatTimestamp(pt.Shots[0].TimeS)
		last := pt.Shots[len(pt.Shots)-1]
		fields[2] = FormatTimestamp(last.TimeS - pt.Shots[0].TimeS)
	}

	if pt.Server == 0 {
		fields[3] = "P1"
	} else if pt.Server == 1 {
		fields[3] = "P2"
	}

	fields[16] = pt.WinnerOrError
	switch pt.WinnerOrError {
	case "Winner":
		fields[17] = "Winner"
	case "UnforcedError", "ForcedError":
		fields[18] = pt.WinnerOrError
	}

	fields[31] = fmt.Sprintf("%d", len(pt.Shots))
	return ResultRow{Fields: fields}
}
