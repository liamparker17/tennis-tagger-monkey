# Phase 4: Match Context & Tactical Analysis

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tennis game state tracking, pattern detection, and tactical summary generation so coaches get actionable insights from processed videos.

**Architecture:** Pure Go implementation. A state machine tracks points/games/sets from detection results. A pattern analyzer identifies tactical tendencies. A report generator produces structured summaries. All new code lives in `internal/tactics/`.

**Tech Stack:** Go, no new dependencies

**Spec:** Phase 4 section of `docs/superpowers/specs/2026-03-20-tennis-tagger-rewrite-design.md`

---

## File Structure

```
New files:
  internal/tactics/
    score.go              — Tennis score state machine
    score_test.go
    patterns.go           — Pattern detection (stroke tendencies, pressure stats)
    patterns_test.go
    report.go             — Tactical summary generation (JSON/text)
    report_test.go
    history.go            — Cross-match historical comparison
    history_test.go

Modified files:
  internal/pipeline/pipeline.go   — Wire tactics into post-processing
  internal/app/app.go             — Add GetTacticalReport binding
  cmd/tagger/main.go              — Print tactical summary
```

---

## Task Groups

| Group | Name | Depends On | Tasks |
|-------|------|------------|-------|
| 1 | Score state machine | — | 1-2 |
| 2 | Pattern detection | — | 3-4 |
| 3 | Report generation | Groups 1, 2 | 5-6 |
| 4 | History + wiring | Group 3 | 7-8 |

Groups 1 and 2 can run in parallel.

---

## Group 1: Score State Machine

### Task 1: Tennis score tracker

**Files:**
- Create: `internal/tactics/score.go`
- Create: `internal/tactics/score_test.go`

Implement tennis scoring rules in Go:

```go
package tactics

// GameScore tracks points within a game (0, 15, 30, 40, Ad).
type GameScore struct {
    Server   int // Points: 0,1,2,3 = 0,15,30,40. 4+ for deuce/ad tracking
    Returner int
}

// SetScore tracks games within a set.
type SetScore struct {
    Server   int
    Returner int
    Tiebreak bool
}

// MatchScore tracks the full match state.
type MatchScore struct {
    Sets        []SetScore
    CurrentSet  SetScore
    CurrentGame GameScore
    Server      int  // 0 or 1 (player index)
    IsComplete  bool
}

// NewMatchScore creates a new match starting at 0-0.
func NewMatchScore() *MatchScore

// PointWon records a point for the given player (0=server, 1=returner).
// Returns a list of events that occurred (e.g., "game", "set", "break").
func (m *MatchScore) PointWon(player int) []string

// String returns human-readable score like "6-4 3-2 (30-15)"
func (m *MatchScore) String() string

// DisplayPoints returns traditional tennis point display (0, 15, 30, 40, Ad).
func DisplayPoints(points int) string

// IsBreakPoint returns true if the returner can win the game.
func (m *MatchScore) IsBreakPoint() bool

// IsSetPoint returns true if either player can win the set.
func (m *MatchScore) IsSetPoint() bool

// IsMatchPoint returns true if either player can win the match.
func (m *MatchScore) IsMatchPoint(setsToWin int) bool
```

**Scoring rules:**
- Game: First to 4 points, win by 2. At deuce (3-3+), advantage then game.
- Set: First to 6 games, win by 2. Tiebreak at 6-6 (first to 7 points, win by 2).
- Match: Best of 3 sets (configurable).
- Server alternates each game. In tiebreak, server changes every 2 points.

### Task 2: Score tests

```go
func TestPointWon_BasicGame(t *testing.T) {
    m := NewMatchScore()
    // Server wins 4 straight points = game
    for i := 0; i < 3; i++ {
        events := m.PointWon(0)
        if contains(events, "game") { t.Error("game too early") }
    }
    events := m.PointWon(0) // 4th point
    if !contains(events, "game") { t.Error("expected game event") }
    if m.CurrentSet.Server != 1 { t.Errorf("expected 1-0, got %d-%d", m.CurrentSet.Server, m.CurrentSet.Returner) }
}

func TestPointWon_Deuce(t *testing.T) {
    m := NewMatchScore()
    // Get to deuce (3-3)
    for i := 0; i < 3; i++ { m.PointWon(0) }
    for i := 0; i < 3; i++ { m.PointWon(1) }
    // Advantage server
    m.PointWon(0)
    if m.CurrentGame.Server != 4 { t.Error("expected advantage") }
    // Back to deuce
    m.PointWon(1)
    if m.CurrentGame.Server != 3 || m.CurrentGame.Returner != 3 { t.Error("expected deuce") }
}

func TestPointWon_SetWin(t *testing.T) {
    m := NewMatchScore()
    // Server wins 6 games (24 points)
    for game := 0; game < 6; game++ {
        for pt := 0; pt < 4; pt++ { m.PointWon(0) }
    }
    if len(m.Sets) != 1 { t.Error("expected 1 completed set") }
}

func TestPointWon_Tiebreak(t *testing.T) {
    m := NewMatchScore()
    // Get to 6-6
    for game := 0; game < 6; game++ {
        for pt := 0; pt < 4; pt++ { m.PointWon(0) }  // server holds
        for pt := 0; pt < 4; pt++ { m.PointWon(1) }  // returner holds (breaks server)
    }
    // Now in tiebreak
    if !m.CurrentSet.Tiebreak { t.Error("expected tiebreak") }
    // Win tiebreak 7-0
    for pt := 0; pt < 7; pt++ { m.PointWon(0) }
    if len(m.Sets) != 1 { t.Error("expected 1 completed set from tiebreak") }
}

func TestIsBreakPoint(t *testing.T) {
    m := NewMatchScore()
    for i := 0; i < 3; i++ { m.PointWon(1) } // 0-40
    if !m.IsBreakPoint() { t.Error("expected break point at 0-40") }
}

func TestDisplayPoints(t *testing.T) {
    tests := map[int]string{0:"0", 1:"15", 2:"30", 3:"40"}
    for pts, want := range tests {
        if got := DisplayPoints(pts); got != want {
            t.Errorf("DisplayPoints(%d) = %q, want %q", pts, got, want)
        }
    }
}
```

Commit after tests pass.

---

## Group 2: Pattern Detection

### Task 3: Pattern analyzer

**Files:**
- Create: `internal/tactics/patterns.go`
- Create: `internal/tactics/patterns_test.go`

```go
package tactics

import "github.com/liamp/tennis-tagger/internal/bridge"

// StrokeStats tracks aggregate stroke statistics for a player.
type StrokeStats struct {
    Total       int
    Winners     int
    Errors      int
    ByType      map[string]int  // "forehand": 25, "backhand": 18, ...
    ByZone      map[string]int  // "Deuce Wide": 10, "Ad T": 5, ...
    ByDepth     map[string]int  // "Baseline": 30, "Net": 5, ...
}

// PressureStats tracks performance under pressure.
type PressureStats struct {
    BreakPointsPlayed   int
    BreakPointsWon      int
    SetPointsPlayed     int
    SetPointsWon        int
    PointsWonOnServe    int
    TotalServePoints    int
    PointsWonOnReturn   int
    TotalReturnPoints   int
}

// PlayerPattern captures tactical tendencies for one player.
type PlayerPattern struct {
    PlayerIndex int
    Strokes     StrokeStats
    Pressure    PressureStats
    Tendencies  []string  // Natural language insights
}

// AnalyzePatterns examines detection results and score context
// to identify player tendencies.
func AnalyzePatterns(
    detections []bridge.DetectionResult,
    placements []bridge.PlacementResult,
    strokes []bridge.StrokeResult,
    rallies []bridge.RallyResult,
    score *MatchScore,
) []PlayerPattern
```

The pattern analyzer should:
1. Count strokes by type per player
2. Count placements by zone per player
3. Track break point conversion rates
4. Generate tendency strings like:
   - "Targets backhand side under pressure (67%)"
   - "Approaches net on 45% of short balls"
   - "First serve percentage: 62%"
   - "Forehand is dominant stroke (58% of groundstrokes)"

### Task 4: Pattern tests

```go
func TestAnalyzePatterns_EmptyInput(t *testing.T) {
    patterns := AnalyzePatterns(nil, nil, nil, nil, NewMatchScore())
    if len(patterns) != 2 { t.Error("expected 2 player patterns (empty)") }
}

func TestStrokeStats_CountByType(t *testing.T) {
    strokes := []bridge.StrokeResult{
        {Type: "forehand", PlayerID: 0},
        {Type: "forehand", PlayerID: 0},
        {Type: "backhand", PlayerID: 0},
        {Type: "forehand", PlayerID: 1},
    }
    patterns := AnalyzePatterns(nil, nil, strokes, nil, NewMatchScore())
    p0 := patterns[0]
    if p0.Strokes.ByType["forehand"] != 2 { t.Errorf("expected 2 forehands for P0") }
    if p0.Strokes.ByType["backhand"] != 1 { t.Errorf("expected 1 backhand for P0") }
    if p0.Strokes.Total != 3 { t.Errorf("expected 3 total for P0") }
}

func TestTendencies_Generated(t *testing.T) {
    // Create enough data for tendencies to be generated
    strokes := make([]bridge.StrokeResult, 20)
    for i := range strokes {
        strokes[i] = bridge.StrokeResult{Type: "forehand", PlayerID: 0}
    }
    patterns := AnalyzePatterns(nil, nil, strokes, nil, NewMatchScore())
    if len(patterns[0].Tendencies) == 0 { t.Error("expected at least one tendency") }
}
```

---

## Group 3: Report Generation

### Task 5: Tactical report

**Files:**
- Create: `internal/tactics/report.go`
- Create: `internal/tactics/report_test.go`

```go
package tactics

// TacticalReport is the full tactical analysis output.
type TacticalReport struct {
    MatchScore    string           `json:"match_score"`
    Duration      string           `json:"duration"`
    TotalRallies  int              `json:"total_rallies"`
    TotalPoints   int              `json:"total_points"`
    Players       []PlayerReport   `json:"players"`
    KeyMoments    []string         `json:"key_moments"`
}

// PlayerReport is per-player tactical data.
type PlayerReport struct {
    Label         string           `json:"label"`  // "Player 1" or "Player 2"
    ServeStats    ServeReport      `json:"serve_stats"`
    StrokeBreakdown map[string]int `json:"stroke_breakdown"`
    PlacementHeatmap map[string]int `json:"placement_heatmap"`
    PressureStats PressureStats    `json:"pressure_stats"`
    Tendencies    []string         `json:"tendencies"`
}

type ServeReport struct {
    TotalServes    int     `json:"total_serves"`
    FirstServeIn   int     `json:"first_serve_in"`
    FirstServePct  float64 `json:"first_serve_pct"`
    Aces           int     `json:"aces"`
    DoubleFaults   int     `json:"double_faults"`
}

// GenerateReport builds a tactical summary from all analysis data.
func GenerateReport(
    patterns []PlayerPattern,
    score *MatchScore,
    rallies []bridge.RallyResult,
    durationSec float64,
) *TacticalReport

// FormatText renders the report as human-readable text.
func (r *TacticalReport) FormatText() string
```

### Task 6: Report tests

```go
func TestGenerateReport_Basic(t *testing.T) {
    score := NewMatchScore()
    // Play a quick game
    for i := 0; i < 4; i++ { score.PointWon(0) }

    patterns := []PlayerPattern{
        {PlayerIndex: 0, Strokes: StrokeStats{Total: 10, ByType: map[string]int{"forehand": 7, "backhand": 3}}},
        {PlayerIndex: 1, Strokes: StrokeStats{Total: 8, ByType: map[string]int{"forehand": 5, "backhand": 3}}},
    }

    report := GenerateReport(patterns, score, nil, 120.0)
    if report.TotalPoints != 4 { t.Errorf("expected 4 points, got %d", report.TotalPoints) }
    if len(report.Players) != 2 { t.Error("expected 2 player reports") }
}

func TestFormatText_NotEmpty(t *testing.T) {
    report := &TacticalReport{
        MatchScore: "6-4 3-2",
        Players: []PlayerReport{{Label: "Player 1"}, {Label: "Player 2"}},
    }
    text := report.FormatText()
    if len(text) == 0 { t.Error("expected non-empty text") }
    if !strings.Contains(text, "Player 1") { t.Error("expected player name in text") }
}
```

---

## Group 4: History + Wiring

### Task 7: Historical comparison

**Files:**
- Create: `internal/tactics/history.go`
- Create: `internal/tactics/history_test.go`

```go
package tactics

import (
    "encoding/json"
    "os"
    "path/filepath"
    "time"
)

// MatchRecord stores a completed match's tactical data for history.
type MatchRecord struct {
    ID        string          `json:"id"`
    Date      time.Time       `json:"date"`
    VideoPath string          `json:"video_path"`
    Score     string          `json:"score"`
    Report    *TacticalReport `json:"report"`
}

// MatchHistory manages historical match records.
type MatchHistory struct {
    dir     string
    records []MatchRecord
}

// NewMatchHistory loads history from the given directory.
func NewMatchHistory(dir string) *MatchHistory

// Save persists a match record.
func (h *MatchHistory) Save(record MatchRecord) error

// List returns all recorded matches, newest first.
func (h *MatchHistory) List() []MatchRecord

// Compare returns a comparison between two matches.
func (h *MatchHistory) Compare(id1, id2 string) *MatchComparison

type MatchComparison struct {
    Match1    MatchRecord `json:"match1"`
    Match2    MatchRecord `json:"match2"`
    StrokeDiff map[string][2]int `json:"stroke_diff"` // type -> [match1_count, match2_count]
    Insights  []string          `json:"insights"`     // "Forehand winners up 40% vs previous match"
}
```

Tests:
- Save and List round-trip
- Compare two matches generates insights

### Task 8: Wire into pipeline and CLI

**Files:**
- Modify: `internal/pipeline/pipeline.go`
- Modify: `internal/app/app.go`
- Modify: `cmd/tagger/main.go`

Pipeline post-processing addition:
```go
// After existing post-processing (rallies, placements):
patterns := tactics.AnalyzePatterns(result.Detections, result.Placements, result.Strokes, result.Rallies, nil)
report := tactics.GenerateReport(patterns, nil, result.Rallies, result.Duration)
result.TacticalReport = report
```

Add to Result struct:
```go
TacticalReport *tactics.TacticalReport
```

App binding:
```go
func (a *App) GetTacticalReport() *tactics.TacticalReport {
    if a.result == nil { return nil }
    return a.result.TacticalReport
}
```

CLI: print `report.FormatText()` after processing.

Run all tests: `go test ./internal/... -count=1`

---

## Summary

| Group | Tasks | Deliverable |
|-------|-------|-------------|
| 1. Score machine | 1-2 | Tennis scoring (game/set/match), break/set/match point detection |
| 2. Patterns | 3-4 | Stroke/placement stats, pressure analysis, tendency generation |
| 3. Report | 5-6 | Structured tactical summary with text formatting |
| 4. History | 7-8 | Cross-match comparison, pipeline + CLI integration |
