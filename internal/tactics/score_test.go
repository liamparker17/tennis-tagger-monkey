package tactics

import (
	"testing"
)

func contains(events []string, target string) bool {
	for _, e := range events {
		if e == target {
			return true
		}
	}
	return false
}

func TestDisplayPoints(t *testing.T) {
	tests := []struct {
		points int
		want   string
	}{
		{0, "0"},
		{1, "15"},
		{2, "30"},
		{3, "40"},
	}
	for _, tt := range tests {
		got := DisplayPoints(tt.points)
		if got != tt.want {
			t.Errorf("DisplayPoints(%d) = %q, want %q", tt.points, got, tt.want)
		}
	}
}

func TestPointWon_BasicGame(t *testing.T) {
	m := NewMatchScore()

	// Server (player 0) wins 4 straight points
	for i := 0; i < 3; i++ {
		events := m.PointWon(0)
		if contains(events, "game") {
			t.Fatalf("Game should not be won after %d points", i+1)
		}
	}

	events := m.PointWon(0)
	if !contains(events, "game") {
		t.Fatal("Expected 'game' event after 4th point")
	}

	// Set score should be 1-0
	if m.CurrentSet.Server != 1 || m.CurrentSet.Returner != 0 {
		t.Errorf("Expected set score 1-0, got %d-%d", m.CurrentSet.Server, m.CurrentSet.Returner)
	}
}

func TestPointWon_Deuce(t *testing.T) {
	m := NewMatchScore()

	// Get to 3-3 (deuce): alternate winning 3 points each
	for i := 0; i < 3; i++ {
		m.PointWon(0) // server point
	}
	for i := 0; i < 3; i++ {
		m.PointWon(1) // returner point
	}

	// At deuce (40-40), game score should be 3-3
	if m.CurrentGame.Server != 3 || m.CurrentGame.Returner != 3 {
		t.Errorf("Expected deuce (3-3), got %d-%d", m.CurrentGame.Server, m.CurrentGame.Returner)
	}

	// Server wins a point -> advantage server (4-3)
	events := m.PointWon(0)
	if contains(events, "game") {
		t.Fatal("Game should not be won on advantage point")
	}
	if m.CurrentGame.Server != 4 || m.CurrentGame.Returner != 3 {
		t.Errorf("Expected advantage (4-3), got %d-%d", m.CurrentGame.Server, m.CurrentGame.Returner)
	}

	// Returner wins a point -> back to deuce (3-3)
	events = m.PointWon(1)
	if contains(events, "game") {
		t.Fatal("Game should not be won when going back to deuce")
	}
	if m.CurrentGame.Server != 3 || m.CurrentGame.Returner != 3 {
		t.Errorf("Expected deuce (3-3) after advantage lost, got %d-%d",
			m.CurrentGame.Server, m.CurrentGame.Returner)
	}

	// Server wins two straight -> game
	m.PointWon(0) // advantage
	events = m.PointWon(0)
	if !contains(events, "game") {
		t.Fatal("Expected 'game' event after winning from advantage")
	}
}

func TestPointWon_SetWin(t *testing.T) {
	m := NewMatchScore()

	// Player 0 wins 6 games straight (each game is 4 points for player 0)
	var setEvent bool
	for game := 0; game < 6; game++ {
		for pt := 0; pt < 4; pt++ {
			events := m.PointWon(0)
			if contains(events, "set") {
				setEvent = true
			}
		}
	}

	if !setEvent {
		t.Fatal("Expected 'set' event after winning 6 games")
	}

	// Should have a completed set
	if len(m.Sets) != 1 {
		t.Fatalf("Expected 1 completed set, got %d", len(m.Sets))
	}
	if m.Sets[0].Server != 6 || m.Sets[0].Returner != 0 {
		t.Errorf("Expected set score 6-0, got %d-%d", m.Sets[0].Server, m.Sets[0].Returner)
	}
}

func TestPointWon_Tiebreak(t *testing.T) {
	m := NewMatchScore()

	// Get to 6-6: each player holds serve for 6 games each
	for game := 0; game < 12; game++ {
		server := m.Server
		for pt := 0; pt < 4; pt++ {
			m.PointWon(server)
		}
	}

	// Should be in tiebreak at 6-6
	if !m.CurrentSet.Tiebreak {
		t.Fatalf("Expected tiebreak at 6-6, set score is %d-%d",
			m.CurrentSet.Server, m.CurrentSet.Returner)
	}

	// Record who is serving first in tiebreak
	tbServer := m.Server

	// Win tiebreak 7-0: server of each point wins
	// Point 1: tbServer serves
	m.PointWon(tbServer)
	// After point 1, server switches
	// Points 2-3: other player serves
	m.PointWon(1 - tbServer)
	m.PointWon(1 - tbServer)
	// Points 4-5: tbServer serves again
	m.PointWon(tbServer)
	m.PointWon(tbServer)
	// Points 6-7: other player serves
	m.PointWon(1 - tbServer)
	events := m.PointWon(1 - tbServer)

	// The tiebreak should be won
	if !contains(events, "set") {
		t.Fatal("Expected 'set' event after tiebreak win")
	}

	if len(m.Sets) < 1 {
		t.Fatal("Expected at least 1 completed set")
	}
	lastSet := m.Sets[len(m.Sets)-1]
	// One side should have 7
	if lastSet.Server != 7 && lastSet.Returner != 7 {
		t.Errorf("Expected tiebreak set with 7 games on one side, got %d-%d",
			lastSet.Server, lastSet.Returner)
	}
}

func TestIsBreakPoint(t *testing.T) {
	m := NewMatchScore()

	// Score: 0-40 (server 0, returner has 3 points)
	m.PointWon(1)
	m.PointWon(1)
	m.PointWon(1)

	if !m.IsBreakPoint() {
		t.Error("Expected break point at 0-40")
	}

	// Reset: new match at 30-30 (not break point)
	m2 := NewMatchScore()
	m2.PointWon(0)
	m2.PointWon(0)
	m2.PointWon(1)
	m2.PointWon(1)

	if m2.IsBreakPoint() {
		t.Error("Should not be break point at 30-30")
	}

	// 30-40 is break point
	m2.PointWon(1)
	if !m2.IsBreakPoint() {
		t.Error("Expected break point at 30-40")
	}
}

func TestString(t *testing.T) {
	m := NewMatchScore()

	// Initial score
	if got := m.String(); got != "0-0" {
		t.Errorf("Expected '0-0', got %q", got)
	}

	// After one point
	m.PointWon(0)
	if got := m.String(); got != "0-0 (15-0)" {
		t.Errorf("Expected '0-0 (15-0)', got %q", got)
	}

	// After server wins a game (3 more points)
	m.PointWon(0)
	m.PointWon(0)
	m.PointWon(0)
	if got := m.String(); got != "1-0" {
		t.Errorf("Expected '1-0', got %q", got)
	}
}

func TestPointWon_MatchWin(t *testing.T) {
	m := NewMatchScore()

	var matchEvent bool

	// Player 0 wins 2 sets: each set 6-0, each game won by player 0
	for set := 0; set < 2; set++ {
		for game := 0; game < 6; game++ {
			for pt := 0; pt < 4; pt++ {
				events := m.PointWon(0)
				if contains(events, "match") {
					matchEvent = true
				}
			}
		}
	}

	if !matchEvent {
		t.Fatal("Expected 'match' event after winning 2 sets")
	}

	if !m.IsComplete {
		t.Fatal("Expected IsComplete to be true after match win")
	}

	if len(m.Sets) != 2 {
		t.Errorf("Expected 2 completed sets, got %d", len(m.Sets))
	}
}

func TestPointWon_BreakEvent(t *testing.T) {
	m := NewMatchScore()

	// Player 0 serves, player 1 wins 4 points -> break
	for i := 0; i < 3; i++ {
		m.PointWon(1)
	}
	events := m.PointWon(1)

	if !contains(events, "game") {
		t.Fatal("Expected 'game' event")
	}
	if !contains(events, "break") {
		t.Fatal("Expected 'break' event when returner wins game")
	}
}

func TestPointWon_NoEventsAfterMatchComplete(t *testing.T) {
	m := NewMatchScore()

	// Player 0 wins match quickly
	for set := 0; set < 2; set++ {
		for game := 0; game < 6; game++ {
			for pt := 0; pt < 4; pt++ {
				m.PointWon(0)
			}
		}
	}

	// Try to score another point
	events := m.PointWon(0)
	if len(events) != 0 {
		t.Errorf("Expected no events after match complete, got %v", events)
	}
}

func TestServerAlternates(t *testing.T) {
	m := NewMatchScore()

	if m.Server != 0 {
		t.Fatal("Expected player 0 to serve first")
	}

	// Player 0 wins first game
	for i := 0; i < 4; i++ {
		m.PointWon(0)
	}

	if m.Server != 1 {
		t.Error("Expected server to switch to player 1 after first game")
	}

	// Player 1 wins second game
	for i := 0; i < 4; i++ {
		m.PointWon(1)
	}

	if m.Server != 0 {
		t.Error("Expected server to switch back to player 0 after second game")
	}
}

func TestIsSetPoint(t *testing.T) {
	m := NewMatchScore()

	// Player 0 wins 5 games straight -> set score 5-0
	for game := 0; game < 5; game++ {
		for pt := 0; pt < 4; pt++ {
			m.PointWon(0)
		}
	}

	if m.CurrentSet.Server != 5 || m.CurrentSet.Returner != 0 {
		t.Fatalf("Expected set score 5-0, got %d-%d", m.CurrentSet.Server, m.CurrentSet.Returner)
	}

	// Player 0 wins 3 more points to get to 40-0 in the game
	m.PointWon(0)
	m.PointWon(0)
	m.PointWon(0)

	if !m.IsSetPoint() {
		t.Error("Expected set point at 5-0, 40-0")
	}
}

func TestIsMatchPoint(t *testing.T) {
	m := NewMatchScore()

	// Player 0 wins first set 6-0
	for game := 0; game < 6; game++ {
		for pt := 0; pt < 4; pt++ {
			m.PointWon(0)
		}
	}

	if len(m.Sets) != 1 {
		t.Fatalf("Expected 1 completed set, got %d", len(m.Sets))
	}

	// In second set, player 0 wins 5 games -> 5-0
	for game := 0; game < 5; game++ {
		for pt := 0; pt < 4; pt++ {
			m.PointWon(0)
		}
	}

	// Player 0 gets to 40-0
	m.PointWon(0)
	m.PointWon(0)
	m.PointWon(0)

	if !m.IsMatchPoint(2) {
		t.Error("Expected match point")
	}
}

func TestTiebreakStartEvent(t *testing.T) {
	m := NewMatchScore()

	var tiebreakStarted bool

	// Get to 6-6
	for game := 0; game < 12; game++ {
		server := m.Server
		events := []string{}
		for pt := 0; pt < 4; pt++ {
			events = m.PointWon(server)
		}
		if contains(events, "tiebreak_start") {
			tiebreakStarted = true
		}
	}

	if !tiebreakStarted {
		t.Fatal("Expected 'tiebreak_start' event at 6-6")
	}

	if !m.CurrentSet.Tiebreak {
		t.Fatal("Expected tiebreak mode to be active")
	}
}
