package tactics

import "fmt"

// DisplayPoints converts a numeric point count to tennis display string.
func DisplayPoints(points int) string {
	switch points {
	case 0:
		return "0"
	case 1:
		return "15"
	case 2:
		return "30"
	case 3:
		return "40"
	default:
		return "40"
	}
}

// GameScore tracks points within a single game.
// Server/Returner are relative to who is serving that game.
type GameScore struct {
	Server   int
	Returner int
}

// SetScore tracks games within a single set.
// Server/Returner are relative to who served first in the match (player 0 / player 1).
// We use a fixed perspective: Server = player 0's games, Returner = player 1's games.
type SetScore struct {
	Server   int
	Returner int
	Tiebreak bool
}

// MatchScore is the full tennis score state machine.
type MatchScore struct {
	Sets        []SetScore
	CurrentSet  SetScore
	CurrentGame GameScore
	Server      int  // 0 = player0 serving, 1 = player1 serving
	IsComplete  bool
	SetsToWin   int  // Default 2 (best of 3)

	tiebreakPoints    int // total points played in current tiebreak
	tiebreakInitServe int // who served first point of tiebreak
}

// NewMatchScore creates a new match score with default best-of-3.
func NewMatchScore() *MatchScore {
	return &MatchScore{
		SetsToWin: 2,
	}
}

// PointWon records a point won by the given player (0 or 1) and returns
// a list of events triggered: "game", "break", "set", "tiebreak_start", "match".
func (m *MatchScore) PointWon(player int) []string {
	if m.IsComplete {
		return nil
	}

	if m.CurrentSet.Tiebreak {
		return m.tiebreakPointWon(player)
	}
	return m.regularPointWon(player)
}

func (m *MatchScore) regularPointWon(player int) []string {
	var events []string

	isServer := player == m.Server

	if isServer {
		m.CurrentGame.Server++
	} else {
		m.CurrentGame.Returner++
	}

	s := m.CurrentGame.Server
	r := m.CurrentGame.Returner

	// Check for game won
	gameWon := false
	var gameWinner int

	if s >= 4 && s-r >= 2 {
		gameWon = true
		gameWinner = m.Server
	} else if r >= 4 && r-s >= 2 {
		gameWon = true
		gameWinner = 1 - m.Server
	}

	// Handle deuce reset: if both >= 4 and equal, reset to 3-3
	if !gameWon && s >= 3 && r >= 3 && s == r {
		m.CurrentGame.Server = 3
		m.CurrentGame.Returner = 3
	}

	if gameWon {
		events = append(events, "game")

		// Break: returner won the game
		if gameWinner != m.Server {
			events = append(events, "break")
		}

		// Add game to set score (player 0 = Server side, player 1 = Returner side)
		if gameWinner == 0 {
			m.CurrentSet.Server++
		} else {
			m.CurrentSet.Returner++
		}

		m.CurrentGame = GameScore{}

		setEvents := m.checkSetWon()
		events = append(events, setEvents...)

		// Alternate server (unless match is complete)
		if !m.IsComplete {
			m.Server = 1 - m.Server
		}
	}

	return events
}

func (m *MatchScore) checkSetWon() []string {
	var events []string
	s := m.CurrentSet.Server
	r := m.CurrentSet.Returner

	setWon := false
	if s >= 6 && s-r >= 2 {
		setWon = true
	} else if r >= 6 && r-s >= 2 {
		setWon = true
	}

	if setWon {
		events = append(events, "set")
		m.Sets = append(m.Sets, m.CurrentSet)
		if m.checkMatchWon() {
			events = append(events, "match")
		} else {
			m.CurrentSet = SetScore{}
		}
	} else if s == 6 && r == 6 {
		m.CurrentSet.Tiebreak = true
		m.tiebreakPoints = 0
		m.tiebreakInitServe = m.Server
		events = append(events, "tiebreak_start")
	}

	return events
}

func (m *MatchScore) tiebreakPointWon(player int) []string {
	var events []string

	// Track the point
	if player == m.Server {
		m.CurrentGame.Server++
	} else {
		m.CurrentGame.Returner++
	}
	m.tiebreakPoints++

	s := m.CurrentGame.Server
	r := m.CurrentGame.Returner

	// Check tiebreak won
	tiebreakWon := false
	var winner int

	if s >= 7 && s-r >= 2 {
		tiebreakWon = true
		winner = m.Server
	} else if r >= 7 && r-s >= 2 {
		tiebreakWon = true
		winner = 1 - m.Server
	}

	if tiebreakWon {
		events = append(events, "game")

		// Add tiebreak game to set score
		if winner == 0 {
			m.CurrentSet.Server++
		} else {
			m.CurrentSet.Returner++
		}

		m.CurrentGame = GameScore{}

		events = append(events, "set")
		m.Sets = append(m.Sets, m.CurrentSet)

		if m.checkMatchWon() {
			events = append(events, "match")
		} else {
			m.CurrentSet = SetScore{}
		}

		// After tiebreak, the player who received first in the tiebreak serves first in next set
		m.Server = 1 - m.tiebreakInitServe
	} else {
		// Server rotation in tiebreak:
		// First point: initial server. After 1 point, switch. Then switch every 2 points.
		// Switch after point 1, then after points 3, 5, 7, ...
		tp := m.tiebreakPoints
		if tp == 1 || (tp > 1 && tp%2 == 1) {
			m.Server = 1 - m.Server
		}
	}

	return events
}

func (m *MatchScore) checkMatchWon() bool {
	var p0Sets, p1Sets int
	for _, s := range m.Sets {
		if s.Server > s.Returner {
			p0Sets++
		} else {
			p1Sets++
		}
	}

	if p0Sets >= m.SetsToWin || p1Sets >= m.SetsToWin {
		m.IsComplete = true
		return true
	}
	return false
}

// String returns a human-readable score string like "6-4 3-2 (30-15)".
func (m *MatchScore) String() string {
	result := ""
	for i, s := range m.Sets {
		if i > 0 {
			result += " "
		}
		result += fmt.Sprintf("%d-%d", s.Server, s.Returner)
	}

	if m.IsComplete {
		return result
	}

	if len(m.Sets) > 0 {
		result += " "
	}
	result += fmt.Sprintf("%d-%d", m.CurrentSet.Server, m.CurrentSet.Returner)

	if m.CurrentGame.Server > 0 || m.CurrentGame.Returner > 0 {
		if m.CurrentSet.Tiebreak {
			result += fmt.Sprintf(" (%d-%d)", m.CurrentGame.Server, m.CurrentGame.Returner)
		} else {
			result += fmt.Sprintf(" (%s-%s)",
				DisplayPoints(m.CurrentGame.Server),
				DisplayPoints(m.CurrentGame.Returner))
		}
	}

	return result
}

// IsBreakPoint returns true if the returner can win the game on the next point.
func (m *MatchScore) IsBreakPoint() bool {
	if m.CurrentSet.Tiebreak || m.IsComplete {
		return false
	}
	s := m.CurrentGame.Server
	r := m.CurrentGame.Returner

	// Returner can win if they need one more point: r >= 3 and r > s
	// At deuce (3-3), no one has advantage yet, so not break point.
	// At 3-4 (advantage returner), it is break point.
	return r >= 3 && r > s
}

// IsSetPoint returns true if either player can win the set on the next point.
func (m *MatchScore) IsSetPoint() bool {
	if m.IsComplete {
		return false
	}

	if m.CurrentSet.Tiebreak {
		s := m.CurrentGame.Server
		r := m.CurrentGame.Returner
		return (s >= 6 && s > r) || (r >= 6 && r > s)
	}

	ss := m.CurrentSet.Server
	sr := m.CurrentSet.Returner
	gs := m.CurrentGame.Server
	gr := m.CurrentGame.Returner

	// wouldWinSet checks if adding a game for the given player would win the set.
	wouldWinSet := func(player int) bool {
		p0, p1 := ss, sr
		if player == 0 {
			p0++
		} else {
			p1++
		}
		return (p0 >= 6 && p0-p1 >= 2) || (p1 >= 6 && p1-p0 >= 2)
	}

	// Game server has game point
	if gs >= 3 && gs > gr && wouldWinSet(m.Server) {
		return true
	}

	// Game returner has game point (break point)
	if gr >= 3 && gr > gs && wouldWinSet(1-m.Server) {
		return true
	}

	return false
}

// IsMatchPoint returns true if either player can win the match on the next point.
func (m *MatchScore) IsMatchPoint(setsToWin int) bool {
	if m.IsComplete {
		return false
	}

	if !m.IsSetPoint() {
		return false
	}

	var p0Sets, p1Sets int
	for _, s := range m.Sets {
		if s.Server > s.Returner {
			p0Sets++
		} else {
			p1Sets++
		}
	}

	return p0Sets == setsToWin-1 || p1Sets == setsToWin-1
}
