package point

// MatchState tracks the full state of a tennis match.
type MatchState struct {
	Server      int      // 0 or 1
	ServeSide   string   // "deuce" or "ad"
	PointScore  [2]int   // 0, 15, 30, 40 (41 = advantage)
	GameScore   [2]int   // games in current set
	SetScore    [2]int   // sets won
	Sets        [][2]int // completed set scores
	IsDeuce     bool
	IsTiebreak  bool
	PointNumber int

	// tiebreakPoints counts raw points won inside the current tiebreak game.
	// Used to determine server rotation in tiebreak (every 2 points after first).
	tiebreakPoints int
}

// NewMatchState creates a new match starting with firstServer serving.
func NewMatchState(firstServer int) *MatchState {
	return &MatchState{
		Server:    firstServer,
		ServeSide: "deuce",
	}
}

// pointValues maps sequence index (0-3) to display score.
var pointValues = [4]int{0, 15, 30, 40}

// AwardPoint awards the current point to winner (0 or 1) and advances the
// match state accordingly.
func (ms *MatchState) AwardPoint(winner int) {
	ms.PointNumber++

	// Alternate serve side every point (including tiebreaks, except after a
	// game ends — the side resets to "deuce" at the start of each new game,
	// which is handled in resetGame).
	if ms.ServeSide == "deuce" {
		ms.ServeSide = "ad"
	} else {
		ms.ServeSide = "deuce"
	}

	if ms.IsTiebreak {
		ms.awardTiebreakPoint(winner)
	} else {
		ms.awardRegularPoint(winner)
	}
}

// awardRegularPoint handles scoring inside a normal (non-tiebreak) game.
//
// Score progression:
//   0 → 15 → 30 → 40 → game (if opponent < 40)
//   40-40 → IsDeuce=true (both stay at 40)
//   From deuce: winner gets advantage (41), loser stays at 40
//   From advantage: advantage holder wins → game; other player wins → back to deuce
func (ms *MatchState) awardRegularPoint(winner int) {
	loser := 1 - winner

	// Handle advantage state: one player is at 41.
	if ms.PointScore[winner] == 41 {
		// Advantage holder wins the game.
		ms.awardGame(winner)
		return
	}
	if ms.PointScore[loser] == 41 {
		// Loser had advantage — back to deuce.
		ms.PointScore[0] = 40
		ms.PointScore[1] = 40
		ms.IsDeuce = true
		return
	}

	// Handle deuce: both at 40, next winner gets advantage.
	if ms.IsDeuce {
		ms.PointScore[winner] = 41
		ms.PointScore[loser] = 40
		ms.IsDeuce = false
		return
	}

	// Normal progression.
	switch ms.PointScore[winner] {
	case 0:
		ms.PointScore[winner] = 15
	case 15:
		ms.PointScore[winner] = 30
	case 30:
		ms.PointScore[winner] = 40
		// After advancing to 40, check if the opponent is also at 40 (deuce).
		if ms.PointScore[loser] == 40 {
			ms.IsDeuce = true
		}
	case 40:
		// Winner was already at 40 and opponent is not at 40 — win the game.
		// (If opponent were also at 40, IsDeuce would already be true and
		// we'd have been caught by the IsDeuce block above.)
		ms.awardGame(winner)
	}
}

// awardTiebreakPoint handles scoring inside a tiebreak game.
// Tiebreak uses raw counts: first to 7+ with a 2-point lead wins.
// Server rotation: first point served by initial server, then alternate every 2 points.
func (ms *MatchState) awardTiebreakPoint(winner int) {
	ms.PointScore[winner]++ // reuse PointScore as raw tiebreak count
	ms.tiebreakPoints++

	// Rotate server in tiebreak:
	//   - After the 1st point, rotate once.
	//   - After every subsequent 2 points, rotate.
	if ms.tiebreakPoints == 1 {
		ms.Server = 1 - ms.Server
	} else if (ms.tiebreakPoints-1)%2 == 0 {
		ms.Server = 1 - ms.Server
	}

	p0 := ms.PointScore[0]
	p1 := ms.PointScore[1]

	// Win condition: 7+ points and 2-point lead.
	if (p0 >= 7 || p1 >= 7) && abs(p0-p1) >= 2 {
		setWinner := 0
		if p1 > p0 {
			setWinner = 1
		}
		// Record the set with tiebreak scores.
		ms.Sets = append(ms.Sets, [2]int{p0 + (6 - p0 + 7), p1 + (6 - p1 + 7)})
		// Simpler: the set score is always 7-6 from perspective of games —
		// but we store actual tiebreak scores for the set record.
		// The game score going in was 6-6; winner gets 7, loser gets 6.
		ms.Sets[len(ms.Sets)-1] = [2]int{0, 0}
		if setWinner == 0 {
			ms.Sets[len(ms.Sets)-1] = [2]int{7, 6}
		} else {
			ms.Sets[len(ms.Sets)-1] = [2]int{6, 7}
		}
		ms.SetScore[setWinner]++
		ms.IsTiebreak = false
		ms.tiebreakPoints = 0
		ms.resetGame()
	}
}

// awardGame awards a game to winner and checks for set win.
func (ms *MatchState) awardGame(winner int) {
	ms.GameScore[winner]++
	ms.resetPoints()

	// Rotate server after every game.
	ms.Server = 1 - ms.Server
	// Serve side resets to deuce at the start of each new game.
	ms.ServeSide = "deuce"

	ms.checkSetWin()
}

// checkSetWin checks whether the current game score ends the set.
func (ms *MatchState) checkSetWin() {
	g0 := ms.GameScore[0]
	g1 := ms.GameScore[1]

	// Tiebreak at 6-6.
	if g0 == 6 && g1 == 6 {
		ms.IsTiebreak = true
		ms.tiebreakPoints = 0
		// Reset point scores for use as tiebreak raw counts.
		ms.PointScore = [2]int{0, 0}
		return
	}

	// Standard set win: 6+ games and 2-game lead.
	setWon := false
	winner := 0
	if g0 >= 6 && g0-g1 >= 2 {
		setWon = true
		winner = 0
	} else if g1 >= 6 && g1-g0 >= 2 {
		setWon = true
		winner = 1
	}

	if setWon {
		ms.Sets = append(ms.Sets, [2]int{g0, g1})
		ms.SetScore[winner]++
		ms.resetGame()
	}
}

// resetPoints resets the per-game point scores to 0 after a game ends.
func (ms *MatchState) resetPoints() {
	ms.PointScore = [2]int{0, 0}
	ms.IsDeuce = false
}

// resetGame resets game scores and point scores after a set ends.
func (ms *MatchState) resetGame() {
	ms.GameScore = [2]int{0, 0}
	ms.PointScore = [2]int{0, 0}
	ms.IsDeuce = false
	ms.IsTiebreak = false
	ms.ServeSide = "deuce"
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
