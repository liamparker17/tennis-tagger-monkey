package point

import (
	"testing"
)

// helper: award n points to the given winner
func awardN(ms *MatchState, winner, n int) {
	for i := 0; i < n; i++ {
		ms.AwardPoint(winner)
	}
}

// TestScoreBasicGame: player 0 wins 4 straight → game, server rotates.
func TestScoreBasicGame(t *testing.T) {
	ms := NewMatchState(0)

	// Before any points
	if ms.Server != 0 {
		t.Fatalf("expected server=0, got %d", ms.Server)
	}

	// Win a full game 4-0 for player 0
	awardN(ms, 0, 4)

	// Game score: player 0 should have 1 game, player 1 should have 0
	if ms.GameScore[0] != 1 || ms.GameScore[1] != 0 {
		t.Errorf("expected GameScore [1,0], got %v", ms.GameScore)
	}

	// Point scores reset to 0 after game
	if ms.PointScore[0] != 0 || ms.PointScore[1] != 0 {
		t.Errorf("expected PointScore reset to [0,0], got %v", ms.PointScore)
	}

	// Server should have rotated to player 1
	if ms.Server != 1 {
		t.Errorf("expected server=1 after game, got %d", ms.Server)
	}
}

// TestScoreDeuce: 40-40, advantage, back to deuce, advantage + win.
func TestScoreDeuce(t *testing.T) {
	ms := NewMatchState(0)

	// Get both players to 40 (3 points each)
	for i := 0; i < 3; i++ {
		ms.AwardPoint(0)
		ms.AwardPoint(1)
	}

	if !ms.IsDeuce {
		t.Fatal("expected IsDeuce=true at 40-40")
	}
	// Both should show 40
	if ms.PointScore[0] != 40 || ms.PointScore[1] != 40 {
		t.Errorf("expected PointScore [40,40] at deuce, got %v", ms.PointScore)
	}

	// Player 0 gets advantage
	ms.AwardPoint(0)
	if ms.IsDeuce {
		t.Error("expected IsDeuce=false after advantage awarded")
	}
	// Advantage is represented as 41
	if ms.PointScore[0] != 41 {
		t.Errorf("expected PointScore[0]=41 (advantage), got %d", ms.PointScore[0])
	}

	// Player 1 wins point → back to deuce
	ms.AwardPoint(1)
	if !ms.IsDeuce {
		t.Error("expected IsDeuce=true after advantage lost")
	}
	if ms.PointScore[0] != 40 || ms.PointScore[1] != 40 {
		t.Errorf("expected PointScore back to [40,40], got %v", ms.PointScore)
	}

	// Player 0 advantage again
	ms.AwardPoint(0)
	// Player 0 wins game
	ms.AwardPoint(0)

	if ms.GameScore[0] != 1 {
		t.Errorf("expected GameScore[0]=1 after winning from advantage, got %d", ms.GameScore[0])
	}
	if ms.IsDeuce {
		t.Error("expected IsDeuce=false after game won")
	}
}

// TestScoreTiebreak: get to 6-6, tiebreak activates, first to 7 with 2+ lead wins set.
func TestScoreTiebreak(t *testing.T) {
	ms := NewMatchState(0)

	// Play 12 games to reach 6-6: alternate winners
	for i := 0; i < 6; i++ {
		// Player 0 wins a game
		awardN(ms, 0, 4)
		// Player 1 wins a game
		awardN(ms, 1, 4)
	}

	if ms.GameScore[0] != 6 || ms.GameScore[1] != 6 {
		t.Fatalf("expected GameScore [6,6] before tiebreak, got %v", ms.GameScore)
	}
	if !ms.IsTiebreak {
		t.Fatal("expected IsTiebreak=true at 6-6")
	}

	// Tiebreak: player 0 wins 7-0 (dominant win)
	awardN(ms, 0, 7)

	// Set should be won by player 0
	if ms.SetScore[0] != 1 || ms.SetScore[1] != 0 {
		t.Errorf("expected SetScore [1,0] after tiebreak win, got %v", ms.SetScore)
	}
	if ms.IsTiebreak {
		t.Error("expected IsTiebreak=false after tiebreak ends")
	}
	// The completed set should be recorded in Sets
	if len(ms.Sets) != 1 {
		t.Fatalf("expected 1 completed set, got %d", len(ms.Sets))
	}
	if ms.Sets[0][0] != 7 || ms.Sets[0][1] != 6 {
		t.Errorf("expected completed set score [7,6], got %v", ms.Sets[0])
	}
}

// TestScoreTiebreakCloseFinish: tiebreak at 6-6 within the tiebreak requires 2-point lead.
func TestScoreTiebreakCloseFinish(t *testing.T) {
	ms := NewMatchState(0)

	// Reach 6-6 in games
	for i := 0; i < 6; i++ {
		awardN(ms, 0, 4)
		awardN(ms, 1, 4)
	}

	// In tiebreak: get to 6-6 within the tiebreak
	for i := 0; i < 6; i++ {
		ms.AwardPoint(0)
		ms.AwardPoint(1)
	}

	// At 6-6, no winner yet — need 2-point lead
	if ms.SetScore[0] != 0 || ms.SetScore[1] != 0 {
		t.Errorf("expected no set winner yet at TB 6-6, got %v", ms.SetScore)
	}
	if !ms.IsTiebreak {
		t.Error("expected IsTiebreak=true at TB 6-6")
	}

	// Player 0 goes to 7, player 1 ties at 7
	ms.AwardPoint(0) // 7-6
	if ms.SetScore[0] != 0 {
		t.Error("7-6 in tiebreak should not yet win the set (need 2-point lead)")
	}
	ms.AwardPoint(1) // 7-7

	// Player 0 wins two in a row → wins tiebreak 9-7
	ms.AwardPoint(0) // 8-7
	ms.AwardPoint(0) // 9-7

	if ms.SetScore[0] != 1 {
		t.Errorf("expected SetScore[0]=1 after 9-7 tiebreak win, got %d", ms.SetScore[0])
	}
}

// TestServeSideAlternates: serve side should alternate each point.
func TestServeSideAlternates(t *testing.T) {
	ms := NewMatchState(0)

	if ms.ServeSide != "deuce" {
		t.Fatalf("expected initial ServeSide=deuce, got %s", ms.ServeSide)
	}

	ms.AwardPoint(0)
	if ms.ServeSide != "ad" {
		t.Errorf("expected ServeSide=ad after 1st point, got %s", ms.ServeSide)
	}

	ms.AwardPoint(0)
	if ms.ServeSide != "deuce" {
		t.Errorf("expected ServeSide=deuce after 2nd point, got %s", ms.ServeSide)
	}

	ms.AwardPoint(0)
	if ms.ServeSide != "ad" {
		t.Errorf("expected ServeSide=ad after 3rd point, got %s", ms.ServeSide)
	}
}

// TestServerRotation: server alternates every game.
func TestServerRotation(t *testing.T) {
	ms := NewMatchState(0)

	if ms.Server != 0 {
		t.Fatalf("expected initial server=0, got %d", ms.Server)
	}

	// Player 0 wins game 1
	awardN(ms, 0, 4)
	if ms.Server != 1 {
		t.Errorf("expected server=1 after game 1, got %d", ms.Server)
	}

	// Player 1 wins game 2
	awardN(ms, 1, 4)
	if ms.Server != 0 {
		t.Errorf("expected server=0 after game 2, got %d", ms.Server)
	}

	// Player 0 wins game 3
	awardN(ms, 0, 4)
	if ms.Server != 1 {
		t.Errorf("expected server=1 after game 3, got %d", ms.Server)
	}
}

// TestSetWin: 6-4 wins a set, set score increments, games reset.
func TestSetWin(t *testing.T) {
	ms := NewMatchState(0)

	// Player 0 wins 6 games, player 1 wins 4 games
	// Interleave: P1 wins 4 games then P0 wins remaining 6
	for i := 0; i < 4; i++ {
		awardN(ms, 1, 4)
		awardN(ms, 0, 4)
	}
	// Score is now 4-4; player 0 wins 2 more
	awardN(ms, 0, 4)
	awardN(ms, 0, 4)

	// Player 0 should have won the set 6-4
	if ms.SetScore[0] != 1 || ms.SetScore[1] != 0 {
		t.Errorf("expected SetScore [1,0] after 6-4, got %v", ms.SetScore)
	}
	if len(ms.Sets) != 1 {
		t.Fatalf("expected 1 completed set, got %d", len(ms.Sets))
	}
	if ms.Sets[0][0] != 6 || ms.Sets[0][1] != 4 {
		t.Errorf("expected completed set [6,4], got %v", ms.Sets[0])
	}

	// Games should reset
	if ms.GameScore[0] != 0 || ms.GameScore[1] != 0 {
		t.Errorf("expected GameScore reset to [0,0] after set, got %v", ms.GameScore)
	}
}

// TestTiebreakServerRotation: in tiebreak, server rotates every 2 points after the first.
func TestTiebreakServerRotation(t *testing.T) {
	ms := NewMatchState(0)

	// Reach 6-6 in games; need to track who serves into tiebreak
	// Alternate game wins so we know the server at 6-6
	for i := 0; i < 6; i++ {
		awardN(ms, 0, 4)
		awardN(ms, 1, 4)
	}
	// After 12 games (6 each), server should be back to player 0
	// (started with 0, rotates each game: 0,1,0,1,...,0 after 12 games → 0)
	if !ms.IsTiebreak {
		t.Fatal("expected tiebreak at 6-6")
	}

	initialServer := ms.Server

	// Point 1: only 1 point before first rotation
	ms.AwardPoint(initialServer)
	if ms.Server == initialServer {
		t.Errorf("expected server to rotate after 1st tiebreak point")
	}

	serverAfterFirst := ms.Server

	// Points 2-3: serve rotates every 2 points
	ms.AwardPoint(serverAfterFirst)
	if ms.Server != serverAfterFirst {
		t.Errorf("expected same server for point 2 (no rotation yet)")
	}
	ms.AwardPoint(serverAfterFirst)
	// After 2 points by serverAfterFirst, should rotate back
	if ms.Server == serverAfterFirst {
		t.Errorf("expected server to rotate after 2 points in tiebreak")
	}
}

// TestPointNumberIncrements: PointNumber increments with each point.
func TestPointNumberIncrements(t *testing.T) {
	ms := NewMatchState(0)

	if ms.PointNumber != 0 {
		t.Fatalf("expected PointNumber=0 initially, got %d", ms.PointNumber)
	}

	for i := 1; i <= 5; i++ {
		ms.AwardPoint(0)
		if ms.PointNumber != i {
			t.Errorf("expected PointNumber=%d after %d points, got %d", i, i, ms.PointNumber)
		}
	}
}
