package export

// DartfishColumns is the exact 116-column header produced by the real
// Dartfish tagging workflow the human tagger uses. Taken verbatim from
// a reference match export so that our CSV can be opened / diffed / fed
// back into the same downstream tooling the human's export feeds.
//
// IMPORTANT: this header is the source of truth. The column INDEX
// constants below are what the pipeline writes to — keep them in sync
// if you re-order.
var DartfishColumns = []string{
	/* 00 */ "Name",
	/* 01 */ "Position",
	/* 02 */ "Duration",
	/* 03 */ "A1: Server",
	/* 04 */ "A2: Serve Data",
	/* 05 */ "A3: Serve Placement",
	/* 06 */ "B1: Returner",
	/* 07 */ "B2: Return Data",
	/* 08 */ "B3: Return Placement",
	/* 09 */ "B7: Return Type",
	/* 10 */ "C1: Serve +1 Stroke",
	/* 11 */ "C2: Serve +1 Data",
	/* 12 */ "C3: Serve +1 Placement",
	/* 13 */ "D1: Return +1 Stroke",
	/* 14 */ "D2: Return +1 Data",
	/* 15 */ "D3: Return +1 Placement",
	/* 16 */ "E1: Last Shot",
	/* 17 */ "E2: Last Shot Winner",
	/* 18 */ "E3: Last Shot Error",
	/* 19 */ "E4: Last Shot Placement",
	/* 20 */ "F1: Point Won",
	/* 21 */ "F2: Point Score",
	/* 22 */ "F3: Tiebreak Score",
	/* 23 */ "F4: Break Point",
	/* 24 */ "F5: Point Penalty",
	/* 25 */ "G1: Game Score",
	/* 26 */ "G2: Game Won",
	/* 27 */ "G3: Set #",
	/* 28 */ "G4: Set Score",
	/* 29 */ "G5: Set Won",
	/* 30 */ "G6: Tiebreaker Won",
	/* 31 */ "H1: Stroke Count",
	/* 32 */ "H2: Deuce Ad",
	/* 33 */ "I1: Net Appearance Player A",
	/* 34 */ "I2: Net Appearance Player B",
	/* 35 */ "K1: Final Shot Player",
	/* 36 */ "K2: Final Shot Stroke",
	/* 37 */ "K3: Final Shot Player A",
	/* 38 */ "K5: Final Shot Player B",
	/* 39 */ "x: Client Team",
	/* 40 */ "x: Date",
	/* 41 */ "x: Division",
	/* 42 */ "x: Event",
	/* 43 */ "x: Lineup Position",
	/* 44 */ "x: Match Details",
	/* 45 */ "x: Opponent Team",
	/* 46 */ "x: Player A",
	/* 47 */ "x: Player A Hand",
	/* 48 */ "x: Player A Set Score",
	/* 49 */ "x: Player B",
	/* 50 */ "x: Player B Hand",
	/* 51 */ "x: Player B Set Score",
	/* 52 */ "x: Round",
	/* 53 */ "x: Surface",
	/* 54 */ "XY2 Last Shot Contact",
	/* 55 */ "XY2 Ret+1 Contact",
	/* 56 */ "XY2 Return Contact",
	/* 57 */ "XY2 Srv+1 Contact",
	/* 58 */ "z - 1st Serve Data",
	/* 59 */ "z - 1st Serve Zone",
	/* 60 */ "z - 2nd Serve Zone",
	/* 61 */ "z - Final Shot Stroke",
	/* 62 */ "z - Game Score",
	/* 63 */ "z - Last Shot Contact",
	/* 64 */ "z - Last Shot Contact Depth",
	/* 65 */ "z - Last Shot Contact Lane",
	/* 66 */ "z - Last Shot Placement",
	/* 67 */ "z - Point Score",
	/* 68 */ "z - Return +1 Contact",
	/* 69 */ "z - Return +1 Contact Depth",
	/* 70 */ "z - Return +1 Contact Lane",
	/* 71 */ "z - Return +1 Placement",
	/* 72 */ "z - Return +1 Stroke",
	/* 73 */ "z - Return Contact",
	/* 74 */ "z - Return Contact Depth",
	/* 75 */ "z - Return Contact Lane",
	/* 76 */ "z - Return Placement",
	/* 77 */ "z - Serve +1 Contact",
	/* 78 */ "z - Serve +1 Contact Depth",
	/* 79 */ "z - Serve +1 Contact Lane",
	/* 80 */ "z - Serve +1 Placement",
	/* 81 */ "z - Serve +1 Stroke",
	/* 82 */ "z - Set Score",
	/* 83 */ "z - Stroke Count",
	/* 84 */ "z - Tiebreak Score",
	/* 85 */ "XY Deuce",
	/* 86 */ "z - Point Lost",
	/* 87 */ "zz - Final Shot",
	/* 88 */ "zz - Point Type",
	/* 89 */ "zz - Pt Won",
	/* 90 */ "zz - Rally Length",
	/* 91 */ "zz - Serve Data",
	/* 92 */ "zz - Server",
	/* 93 */ "zz - W-E",
	/* 94 */ "zz - W-E Player",
	/* 95 */ "x: 1st Serve Fault Data",
	/* 96 */ "x: 1st Serve Fault Placement",
	/* 97 */ "XY Ad",
	/* 98 */ "XY Ad 1st Fault",
	/* 99 */ "XY Last Shot",
	/*100 */ "XY Ret+1",
	/*101 */ "XY Return",
	/*102 */ "XY Srv+1",
	/*103 */ "z - Last Shot Stroke",
	/*104 */ "zz - Rally Contact Lane",
	/*105 */ "zz - Rally Depth",
	/*106 */ "zz - Rally FH-BH",
	/*107 */ "zz - Rally Placement",
	/*108 */ "zz - Rally Stroke",
	/*109 */ "zz - Return Stroke",
	/*110 */ "z - Net Appearance B",
	/*111 */ "zz - Net B",
	/*112 */ "XY Deuce 1st Fault",
	/*113 */ "zz - Rally Error",
	/*114 */ "z - Net Appearance A",
	/*115 */ "zz - Net A",
}

// Column indices — write to these instead of magic numbers.
const (
	ColName                 = 0
	ColPosition             = 1
	ColDuration             = 2
	ColServer               = 3   // A1
	ColServeData            = 4   // A2
	ColServePlacement       = 5   // A3
	ColReturner             = 6   // B1
	ColReturnData           = 7   // B2
	ColReturnPlacement      = 8   // B3
	ColReturnType           = 9   // B7
	ColServePlus1Stroke     = 10  // C1
	ColServePlus1Data       = 11  // C2
	ColServePlus1Placement  = 12  // C3
	ColReturnPlus1Stroke    = 13  // D1
	ColReturnPlus1Data      = 14  // D2
	ColReturnPlus1Placement = 15  // D3
	ColLastShot             = 16  // E1
	ColLastShotWinner       = 17  // E2
	ColLastShotError        = 18  // E3
	ColLastShotPlacement    = 19  // E4
	ColPointWon             = 20  // F1
	ColPointScore           = 21  // F2
	ColTiebreakScore        = 22  // F3
	ColBreakPoint           = 23  // F4
	ColPointPenalty         = 24  // F5
	ColGameScore            = 25  // G1
	ColGameWon              = 26  // G2
	ColSetNumber            = 27  // G3
	ColSetScore             = 28  // G4
	ColSetWon               = 29  // G5
	ColTiebreakerWon        = 30  // G6
	ColStrokeCount          = 31  // H1
	ColDeuceAd              = 32  // H2
	ColNetAppearanceA       = 33  // I1
	ColNetAppearanceB       = 34  // I2
	ColFinalShotPlayer      = 35  // K1
	ColFinalShotStroke      = 36  // K2
	ColPlayerA              = 46  // x: Player A
	ColPlayerB              = 49  // x: Player B
	ColXYDeuce              = 85  // XY Deuce (serve bounce, deuce court)
	ColXYAd                 = 97  // XY Ad    (serve bounce, ad court)
	ColXYLastShot           = 99  // XY Last Shot
	ColXYRetPlus1           = 100 // XY Ret+1
	ColXYReturn             = 101 // XY Return
	ColXYSrvPlus1           = 102 // XY Srv+1
)
