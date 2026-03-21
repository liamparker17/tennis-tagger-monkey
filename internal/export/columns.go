package export

// DartfishColumns defines the 62-column header for Dartfish CSV export.
// These names are copied character-for-character from the Python
// DartfishCSVGenerator.COLUMNS list in files/analysis/csv_generator.py.
var DartfishColumns = []string{
	// Basic info
	"Name", "Position", "Duration",

	// Point level
	"0 - Point Level",

	// Server info (A)
	"A1: Server", "A2: Serve Data", "A3: Serve Placement",

	// Returner info (B)
	"B1: Returner", "B2: Return Data", "B3: Return Placement",

	// Serve +1 (C)
	"C1: Serve +1 Stroke", "C2: Serve +1 Data", "C3: Serve +1 Placement",

	// Return +1 (D)
	"D1: Return +1 Stroke", "D2: Return +1 Data", "D3: Return +1 Placement",

	// Last shot (E)
	"E1: Last Shot", "E2: Last Shot Winner", "E3: Last Shot Error", "E4: Last Shot Placement",

	// Point outcome (F)
	"F1: Point Won", "F2: Point Score",

	// Rally data (G)
	"G1: Rally Length", "G2: Total Strokes",

	// Net appearances (H)
	"H1: Server Net", "H2: Returner Net",

	// Additional metrics (I-Z)
	"I1: Winner Type", "I2: Error Type",
	"J1: Break Point", "J2: Set Point", "J3: Match Point",
	"K1: Deuce/Ad", "K2: Game Score",
	"L1: Set Number", "L2: Game Number",
	"M1: Serve Speed", "M2: Serve Number",
	"N1: First Serve In", "N2: Second Serve",
	"O1: Return Quality", "O2: Return Depth",
	"P1: Approach Shot", "P2: Passing Shot",
	"Q1: Court Position Server", "Q2: Court Position Returner",
	"R1: Dominant Hand Server", "R2: Dominant Hand Returner",
	"S1: Time Between Points",
	"T1: Video Timestamp", "T2: Frame Number",
	"U1: Confidence Score",
	"V1: Notes", "V2: Tags",
	"W1: Weather Conditions", "W2: Court Surface",
	"X1: Player 1 Name", "X2: Player 2 Name",
	"Y1: Tournament", "Y2: Round",
	"Z1: Serve Contact Depth", "Z2: Return Contact Depth",
	"Z3: Serve +1 Contact Depth",
}
