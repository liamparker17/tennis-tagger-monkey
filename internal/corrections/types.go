package corrections

// Correction represents a single user correction to a model prediction.
type Correction struct {
	ID         string `json:"id"`
	VideoPath  string `json:"video_path"`
	FrameIndex int    `json:"frame_index"`
	Type       string `json:"type"` // "stroke", "placement", "detection", "false_positive"
	Original   string `json:"original"`
	Corrected  string `json:"corrected"`
	PlayerID   int    `json:"player_id"`
	Timestamp  string `json:"timestamp"`
}

// CorrectionBatch groups corrections for a training/fine-tuning run.
type CorrectionBatch struct {
	Corrections  []Correction `json:"corrections"`
	ModelVersion string       `json:"model_version"`
}
