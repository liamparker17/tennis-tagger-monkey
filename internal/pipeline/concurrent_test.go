package pipeline

import (
	"testing"

	"github.com/liamp/tennis-tagger/internal/bridge"
	"github.com/liamp/tennis-tagger/internal/config"
)

func TestProcessConcurrent_WithMock(t *testing.T) {
	cfg := config.Default()
	mock := bridge.NewMockBridge()
	p := New(mock, cfg)

	result, err := p.ProcessConcurrent(testVideoPath(t))
	if err != nil {
		t.Fatalf("ProcessConcurrent: %v", err)
	}
	if result.ProcessedFrames == 0 {
		t.Error("expected non-zero processed frames")
	}
	if result.TotalFrames == 0 {
		t.Error("expected non-zero total frames")
	}
	if p.Progress().Stage != "complete" {
		t.Errorf("stage = %q, want complete", p.Progress().Stage)
	}
}
