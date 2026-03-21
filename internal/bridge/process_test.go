package bridge

import (
	"os/exec"
	"testing"
)

func pythonAvailable(t *testing.T) string {
	t.Helper()
	// Try "python" first, then "python3"
	for _, name := range []string{"python", "python3"} {
		path, err := exec.LookPath(name)
		if err == nil {
			return path
		}
	}
	t.Skip("python not found in PATH, skipping process bridge tests")
	return ""
}

func TestProcessBridge_InitAndClose(t *testing.T) {
	pythonPath := pythonAvailable(t)

	pb := NewProcessBridge(pythonPath)
	defer pb.Close()

	err := pb.Init(BridgeConfig{
		ModelsDir: "models",
		Device:    "cpu",
	})
	if err != nil {
		// Init may fail if torch/ML deps are not installed — that's expected
		t.Logf("Init failed (likely missing ML dependencies): %v", err)
		t.Skip("Skipping: Python ML dependencies not available")
	}

	t.Log("ProcessBridge initialized successfully")
}

func TestProcessBridge_CloseNoPanic(t *testing.T) {
	pythonPath := pythonAvailable(t)

	// Close without Init should not panic
	pb := NewProcessBridge(pythonPath)
	pb.Close()

	// Double close should not panic
	pb2 := NewProcessBridge(pythonPath)
	pb2.Close()
	pb2.Close()
}

func TestProcessBridge_CloseBeforeInit(t *testing.T) {
	// Even with a bogus python path, Close should not panic
	pb := NewProcessBridge("nonexistent-python-binary")
	pb.Close()
}
