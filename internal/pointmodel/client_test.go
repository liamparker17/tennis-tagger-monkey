package pointmodel

import (
	"os/exec"
	"runtime"
	"testing"
)

// Fake server: for each line of stdin, emit a canned FusedPointPrediction JSON-RPC result,
// except for the shutdown method which emits "bye".
const fakeServerPy = `
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    rid = req.get("id"); m = req.get("method")
    if m == "shutdown":
        print(json.dumps({"jsonrpc":"2.0","id":rid,"result":"bye"}), flush=True); break
    if m == "ping":
        print(json.dumps({"jsonrpc":"2.0","id":rid,"result":"pong"}), flush=True); continue
    if m == "predict_point":
        result = {
            "contact_frames": [10, 40],
            "bounce_frames": [],
            "strokes": [
                {"index":0,"stroke":"Serve","hitter":0,"in_court":True,"prob":0.9,"contact_frame":10},
                {"index":1,"stroke":"Forehand","hitter":1,"in_court":True,"prob":0.8,"contact_frame":40},
            ],
            "outcome": "Winner", "outcome_prob": 0.7, "low_confidence": False,
        }
        print(json.dumps({"jsonrpc":"2.0","id":rid,"result":result}), flush=True); continue
    print(json.dumps({"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"?"}}), flush=True)
`

func TestClientPredictPoint(t *testing.T) {
	py, err := exec.LookPath("python")
	if err != nil && runtime.GOOS == "windows" {
		py, err = exec.LookPath("python.exe")
	}
	if err != nil {
		t.Skip("python not on PATH")
	}
	cmd := exec.Command(py, "-c", fakeServerPy)
	c, err := StartCmd(cmd)
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	defer c.Close()

	if err := c.Ping(); err != nil {
		t.Fatalf("ping: %v", err)
	}
	pred, err := c.PredictPoint("x.mp4")
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	if len(pred.Strokes) != 2 {
		t.Fatalf("want 2 strokes, got %d", len(pred.Strokes))
	}
	if pred.Strokes[0].Stroke != "Serve" {
		t.Errorf("stroke[0]=%q", pred.Strokes[0].Stroke)
	}
	if pred.Outcome != "Winner" {
		t.Errorf("outcome=%q", pred.Outcome)
	}
}
