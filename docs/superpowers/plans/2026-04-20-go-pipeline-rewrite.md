# Go Pipeline Rewrite + Bayesian Fusion + CSV Writer — Plan 4

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Replace `internal/point/SegmentShots` and `RecognizePoint` with output from the Plan 3 multi-task model, run that output through a **Bayesian fusion layer** that combines noisy head probabilities with structural priors and cross-modal markers, and emit a Dartfish-format CSV that matches a human-tagged CSV within 5% on aggregate metrics.

**Architecture:**

```
match.mp4
  ↓
[Go] internal/pipeline (existing) extracts frames, points already segmented coarsely
  ↓
[Go] internal/pointmodel (NEW) — for each candidate point window, JSON-RPC call to Python
  ↓
[Python] ml/inference_server — loads PointModel checkpoint, runs feature extractor inline,
         returns FusedPointPrediction (Bayesian-fused over raw head logits)
  ↓
[Go] internal/point.PointFromPrediction — adapter from FusedPointPrediction → existing Shot/Point
  ↓
[Go] internal/export.WriteDartfish (extended) — fills 120 cols from fused fields
```

**Why Bayesian fusion (not argmax):** the model's per-head argmax ignores structural constraints (shot 1 of a point is always a Serve, hitters alternate, an Ace implies stroke_count==1 AND outcome was decided on the serve). It also ignores cross-modal markers we can derive cheaply at fusion time without retraining: audio impulse alignment with predicted contact frames, near-net court position gating Volley, deuce-vs-ad serve box gating serve placement. We treat each head's softmax as a likelihood over the latent variable, multiply by structural priors in log-space, and renormalize.

**Tech stack:** Go 1.26 (existing), Python 3.11 (existing JSON-RPC bridge pattern from `internal/bridge/`), PyTorch (loaded once in long-lived inference server).

---

## File layout

```
ml/inference_server/
  __init__.py
  server.py              # JSON-RPC server (extends pattern from ml/bridge_server.py)
  predict.py             # load checkpoint, run feature_extractor + PointModel + fusion
ml/point_model/
  fusion.py              # Bayesian fusion: heads × priors → FusedPointPrediction
ml/tests/
  test_fusion.py
  test_inference_server.py

internal/pointmodel/
  client.go              # JSON-RPC client to ml/inference_server
  client_test.go
  types.go               # Go mirror of FusedPointPrediction
internal/point/
  from_prediction.go     # FusedPointPrediction → []Shot, Point, MatchState updates
  from_prediction_test.go
internal/export/
  dartfish.go            # extend WriteDartfish to read fused fields (modify, don't replace)
  dartfish_test.go       # extend
cmd/tagger/
  main.go                # add --use-pointmodel flag (modify)
```

---

## Bayesian fusion — design

For each candidate point clip, the model emits raw heads:

- `contact_logits` (T,) — sigmoid → per-frame contact probability
- `bounce_logits` (T,) — per-frame bounce probability
- `hitter_per_frame_logits` (T, 2)
- `stroke_logits` (4, 9) — 4 shot slots × 9 stroke classes
- `inout_logits` (4,)
- `outcome_logits` (5,)

**Step 1 — extract candidate contact frames.** Run NMS on `contact_prob` with min-distance ≈ 12 frames (0.4s @ 30fps). Threshold = 0.3 (intentionally loose; fusion will drop false positives). This yields candidate shot timestamps.

**Step 2 — apply structural priors per shot:**

| Prior | Form | Strength |
|---|---|---|
| Shot 1 is Serve | `log P(stroke=Serve\|i=0) += 4.0` | hard |
| Shot i+1 hitter ≠ shot i hitter | `log P(hitter=1-prev) += 3.0` | hard-ish |
| Volley requires court y < 0.4 (player near net) | `log P(stroke=Volley) += -3.0` if not near net | gating |
| Smash requires `pose_court[hitter].arms_above_head` true | `log P(stroke=Smash) += -3.0` if not | gating |
| Ace ⇒ stroke_count == 1 | `log P(outcome=Ace) += -inf` if N_shots > 1 | hard |
| DoubleFault ⇒ stroke_count ∈ {1, 2} of all serves | similar | hard |
| Audio impulse within ±2 frames of contact_t | `log P(contact_t) += 1.5` per matched impulse | soft |

**Step 3 — combine in log-space:** `log P_final(class) = log P_model(class) + sum_priors(log P_prior(class))`, renormalize per categorical head.

**Step 4 — emit `FusedPointPrediction`** with the argmax **after** fusion, plus calibrated probabilities so the CSV writer can flag low-confidence rows.

Audio impulses: extracted from `audio_mel` by detecting frames where the mel-spectrogram's high-frequency band (mel bins 40–60) jumps > 1.5 σ above local mean — proxy for racquet contact "pop". Already in the `.npz` from Plan 2A; no new feature extraction.

---

## JSON-RPC contract

```json
// Request
{"jsonrpc":"2.0","id":1,"method":"predict_point","params":{
   "clip_path": "<absolute path to .mp4>",
   "point_index": 0
}}

// Response
{"jsonrpc":"2.0","id":1,"result":{
   "contact_frames": [12, 47, 81],
   "bounce_frames":  [5, 24, 60, 88],
   "strokes": [
     {"index":0, "stroke":"Serve",    "hitter":0, "in_court":true,  "prob":0.94},
     {"index":1, "stroke":"Forehand", "hitter":1, "in_court":true,  "prob":0.81},
     {"index":2, "stroke":"Backhand", "hitter":0, "in_court":false, "prob":0.55}
   ],
   "outcome": "Winner",
   "outcome_prob": 0.72,
   "low_confidence": false
}}
```

`low_confidence = true` when any of: outcome_prob < 0.5, any stroke prob < 0.4, or contact_frames count disagrees with stroke count after fusion.

---

### Task 4.1 — Bayesian fusion module

**Files:** create `ml/point_model/fusion.py`, `ml/tests/test_fusion.py`.

- [ ] Failing test:

```python
# ml/tests/test_fusion.py
import numpy as np, torch
from ml.point_model.fusion import fuse, FusedPointPrediction

def _logits(probs):
    p = np.asarray(probs, np.float32)
    return torch.from_numpy(np.log(np.clip(p, 1e-6, 1.0)))

def test_serve_prior_overrides_weak_model():
    # model says shot 0 is 40% Forehand, 30% Serve — prior should flip to Serve.
    stroke = torch.zeros(4, 9)
    stroke[0, 0] = np.log(0.40)   # Forehand
    stroke[0, 2] = np.log(0.30)   # Serve
    contact = torch.full((60,), -3.0); contact[10] = 2.0; contact[40] = 2.0
    hitter = torch.zeros(60, 2); hitter[10, 0] = 2.0; hitter[40, 1] = 2.0
    inout = torch.zeros(4); outcome = torch.zeros(5)
    audio_impulse = np.zeros(60, np.bool_); audio_impulse[10] = True; audio_impulse[40] = True
    pose_court = np.zeros((60, 2, 17, 2), np.float32)  # stays at baseline → no Volley/Smash gating

    out = fuse(contact, hitter, stroke, inout, outcome,
               audio_impulse=audio_impulse, pose_court=pose_court, fps=30)
    assert isinstance(out, FusedPointPrediction)
    assert out.strokes[0].stroke == "Serve"
    assert out.strokes[0].hitter == 0
    assert out.strokes[1].hitter == 1   # alternation prior

def test_ace_requires_single_shot():
    stroke = torch.zeros(4, 9); stroke[0, 2] = 5.0   # confidently Serve
    contact = torch.full((60,), -3.0); contact[10] = 5.0; contact[40] = 5.0  # 2 contacts
    hitter = torch.zeros(60, 2); hitter[10, 0] = 5.0; hitter[40, 1] = 5.0
    outcome = torch.full((5,), -3.0); outcome[0] = 4.0   # model says Ace
    inout = torch.zeros(4)
    out = fuse(contact, hitter, stroke, inout, outcome,
               audio_impulse=np.zeros(60, np.bool_),
               pose_court=np.zeros((60, 2, 17, 2), np.float32), fps=30)
    assert out.outcome != "Ace"          # ruled out by N_shots > 1
```

- [ ] Implement:

```python
# ml/point_model/fusion.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from .vocab import STROKE_CLASSES, OUTCOME_CLASSES

CONTACT_NMS_DIST = 12        # frames
CONTACT_THRESHOLD = 0.3
SERVE_PRIOR = 4.0
ALTERNATE_PRIOR = 3.0
VOLLEY_NEAR_NET_Y = 0.4      # normalized court y; lower = near net
GATE_PENALTY = 3.0
AUDIO_BONUS = 1.5
LOW_CONF_OUTCOME = 0.5
LOW_CONF_STROKE = 0.4

SERVE_IDX   = STROKE_CLASSES.index("Serve")
VOLLEY_IDX  = STROKE_CLASSES.index("Volley")
SMASH_IDX   = STROKE_CLASSES.index("Smash")
ACE_IDX     = OUTCOME_CLASSES.index("Ace")
DF_IDX      = OUTCOME_CLASSES.index("DoubleFault")

@dataclass
class FusedStroke:
    index: int
    stroke: str
    hitter: int
    in_court: bool
    prob: float
    contact_frame: int

@dataclass
class FusedPointPrediction:
    contact_frames: list[int]
    bounce_frames: list[int]
    strokes: list[FusedStroke]
    outcome: str
    outcome_prob: float
    low_confidence: bool = False
    raw: dict = field(default_factory=dict)

def _nms_1d(probs: np.ndarray, threshold: float, min_dist: int) -> list[int]:
    cand = np.where(probs >= threshold)[0]
    if cand.size == 0: return []
    cand = sorted(cand, key=lambda i: -probs[i])
    chosen: list[int] = []
    for i in cand:
        if all(abs(i - c) >= min_dist for c in chosen):
            chosen.append(int(i))
    return sorted(chosen)

def _arms_above_head(pose_court: np.ndarray, frame: int, hitter: int) -> bool:
    if frame >= pose_court.shape[0]: return False
    kp = pose_court[frame, hitter]              # (17, 2)
    # COCO keypoints: 0=nose, 9=l_wrist, 10=r_wrist
    nose_y = kp[0, 1]
    wrist_y = min(kp[9, 1], kp[10, 1])
    return wrist_y < nose_y                     # smaller y = higher in image-court

def _hitter_court_y(pose_court: np.ndarray, frame: int, hitter: int) -> float:
    if frame >= pose_court.shape[0]: return 0.5
    return float(pose_court[frame, hitter, :, 1].mean())

def fuse(contact_logits: torch.Tensor, hitter_logits: torch.Tensor,
         stroke_logits: torch.Tensor, inout_logits: torch.Tensor,
         outcome_logits: torch.Tensor, *,
         audio_impulse: np.ndarray, pose_court: np.ndarray,
         fps: int) -> FusedPointPrediction:
    contact_p = torch.sigmoid(contact_logits).cpu().numpy()
    # Audio bonus on contact prob
    bonus = np.zeros_like(contact_p)
    impulse_idx = np.where(audio_impulse)[0]
    for ti in impulse_idx:
        lo, hi = max(0, ti - 2), min(len(contact_p), ti + 3)
        bonus[lo:hi] += AUDIO_BONUS
    contact_p_post = 1 / (1 + np.exp(-(np.log(contact_p / (1 - contact_p + 1e-6) + 1e-6) + bonus)))
    contacts = _nms_1d(contact_p_post, CONTACT_THRESHOLD, CONTACT_NMS_DIST)
    n_shots = len(contacts)

    # Per-frame hitter softmax
    hitter_p = F.softmax(hitter_logits, dim=-1).cpu().numpy()

    # Per-shot stroke priors
    n_slots = stroke_logits.shape[0]
    stroke_lp = F.log_softmax(stroke_logits, dim=-1).cpu().numpy().copy()  # (4, 9)
    fused_strokes: list[FusedStroke] = []
    prev_hitter = -1
    for i in range(min(n_shots, n_slots)):
        f = contacts[i]
        # Serve prior on shot 0
        if i == 0:
            stroke_lp[i, SERVE_IDX] += SERVE_PRIOR

        # Volley gating: only allowed near net
        if i < n_shots:
            # Determine candidate hitter from per-frame logits + alternation prior
            hp = np.log(hitter_p[f] + 1e-6).copy()
            if prev_hitter >= 0:
                hp[1 - prev_hitter] += ALTERNATE_PRIOR
            this_hitter = int(np.argmax(hp))

            y = _hitter_court_y(pose_court, f, this_hitter)
            if y > VOLLEY_NEAR_NET_Y:
                stroke_lp[i, VOLLEY_IDX] -= GATE_PENALTY
            if not _arms_above_head(pose_court, f, this_hitter):
                stroke_lp[i, SMASH_IDX] -= GATE_PENALTY
        else:
            this_hitter = 0

        probs = np.exp(stroke_lp[i] - stroke_lp[i].max())
        probs /= probs.sum()
        cls = int(np.argmax(probs))
        fused_strokes.append(FusedStroke(
            index=i, stroke=STROKE_CLASSES[cls], hitter=this_hitter,
            in_court=bool(torch.sigmoid(inout_logits[i]) > 0.5),
            prob=float(probs[cls]), contact_frame=f,
        ))
        prev_hitter = this_hitter

    # Outcome priors
    outcome_lp = F.log_softmax(outcome_logits, dim=-1).cpu().numpy().copy()
    if n_shots != 1:
        outcome_lp[ACE_IDX] = -1e9
    if n_shots not in (1, 2) and n_shots != 0:
        outcome_lp[DF_IDX] = -1e9
    op = np.exp(outcome_lp - outcome_lp.max()); op /= op.sum()
    out_idx = int(np.argmax(op)); outcome_name = OUTCOME_CLASSES[out_idx]
    outcome_prob = float(op[out_idx])

    low = (outcome_prob < LOW_CONF_OUTCOME
           or any(s.prob < LOW_CONF_STROKE for s in fused_strokes))

    return FusedPointPrediction(
        contact_frames=contacts, bounce_frames=[],
        strokes=fused_strokes, outcome=outcome_name,
        outcome_prob=outcome_prob, low_confidence=low,
    )
```

- [ ] Test passes. Commit: `feat(plan4): Bayesian fusion over multi-task heads`.

---

### Task 4.2 — Audio impulse extractor (used by fusion + server)

**Files:** add to `ml/point_model/fusion.py` (or new `audio_markers.py`).

- [ ] Extend `fusion.py`:

```python
def detect_audio_impulses(audio_mel: np.ndarray, T: int,
                          hi_band: tuple[int, int] = (40, 60),
                          z_threshold: float = 1.5) -> np.ndarray:
    """Returns (T,) bool array of impulse frames."""
    if audio_mel.size == 0: return np.zeros(T, np.bool_)
    n_mels, F_len = audio_mel.shape
    band = audio_mel[hi_band[0]:hi_band[1]].mean(axis=0)
    if band.size < 5: return np.zeros(T, np.bool_)
    mu = band.mean(); sd = band.std() + 1e-6
    impulse_F = (band - mu) / sd > z_threshold
    # Resample F → T
    idx = np.linspace(0, F_len - 1, T).astype(np.int64)
    return impulse_F[idx]
```

- [ ] Quick test:

```python
def test_impulse_resamples_to_T():
    from ml.point_model.fusion import detect_audio_impulses
    mel = np.zeros((64, 200), np.float32); mel[40:60, 100] = 5.0
    out = detect_audio_impulses(mel, T=60)
    assert out.shape == (60,)
    assert out.any()
```

Commit: `feat(plan4): audio impulse marker for fusion`.

---

### Task 4.3 — Inference adapter

**Files:** create `ml/inference_server/predict.py`, `ml/inference_server/__init__.py`.

```python
# ml/inference_server/predict.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from ml.feature_extractor.process import process_clip
from ml.feature_extractor.schema import load_npz
from ml.point_model.model import PointModel, PointModelConfig
from ml.point_model.features import build_feature_tensor
from ml.point_model.fusion import fuse, detect_audio_impulses, FusedPointPrediction

class Predictor:
    def __init__(self, ckpt: Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(ckpt, map_location=self.device)
        self.model = PointModel(PointModelConfig(**sd["config"])).to(self.device)
        self.model.load_state_dict(sd["model"]); self.model.eval()

    @torch.no_grad()
    def predict_clip(self, clip_path: Path, tmp_dir: Path) -> FusedPointPrediction:
        npz_path = tmp_dir / (clip_path.stem + ".npz")
        process_clip(clip_path, npz_path)
        fs = load_npz(npz_path)
        feats = build_feature_tensor(
            pose_px=fs.pose_px, pose_court=fs.pose_court,
            pose_conf=fs.pose_conf, pose_valid=fs.pose_valid,
            ball=fs.ball, audio_mel=fs.audio_mel, clip_meta=fs.clip_meta,
        )
        x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        mask = torch.ones(1, feats.shape[0], device=self.device)
        out = self.model(x, mask)
        T = feats.shape[0]
        impulses = detect_audio_impulses(fs.audio_mel, T)
        return fuse(
            out["contact_logits"][0], out["hitter_per_frame_logits"][0],
            out["stroke_logits"][0], out["inout_logits"][0],
            out["outcome_logits"][0],
            audio_impulse=impulses, pose_court=fs.pose_court, fps=fs.fps,
        )
```

Commit: `feat(plan4): Predictor wraps feature_extractor + model + fusion`.

---

### Task 4.4 — JSON-RPC inference server

**Files:** create `ml/inference_server/server.py`, `ml/tests/test_inference_server.py`.

Pattern after existing `ml/bridge_server.py`. Stdio-based JSON-RPC for low-latency Go ↔ Python.

- [ ] Test (smoke — gated on checkpoint):

```python
import json, subprocess, sys, pytest
from pathlib import Path

CKPT = Path("files/models/point_model/run0/best.pt")
CLIP = Path("files/data/clips/_smoke/p_0001.mp4")

@pytest.mark.skipif(not CKPT.exists() or not CLIP.exists(), reason="needs trained ckpt + clip")
def test_predict_point(tmp_path):
    p = subprocess.Popen(
        [sys.executable, "-m", "ml.inference_server.server", "--ckpt", str(CKPT)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    req = json.dumps({"jsonrpc":"2.0","id":1,"method":"predict_point",
                       "params":{"clip_path": str(CLIP), "point_index": 0}}) + "\n"
    p.stdin.write(req.encode()); p.stdin.flush()
    line = p.stdout.readline()
    p.stdin.close(); p.wait(timeout=30)
    resp = json.loads(line)
    assert "result" in resp and "strokes" in resp["result"]
```

- [ ] Implement:

```python
# ml/inference_server/server.py
from __future__ import annotations
import argparse, json, sys, tempfile
from dataclasses import asdict
from pathlib import Path
from .predict import Predictor

def _to_dict(pred):
    return {
        "contact_frames": pred.contact_frames,
        "bounce_frames": pred.bounce_frames,
        "strokes": [asdict(s) for s in pred.strokes],
        "outcome": pred.outcome,
        "outcome_prob": pred.outcome_prob,
        "low_confidence": pred.low_confidence,
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser("inference_server")
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args(argv)

    pred = Predictor(args.ckpt)
    tmp = Path(tempfile.mkdtemp())

    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            method = req.get("method"); rid = req.get("id")
            if method == "predict_point":
                clip = Path(req["params"]["clip_path"])
                fp = pred.predict_clip(clip, tmp)
                resp = {"jsonrpc":"2.0","id":rid,"result": _to_dict(fp)}
            elif method == "ping":
                resp = {"jsonrpc":"2.0","id":rid,"result":"pong"}
            elif method == "shutdown":
                print(json.dumps({"jsonrpc":"2.0","id":rid,"result":"bye"}), flush=True)
                return 0
            else:
                resp = {"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"no such method"}}
        except Exception as e:
            resp = {"jsonrpc":"2.0","id":req.get("id") if 'req' in locals() else None,
                    "error":{"code":-32000,"message":str(e)}}
        print(json.dumps(resp), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Commit: `feat(plan4): JSON-RPC inference server`.

---

### Task 4.5 — Go client + types

**Files:** create `internal/pointmodel/types.go`, `internal/pointmodel/client.go`, `internal/pointmodel/client_test.go`.

- [ ] `types.go`:

```go
package pointmodel

type FusedStroke struct {
	Index        int     `json:"index"`
	Stroke       string  `json:"stroke"`
	Hitter       int     `json:"hitter"`
	InCourt      bool    `json:"in_court"`
	Prob         float64 `json:"prob"`
	ContactFrame int     `json:"contact_frame"`
}

type FusedPointPrediction struct {
	ContactFrames []int         `json:"contact_frames"`
	BounceFrames  []int         `json:"bounce_frames"`
	Strokes       []FusedStroke `json:"strokes"`
	Outcome       string        `json:"outcome"`
	OutcomeProb   float64       `json:"outcome_prob"`
	LowConfidence bool          `json:"low_confidence"`
}
```

- [ ] `client.go`:

```go
package pointmodel

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"sync/atomic"
)

type Client struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex
	nextID int64
}

type rpcReq struct {
	JSONRPC string         `json:"jsonrpc"`
	ID      int64          `json:"id"`
	Method  string         `json:"method"`
	Params  map[string]any `json:"params,omitempty"`
}

type rpcResp struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func Start(python string, ckpt string) (*Client, error) {
	cmd := exec.Command(python, "-m", "ml.inference_server.server", "--ckpt", ckpt)
	stdin, err := cmd.StdinPipe()
	if err != nil { return nil, err }
	stdout, err := cmd.StdoutPipe()
	if err != nil { return nil, err }
	if err := cmd.Start(); err != nil { return nil, err }
	return &Client{cmd: cmd, stdin: stdin, stdout: bufio.NewReader(stdout)}, nil
}

func (c *Client) call(method string, params map[string]any, out any) error {
	c.mu.Lock(); defer c.mu.Unlock()
	id := atomic.AddInt64(&c.nextID, 1)
	req := rpcReq{JSONRPC: "2.0", ID: id, Method: method, Params: params}
	b, err := json.Marshal(req); if err != nil { return err }
	if _, err := c.stdin.Write(append(b, '\n')); err != nil { return err }
	line, err := c.stdout.ReadBytes('\n'); if err != nil { return err }
	var resp rpcResp
	if err := json.Unmarshal(line, &resp); err != nil { return err }
	if resp.Error != nil { return fmt.Errorf("rpc: %s", resp.Error.Message) }
	if out == nil { return nil }
	return json.Unmarshal(resp.Result, out)
}

func (c *Client) PredictPoint(clipPath string) (*FusedPointPrediction, error) {
	var out FusedPointPrediction
	err := c.call("predict_point", map[string]any{"clip_path": clipPath}, &out)
	if err != nil { return nil, err }
	return &out, nil
}

func (c *Client) Close() error {
	_ = c.call("shutdown", nil, nil)
	_ = c.stdin.Close()
	return c.cmd.Wait()
}
```

- [ ] `client_test.go` — table-driven against a fake stdio process (use `os/exec` with `cat`-style echo script), or skip if not feasible. Commit: `feat(plan4): Go JSON-RPC client for inference server`.

---

### Task 4.6 — Adapter: FusedPointPrediction → Shot/Point

**Files:** create `internal/point/from_prediction.go`, `internal/point/from_prediction_test.go`.

- [ ] Test:

```go
package point

import (
	"testing"
	"tennis-tagger/internal/pointmodel"
)

func TestPointFromPrediction(t *testing.T) {
	pred := &pointmodel.FusedPointPrediction{
		ContactFrames: []int{10, 40, 80},
		Strokes: []pointmodel.FusedStroke{
			{Index: 0, Stroke: "Serve",    Hitter: 0, InCourt: true, ContactFrame: 10, Prob: 0.94},
			{Index: 1, Stroke: "Forehand", Hitter: 1, InCourt: true, ContactFrame: 40, Prob: 0.82},
			{Index: 2, Stroke: "Backhand", Hitter: 0, InCourt: true, ContactFrame: 80, Prob: 0.65},
		},
		Outcome: "Winner", OutcomeProb: 0.71,
	}
	pt := PointFromPrediction(pred, 0.0, 30.0)
	if len(pt.Shots) != 3 {
		t.Fatalf("want 3 shots, got %d", len(pt.Shots))
	}
	if pt.Shots[0].StrokeType != "Serve" {
		t.Errorf("shot 0 stroke = %q, want Serve", pt.Shots[0].StrokeType)
	}
	if pt.WinnerOrError != "Winner" { t.Errorf("outcome = %q", pt.WinnerOrError) }
}
```

- [ ] Implement (signatures may need adjustment to existing `Shot`/`Point` definitions in `internal/point/`):

```go
package point

import (
	"tennis-tagger/internal/pointmodel"
)

func PointFromPrediction(pred *pointmodel.FusedPointPrediction, clipStartS, fps float64) *Point {
	shots := make([]Shot, 0, len(pred.Strokes))
	for _, s := range pred.Strokes {
		shots = append(shots, Shot{
			StrokeType:    s.Stroke,
			Hitter:        s.Hitter,
			InCourt:       s.InCourt,
			TimeS:         clipStartS + float64(s.ContactFrame)/fps,
			Confidence:    s.Prob,
		})
	}
	return &Point{
		Shots:         shots,
		WinnerOrError: pred.Outcome,
		Confidence:    pred.OutcomeProb,
		LowConfidence: pred.LowConfidence,
	}
}
```

If `Point`/`Shot` lack the fields above, add them — Plan 3's `PointPrediction` is the contract; the existing structs predate it. Keep backwards-compatible when possible. Commit: `feat(plan4): adapter from fused prediction to Shot/Point`.

---

### Task 4.7 — Wire into pipeline behind a flag

**Files:** modify `cmd/tagger/main.go`, `internal/app/app.go`, `internal/pipeline/concurrent.go` as needed.

- [ ] Add `--use-pointmodel` flag and `--pointmodel-ckpt` (default `files/models/point_model/run0/best.pt`).
- [ ] When set, the pipeline:
  1. Cuts each candidate point window to a temp clip via existing ffmpeg helpers.
  2. Calls `pointmodel.Client.PredictPoint(clipPath)`.
  3. Replaces the existing `SegmentShots`/`RecognizePoint` output with `PointFromPrediction`.
- [ ] When unset: zero behavior change — old path runs.
- [ ] Test: run `./tagger.exe --use-pointmodel testdata/sample_a.mp4` after model exists; confirm CSV is written and not crashing.

Commit: `feat(plan4): --use-pointmodel flag wires inference server into pipeline`.

---

### Task 4.8 — Extend CSV writer

**Files:** modify `internal/export/dartfish.go`, `internal/export/dartfish_test.go`.

The existing writer takes `MatchState`/`Point`/`Shot`. After the adapter in Task 4.6, the same structs carry fused fields — most columns just work. New behavior:

- [ ] If `Point.LowConfidence` is true, prefix the row's "Notes" column (or whatever free-text column the format uses) with `[LOW_CONF]`.
- [ ] When `Point.Confidence < 0.5`, leave the speed column blank (we have no source for it from the model).
- [ ] Update test fixtures.

Commit: `feat(plan4): CSV writer flags low-confidence rows`.

---

### Task 4.9 — End-to-end

- [ ] Pick one held-out match (one of the 10 from Plan 3's deterministic split). Get its human-tagged CSV from `files/data/training_pairs/<match>/`.
- [ ] Run: `./tagger.exe --use-pointmodel files/data/training_pairs/<match>/<video>.mp4 -o /tmp/predicted.csv`.
- [ ] Compare aggregate metrics with a small Python script (one-off, do not check in unless useful):
  - Total points, total shots
  - Winner / Error / Ace / DoubleFault counts
  - Per-stroke counts (Serve/Forehand/Backhand)
- [ ] **Pass criterion:** all aggregate counts within 5% of human-tagged values.
- [ ] Document the comparison numbers in the commit message.

Commit: `test(plan4): end-to-end on held-out match — within Xpct of human tagger`.

---

## Done when

- `pytest ml/tests/test_fusion.py ml/tests/test_inference_server.py -v` is green.
- `go test ./internal/pointmodel/... ./internal/point/... ./internal/export/... -v` is green.
- `./tagger.exe --use-pointmodel <held-out>.mp4` produces a Dartfish CSV that aggregates within 5% of the corresponding human-tagged CSV.
- The old non-pointmodel path still runs unchanged when the flag is absent.

## Out of scope

- Per-frame ball trajectory in CSV (optional later visualization pass).
- Re-training the model from corrections (existing `internal/corrections` can feed Plan 3's training loop later).
- Live / real-time tagging.
