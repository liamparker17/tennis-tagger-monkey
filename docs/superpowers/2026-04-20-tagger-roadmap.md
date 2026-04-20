# Tennis Tagger — Human-Parity Roadmap

**Date:** 2026-04-20
**Goal:** Replace the ball-tracking pipeline with an event + scene model trained on the full Dartfish label set, and reach aggregate-stat parity with a human tagger on ~100 hours of diverse footage.

---

## Why we're rewriting

The current pipeline is **ball-tracker-centric**: detect the ball every frame, fit trajectories, infer everything else. Two fundamental problems:

1. **Ball detection is unreliable** (~15-20% recall) and the architecture has no graceful fallback.
2. **Human taggers don't track the ball continuously.** They detect events (bounce, contact, direction change) and read player movement to infer where the ball is.

We're shifting to a per-point clip model that consumes pose, audio, court geometry, and (optionally) ball detections, and predicts everything Dartfish encodes via parallel multi-task heads.

## Target architecture

```
Match video
  ↓
[Per-point clipper]                    ← Plan 1
  ↓
[Per-frame feature extractor]          ← Plan 2A   ┐ both feed Plan 3
[Contact-frame labeling tool]          ← Plan 2B   ┘
  ↓
[Multi-task model (6 heads)]           ← Plan 3
  ↓
[Inference + CSV writer]               ← Plan 4
```

## Plan list

| # | Plan | Depends on | Independently testable? |
|---|---|---|---|
| 1 | Clip + multi-label extractor | none | yes — inspect output JSON + clips |
| 2 | Per-frame feature extractor **+ contact-frame labeling tool** (parallel sub-streams 2A and 2B) | 1 | yes — `.npz` cache + labeler app |
| 3 | Multi-task model + training | 1, 2 | yes — eval on holdout matches |
| 4 | Go pipeline rewrite + CSV writer | 3 | yes — end-to-end on a test match |

Sub-streams **2A** (Python feature extractor) and **2B** (browser labeling tool) can be done in parallel by different workers; both feed Plan 3.

## Scope of each plan

### Plan 1 — Clip + multi-label extractor
For every `<video>.mp4` + `<video>.csv` pair under `files/data/training_pairs/`:
- Cut a clip per Dartfish point (point start − 1s to point end + 1s).
- Parse every Dartfish supervision column we'll use (server, returner, all 4 contact XYs, all 4 placements, all 4 stroke types, last shot, winner/error, point won, score state, surface, hands, speed).
- Emit `files/data/clips/<match>/p_NNNN.mp4` + one `labels.json` per match.

No ML, no GPU.

### Plan 2 — Features + contact labeler (parallel)
**2A — Per-frame feature extractor.** For each clip from Plan 1: pose (YOLOv8s-pose) per frame, court homography per clip, log-mel audio, sparse YOLO-ball. Cache one `.npz` per clip + `index.json` per match.

**2B — Contact-frame labeling tool.** Tiny stdlib HTTP server + single-page UI for hand-labeling per-shot contact frames into `<match>/contact_labels.json`. Target: 5–10 hours of footage labeled. Feeds the contact-event head in Plan 3.

### Plan 3 — Multi-task model + training
- Transformer encoder (~5M params) over per-frame 243-dim features.
- 6 heads: contact event, bounce event, stroke type, in/out, outcome, hitter.
- Strong supervision (Plan 2B contact frames) where available, weak heuristics elsewhere.
- Deterministic 10% holdout by match-name SHA1.
- Reports per-head accuracy / F1 on holdout.

### Plan 4 — Go pipeline rewrite + CSV writer
- New `internal/pointmodel/` package: shells out to a Python inference server running the trained model on a clip.
- Replace `internal/point/SegmentShots` and `RecognizePoint` with model-output adapters.
- CSV writer fills Dartfish columns directly from `PointPrediction` fields.
- End-to-end test on `testdata/sample_a.mp4` and one held-out training match.

**Done:** `./tagger.exe video.mp4` produces a Dartfish CSV that matches a human-tagged CSV within 5% on aggregate metrics.

## Non-goals

- Real-time / live tagging (on hold).
- Frontend match-review UI beyond the labeling tool in Plan 2B.
- Per-frame ball trajectory in the output CSV — optional YOLO-ball pass for visualization can be added later.
- Replacing the human tagger entirely — target is "human spot-checks".

## Data budget reminder

- ~100 hours of diverse footage → comfortably mid-Tier-3 (shot-by-shot human parity) for well-supervised heads.
- Rare strokes (smash, drop shot) lag — flagged demo-quality until ~250h.
- Contact-frame head's data hunger is fixed by Plan 2B labeling, not by collecting more matches.

## Order of operations

1. **Now:** Plan 1.
2. After Plan 1 merges: Plan 2A and 2B in parallel.
3. After both: Plan 3.
4. After Plan 3 hits accuracy targets: Plan 4.

Each plan ships standalone — no big-bang merge.
