# Ball Trajectory Prediction & Point Recognition System

**Date:** 2026-03-23
**Status:** Approved (pending implementation)

## Overview

A 5-layer system that transforms sparse ball detections into full shot-level tennis analytics: trajectory prediction, bounce detection, in/out calls, point outcomes, and running score tracking. Designed for fixed-camera footage of any court surface and resolution.

## Constraints

- Fixed camera only (no pan/tilt/zoom). Court detection runs once per video.
- No scoreboard dependency. Score is derived entirely from ball trajectory + court boundaries + player positions. OCR scoreboard is a validation signal when available, not a requirement.
- Must handle ball speeds from 40kph (lob) to 220kph (serve). Fast shots produce near-linear trajectories; slow shots are parabolic.
- Target: accurate enough to auto-score matches. Mistakes caught in QC feedback loop.

## Architecture

```
Layer 1: Ball Detection (dense)
  TrackNet (heatmap, ~85% frames) + YOLO (bounding box, ~5% frames)
  -> Merged ball positions with confidence scores

Layer 2: Trajectory Fitting (physics)
  Ball positions -> Arc fitting in court-space (linear to parabolic)
  -> Predicted full trajectory including bounce point

Layer 3: Shot Segmentation
  Trajectory + bounce points + player positions
  -> Individual shots: who hit, where it landed

Layer 4: Point Recognition
  Shot sequence + court boundaries + tennis rules
  -> Point outcome: winner/error/ace/double-fault, who won

Layer 5: Score Tracking
  Point outcomes + tennis scoring rules
  -> Running score: games, sets, match score
```

Each layer is a pure function of its inputs. No circular dependencies. Layer 1 runs per-frame during video processing. Layers 2-5 run as post-processing.

---

## Layer 1: Ball Detection (TrackNet + YOLO)

### TrackNet Model

Standard TrackNet v2 encoder-decoder (U-Net style), ~2.5M parameters.

**Input:** 3 background-subtracted frames, every-other-frame spacing (frames N-4, N-2, N). This gives a 133ms temporal baseline at 30fps — enough for a ball at 100kph to travel ~3.7m, creating a clear motion signal.

**Background subtraction:** Computed once at video start. Take the median of ~30 frames spread across the first 10 seconds. The median removes moving objects (players, ball), leaving only the static court surface and surroundings. Per-frame diff: `abs(frame - reference)`. Cost: one subtraction per frame (essentially free on GPU).

**Why background subtraction matters:** Removes white objects that look like balls (line markings, scoreboard text, umbrellas). Makes the ball's motion streak stand out against a near-black background. Dramatically reduces false positives on courts with white lines.

**Output:** 640x360x1 probability heatmap. Each pixel is 0.0-1.0 representing probability the ball is at that location.

**Frame pipeline:**
```
Video frames:  1  2  3  4  5  6  7  8  9  10  11  12 ...
               |     |     |     |     |      |
TrackNet in:  [1,3,5]  [3,5,7]  [5,7,9]  [7,9,11] ...
TrackNet out:     v       v        v         v
Ball pos:         5       7        9         11
                  (15 predictions per second)
```

**Peak detection from heatmap:**
1. Threshold heatmap at 0.5
2. Find brightest connected region
3. Weighted centroid of that region (sub-pixel accuracy)
4. If peak < 0.5: no ball detected this frame
5. Output: `(x, y, confidence)` in pixel coords

### YOLO Ball Detection

Existing yolov8s detector with `ball_conf=0.2`. Catches the ball in ~1-5% of frames, primarily when it's slow (bouncing, at arc peak, just after contact).

### Merging Detections

For each frame, combine TrackNet and YOLO ball positions:
- If both detect a ball within 20px of each other: use TrackNet position (sub-pixel), boost confidence
- If only TrackNet: use TrackNet position
- If only YOLO: use YOLO position
- If neither: no ball this frame

### Pre-trained Weights

Start with publicly available TrackNet v2 weights (trained on broadcast tennis). Fine-tune on our data if accuracy is insufficient.

### YOLO Self-Training Pipeline (future)

TrackNet's dense detections become pseudo-labels for YOLO fine-tuning:
1. Run TrackNet on videos -> ball positions in ~85% of frames
2. Convert to YOLO bounding box format
3. Fine-tune yolov8s on pseudo-labels
4. Result: YOLO improves at ball detection over time

This is a separate future step, not part of initial build.

### Expected Detection Rates

| Ball speed | TrackNet rate | YOLO rate | Combined |
|-----------|--------------|-----------|----------|
| Slow (bounce, arc peak) | ~95% | ~10% | ~95% |
| Medium (rally) | ~80% | ~3% | ~80% |
| Fast (serve, smash) | ~50-60% | ~1% | ~55% |
| Very fast (ace) | ~30-40% | ~0% | ~35% |

Even at worst case (35%), we get 5+ detections per second — enough for trajectory fitting.

---

## Layer 2: Trajectory Fitting

### Coordinate System

Ball detections arrive in pixel coordinates. The court homography (computed once from first frame) maps these to court-space:

- `cx, cy`: horizontal position on court in meters (0-23.77m length, 0-10.97m width)
- `h`: height above ground, estimated from the difference between the ball's pixel-y and the court surface's expected pixel-y at that court position

### Height Estimation

From a fixed behind-baseline camera, vertical position in the frame correlates with height off the ground. When the ball is on the court surface, it sits at the expected y-position per the homography. When airborne, it appears higher.

```
court_y = homography maps (cx, cy, h=0) back to pixel space -> expected pixel y
ball_y  = actual ball pixel y
height  = f(court_y - ball_y, camera_geometry)
```

### Trajectory Model

General model that handles the full spectrum from linear (fast) to parabolic (slow):

```
cx(t) = cx0 + vx*t                     (horizontal velocity)
cy(t) = cy0 + vy*t                     (lateral velocity)
h(t)  = h0 + vz*t - 0.5*g_eff*t^2     (gravity + drag)
```

`g_eff` (effective gravity) naturally adapts to ball speed:
- Lob: g_eff ~ 9.8 (real gravity dominates)
- Rally groundstroke: g_eff ~ 6-8 (spin + drag)
- Fast serve: g_eff ~ 3-5 (near-linear, minimal gravity effect)
- Flat bullet: g_eff ~ 1-2 (essentially linear trajectory)

The fitter solves for 7 unknowns (cx0, vx, cy0, vy, h0, vz, g_eff) using least-squares regression on N detected ball positions. With 3+ detections, the system is over-determined.

### Bounce Detection

A bounce occurs when h(t) = 0 (ball hits the ground):

```
t_bounce = (vz + sqrt(vz^2 + 2*g_eff*h0)) / g_eff
```

At t_bounce, the court position (cx(t_bounce), cy(t_bounce)) is the landing spot.

### In/Out Determination

Tennis court boundaries (meters):
- Full court: 23.77m x 10.97m (doubles) / 8.23m (singles)
- Service box: 6.40m from net, 4.115m from center line
- Net is at 11.885m from each baseline

Landing point classification:
- Inside singles/doubles lines and between baselines -> IN
- Inside service box (for serves) -> SERVICE IN
- Outside any boundary -> OUT
- Within 5cm of a line -> CLOSE CALL (flagged for review)

### Trajectory Confidence

Each trajectory gets a confidence score based on:
- Number of input detections (3 = minimum, 10+ = high)
- Residual error of the least-squares fit
- Mean detection confidence from TrackNet/YOLO
- Consistency with physics (g_eff between 0 and 15)

Confidence levels:
- High (>0.8): clear trajectory, reliable in/out call
- Medium (0.5-0.8): reasonable fit, apply but flag for QC
- Low (<0.5): too few detections or poor fit, skip auto-scoring

### Known Limitations

- **Net hits:** Cannot detect if ball clips the net. Would need net position tracking or audio analysis.
- **Spin effects:** Topspin dip and slice float are partially absorbed by g_eff but not perfectly modeled.
- **< 3 detections:** Cannot fit a parabola. Fall back to linear interpolation with assumed gravity.

---

## Layer 3: Shot Segmentation

### Shot Boundaries

A shot is one player hitting the ball to the other. Boundaries are detected by:

1. **Direction change:** Ball reverses its court-length direction (cy derivative flips sign). This means a player made contact. The ball's trajectory alone tells us — no racquet detection needed.

2. **Bounce detection:** h(t) crosses zero. After a bounce, either the opponent hits (new shot) or the ball bounces again / goes out (point over).

### Player Assignment

For fixed behind-baseline camera:
- Near player (bottom of frame): shots travel away from camera (cy increasing)
- Far player (top of frame): shots travel toward camera (cy decreasing)

When the ball changes direction, the player on that side of the net is the hitter. Player positions from the 89% two-player detection confirm this.

### Serve Detection

The first shot of each point. Identified by:
- Ball originates from behind the baseline on the server's side
- High initial height (ball toss) followed by fast trajectory
- Server alternates deuce/ad side each point (tracked by score state)

### Shot Output

```
Shot {
    index:       int          // shot number within the rally (1 = serve)
    hitter:      int          // 0 = near player, 1 = far player
    trajectory:  Trajectory   // fitted arc
    bounce:      CourtPoint   // where it landed (cx, cy in meters)
    in_out:      string       // "in", "out", "close_call"
    speed_kph:   float        // estimated from trajectory
    confidence:  float        // trajectory fit confidence
}
```

---

## Layer 4: Point Recognition

### Point End Conditions

A point ends when one of these is detected:

| # | Condition | Detection method | Point to | Category |
|---|-----------|-----------------|----------|----------|
| 1 | Ball bounces out | Bounce point outside court lines | Opponent of last hitter | Unforced error |
| 2 | Double bounce | Two bounces on same side of net without direction change | Last hitter | Winner |
| 3 | Ace | Serve bounces in service box, no return | Server | Ace |
| 4 | Double fault | Two consecutive serves outside service box | Returner | Double fault |
| 5 | Unreturned winner | Ball bounces in, trajectory continues past baseline, no direction change | Last hitter | Winner |
| 6 | Ball into net | Trajectory ends before cy reaches net position | Opponent of last hitter | Error |

### Point Output

```
Point {
    number:       int          // point number in the match
    shots:        []Shot       // all shots in this point
    rally_length: int          // number of shots
    winner:       int          // 0 = near player, 1 = far player
    outcome:      string       // "winner", "unforced_error", "ace", "double_fault", "error"
    server:       int          // who served
    serve_side:   string       // "deuce" or "ad"
    confidence:   float        // min confidence across all shots
}
```

### Point Confidence

- High (>0.8): clear shots, reliable in/out calls throughout
- Medium (0.5-0.8): some uncertain shots but outcome is likely correct
- Low (<0.5): too many gaps, flag for human review

---

## Layer 5: Score Tracking

### Tennis Scoring State Machine

```
MatchState {
    server:       int        // 0 or 1
    serve_side:   string     // "deuce" or "ad"
    point_score:  [2]int     // point values (0, 15, 30, 40; 41 = advantage)
    game_score:   [2]int     // games in current set
    set_score:    [2]int     // sets won
    sets:         []SetResult
    is_tiebreak:  bool
    point_number: int
}
```

### State Transitions

On each point outcome:
1. Award point to winner
2. Check game won: 40 vs <40, advantage resolution, tiebreak rules
3. Check set won: first to 6 with 2+ lead, or tiebreak at 6-6
4. Check match won: best of 3 sets
5. Update server: alternates every game; tiebreak every 2 points after first
6. Update serve side: alternates every point (deuce/ad)

### Server Detection (No Scoreboard)

1. First point: player behind baseline on deuce side whose position correlates with ball origin = server
2. Subsequent games: server alternates automatically via state machine
3. Validation: if a "serve" originates from the wrong player, flag as tracking error

### OCR Cross-Check (When Available)

When scoreboard is visible, OCR reads the score and compares against tracked score. Mismatches flag potential errors for review. This is a validation layer, not a dependency.

### Error Resilience

- High confidence point: apply normally
- Medium confidence: apply but flag for QC
- Low confidence: insert "unknown point" placeholder
- Contradictory data: hold score, flag for human review

The QC loop corrects individual point outcomes; score recalculates from the corrected sequence.

---

## File Structure

### New Python Files

```
ml/tracknet.py          # TrackNet v2 model, inference wrapper, background subtraction
ml/trajectory.py        # TrajectoryFitter, bounce detection, in/out classification
```

### New Go Files

```
internal/point/shot.go       # Shot struct and segmentation logic
internal/point/point.go      # Point recognition state machine
internal/point/score.go      # MatchState and tennis scoring rules
internal/point/rules.go      # Court boundaries, service boxes, in/out geometry
internal/point/shot_test.go  # Shot segmentation tests
internal/point/point_test.go # Point recognition tests
internal/point/score_test.go # Scoring edge cases (deuce, tiebreak, set transitions)
internal/point/rules_test.go # Boundary geometry tests
```

### Modified Files

```
ml/bridge_server.py             # New RPCs: tracknet_detect_batch, fit_trajectories
internal/bridge/types.go        # New types: TrajectoryResult, ShotResult, PointResult, MatchScore
internal/bridge/process.go      # New bridge calls for trajectory + point recognition
internal/pipeline/concurrent.go # Post-processing stages for trajectory -> shot -> point -> score
internal/app/app.go             # Updated CSV export with shot/point/score columns
```

---

## Implementation Order

1. **TrackNet model** — get dense ball detection working first (biggest accuracy gain)
2. **Background subtraction** — improve TrackNet accuracy on real footage
3. **Trajectory fitting** — physics layer, bounce detection, in/out calls
4. **Shot segmentation** — break trajectories into individual shots
5. **Point recognition** — determine point outcomes from shot sequences
6. **Score tracking** — tennis scoring state machine
7. **CSV export** — full tagged output in Dartfish format
8. **QC integration** — flag low-confidence points for human review
9. **YOLO self-training** — use TrackNet labels to improve YOLO (future)
