# Tennis Tagger MVP — Handover

## What Happened This Session

Tested the pipeline against real Wimbledon broadcast footage using a professionally tagged Dartfish CSV as ground truth. Built a validation framework and attempted to improve ball detection with a better pretrained model.

### Ground Truth Data

Match: **Marcos Giron vs Camilo Ugo-Carabelli, 2025 Wimbledon R1**
- Video: `E:\TaggaBot\Marcos Giron vs Camilo Ugo-Carabelli 2025 Wimbledon R1 (HE).mp4` (1280x720, 50fps, 61 min)
- CSV: `E:\TaggaBot\...mp4.csv` — 234 tagged points with serve speeds, stroke counts, XY court positions
- Dartclip: `E:\TaggaBot\...mp4.dartclip` — same data in Dartfish XML format

### Dartfish XY Coordinate Mapping (Derived)

The CSV's XY columns (XY Deuce, XY Ad, XY Last Shot, etc.) use a ~337x326 pixel court diagram. Mapping to court meters:

```
cx_meters = 0.04571 * x_px - 4.571
cy_meters = 0.03832 * y_px + 1.845
```

Calibrated using serve placements: T serves at x≈220, Wide at x≈125, Net faults at y≈262, service line at y≈95.

### Validation Framework

Created `scripts/validate_against_ground_truth.py` — extracts individual point clips using CSV timestamps, runs the tagger, compares against ground truth (shot count, serve speed, court positions). Usage:

```bash
python scripts/validate_against_ground_truth.py \
    --video "E:/TaggaBot/Marcos Giron...mp4" \
    --csv "E:/TaggaBot/Marcos Giron...mp4.csv" \
    --points 10 --offset 0
```

### Validation Results (5 points)

| Point | GT Strokes | Tagger | GT Speed | Tagger Speed | Notes |
|-------|-----------|--------|----------|-------------|-------|
| 0-0 Ace | 1-4 | 13 | 116 | 0 | Over-segmented |
| 15-0 Rally | 13+ | 11 | 109 | 0 | Split into 3 points |
| 30-0 | 5-8 | 8 ✓ | 105 | 0 | Split into 2 points |
| 30-15 | 5-8 | 41 | 123 | 0 | Massive over-segmentation |
| 40-15 | 1-4 | 17 | 125 | 122 ✓ | Over-segmented but speed good |

Serve speed MAE: **3.4 km/h** (when detected — from YOLO, not TrackNet).

### WASB TrackNetV2 Model (Downloaded, Does Not Work)

Downloaded `nttcom/WASB-SBDT` tennis-trained TrackNetV2 weights (MIT license). Architecture class `WASBTrackNetV2` added to `ml/tracknet.py`. Model loads and runs but **produces zero detections** on Wimbledon footage — max heatmap value is 0.05 (threshold is 0.5). The model was trained on 2017 Summer Universiade footage and doesn't generalize to grass court broadcast.

Model file: `models/tracknetv2_tennis_wasb.pt` (45.4 MB)

### Key Discovery: No Public Pretrained Model Works

Tested three approaches:
1. **BallTrackerNet (yastrebksv)**: ~15% detection rate. Trained on same Universiade dataset.
2. **WASB TrackNetV2 tennis**: 0% detection rate on Wimbledon footage. Trained on same Universiade dataset.
3. **YOLO ball detection**: Produces useful detections (accurate serve speeds). This is currently the only thing working.

**Root cause**: All public tennis TrackNet models were trained on one dataset (10 matches from 2017 Universiade). Wimbledon broadcast footage looks completely different — grass court, different camera angle/height, different graphics/overlays, 50fps vs 30fps.

### Code Changes (Uncommitted)

| File | Changes |
|------|---------|
| `ml/tracknet.py` | Added `WASBTrackNetV2` architecture class, auto-detect model type from weights, resize frames to 512x288 for WASB, scale coords back |
| `ml/bridge_server.py` | Prefer WASB weights, skip background subtraction for WASB, make stroke classifier optional (graceful fallback when weights missing), homography rescaling to 640x360 detection space, broadcast fallback improvements |
| `ml/trajectory.py` | Time-based segmentation params (FPS-aware), `_sec_to_frames()` helper |
| `ml/tests/test_trajectory.py` | Updated imports for time-based constants |
| `scripts/validate_against_ground_truth.py` | New: ground truth validation script |
| `validation_results.json` | Baseline results (BallTrackerNet) |
| `validation_results_wasb.json` | Results with WASB model |

All tests pass: 41 Python, 11 Go packages.

## What To Do Next

### Priority 1: Fine-tune TrackNet on real broadcast data

The user has **several hundred GB of tagged footage**. The path:

1. **Use YOLO ball detections as seed labels** — YOLO is already producing useful ball positions. Run the pipeline on tagged clips, extract frames where YOLO detects the ball with high confidence.
2. **Build a training dataset** — pairs of (3 consecutive frames, ball position heatmap). Target: 10,000+ labeled frames from diverse matches.
3. **Fine-tune the WASB TrackNetV2** on this data — the architecture is ready (`WASBTrackNetV2` class), just needs training loop and data loader.
4. **Validate iteratively** using the ground truth framework against the Wimbledon CSV.

The training infrastructure partially exists (`ml/trainer.py`, 330 LOC) but hasn't been tested end-to-end and is oriented toward BallTrackerNet, not WASB.

### Priority 2: Fix over-segmentation

Even with good detection, the pipeline produces too many shots per point. Root causes:
- Ball detected during non-play periods (replays, crowd shots, between points)
- Segmentation splits continuous rallies into fragments
- The `groupShotsIntoPoints` 3-second gap may be too aggressive for broadcast footage with camera cuts

Possible fixes:
- Filter detections to the court area using player positions as anchors
- Use scene-change detection to exclude replays/cutaways
- Increase the point-grouping gap threshold for broadcast footage

### Priority 3: Fix homography

The broadcast fallback homography is still producing invalid court coordinates (cy=-43 to +331). The homography needs to:
- Be computed in 640x360 detection space (fixed but needs verification)
- Account for ball-in-flight parallax (ball is 1-3m above court plane)
- Use actual court corner positions from each specific match

The Dartfish XY data could be used to calibrate: if we can identify which video frame corresponds to a tagged XY position, we can derive the ground truth homography.

## Architecture Notes

The pipeline's detection path:
```
Video → Go (640x360 frames via ffmpeg)
    → Python bridge (rpc_tracknet_batch)
        → Background subtraction (skip for WASB)
        → TrackNet (frame triplets → heatmap → peak detection)
    → Python bridge (rpc_detect_batch)
        → YOLO (ball + player bounding boxes)
    → Go: mergeBallPositions (TrackNet + YOLO)
    → Python bridge (rpc_fit_trajectories)
        → segment_detections → TrajectoryFitter.fit per segment
    → Go: SegmentShots → groupShotsIntoPoints → RecognizePoint
    → CSV export
```

Currently, TrackNet contributes ~0% of useful detections on broadcast footage. YOLO is carrying the detection entirely.

## How To Resume

Tell Claude: "Read HANDOVER.md. Next step is building the TrackNet fine-tuning pipeline using YOLO detections as seed labels. The user has hundreds of GB of tagged footage at E:\TaggaBot\."
