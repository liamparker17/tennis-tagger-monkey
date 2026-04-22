from __future__ import annotations
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from .parse import parse_dartfish_csv
from .clip import cut_clip, window_seconds

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

def process_match(video: Path, csv: Path, out_root: Path, match_name: str) -> int:
    points = parse_dartfish_csv(csv)
    out_dir = out_root / match_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_src = video.parent / (video.name + ".setup.json")
    if sidecar_src.exists():
        try:
            shutil.copy2(sidecar_src, out_dir / "setup.json")
        except Exception as e:
            log.warning("failed to copy setup sidecar: %s", e)
    records = []
    skipped = []
    for n, p in enumerate(points, start=1):
        clip_name = f"p_{n:04d}.mp4"
        clip_path = out_dir / clip_name
        start_s, dur_s = window_seconds(p.start_ms, p.duration_ms)
        point_offset_s = (p.start_ms / 1000.0) - start_s
        point_end_in_clip_s = point_offset_s + (p.duration_ms / 1000.0)
        serve_xy = p.serve_bounce_xy
        # placement_xy slots: [serve_bounce, return, srv+1, ret+1, last_shot].
        # last_shot is only tagged for rallies > 4; otherwise take the last
        # non-None rally shot as the final bounce.
        last_shot_xy = next((xy for xy in reversed(p.placement_xy[1:]) if xy is not None), None)
        ball_anchors = []
        if serve_xy is not None:
            ball_anchors.append({
                "shot": "serve_contact", "t_clip_s": round(point_offset_s, 3),
                "xy_court": [float(serve_xy[0]), float(serve_xy[1])],
            })
        if last_shot_xy is not None:
            ball_anchors.append({
                "shot": "last_bounce", "t_clip_s": round(point_end_in_clip_s, 3),
                "xy_court": [float(last_shot_xy[0]), float(last_shot_xy[1])],
            })
        if clip_path.exists() and clip_path.stat().st_size > 0:
            rec = asdict(p)
            rec["clip"] = clip_name
            rec["clip_start_s"] = start_s
            rec["clip_duration_s"] = dur_s
            rec["ball_anchors"] = ball_anchors
            records.append(rec)
            continue
        try:
            cut_clip(video, start_s, dur_s, clip_path)
        except RuntimeError as e:
            log.warning("clip %s failed: %s", clip_name, e)
            skipped.append({"index": p.index, "clip": clip_name, "reason": str(e)})
            continue
        rec = asdict(p)
        rec["clip"] = clip_name
        rec["clip_start_s"] = start_s
        rec["clip_duration_s"] = dur_s
        rec["ball_anchors"] = ball_anchors
        records.append(rec)
    (out_dir / "labels.json").write_text(json.dumps({
        "schema": SCHEMA_VERSION, "match": match_name,
        "video": str(video), "csv": str(csv), "points": records,
        "skipped": skipped,
    }, indent=2))
    return len(records)
