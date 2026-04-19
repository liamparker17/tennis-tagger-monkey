from __future__ import annotations
import json
import logging
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
    records = []
    skipped = []
    for n, p in enumerate(points, start=1):
        clip_name = f"p_{n:04d}.mp4"
        start_s, dur_s = window_seconds(p.start_ms, p.duration_ms)
        try:
            cut_clip(video, start_s, dur_s, out_dir / clip_name)
        except RuntimeError as e:
            log.warning("clip %s failed: %s", clip_name, e)
            skipped.append({"index": p.index, "clip": clip_name, "reason": str(e)})
            continue
        rec = asdict(p)
        rec["clip"] = clip_name
        rec["clip_start_s"] = start_s
        rec["clip_duration_s"] = dur_s
        records.append(rec)
    (out_dir / "labels.json").write_text(json.dumps({
        "schema": SCHEMA_VERSION, "match": match_name,
        "video": str(video), "csv": str(csv), "points": records,
        "skipped": skipped,
    }, indent=2))
    return len(records)
