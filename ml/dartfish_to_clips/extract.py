from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from .parse import parse_dartfish_csv
from .clip import cut_clip, window_seconds

SCHEMA_VERSION = 1

def process_match(video: Path, csv: Path, out_root: Path, match_name: str) -> int:
    points = parse_dartfish_csv(csv)
    out_dir = out_root / match_name
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for n, p in enumerate(points, start=1):
        clip_name = f"p_{n:04d}.mp4"
        start_s, dur_s = window_seconds(p.start_ms, p.duration_ms)
        try:
            cut_clip(video, start_s, dur_s, out_dir / clip_name)
        except RuntimeError:
            continue
        rec = asdict(p)
        rec["clip"] = clip_name
        rec["clip_start_s"] = start_s
        rec["clip_duration_s"] = dur_s
        records.append(rec)
    (out_dir / "labels.json").write_text(json.dumps({
        "schema": SCHEMA_VERSION, "match": match_name,
        "video": str(video), "csv": str(csv), "points": records,
    }, indent=2))
    return len(records)
