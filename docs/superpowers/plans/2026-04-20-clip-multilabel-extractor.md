# Clip + Multi-Label Extractor — Plan 1

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Walk every `<video>.mp4` + `<video>.csv` pair under `files/data/training_pairs/`, cut one short clip per Dartfish point, and emit a `labels.json` carrying every supervision column the multi-task model will use.

**Architecture:** Pure-Python module `ml/dartfish_to_clips/`. Parser → clipper → orchestrator → CLI. ffmpeg subprocess for cutting. No ML, no GPU.

**Tech stack:** Python 3.11, ffmpeg on PATH, pytest.

---

## File layout

```
ml/dartfish_to_clips/
  __init__.py
  parse.py        # CSV → PointLabels list
  clip.py         # ffmpeg slow-seek clip cutter
  extract.py      # match-level orchestrator
  __main__.py     # CLI: python -m ml.dartfish_to_clips files/data/training_pairs
ml/tests/
  test_parse.py
  test_clip.py
  test_extract.py
```

Output:
```
files/data/clips/<match>/p_0001.mp4
files/data/clips/<match>/labels.json
```

---

## Dartfish column map

```python
COL_POSITION_MS       = 2     # point start (ms)
COL_DURATION_MS       = 3     # point duration (ms)
COL_SERVER            = 5
COL_RETURNER          = 6
COL_SURFACE           = 7
COL_SPEED_KMH         = 8
COL_HANDS_P1          = 9
COL_HANDS_P2          = 10
COL_STROKE_BASE       = 13    # 4 cols (per-shot stroke types)
COL_LAST_SHOT_STROKE  = 17
COL_STROKE_COUNT      = 32
COL_POINT_WON_BY      = 33
COL_WINNER_OR_ERROR   = 34
COL_SCORE_STATE       = 35
COL_CONTACT_XY_BASE   = 55    # 4 (x,y) pairs → cols 55..62
COL_PLACEMENT_XY_BASE = 63    # 4 (x,y) pairs → cols 63..70
COL_XY_SERVE_BOUNCE   = 97
COL_XY_AD_BOUNCE      = 99
```

If an index turns out wrong on real data, fix in `parse.py` and update Task 1's fixture.

---

### Task 1 — `PointLabels` parser

**Files:** create `ml/dartfish_to_clips/parse.py`, `ml/tests/test_parse.py`.

- [ ] Failing test:

```python
# ml/tests/test_parse.py
from pathlib import Path
from ml.dartfish_to_clips.parse import parse_dartfish_csv, PointLabels

def test_parse_minimal(tmp_path: Path):
    csv = tmp_path / "m.csv"
    header = ",".join(f"c{i}" for i in range(120)) + "\n"
    row = [""] * 120
    row[2] = "12000"; row[3] = "8000"; row[5] = "P1"; row[6] = "P2"
    row[17] = "Forehand"; row[32] = "3"; row[33] = "P1"; row[34] = "Winner"
    row[13] = "Serve"; row[14] = "Forehand"; row[15] = "Forehand"
    row[55] = "100"; row[56] = "200"
    csv.write_text(header + ",".join(row) + "\n", encoding="utf-8")

    pts = parse_dartfish_csv(csv)
    assert len(pts) == 1
    p: PointLabels = pts[0]
    assert p.start_ms == 12000 and p.duration_ms == 8000
    assert p.server == "P1" and p.returner == "P2"
    assert p.last_shot_stroke == "Forehand" and p.stroke_count == 3
    assert p.point_won_by == "P1" and p.winner_or_error == "Winner"
    assert p.stroke_types[:3] == ["Serve", "Forehand", "Forehand"]
    assert p.contact_xy[0] == (100.0, 200.0)
```

- [ ] `pytest ml/tests/test_parse.py -v` → fails (module missing).

- [ ] Implement:

```python
# ml/dartfish_to_clips/parse.py
from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

COL_POSITION_MS = 2
COL_DURATION_MS = 3
COL_SERVER = 5
COL_RETURNER = 6
COL_SURFACE = 7
COL_SPEED_KMH = 8
COL_HANDS_P1 = 9
COL_HANDS_P2 = 10
COL_STROKE_BASE = 13
COL_LAST_SHOT_STROKE = 17
COL_STROKE_COUNT = 32
COL_POINT_WON_BY = 33
COL_WINNER_OR_ERROR = 34
COL_SCORE_STATE = 35
COL_CONTACT_XY_BASE = 55
COL_PLACEMENT_XY_BASE = 63
COL_XY_SERVE_BOUNCE = 97
COL_XY_AD_BOUNCE = 99

@dataclass
class PointLabels:
    index: int
    start_ms: int
    duration_ms: int
    server: str = ""
    returner: str = ""
    surface: str = ""
    speed_kmh: Optional[float] = None
    hands: tuple[str, str] = ("", "")
    stroke_types: list[str] = field(default_factory=list)
    last_shot_stroke: str = ""
    stroke_count: int = 0
    point_won_by: str = ""
    winner_or_error: str = ""
    score_state: str = ""
    contact_xy: list[tuple[float, float]] = field(default_factory=list)
    placement_xy: list[tuple[float, float]] = field(default_factory=list)
    serve_bounce_xy: Optional[tuple[float, float]] = None
    ad_bounce_xy: Optional[tuple[float, float]] = None

def _f(s: str) -> Optional[float]:
    s = s.strip()
    if not s: return None
    try: return float(s)
    except ValueError: return None

def _i(s: str, default: int = 0) -> int:
    v = _f(s); return int(v) if v is not None else default

def _xy(row: list[str], base: int) -> Optional[tuple[float, float]]:
    if base + 1 >= len(row): return None
    x, y = _f(row[base]), _f(row[base + 1])
    return (x, y) if x is not None and y is not None else None

def parse_dartfish_csv(path: Path) -> list[PointLabels]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    out: list[PointLabels] = []
    for i, row in enumerate(rows[1:]):
        if len(row) < 36: continue
        start = _i(row[COL_POSITION_MS], -1); dur = _i(row[COL_DURATION_MS])
        if start < 0 or dur <= 0: continue
        strokes = [row[COL_STROKE_BASE + k].strip() for k in range(4) if COL_STROKE_BASE + k < len(row)]
        strokes = [s for s in strokes if s]
        contacts = [xy for xy in (_xy(row, COL_CONTACT_XY_BASE + 2*k) for k in range(4)) if xy]
        placements = [xy for xy in (_xy(row, COL_PLACEMENT_XY_BASE + 2*k) for k in range(4)) if xy]
        out.append(PointLabels(
            index=i, start_ms=start, duration_ms=dur,
            server=row[COL_SERVER].strip(), returner=row[COL_RETURNER].strip(),
            surface=row[COL_SURFACE].strip(), speed_kmh=_f(row[COL_SPEED_KMH]),
            hands=(row[COL_HANDS_P1].strip(), row[COL_HANDS_P2].strip()),
            stroke_types=strokes, last_shot_stroke=row[COL_LAST_SHOT_STROKE].strip(),
            stroke_count=_i(row[COL_STROKE_COUNT]),
            point_won_by=row[COL_POINT_WON_BY].strip(),
            winner_or_error=row[COL_WINNER_OR_ERROR].strip(),
            score_state=row[COL_SCORE_STATE].strip() if COL_SCORE_STATE < len(row) else "",
            contact_xy=contacts, placement_xy=placements,
            serve_bounce_xy=_xy(row, COL_XY_SERVE_BOUNCE),
            ad_bounce_xy=_xy(row, COL_XY_AD_BOUNCE),
        ))
    return out
```

- [ ] Test passes. Commit: `feat(plan1): dartfish CSV → PointLabels parser`.

---

### Task 2 — ffmpeg clip cutter

**Files:** create `ml/dartfish_to_clips/clip.py`, `ml/tests/test_clip.py`.

- [ ] Test:

```python
# ml/tests/test_clip.py
import shutil, subprocess, pytest
from pathlib import Path
from ml.dartfish_to_clips.clip import cut_clip

SAMPLE = Path("testdata/sample_a.mp4")

@pytest.mark.skipif(not SAMPLE.exists() or shutil.which("ffmpeg") is None, reason="needs sample + ffmpeg")
def test_cut_clip(tmp_path):
    out = tmp_path / "c.mp4"
    cut_clip(SAMPLE, start_s=1.0, duration_s=2.0, out=out)
    assert out.exists() and out.stat().st_size > 0
    p = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=nw=1:nk=1", str(out)], capture_output=True, text=True)
    assert 1.5 < float(p.stdout.strip()) < 2.5
```

- [ ] Implement:

```python
# ml/dartfish_to_clips/clip.py
from __future__ import annotations
import subprocess
from pathlib import Path

PAD_BEFORE_S = 1.0
PAD_AFTER_S  = 1.0

def cut_clip(src: Path, start_s: float, duration_s: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ss", f"{max(0.0, start_s):.3f}",
        "-t",  f"{duration_s:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "96k",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr}")

def window_seconds(start_ms: int, duration_ms: int) -> tuple[float, float]:
    s = max(0.0, start_ms/1000.0 - PAD_BEFORE_S)
    d = duration_ms/1000.0 + PAD_BEFORE_S + PAD_AFTER_S
    return s, d
```

- [ ] Test passes (or skips). Commit: `feat(plan1): ffmpeg clip cutter`.

---

### Task 3 — Match orchestrator

**Files:** create `ml/dartfish_to_clips/extract.py`, `ml/tests/test_extract.py`.

- [ ] Test:

```python
# ml/tests/test_extract.py
import json, shutil, pytest
from pathlib import Path
from ml.dartfish_to_clips.extract import process_match

SRC = Path("testdata/sample_a.mp4")

@pytest.mark.skipif(not SRC.exists() or shutil.which("ffmpeg") is None, reason="needs sample + ffmpeg")
def test_process_match(tmp_path):
    csv = tmp_path / "m.csv"
    header = ",".join(f"c{i}" for i in range(120)) + "\n"
    row = [""] * 120
    row[2] = "1000"; row[3] = "1500"; row[5] = "P1"; row[6] = "P2"
    row[13] = "Serve"; row[17] = "Serve"; row[32] = "1"; row[33] = "P1"; row[34] = "Ace"
    csv.write_text(header + ",".join(row) + "\n", encoding="utf-8")

    out = tmp_path / "out"
    n = process_match(SRC, csv, out, match_name="m")
    assert n == 1
    assert (out / "m" / "p_0001.mp4").exists()
    labels = json.loads((out / "m" / "labels.json").read_text())
    assert labels["match"] == "m" and len(labels["points"]) == 1
    assert labels["points"][0]["clip"] == "p_0001.mp4"
```

- [ ] Implement:

```python
# ml/dartfish_to_clips/extract.py
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
```

- [ ] Test passes. Commit: `feat(plan1): match orchestrator emits clips + labels.json`.

---

### Task 4 — CLI

**Files:** create `ml/dartfish_to_clips/__main__.py`, `ml/dartfish_to_clips/__init__.py` (empty).

- [ ] Implement:

```python
# ml/dartfish_to_clips/__main__.py
import argparse, sys
from pathlib import Path
from .extract import process_match

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("dartfish_to_clips")
    ap.add_argument("pairs_root", type=Path, help="files/data/training_pairs")
    ap.add_argument("--out", type=Path, default=Path("files/data/clips"))
    ap.add_argument("--only", default=None, help="substring filter on match folder name")
    args = ap.parse_args(argv)

    matches = [d for d in sorted(args.pairs_root.iterdir()) if d.is_dir()]
    if args.only:
        matches = [d for d in matches if args.only in d.name]
    total = 0
    for d in matches:
        vids = list(d.glob("*.mp4")); csvs = list(d.glob("*.csv"))
        if not vids or not csvs:
            print(f"skip {d.name}: missing video or csv"); continue
        n = process_match(vids[0], csvs[0], args.out, match_name=d.name)
        print(f"{d.name}: {n} clips"); total += n
    print(f"TOTAL: {total}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] Smoke run:

```bash
python -m ml.dartfish_to_clips files/data/training_pairs --only "Marcos Giron"
```

Expected: clips appear under `files/data/clips/<match>/`, `labels.json` parses as JSON.

- [ ] Commit: `feat(plan1): CLI for clip extraction`.

---

### Task 5 — Sanity check

- [ ] Open one extracted clip in any player; confirm it spans the point with ~1s pad on each side.
- [ ] `python -c "import json; d=json.load(open('files/data/clips/<match>/labels.json')); print(d['points'][0])"` — confirm stroke_types, contact_xy, winner_or_error are populated where the source CSV had them.
- [ ] If any module imports the deleted `ml/classifier.py` or `ml/tracknet.py`, remove the dangling import. Commit: `chore(plan1): drop dangling refs to removed ML modules`.

---

## Done when

- `python -m ml.dartfish_to_clips files/data/training_pairs` runs over all matches without crashing.
- Each match folder under `files/data/clips/` has N `p_*.mp4` and one `labels.json` with `points` entries carrying start/duration, server/returner, stroke_types, contact_xy, winner_or_error.
- `pytest ml/tests/test_parse.py ml/tests/test_clip.py ml/tests/test_extract.py -v` is green.
