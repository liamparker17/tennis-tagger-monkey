from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Column indices, remapped against the real Dartfish CSV header
# (see files/data/training_pairs/Andrew Johnson vs Alberto Pulido Moreno_*/*.csv).
COL_POSITION_MS = 1
COL_DURATION_MS = 2
COL_SERVER = 3
COL_RETURNER = 6
COL_SURFACE = 53
COL_SPEED_KMH = -1  # not present in this CSV layout
COL_HANDS_P1 = 47
COL_HANDS_P2 = 50
# Per-shot stroke labels are non-contiguous in the real layout:
# 4=A2 Serve Data, 10=C1 Serve+1 Stroke, 13=D1 Return+1 Stroke, 16=E1 Last Shot.
STROKE_INDICES = (4, 10, 13, 16)
COL_LAST_SHOT_STROKE = 16
COL_STROKE_COUNT = 31
COL_POINT_WON_BY = 20
COL_WINNER_OR_ERROR = 93  # "zz - W-E" carries outcome label
COL_SCORE_STATE = 21
# Per-shot contact XY stored as single "x;y" cells (not x/y in consecutive cols).
CONTACT_XY_INDICES = (57, 56, 55, 54)  # Srv+1, Return, Ret+1, Last Shot Contact
PLACEMENT_XY_INDICES = (102, 101, 100, 99)  # Srv+1, Return, Ret+1, Last Shot placement
COL_XY_SERVE_BOUNCE = 85  # XY Deuce (serve-side bounce)
COL_XY_AD_BOUNCE = 97  # XY Ad

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

def _cell(row: list[str], idx: int) -> str:
    if idx < 0 or idx >= len(row): return ""
    return row[idx]

def _xy_cell(row: list[str], idx: int) -> Optional[tuple[float, float]]:
    """Parse an 'x;y' formatted cell into a (float, float) pair."""
    s = _cell(row, idx).strip()
    if not s or ";" not in s: return None
    parts = s.split(";")
    if len(parts) < 2: return None
    x, y = _f(parts[0]), _f(parts[1])
    return (x, y) if x is not None and y is not None else None

def parse_dartfish_csv(path: Path) -> list[PointLabels]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    out: list[PointLabels] = []
    for i, row in enumerate(rows[1:]):
        if len(row) < 36: continue
        start = _i(_cell(row, COL_POSITION_MS), -1)
        dur = _i(_cell(row, COL_DURATION_MS))
        if start < 0 or dur <= 0: continue
        strokes = [_cell(row, idx).strip() for idx in STROKE_INDICES]
        strokes = [s for s in strokes if s]
        contacts = [xy for xy in (_xy_cell(row, idx) for idx in CONTACT_XY_INDICES) if xy]
        placements = [xy for xy in (_xy_cell(row, idx) for idx in PLACEMENT_XY_INDICES) if xy]
        speed = _f(_cell(row, COL_SPEED_KMH)) if COL_SPEED_KMH >= 0 else None
        out.append(PointLabels(
            index=i, start_ms=start, duration_ms=dur,
            server=_cell(row, COL_SERVER).strip(),
            returner=_cell(row, COL_RETURNER).strip(),
            surface=_cell(row, COL_SURFACE).strip(),
            speed_kmh=speed,
            hands=(_cell(row, COL_HANDS_P1).strip(), _cell(row, COL_HANDS_P2).strip()),
            stroke_types=strokes,
            last_shot_stroke=_cell(row, COL_LAST_SHOT_STROKE).strip(),
            stroke_count=_i(_cell(row, COL_STROKE_COUNT)),
            point_won_by=_cell(row, COL_POINT_WON_BY).strip(),
            winner_or_error=_cell(row, COL_WINNER_OR_ERROR).strip(),
            score_state=_cell(row, COL_SCORE_STATE).strip(),
            contact_xy=contacts, placement_xy=placements,
            serve_bounce_xy=_xy_cell(row, COL_XY_SERVE_BOUNCE),
            ad_bounce_xy=_xy_cell(row, COL_XY_AD_BOUNCE),
        ))
    return out
