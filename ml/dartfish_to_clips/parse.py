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
