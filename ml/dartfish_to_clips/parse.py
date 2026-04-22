from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Header names (not indices) — the CSV column order drifts between panel
# versions, so we always look up by header string.
H_POSITION     = "Position"
H_DURATION     = "Duration"
H_SERVER       = "A1: Server"
H_RETURNER     = "B1: Returner"
H_SERVE_DATA   = "A2: Serve Data"
H_RETURN_DATA  = "B2: Return Data"
H_SERVE1      = "C1: Serve +1 Stroke"
H_RETURN1     = "D1: Return +1 Stroke"
H_LAST_SHOT   = "E1: Last Shot"
H_LAST_WINNER = "E2: Last Shot Winner"
H_LAST_ERROR  = "E3: Last Shot Error"
H_POINT_WON   = "F1: Point Won"
H_POINT_SCORE = "F2: Point Score"
H_STROKE_COUNT = "H1: Stroke Count"
H_DEUCE_AD    = "H2: Deuce Ad"
H_SURFACE     = "x: Surface"
H_PLAYER_A    = "x: Player A"
H_PLAYER_B    = "x: Player B"
H_HAND_A      = "x: Player A Hand"
H_HAND_B      = "x: Player B Hand"
H_FINAL_SHOT_OUTCOME = "zz - Final Shot"  # Winner / Unforced Error / Forced Error
H_W_E         = "zz - W-E"                 # Winner A/B, Error A/B
H_W_E_PLAYER  = "zz - W-E Player"
H_XY_DEUCE    = "XY Deuce"    # serve bounce when serving to deuce
H_XY_AD       = "XY Ad"       # serve bounce when serving to ad
H_XY_RETURN   = "XY Return"
H_XY_SRV1     = "XY Srv+1"
H_XY_RET1     = "XY Ret+1"
H_XY_LAST     = "XY Last Shot"
H_XY2_SRV1    = "XY2 Srv+1 Contact"
H_XY2_RETURN  = "XY2 Return Contact"
H_XY2_RET1    = "XY2 Ret+1 Contact"
H_XY2_LAST    = "XY2 Last Shot Contact"


@dataclass
class PointLabels:
    index: int
    start_ms: int
    duration_ms: int
    server: str = ""
    returner: str = ""
    player_a: str = ""
    player_b: str = ""
    surface: str = ""
    speed_kmh: Optional[float] = None
    hands: tuple[str, str] = ("", "")
    # Stroke list in rally order. Each entry is a raw stroke string from Dartfish.
    # Position 0 = serve, 1 = return, 2 = serve+1, 3 = return+1, 4 = last shot (if rally > 4).
    stroke_types: list[str] = field(default_factory=list)
    last_shot_stroke: str = ""
    stroke_count_lo: int = 0   # lower bound of "1 to 4" bucket
    stroke_count_hi: int = 0   # upper bound
    stroke_count_bucket: str = ""  # raw string e.g. "1 to 4"
    point_won_by: str = ""        # name of winner of point
    winner_or_error_raw: str = ""  # raw zz-Final Shot value
    outcome: str = ""              # derived: Ace / DoubleFault / Winner / ForcedError / UnforcedError
    outcome_player: str = ""       # name of winner/error-maker
    score_state: str = ""
    deuce_or_ad: str = ""
    contact_xy: list[Optional[tuple[float, float]]] = field(default_factory=list)
    placement_xy: list[Optional[tuple[float, float]]] = field(default_factory=list)
    serve_bounce_xy: Optional[tuple[float, float]] = None


def _f(s: str) -> Optional[float]:
    s = s.strip()
    if not s: return None
    try: return float(s)
    except ValueError: return None


def _parse_stroke_count_bucket(s: str) -> tuple[int, int]:
    """Dartfish writes '1 to 4', '5 to 9', etc. Return (lo, hi)."""
    s = s.strip()
    if not s: return (0, 0)
    parts = s.replace("+", "").split("to")
    if len(parts) == 2:
        lo = _f(parts[0]); hi = _f(parts[1])
        return (int(lo) if lo else 0, int(hi) if hi else 0)
    v = _f(s)
    if v is not None:
        return (int(v), int(v))
    return (0, 0)


def _xy(s: str) -> Optional[tuple[float, float]]:
    s = s.strip()
    if not s or ";" not in s: return None
    a, b = s.split(";", 1)
    x, y = _f(a), _f(b)
    return (x, y) if x is not None and y is not None else None


def _derive_outcome(serve_data: str, final_shot: str,
                    last_winner: str, last_error: str,
                    we: str, we_player: str
                    ) -> tuple[str, str]:
    """Return (canonical_outcome, player_name)."""
    sd = serve_data.strip().lower()
    if "ace" in sd: return "Ace", we_player.strip()
    if "double fault" in sd: return "DoubleFault", we_player.strip()
    fs = final_shot.strip().lower()
    if "winner" in fs: return "Winner", last_winner.strip() or we_player.strip()
    if "forced error" in fs: return "ForcedError", last_error.strip() or we_player.strip()
    if "unforced error" in fs: return "UnforcedError", last_error.strip() or we_player.strip()
    # Fallback from zz-W-E: "Winner A", "Error B", ...
    we_s = we.strip().lower()
    if "winner" in we_s: return "Winner", we_player.strip()
    if "error" in we_s:  return "UnforcedError", we_player.strip()
    return "UnforcedError", ""


class _HeaderMap:
    def __init__(self, header: list[str]):
        self.by_name = {h: i for i, h in enumerate(header)}
    def cell(self, row: list[str], name: str) -> str:
        i = self.by_name.get(name, -1)
        if i < 0 or i >= len(row): return ""
        return row[i]


def parse_dartfish_csv(path: Path) -> list[PointLabels]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows: return []
    hm = _HeaderMap(rows[0])

    out: list[PointLabels] = []
    for i, row in enumerate(rows[2:]):  # rows[1] is a label-row
        pos_raw = hm.cell(row, H_POSITION)
        dur_raw = hm.cell(row, H_DURATION)
        pv = _f(pos_raw); dv = _f(dur_raw)
        if pv is None or dv is None or dv <= 0: continue
        start_ms = int(pv); duration_ms = int(dv)
        # skip the "SAVE MATCH DETAILS - DELETE" sentinel
        name_cell = row[0].strip() if row else ""
        if name_cell.lower().startswith("save match"): continue

        server = hm.cell(row, H_SERVER).strip()
        returner = hm.cell(row, H_RETURNER).strip()
        player_a = hm.cell(row, H_PLAYER_A).strip()
        player_b = hm.cell(row, H_PLAYER_B).strip()

        # Strokes in rally order: serve, return, srv+1, ret+1, last-shot-if-long
        stroke_cols = [H_SERVE_DATA, H_RETURN_DATA, H_SERVE1, H_RETURN1]
        stroke_types: list[str] = []
        for c in stroke_cols:
            v = hm.cell(row, c).strip()
            if v: stroke_types.append(v)
        last_shot = hm.cell(row, H_LAST_SHOT).strip()
        if last_shot:
            stroke_types.append(last_shot)

        # Stroke count bucket
        bucket = hm.cell(row, H_STROKE_COUNT).strip()
        lo, hi = _parse_stroke_count_bucket(bucket)

        # Outcome
        outcome, outcome_player = _derive_outcome(
            hm.cell(row, H_SERVE_DATA),
            hm.cell(row, H_FINAL_SHOT_OUTCOME),
            hm.cell(row, H_LAST_WINNER),
            hm.cell(row, H_LAST_ERROR),
            hm.cell(row, H_W_E),
            hm.cell(row, H_W_E_PLAYER),
        )

        # Contact + placement XY per shot (ordered like stroke_types)
        contact_xy: list[Optional[tuple[float, float]]] = [
            None,  # serve contact (not tagged in this panel)
            _xy(hm.cell(row, H_XY2_RETURN)),
            _xy(hm.cell(row, H_XY2_SRV1)),
            _xy(hm.cell(row, H_XY2_RET1)),
            _xy(hm.cell(row, H_XY2_LAST)),
        ]

        deuce_ad = hm.cell(row, H_DEUCE_AD).strip().lower()
        serve_bounce = _xy(hm.cell(row, H_XY_AD if "ad" in deuce_ad
                                    else H_XY_DEUCE))
        placement_xy: list[Optional[tuple[float, float]]] = [
            serve_bounce,
            _xy(hm.cell(row, H_XY_RETURN)),
            _xy(hm.cell(row, H_XY_SRV1)),
            _xy(hm.cell(row, H_XY_RET1)),
            _xy(hm.cell(row, H_XY_LAST)),
        ]

        out.append(PointLabels(
            index=i, start_ms=start_ms, duration_ms=duration_ms,
            server=server, returner=returner,
            player_a=player_a, player_b=player_b,
            surface=hm.cell(row, H_SURFACE).strip(),
            speed_kmh=None,
            hands=(hm.cell(row, H_HAND_A).strip(), hm.cell(row, H_HAND_B).strip()),
            stroke_types=stroke_types,
            last_shot_stroke=last_shot,
            stroke_count_lo=lo, stroke_count_hi=hi,
            stroke_count_bucket=bucket,
            point_won_by=hm.cell(row, H_POINT_WON).strip(),
            winner_or_error_raw=hm.cell(row, H_FINAL_SHOT_OUTCOME).strip(),
            outcome=outcome, outcome_player=outcome_player,
            score_state=hm.cell(row, H_POINT_SCORE).strip(),
            deuce_or_ad="Ad" if "ad" in deuce_ad else "Deuce",
            contact_xy=contact_xy,
            placement_xy=placement_xy,
            serve_bounce_xy=serve_bounce,
        ))
    return out
