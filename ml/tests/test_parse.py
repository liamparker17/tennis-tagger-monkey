from pathlib import Path
from ml.dartfish_to_clips.parse import parse_dartfish_csv, PointLabels

# parse.py now looks up columns by header NAME, so these tests build CSVs
# using the real Dartfish header strings rather than column indices.

HEADERS = [
    "Name", "Position", "Duration",
    "A1: Server", "A2: Serve Data", "B1: Returner", "B2: Return Data",
    "C1: Serve +1 Stroke", "D1: Return +1 Stroke",
    "E1: Last Shot", "E2: Last Shot Winner", "E3: Last Shot Error",
    "F1: Point Won", "F2: Point Score",
    "H1: Stroke Count", "H2: Deuce Ad",
    "x: Surface", "x: Player A", "x: Player B",
    "x: Player A Hand", "x: Player B Hand",
    "zz - Final Shot", "zz - W-E", "zz - W-E Player",
    "XY Deuce", "XY Ad", "XY Return", "XY Srv+1", "XY Ret+1", "XY Last Shot",
    "XY2 Srv+1 Contact", "XY2 Return Contact", "XY2 Ret+1 Contact", "XY2 Last Shot Contact",
]
IDX = {h: i for i, h in enumerate(HEADERS)}


def _row(**kv) -> list[str]:
    r = [""] * len(HEADERS)
    for k, v in kv.items():
        r[IDX[k]] = v
    return r


def _write(path: Path, rows: list[list[str]]) -> None:
    lines = [",".join(HEADERS)]
    lines.append(",".join([""] * len(HEADERS)))  # Dartfish writes a label row
    for r in rows:
        lines.append(",".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_minimal(tmp_path: Path):
    csv = tmp_path / "m.csv"
    _write(csv, [_row(
        **{"Position": "12000", "Duration": "8000",
           "A1: Server": "Alice", "B1: Returner": "Bob",
           "x: Player A": "Alice", "x: Player B": "Bob",
           "A2: Serve Data": "1st Serve Made",
           "B2: Return Data": "Backhand Return Made",
           "C1: Serve +1 Stroke": "Forehand",
           "E1: Last Shot": "Forehand",
           "E2: Last Shot Winner": "Alice",
           "H1: Stroke Count": "1 to 4",
           "H2: Deuce Ad": "Deuce",
           "zz - Final Shot": "Winner",
           "zz - W-E": "Winner A",
           "zz - W-E Player": "Alice",
           "XY Deuce": "100;50",
           "F1: Point Won": "Alice"}
    )])
    pts = parse_dartfish_csv(csv)
    assert len(pts) == 1
    p: PointLabels = pts[0]
    assert p.start_ms == 12000 and p.duration_ms == 8000
    assert p.server == "Alice" and p.returner == "Bob"
    assert p.player_a == "Alice" and p.player_b == "Bob"
    assert p.stroke_count_lo == 1 and p.stroke_count_hi == 4
    assert p.outcome == "Winner" and p.outcome_player == "Alice"
    assert p.deuce_or_ad == "Deuce"
    assert p.serve_bounce_xy == (100.0, 50.0)
    assert p.stroke_types[:3] == ["1st Serve Made", "Backhand Return Made", "Forehand"]


def test_ad_picks_xy_ad(tmp_path: Path):
    csv = tmp_path / "m.csv"
    _write(csv, [_row(
        **{"Position": "0", "Duration": "5000",
           "H2: Deuce Ad": "Ad", "XY Ad": "200;75", "XY Deuce": "999;999",
           "A2: Serve Data": "Ace"}
    )])
    pts = parse_dartfish_csv(csv)
    assert len(pts) == 1
    assert pts[0].serve_bounce_xy == (200.0, 75.0)
    assert pts[0].outcome == "Ace"


def test_skip_invalid_duration(tmp_path: Path):
    csv = tmp_path / "m.csv"
    _write(csv, [
        _row(**{"Position": "1000", "Duration": "0"}),
        _row(**{"Position": "", "Duration": "5000"}),
    ])
    assert parse_dartfish_csv(csv) == []


def test_skip_save_sentinel(tmp_path: Path):
    csv = tmp_path / "m.csv"
    _write(csv, [
        _row(**{"Name": "SAVE MATCH DETAILS - DELETE",
                "Position": "1000", "Duration": "5000"}),
    ])
    assert parse_dartfish_csv(csv) == []


def test_multi_row(tmp_path: Path):
    csv = tmp_path / "m.csv"
    _write(csv, [
        _row(**{"Position": "1000", "Duration": "5000"}),
        _row(**{"Position": "8000", "Duration": "4000"}),
    ])
    pts = parse_dartfish_csv(csv)
    assert len(pts) == 2
    assert pts[0].index == 0 and pts[0].start_ms == 1000
    assert pts[1].index == 1 and pts[1].start_ms == 8000


def test_blank_strokes_filtered(tmp_path: Path):
    csv = tmp_path / "m.csv"
    _write(csv, [_row(
        **{"Position": "0", "Duration": "5000",
           "A2: Serve Data": "1st Serve Made",
           "C1: Serve +1 Stroke": "Forehand"}
    )])
    pts = parse_dartfish_csv(csv)
    assert len(pts) == 1
    # Empty return + empty ret+1 should not become entries.
    assert pts[0].stroke_types == ["1st Serve Made", "Forehand"]
