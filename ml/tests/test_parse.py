from pathlib import Path
from ml.dartfish_to_clips.parse import parse_dartfish_csv, PointLabels

# Indices match the real Dartfish CSV header (see parse.py for mapping).
# Row layout recap:
#   1=Position, 2=Duration, 3=Server, 6=Returner,
#   4/10/13/16 = stroke types per shot,
#   16=Last Shot, 20=Point Won, 21=Score, 31=Stroke Count,
#   47/50 = hands, 53=Surface, 93 = W-E,
#   54-57 = contact xy ("x;y"), 99-102 = placement xy ("x;y"),
#   85 = serve bounce xy, 97 = ad bounce xy.

ROW_WIDTH = 120


def _blank_row() -> list[str]:
    return [""] * ROW_WIDTH


def _make_row(**overrides) -> list[str]:
    row = _blank_row()
    row[1] = "1000"
    row[2] = "5000"
    for k, v in overrides.items():
        row[int(k[1:])] = v
    return row


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    header = ",".join(f"c{i}" for i in range(ROW_WIDTH)) + "\n"
    body = "\n".join(",".join(r) for r in rows) + "\n"
    path.write_text(header + body, encoding="utf-8")


def test_parse_minimal(tmp_path: Path):
    csv = tmp_path / "m.csv"
    row = _blank_row()
    row[1] = "12000"  # Position (ms)
    row[2] = "8000"   # Duration (ms)
    row[3] = "P1"     # Server
    row[6] = "P2"     # Returner
    row[16] = "Forehand"  # Last Shot
    row[31] = "3"     # Stroke Count
    row[20] = "P1"    # Point Won
    row[93] = "Winner A"  # W-E
    # Per-shot strokes: indices 4, 10, 13, 16.
    row[4] = "Serve"
    row[10] = "Forehand"
    row[13] = "Backhand"
    # Contact XY (first slot, index 57 = Srv+1 Contact) as "x;y" cell.
    row[57] = "100;200"
    _write_csv(csv, [row])

    pts = parse_dartfish_csv(csv)
    assert len(pts) == 1
    p: PointLabels = pts[0]
    assert p.start_ms == 12000 and p.duration_ms == 8000
    assert p.server == "P1" and p.returner == "P2"
    assert p.last_shot_stroke == "Forehand" and p.stroke_count == 3
    assert p.point_won_by == "P1" and p.winner_or_error == "Winner A"
    assert p.stroke_types[:4] == ["Serve", "Forehand", "Backhand", "Forehand"]
    assert p.contact_xy[0] == (100.0, 200.0)


def test_skip_short_row(tmp_path: Path):
    csv = tmp_path / "m.csv"
    header = ",".join(f"c{i}" for i in range(ROW_WIDTH)) + "\n"
    short = ",".join([""] * 20) + "\n"  # only 20 cols < 36
    csv.write_text(header + short, encoding="utf-8")
    assert parse_dartfish_csv(csv) == []


def test_skip_negative_start(tmp_path: Path):
    csv = tmp_path / "m.csv"
    empty_start = _make_row()
    empty_start[1] = ""  # empty -> treated as < 0
    negative_start = _make_row()
    negative_start[1] = "-100"
    _write_csv(csv, [empty_start, negative_start])
    assert parse_dartfish_csv(csv) == []


def test_skip_nonpositive_duration(tmp_path: Path):
    csv = tmp_path / "m.csv"
    zero_dur = _make_row()
    zero_dur[2] = "0"
    _write_csv(csv, [zero_dur])
    assert parse_dartfish_csv(csv) == []


def test_multi_row(tmp_path: Path):
    csv = tmp_path / "m.csv"
    r1 = _make_row()
    r1[1] = "1000"; r1[2] = "5000"
    r2 = _make_row()
    r2[1] = "8000"; r2[2] = "4000"
    _write_csv(csv, [r1, r2])
    pts = parse_dartfish_csv(csv)
    assert len(pts) == 2
    assert pts[0].index == 0 and pts[0].start_ms == 1000
    assert pts[1].index == 1 and pts[1].start_ms == 8000


def test_blank_strokes_filtered(tmp_path: Path):
    csv = tmp_path / "m.csv"
    row = _make_row()
    # Stroke indices: 4, 10, 13, 16. Populate only 4 and 13.
    row[4] = "Serve"
    row[10] = ""
    row[13] = "Forehand"
    row[16] = ""
    _write_csv(csv, [row])
    pts = parse_dartfish_csv(csv)
    assert len(pts) == 1
    assert pts[0].stroke_types == ["Serve", "Forehand"]
