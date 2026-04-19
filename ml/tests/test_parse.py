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


def _make_row(**overrides) -> list[str]:
    row = [""] * 120
    row[2] = "1000"
    row[3] = "5000"
    for k, v in overrides.items():
        row[int(k[1:])] = v
    return row


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    header = ",".join(f"c{i}" for i in range(120)) + "\n"
    body = "\n".join(",".join(r) for r in rows) + "\n"
    path.write_text(header + body, encoding="utf-8")


def test_skip_short_row(tmp_path: Path):
    csv = tmp_path / "m.csv"
    header = ",".join(f"c{i}" for i in range(120)) + "\n"
    short = ",".join([""] * 20) + "\n"  # only 20 cols < 36
    csv.write_text(header + short, encoding="utf-8")
    assert parse_dartfish_csv(csv) == []


def test_skip_negative_start(tmp_path: Path):
    csv = tmp_path / "m.csv"
    empty_start = _make_row()
    empty_start[2] = ""  # empty -> treated as < 0
    negative_start = _make_row()
    negative_start[2] = "-100"
    _write_csv(csv, [empty_start, negative_start])
    assert parse_dartfish_csv(csv) == []


def test_skip_nonpositive_duration(tmp_path: Path):
    csv = tmp_path / "m.csv"
    zero_dur = _make_row()
    zero_dur[3] = "0"
    _write_csv(csv, [zero_dur])
    assert parse_dartfish_csv(csv) == []


def test_multi_row(tmp_path: Path):
    csv = tmp_path / "m.csv"
    r1 = _make_row()
    r1[2] = "1000"; r1[3] = "5000"
    r2 = _make_row()
    r2[2] = "8000"; r2[3] = "4000"
    _write_csv(csv, [r1, r2])
    pts = parse_dartfish_csv(csv)
    assert len(pts) == 2
    assert pts[0].index == 0 and pts[0].start_ms == 1000
    assert pts[1].index == 1 and pts[1].start_ms == 8000


def test_blank_strokes_filtered(tmp_path: Path):
    csv = tmp_path / "m.csv"
    row = _make_row()
    row[13] = "Serve"; row[14] = ""; row[15] = "Forehand"; row[16] = ""
    _write_csv(csv, [row])
    pts = parse_dartfish_csv(csv)
    assert len(pts) == 1
    assert pts[0].stroke_types == ["Serve", "Forehand"]
