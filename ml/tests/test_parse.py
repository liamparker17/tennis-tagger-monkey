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
