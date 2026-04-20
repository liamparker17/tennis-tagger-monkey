import json, shutil, pytest
from pathlib import Path
from ml.dartfish_to_clips.extract import process_match
import ml.dartfish_to_clips.extract as extract_mod

SRC = Path("testdata/sample_a.mp4")


def _make_csv_with_two_rows(csv_path: Path) -> None:
    header = ",".join(f"c{i}" for i in range(120)) + "\n"
    rows = []
    for _ in range(2):
        row = [""] * 120
        # Real-layout indices: 1=Position, 2=Duration, 3=Server, 6=Returner,
        # 4=serve stroke, 16=Last Shot, 20=Point Won, 31=Stroke Count, 93=W-E.
        row[1] = "1000"; row[2] = "1500"; row[3] = "P1"; row[6] = "P2"
        row[4] = "Serve"; row[16] = "Serve"; row[31] = "1"; row[20] = "P1"; row[93] = "Winner A"
        rows.append(",".join(row))
    csv_path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_process_match_records_skips(tmp_path, monkeypatch):
    csv = tmp_path / "m.csv"
    _make_csv_with_two_rows(csv)

    calls = {"n": 0}

    def fake_cut_clip(video, start_s, dur_s, out_path):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        Path(out_path).write_bytes(b"")

    monkeypatch.setattr(extract_mod, "cut_clip", fake_cut_clip)

    out = tmp_path / "out"
    n = process_match(Path("dummy.mp4"), csv, out, match_name="m")
    assert n == 1
    labels = json.loads((out / "m" / "labels.json").read_text())
    assert len(labels["points"]) == 1
    assert len(labels["skipped"]) == 1
    assert labels["skipped"][0]["index"] == 1
    assert labels["skipped"][0]["clip"] == "p_0002.mp4"
    assert labels["skipped"][0]["reason"] == "boom"

@pytest.mark.skipif(not SRC.exists() or shutil.which("ffmpeg") is None, reason="needs sample + ffmpeg")
def test_process_match(tmp_path):
    csv = tmp_path / "m.csv"
    header = ",".join(f"c{i}" for i in range(120)) + "\n"
    row = [""] * 120
    row[1] = "1000"; row[2] = "1500"; row[3] = "P1"; row[6] = "P2"
    row[4] = "Serve"; row[16] = "Serve"; row[31] = "1"; row[20] = "P1"; row[93] = "Winner A"
    csv.write_text(header + ",".join(row) + "\n", encoding="utf-8")

    out = tmp_path / "out"
    n = process_match(SRC, csv, out, match_name="m")
    assert n == 1
    assert (out / "m" / "p_0001.mp4").exists()
    labels = json.loads((out / "m" / "labels.json").read_text())
    assert labels["match"] == "m" and len(labels["points"]) == 1
    assert labels["points"][0]["clip"] == "p_0001.mp4"
