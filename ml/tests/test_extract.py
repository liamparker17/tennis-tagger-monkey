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
