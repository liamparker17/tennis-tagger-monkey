import json, shutil, pytest
from pathlib import Path
from ml.dartfish_to_clips.extract import process_match
import ml.dartfish_to_clips.extract as extract_mod

SRC = Path("testdata/sample_a.mp4")

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
    lines = [",".join(HEADERS), ",".join([""] * len(HEADERS))]
    for r in rows:
        lines.append(",".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_point(**extra) -> list[str]:
    base = {
        "Position": "1000", "Duration": "1500",
        "A1: Server": "P1", "B1: Returner": "P2",
        "x: Player A": "P1", "x: Player B": "P2",
        "A2: Serve Data": "1st Serve Made",
        "E1: Last Shot": "Forehand",
        "H1: Stroke Count": "1 to 4", "H2: Deuce Ad": "Deuce",
        "zz - Final Shot": "Winner", "zz - W-E": "Winner A",
        "zz - W-E Player": "P1", "F1: Point Won": "P1",
    }
    base.update(extra)
    return _row(**base)


def test_process_match_records_skips(tmp_path, monkeypatch):
    csv = tmp_path / "m.csv"
    _write(csv, [_sample_point(), _sample_point(**{"Position": "5000"})])

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
    _write(csv, [_sample_point()])

    out = tmp_path / "out"
    n = process_match(SRC, csv, out, match_name="m")
    assert n == 1
    assert (out / "m" / "p_0001.mp4").exists()
    labels = json.loads((out / "m" / "labels.json").read_text())
    assert labels["match"] == "m" and len(labels["points"]) == 1
    assert labels["points"][0]["clip"] == "p_0001.mp4"
