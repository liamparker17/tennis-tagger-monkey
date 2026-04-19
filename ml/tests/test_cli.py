from pathlib import Path
import ml.dartfish_to_clips.__main__ as cli


def _make_match(root: Path, name: str, with_mp4: bool = True, with_csv: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if with_mp4:
        (d / "match.mp4").write_bytes(b"")
    if with_csv:
        header = ",".join(f"c{i}" for i in range(120)) + "\n"
        row = [""] * 120
        row[2] = "1000"; row[3] = "1500"; row[5] = "P1"; row[6] = "P2"
        row[13] = "Serve"; row[17] = "Serve"; row[32] = "1"; row[33] = "P1"; row[34] = "Ace"
        (d / "match.csv").write_text(header + ",".join(row) + "\n", encoding="utf-8")
    return d


def test_cli_basic(tmp_path, monkeypatch, capsys):
    pairs = tmp_path / "pairs"
    _make_match(pairs, "MatchA")

    def fake_process_match(video, csv, out, match_name):
        return 3

    monkeypatch.setattr(cli, "process_match", fake_process_match)

    rc = cli.main([str(pairs), "--out", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MatchA: 3 clips" in out
    assert "TOTAL: 3" in out


def test_cli_only_filter(tmp_path, monkeypatch, capsys):
    pairs = tmp_path / "pairs"
    _make_match(pairs, "MatchA")
    _make_match(pairs, "MatchB")

    def fake_process_match(video, csv, out, match_name):
        return 5

    monkeypatch.setattr(cli, "process_match", fake_process_match)

    rc = cli.main([str(pairs), "--out", str(tmp_path / "out"), "--only", "MatchA"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MatchA: 5 clips" in out
    assert "MatchB" not in out
    assert "TOTAL: 5" in out


def test_cli_skips_missing(tmp_path, monkeypatch, capsys):
    pairs = tmp_path / "pairs"
    _make_match(pairs, "NoCsv", with_mp4=True, with_csv=False)
    _make_match(pairs, "NoMp4", with_mp4=False, with_csv=True)

    def fake_process_match(*a, **kw):
        raise AssertionError("should not be called")

    monkeypatch.setattr(cli, "process_match", fake_process_match)

    rc = cli.main([str(pairs), "--out", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "skip NoCsv: missing video or csv" in out
    assert "skip NoMp4: missing video or csv" in out
    assert "TOTAL: 0" in out
