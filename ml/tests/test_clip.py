import shutil
import subprocess
from pathlib import Path

import pytest

from ml.dartfish_to_clips.clip import cut_clip, window_seconds

SAMPLE = Path("testdata/sample_a.mp4")


@pytest.mark.skipif(
    not SAMPLE.exists() or shutil.which("ffmpeg") is None,
    reason="needs sample + ffmpeg",
)
def test_cut_clip(tmp_path):
    out = tmp_path / "c.mp4"
    cut_clip(SAMPLE, start_s=1.0, duration_s=2.0, out=out)
    assert out.exists() and out.stat().st_size > 0
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert 1.5 < float(p.stdout.strip()) < 2.5


def test_window_seconds():
    s, d = window_seconds(12000, 8000)
    assert s == 11.0 and d == 10.0


def test_window_seconds_clamps_negative_start():
    s, d = window_seconds(500, 1000)
    assert s == 0.0 and d == 3.0
