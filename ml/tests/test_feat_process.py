import os, pytest, json
from pathlib import Path
from ml.feature_extractor.process import process_clip

CLIP = Path("files/data/clips/_smoke/p_0001.mp4")


@pytest.mark.skipif(not CLIP.exists(), reason="needs Plan 1 smoke output")
def test_process_clip(tmp_path):
    out = tmp_path / "p.npz"
    process_clip(CLIP, out, fps=30)
    assert out.exists()
