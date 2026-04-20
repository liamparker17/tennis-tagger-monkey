import shutil, pytest, numpy as np
from pathlib import Path
from ml.feature_extractor.audio import extract_log_mel

SAMPLE = Path("testdata/sample_a.mp4")

@pytest.mark.skipif(not SAMPLE.exists() or shutil.which("ffmpeg") is None, reason="needs sample+ffmpeg")
def test_audio_shape(tmp_path):
    mel = extract_log_mel(SAMPLE, tmp_dir=tmp_path)
    assert mel.ndim == 2 and mel.shape[0] == 64 and mel.shape[1] > 0
    assert mel.dtype == np.float32
