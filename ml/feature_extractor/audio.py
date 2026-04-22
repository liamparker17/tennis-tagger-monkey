from __future__ import annotations
import subprocess, tempfile
from pathlib import Path
import numpy as np

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

SR = 22050; N_MELS = 64; HOP = 512

def extract_log_mel(video: Path, tmp_dir: Path | None = None) -> np.ndarray:
    import librosa
    tmp = Path(tempfile.mkdtemp() if tmp_dir is None else tmp_dir)
    wav = tmp / (video.stem + ".wav")
    subprocess.run([
        "ffmpeg","-nostdin","-y","-loglevel","error",
        "-i", str(video), "-ac","1","-ar", str(SR), str(wav)
    ], check=True, creationflags=_NO_WINDOW)
    y, _ = librosa.load(wav, sr=SR, mono=True)
    if y.size == 0: return np.zeros((N_MELS, 1), np.float32)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP)
    return librosa.power_to_db(mel + 1e-10).astype(np.float32)
