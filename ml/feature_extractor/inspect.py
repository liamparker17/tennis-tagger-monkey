# ml/feature_extractor/inspect.py
import sys
from pathlib import Path
from .schema import load_npz

def main():
    fs = load_npz(Path(sys.argv[1]))
    print(f"schema={fs.schema} fps={fs.fps}")
    print(f"pose_px={fs.pose_px.shape} pose_conf mean={fs.pose_conf.mean():.3f}")
    print(f"pose_valid per-slot frac={fs.pose_valid.mean(axis=0)}")
    print(f"ball detections={int((fs.ball[:,2] > 0).sum())}/{len(fs.ball)}")
    print(f"court_h det={float(__import__('numpy').linalg.det(fs.court_h)):.3f}")
    print(f"audio_mel={fs.audio_mel.shape}")
    print(f"meta duration_s/w/h/native_fps={fs.clip_meta.tolist()}")

if __name__ == "__main__": main()
