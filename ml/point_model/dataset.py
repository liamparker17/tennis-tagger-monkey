from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from torch.utils.data import Dataset
from .features import build_feature_tensor, FEATURE_DIM
from .labels import build_targets, Targets

T_MAX = 900  # 30 s @ 30 fps — long enough for baseline rallies w/o truncation

@dataclass
class Sample:
    features: np.ndarray
    mask: np.ndarray
    targets: Targets
    meta: dict

class ClipDataset(Dataset):
    def __init__(self, clips_root: Path, features_root: Path,
                 match_filter: Optional[list[str]] = None, T_max: int = T_MAX):
        self.clips_root = clips_root; self.features_root = features_root
        self.T_max = T_max
        self.items: list[tuple[str, dict, Path, Optional[Path]]] = []
        for d in sorted(clips_root.iterdir()):
            if not d.is_dir(): continue
            if match_filter is not None and d.name not in match_filter: continue
            labels_path = d / "labels.json"
            if not labels_path.exists(): continue
            doc = json.loads(labels_path.read_text())
            contact_path = d / "contact_labels.json"
            contact_doc = json.loads(contact_path.read_text()) if contact_path.exists() else None
            for p in doc["points"]:
                npz = features_root / d.name / (Path(p["clip"]).stem + ".npz")
                if not npz.exists(): continue
                self.items.append((d.name, p, npz, contact_path if contact_doc else None))
                p["_contact"] = (contact_doc.get("clips", {}).get(Path(p["clip"]).stem)
                                  if contact_doc else None)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        match, p, npz, _ = self.items[i]
        z = np.load(npz)
        feats = build_feature_tensor(
            pose_px=z["pose_px"], pose_court=z["pose_court"],
            pose_conf=z["pose_conf"], pose_valid=z["pose_valid"],
            ball=z["ball"], audio_mel=z["audio_mel"], clip_meta=z["clip_meta"],
        )
        T = feats.shape[0]
        Tm = self.T_max
        out = np.zeros((Tm, FEATURE_DIM), np.float32)
        mask = np.zeros((Tm,), np.float32)
        n = min(T, Tm)
        out[:n] = feats[:n]; mask[:n] = 1.0

        strong = p.get("_contact")
        strong_pairs = [(int(e["frame"]), int(e["hitter"])) for e in strong] if strong else None
        targets = build_targets(
            T=Tm, fps=int(z["fps"]),
            stroke_count_lo=int(p.get("stroke_count_lo", 0)),
            stroke_count_hi=int(p.get("stroke_count_hi", 0)),
            stroke_types=list(p.get("stroke_types", [])),
            outcome=p.get("outcome", "UnforcedError"),
            point_won_by=p.get("point_won_by", ""),
            server=p.get("server", ""),
            player_a=p.get("player_a", ""),
            player_b=p.get("player_b", ""),
            strong_contact_frames=strong_pairs,
        )
        return {"features": out, "mask": mask, "targets": targets,
                "meta": {"match": match, "clip": p["clip"]}}
