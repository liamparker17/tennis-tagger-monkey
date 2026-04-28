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
                 match_filter: Optional[list[str]] = None, T_max: int = T_MAX,
                 clip_filter: Optional[set[str]] = None):
        """clip_filter: when set, only clips whose stem is in the set are
        kept. Used by the single-match clip-level val fallback."""
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
                stem = Path(p["clip"]).stem
                if clip_filter is not None and stem not in clip_filter: continue
                npz = features_root / d.name / (stem + ".npz")
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

        fps = int(z["fps"])
        strong_pairs, bounce_frames = _supervision_from_point(p, fps, Tm)
        targets = build_targets(
            T=Tm, fps=fps,
            stroke_count_lo=int(p.get("stroke_count_lo", 0)),
            stroke_count_hi=int(p.get("stroke_count_hi", 0)),
            stroke_types=list(p.get("stroke_types", [])),
            outcome=p.get("outcome", "UnforcedError"),
            point_won_by=p.get("point_won_by", ""),
            server=p.get("server", ""),
            player_a=p.get("player_a", ""),
            player_b=p.get("player_b", ""),
            strong_contact_frames=strong_pairs,
            bounce_frames=bounce_frames,
        )
        return {"features": out, "mask": mask, "targets": targets,
                "meta": {"match": match, "clip": p["clip"]}}


def _supervision_from_point(p: dict, fps: int, T_max: int
                            ) -> tuple[Optional[list[tuple[int, int]]], Optional[list[int]]]:
    """Build (contact_strong, bounce_frames) supervision for one point.

    Priority for contacts: contact_labels.json (dense, from contact_labeler GUI)
    if present, else ball_anchors from labels.json (sparse: typically just
    serve_contact). Bounces always come from ball_anchors *_bounce entries
    since contact_labels only tracks contacts, not bounces.

    Hitter for serve_contact = the server's pose-axis index (0 = player_a).
    For other contact types we don't infer hitter here — the alternation
    prior in build_targets handles the rest.
    """
    contacts = p.get("_contact")
    strong_pairs: Optional[list[tuple[int, int]]]
    if contacts:
        strong_pairs = [(int(e["frame"]), int(e["hitter"])) for e in contacts]
    else:
        strong_pairs = None
        anchors = p.get("ball_anchors") or []
        # Only use anchors for contact supervision when we know the hitter.
        # serve_contact => server. Other anchors don't tell us who hit.
        server = (p.get("server") or "").strip().lower()
        a_name = (p.get("player_a") or "").strip().lower()
        b_name = (p.get("player_b") or "").strip().lower()
        server_idx = 0 if (a_name and server == a_name) else (
            1 if (b_name and server == b_name) else (0 if server == "p1" else None))
        c_pairs: list[tuple[int, int]] = []
        for a in anchors:
            shot = str(a.get("shot", ""))
            if shot == "serve_contact" and server_idx is not None:
                f = int(round(float(a.get("t_clip_s", 0.0)) * fps))
                if 0 <= f < T_max:
                    c_pairs.append((f, server_idx))
        if c_pairs:
            strong_pairs = c_pairs

    bounce_frames: Optional[list[int]] = None
    anchors = p.get("ball_anchors") or []
    bf: list[int] = []
    for a in anchors:
        shot = str(a.get("shot", ""))
        if shot.endswith("_bounce"):
            f = int(round(float(a.get("t_clip_s", 0.0)) * fps))
            if 0 <= f < T_max:
                bf.append(f)
    if bf:
        bounce_frames = bf
    return strong_pairs, bounce_frames
