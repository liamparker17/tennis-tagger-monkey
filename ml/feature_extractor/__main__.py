import argparse, json, sys
from pathlib import Path
from .process import process_clip
from .pose import PoseExtractor
from .ball import BallDetector
from .ball_color import BallColorModel, calibrate_from_setup as calibrate_ball_color
from .player_color import PlayerColorModel, calibrate_from_setup as calibrate_player_color
from .schema import SCHEMA_VERSION


def _load_or_build_ball_color(match_dir: Path, setup: dict | None) -> BallColorModel | None:
    """Returns a per-match ball color model, building it from preflight
    ball_labels the first time and caching to <match>/ball_color.json on
    subsequent runs. Returns None if preflight didn't collect ball labels."""
    if setup is None: return None
    cache = match_dir / "ball_color.json"
    if cache.exists():
        try: return BallColorModel.load(cache)
        except Exception: pass
    labels = setup.get("ball_labels") or []
    video = setup.get("video")
    if not labels or not video: return None
    model = calibrate_ball_color(Path(video), labels)
    if model is None: return None
    try: model.save(cache)
    except Exception: pass
    return model


def _load_or_build_player_color(match_dir: Path, setup: dict | None) -> PlayerColorModel | None:
    """Same pattern as ball color. Expects preflight step 5 to have populated
    setup['player_colors'] with {near: [[L,a,b],...], far: [[L,a,b],...]}."""
    if setup is None: return None
    cache = match_dir / "player_color.json"
    if cache.exists():
        try: return PlayerColorModel.load(cache)
        except Exception: pass
    player_colors = setup.get("player_colors") or {}
    video = setup.get("video")
    ref_frame = int(setup.get("reference_frame_index", 0))
    if not player_colors or not video: return None
    model = calibrate_player_color(Path(video), ref_frame, player_colors)
    if model is None: return None
    try: model.save(cache)
    except Exception: pass
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("feature_extractor")
    ap.add_argument("clips_root", type=Path, default=Path("files/data/clips"))
    ap.add_argument("--out", type=Path, default=Path("files/data/features"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    pose = PoseExtractor(Path("models/yolov8s-pose.pt"))
    # Prefer the fine-tuned single-class ball detector if it exists; class 0
    # is the sole class in that corpus. Fall back to stock COCO (class 32 =
    # "sports ball") when the fine-tune hasn't been built yet.
    ball_finetuned = Path("files/models/yolo_ball/best.pt")
    if ball_finetuned.is_file():
        ball = BallDetector(ball_finetuned, ball_class_id=0)
        print(f"  ball detector: fine-tuned weights ({ball_finetuned})")
    else:
        ball = BallDetector(Path("models/yolov8s.pt"), ball_class_id=32)
        print(f"  ball detector: stock COCO fallback (fine-tune not built)")

    matches = [d for d in sorted(args.clips_root.iterdir()) if d.is_dir()]
    if args.only: matches = [d for d in matches if args.only in d.name]
    for m in matches:
        out_dir = args.out / m.name
        out_dir.mkdir(parents=True, exist_ok=True)
        setup_path = m / "setup.json"
        setup = None
        if setup_path.exists():
            try:
                setup = json.loads(setup_path.read_text())
                print(f"  {m.name}: using preflight setup (manual court corners)")
            except Exception as e:
                print(f"  {m.name}: failed to read setup.json: {e}")

        # Per-match color calibration (auto, cached). Attach to detectors.
        ball_cm = _load_or_build_ball_color(m, setup)
        player_cm = _load_or_build_player_color(m, setup)
        ball.set_color_model(ball_cm); ball.reset_stats()
        pose.set_color_model(player_cm); pose.reset_stats()
        if ball_cm is not None:
            print(f"    ball color model: {ball_cm.n_samples} px calibrated, sigma={ball_cm.residual_sigma:.1f}")
        if player_cm is not None:
            print(f"    player color model: a={len(player_cm.a_anchors)} b={len(player_cm.b_anchors)} anchors")

        labels_path = m / "labels.json"
        anchors_by_clip: dict[str, list[dict]] = {}
        fps = 30
        if labels_path.exists():
            try:
                labels = json.loads(labels_path.read_text())
                for rec in labels.get("points", []):
                    clip_name = rec.get("clip")
                    if not clip_name: continue
                    T_clip = int(round(float(rec.get("clip_duration_s", 0.0)) * fps))
                    anchors = []
                    for a in rec.get("ball_anchors", []):
                        f = int(round(float(a.get("t_clip_s", 0.0)) * fps))
                        if 0 <= f < T_clip:
                            anchors.append({"frame": f, "xy_court": a.get("xy_court")})
                    anchors_by_clip[Path(clip_name).stem] = anchors
            except Exception as e:
                print(f"  {m.name}: failed to read labels.json: {e}")

        index = []
        for clip in sorted(m.glob("p_*.mp4")):
            out_npz = out_dir / (clip.stem + ".npz")
            if out_npz.exists() and not args.force:
                index.append(clip.stem); continue
            try:
                anchors = anchors_by_clip.get(clip.stem)
                process_clip(clip, out_npz, pose=pose, ball=ball, setup=setup,
                             ball_anchors=anchors)
                index.append(clip.stem)
                print(f"  {m.name}/{clip.stem} ok")
            except Exception as e:
                print(f"  {m.name}/{clip.stem} FAIL: {e}")

        # Per-match detection rate summary so you can tell if a color
        # calibration is actually helping.
        bs = ball.stats
        if bs["frames"] > 0:
            total = bs["frames"]
            hit_pct = 100.0 * (bs["yolo_hit"] + bs["color_rescued"]) / total
            print(f"    ball: {bs['yolo_hit']} yolo, {bs['color_rescued']} color-rescued, "
                  f"{bs['miss']} missed ({hit_pct:.1f}% detected)")
        ps = pose.stats
        if ps["frames"] > 0 and player_cm is not None:
            used_pct = 100.0 * ps["color_used"] / ps["frames"]
            print(f"    pose identity: color used on {used_pct:.1f}% of frames, "
                  f"y-fallback on {100 - used_pct:.1f}%")

        (out_dir / "index.json").write_text(json.dumps({
            "schema": SCHEMA_VERSION, "match": m.name, "clips": index
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
