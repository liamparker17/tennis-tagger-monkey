"""Validate tagger output against Dartfish ground truth CSV.

Extracts individual point clips from a match video using timestamps
from the ground truth CSV, runs the tagger on each, and compares
the output against the human-tagged data.

Usage:
    python scripts/validate_against_ground_truth.py \
        --video "E:/TaggaBot/Marcos Giron vs Camilo Ugo-Carabelli 2025 Wimbledon R1 (HE).mp4" \
        --csv "E:/TaggaBot/Marcos Giron vs Camilo Ugo-Carabelli 2025 Wimbledon R1 (HE).mp4.csv" \
        --points 5 \
        --output validation_results.json
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# Dartfish XY pixel -> court meters conversion
# Derived from serve placement calibration (T/Wide/Body/Net positions)
def dartfish_xy_to_court(x_px: int, y_px: int) -> tuple[float, float]:
    """Convert Dartfish tagging panel XY coords to court meters.

    Returns (cx, cy) where:
        cx: 0 = doubles sideline, ~5.5 = center, ~11 = far sideline
        cy: 0 = far baseline, 11.885 = net, 23.77 = near baseline
    """
    cx = 0.04571 * x_px - 4.571
    cy = 0.03832 * y_px + 1.845
    return round(cx, 2), round(cy, 2)


@dataclass
class GroundTruthPoint:
    name: str
    position_ms: int
    duration_ms: int
    server: str
    serve_data: str
    serve_placement: str
    serve_speed_1st: Optional[int]
    serve_speed_2nd: Optional[int]
    stroke_count: str
    point_won: str
    score: str
    game_score: str
    set_num: str
    xy_deuce: Optional[tuple[float, float]]  # court meters
    xy_ad: Optional[tuple[float, float]]
    xy_last_shot: Optional[tuple[float, float]]
    xy_return: Optional[tuple[float, float]]
    xy_srv1: Optional[tuple[float, float]]


@dataclass
class TaggerPoint:
    name: str
    rally_length: int
    serve_speed: float
    shots: list[dict]


@dataclass
class ComparisonResult:
    point_name: str
    ground_truth_strokes: str
    tagger_shots: int
    gt_serve_speed: Optional[int]
    tagger_serve_speed: float
    gt_server: str
    gt_point_won: str
    gt_xy_positions: dict
    tagger_positions: list[dict]
    notes: list[str]


def parse_ground_truth(csv_path: str) -> list[GroundTruthPoint]:
    """Parse the Dartfish ground truth CSV into structured points."""
    points = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        next(reader)  # skip metadata row

        for row in reader:
            name = row[0]
            if not name or "SAVE" in name or "DELETE" in name:
                continue

            def get(colname: str) -> str:
                return row[col[colname]] if colname in col else ""

            def parse_xy(val: str) -> Optional[tuple[float, float]]:
                if val and ";" in val:
                    x, y = val.split(";")
                    return dartfish_xy_to_court(int(x), int(y))
                return None

            def parse_speed(val: str) -> Optional[int]:
                try:
                    return int(val) if val else None
                except ValueError:
                    return None

            points.append(GroundTruthPoint(
                name=name,
                position_ms=int(row[1]),
                duration_ms=int(row[2]),
                server=get("A1: Server"),
                serve_data=get("A2: Serve Data"),
                serve_placement=get("A3: Serve Placement"),
                serve_speed_1st=parse_speed(get("A4: 1st Serve Speed")),
                serve_speed_2nd=parse_speed(get("A4: 2nd Serve Speed")),
                stroke_count=get("H1: Stroke Count"),
                point_won=get("F1: Point Won"),
                score=get("F2: Point Score"),
                game_score=get("G1: Game Score"),
                set_num=get("G3: Set #"),
                xy_deuce=parse_xy(get("XY Deuce")),
                xy_ad=parse_xy(get("XY Ad")),
                xy_last_shot=parse_xy(get("XY Last Shot")),
                xy_return=parse_xy(get("XY Return")),
                xy_srv1=parse_xy(get("XY Srv+1")),
            ))
    return points


def extract_clip(video_path: str, start_ms: int, duration_ms: int, output_path: str) -> bool:
    """Extract a video clip using ffmpeg."""
    start_sec = start_ms / 1000.0
    duration_sec = duration_ms / 1000.0
    # Add 1s padding before and after for context
    start_sec = max(0, start_sec - 1.0)
    duration_sec += 2.0

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration_sec),
        "-c:v", "libx264", "-crf", "23", "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def run_tagger(clip_path: str, tagger_dir: str) -> Optional[list[dict]]:
    """Run the tagger on a clip and parse the CSV output."""
    cmd = ["go", "run", "./cmd/tagger", clip_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=tagger_dir)
    if result.returncode != 0:
        print(f"  Tagger failed: {result.stderr[:200]}", file=sys.stderr)
        return None

    csv_path = clip_path + "_output.csv"
    if not os.path.exists(csv_path):
        return None

    points = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            notes = row.get("V1: Notes", "")
            shots = []
            if notes:
                for shot_str in notes.split(";"):
                    shot = {}
                    for kv in shot_str.split(","):
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            # Strip "shotN:" prefix from first field
                            if ":" in k:
                                k = k.split(":")[-1]
                            shot[k] = v
                    if shot:
                        shots.append(shot)

            points.append({
                "name": row.get("Name", ""),
                "rally_length": int(row.get("G1: Rally Length", "0")),
                "serve_speed": float(row.get("M1: Serve Speed", "0") or "0"),
                "shots": shots,
            })

    # Clean up
    try:
        os.remove(csv_path)
    except OSError:
        pass

    return points


def compare_point(gt: GroundTruthPoint, tagger_points: list[dict]) -> ComparisonResult:
    """Compare tagger output against ground truth for one point."""
    notes = []

    # Tagger may produce 0 or more points from the clip
    total_shots = sum(p["rally_length"] for p in tagger_points)
    tagger_speed = tagger_points[0]["serve_speed"] if tagger_points else 0.0

    # Stroke count comparison
    gt_count_str = gt.stroke_count  # e.g. "1 to 4", "5 to 8", "13+"
    if gt_count_str:
        if "+" in gt_count_str:
            gt_min = int(gt_count_str.replace("+", ""))
            if total_shots < gt_min:
                notes.append(f"Too few shots: {total_shots} < {gt_min}+")
        elif "to" in gt_count_str:
            lo, hi = [int(x.strip()) for x in gt_count_str.split("to")]
            if total_shots < lo:
                notes.append(f"Too few shots: {total_shots} < {lo}")
            elif total_shots > hi * 3:  # generous upper bound
                notes.append(f"Way too many shots: {total_shots} >> {hi}")

    # Serve speed comparison
    gt_speed = gt.serve_speed_1st or gt.serve_speed_2nd
    if gt_speed and tagger_speed > 0:
        speed_error = abs(tagger_speed - gt_speed)
        if speed_error > 30:
            notes.append(f"Speed off by {speed_error:.0f} km/h ({tagger_speed:.0f} vs {gt_speed})")

    # Collect all tagger court positions
    tagger_positions = []
    for p in tagger_points:
        for s in p["shots"]:
            cx = float(s.get("cx", 0))
            cy = float(s.get("cy", 0))
            if cx != 0 or cy != 0:
                tagger_positions.append({"cx": cx, "cy": cy})

    # Ground truth XY positions
    gt_xy = {}
    if gt.xy_deuce:
        gt_xy["deuce_serve"] = {"cx": gt.xy_deuce[0], "cy": gt.xy_deuce[1]}
    if gt.xy_ad:
        gt_xy["ad_serve"] = {"cx": gt.xy_ad[0], "cy": gt.xy_ad[1]}
    if gt.xy_return:
        gt_xy["return"] = {"cx": gt.xy_return[0], "cy": gt.xy_return[1]}
    if gt.xy_srv1:
        gt_xy["srv+1"] = {"cx": gt.xy_srv1[0], "cy": gt.xy_srv1[1]}
    if gt.xy_last_shot:
        gt_xy["last_shot"] = {"cx": gt.xy_last_shot[0], "cy": gt.xy_last_shot[1]}

    if not tagger_points:
        notes.append("Tagger produced no points")
    elif len(tagger_points) > 1:
        notes.append(f"Tagger split into {len(tagger_points)} points (expected 1)")

    return ComparisonResult(
        point_name=gt.name,
        ground_truth_strokes=gt.stroke_count,
        tagger_shots=total_shots,
        gt_serve_speed=gt_speed,
        tagger_serve_speed=tagger_speed,
        gt_server=gt.server,
        gt_point_won=gt.point_won,
        gt_xy_positions=gt_xy,
        tagger_positions=tagger_positions[:5],  # first 5 for brevity
        notes=notes,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate tagger against ground truth")
    parser.add_argument("--video", required=True, help="Path to match video")
    parser.add_argument("--csv", required=True, help="Path to ground truth CSV")
    parser.add_argument("--points", type=int, default=5, help="Number of points to test")
    parser.add_argument("--offset", type=int, default=0, help="Start from this point index")
    parser.add_argument("--output", default="validation_results.json", help="Output JSON path")
    args = parser.parse_args()

    tagger_dir = str(Path(__file__).parent.parent)

    print(f"Parsing ground truth from {args.csv}...")
    gt_points = parse_ground_truth(args.csv)
    print(f"Found {len(gt_points)} tagged points")

    results = []
    test_points = gt_points[args.offset : args.offset + args.points]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, gt in enumerate(test_points):
            print(f"\n--- Point {i+1}/{len(test_points)}: {gt.name} ---")
            print(f"  Server: {gt.server}, Score: {gt.score}")
            print(f"  Position: {gt.position_ms}ms, Duration: {gt.duration_ms}ms")
            print(f"  Serve: {gt.serve_data} ({gt.serve_speed_1st or gt.serve_speed_2nd} km/h)")
            print(f"  Strokes: {gt.stroke_count}")

            clip_path = os.path.join(tmpdir, f"point_{i}.mp4")
            print(f"  Extracting clip...")
            if not extract_clip(args.video, gt.position_ms, gt.duration_ms, clip_path):
                print(f"  FAILED to extract clip")
                continue

            print(f"  Running tagger...")
            tagger_points = run_tagger(clip_path, tagger_dir)
            if tagger_points is None:
                print(f"  FAILED to run tagger")
                results.append(ComparisonResult(
                    point_name=gt.name,
                    ground_truth_strokes=gt.stroke_count,
                    tagger_shots=0,
                    gt_serve_speed=gt.serve_speed_1st or gt.serve_speed_2nd,
                    tagger_serve_speed=0,
                    gt_server=gt.server,
                    gt_point_won=gt.point_won,
                    gt_xy_positions={},
                    tagger_positions=[],
                    notes=["Tagger execution failed"],
                ))
                continue

            comparison = compare_point(gt, tagger_points)
            results.append(comparison)

            print(f"  Result: {comparison.tagger_shots} shots (GT: {gt.stroke_count})")
            print(f"  Speed: {comparison.tagger_serve_speed:.0f} km/h (GT: {comparison.gt_serve_speed or 'N/A'})")
            if comparison.notes:
                for note in comparison.notes:
                    print(f"  NOTE: {note}")

    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY ({len(results)} points)")
    print(f"{'='*60}")

    issues = sum(1 for r in results if r.notes)
    print(f"Points with issues: {issues}/{len(results)}")

    speed_errors = []
    for r in results:
        if r.gt_serve_speed and r.tagger_serve_speed > 0:
            speed_errors.append(abs(r.tagger_serve_speed - r.gt_serve_speed))
    if speed_errors:
        print(f"Serve speed MAE: {sum(speed_errors)/len(speed_errors):.1f} km/h")

    # Save results
    output_path = os.path.join(tagger_dir, args.output)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
