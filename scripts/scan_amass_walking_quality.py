from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "ember-lab-berkeley/AMASS_Retargeted_for_G1"


def decode_names(arr) -> list[str]:
    names = []
    for x in arr:
        if isinstance(x, bytes):
            names.append(x.decode("utf-8"))
        else:
            names.append(str(x))
    return names


def choose_root_body_index(body_names: list[str]) -> int:
    preferred = ["pelvis", "torso", "trunk", "base", "waist"]
    lower_names = [name.lower() for name in body_names]

    for word in preferred:
        for i, name in enumerate(lower_names):
            if word in name:
                return i

    return 0


def path_length_xy(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0

    diffs = np.diff(points[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def score_motion(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)

    if "dof_positions" not in data.files:
        raise ValueError("Missing dof_positions")

    if "body_positions" not in data.files:
        raise ValueError("Missing body_positions")

    if "body_names" not in data.files:
        raise ValueError("Missing body_names")

    fps = 30.0
    if "fps" in data.files:
        fps_arr = np.asarray(data["fps"]).reshape(-1)
        if len(fps_arr) > 0:
            fps = float(fps_arr[0])

    dof_positions = np.asarray(data["dof_positions"], dtype=np.float64)
    body_positions = np.asarray(data["body_positions"], dtype=np.float64)
    body_names = decode_names(data["body_names"])

    root_index = choose_root_body_index(body_names)
    root_name = body_names[root_index]

    frames = int(dof_positions.shape[0])
    duration_sec = frames / fps if fps > 1e-6 else 0.0

    root_xy = body_positions[:, root_index, :2]
    delta_xy = root_xy[-1] - root_xy[0]

    displacement_xy = float(np.linalg.norm(delta_xy))
    path_len_xy = path_length_xy(root_xy)
    straightness = displacement_xy / path_len_xy if path_len_xy > 1e-6 else 0.0
    speed_mps = displacement_xy / duration_sec if duration_sec > 1e-6 else 0.0

    joint_std = float(np.mean(np.std(dof_positions, axis=0)))
    joint_range = float(np.mean(np.max(dof_positions, axis=0) - np.min(dof_positions, axis=0)))

    duration_score = min(duration_sec / 5.0, 1.0)
    displacement_score = min(displacement_xy / 2.5, 1.0)

    if 0.25 <= speed_mps <= 1.80:
        speed_score = 1.0
    elif speed_mps < 0.25:
        speed_score = speed_mps / 0.25
    else:
        speed_score = max(0.0, 1.0 - ((speed_mps - 1.80) / 1.80))

    if 0.03 <= joint_std <= 0.60:
        joint_score = 1.0
    else:
        joint_score = 0.5

    final_score = (
        0.30 * displacement_score
        + 0.25 * straightness
        + 0.20 * duration_score
        + 0.15 * speed_score
        + 0.10 * joint_score
    )

    return {
        "frames": frames,
        "fps": fps,
        "duration_sec": duration_sec,
        "root_body": root_name,
        "dx": float(delta_xy[0]),
        "dy": float(delta_xy[1]),
        "displacement_xy": displacement_xy,
        "path_len_xy": path_len_xy,
        "straightness": straightness,
        "speed_mps": speed_mps,
        "joint_std": joint_std,
        "joint_range": joint_range,
        "score": final_score,
    }


def is_candidate(filename: str) -> bool:
    lower = filename.lower()

    if not filename.startswith("g1/"):
        return False

    if not filename.endswith(".npz"):
        return False

    include_terms = [
        "normal_walk",
        "walk1",
        "walk2",
        "walk3",
        "walk4",
        "walking_run",
        "b3-walk",
        "qkwalk",
    ]

    if not any(term in lower for term in include_terms):
        return False

    exclude_terms = [
        "turn",
        "sidestep",
        "backward",
        "hop",
        "leap",
        "skip",
        "crouch",
        "pickup",
        "box",
        "runto",
        "walktorun",
        "standtowalk",
        "walktostand",
        "stairs",
        "jump",
        "dance",
        "martial",
    ]

    if any(term in lower for term in exclude_terms):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_downloads", type=int, default=80)
    parser.add_argument("--local_dir", type=str, default="datasets/raw/amass_candidates")
    parser.add_argument("--out_csv", type=str, default="reports/amass_walking_quality_scan.csv")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    candidates = [f for f in files if is_candidate(f)]

    print("=" * 100)
    print("Candidate files found:", len(candidates))
    print("Max downloads:", args.max_downloads)
    print("=" * 100)

    rows = []

    limit = min(args.max_downloads, len(candidates))

    for i, filename in enumerate(candidates[:limit], 1):
        print()
        print(f"[{i}/{limit}] {filename}")

        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=str(local_dir),
            )

            stats = score_motion(Path(local_path))

            row = {
                "rank_input": i,
                "filename": filename,
                "local_path": local_path,
                **stats,
            }

            rows.append(row)

            print(
                f"  frames={stats['frames']} "
                f"duration={stats['duration_sec']:.2f}s "
                f"disp={stats['displacement_xy']:.3f}m "
                f"straight={stats['straightness']:.3f} "
                f"speed={stats['speed_mps']:.3f} "
                f"score={stats['score']:.3f}"
            )

        except Exception as error:
            print("  ERROR:", repr(error))

    rows.sort(key=lambda row: row["score"], reverse=True)

    fieldnames = [
        "score",
        "filename",
        "local_path",
        "frames",
        "fps",
        "duration_sec",
        "root_body",
        "dx",
        "dy",
        "displacement_xy",
        "path_len_xy",
        "straightness",
        "speed_mps",
        "joint_std",
        "joint_range",
        "rank_input",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print()
    print("=" * 100)
    print("TOP 15 AMASS WALKING CANDIDATES")
    print("=" * 100)

    for rank, row in enumerate(rows[:15], 1):
        print(
            f"{rank:02d}. "
            f"score={row['score']:.3f} "
            f"frames={row['frames']} "
            f"duration={row['duration_sec']:.2f}s "
            f"disp={row['displacement_xy']:.3f}m "
            f"straight={row['straightness']:.3f} "
            f"speed={row['speed_mps']:.3f} "
            f"| {row['filename']}"
        )

    print()
    print("Saved CSV:", out_csv)


if __name__ == "__main__":
    main()
