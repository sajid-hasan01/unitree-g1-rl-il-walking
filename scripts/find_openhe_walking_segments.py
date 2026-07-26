import argparse
import csv
import pathlib
import pickle
from pathlib import Path

import joblib
import numpy as np
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "openhe/g1-retargeted-motions"
REPO_TYPE = "dataset"

DOWNLOAD_DIR = Path("datasets") / "raw" / "openhe_g1_retargeted_motions"
OUTPUT_CSV = Path("results") / "openhe_g1_best_walking_segments.csv"


POSITIVE_KEYWORDS = [
    "walk",
    "walking",
    "walk1",
    "walk2",
    "walk3",
    "walk4",
]

NEGATIVE_KEYWORDS = [
    "run",
    "running",
    "sprint",
    "jump",
    "fall",
    "dance",
    "fight",
    "kick",
    "punch",
    "cartwheel",
    "crawl",
    "sit",
    "stand",
    "lie",
    "crouch",
    "turn",
]


def is_walking_candidate(repo_file):
    lower_path = repo_file.lower()

    if not lower_path.endswith(".pkl"):
        return False

    if not any(keyword in lower_path for keyword in POSITIVE_KEYWORDS):
        return False

    if any(keyword in lower_path for keyword in NEGATIVE_KEYWORDS):
        return False

    return True


def score_file(repo_file):
    lower_path = repo_file.lower()
    score = 0

    if "lafan1" in lower_path:
        score += 50

    if "walk1" in lower_path:
        score += 40
    if "walk2" in lower_path:
        score += 40
    if "walk3" in lower_path:
        score += 40
    if "walk4" in lower_path:
        score += 40

    if "walk" in lower_path:
        score += 30

    for keyword in NEGATIVE_KEYWORDS:
        if keyword in lower_path:
            score -= 100

    return score


def download_file(repo_file):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=repo_file,
        local_dir=str(DOWNLOAD_DIR),
    )

    return Path(local_path)


def load_motion(local_path):
    original_posix_path = pathlib.PosixPath

    try:
        pathlib.PosixPath = pathlib.WindowsPath

        try:
            data = joblib.load(local_path)
        except Exception:
            with open(local_path, "rb") as file:
                data = pickle.load(file)

    finally:
        pathlib.PosixPath = original_posix_path

    if not isinstance(data, dict):
        raise ValueError(f"Loaded object is not dict. Type: {type(data)}")

    keys = list(data.keys())

    if len(keys) == 0:
        raise ValueError("Loaded dictionary is empty.")

    first_key = keys[0]
    first_value = data[first_key]

    if isinstance(first_value, dict):
        motion_key = str(first_key)
        motion = first_value
    else:
        motion_key = "direct"
        motion = data

    if not isinstance(motion, dict):
        raise ValueError(f"Motion object is not dict. Type: {type(motion)}")

    return motion_key, motion


def get_array(motion, key):
    if key not in motion:
        return None

    value = np.asarray(motion[key])

    if value.size == 0:
        return None

    return value


def get_fps(motion):
    if "fps" not in motion:
        return 30.0

    value = motion["fps"]

    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]

    return float(value)


def scan_segments(repo_file, window_frames, stride_frames):
    local_path = download_file(repo_file)
    motion_key, motion = load_motion(local_path)

    root_pos = get_array(motion, "root_trans_offset")
    dof = get_array(motion, "dof")
    contact_mask = get_array(motion, "contact_mask")
    fps = get_fps(motion)

    if root_pos is None:
        raise KeyError("Missing root_trans_offset")

    if dof is None:
        raise KeyError("Missing dof")

    root_pos = root_pos.astype(np.float32)
    dof = dof.astype(np.float32)

    if root_pos.ndim != 2 or root_pos.shape[1] < 3:
        raise ValueError(f"Invalid root_trans_offset shape: {root_pos.shape}")

    if dof.ndim != 2:
        raise ValueError(f"Invalid dof shape: {dof.shape}")

    frames = min(root_pos.shape[0], dof.shape[0])

    if frames < window_frames:
        return []

    results = []

    for start in range(0, frames - window_frames + 1, stride_frames):
        end = start + window_frames

        segment_root = root_pos[start:end, :3]
        delta = segment_root[-1] - segment_root[0]

        dx = float(delta[0])
        dy = float(delta[1])
        dz = float(delta[2])

        xy_displacement = float(np.linalg.norm(delta[:2]))
        duration_sec = window_frames / fps if fps > 0 else 0.0
        avg_xy_speed = xy_displacement / duration_sec if duration_sec > 0 else 0.0

        dominant_axis = "x" if abs(dx) >= abs(dy) else "y"

        vertical_range = float(
            np.max(segment_root[:, 2]) - np.min(segment_root[:, 2])
        )

        mean_height = float(np.mean(segment_root[:, 2]))

        dof_segment = dof[start:end]
        dof_range_mean = float(np.mean(np.max(dof_segment, axis=0) - np.min(dof_segment, axis=0)))
        dof_abs_mean = float(np.mean(np.abs(dof_segment)))

        if contact_mask is not None:
            contact_segment = np.asarray(contact_mask[start:end])
            contact_mean = np.mean(contact_segment, axis=0).tolist()
            contact_shape = str(tuple(contact_mask.shape))
        else:
            contact_mean = ""
            contact_shape = ""

        score = (
            3.0 * xy_displacement
            + 1.0 * avg_xy_speed
            + 0.1 * dof_range_mean
            - 0.5 * abs(dz)
            - 0.2 * vertical_range
        )

        results.append(
            {
                "score": score,
                "repo_file": repo_file,
                "local_path": str(local_path),
                "motion_key": motion_key,
                "start_frame": start,
                "end_frame": end,
                "window_frames": window_frames,
                "fps": fps,
                "duration_sec": duration_sec,
                "total_frames": frames,
                "dof_shape": str(tuple(dof.shape)),
                "dof_dim": int(dof.shape[1]),
                "contact_shape": contact_shape,
                "contact_mean": contact_mean,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "xy_displacement": xy_displacement,
                "avg_xy_speed": avg_xy_speed,
                "dominant_axis": dominant_axis,
                "mean_height": mean_height,
                "vertical_range": vertical_range,
                "dof_range_mean": dof_range_mean,
                "dof_abs_mean": dof_abs_mean,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inspect_top", type=int, default=12)
    parser.add_argument("--top_segments", type=int, default=30)
    parser.add_argument("--window_frames", type=int, default=300)
    parser.add_argument("--stride_frames", type=int, default=60)

    args = parser.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)

    api = HfApi()

    print("Listing files from Hugging Face dataset...")
    print("Repo:", REPO_ID)
    print()

    repo_files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    candidates = []

    for repo_file in repo_files:
        if is_walking_candidate(repo_file):
            candidates.append(
                {
                    "repo_file": repo_file,
                    "score": score_file(repo_file),
                }
            )

    candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True,
    )

    candidates = candidates[: args.inspect_top]

    print("Walking candidates scanned:", len(candidates))
    print("Window frames:", args.window_frames)
    print("Stride frames:", args.stride_frames)
    print()

    all_segments = []

    for i, item in enumerate(candidates, start=1):
        repo_file = item["repo_file"]
        print(f"[{i}/{len(candidates)}] scanning {repo_file}")

        try:
            segments = scan_segments(
                repo_file=repo_file,
                window_frames=args.window_frames,
                stride_frames=args.stride_frames,
            )

            all_segments.extend(segments)

            if len(segments) > 0:
                best = max(segments, key=lambda row: row["score"])
                print(
                    "  best:",
                    "start",
                    best["start_frame"],
                    "end",
                    best["end_frame"],
                    "| xy_disp",
                    round(best["xy_displacement"], 4),
                    "m | avg_speed",
                    round(best["avg_xy_speed"], 4),
                    "m/s | axis",
                    best["dominant_axis"],
                )
            else:
                print("  no valid segments")

        except Exception as error:
            print("  ERROR:", error)

    all_segments = sorted(
        all_segments,
        key=lambda row: row["score"],
        reverse=True,
    )

    print()
    print("Top walking segments:")
    print()

    for i, row in enumerate(all_segments[: args.top_segments], start=1):
        print(
            f"{i:03d}. score={row['score']:.4f} | "
            f"{row['repo_file']} | "
            f"frames {row['start_frame']}:{row['end_frame']} | "
            f"xy_disp={row['xy_displacement']:.4f}m | "
            f"avg_speed={row['avg_xy_speed']:.4f}m/s | "
            f"dx={row['dx']:.4f} | dy={row['dy']:.4f} | "
            f"axis={row['dominant_axis']} | dof={row['dof_dim']}"
        )

    fieldnames = [
        "score",
        "repo_file",
        "local_path",
        "motion_key",
        "start_frame",
        "end_frame",
        "window_frames",
        "fps",
        "duration_sec",
        "total_frames",
        "dof_shape",
        "dof_dim",
        "contact_shape",
        "contact_mean",
        "dx",
        "dy",
        "dz",
        "xy_displacement",
        "avg_xy_speed",
        "dominant_axis",
        "mean_height",
        "vertical_range",
        "dof_range_mean",
        "dof_abs_mean",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_segments:
            writer.writerow(row)

    print()
    print("Saved segment CSV:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()