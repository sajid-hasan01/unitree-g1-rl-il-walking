import argparse
import csv
import pathlib
import pickle
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "openhe/g1-retargeted-motions"
REPO_TYPE = "dataset"

DOWNLOAD_DIR = Path("datasets") / "raw" / "openhe_g1_retargeted_motions"
OUTPUT_CSV = Path("results") / "openhe_g1_walking_candidates.csv"


POSITIVE_KEYWORDS = [
    "walk",
    "walking",
    "walk1",
    "walk2",
    "walk3",
    "walk4",
    "locomotion",
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


def is_pkl_file(path):
    return path.lower().endswith(".pkl")


def is_walking_candidate(path):
    lower_path = path.lower()

    if not is_pkl_file(path):
        return False

    if not any(keyword in lower_path for keyword in POSITIVE_KEYWORDS):
        return False

    if any(keyword in lower_path for keyword in NEGATIVE_KEYWORDS):
        return False

    return True


def score_file(path):
    lower_path = path.lower()
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


def download_file(repo_file, force_download=False):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=repo_file,
        local_dir=str(DOWNLOAD_DIR),
        force_download=force_download,
    )

    return Path(local_path)


def load_with_joblib(local_path):
    import joblib

    original_posix_path = pathlib.PosixPath

    try:
        pathlib.PosixPath = pathlib.WindowsPath
        data = joblib.load(local_path)
    finally:
        pathlib.PosixPath = original_posix_path

    return data


def load_with_pickle(local_path):
    original_posix_path = pathlib.PosixPath

    try:
        pathlib.PosixPath = pathlib.WindowsPath

        with open(local_path, "rb") as file:
            data = pickle.load(file)

    finally:
        pathlib.PosixPath = original_posix_path

    return data


def load_motion_pkl(local_path):
    errors = []

    try:
        data = load_with_joblib(local_path)
    except Exception as error:
        errors.append(f"joblib.load failed: {error}")
        data = None

    if data is None:
        try:
            data = load_with_pickle(local_path)
        except Exception as error:
            errors.append(f"pickle.load failed: {error}")
            joined_errors = " | ".join(errors)
            raise RuntimeError(joined_errors)

    if not isinstance(data, dict):
        raise ValueError(f"Loaded object is not a dictionary. Type: {type(data)}")

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
        raise ValueError(f"Motion object is not a dictionary. Type: {type(motion)}")

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


def inspect_motion(repo_file, force_download=False):
    local_path = download_file(repo_file, force_download=force_download)

    motion_key, motion = load_motion_pkl(local_path)

    root_pos = get_array(motion, "root_trans_offset")
    root_rot = get_array(motion, "root_rot")
    dof = get_array(motion, "dof")
    contact_mask = get_array(motion, "contact_mask")
    fps = get_fps(motion)

    if root_pos is None:
        raise KeyError("Missing root_trans_offset")

    if dof is None:
        raise KeyError("Missing dof")

    if root_pos.ndim != 2 or root_pos.shape[1] < 3:
        raise ValueError(f"Invalid root_trans_offset shape: {root_pos.shape}")

    if dof.ndim != 2:
        raise ValueError(f"Invalid dof shape: {dof.shape}")

    frames = int(root_pos.shape[0])
    duration_sec = frames / fps if fps > 0 else 0.0

    root_delta = root_pos[-1, :3] - root_pos[0, :3]

    dx = float(root_delta[0])
    dy = float(root_delta[1])
    dz = float(root_delta[2])
    xy_displacement = float(np.linalg.norm(root_delta[:2]))

    avg_xy_speed = xy_displacement / duration_sec if duration_sec > 0 else 0.0
    dominant_axis = "x" if abs(dx) >= abs(dy) else "y"

    if contact_mask is not None:
        contact_shape = str(tuple(contact_mask.shape))
        contact_mean = np.mean(contact_mask, axis=0).tolist()
    else:
        contact_shape = ""
        contact_mean = ""

    if root_rot is not None:
        root_rot_shape = str(tuple(root_rot.shape))
    else:
        root_rot_shape = ""

    result = {
        "repo_file": repo_file,
        "local_path": str(local_path),
        "motion_key": motion_key,
        "fps": fps,
        "frames": frames,
        "duration_sec": duration_sec,
        "dof_shape": str(tuple(dof.shape)),
        "dof_dim": int(dof.shape[1]),
        "root_shape": str(tuple(root_pos.shape)),
        "root_rot_shape": root_rot_shape,
        "contact_shape": contact_shape,
        "contact_mean": contact_mean,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "xy_displacement": xy_displacement,
        "avg_xy_speed": avg_xy_speed,
        "dominant_axis": dominant_axis,
        "keys": ", ".join(sorted(list(motion.keys()))),
        "error": "",
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--inspect_top", type=int, default=20)
    parser.add_argument("--force_download", action="store_true")

    args = parser.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)

    print("Listing files from Hugging Face dataset...")
    print("Repo:", REPO_ID)
    print()

    api = HfApi()

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

    print("Walking candidate files found:", len(candidates))
    print()

    inspected_results = []

    inspect_count = min(args.inspect_top, len(candidates))

    print(f"Inspecting top {inspect_count} candidates...")
    print()

    for i in range(inspect_count):
        repo_file = candidates[i]["repo_file"]

        print(f"[{i + 1}/{inspect_count}] {repo_file}")

        try:
            result = inspect_motion(
                repo_file,
                force_download=args.force_download,
            )

            result["score"] = candidates[i]["score"]
            inspected_results.append(result)

            print(
                "  frames:",
                result["frames"],
                "| duration:",
                round(result["duration_sec"], 3),
                "sec | dof:",
                result["dof_dim"],
                "| xy_disp:",
                round(result["xy_displacement"], 4),
                "m | avg_speed:",
                round(result["avg_xy_speed"], 4),
                "m/s | axis:",
                result["dominant_axis"],
            )

            print("  keys:", result["keys"])
            print("  contact:", result["contact_shape"])

        except Exception as error:
            result = {
                "repo_file": repo_file,
                "score": candidates[i]["score"],
                "error": str(error),
            }

            inspected_results.append(result)
            print("  ERROR:", error)

    print()
    print("Top listed candidates:")
    print()

    for i, item in enumerate(candidates[: args.limit], start=1):
        print(f"{i:03d}. score={item['score']} | {item['repo_file']}")

    fieldnames = [
        "score",
        "repo_file",
        "local_path",
        "motion_key",
        "fps",
        "frames",
        "duration_sec",
        "dof_shape",
        "dof_dim",
        "root_shape",
        "root_rot_shape",
        "contact_shape",
        "contact_mean",
        "dx",
        "dy",
        "dz",
        "xy_displacement",
        "avg_xy_speed",
        "dominant_axis",
        "keys",
        "error",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in inspected_results:
            clean_row = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(clean_row)

    print()
    print("Saved inspection CSV:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()