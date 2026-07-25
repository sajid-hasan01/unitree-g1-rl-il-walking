import argparse
import csv
import os
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "ember-lab-berkeley/AMASS_Retargeted_for_G1"
REPO_TYPE = "dataset"

OUTPUT_CSV = Path("results") / "amass_walking_candidates.csv"
DOWNLOAD_DIR = Path("datasets") / "raw" / "amass_g1_candidates"

POSITIVE_KEYWORDS = [
    "walk",
    "walking",
    "normal_walk",
    "normalwalk",
    "walk1",
    "walk2",
    "walk_",
    "_walk",
    "straight",
    "forward",
]

NEGATIVE_KEYWORDS = [
    "standtowalk",
    "stand_to_walk",
    "stand-to-walk",
    "stand",
    "sit",
    "sitting",
    "jump",
    "run",
    "running",
    "jog",
    "jogging",
    "dance",
    "turn",
    "turning",
    "stair",
    "stairs",
    "squat",
    "kick",
    "punch",
    "crawl",
    "climb",
    "throw",
    "bend",
    "stretch",
    "cartwheel",
]


def score_file(path):
    lower_path = path.lower()
    filename = Path(path).name.lower()

    score = 0

    if path.startswith("g1/"):
        score += 50

    if filename.endswith("_jpos.npz"):
        score += 30

    for keyword in POSITIVE_KEYWORDS:
        if keyword in lower_path:
            score += 20

    for keyword in NEGATIVE_KEYWORDS:
        if keyword in lower_path:
            score -= 40

    if "walking" in lower_path:
        score += 60

    if "normal_walk" in lower_path or "normalwalk" in lower_path:
        score += 80

    if "standtowalk" in lower_path:
        score -= 150

    return score


def is_motion_file(path):
    return (
        path.startswith("g1/")
        and path.endswith(".npz")
        and "_jpos" in path
    )


def is_walking_candidate(path, include_transitions=False):
    lower_path = path.lower()

    if not is_motion_file(path):
        return False

    if "walk" not in lower_path and "walking" not in lower_path:
        return False

    if not include_transitions:
        transition_terms = [
            "standtowalk",
            "stand_to_walk",
            "stand-to-walk",
        ]

        for term in transition_terms:
            if term in lower_path:
                return False

    strong_bad_terms = [
        "run",
        "jog",
        "jump",
        "dance",
        "sit",
        "stair",
        "squat",
        "kick",
        "punch",
        "crawl",
        "climb",
    ]

    for term in strong_bad_terms:
        if term in lower_path:
            return False

    return True


def extract_npz_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    if "fps" in data:
        fps = float(data["fps"][0])
    else:
        fps = None

    if "dof_positions" in data:
        frames = int(data["dof_positions"].shape[0])
    else:
        frames = None

    root_positions = None

    if "body_positions" in data and "body_names" in data:
        body_positions = data["body_positions"].astype(np.float32)
        body_names = [str(name) for name in data["body_names"]]

        if "pelvis" in body_names:
            pelvis_index = body_names.index("pelvis")
        else:
            pelvis_index = 0

        root_positions = body_positions[:, pelvis_index, :]

    if root_positions is None:
        return {
            "fps": fps,
            "frames": frames,
            "duration_sec": None,
            "dx": None,
            "dy": None,
            "dz": None,
            "xy_displacement": None,
            "dominant_axis": None,
            "avg_xy_speed": None,
        }

    start = root_positions[0]
    end = root_positions[-1]
    delta = end - start

    dx = float(delta[0])
    dy = float(delta[1])
    dz = float(delta[2])

    xy_displacement = float(np.linalg.norm(delta[:2]))

    if abs(dx) >= abs(dy):
        dominant_axis = "x"
    else:
        dominant_axis = "y"

    duration_sec = None
    avg_xy_speed = None

    if fps is not None and frames is not None and fps > 0:
        duration_sec = frames / fps
        avg_xy_speed = xy_displacement / duration_sec

    return {
        "fps": fps,
        "frames": frames,
        "duration_sec": duration_sec,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "xy_displacement": xy_displacement,
        "dominant_axis": dominant_axis,
        "avg_xy_speed": avg_xy_speed,
    }


def inspect_candidate(path):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=path,
        local_dir=str(DOWNLOAD_DIR),
    )

    info = extract_npz_data(local_path)
    return info


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--inspect_top", type=int, default=0)
    parser.add_argument("--include_transitions", action="store_true")

    args = parser.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)

    api = HfApi()

    print("Listing files from Hugging Face dataset...")
    print("Repo:", REPO_ID)
    print()

    files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    candidates = []

    for path in files:
        if is_walking_candidate(path, include_transitions=args.include_transitions):
            candidates.append(
                {
                    "path": path,
                    "score": score_file(path),
                    "fps": "",
                    "frames": "",
                    "duration_sec": "",
                    "dx": "",
                    "dy": "",
                    "dz": "",
                    "xy_displacement": "",
                    "dominant_axis": "",
                    "avg_xy_speed": "",
                }
            )

    candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True,
    )

    print(f"Total walking candidates found: {len(candidates)}")
    print()

    inspect_count = min(args.inspect_top, len(candidates))

    if inspect_count > 0:
        print(f"Inspecting top {inspect_count} candidates...")
        print("This downloads only the selected top files.")
        print()

        for i in range(inspect_count):
            path = candidates[i]["path"]

            try:
                info = inspect_candidate(path)

                for key, value in info.items():
                    if value is None:
                        candidates[i][key] = ""
                    elif isinstance(value, float):
                        candidates[i][key] = round(value, 4)
                    else:
                        candidates[i][key] = value

            except Exception as error:
                candidates[i]["frames"] = f"ERROR: {error}"

    print("Top candidates:")
    print()

    for i, item in enumerate(candidates[: args.limit], start=1):
        print(f"{i:03d}. score={item['score']} | {item['path']}")

        if item["frames"] != "":
            print(
                f"     frames={item['frames']}, "
                f"fps={item['fps']}, "
                f"duration={item['duration_sec']}s, "
                f"dx={item['dx']}, "
                f"dy={item['dy']}, "
                f"xy_disp={item['xy_displacement']}, "
                f"axis={item['dominant_axis']}, "
                f"avg_speed={item['avg_xy_speed']}"
            )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "score",
                "path",
                "fps",
                "frames",
                "duration_sec",
                "dx",
                "dy",
                "dz",
                "xy_displacement",
                "dominant_axis",
                "avg_xy_speed",
            ],
        )

        writer.writeheader()

        for item in candidates:
            writer.writerow(item)

    print()
    print("Saved candidate list:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()