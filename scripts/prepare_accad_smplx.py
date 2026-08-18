from pathlib import Path
import numpy as np
import shutil
import subprocess
import zipfile
import tarfile
import re


PROJECT = Path(r"C:\Projects\unitree-g1-rl-il-walking")
DOWNLOADS = Path.home() / "Downloads"

DEST_ROOT = PROJECT / "datasets" / "raw" / "amass_smplx"
DEST = DEST_ROOT / "ACCAD"

DEST.mkdir(parents=True, exist_ok=True)


print("=" * 110)
print("STAGE A1 — LOCATE ACCAD SMPL-X DOWNLOAD")
print("=" * 110)


patterns = [
    re.compile(r"accad", re.I),
    re.compile(r"smpl.?x", re.I),
]


candidates = []

for p in DOWNLOADS.iterdir():

    if not p.is_file():
        continue

    name = p.name.lower()

    if (
        "accad" in name
        or
        ("amass" in name and "smpl" in name)
    ):
        candidates.append(p)


candidates.sort(
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)


print("Downloads directory:", DOWNLOADS)

if candidates:

    print("\nPotential archives:")

    for i, p in enumerate(candidates[:10]):

        size_gb = p.stat().st_size / 1024**3

        print(
            f"[{i}] {p.name}\n"
            f"    {p}\n"
            f"    size={size_gb:.3f} GB"
        )

else:

    print("\nNo obvious ACCAD archive found in Downloads.")


# ---------------------------------------------------------
# If already extracted, skip extraction
# ---------------------------------------------------------

existing_npz = list(
    DEST.rglob("*.npz")
)


if existing_npz:

    print()
    print("=" * 110)
    print("ACCAD ALREADY APPEARS EXTRACTED")
    print("=" * 110)

    print("NPZ count:", len(existing_npz))

else:

    if not candidates:

        raise SystemExit(
            "\nCould not find the ACCAD archive automatically."
        )


    archive = candidates[0]

    print()
    print("=" * 110)
    print("EXTRACTING")
    print("=" * 110)

    print("archive:", archive)
    print("destination:", DEST)


    lower = archive.name.lower()


    if lower.endswith(".zip"):

        with zipfile.ZipFile(archive, "r") as z:

            z.extractall(DEST)


    elif (
        lower.endswith(".tar.gz")
        or lower.endswith(".tgz")
        or lower.endswith(".tar.bz2")
        or lower.endswith(".tbz2")
        or lower.endswith(".tar")
    ):

        with tarfile.open(archive, "r:*") as t:

            t.extractall(DEST)


    else:

        print(
            "\nPython does not recognize the archive extension."
        )

        print(
            "Trying Windows tar..."
        )

        result = subprocess.run(
            [
                "tar",
                "-xf",
                str(archive),
                "-C",
                str(DEST),
            ]
        )

        if result.returncode != 0:

            raise SystemExit(
                "Extraction failed."
            )


# ---------------------------------------------------------
# Discover NPZ files
# ---------------------------------------------------------

npz_files = sorted(
    DEST.rglob("*.npz")
)


print()
print("=" * 110)
print("STAGE A2 — RAW DATA DISCOVERY")
print("=" * 110)

print("ACCAD root:", DEST)
print("NPZ files:", len(npz_files))


if not npz_files:

    raise SystemExit(
        "\nNo NPZ files were found after extraction."
    )


print("\nFirst files:")

for p in npz_files[:15]:

    print(" ", p.relative_to(DEST))


# ---------------------------------------------------------
# Inspect internal structure
# ---------------------------------------------------------

human_keys_expected = {
    "poses",
    "trans",
    "transl",
    "betas",
    "gender",
    "mocap_frame_rate",
    "mocap_framerate",
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
}


robot_keys = {
    "dof_positions",
    "dof_velocities",
    "body_positions",
    "body_rotations",
    "full_qpos",
    "full_qvel",
}


print()
print("=" * 110)
print("STAGE A3 — VERIFY RAW SMPL-X FORMAT")
print("=" * 110)


valid_human = []
robot_like = []
failed = []


for path in npz_files[:100]:

    try:

        with np.load(
            path,
            allow_pickle=True,
        ) as f:

            keys = set(f.files)

            h_hits = sorted(
                keys & human_keys_expected
            )

            r_hits = sorted(
                keys & robot_keys
            )


            if len(h_hits) >= 3:

                valid_human.append(
                    (
                        path,
                        h_hits,
                    )
                )

            elif r_hits:

                robot_like.append(
                    (
                        path,
                        r_hits,
                    )
                )

    except Exception as e:

        failed.append(
            (
                path,
                str(e),
            )
        )


print(
    "raw-human candidates:",
    len(valid_human)
)

print(
    "robot-like files:",
    len(robot_like)
)

print(
    "read failures:",
    len(failed)
)


if not valid_human:

    print()
    print(
        "RESULT: RAW SMPL-X FORMAT NOT YET CONFIRMED"
    )

    if robot_like:

        print(
            "WARNING: files appear robot-retargeted."
        )

    raise SystemExit(1)


print()
print(
    "RESULT: RAW HUMAN SMPL-X DATA CONFIRMED"
)


# ---------------------------------------------------------
# Detailed inspection of first valid motion
# ---------------------------------------------------------

sample = valid_human[0][0]


print()
print("=" * 110)
print("STAGE A4 — SAMPLE STRUCTURE")
print("=" * 110)

print("sample:")
print(sample)


with np.load(
    sample,
    allow_pickle=True,
) as f:

    print("\nkeys:")
    print(f.files)

    print("\narrays:")

    for key in f.files:

        arr = np.asarray(
            f[key]
        )

        print(
            f"{key:30s} "
            f"shape={str(arr.shape):22s} "
            f"dtype={arr.dtype}"
        )


print()
print("=" * 110)
print("ACCAD SMPL-X VALIDATION COMPLETE")
print("=" * 110)
