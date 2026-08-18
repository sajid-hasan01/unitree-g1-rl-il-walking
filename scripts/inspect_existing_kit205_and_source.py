from pathlib import Path
import numpy as np
import re


ROOT = Path(
    r"C:\Projects\unitree-g1-rl-il-walking"
)

KIT = (
    ROOT
    / "datasets"
    / "raw"
    / "amass_candidates"
    / "g1"
    / "KIT"
    / "205"
    / "walking_run01_poses_100_jpos.npz"
)


print("=" * 110)
print("1. INSPECT EXISTING KIT205 FILE")
print("=" * 110)

print("path:", KIT)
print("exists:", KIT.exists())


if KIT.exists():

    with np.load(
        KIT,
        allow_pickle=True,
    ) as f:

        print()
        print("keys:")
        print(f.files)

        print()
        print("shapes / dtype:")

        for key in f.files:

            value = np.asarray(
                f[key]
            )

            print(
                f"{key:35s} "
                f"shape={str(value.shape):20s} "
                f"dtype={value.dtype}"
            )


        print()
        print("ROBOT/HUMAN CLASSIFICATION")

        keys = set(
            f.files
        )

        human_keys = {
            "poses",
            "trans",
            "transl",
            "betas",
            "gender",
            "global_orient",
            "body_pose",
            "left_hand_pose",
            "right_hand_pose",
        }


        human_hits = sorted(
            keys
            &
            human_keys
        )


        robot_like = False


        for key in f.files:

            value = np.asarray(
                f[key]
            )

            if (
                value.ndim >= 2
                and
                value.shape[-1]
                in (
                    29,
                    35,
                    36,
                )
            ):

                robot_like = True


        print(
            "human SMPL-X keys:",
            human_hits,
        )

        print(
            "robot-shaped arrays found:",
            robot_like,
        )


        if (
            len(
                human_hits
            )
            >=
            3
        ):

            print(
                "RESULT: POSSIBLE RAW HUMAN DATA"
            )

        elif robot_like:

            print(
                "RESULT: ALREADY-RETARGETED ROBOT DATA"
            )

        else:

            print(
                "RESULT: NEED MANUAL FORMAT REVIEW"
            )


else:

    print(
        "KIT205 file was not found at expected path."
    )


print()
print("=" * 110)
print("2. FIND OLD HUGGING FACE DOWNLOAD SOURCE")
print("=" * 110)


scripts = [
    ROOT
    / "scripts"
    / "download_amass_walking_sample.py",

    ROOT
    / "scripts"
    / "select_kit205_g1_forward_walks.py",

    ROOT
    / "scripts"
    / "list_amass_walking_files.py",
]


patterns = [
    r"repo_id",
    r"snapshot_download",
    r"hf_hub_download",
    r"huggingface",
    r"allow_patterns",
    r"local_dir",
    r"AMASS_Retargeted",
    r"EMBER",
]


for path in scripts:

    print()
    print("-" * 110)
    print(path.name)
    print("-" * 110)

    if not path.exists():

        print("NOT FOUND")
        continue


    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )


    lines = text.splitlines()


    matches = []


    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        if any(
            re.search(
                pattern,
                line,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        ):

            matches.append(
                (
                    line_number,
                    line,
                )
            )


    if not matches:

        print(
            "No Hugging Face/source lines found."
        )

    else:

        for line_number, line in matches:

            print(
                f"{line_number:4d}: {line}"
            )


print()
print("=" * 110)
print("DONE")
print("=" * 110)
