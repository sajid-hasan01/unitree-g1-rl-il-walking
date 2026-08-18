from pathlib import Path
import numpy as np
import csv
import re


ROOT = Path(
    r"C:\Projects\unitree-g1-rl-il-walking"
)

ACCAD = (
    ROOT
    / "datasets"
    / "raw"
    / "amass_smplx"
    / "ACCAD"
)

OUT_CSV = (
    ROOT
    / "results"
    / "accad_smplx_walk_ranking.csv"
)

OUT_LOG = (
    ROOT
    / "results"
    / "accad_smplx_walk_ranking.log"
)

OUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Filename filtering
# ---------------------------------------------------------

GOOD_WORDS = (
    "walk",
    "walking",
)

BAD_WORDS = (
    "backward",
    "backwards",
    "turn",
    "side",
    "sidestep",
    "skip",
    "hop",
    "jump",
    "run",
    "crouch",
    "crawl",
    "box",
    "pickup",
    "pick_up",
    "lie",
    "martial",
    "kick",
    "punch",
    "leap",
    "stairs",
)


def looks_like_plain_walk(path: Path):

    s = path.name.lower()

    if not any(
        w in s
        for w in GOOD_WORDS
    ):
        return False

    if any(
        w in s
        for w in BAD_WORDS
    ):
        return False

    return True


# ---------------------------------------------------------
# Coordinate-independent trajectory metrics
# ---------------------------------------------------------

def trajectory_metrics(trans, fps):

    trans = np.asarray(
        trans,
        dtype=np.float64,
    )

    n = len(trans)

    if n < 20:
        return None


    duration = (
        (n - 1) / fps
        if fps > 0
        else 0.0
    )


    # Remove initial position.
    x = (
        trans
        -
        trans[0]
    )


    # -----------------------------------------------------
    # PCA gives the dominant travel direction without
    # assuming whether the source is X-forward/Y-forward.
    # -----------------------------------------------------

    centered = (
        x
        -
        x.mean(
            axis=0,
            keepdims=True,
        )
    )


    try:

        _, _, vh = np.linalg.svd(
            centered,
            full_matrices=False,
        )

    except np.linalg.LinAlgError:

        return None


    forward_axis = vh[0]


    scalar_progress = (
        x
        @
        forward_axis
    )


    # Make final displacement positive.
    if scalar_progress[-1] < 0:

        forward_axis = (
            -forward_axis
        )

        scalar_progress = (
            -scalar_progress
        )


    forward_displacement = (
        scalar_progress[-1]
    )


    net_3d = float(
        np.linalg.norm(
            trans[-1]
            -
            trans[0]
        )
    )


    diffs = np.diff(
        trans,
        axis=0,
    )


    segment_length = (
        np.linalg.norm(
            diffs,
            axis=1,
        )
    )


    path_length = float(
        np.sum(
            segment_length
        )
    )


    straightness = (
        net_3d / path_length
        if path_length > 1e-9
        else 0.0
    )


    avg_speed = (
        path_length / duration
        if duration > 0
        else 0.0
    )


    net_speed = (
        net_3d / duration
        if duration > 0
        else 0.0
    )


    # Distance away from dominant trajectory line.
    projected = np.outer(
        scalar_progress,
        forward_axis,
    )


    residual = (
        x
        -
        projected
    )


    lateral_distance = (
        np.linalg.norm(
            residual,
            axis=1,
        )
    )


    lateral_p50 = float(
        np.percentile(
            lateral_distance,
            50,
        )
    )


    lateral_p95 = float(
        np.percentile(
            lateral_distance,
            95,
        )
    )


    # Backtracking along principal direction.
    dprogress = np.diff(
        scalar_progress
    )


    backward_fraction = float(
        np.mean(
            dprogress < -1e-4
        )
    )


    # Root acceleration roughness.
    if len(diffs) >= 2:

        velocity = (
            diffs * fps
        )

        acceleration = (
            np.diff(
                velocity,
                axis=0,
            )
            *
            fps
        )

        acc_norm = np.linalg.norm(
            acceleration,
            axis=1,
        )


        acc_p95 = float(
            np.percentile(
                acc_norm,
                95,
            )
        )

    else:

        acc_p95 = 0.0


    return {
        "frames": n,
        "duration_s": duration,
        "forward_m": float(
            forward_displacement
        ),
        "net_disp_m": net_3d,
        "path_length_m": path_length,
        "avg_speed_mps": avg_speed,
        "net_speed_mps": net_speed,
        "straightness": straightness,
        "lateral_p50_m": lateral_p50,
        "lateral_p95_m": lateral_p95,
        "backward_fraction": backward_fraction,
        "acc_p95": acc_p95,
        "principal_x": float(
            forward_axis[0]
        ),
        "principal_y": float(
            forward_axis[1]
        ),
        "principal_z": float(
            forward_axis[2]
        ),
    }


def score_motion(m):

    # -----------------------------------------------------
    # Prefer:
    # - useful displacement
    # - normal walking speed
    # - straight trajectory
    # - low sideways wandering
    # - little backtracking
    #
    # This is ONLY a source-motion ranking metric.
    # It is NOT our physical acceptance score.
    # -----------------------------------------------------

    displacement_score = min(
        m["net_disp_m"] / 2.0,
        1.0,
    )


    speed = m[
        "net_speed_mps"
    ]


    # Broad normal-walk preference.
    if 0.45 <= speed <= 1.8:

        speed_score = 1.0

    elif 0.25 <= speed < 0.45:

        speed_score = (
            (speed - 0.25)
            /
            0.20
        )

    elif 1.8 < speed <= 2.3:

        speed_score = (
            1.0
            -
            (speed - 1.8)
            /
            0.5
        )

    else:

        speed_score = 0.0


    straight_score = np.clip(
        (
            m["straightness"]
            -
            0.70
        )
        /
        0.30,
        0.0,
        1.0,
    )


    lateral_score = np.clip(
        1.0
        -
        m["lateral_p95_m"]
        /
        0.50,
        0.0,
        1.0,
    )


    backtrack_score = np.clip(
        1.0
        -
        m["backward_fraction"]
        /
        0.20,
        0.0,
        1.0,
    )


    duration_score = np.clip(
        (
            m["duration_s"]
            -
            1.0
        )
        /
        3.0,
        0.0,
        1.0,
    )


    score = (
        30.0
        *
        straight_score

        +
        25.0
        *
        displacement_score

        +
        20.0
        *
        speed_score

        +
        10.0
        *
        lateral_score

        +
        10.0
        *
        backtrack_score

        +
        5.0
        *
        duration_score
    )


    return float(
        score
    )


# ---------------------------------------------------------
# Scan ALL ACCAD files
# ---------------------------------------------------------

all_npz = sorted(
    ACCAD.rglob("*.npz")
)


plain_walks = [
    p
    for p in all_npz
    if looks_like_plain_walk(p)
]


rows = []

invalid = []


for path in plain_walks:

    try:

        with np.load(
            path,
            allow_pickle=True,
        ) as f:

            keys = set(
                f.files
            )


            if (
                "trans" not in keys
                or
                "poses" not in keys
            ):

                invalid.append(
                    (
                        path,
                        "missing trans/poses",
                    )
                )

                continue


            trans = np.asarray(
                f["trans"],
                dtype=np.float64,
            )


            poses = np.asarray(
                f["poses"],
            )


            fps = float(
                np.asarray(
                    f["mocap_frame_rate"]
                ).reshape(-1)[0]
            )


            gender = str(
                np.asarray(
                    f["gender"]
                ).reshape(-1)[0]
            )


            surface = str(
                np.asarray(
                    f["surface_model_type"]
                ).reshape(-1)[0]
            )


            if (
                trans.ndim != 2
                or
                trans.shape[1] != 3
            ):

                invalid.append(
                    (
                        path,
                        f"bad trans shape {trans.shape}",
                    )
                )

                continue


            if (
                poses.ndim != 2
                or
                poses.shape[0]
                !=
                trans.shape[0]
            ):

                invalid.append(
                    (
                        path,
                        f"bad poses shape {poses.shape}",
                    )
                )

                continue


            m = trajectory_metrics(
                trans,
                fps,
            )


            if m is None:

                continue


            row = {
                "file": str(
                    path.relative_to(
                        ACCAD
                    )
                ),
                "full_path": str(
                    path
                ),
                "gender": gender,
                "surface_model_type": surface,
                "fps": fps,
                "pose_dim": poses.shape[1],
                **m,
            }


            row["score"] = (
                score_motion(
                    row
                )
            )


            rows.append(
                row
            )


    except Exception as e:

        invalid.append(
            (
                path,
                repr(e),
            )
        )


rows.sort(
    key=lambda r: r["score"],
    reverse=True,
)


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

if rows:

    fieldnames = list(
        rows[0].keys()
    )


    with OUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ---------------------------------------------------------
# Log
# ---------------------------------------------------------

lines = []

lines.append(
    "=" * 130
)

lines.append(
    "ACCAD RAW SMPL-X STRAIGHT-WALK SOURCE RANKING"
)

lines.append(
    "=" * 130
)

lines.append(
    f"Dataset root: {ACCAD}"
)

lines.append(
    f"All NPZ files: {len(all_npz)}"
)

lines.append(
    f"Plain-walk filename candidates: {len(plain_walks)}"
)

lines.append(
    f"Successfully analyzed: {len(rows)}"
)

lines.append(
    f"Rejected/unreadable: {len(invalid)}"
)

lines.append("")

lines.append(
    "IMPORTANT: ranking is source-kinematic only; "
    "it does NOT prove G1 dynamic feasibility."
)

lines.append("")

lines.append(
    "RANK SCORE  DUR(s) FPS   DISP(m) SPEED  STRAIGHT "
    "LAT95(m) BACK%  FILE"
)

lines.append(
    "-" * 130
)


for i, row in enumerate(
    rows[:30],
    start=1,
):

    lines.append(

        f"{i:4d} "
        f"{row['score']:5.1f} "
        f"{row['duration_s']:7.2f} "
        f"{row['fps']:5.1f} "
        f"{row['net_disp_m']:7.3f} "
        f"{row['net_speed_mps']:6.3f} "
        f"{row['straightness']:8.4f} "
        f"{row['lateral_p95_m']:8.3f} "
        f"{100*row['backward_fraction']:5.1f} "
        f"{row['file']}"
    )


lines.append("")

lines.append(
    "=" * 130
)

lines.append(
    "TOP 5 FULL PATHS"
)

lines.append(
    "=" * 130
)


for i, row in enumerate(
    rows[:5],
    start=1,
):

    lines.append(
        f"\n[{i}] score={row['score']:.2f}"
    )

    lines.append(
        row["full_path"]
    )

    lines.append(
        "  "
        f"frames={row['frames']}, "
        f"duration={row['duration_s']:.2f}s, "
        f"fps={row['fps']:.1f}"
    )

    lines.append(
        "  "
        f"displacement={row['net_disp_m']:.3f}m, "
        f"speed={row['net_speed_mps']:.3f}m/s, "
        f"straightness={row['straightness']:.4f}"
    )

    lines.append(
        "  "
        f"lateral95={row['lateral_p95_m']:.3f}m, "
        f"backtracking={100*row['backward_fraction']:.1f}%"
    )

    lines.append(
        "  dominant travel axis="
        f"[{row['principal_x']:+.3f}, "
        f"{row['principal_y']:+.3f}, "
        f"{row['principal_z']:+.3f}]"
    )


if invalid:

    lines.append("")

    lines.append(
        "=" * 130
    )

    lines.append(
        "REJECTED FILES"
    )

    lines.append(
        "=" * 130
    )

    for path, reason in invalid[:30]:

        lines.append(
            f"{path}: {reason}"
        )


text = "\n".join(
    lines
)


OUT_LOG.write_text(
    text,
    encoding="utf-8",
)


print(text)

print()

print(
    "CSV:",
    OUT_CSV,
)

print(
    "LOG:",
    OUT_LOG,
)

