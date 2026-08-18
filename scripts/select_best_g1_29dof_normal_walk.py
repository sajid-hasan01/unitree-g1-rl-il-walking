from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import csv
import math
import shutil
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DOWNLOAD_DIR = (
    ROOT
    / "datasets"
    / "candidate_29dof_walks"
)

RESULT_CSV = (
    ROOT
    / "results"
    / "g1_29dof_walk_candidate_ranking.csv"
)

SELECTED_DIR = (
    ROOT
    / "datasets"
    / "selected_29dof_walk"
)


REPO = (
    "https://huggingface.co/datasets/"
    "ember-lab-berkeley/"
    "AMASS_Retargeted_for_G1"
)


CANDIDATES = [
    (
        "rub040_walk1",
        "g1/BioMotionLab_NTroje/rub040/"
        "0000_normal_walk1_poses_120_jpos.npz",
    ),
    (
        "rub040_walk2",
        "g1/BioMotionLab_NTroje/rub040/"
        "0001_normal_walk2_poses_120_jpos.npz",
    ),
    (
        "rub027_walk1",
        "g1/BioMotionLab_NTroje/rub027/"
        "0005_normal_walk1_poses_120_jpos.npz",
    ),
    (
        "rub027_walk2",
        "g1/BioMotionLab_NTroje/rub027/"
        "0006_normal_walk2_poses_120_jpos.npz",
    ),
    (
        "rub027_walk3",
        "g1/BioMotionLab_NTroje/rub027/"
        "0007_normal_walk3_poses_120_jpos.npz",
    ),
    (
        "rub027_walk4",
        "g1/BioMotionLab_NTroje/rub027/"
        "0008_normal_walk4_poses_120_jpos.npz",
    ),
]


# Older verified revision fallback.
REVISIONS = [
    "main",
    "da464c1f553eb982291656326c3aa02bd080fd35",
]


DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SELECTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def download_file(
    relative_path,
    destination,
):

    if destination.exists():

        try:
            check = np.load(
                destination,
                allow_pickle=True,
            )

            if (
                "dof_positions"
                in check.files
            ):

                print(
                    "Already downloaded:",
                    destination.name,
                )

                return

        except Exception:
            destination.unlink(
                missing_ok=True
            )


    last_error = None


    for revision in REVISIONS:

        url = (
            f"{REPO}/resolve/"
            f"{revision}/"
            f"{relative_path}"
            "?download=true"
        )


        try:

            print(
                "Downloading:",
                relative_path,
            )


            req = Request(
                url,
                headers={
                    "User-Agent":
                        "Mozilla/5.0"
                },
            )


            with urlopen(
                req,
                timeout=120,
            ) as response:

                data = response.read()


            destination.write_bytes(
                data
            )


            # Verify actual NPZ, not HTML/pointer.
            test = np.load(
                destination,
                allow_pickle=True,
            )


            if (
                "dof_positions"
                not in test.files
            ):

                raise RuntimeError(
                    "Downloaded file is not "
                    "the expected motion NPZ."
                )


            print(
                "  OK:",
                destination.name,
                f"({len(data)/1024:.1f} KiB)",
            )

            return


        except Exception as exc:

            last_error = exc

            destination.unlink(
                missing_ok=True
            )


    raise RuntimeError(
        f"Could not download "
        f"{relative_path}: "
        f"{last_error}"
    )


def qyaw_wxyz(
    q,
):

    q = np.asarray(
        q,
        dtype=np.float64,
    )


    norm = np.linalg.norm(
        q,
        axis=1,
        keepdims=True,
    )


    q = (
        q
        / np.maximum(
            norm,
            1e-12,
        )
    )


    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]


    return np.unwrap(
        np.arctan2(
            2.0 * (
                w*z
                + x*y
            ),

            1.0
            - 2.0
            * (
                y*y
                + z*z
            ),
        )
    )


def clip01(x):

    return float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )


def periodicity_metric(
    q,
    fps,
):

    # 12 lower-body joints.
    x = np.asarray(
        q[:, :12],
        dtype=np.float64,
    )


    x = x - np.mean(
        x,
        axis=0,
        keepdims=True,
    )


    std = np.std(
        x,
        axis=0,
        keepdims=True,
    )


    valid = (
        std[0]
        > 1e-4
    )


    if np.sum(valid) < 4:

        return (
            0.0,
            float("nan"),
        )


    x = (
        x[:, valid]
        / std[:, valid]
    )


    min_lag = max(
        2,
        int(
            round(
                0.55 * fps
            )
        ),
    )

    max_lag = min(
        len(x) // 2,
        int(
            round(
                1.60 * fps
            )
        ),
    )


    if max_lag <= min_lag:

        return (
            0.0,
            float("nan"),
        )


    best_corr = -1.0
    best_lag = None


    for lag in range(
        min_lag,
        max_lag + 1,
    ):

        a = x[:-lag]
        b = x[lag:]


        corr = float(
            np.mean(
                a * b
            )
        )


        if corr > best_corr:

            best_corr = corr
            best_lag = lag


    period = (
        best_lag / fps
        if best_lag
        else float("nan")
    )


    return (
        best_corr,
        period,
    )


def analyze(
    name,
    path,
):

    d = np.load(
        path,
        allow_pickle=True,
    )


    required = [
        "dof_names",
        "dof_positions",
        "body_names",
        "body_positions",
        "body_rotations",
        "fps",
    ]


    for key in required:

        if key not in d.files:

            raise RuntimeError(
                f"{name}: missing {key}"
            )


    q = np.asarray(
        d["dof_positions"],
        dtype=np.float64,
    )


    if (
        q.ndim != 2
        or q.shape[1] != 29
    ):

        raise RuntimeError(
            f"{name}: expected "
            f"(T,29), got {q.shape}"
        )


    dof_names = [
        str(x)
        for x in d["dof_names"]
    ]


    body_names = [
        str(x)
        for x in d["body_names"]
    ]


    body_pos = np.asarray(
        d["body_positions"],
        dtype=np.float64,
    )

    body_rot = np.asarray(
        d["body_rotations"],
        dtype=np.float64,
    )


    fps = float(
        np.asarray(
            d["fps"]
        ).reshape(-1)[0]
    )


    n = q.shape[0]

    duration = (
        (n - 1)
        / fps
    )


    pelvis = body_names.index(
        "pelvis"
    )

    left_foot = body_names.index(
        "left_ankle_roll_link"
    )

    right_foot = body_names.index(
        "right_ankle_roll_link"
    )


    root = body_pos[
        :,
        pelvis,
        :
    ]


    delta_xy = (
        root[-1, :2]
        - root[0, :2]
    )


    net_xy = float(
        np.linalg.norm(
            delta_xy
        )
    )


    theta = math.atan2(
        delta_xy[1],
        delta_xy[0],
    )


    c = math.cos(theta)
    s = math.sin(theta)


    relative_xy = (
        root[:, :2]
        - root[0, :2]
    )


    # Rotate net travel onto +X.
    aligned_x = (
        c * relative_xy[:, 0]
        + s * relative_xy[:, 1]
    )

    aligned_y = (
        -s * relative_xy[:, 0]
        + c * relative_xy[:, 1]
    )


    forward = float(
        aligned_x[-1]
    )


    lateral_rms = float(
        np.sqrt(
            np.mean(
                aligned_y ** 2
            )
        )
    )


    lateral_max = float(
        np.max(
            np.abs(
                aligned_y
            )
        )
    )


    root_diff = np.diff(
        root[:, :2],
        axis=0,
    )


    path_length = float(
        np.sum(
            np.linalg.norm(
                root_diff,
                axis=1,
            )
        )
    )


    straightness = (
        net_xy
        / max(
            path_length,
            1e-9,
        )
    )


    speed = np.linalg.norm(
        root_diff,
        axis=1,
    ) * fps


    mean_speed = float(
        np.mean(speed)
    )


    # ---------------------------------------------
    # Beginning/end steady-walking quality.
    # ---------------------------------------------

    m = len(speed)

    edge = max(
        3,
        int(
            round(
                0.12 * m
            )
        ),
    )


    middle_start = edge
    middle_end = max(
        middle_start + 1,
        m - edge,
    )


    first_speed = float(
        np.mean(
            speed[:edge]
        )
    )

    last_speed = float(
        np.mean(
            speed[-edge:]
        )
    )

    mid_speed = float(
        np.mean(
            speed[
                middle_start:
                middle_end
            ]
        )
    )


    if mid_speed > 0.05:

        boundary_stability = (
            1.0
            - 0.5
            * (
                abs(
                    first_speed
                    / mid_speed
                    - 1.0
                )
                +
                abs(
                    last_speed
                    / mid_speed
                    - 1.0
                )
            )
        )

    else:

        boundary_stability = 0.0


    boundary_stability = clip01(
        boundary_stability
    )


    # ---------------------------------------------
    # Pelvis heading stability relative to travel.
    # ---------------------------------------------

    yaw = qyaw_wxyz(
        body_rot[
            :,
            pelvis,
            :
        ]
    )


    yaw_rel = np.unwrap(
        yaw - theta
    )


    # Remove whole multiples of 2*pi.
    yaw_rel = (
        yaw_rel
        - round(
            float(
                np.mean(
                    yaw_rel
                )
                / (
                    2.0
                    * math.pi
                )
            )
        )
        * (
            2.0
            * math.pi
        )
    )


    yaw_range_deg = math.degrees(
        float(
            np.max(
                yaw_rel
            )
            - np.min(
                yaw_rel
            )
        )
    )


    yaw_drift_deg = math.degrees(
        float(
            yaw_rel[-1]
            - yaw_rel[0]
        )
    )


    mean_heading_error_deg = math.degrees(
        float(
            np.mean(
                np.abs(
                    np.arctan2(
                        np.sin(
                            yaw_rel
                        ),
                        np.cos(
                            yaw_rel
                        ),
                    )
                )
            )
        )
    )


    # ---------------------------------------------
    # Gait periodicity.
    # ---------------------------------------------

    periodicity, gait_period = (
        periodicity_metric(
            q,
            fps,
        )
    )


    if (
        np.isfinite(
            gait_period
        )
        and gait_period > 0
    ):

        cycles = (
            duration
            / gait_period
        )

    else:

        cycles = 0.0


    # ---------------------------------------------
    # Foot-contact proxy using retargeted world
    # ankle positions.
    #
    # This is only for candidate ranking.
    # Final chosen clip will later use exact
    # MuJoCo sole geometry and V4 grounding.
    # ---------------------------------------------

    lf = body_pos[
        :,
        left_foot,
        :
    ]

    rf = body_pos[
        :,
        right_foot,
        :
    ]


    lf_ground = float(
        np.percentile(
            lf[:, 2],
            5,
        )
    )

    rf_ground = float(
        np.percentile(
            rf[:, 2],
            5,
        )
    )


    lf_clear = (
        lf[:, 2]
        - lf_ground
    )

    rf_clear = (
        rf[:, 2]
        - rf_ground
    )


    lf_speed = np.zeros(
        n,
        dtype=np.float64,
    )

    rf_speed = np.zeros(
        n,
        dtype=np.float64,
    )


    lf_speed[1:] = (
        np.linalg.norm(
            np.diff(
                lf[:, :2],
                axis=0,
            ),
            axis=1,
        )
        * fps
    )

    rf_speed[1:] = (
        np.linalg.norm(
            np.diff(
                rf[:, :2],
                axis=0,
            ),
            axis=1,
        )
        * fps
    )


    contact_height = 0.025


    left_low = (
        lf_clear
        <= contact_height
    )

    right_low = (
        rf_clear
        <= contact_height
    )


    left_support = (
        left_low
        & (
            lf_speed
            <= 0.35
        )
    )

    right_support = (
        right_low
        & (
            rf_speed
            <= 0.35
        )
    )


    support_coverage = float(
        np.mean(
            left_support
            | right_support
        )
    )


    contact_coverage = float(
        np.mean(
            left_low
            | right_low
        )
    )


    double_support = float(
        np.mean(
            left_support
            & right_support
        )
    )


    left_support_count = int(
        np.sum(
            left_support
        )
    )

    right_support_count = int(
        np.sum(
            right_support
        )
    )


    support_symmetry = (
        min(
            left_support_count,
            right_support_count,
        )
        /
        max(
            left_support_count,
            right_support_count,
            1,
        )
    )


    stance_speeds = np.concatenate(
        [
            lf_speed[
                left_low
            ],
            rf_speed[
                right_low
            ],
        ]
    )


    if len(
        stance_speeds
    ):

        stance_slip = float(
            np.median(
                stance_speeds
            )
        )

    else:

        stance_slip = 999.0


    left_clear95 = float(
        np.percentile(
            lf_clear,
            95,
        )
    )

    right_clear95 = float(
        np.percentile(
            rf_clear,
            95,
        )
    )


    clearance_symmetry = (
        min(
            left_clear95,
            right_clear95,
        )
        /
        max(
            left_clear95,
            right_clear95,
            1e-6,
        )
    )


    # ---------------------------------------------
    # SCORE / 100
    # ---------------------------------------------

    duration_score = (
        8.0
        * clip01(
            duration
            / 6.0
        )
    )


    travel_score = (
        8.0
        * clip01(
            forward
            / 3.0
        )
    )


    straight_score = (
        12.0
        * clip01(
            (
                straightness
                - 0.85
            )
            / 0.15
        )
    )


    lateral_score = (
        8.0
        * clip01(
            1.0
            - (
                lateral_rms
                / 0.15
            )
        )
    )


    yaw_score = (
        8.0
        * clip01(
            1.0
            - (
                yaw_range_deg
                / 30.0
            )
        )
    )


    periodicity_score = (
        20.0
        * clip01(
            (
                periodicity
                - 0.35
            )
            / 0.55
        )
    )


    cycles_score = (
        6.0
        * clip01(
            (
                cycles
                - 1.5
            )
            / 3.5
        )
    )


    support_score = (
        10.0
        * clip01(
            (
                support_coverage
                - 0.45
            )
            / 0.50
        )
    )


    slip_score = (
        6.0
        * clip01(
            1.0
            - (
                stance_slip
                / 0.30
            )
        )
    )


    symmetry_score = (
        6.0
        * clip01(
            0.5
            * (
                support_symmetry
                + clearance_symmetry
            )
        )
    )


    boundary_score = (
        8.0
        * boundary_stability
    )


    total_score = (
        duration_score
        + travel_score
        + straight_score
        + lateral_score
        + yaw_score
        + periodicity_score
        + cycles_score
        + support_score
        + slip_score
        + symmetry_score
        + boundary_score
    )


    return {
        "name":
            name,

        "file":
            str(path),

        "frames":
            n,

        "fps":
            fps,

        "duration_s":
            duration,

        "forward_m":
            forward,

        "path_m":
            path_length,

        "straightness":
            straightness,

        "mean_speed":
            mean_speed,

        "lateral_rms_m":
            lateral_rms,

        "lateral_max_m":
            lateral_max,

        "yaw_range_deg":
            yaw_range_deg,

        "yaw_drift_deg":
            yaw_drift_deg,

        "heading_error_deg":
            mean_heading_error_deg,

        "periodicity":
            periodicity,

        "gait_period_s":
            gait_period,

        "cycles":
            cycles,

        "contact_coverage":
            contact_coverage,

        "support_coverage":
            support_coverage,

        "double_support":
            double_support,

        "support_symmetry":
            support_symmetry,

        "stance_slip_mps":
            stance_slip,

        "left_swing_clear_m":
            left_clear95,

        "right_swing_clear_m":
            right_clear95,

        "clearance_symmetry":
            clearance_symmetry,

        "boundary_stability":
            boundary_stability,

        "score":
            total_score,
    }


# ================================================================
# DOWNLOAD
# ================================================================

print("=" * 170)
print("DOWNLOADING 29-DOF G1 NORMAL-WALK CANDIDATES")
print("=" * 170)


local_files = []


for name, relative_path in CANDIDATES:

    destination = (
        DOWNLOAD_DIR
        / (
            name
            + ".npz"
        )
    )


    download_file(
        relative_path,
        destination,
    )


    local_files.append(
        (
            name,
            relative_path,
            destination,
        )
    )


# ================================================================
# ANALYZE
# ================================================================

print()
print("=" * 170)
print("ANALYZING CANDIDATES")
print("=" * 170)


results = []


for (
    name,
    relative_path,
    destination,
) in local_files:

    r = analyze(
        name,
        destination,
    )


    r[
        "source_path"
    ] = relative_path


    results.append(
        r
    )


results.sort(
    key=lambda x:
        x["score"],
    reverse=True,
)


# ================================================================
# TABLE
# ================================================================

print()
print("=" * 190)
print("29-DOF G1 WALKING DATASET RANKING")
print("=" * 190)

print(
    f"{'RANK':>4s} "
    f"{'CANDIDATE':16s} "
    f"{'SCORE':>6s} "
    f"{'SEC':>6s} "
    f"{'FWD':>7s} "
    f"{'SPEED':>7s} "
    f"{'STRAIGHT':>8s} "
    f"{'LAT-RMS':>8s} "
    f"{'YAW-R':>7s} "
    f"{'PERIOD':>7s} "
    f"{'GAIT-T':>7s} "
    f"{'CYCLES':>7s} "
    f"{'SUPPORT':>8s} "
    f"{'SLIP':>7s} "
    f"{'SYM':>6s} "
    f"{'BOUND':>6s}"
)

print("-" * 190)


for rank, r in enumerate(
    results,
    start=1,
):

    print(
        f"{rank:4d} "
        f"{r['name']:16s} "
        f"{r['score']:6.1f} "
        f"{r['duration_s']:6.2f} "
        f"{r['forward_m']:7.2f} "
        f"{r['mean_speed']:7.2f} "
        f"{r['straightness']:8.3f} "
        f"{r['lateral_rms_m']:8.3f} "
        f"{r['yaw_range_deg']:7.1f} "
        f"{r['periodicity']:7.3f} "
        f"{r['gait_period_s']:7.3f} "
        f"{r['cycles']:7.2f} "
        f"{r['support_coverage']:8.3f} "
        f"{r['stance_slip_mps']:7.3f} "
        f"{r['support_symmetry']:6.3f} "
        f"{r['boundary_stability']:6.3f}"
    )


# ================================================================
# SAVE CSV
# ================================================================

with RESULT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            results[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ================================================================
# SELECT WINNER
# ================================================================

best = results[0]

best_source = Path(
    best["file"]
)

selected_npz = (
    SELECTED_DIR
    / "best_29dof_normal_walk_raw.npz"
)


shutil.copy2(
    best_source,
    selected_npz,
)


selection_txt = (
    ROOT
    / "results"
    / "g1_29dof_walk_selection.txt"
)


selection_txt.write_text(
    "\n".join(
        [
            f"winner={best['name']}",
            f"score={best['score']:.3f}",
            f"source={best['source_path']}",
            f"local={selected_npz}",
            f"duration_s={best['duration_s']:.3f}",
            f"forward_m={best['forward_m']:.3f}",
            f"straightness={best['straightness']:.5f}",
            f"lateral_rms_m={best['lateral_rms_m']:.5f}",
            f"yaw_range_deg={best['yaw_range_deg']:.3f}",
            f"periodicity={best['periodicity']:.5f}",
            f"gait_period_s={best['gait_period_s']:.5f}",
            f"cycles={best['cycles']:.3f}",
            f"support_coverage={best['support_coverage']:.5f}",
            f"stance_slip_mps={best['stance_slip_mps']:.5f}",
            f"support_symmetry={best['support_symmetry']:.5f}",
            f"boundary_stability={best['boundary_stability']:.5f}",
        ]
    ),
    encoding="utf-8",
)


print()
print("=" * 190)
print("SELECTED 29-DOF WALK")
print("=" * 190)

print(
    "winner:",
    best["name"],
)

print(
    "score:",
    f"{best['score']:.1f}/100",
)

print(
    "source:",
    best["source_path"],
)

print(
    "duration:",
    f"{best['duration_s']:.2f}s",
)

print(
    "forward travel:",
    f"{best['forward_m']:.2f}m",
)

print(
    "mean speed:",
    f"{best['mean_speed']:.2f}m/s",
)

print(
    "straightness:",
    f"{best['straightness']:.3f}",
)

print(
    "lateral RMS:",
    f"{best['lateral_rms_m']:.3f}m",
)

print(
    "yaw range:",
    f"{best['yaw_range_deg']:.2f}deg",
)

print(
    "gait periodicity:",
    f"{best['periodicity']:.3f}",
)

print(
    "gait period:",
    f"{best['gait_period_s']:.3f}s",
)

print(
    "estimated complete cycles:",
    f"{best['cycles']:.2f}",
)

print(
    "support coverage:",
    f"{100*best['support_coverage']:.1f}%",
)

print(
    "stance-foot slip proxy:",
    f"{best['stance_slip_mps']:.3f}m/s",
)

print(
    "left/right support symmetry:",
    f"{best['support_symmetry']:.3f}",
)

print(
    "boundary walking stability:",
    f"{best['boundary_stability']:.3f}",
)

print()
print(
    "selected copy:",
    selected_npz,
)

print(
    "CSV:",
    RESULT_CSV,
)

print(
    "selection record:",
    selection_txt,
)


print()
print("=" * 190)


if (
    best["score"] >= 75.0
    and best["periodicity"] >= 0.65
    and best["straightness"] >= 0.93
    and best["cycles"] >= 3.0
):

    print(
        "DATASET DECISION: EXCELLENT RESTART REFERENCE"
    )

elif (
    best["score"] >= 60.0
    and best["periodicity"] >= 0.50
):

    print(
        "DATASET DECISION: GOOD RESTART REFERENCE"
    )

else:

    print(
        "DATASET DECISION: NO CANDIDATE IS CLEAN ENOUGH"
    )

    print(
        "Do not start training yet; expand the search pool."
    )


print("=" * 190)
print("NO PPO TRAINING WAS PERFORMED.")
print("=" * 190)
