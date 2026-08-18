from pathlib import Path
from urllib.request import Request, urlopen

import csv
import math
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = (
    ROOT
    / "datasets"
    / "candidate_29dof_walks_kit"
)

CSV_PATH = (
    ROOT
    / "results"
    / "g1_kit205_forward_walk_ranking.csv"
)

REPO = (
    "https://huggingface.co/datasets/"
    "ember-lab-berkeley/"
    "AMASS_Retargeted_for_G1"
)

REVISION = "main"


# ================================================================
# 20 explicit walking candidates.
# No running.
# No turning.
# ================================================================

CANDIDATES = []


for i in range(1, 11):

    CANDIDATES.append(
        (
            f"medium_{i:02d}",

            (
                "g1/KIT/205/"
                f"walking_medium{i:02d}"
                "_poses_100_jpos.npz"
            ),
        )
    )


for i in range(1, 11):

    CANDIDATES.append(
        (
            f"slow_{i:02d}",

            (
                "g1/KIT/205/"
                f"walking_slow{i:02d}"
                "_poses_100_jpos.npz"
            ),
        )
    )


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# DOWNLOAD
# ================================================================

def download(
    relative,
    destination,
):

    if destination.exists():

        try:

            d = np.load(
                destination,
                allow_pickle=True,
            )

            if (
                "dof_positions" in d.files
                and
                "body_positions" in d.files
            ):

                print(
                    "cached:",
                    destination.name,
                )

                return

        except Exception:
            pass


    url = (
        f"{REPO}/resolve/"
        f"{REVISION}/"
        f"{relative}"
        "?download=true"
    )


    print(
        "downloading:",
        relative,
    )


    request = Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0"
        },
    )


    with urlopen(
        request,
        timeout=120,
    ) as response:

        raw = response.read()


    destination.write_bytes(
        raw
    )


    # Real-file verification.
    d = np.load(
        destination,
        allow_pickle=True,
    )


    if "dof_positions" not in d.files:

        raise RuntimeError(
            f"Invalid NPZ: {destination}"
        )


# ================================================================
# QUAT → YAW
# ================================================================

def yaw_wxyz(
    quat,
):

    q = np.asarray(
        quat,
        dtype=np.float64,
    )


    q /= np.maximum(
        np.linalg.norm(
            q,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )


    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]


    return np.unwrap(
        np.arctan2(
            2.0 * (
                w*z + x*y
            ),
            1.0
            - 2.0 * (
                y*y + z*z
            ),
        )
    )


# ================================================================
# GAIT PERIODICITY
#
# Uses lower-body DOFs only.
# ================================================================

def periodicity(
    q,
    fps,
):

    legs = np.asarray(
        q[:, :12],
        dtype=np.float64,
    )


    legs = (
        legs
        - np.mean(
            legs,
            axis=0,
            keepdims=True,
        )
    )


    std = np.std(
        legs,
        axis=0,
        keepdims=True,
    )


    valid = (
        std[0] > 1e-4
    )


    if np.sum(valid) < 4:

        return (
            0.0,
            float("nan"),
        )


    legs = (
        legs[:, valid]
        / std[:, valid]
    )


    # Human normal walking:
    # broad cycle-search range.
    min_lag = int(
        round(
            0.55 * fps
        )
    )

    max_lag = min(
        int(
            round(
                1.60 * fps
            )
        ),
        len(legs) // 2,
    )


    best = -999.0
    best_lag = None


    for lag in range(
        min_lag,
        max_lag + 1,
    ):

        a = legs[:-lag]
        b = legs[lag:]


        corr = float(
            np.mean(
                a * b
            )
        )


        if corr > best:

            best = corr
            best_lag = lag


    if best_lag is None:

        return (
            0.0,
            float("nan"),
        )


    return (
        best,
        best_lag / fps,
    )


def clip01(x):

    return float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )


# ================================================================
# ANALYSIS
# ================================================================

def analyze(
    name,
    path,
    relative,
):

    d = np.load(
        path,
        allow_pickle=True,
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
            f"{name}: not a 29-DOF motion: "
            f"{q.shape}"
        )


    body_names = [
        str(x)
        for x in d["body_names"]
    ]


    if "pelvis" not in body_names:

        raise RuntimeError(
            f"{name}: pelvis missing."
        )


    pelvis_id = (
        body_names.index(
            "pelvis"
        )
    )


    pos = np.asarray(
        d["body_positions"],
        dtype=np.float64,
    )[
        :,
        pelvis_id,
        :
    ]


    quat = np.asarray(
        d["body_rotations"],
        dtype=np.float64,
    )[
        :,
        pelvis_id,
        :
    ]


    fps = float(
        np.asarray(
            d["fps"]
        ).reshape(-1)[0]
    )


    n = len(q)


    duration = (
        (n - 1)
        / fps
    )


    # ============================================================
    # ROOT TRAJECTORY
    # ============================================================

    xy = (
        pos[:, :2]
        - pos[0, :2]
    )


    delta = (
        xy[-1]
        - xy[0]
    )


    net = float(
        np.linalg.norm(
            delta
        )
    )


    if net < 1e-8:

        heading = 0.0

    else:

        heading = math.atan2(
            delta[1],
            delta[0],
        )


    c = math.cos(
        heading
    )

    s = math.sin(
        heading
    )


    # Rotate net direction onto +X.
    x = (
        c * xy[:, 0]
        +
        s * xy[:, 1]
    )


    y = (
        -s * xy[:, 0]
        +
        c * xy[:, 1]
    )


    forward = float(
        x[-1]
    )


    # ============================================================
    # PATH LENGTH
    # ============================================================

    diff = np.diff(
        xy,
        axis=0,
    )


    step_distance = np.linalg.norm(
        diff,
        axis=1,
    )


    path_length = float(
        np.sum(
            step_distance
        )
    )


    straightness = (
        forward
        /
        max(
            path_length,
            1e-9,
        )
    )


    mean_speed = (
        path_length
        / max(
            duration,
            1e-9,
        )
    )


    net_speed = (
        forward
        / max(
            duration,
            1e-9,
        )
    )


    lateral_rms = float(
        np.sqrt(
            np.mean(
                y ** 2
            )
        )
    )


    lateral_max = float(
        np.max(
            np.abs(
                y
            )
        )
    )


    # ============================================================
    # FORWARD MONOTONICITY
    #
    # Percentage of frames whose displacement is broadly forward.
    # ============================================================

    dx = np.diff(
        x
    )


    forward_fraction = float(
        np.mean(
            dx > -0.001
        )
    )


    backward_distance = float(
        -np.sum(
            np.minimum(
                dx,
                0.0,
            )
        )
    )


    # ============================================================
    # PELVIS YAW
    # ============================================================

    yaw = yaw_wxyz(
        quat
    )


    # Difference from trajectory heading.
    yaw_rel = (
        yaw
        - heading
    )


    yaw_rel = np.unwrap(
        yaw_rel
    )


    # Remove whole-turn offset.
    yaw_rel -= (
        round(
            float(
                np.mean(
                    yaw_rel
                )
                /
                (
                    2.0
                    * math.pi
                )
            )
        )
        *
        (
            2.0
            * math.pi
        )
    )


    yaw_range = math.degrees(
        float(
            np.ptp(
                yaw_rel
            )
        )
    )


    yaw_drift = math.degrees(
        float(
            yaw_rel[-1]
            - yaw_rel[0]
        )
    )


    heading_error = math.degrees(
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


    # ============================================================
    # GAIT PERIOD
    # ============================================================

    periodic, period = (
        periodicity(
            q,
            fps,
        )
    )


    cycles = (

        duration / period

        if (
            np.isfinite(
                period
            )
            and period > 0.0
        )

        else 0.0
    )


    # ============================================================
    # START / END CONSISTENCY
    # ============================================================

    velocity = (
        step_distance
        * fps
    )


    edge = max(
        3,
        int(
            0.15 * len(
                velocity
            )
        ),
    )


    first_speed = float(
        np.mean(
            velocity[:edge]
        )
    )


    last_speed = float(
        np.mean(
            velocity[-edge:]
        )
    )


    middle = velocity[
        edge:
        max(
            edge + 1,
            len(velocity)
            - edge,
        )
    ]


    middle_speed = float(
        np.mean(
            middle
        )
    )


    if middle_speed > 0.05:

        boundary = (
            1.0
            -
            0.5
            * (
                abs(
                    first_speed
                    /
                    middle_speed
                    - 1.0
                )
                +
                abs(
                    last_speed
                    /
                    middle_speed
                    - 1.0
                )
            )
        )

    else:

        boundary = 0.0


    boundary = clip01(
        boundary
    )


    # ============================================================
    # HARD ELIGIBILITY
    #
    # We want ACTUAL forward locomotion.
    # ============================================================

    eligible = (
        duration >= 3.0
        and
        forward >= 1.0
        and
        net_speed >= 0.25
        and
        straightness >= 0.75
        and
        forward_fraction >= 0.80
        and
        periodic >= 0.45
        and
        cycles >= 2.0
    )


    # ============================================================
    # SCORE / 100
    # ============================================================

    duration_score = (
        8.0
        * clip01(
            duration
            / 6.0
        )
    )


    forward_score = (
        18.0
        * clip01(
            forward
            / 4.0
        )
    )


    speed_score = (
        8.0
        * clip01(
            (
                net_speed
                - 0.20
            )
            / 0.80
        )
    )


    straight_score = (
        16.0
        * clip01(
            (
                straightness
                - 0.70
            )
            / 0.30
        )
    )


    monotonic_score = (
        8.0
        * clip01(
            (
                forward_fraction
                - 0.70
            )
            / 0.30
        )
    )


    lateral_score = (
        7.0
        * clip01(
            1.0
            -
            lateral_rms
            / 0.25
        )
    )


    yaw_score = (
        7.0
        * clip01(
            1.0
            -
            yaw_range
            / 35.0
        )
    )


    periodicity_score = (
        16.0
        * clip01(
            (
                periodic
                - 0.35
            )
            / 0.60
        )
    )


    cycle_score = (
        6.0
        * clip01(
            (
                cycles
                - 1.5
            )
            / 3.5
        )
    )


    boundary_score = (
        6.0
        * boundary
    )


    score = (
        duration_score
        + forward_score
        + speed_score
        + straight_score
        + monotonic_score
        + lateral_score
        + yaw_score
        + periodicity_score
        + cycle_score
        + boundary_score
    )


    # In-place clips cannot win simply because
    # their leg motion is periodic.
    if forward < 0.50:

        score *= 0.25


    elif forward < 1.00:

        score *= 0.60


    return {

        "name":
            name,

        "source":
            relative,

        "frames":
            n,

        "fps":
            fps,

        "duration":
            duration,

        "forward":
            forward,

        "path":
            path_length,

        "net_speed":
            net_speed,

        "mean_speed":
            mean_speed,

        "straightness":
            straightness,

        "forward_fraction":
            forward_fraction,

        "backward_distance":
            backward_distance,

        "lateral_rms":
            lateral_rms,

        "lateral_max":
            lateral_max,

        "yaw_range":
            yaw_range,

        "yaw_drift":
            yaw_drift,

        "heading_error":
            heading_error,

        "periodicity":
            periodic,

        "period":
            period,

        "cycles":
            cycles,

        "boundary":
            boundary,

        "eligible":
            eligible,

        "score":
            score,
    }


# ================================================================
# DOWNLOAD ALL
# ================================================================

print("=" * 180)
print("KIT/205 29-DOF FORWARD-WALK SEARCH")
print("MEDIUM + SLOW WALKING")
print("NO TRAINING")
print("=" * 180)


local = []


for name, relative in CANDIDATES:

    path = (
        OUT_DIR
        / (
            name
            + ".npz"
        )
    )


    download(
        relative,
        path,
    )


    local.append(
        (
            name,
            relative,
            path,
        )
    )


# ================================================================
# ANALYZE
# ================================================================

results = []


for (
    name,
    relative,
    path,
) in local:

    results.append(
        analyze(
            name,
            path,
            relative,
        )
    )


results.sort(
    key=lambda r: (
        r["eligible"],
        r["score"],
        r["forward"],
    ),
    reverse=True,
)


# ================================================================
# TABLE
# ================================================================

print()
print("=" * 205)
print("KIT/205 FORWARD WALK RANKING")
print("=" * 205)

print(
    f"{'#':>2s} "
    f"{'NAME':12s} "
    f"{'OK':>3s} "
    f"{'SCORE':>6s} "
    f"{'SEC':>6s} "
    f"{'FWD':>7s} "
    f"{'NETV':>6s} "
    f"{'PATH':>7s} "
    f"{'STR':>6s} "
    f"{'FWD%':>6s} "
    f"{'BACK':>6s} "
    f"{'LAT':>6s} "
    f"{'YAW-R':>7s} "
    f"{'PER':>6s} "
    f"{'GAIT':>6s} "
    f"{'CYC':>5s} "
    f"{'BOUND':>6s}"
)

print("-" * 205)


for rank, r in enumerate(
    results,
    1,
):

    print(
        f"{rank:2d} "
        f"{r['name']:12s} "
        f"{str(r['eligible']):>3s} "
        f"{r['score']:6.1f} "
        f"{r['duration']:6.2f} "
        f"{r['forward']:7.2f} "
        f"{r['net_speed']:6.2f} "
        f"{r['path']:7.2f} "
        f"{r['straightness']:6.3f} "
        f"{100*r['forward_fraction']:5.1f}% "
        f"{r['backward_distance']:6.2f} "
        f"{r['lateral_rms']:6.3f} "
        f"{r['yaw_range']:7.1f} "
        f"{r['periodicity']:6.3f} "
        f"{r['period']:6.3f} "
        f"{r['cycles']:5.2f} "
        f"{r['boundary']:6.3f}"
    )


# ================================================================
# CSV
# ================================================================

with CSV_PATH.open(
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
# FINAL DECISION
# ================================================================

eligible = [
    r
    for r in results
    if r["eligible"]
]


print()
print("=" * 180)
print("FORWARD WALK SELECTION")
print("=" * 180)


if not eligible:

    print(
        "RESULT: NO KIT/205 CLIP PASSED "
        "THE FORWARD-LOCOMOTION FILTER."
    )

    print(
        "Do not train."
    )

    print(
        "Next search should expand to "
        "CMU / ACCAD / other AMASS walking sets."
    )


else:

    print(
        "eligible clips:",
        len(eligible),
    )


    print()


    for rank, r in enumerate(
        eligible[:3],
        1,
    ):

        print(
            f"TOP {rank}: "
            f"{r['name']}"
        )

        print(
            f"  score       : "
            f"{r['score']:.1f}/100"
        )

        print(
            f"  source      : "
            f"{r['source']}"
        )

        print(
            f"  duration    : "
            f"{r['duration']:.2f}s"
        )

        print(
            f"  forward     : "
            f"{r['forward']:.2f}m"
        )

        print(
            f"  net speed   : "
            f"{r['net_speed']:.2f}m/s"
        )

        print(
            f"  straightness: "
            f"{r['straightness']:.3f}"
        )

        print(
            f"  periodicity : "
            f"{r['periodicity']:.3f}"
        )

        print(
            f"  gait period : "
            f"{r['period']:.3f}s"
        )

        print(
            f"  cycles      : "
            f"{r['cycles']:.2f}"
        )

        print(
            f"  yaw range   : "
            f"{r['yaw_range']:.1f}deg"
        )

        print(
            f"  lateral RMS : "
            f"{r['lateral_rms']:.3f}m"
        )

        print()


    winner = eligible[0]


    print(
        "PROVISIONAL WINNER:",
        winner["name"],
    )


    print(
        "Do NOT train yet."
    )

    print(
        "Next: exact MuJoCo sole/contact/"
        "grounding validation of top 3."
    )


print()
print(
    "CSV:",
    CSV_PATH,
)

print()
print(
    "NO PPO TRAINING WAS PERFORMED."
)

print("=" * 180)
