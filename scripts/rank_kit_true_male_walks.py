from pathlib import Path
import numpy as np

ROOT = Path("/mnt/d/KIT (1)/KIT")

BAD = {
    "turn": 30,
    "back": 30,
    "run": 30,
    "jog": 30,
    "jump": 30,
    "hop": 30,
    "skip": 30,
    "kick": 25,
    "sit": 25,
    "stand": 15,
    "lie": 25,
    "crouch": 25,
    "stair": 25,
    "carry": 25,
    "box": 25,
    "pick": 25,
    "dance": 25,
}

results = []

for p in ROOT.rglob("*_stageii.npz"):

    if "walk" not in p.name.lower():
        continue

    try:
        d = np.load(p, allow_pickle=True)

        if not all(
            k in d.files
            for k in ["gender", "poses", "trans", "mocap_frame_rate"]
        ):
            continue

        gender = np.asarray(d["gender"]).item()

        if isinstance(gender, bytes):
            gender = gender.decode()

        if str(gender).lower().strip() != "male":
            continue

        poses = np.asarray(d["poses"])
        trans = np.asarray(d["trans"], dtype=float)

        if poses.ndim != 2 or poses.shape[1] != 165:
            continue

        fps = float(
            np.asarray(d["mocap_frame_rate"]).reshape(-1)[0]
        )

        if len(trans) < 2 or fps <= 0:
            continue

        duration = (len(trans) - 1) / fps

        xy = trans[:, :2]
        delta = xy[-1] - xy[0]

        displacement = float(np.linalg.norm(delta))

        if displacement < 1.0:
            continue

        speed = displacement / duration

        dxy = np.diff(xy, axis=0)

        path_length = float(
            np.linalg.norm(dxy, axis=1).sum()
        )

        if path_length <= 1e-9:
            continue

        straightness = displacement / path_length

        direction = delta / (displacement + 1e-12)

        lateral_axis = np.array([
            -direction[1],
            direction[0],
        ])

        lateral = (xy - xy[0]) @ lateral_axis

        lateral95 = float(
            np.percentile(np.abs(lateral), 95)
        )

        forward_step = dxy @ direction

        backtracking = float(
            np.mean(forward_step < -0.0005)
        )

        # -------------------------
        # SCORE
        # -------------------------
        score = 100.0

        # Normal walking speed
        if speed < 0.70:
            score -= (0.70 - speed) * 40

        elif speed > 1.20:
            score -= (speed - 1.20) * 45

        # Prefer useful duration
        if duration < 5:
            score -= (5 - duration) * 6

        elif duration > 15:
            score -= duration - 15

        # Prefer straight motion
        score -= max(
            0.0,
            0.99 - straightness,
        ) * 350

        # Low lateral drift
        score -= lateral95 * 45

        # Almost no backward frame movement
        score -= backtracking * 200

        name = p.name.lower()

        penalties = []

        for word, value in BAD.items():
            if word in name:
                score -= value
                penalties.append(word)

        simple = len(penalties) == 0

        if simple:
            score += 12

        results.append({
            "score": score,
            "duration": duration,
            "frames": len(trans),
            "fps": fps,
            "speed": speed,
            "disp": displacement,
            "straight": straightness,
            "lat95": lateral95,
            "back": backtracking,
            "simple": simple,
            "penalties": penalties,
            "path": p,
        })

    except Exception:
        continue


results.sort(
    key=lambda x: x["score"],
    reverse=True,
)

print("=" * 125)
print("BEST TRUE-MALE KIT SMPL-X WALKING MOTIONS")
print("=" * 125)

for i, r in enumerate(results[:30], 1):

    print(
        f"\n#{i:02d} "
        f"SCORE={r['score']:.1f} "
        f"SIMPLE={'YES' if r['simple'] else 'NO'}"
    )

    print(
        f" duration={r['duration']:.2f}s"
        f" frames={r['frames']}"
        f" fps={r['fps']:.0f}"
        f" speed={r['speed']:.3f}m/s"
    )

    print(
        f" displacement={r['disp']:.3f}m"
        f" straightness={r['straight']:.4f}"
        f" lateral95={r['lat95']:.3f}m"
        f" backtracking={100*r['back']:.2f}%"
    )

    print(
        " penalties="
        + (
            ",".join(r["penalties"])
            if r["penalties"]
            else "NONE"
        )
    )

    print(r["path"])

print()
print("TOTAL MALE WALK CANDIDATES:", len(results))
