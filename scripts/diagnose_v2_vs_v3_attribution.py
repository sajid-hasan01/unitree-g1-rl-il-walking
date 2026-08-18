"""
diagnose_v2_vs_v3_attribution.py

Read-only diagnostic. Loads the trained V2 and V3 PPO checkpoints, runs each
through a deterministic rollout in its own environment, and logs per-joint
residual magnitude, root angular velocity, and Y/yaw/up_z at matched steps.

Does NOT modify any existing file. Does NOT retrain anything.

Output: two CSVs (results/attribution_v2.csv, results/attribution_v3.csv)
plus a printed comparison report.
"""

import os
import csv
import importlib
import numpy as np
from stable_baselines3 import PPO

# ======================================================================
# CONFIG -- EDIT THESE THREE THINGS BEFORE RUNNING
# ======================================================================

# 1) Checkpoint paths
MODELS = [
    {
        "name": "V2",
        "checkpoint": r"models\g1_ppo_bc_line_residual_v2_50k.zip",
        "env_module": "envs.g1_bc_line_residual_env_v2",
        "env_class": "G1BCLineResidualEnvV2",
        "env_kwargs": {
            "start_frame": 25,
            "stand_frames": 45,
            "transition_frames": 90,
            "target_smoothing": 0.35,
            "residual_scale": 0.14,
            "residual_ramp_frames": 30,
            "target_velocity": -0.18,
            "max_episode_steps": 600,
        },
    },
    {
        "name": "V3",
        "checkpoint": r"models\g1_ppo_bc_line_residual_v3_50k.zip",
        "env_module": "envs.g1_bc_line_residual_env_v3",
        "env_class": "G1BCLineResidualEnvV3",
        "env_kwargs": {
            "start_frame": 25,
            "stand_frames": 45,
            "transition_frames": 90,
            "target_smoothing": 0.35,
            "residual_scale": 0.14,
            "residual_ramp_frames": 30,
            "target_velocity": -0.18,
            "max_episode_steps": 600,
        },
    },
]

MAX_STEPS = 200
OUTPUT_DIR = "results"

# Joint index -> name, per the 15-DOF ordering used throughout this project
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
]

JOINT_GROUPS = {
    "hip_pitch":  [0, 6],
    "hip_roll":   [1, 7],
    "hip_yaw":    [2, 8],
    "knee":       [3, 9],
    "ankle_pitch": [4, 10],
    "ankle_roll":  [5, 11],
    "waist":       [12, 13, 14],
}


# ======================================================================
# EDIT THIS FUNCTION ONLY IF your env doesn't expose data via env.unwrapped.data
# (standard MuJoCo gym pattern: env.unwrapped.data is an mjData struct with
# .qpos / .qvel). If your env wraps this differently, adjust the three lines
# marked below -- everything else in the script is env-agnostic.
# ======================================================================
def get_root_state(env):
    """Return (up_z, y_pos, yaw_deg, root_ang_vel_xyz) for the current sim step."""
    data = env.unwrapped.data          # <-- EDIT if your env stores mjData elsewhere
    qpos = data.qpos
    qvel = data.qvel

    # Free-joint convention: qpos[0:3]=xyz, qpos[3:7]=quat(w,x,y,z)
    # qvel[0:3]=linear vel, qvel[3:6]=angular vel (body frame)
    y_pos = float(qpos[1])
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]

    # up_z from rotated body z-axis
    up_z = float(1 - 2 * (qx * qx + qy * qy))

    # yaw from quaternion (Z-up convention)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw_deg = float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))

    root_ang_vel = np.array(qvel[3:6], dtype=float)
    return up_z, y_pos, yaw_deg, root_ang_vel


def run_rollout(model_cfg):
    module = importlib.import_module(model_cfg["env_module"])
    env_cls = getattr(module, model_cfg["env_class"])
    env = env_cls(**model_cfg["env_kwargs"])

    policy = PPO.load(model_cfg["checkpoint"])

    obs, _ = env.reset()
    rows = []
    for step in range(MAX_STEPS):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        up_z, y_pos, yaw_deg, root_ang_vel = get_root_state(env)

        row = {
            "step": step,
            "up_z": up_z,
            "y_pos": y_pos,
            "yaw_deg": yaw_deg,
            "root_ang_vel_x": root_ang_vel[0],
            "root_ang_vel_y": root_ang_vel[1],
            "root_ang_vel_z": root_ang_vel[2],
            "max_abs_action": float(np.max(np.abs(action))),
        }
        for i, jname in enumerate(JOINT_NAMES):
            row[f"action_{jname}"] = float(action[i])
        rows.append(row)

        if terminated or truncated:
            break

    env.close()
    return rows


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_stats(rows, window):
    """Mean/max |action| per joint group, over the given step-index window."""
    sub = [r for r in rows if window[0] <= r["step"] <= window[1]]
    if not sub:
        return {}
    out = {}
    for group, idxs in JOINT_GROUPS.items():
        vals = []
        for r in sub:
            for i in idxs:
                vals.append(abs(r[f"action_{JOINT_NAMES[i]}"]))
        out[group] = {"mean": float(np.mean(vals)), "max": float(np.max(vals))}
    return out


def report(results):
    print("\n" + "=" * 70)
    print("SURVIVAL / TRAJECTORY SUMMARY")
    print("=" * 70)
    for name, rows in results.items():
        last = rows[-1]
        max_abs_y = max(abs(r["y_pos"]) for r in rows)
        max_abs_yaw = max(abs(r["yaw_deg"]) for r in rows)
        max_action = max(r["max_abs_action"] for r in rows)
        print(f"{name}: steps={len(rows)}  max|Y|={max_abs_y:.3f}  "
              f"max|yaw|={max_abs_yaw:.1f}  max|action|={max_action:.3f}")

    print("\n" + "=" * 70)
    print("PER-JOINT-GROUP ATTRIBUTION -- matched window step 70-110")
    print("(edit the window below if your V2/V3 runs diverge at a different step)")
    print("=" * 70)
    window = (70, 110)
    stats = {name: group_stats(rows, window) for name, rows in results.items()}
    groups = list(JOINT_GROUPS.keys())
    header = f"{'group':<14}" + "".join(f"{n+' mean':>12}{n+' max':>12}" for n in results)
    print(header)
    for g in groups:
        line = f"{g:<14}"
        for name in results:
            s = stats[name].get(g, {"mean": float('nan'), "max": float('nan')})
            line += f"{s['mean']:>12.4f}{s['max']:>12.4f}"
        print(line)

    print("\n" + "=" * 70)
    print("ROOT ANGULAR VELOCITY (pitch=y-axis, yaw=z-axis) -- same window")
    print("=" * 70)
    for name, rows in results.items():
        sub = [r for r in rows if window[0] <= r["step"] <= window[1]]
        if not sub:
            continue
        pitch = [abs(r["root_ang_vel_y"]) for r in sub]
        yaw_r = [abs(r["root_ang_vel_z"]) for r in sub]
        print(f"{name}: mean|pitch_rate|={np.mean(pitch):.3f}  max|pitch_rate|={np.max(pitch):.3f}  "
              f"mean|yaw_rate|={np.mean(yaw_r):.3f}  max|yaw_rate|={np.max(yaw_r):.3f}")

    print("\nInterpretation guide:")
    print("- If ALL joint groups show similarly suppressed mean/max in V3 vs V2")
    print("  -> global reward-driven conservatism (angular-vel / action-saturation")
    print("     / line-yaw penalties). Fix: loosen those penalty weights, not ankle-roll.")
    print("- If hip_pitch/knee specifically drop much more than ankle_roll in V3")
    print("  -> sagittal stabilization was disproportionately suppressed (hypothesis A).")
    print("- If V3 pitch_rate is HIGHER than V2 despite smaller sagittal action")
    print("  -> V3 isn't correcting sagittal drift in time (also supports hypothesis A).")
    print("- If V3 ankle_roll max is pinned near its scale ceiling while Y still drifts")
    print("  -> the 0.4225 ankle-roll ceiling may genuinely be too low for a retrained")
    print("     policy (hypothesis D), not just an artifact of the earlier ablation.")


if __name__ == "__main__":
    results = {}
    for cfg in MODELS:
        print(f"Running rollout for {cfg['name']}  ({cfg['checkpoint']}) ...")
        rows = run_rollout(cfg)
        results[cfg["name"]] = rows
        csv_path = os.path.join(OUTPUT_DIR, f"attribution_{cfg['name'].lower()}.csv")
        save_csv(rows, csv_path)
        print(f"  -> {len(rows)} steps logged, saved to {csv_path}")

    report(results)
