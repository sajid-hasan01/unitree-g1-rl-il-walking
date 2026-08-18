from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from envs.g1_phase_bc_residual_env_v2 import G1PhaseBCResidualEnvV2


def get_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return str(name) if name is not None else f"unnamed_{obj_id}"


def find_foot_geoms(model: mujoco.MjModel, side: str) -> list[int]:
    """
    Correct side-specific detection.

    Uses body name prefix:
    left_ankle...
    right_ankle...

    Avoids substring markers like "_r", because "_r" wrongly matches "roll".
    """
    assert side in ["left", "right"]

    prefix = f"{side}_"
    allowed_keywords = ["ankle", "foot", "toe", "sole"]

    geom_ids: list[int] = []

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = get_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id).lower()

        if not body_name.startswith(prefix):
            continue

        if any(k in body_name for k in allowed_keywords):
            geom_ids.append(geom_id)

    return sorted(set(geom_ids))


def find_floor_geoms(model: mujoco.MjModel) -> list[int]:
    floor_ids: list[int] = []

    for geom_id in range(model.ngeom):
        geom_name = get_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).lower()
        geom_type = int(model.geom_type[geom_id])

        if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE) or "floor" in geom_name or "ground" in geom_name:
            floor_ids.append(geom_id)

    return sorted(set(floor_ids))


def print_geom_list(model: mujoco.MjModel, title: str, geom_ids: list[int]) -> None:
    print(title)

    if not geom_ids:
        print("  NONE FOUND")
        return

    for geom_id in geom_ids:
        geom_name = get_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = int(model.geom_bodyid[geom_id])
        body_name = get_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        print(f"  geom_id={geom_id:03d} geom={geom_name} body={body_name}")


def geom_centroid_xy(data: mujoco.MjData, geom_ids: list[int]) -> np.ndarray:
    if not geom_ids:
        return np.zeros(2, dtype=np.float64)

    return np.asarray([data.geom_xpos[g, :2] for g in geom_ids], dtype=np.float64).mean(axis=0)


def geom_min_z(data: mujoco.MjData, geom_ids: list[int]) -> float:
    if not geom_ids:
        return float("nan")

    return float(min(float(data.geom_xpos[g, 2]) for g in geom_ids))


def has_foot_floor_contact(
    data: mujoco.MjData,
    foot_geom_ids: list[int],
    floor_geom_ids: list[int],
    contact_margin: float,
) -> tuple[bool, int]:
    foot_set = set(foot_geom_ids)
    floor_set = set(floor_geom_ids)

    count = 0

    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        dist = float(contact.dist)

        if dist > contact_margin:
            continue

        if (g1 in foot_set and g2 in floor_set) or (g2 in foot_set and g1 in floor_set):
            count += 1

    return count > 0, count


def contact_state(left_contact: bool, right_contact: bool) -> str:
    if left_contact and right_contact:
        return "both"
    if left_contact:
        return "left"
    if right_contact:
        return "right"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="experiments/amass_b3_15dof_residual_ppo_v3/models/phase_bc_residual_ppo_v3a_stable_drift_20k.zip",
    )
    parser.add_argument(
        "--out_csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3/reports/foot_contact_diagnostic_fixed_v3a_20k.csv",
    )

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--steps", type=int, default=600)

    parser.add_argument("--start_frame", type=int, default=90)
    parser.add_argument("--gait_scale", type=float, default=0.18)
    parser.add_argument("--residual_scale", type=float, default=0.14)
    parser.add_argument("--target_smoothing", type=float, default=0.07)
    parser.add_argument("--transition_steps", type=int, default=240)
    parser.add_argument("--target_velocity", type=float, default=0.035)

    parser.add_argument("--contact_margin", type=float, default=0.015)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--sleep_time", type=float, default=0.0)

    args = parser.parse_args()

    model_path = Path(args.model)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = G1PhaseBCResidualEnvV2(
        render_mode="human" if args.render else None,
        start_frame=args.start_frame,
        gait_scale=args.gait_scale,
        residual_scale=args.residual_scale,
        target_smoothing=args.target_smoothing,
        transition_steps=args.transition_steps,
        target_velocity=args.target_velocity,
        max_episode_steps=args.steps,
    )

    model = PPO.load(str(model_path), env=env)

    left_geom_ids = find_foot_geoms(env.model, "left")
    right_geom_ids = find_foot_geoms(env.model, "right")
    floor_geom_ids = find_floor_geoms(env.model)

    print("=" * 100)
    print("FIXED FOOT CONTACT DIAGNOSTIC")
    print("=" * 100)
    print("Model:", model_path)
    print("Output CSV:", out_csv)
    print("Steps:", args.steps)
    print("Gait scale:", args.gait_scale)
    print("Residual scale:", args.residual_scale)
    print("Target velocity:", args.target_velocity)
    print("Contact margin:", args.contact_margin)
    print("=" * 100)

    print_geom_list(env.model, "LEFT FOOT GEOMS:", left_geom_ids)
    print_geom_list(env.model, "RIGHT FOOT GEOMS:", right_geom_ids)
    print_geom_list(env.model, "FLOOR GEOMS:", floor_geom_ids)

    obs, info = env.reset()

    previous_left_xy = geom_centroid_xy(env.data, left_geom_ids)
    previous_right_xy = geom_centroid_xy(env.data, right_geom_ids)

    rows: list[dict[str, object]] = []

    left_contact_steps = 0
    right_contact_steps = 0
    both_contact_steps = 0
    no_contact_steps = 0
    left_only_steps = 0
    right_only_steps = 0

    left_contact_switches = 0
    right_contact_switches = 0

    prev_left_contact = False
    prev_right_contact = False

    max_left_z = -1e9
    max_right_z = -1e9

    total_left_slip = 0.0
    total_right_slip = 0.0
    left_slip_count = 0
    right_slip_count = 0

    total_reward = 0.0
    final_info = info

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        final_info = info

        left_z = geom_min_z(env.data, left_geom_ids)
        right_z = geom_min_z(env.data, right_geom_ids)

        max_left_z = max(max_left_z, left_z)
        max_right_z = max(max_right_z, right_z)

        left_contact, left_contact_count = has_foot_floor_contact(
            env.data,
            left_geom_ids,
            floor_geom_ids,
            args.contact_margin,
        )
        right_contact, right_contact_count = has_foot_floor_contact(
            env.data,
            right_geom_ids,
            floor_geom_ids,
            args.contact_margin,
        )

        left_xy = geom_centroid_xy(env.data, left_geom_ids)
        right_xy = geom_centroid_xy(env.data, right_geom_ids)

        dt = max(env.frame_skip * env.sim_dt, 1e-6)

        left_slip_speed = float(np.linalg.norm(left_xy - previous_left_xy) / dt)
        right_slip_speed = float(np.linalg.norm(right_xy - previous_right_xy) / dt)

        previous_left_xy = left_xy.copy()
        previous_right_xy = right_xy.copy()

        if left_contact:
            left_contact_steps += 1
            total_left_slip += left_slip_speed
            left_slip_count += 1

        if right_contact:
            right_contact_steps += 1
            total_right_slip += right_slip_speed
            right_slip_count += 1

        if left_contact and right_contact:
            both_contact_steps += 1
        elif left_contact and not right_contact:
            left_only_steps += 1
        elif right_contact and not left_contact:
            right_only_steps += 1
        else:
            no_contact_steps += 1

        if step > 0:
            if left_contact != prev_left_contact:
                left_contact_switches += 1
            if right_contact != prev_right_contact:
                right_contact_switches += 1

        prev_left_contact = left_contact
        prev_right_contact = right_contact

        state = contact_state(left_contact, right_contact)

        row = {
            "step": step,
            "base_x": float(info["base_x"]),
            "base_y": float(info["base_y"]),
            "base_z": float(info["base_z"]),
            "up_z": float(info["up_z"]),
            "vx": float(info["vx"]),
            "left_z": float(left_z),
            "right_z": float(right_z),
            "left_contact": int(left_contact),
            "right_contact": int(right_contact),
            "left_contact_count": int(left_contact_count),
            "right_contact_count": int(right_contact_count),
            "contact_state": state,
            "left_slip_speed": float(left_slip_speed),
            "right_slip_speed": float(right_slip_speed),
            "action_mean_abs": float(info["action_mean_abs"]),
            "reward": float(reward),
            "terminated": int(terminated),
            "truncated": int(truncated),
        }

        rows.append(row)

        if step % args.print_every == 0 or terminated or truncated:
            print(
                f"step={step:04d} "
                f"x={info['base_x']:+.3f} "
                f"z={info['base_z']:+.3f} "
                f"up_z={info['up_z']:+.3f} "
                f"vx={info['vx']:+.3f} "
                f"Lz={left_z:+.3f} "
                f"Rz={right_z:+.3f} "
                f"Lc={int(left_contact)} "
                f"Rc={int(right_contact)} "
                f"state={state} "
                f"Lslip={left_slip_speed:.3f} "
                f"Rslip={right_slip_speed:.3f} "
                f"fallen={terminated}"
            )

        if args.sleep_time > 0:
            time.sleep(args.sleep_time)

        if terminated or truncated:
            break

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    completed_steps = len(rows)
    mean_left_slip = total_left_slip / max(left_slip_count, 1)
    mean_right_slip = total_right_slip / max(right_slip_count, 1)

    print()
    print("=" * 100)
    print("FIXED FOOT CONTACT DIAGNOSTIC SUMMARY")
    print("=" * 100)
    print("Steps completed:", completed_steps)
    print("Final x:", f"{float(final_info['base_x']):+.3f}")
    print("Final z:", f"{float(final_info['base_z']):+.3f}")
    print("Final up_z:", f"{float(final_info['up_z']):+.3f}")
    print("Total reward:", f"{total_reward:+.3f}")
    print("Terminated:", final_info["terminated"])
    print("Truncated:", final_info["truncated"])
    print()
    print("Left contact steps:", left_contact_steps)
    print("Right contact steps:", right_contact_steps)
    print("Both contact steps:", both_contact_steps)
    print("Left-only contact steps:", left_only_steps)
    print("Right-only contact steps:", right_only_steps)
    print("No contact steps:", no_contact_steps)
    print("Left contact switches:", left_contact_switches)
    print("Right contact switches:", right_contact_switches)
    print("Max left foot z:", f"{max_left_z:+.4f}")
    print("Max right foot z:", f"{max_right_z:+.4f}")
    print("Mean left slip speed while contacting:", f"{mean_left_slip:.4f}")
    print("Mean right slip speed while contacting:", f"{mean_right_slip:.4f}")
    print("CSV saved:", out_csv)
    print("=" * 100)

    env.close()


if __name__ == "__main__":
    main()
