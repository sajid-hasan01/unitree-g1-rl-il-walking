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


EXCLUDE_KEYWORDS = [
    "hand",
    "arm",
    "shoulder",
    "elbow",
    "wrist",
    "head",
]


FOOT_KEYWORDS = [
    "foot",
    "ankle",
    "toe",
    "sole",
]


LEFT_MARKERS = [
    "left",
    "l_",
    "_l",
]


RIGHT_MARKERS = [
    "right",
    "r_",
    "_r",
]


FLOOR_KEYWORDS = [
    "floor",
    "ground",
    "plane",
]


def get_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    if name is None:
        return f"unnamed_{int(obj_id)}"
    return str(name)


def has_any(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(k in lower for k in keywords)


def has_side(text: str, side: str) -> bool:
    lower = text.lower()

    if side == "left":
        return any(marker in lower for marker in LEFT_MARKERS)

    if side == "right":
        return any(marker in lower for marker in RIGHT_MARKERS)

    return False


def is_excluded(text: str) -> bool:
    return has_any(text, EXCLUDE_KEYWORDS)


def find_side_body_ids(model: mujoco.MjModel, side: str) -> set[int]:
    ids: set[int] = set()

    for body_id in range(model.nbody):
        body_name = get_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        if is_excluded(body_name):
            continue

        if has_side(body_name, side) and has_any(body_name, FOOT_KEYWORDS):
            ids.add(body_id)

    return ids


def find_side_geom_ids(model: mujoco.MjModel, side: str) -> list[int]:
    side_body_ids = find_side_body_ids(model, side)
    geom_ids: list[int] = []

    for geom_id in range(model.ngeom):
        geom_name = get_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = int(model.geom_bodyid[geom_id])
        body_name = get_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        name_match = (
            not is_excluded(geom_name)
            and has_side(geom_name, side)
            and has_any(geom_name, FOOT_KEYWORDS)
        )

        body_match = body_id in side_body_ids

        fallback_body_match = (
            not is_excluded(body_name)
            and has_side(body_name, side)
            and has_any(body_name, FOOT_KEYWORDS)
        )

        if name_match or body_match or fallback_body_match:
            geom_ids.append(geom_id)

    return sorted(set(geom_ids))


def find_floor_geom_ids(model: mujoco.MjModel) -> list[int]:
    floor_ids: list[int] = []

    for geom_id in range(model.ngeom):
        geom_name = get_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        geom_type = int(model.geom_type[geom_id])

        is_plane = geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE)
        name_match = has_any(geom_name, FLOOR_KEYWORDS)

        if is_plane or name_match:
            floor_ids.append(geom_id)

    return sorted(set(floor_ids))


def geom_centroid_xy(data: mujoco.MjData, geom_ids: list[int]) -> np.ndarray:
    if not geom_ids:
        return np.zeros(2, dtype=np.float64)

    positions = np.asarray([data.geom_xpos[g, :2] for g in geom_ids], dtype=np.float64)
    return positions.mean(axis=0)


def geom_min_z(data: mujoco.MjData, geom_ids: list[int]) -> float:
    if not geom_ids:
        return float("nan")

    values = [float(data.geom_xpos[g, 2]) for g in geom_ids]
    return float(min(values))


def foot_contact(
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

        if floor_set:
            valid = (g1 in foot_set and g2 in floor_set) or (g2 in foot_set and g1 in floor_set)
        else:
            valid = (g1 in foot_set) or (g2 in foot_set)

        if valid:
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="experiments/amass_b3_15dof_residual_ppo_v3/models/phase_bc_residual_ppo_v3a_stable_drift_20k.zip",
    )
    parser.add_argument(
        "--out_csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3/reports/foot_contact_diagnostic_v3a_20k.csv",
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

    left_geom_ids = find_side_geom_ids(env.model, "left")
    right_geom_ids = find_side_geom_ids(env.model, "right")
    floor_geom_ids = find_floor_geom_ids(env.model)

    print("=" * 100)
    print("FOOT CONTACT DIAGNOSTIC")
    print("=" * 100)
    print("Model:", model_path)
    print("Output CSV:", out_csv)
    print("Steps:", args.steps)
    print("Gait scale:", args.gait_scale)
    print("Residual scale:", args.residual_scale)
    print("Target velocity:", args.target_velocity)
    print("Contact margin:", args.contact_margin)
    print("=" * 100)

    print_geom_list(env.model, "LEFT FOOT GEOM CANDIDATES:", left_geom_ids)
    print_geom_list(env.model, "RIGHT FOOT GEOM CANDIDATES:", right_geom_ids)
    print_geom_list(env.model, "FLOOR GEOM CANDIDATES:", floor_geom_ids)

    if not left_geom_ids or not right_geom_ids:
        print()
        print("WARNING: Foot geom detection is incomplete.")
        print("The script will still run, but contact/height values may be unreliable.")

    obs, info = env.reset()

    previous_left_xy = geom_centroid_xy(env.data, left_geom_ids)
    previous_right_xy = geom_centroid_xy(env.data, right_geom_ids)

    rows: list[dict[str, object]] = []

    left_contact_steps = 0
    right_contact_steps = 0
    both_contact_steps = 0
    no_contact_steps = 0

    left_contact_switches = 0
    right_contact_switches = 0

    prev_left_contact = False
    prev_right_contact = False

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

        left_contact, left_contact_count = foot_contact(
            env.data,
            left_geom_ids,
            floor_geom_ids,
            args.contact_margin,
        )
        right_contact, right_contact_count = foot_contact(
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
        elif not left_contact and not right_contact:
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
            "left_z": left_z,
            "right_z": right_z,
            "left_contact": int(left_contact),
            "right_contact": int(right_contact),
            "left_contact_count": left_contact_count,
            "right_contact_count": right_contact_count,
            "contact_state": state,
            "left_slip_speed": left_slip_speed,
            "right_slip_speed": right_slip_speed,
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
    print("FOOT CONTACT DIAGNOSTIC SUMMARY")
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
    print("No contact steps:", no_contact_steps)
    print("Left contact switches:", left_contact_switches)
    print("Right contact switches:", right_contact_switches)
    print("Mean left slip speed while contacting:", f"{mean_left_slip:.4f}")
    print("Mean right slip speed while contacting:", f"{mean_right_slip:.4f}")
    print("CSV saved:", out_csv)
    print("=" * 100)

    env.close()


if __name__ == "__main__":
    main()
