from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np


def get_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return str(name) if name is not None else f"unnamed_{obj_id}"


def find_foot_geoms(model: mujoco.MjModel, side: str) -> list[int]:
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


def joint_qpos_addresses(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    addresses = []

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

        if joint_id < 0:
            raise RuntimeError(f"Joint not found: {joint_name}")

        addresses.append(int(model.jnt_qposadr[joint_id]))

    return addresses


def apply_joint_positions(qpos: np.ndarray, qpos_addresses: list[int], values: np.ndarray) -> None:
    for value, qadr in zip(values, qpos_addresses):
        qpos[qadr] = float(value)


def geom_min_z(data: mujoco.MjData, geom_ids: list[int]) -> float:
    if not geom_ids:
        return float("nan")

    return float(min(float(data.geom_xpos[g, 2]) for g in geom_ids))


def geom_centroid_xy(data: mujoco.MjData, geom_ids: list[int]) -> np.ndarray:
    if not geom_ids:
        return np.zeros(2, dtype=np.float64)

    return np.asarray([data.geom_xpos[g, :2] for g in geom_ids], dtype=np.float64).mean(axis=0)


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


def extract_windows(flags: np.ndarray) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []

    in_window = False
    start = 0

    for i, value in enumerate(flags):
        if bool(value) and not in_window:
            start = i
            in_window = True

        if in_window and (not bool(value)):
            windows.append((start, i - 1))
            in_window = False

    if in_window:
        windows.append((start, len(flags) - 1))

    return windows


def count_switches(flags: np.ndarray) -> int:
    if len(flags) <= 1:
        return 0

    count = 0

    for i in range(1, len(flags)):
        if bool(flags[i]) != bool(flags[i - 1]):
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="experiments/amass_b3_15dof_il/processed/g1_amass_b3_walk1_il_15dof.npz",
    )
    parser.add_argument(
        "--mujoco_model",
        default="third_party/mujoco_menagerie/unitree_g1/scene.xml",
    )
    parser.add_argument(
        "--out_csv",
        default="experiments/amass_b3_15dof_residual_ppo_v3/reports/amass_b3_reference_foot_schedule.csv",
    )
    parser.add_argument("--contact_margin", type=float, default=0.015)
    parser.add_argument("--swing_clearance_threshold", type=float, default=0.015)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--use_root_z", action="store_true")

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    mujoco_model_path = Path(args.mujoco_model)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not mujoco_model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {mujoco_model_path}")

    dataset = np.load(dataset_path, allow_pickle=True)

    joint_pos_15 = np.asarray(dataset["joint_pos_15"], dtype=np.float64)
    root_positions = np.asarray(dataset["root_positions"], dtype=np.float64)
    joint_names = [str(x) for x in dataset["controlled_joint_names"]]

    model = mujoco.MjModel.from_xml_path(str(mujoco_model_path))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        stand_qpos = model.key_qpos[0].copy()
    else:
        stand_qpos = np.zeros(model.nq, dtype=np.float64)
        stand_qpos[3] = 1.0
        stand_qpos[2] = 0.79

    qpos_addresses = joint_qpos_addresses(model, joint_names)

    left_geoms = find_foot_geoms(model, "left")
    right_geoms = find_foot_geoms(model, "right")
    floor_geoms = find_floor_geoms(model)

    print("=" * 100)
    print("AMASS B3 REFERENCE FOOT-SCHEDULE ANALYSIS")
    print("=" * 100)
    print("Dataset:", dataset_path)
    print("MuJoCo model:", mujoco_model_path)
    print("Output CSV:", out_csv)
    print("Frames:", len(joint_pos_15))
    print("Contact margin:", args.contact_margin)
    print("Swing clearance threshold:", args.swing_clearance_threshold)
    print("=" * 100)

    print_geom_list(model, "LEFT FOOT GEOMS:", left_geoms)
    print_geom_list(model, "RIGHT FOOT GEOMS:", right_geoms)
    print_geom_list(model, "FLOOR GEOMS:", floor_geoms)

    rows: list[dict[str, object]] = []

    left_z_values: list[float] = []
    right_z_values: list[float] = []

    previous_left_xy = None
    previous_right_xy = None

    for frame in range(len(joint_pos_15)):
        qpos = stand_qpos.copy()

        apply_joint_positions(qpos, qpos_addresses, joint_pos_15[frame])

        if frame < len(root_positions):
            qpos[0] = stand_qpos[0] + float(root_positions[frame, 0])
            qpos[1] = stand_qpos[1] + float(root_positions[frame, 1])

            if args.use_root_z:
                qpos[2] = stand_qpos[2] + float(root_positions[frame, 2])
            else:
                qpos[2] = stand_qpos[2]

        data.qpos[:] = qpos
        data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)

        left_z = geom_min_z(data, left_geoms)
        right_z = geom_min_z(data, right_geoms)

        left_contact, left_contact_count = has_foot_floor_contact(
            data,
            left_geoms,
            floor_geoms,
            args.contact_margin,
        )
        right_contact, right_contact_count = has_foot_floor_contact(
            data,
            right_geoms,
            floor_geoms,
            args.contact_margin,
        )

        left_xy = geom_centroid_xy(data, left_geoms)
        right_xy = geom_centroid_xy(data, right_geoms)

        if previous_left_xy is None:
            left_xy_speed = 0.0
        else:
            left_xy_speed = float(np.linalg.norm(left_xy - previous_left_xy))

        if previous_right_xy is None:
            right_xy_speed = 0.0
        else:
            right_xy_speed = float(np.linalg.norm(right_xy - previous_right_xy))

        previous_left_xy = left_xy.copy()
        previous_right_xy = right_xy.copy()

        left_z_values.append(left_z)
        right_z_values.append(right_z)

        row = {
            "frame": frame,
            "root_x": float(data.qpos[0]),
            "root_y": float(data.qpos[1]),
            "root_z": float(data.qpos[2]),
            "left_z": float(left_z),
            "right_z": float(right_z),
            "left_contact": int(left_contact),
            "right_contact": int(right_contact),
            "left_contact_count": int(left_contact_count),
            "right_contact_count": int(right_contact_count),
            "left_xy_step_motion": float(left_xy_speed),
            "right_xy_step_motion": float(right_xy_speed),
        }

        rows.append(row)

        if frame % args.print_every == 0:
            print(
                f"frame={frame:04d} "
                f"root_x={data.qpos[0]:+.3f} "
                f"Lz={left_z:+.4f} "
                f"Rz={right_z:+.4f} "
                f"Lc={int(left_contact)} "
                f"Rc={int(right_contact)} "
                f"Lxy={left_xy_speed:.4f} "
                f"Rxy={right_xy_speed:.4f}"
            )

    left_z_arr = np.asarray(left_z_values, dtype=np.float64)
    right_z_arr = np.asarray(right_z_values, dtype=np.float64)

    left_floor_z = float(np.percentile(left_z_arr, 5))
    right_floor_z = float(np.percentile(right_z_arr, 5))

    left_clearance = left_z_arr - left_floor_z
    right_clearance = right_z_arr - right_floor_z

    left_swing = left_clearance > args.swing_clearance_threshold
    right_swing = right_clearance > args.swing_clearance_threshold

    left_swing_windows = extract_windows(left_swing)
    right_swing_windows = extract_windows(right_swing)

    for i, row in enumerate(rows):
        row["left_clearance"] = float(left_clearance[i])
        row["right_clearance"] = float(right_clearance[i])
        row["left_swing"] = int(left_swing[i])
        row["right_swing"] = int(right_swing[i])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    left_contact_steps = sum(int(r["left_contact"]) for r in rows)
    right_contact_steps = sum(int(r["right_contact"]) for r in rows)
    both_contact_steps = sum(
        1 for r in rows if int(r["left_contact"]) == 1 and int(r["right_contact"]) == 1
    )
    left_only_steps = sum(
        1 for r in rows if int(r["left_contact"]) == 1 and int(r["right_contact"]) == 0
    )
    right_only_steps = sum(
        1 for r in rows if int(r["left_contact"]) == 0 and int(r["right_contact"]) == 1
    )
    no_contact_steps = sum(
        1 for r in rows if int(r["left_contact"]) == 0 and int(r["right_contact"]) == 0
    )

    print()
    print("=" * 100)
    print("AMASS B3 REFERENCE FOOT-SCHEDULE SUMMARY")
    print("=" * 100)
    print("Frames:", len(rows))
    print("Root x start:", f"{float(rows[0]['root_x']):+.3f}")
    print("Root x end:", f"{float(rows[-1]['root_x']):+.3f}")
    print("Root x displacement:", f"{float(rows[-1]['root_x']) - float(rows[0]['root_x']):+.3f}")
    print()
    print("Left contact steps:", left_contact_steps)
    print("Right contact steps:", right_contact_steps)
    print("Both contact steps:", both_contact_steps)
    print("Left-only contact steps:", left_only_steps)
    print("Right-only contact steps:", right_only_steps)
    print("No contact steps:", no_contact_steps)
    print()
    print("Left min z:", f"{float(np.min(left_z_arr)):+.4f}")
    print("Left max z:", f"{float(np.max(left_z_arr)):+.4f}")
    print("Left max clearance:", f"{float(np.max(left_clearance)):+.4f}")
    print("Right min z:", f"{float(np.min(right_z_arr)):+.4f}")
    print("Right max z:", f"{float(np.max(right_z_arr)):+.4f}")
    print("Right max clearance:", f"{float(np.max(right_clearance)):+.4f}")
    print()
    print("Left swing frames:", int(np.sum(left_swing)))
    print("Right swing frames:", int(np.sum(right_swing)))
    print("Left swing switches:", count_switches(left_swing))
    print("Right swing switches:", count_switches(right_swing))
    print("Left swing windows:", left_swing_windows)
    print("Right swing windows:", right_swing_windows)
    print()
    print("CSV saved:", out_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()
