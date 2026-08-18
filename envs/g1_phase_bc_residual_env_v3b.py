from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from envs.g1_phase_bc_residual_env_v2 import G1PhaseBCResidualEnvV2


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


def geom_min_z(data: mujoco.MjData, geom_ids: list[int]) -> float:
    if not geom_ids:
        return 0.0

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


def apply_joint_positions(qpos: np.ndarray, qpos_addresses: list[int], values: np.ndarray) -> None:
    for value, qadr in zip(values, qpos_addresses):
        qpos[qadr] = float(value)


def contact_state(left_contact: bool, right_contact: bool) -> str:
    if left_contact and right_contact:
        return "both"
    if left_contact:
        return "left"
    if right_contact:
        return "right"
    return "none"


class G1PhaseBCResidualEnvV3B(G1PhaseBCResidualEnvV2):
    """
    PPO V3-B contact-aware environment.

    Goal:
    - Keep V2/V3-A balance behavior.
    - Use AMASS B3 reference foot schedule.
    - Reward swing-foot clearance.
    - Penalize both feet planted during expected swing phase.
    - Avoid the V3-A lunge-and-fall shortcut.

    Observation/action spaces are unchanged from V2/V3-A.
    This allows policy transfer from existing PPO checkpoints.
    """

    def __init__(
        self,
        *args: Any,
        reference_swing_clearance_threshold: float = 0.015,
        desired_swing_clearance: float = 0.035,
        contact_margin: float = 0.015,
        contact_match_reward_weight: float = 0.75,
        swing_contact_penalty_weight: float = 0.35,
        double_stance_penalty_weight: float = 0.25,
        no_support_penalty_weight: float = 1.00,
        forward_velocity_weight: float = 0.45,
        velocity_tracking_weight: float = 0.35,
        displacement_reward_weight: float = 0.20,
        backward_velocity_penalty_weight: float = 2.50,
        backward_position_penalty_weight: float = 1.50,
        lunge_velocity_penalty_weight: float = 3.00,
        fall_warning_weight: float = 5.00,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.reference_swing_clearance_threshold = float(reference_swing_clearance_threshold)
        self.desired_swing_clearance = float(desired_swing_clearance)
        self.contact_margin = float(contact_margin)

        self.contact_match_reward_weight = float(contact_match_reward_weight)
        self.swing_contact_penalty_weight = float(swing_contact_penalty_weight)
        self.double_stance_penalty_weight = float(double_stance_penalty_weight)
        self.no_support_penalty_weight = float(no_support_penalty_weight)

        self.forward_velocity_weight = float(forward_velocity_weight)
        self.velocity_tracking_weight = float(velocity_tracking_weight)
        self.displacement_reward_weight = float(displacement_reward_weight)

        self.backward_velocity_penalty_weight = float(backward_velocity_penalty_weight)
        self.backward_position_penalty_weight = float(backward_position_penalty_weight)
        self.lunge_velocity_penalty_weight = float(lunge_velocity_penalty_weight)
        self.fall_warning_weight = float(fall_warning_weight)

        self.left_foot_geom_ids = find_foot_geoms(self.model, "left")
        self.right_foot_geom_ids = find_foot_geoms(self.model, "right")
        self.floor_geom_ids = find_floor_geoms(self.model)

        self.left_stand_z, self.right_stand_z = self._compute_stand_foot_z()

        self.reference_left_swing, self.reference_right_swing = self._build_reference_swing_schedule()

        self.episode_start_x = 0.0

    def reset(self, *args: Any, **kwargs: Any):
        obs, info = super().reset(*args, **kwargs)
        self.episode_start_x = float(self.data.qpos[0])
        return obs, info

    def _compute_stand_foot_z(self) -> tuple[float, float]:
        temp_data = mujoco.MjData(self.model)
        temp_data.qpos[:] = self.stand_qpos
        temp_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, temp_data)

        left_z = geom_min_z(temp_data, self.left_foot_geom_ids)
        right_z = geom_min_z(temp_data, self.right_foot_geom_ids)

        return float(left_z), float(right_z)

    def _build_reference_swing_schedule(self) -> tuple[np.ndarray, np.ndarray]:
        dataset = np.load(self.dataset_path, allow_pickle=True)

        joint_pos_15 = np.asarray(dataset["joint_pos_15"], dtype=np.float64)

        if "root_positions" in dataset.files:
            root_positions = np.asarray(dataset["root_positions"], dtype=np.float64)
        else:
            root_positions = np.zeros((len(joint_pos_15), 3), dtype=np.float64)

        temp_data = mujoco.MjData(self.model)

        left_z_values: list[float] = []
        right_z_values: list[float] = []

        for frame in range(len(joint_pos_15)):
            qpos = self.stand_qpos.copy()

            apply_joint_positions(qpos, self.qpos_addresses, joint_pos_15[frame])

            if frame < len(root_positions):
                qpos[0] = self.stand_qpos[0] + float(root_positions[frame, 0])
                qpos[1] = self.stand_qpos[1] + float(root_positions[frame, 1])
                qpos[2] = self.stand_qpos[2]

            temp_data.qpos[:] = qpos
            temp_data.qvel[:] = 0.0

            mujoco.mj_forward(self.model, temp_data)

            left_z_values.append(geom_min_z(temp_data, self.left_foot_geom_ids))
            right_z_values.append(geom_min_z(temp_data, self.right_foot_geom_ids))

        left_z_arr = np.asarray(left_z_values, dtype=np.float64)
        right_z_arr = np.asarray(right_z_values, dtype=np.float64)

        left_floor_z = float(np.percentile(left_z_arr, 5))
        right_floor_z = float(np.percentile(right_z_arr, 5))

        left_clearance = left_z_arr - left_floor_z
        right_clearance = right_z_arr - right_floor_z

        left_swing = left_clearance > self.reference_swing_clearance_threshold
        right_swing = right_clearance > self.reference_swing_clearance_threshold

        left_swing = self._rollout_sequence(
            left_swing.astype(np.float32),
            start_frame=self.start_frame,
            reverse_time=self.reverse_time,
        ).astype(bool)

        right_swing = self._rollout_sequence(
            right_swing.astype(np.float32),
            start_frame=self.start_frame,
            reverse_time=self.reverse_time,
        ).astype(bool)

        return left_swing, right_swing

    def _current_foot_state(self) -> dict[str, float | bool | int | str]:
        left_z = geom_min_z(self.data, self.left_foot_geom_ids)
        right_z = geom_min_z(self.data, self.right_foot_geom_ids)

        left_clearance = max(0.0, left_z - self.left_stand_z)
        right_clearance = max(0.0, right_z - self.right_stand_z)

        left_contact, left_count = has_foot_floor_contact(
            self.data,
            self.left_foot_geom_ids,
            self.floor_geom_ids,
            self.contact_margin,
        )

        right_contact, right_count = has_foot_floor_contact(
            self.data,
            self.right_foot_geom_ids,
            self.floor_geom_ids,
            self.contact_margin,
        )

        return {
            "left_z": float(left_z),
            "right_z": float(right_z),
            "left_clearance": float(left_clearance),
            "right_clearance": float(right_clearance),
            "left_contact": bool(left_contact),
            "right_contact": bool(right_contact),
            "left_contact_count": int(left_count),
            "right_contact_count": int(right_count),
            "contact_state": contact_state(left_contact, right_contact),
        }

    def _foot_match_score(
        self,
        desired_swing: bool,
        contact: bool,
        clearance: float,
    ) -> float:
        if desired_swing:
            clearance_score = float(np.clip(clearance / max(self.desired_swing_clearance, 1e-6), 0.0, 1.0))
            no_contact_score = 0.0 if contact else 1.0
            return 0.70 * clearance_score + 0.30 * no_contact_score

        return 1.0 if contact else 0.0

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        previous_x = float(self.data.qpos[0])

        target = self._compute_target(action)
        self._apply_target(target)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()

        base_x = float(self.data.qpos[0])
        base_y = float(self.data.qpos[1])
        base_z = float(self.data.qpos[2])
        up_z = self._get_up_z()

        dt = max(self.frame_skip * self.sim_dt, 1e-6)
        vx = (base_x - previous_x) / dt
        relative_x = base_x - self.episode_start_x

        phase_idx = self.phase_index % len(self.reference_left_swing)

        desired_left_swing = bool(self.reference_left_swing[phase_idx])
        desired_right_swing = bool(self.reference_right_swing[phase_idx])

        foot_state = self._current_foot_state()

        left_contact = bool(foot_state["left_contact"])
        right_contact = bool(foot_state["right_contact"])

        left_clearance = float(foot_state["left_clearance"])
        right_clearance = float(foot_state["right_clearance"])

        left_score = self._foot_match_score(
            desired_swing=desired_left_swing,
            contact=left_contact,
            clearance=left_clearance,
        )

        right_score = self._foot_match_score(
            desired_swing=desired_right_swing,
            contact=right_contact,
            clearance=right_clearance,
        )

        up_gate = float(np.clip((up_z - 0.75) / 0.20, 0.0, 1.0))
        height_gate = float(np.clip((base_z - 0.58) / 0.18, 0.0, 1.0))
        balance_gate = up_gate * height_gate

        height_target = float(self.stand_qpos[2])
        height_error = base_z - height_target
        height_reward = math.exp(-(height_error * height_error) / 0.020)

        velocity_tracking_reward = self.velocity_tracking_weight * balance_gate * math.exp(
            -((vx - self.target_velocity) ** 2) / 0.040
        )

        forward_velocity_reward = self.forward_velocity_weight * balance_gate * float(
            np.clip(vx, 0.0, 0.18)
        )

        displacement_reward = self.displacement_reward_weight * balance_gate * float(
            np.clip(relative_x, 0.0, 0.30)
        )

        contact_match_reward = self.contact_match_reward_weight * balance_gate * 0.5 * (
            left_score + right_score
        )

        any_desired_swing = desired_left_swing or desired_right_swing
        both_contact = left_contact and right_contact
        no_contact = (not left_contact) and (not right_contact)

        swing_contact_count = 0
        if desired_left_swing and left_contact:
            swing_contact_count += 1
        if desired_right_swing and right_contact:
            swing_contact_count += 1

        swing_contact_penalty = self.swing_contact_penalty_weight * balance_gate * float(swing_contact_count)

        double_stance_penalty = 0.0
        if any_desired_swing and both_contact:
            double_stance_penalty = self.double_stance_penalty_weight * balance_gate

        no_support_penalty = 0.0
        if no_contact:
            no_support_penalty = self.no_support_penalty_weight

        backward_velocity = max(0.0, -vx)
        backward_velocity_penalty = (
            self.backward_velocity_penalty_weight
            * backward_velocity
            * backward_velocity
        )

        backward_position = max(0.0, -relative_x - 0.03)
        backward_position_penalty = (
            self.backward_position_penalty_weight
            * backward_position
        )

        lunge_velocity = max(0.0, vx - 0.22)
        lunge_velocity_penalty = self.lunge_velocity_penalty_weight * lunge_velocity * lunge_velocity

        lateral_penalty = 0.20 * abs(base_y)

        residual_penalty = 0.015 * float(np.mean(action * action))

        tracking_error = float(
            np.mean(np.abs(self._get_joint_pos().astype(np.float64) - target))
        )
        ctrl_penalty = 0.04 * tracking_error

        up_warning = max(0.0, 0.88 - up_z)
        height_warning = max(0.0, 0.68 - base_z)
        fall_warning_penalty = self.fall_warning_weight * (
            up_warning * up_warning + height_warning * height_warning
        )

        reward = (
            0.30
            + 1.45 * up_z
            + 0.45 * height_reward
            + velocity_tracking_reward
            + forward_velocity_reward
            + displacement_reward
            + contact_match_reward
            - swing_contact_penalty
            - double_stance_penalty
            - no_support_penalty
            - backward_velocity_penalty
            - backward_position_penalty
            - lunge_velocity_penalty
            - lateral_penalty
            - residual_penalty
            - ctrl_penalty
            - fall_warning_penalty
        )

        terminated = bool(base_z < self.fall_height or up_z < self.fall_up_z)

        if terminated:
            reward -= 30.0

        self.step_count += 1
        self.phase_index = (self.phase_index + 1) % len(self.phase_bc_targets)
        self.prev_action = action.copy()

        truncated = bool(self.step_count >= self.max_episode_steps)

        obs = self._get_obs()
        info = self._get_info(action=action, reward=reward)

        info["vx"] = float(vx)
        info["relative_x"] = float(relative_x)
        info["terminated"] = terminated
        info["truncated"] = truncated

        info["desired_left_swing"] = int(desired_left_swing)
        info["desired_right_swing"] = int(desired_right_swing)

        info.update(foot_state)

        info["left_match_score"] = float(left_score)
        info["right_match_score"] = float(right_score)
        info["balance_gate"] = float(balance_gate)

        info["velocity_tracking_reward"] = float(velocity_tracking_reward)
        info["forward_velocity_reward"] = float(forward_velocity_reward)
        info["displacement_reward"] = float(displacement_reward)
        info["contact_match_reward"] = float(contact_match_reward)

        info["swing_contact_penalty"] = float(swing_contact_penalty)
        info["double_stance_penalty"] = float(double_stance_penalty)
        info["no_support_penalty"] = float(no_support_penalty)
        info["backward_velocity_penalty"] = float(backward_velocity_penalty)
        info["backward_position_penalty"] = float(backward_position_penalty)
        info["lunge_velocity_penalty"] = float(lunge_velocity_penalty)
        info["fall_warning_penalty"] = float(fall_warning_penalty)

        return obs, float(reward), terminated, truncated, info
