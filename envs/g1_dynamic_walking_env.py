import os
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class G1DynamicWalkingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path=None,
        dataset_path=None,
        reference_mode="transition",
        target_forward_velocity=0.15,
        action_scale=0.10,
        frame_skip=5,
        max_episode_steps=1000,
        height_offset=0.02,
        reference_speed=0.25,
        initial_stand_steps=80,
        transition_steps=120,
        random_start=False,
        enable_push=False,
        push_window_start=None,
        push_window_end=600,
        push_interval_min=100,
        push_interval_max=200,
        push_force_min=20.0,
        push_force_max=60.0,
        push_duration_steps=5,
    ):
        super().__init__()

        project_root = Path(__file__).resolve().parents[1]

        if model_path is None:
            model_path = (
                project_root
                / "third_party"
                / "mujoco_menagerie"
                / "unitree_g1"
                / "scene.xml"
            )

        if dataset_path is None:
            dataset_path = (
                project_root
                / "datasets"
                / "processed"
                / "g1_amass_walking_il_15dof.npz"
            )

        self.model_path = str(model_path)
        self.dataset_path = str(dataset_path)
        self.reference_mode = str(reference_mode).lower().strip()

        valid_reference_modes = {"transition", "cyclic"}

        if self.reference_mode not in valid_reference_modes:
            raise ValueError(
                f"Invalid reference_mode: {self.reference_mode}. "
                f"Expected one of: {sorted(valid_reference_modes)}"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"IL dataset not found: {self.dataset_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        if self.model.nkey > 0:
            self.stand_qpos = self.model.key_qpos[0].copy()
        else:
            self.stand_qpos = np.zeros(self.model.nq, dtype=np.float64)
            self.stand_qpos[3] = 1.0

        dataset = np.load(self.dataset_path, allow_pickle=True)

        self.fps = float(dataset["fps"][0])
        self.reference_joint_positions = dataset["joint_pos_15"].astype(np.float32)
        self.reference_joint_velocities = dataset["joint_vel_15"].astype(np.float32)

        self.num_frames = self.reference_joint_positions.shape[0]

        if "root_positions" in dataset:
            self.reference_root_positions = dataset["root_positions"].astype(np.float32)
            self.has_reference_root_positions = True
        else:
            self.reference_root_positions = np.zeros((self.num_frames, 3), dtype=np.float32)
            self.reference_root_positions[:, 0] = 0.0
            self.reference_root_positions[:, 1] = 0.0
            self.reference_root_positions[:, 2] = float(self.stand_qpos[2])
            self.has_reference_root_positions = False

        self.controlled_joint_names = [
            str(name) for name in dataset["controlled_joint_names"]
        ]

        self.num_actions = len(self.controlled_joint_names)

        self.target_forward_velocity = float(target_forward_velocity)
        self.action_scale = float(action_scale)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.height_offset = float(height_offset)
        self.reference_speed = float(reference_speed)
        self.initial_stand_steps = int(initial_stand_steps)
        self.transition_steps = int(transition_steps)
        self.random_start = bool(random_start)

        self.enable_push = bool(enable_push)

        self.push_window_start = (
            int(push_window_start)
            if push_window_start is not None
            else self.initial_stand_steps
        )

        self.push_window_end = int(push_window_end)
        self.push_interval_min = int(push_interval_min)
        self.push_interval_max = int(push_interval_max)
        self.push_force_min = float(push_force_min)
        self.push_force_max = float(push_force_max)
        self.push_duration_steps = int(push_duration_steps)

        self.control_dt = self.model.opt.timestep * self.frame_skip

        self.joint_qpos_addresses = []
        self.joint_qvel_addresses = []
        self.actuator_ids = []

        for joint_name in self.controlled_joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                joint_name,
            )

            if joint_id < 0:
                raise ValueError(f"Joint not found: {joint_name}")

            if actuator_id < 0:
                raise ValueError(f"Actuator not found: {joint_name}")

            self.joint_qpos_addresses.append(self.model.jnt_qposadr[joint_id])
            self.joint_qvel_addresses.append(self.model.jnt_dofadr[joint_id])
            self.actuator_ids.append(actuator_id)

        self.upper_body_actuators = []
        controlled_set = set(self.controlled_joint_names)

        for actuator_id in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )

            if actuator_name is None:
                continue

            if actuator_name in controlled_set:
                continue

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                actuator_name,
            )

            if joint_id < 0:
                continue

            qpos_address = self.model.jnt_qposadr[joint_id]

            self.upper_body_actuators.append(
                {
                    "name": actuator_name,
                    "actuator_id": actuator_id,
                    "qpos_address": qpos_address,
                    "target_qpos": float(self.stand_qpos[qpos_address]),
                }
            )

        self.pelvis_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )

        if self.pelvis_body_id < 0:
            raise ValueError("Pelvis body not found.")

        self.left_foot_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "left_foot",
        )

        self.right_foot_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "right_foot",
        )

        if self.left_foot_site_id < 0:
            raise ValueError("left_foot site not found.")

        if self.right_foot_site_id < 0:
            raise ValueError("right_foot site not found.")

        self.left_foot_body_ids = set()
        self.right_foot_body_ids = set()

        for body_name in ["left_ankle_pitch_link", "left_ankle_roll_link"]:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )

            if body_id >= 0:
                self.left_foot_body_ids.add(int(body_id))

        for body_name in ["right_ankle_pitch_link", "right_ankle_roll_link"]:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )

            if body_id >= 0:
                self.right_foot_body_ids.add(int(body_id))

        if len(self.left_foot_body_ids) == 0:
            raise ValueError("No left foot body IDs found.")

        if len(self.right_foot_body_ids) == 0:
            raise ValueError("No right foot body IDs found.")

        self.previous_left_foot_pos = np.zeros(3, dtype=np.float64)
        self.previous_right_foot_pos = np.zeros(3, dtype=np.float64)
        self.ground_foot_height = 0.0

        self.last_foot_info = {
            "left_contact": False,
            "right_contact": False,
            "left_foot_slip": 0.0,
            "right_foot_slip": 0.0,
            "left_foot_clearance": 0.0,
            "right_foot_clearance": 0.0,
        }

        self.episode_step = 0
        self.motion_frame = 0.0
        self.previous_action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_targets = np.zeros(self.num_actions, dtype=np.float32)

        self.next_push_step = None
        self.push_remaining_steps = 0
        self.current_push_force = np.zeros(3, dtype=np.float64)
        self.last_push_info = {"push_active": False, "push_force_magnitude": 0.0}

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        dummy_obs = self._build_observation()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
            dtype=np.float32,
        )

    def _smoothstep(self, x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _interpolate_reference(self, frame_float):
        frame_float = frame_float % self.num_frames

        frame_0 = int(np.floor(frame_float))
        frame_1 = (frame_0 + 1) % self.num_frames
        alpha = frame_float - frame_0

        joint_pos = (
            (1.0 - alpha) * self.reference_joint_positions[frame_0]
            + alpha * self.reference_joint_positions[frame_1]
        )

        joint_vel = (
            (1.0 - alpha) * self.reference_joint_velocities[frame_0]
            + alpha * self.reference_joint_velocities[frame_1]
        )

        root_pos = (
            (1.0 - alpha) * self.reference_root_positions[frame_0]
            + alpha * self.reference_root_positions[frame_1]
        )

        return (
            joint_pos.astype(np.float32),
            joint_vel.astype(np.float32),
            root_pos.astype(np.float32),
        )

    def _get_stand_joint_positions(self):
        return np.array(
            [self.stand_qpos[address] for address in self.joint_qpos_addresses],
            dtype=np.float32,
        )

    def _get_joint_positions(self):
        return np.array(
            [self.data.qpos[address] for address in self.joint_qpos_addresses],
            dtype=np.float32,
        )

    def _get_joint_velocities(self):
        return np.array(
            [self.data.qvel[address] for address in self.joint_qvel_addresses],
            dtype=np.float32,
        )

    def _get_up_z(self):
        return float(self.data.xmat[self.pelvis_body_id, 8])

    def _get_site_position(self, site_id):
        return self.data.site_xpos[site_id].copy()

    def _get_foot_contacts(self):
        left_contact = False
        right_contact = False

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]

            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)

            body_1 = int(self.model.geom_bodyid[geom_1])
            body_2 = int(self.model.geom_bodyid[geom_2])

            if body_1 in self.left_foot_body_ids or body_2 in self.left_foot_body_ids:
                left_contact = True

            if body_1 in self.right_foot_body_ids or body_2 in self.right_foot_body_ids:
                right_contact = True

        return left_contact, right_contact

    def _get_foot_metrics(self):
        left_pos = self._get_site_position(self.left_foot_site_id)
        right_pos = self._get_site_position(self.right_foot_site_id)

        left_vel = (
            left_pos - self.previous_left_foot_pos
        ) / max(self.control_dt, 1e-6)

        right_vel = (
            right_pos - self.previous_right_foot_pos
        ) / max(self.control_dt, 1e-6)

        left_contact, right_contact = self._get_foot_contacts()

        left_foot_slip = 0.0
        right_foot_slip = 0.0

        if left_contact:
            left_foot_slip = float(np.linalg.norm(left_vel[:2]))

        if right_contact:
            right_foot_slip = float(np.linalg.norm(right_vel[:2]))

        left_foot_clearance = float(max(left_pos[2] - self.ground_foot_height, 0.0))
        right_foot_clearance = float(max(right_pos[2] - self.ground_foot_height, 0.0))

        foot_info = {
            "left_contact": bool(left_contact),
            "right_contact": bool(right_contact),
            "left_foot_slip": left_foot_slip,
            "right_foot_slip": right_foot_slip,
            "left_foot_clearance": left_foot_clearance,
            "right_foot_clearance": right_foot_clearance,
        }

        self.last_foot_info = foot_info

        return foot_info

    def _update_previous_foot_positions(self):
        self.previous_left_foot_pos = self._get_site_position(self.left_foot_site_id)
        self.previous_right_foot_pos = self._get_site_position(self.right_foot_site_id)

    def _get_reference_joint_positions_for_step(self):
        stand_joint_pos = self._get_stand_joint_positions()
        walk_joint_pos, _, _ = self._interpolate_reference(self.motion_frame)

        if self.episode_step < self.initial_stand_steps:
            return stand_joint_pos

        transition_step = self.episode_step - self.initial_stand_steps
        alpha = self._smoothstep(transition_step / max(self.transition_steps, 1))

        blended_joint_pos = (
            (1.0 - alpha) * stand_joint_pos
            + alpha * walk_joint_pos
        )

        return blended_joint_pos.astype(np.float32)

    def _get_reference_height_for_step(self):
        if self.episode_step < self.initial_stand_steps:
            return float(self.stand_qpos[2])

        if self.has_reference_root_positions:
            _, _, ref_root_pos = self._interpolate_reference(self.motion_frame)
            return float(ref_root_pos[2] + self.height_offset)

        return float(self.stand_qpos[2] + self.height_offset)

    def _build_observation(self):
        ref_joint_pos = self._get_reference_joint_positions_for_step()

        base_height = np.array([self.data.qpos[2]], dtype=np.float32)
        base_quat = self.data.qpos[3:7].astype(np.float32)
        base_velocity = self.data.qvel[0:6].astype(np.float32)

        joint_pos = self._get_joint_positions()
        joint_vel = self._get_joint_velocities()

        phase = 2.0 * np.pi * (self.motion_frame / max(self.num_frames - 1, 1))

        phase_features = np.array(
            [
                np.sin(phase),
                np.cos(phase),
                self.target_forward_velocity,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                base_height,
                base_quat,
                0.1 * base_velocity,
                joint_pos,
                0.1 * joint_vel,
                ref_joint_pos,
                phase_features,
            ]
        ).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _clip_ctrl(self, actuator_id, value):
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[actuator_id]
        return float(np.clip(value, ctrl_min, ctrl_max))

    def _apply_position_control(self, action):
        ref_joint_pos = self._get_reference_joint_positions_for_step()

        if self.episode_step < self.initial_stand_steps:
            action = np.zeros_like(action, dtype=np.float32)

        target_joint_pos = ref_joint_pos + self.action_scale * action
        applied_targets = np.zeros(self.num_actions, dtype=np.float32)

        for i, actuator_id in enumerate(self.actuator_ids):
            target = self._clip_ctrl(actuator_id, target_joint_pos[i])
            self.data.ctrl[actuator_id] = target
            applied_targets[i] = target

        for item in self.upper_body_actuators:
            actuator_id = item["actuator_id"]
            target_qpos = item["target_qpos"]
            target = self._clip_ctrl(actuator_id, target_qpos)
            self.data.ctrl[actuator_id] = target

        self.last_targets = applied_targets

    def _schedule_next_push(self):
        base_step = max(self.episode_step, self.push_window_start)

        interval = int(
            self.np_random.integers(self.push_interval_min, self.push_interval_max + 1)
        )

        candidate = base_step + interval

        if candidate > self.push_window_end:
            self.next_push_step = None
        else:
            self.next_push_step = candidate

    def _maybe_start_push(self):
        if not self.enable_push:
            return

        if self.next_push_step is None:
            return

        if self.episode_step < self.push_window_start:
            return

        if self.episode_step == self.next_push_step:
            angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            magnitude = float(
                self.np_random.uniform(self.push_force_min, self.push_force_max)
            )

            self.current_push_force = np.array(
                [
                    magnitude * np.cos(angle),
                    magnitude * np.sin(angle),
                    0.0,
                ],
                dtype=np.float64,
            )

            self.push_remaining_steps = self.push_duration_steps

            self.last_push_info = {
                "push_active": True,
                "push_force_magnitude": magnitude,
            }

            self._schedule_next_push()

    def _apply_push_force(self):
        if self.push_remaining_steps > 0:
            self.data.xfrc_applied[self.pelvis_body_id, 0:3] = self.current_push_force
            self.data.xfrc_applied[self.pelvis_body_id, 3:6] = 0.0
        else:
            self.data.xfrc_applied[self.pelvis_body_id, :] = 0.0
            self.last_push_info["push_active"] = False

    def _compute_reward(self, action):
        ref_joint_pos = self._get_reference_joint_positions_for_step()

        joint_pos = self._get_joint_positions()
        joint_vel = self._get_joint_velocities()

        base_height = float(self.data.qpos[2])
        base_y = float(self.data.qpos[1])

        forward_velocity = float(self.data.qvel[0])
        lateral_velocity = float(self.data.qvel[1])

        up_z = self._get_up_z()
        reference_height = self._get_reference_height_for_step()

        foot_info = self._get_foot_metrics()

        left_contact = foot_info["left_contact"]
        right_contact = foot_info["right_contact"]

        left_foot_slip = foot_info["left_foot_slip"]
        right_foot_slip = foot_info["right_foot_slip"]

        left_foot_clearance = foot_info["left_foot_clearance"]
        right_foot_clearance = foot_info["right_foot_clearance"]

        if self.episode_step < self.initial_stand_steps:
            transition_alpha = 0.0
            desired_velocity = 0.0
            direction_sign = 0.0
        else:
            transition_step = self.episode_step - self.initial_stand_steps
            transition_alpha = self._smoothstep(
                transition_step / max(self.transition_steps, 1)
            )
            desired_velocity = self.target_forward_velocity * transition_alpha
            direction_sign = 1.0 if self.target_forward_velocity >= 0.0 else -1.0

        target_speed = abs(desired_velocity)
        allowed_speed = max(target_speed + 0.04, 0.06)

        directional_velocity = direction_sign * forward_velocity

        tracking_error = np.mean((joint_pos - ref_joint_pos) ** 2)
        tracking_reward = np.exp(-tracking_error / 0.07)

        velocity_error = forward_velocity - desired_velocity
        velocity_reward = np.exp(-(velocity_error ** 2) / 0.018)

        upright_reward = np.clip(up_z, 0.0, 1.0)

        height_error = base_height - reference_height
        height_reward = np.exp(-(height_error ** 2) / 0.018)

        if transition_alpha > 0.25:
            progress_reward = float(
                np.clip(directional_velocity, 0.0, target_speed)
            )
        else:
            progress_reward = 0.0

        wrong_direction_penalty = 5.0 * max(-directional_velocity, 0.0)

        overspeed = max(directional_velocity - allowed_speed, 0.0)

        overspeed_penalty = (
            8.0 * overspeed
            + 4.0 * (overspeed ** 2)
        )

        absolute_forward_speed = abs(forward_velocity)
        high_speed_penalty = 3.0 * max(absolute_forward_speed - 0.35, 0.0)

        lateral_position_penalty = 3.0 * (base_y ** 2) + 0.8 * abs(base_y)

        lateral_velocity_penalty = (
            2.0 * (lateral_velocity ** 2)
            + 0.6 * abs(lateral_velocity)
        )

        action_penalty = 0.08 * np.mean(action ** 2)
        smoothness_penalty = 0.14 * np.mean((action - self.previous_action) ** 2)
        joint_velocity_penalty = 0.002 * np.mean(joint_vel ** 2)

        low_height_penalty = 0.0
        if base_height < 0.72:
            low_height_penalty = 5.0 * (0.72 - base_height)

        tilt_penalty = 0.0
        if up_z < 0.92:
            tilt_penalty = 5.0 * (0.92 - up_z)

        collapse_penalty = 0.0
        if self.episode_step >= self.initial_stand_steps and base_height < 0.70:
            collapse_penalty = 4.0 * (0.70 - base_height)

        contact_count = int(left_contact) + int(right_contact)

        no_contact_penalty = 0.0
        double_flight_penalty = 0.0
        single_support_reward = 0.0
        double_support_reward = 0.0

        if self.episode_step >= self.initial_stand_steps:
            if contact_count == 0:
                no_contact_penalty = 2.0

            if transition_alpha > 0.35:
                if contact_count == 1:
                    single_support_reward = 0.35
                elif contact_count == 2:
                    double_support_reward = 0.10
                else:
                    double_flight_penalty = 1.0

        foot_slip_penalty = 0.0

        if transition_alpha > 0.15:
            foot_slip_penalty = 2.5 * (
                left_foot_slip
                + right_foot_slip
            )

        foot_clearance_reward = 0.0

        if transition_alpha > 0.35:
            clearance_target = 0.035

            if not left_contact:
                foot_clearance_reward += min(
                    left_foot_clearance / clearance_target,
                    1.0,
                )

            if not right_contact:
                foot_clearance_reward += min(
                    right_foot_clearance / clearance_target,
                    1.0,
                )

            foot_clearance_reward *= 0.15

        excessive_clearance_penalty = 0.0

        if transition_alpha > 0.35:
            excessive_clearance_penalty = 1.0 * (
                max(left_foot_clearance - 0.16, 0.0)
                + max(right_foot_clearance - 0.16, 0.0)
            )

        reward = (
            1.30 * tracking_reward
            + 1.80 * velocity_reward
            + 2.20 * upright_reward
            + 1.50 * height_reward
            + 0.20 * progress_reward
            + single_support_reward
            + double_support_reward
            + foot_clearance_reward
            + 0.15
            - wrong_direction_penalty
            - overspeed_penalty
            - high_speed_penalty
            - lateral_position_penalty
            - lateral_velocity_penalty
            - action_penalty
            - smoothness_penalty
            - joint_velocity_penalty
            - low_height_penalty
            - tilt_penalty
            - collapse_penalty
            - no_contact_penalty
            - double_flight_penalty
            - foot_slip_penalty
            - excessive_clearance_penalty
        )

        return float(reward)

    def _is_fallen(self):
        if self.episode_step < 20:
            return False

        base_height = float(self.data.qpos[2])
        base_y = float(self.data.qpos[1])
        forward_velocity = float(self.data.qvel[0])
        lateral_velocity = float(self.data.qvel[1])
        up_z = self._get_up_z()

        if base_height < 0.35:
            return True

        if base_height > 1.25:
            return True

        if up_z < 0.45:
            return True

        if self.episode_step > self.initial_stand_steps + 80:
            if abs(base_y) > 0.70:
                return True

            speed_limit = max(0.65, 4.0 * abs(self.target_forward_velocity))

            if abs(forward_velocity) > speed_limit:
                return True

            if abs(lateral_velocity) > 0.80:
                return True

            if base_height < 0.50:
                return True

            if up_z < 0.60:
                return True

        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.episode_step = 0
        self.previous_action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_targets = np.zeros(self.num_actions, dtype=np.float32)

        self.push_remaining_steps = 0
        self.current_push_force = np.zeros(3, dtype=np.float64)
        self.last_push_info = {"push_active": False, "push_force_magnitude": 0.0}
        self.data.xfrc_applied[:, :] = 0.0

        if self.enable_push:
            self._schedule_next_push()
        else:
            self.next_push_step = None

        if self.random_start:
            self.motion_frame = float(self.np_random.integers(0, self.num_frames))
        else:
            self.motion_frame = 0.0

        stand_joint_pos = self._get_stand_joint_positions()

        self.data.qpos[:] = self.stand_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = float(self.stand_qpos[2])

        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0

        for i, qpos_address in enumerate(self.joint_qpos_addresses):
            self.data.qpos[qpos_address] = stand_joint_pos[i]

        mujoco.mj_forward(self.model, self.data)

        self._update_previous_foot_positions()

        left_pos = self._get_site_position(self.left_foot_site_id)
        right_pos = self._get_site_position(self.right_foot_site_id)

        self.ground_foot_height = float(min(left_pos[2], right_pos[2]))

        self.last_foot_info = {
            "left_contact": False,
            "right_contact": False,
            "left_foot_slip": 0.0,
            "right_foot_slip": 0.0,
            "left_foot_clearance": 0.0,
            "right_foot_clearance": 0.0,
        }

        observation = self._build_observation()

        info = {
            "reference_mode": self.reference_mode,
            "dataset_path": self.dataset_path,
            "motion_frame": self.motion_frame,
            "base_height": float(self.data.qpos[2]),
            "up_z": self._get_up_z(),
            "upper_body_joints_held": len(self.upper_body_actuators),
            "push_active": False,
            "push_force_magnitude": 0.0,
            "has_reference_root_positions": self.has_reference_root_positions,
            "left_contact": False,
            "right_contact": False,
            "left_foot_slip": 0.0,
            "right_foot_slip": 0.0,
            "left_foot_clearance": 0.0,
            "right_foot_clearance": 0.0,
        }

        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        self._maybe_start_push()

        for _ in range(self.frame_skip):
            self._apply_position_control(action)
            self._apply_push_force()
            mujoco.mj_step(self.model, self.data)

        if self.push_remaining_steps > 0:
            self.push_remaining_steps -= 1

        if self.episode_step >= self.initial_stand_steps:
            self.motion_frame += self.control_dt * self.fps * self.reference_speed
            self.motion_frame = self.motion_frame % self.num_frames

        reward = self._compute_reward(action)

        terminated = self._is_fallen()
        truncated = self.episode_step >= self.max_episode_steps - 1

        if terminated:
            reward -= 5.0

        observation = self._build_observation()

        info = {
            "reference_mode": self.reference_mode,
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "base_height": float(self.data.qpos[2]),
            "x_velocity": float(self.data.qvel[0]),
            "y_velocity": float(self.data.qvel[1]),
            "up_z": self._get_up_z(),
            "motion_frame": self.motion_frame,
            "reward": reward,
            "upper_body_joints_held": len(self.upper_body_actuators),
            "push_active": bool(self.last_push_info["push_active"]),
            "push_force_magnitude": float(self.last_push_info["push_force_magnitude"]),
            "has_reference_root_positions": self.has_reference_root_positions,
            "left_contact": bool(self.last_foot_info["left_contact"]),
            "right_contact": bool(self.last_foot_info["right_contact"]),
            "left_foot_slip": float(self.last_foot_info["left_foot_slip"]),
            "right_foot_slip": float(self.last_foot_info["right_foot_slip"]),
            "left_foot_clearance": float(self.last_foot_info["left_foot_clearance"]),
            "right_foot_clearance": float(self.last_foot_info["right_foot_clearance"]),
        }

        self.previous_action = action.copy()
        self._update_previous_foot_positions()
        self.episode_step += 1

        return observation, reward, terminated, truncated, info

    def close(self):
        pass