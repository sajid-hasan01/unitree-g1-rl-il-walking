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
        action_target_smoothing=0.55,
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
        include_contact_phase_observation=False,
        use_reference_contact_mask=False,
        reference_start_frame=0,
        use_functional_foot_contact=True,
        functional_contact_threshold=0.025,
        initial_yaw_degrees=None,
        reference_state_initialization=False,
        rsi_start_frame=0,
        rsi_end_frame=None,
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
            self.reference_root_positions = np.zeros(
                (self.num_frames, 3),
                dtype=np.float32,
            )
            self.reference_root_positions[:, 0] = 0.0
            self.reference_root_positions[:, 1] = 0.0
            self.reference_root_positions[:, 2] = float(self.stand_qpos[2])
            self.has_reference_root_positions = False

        if "contact_mask" in dataset:
            self.reference_contact_mask = dataset["contact_mask"].astype(np.float32)
            self.has_reference_contact_mask = True
        else:
            self.reference_contact_mask = None
            self.has_reference_contact_mask = False

        self.controlled_joint_names = [
            str(name) for name in dataset["controlled_joint_names"]
        ]

        self.num_actions = len(self.controlled_joint_names)

        self.target_forward_velocity = float(target_forward_velocity)
        self.action_scale = float(action_scale)

        # v55:
        # Previous runs used a scalar residual action_scale around 0.06 rad.
        # Manual authority tests showed real foot lifting needs much larger joint
        # offsets: roughly hip_pitch +0.55 rad and knee +0.8 rad. This vector keeps
        # high authority only on joints needed for foot lift, while keeping waist
        # and yaw/roll residuals smaller.
        self.per_joint_residual_scale = self._build_per_joint_residual_scale()

        # v42 target smoothing:
        # The PPO action is a residual added to the reference joint pose. Without
        # filtering, the residual target can change abruptly during contact switches.
        # This low-pass filter smooths joint target commands and reduces sideways
        # jerks. Default 0.55 means 55% previous target + 45% requested target.
        self.action_target_smoothing = float(
            np.clip(action_target_smoothing, 0.0, 0.98)
        )

        # v37 forward-facing setup:
        # The selected OpenHE walking segment and trained PPO tasks use negative X
        # as the intended walking direction. With the original identity yaw, the
        # robot can visually look as if it is stepping backward. For negative-X
        # walking tasks, this environment starts the free root with 180 degrees
        # yaw so the robot visually faces the direction of motion.
        #
        # Existing training/evaluation scripts do not need a new argument:
        # if initial_yaw_degrees is None, it is selected automatically from the
        # target direction. For positive-X tasks, yaw remains 0 degrees.
        if initial_yaw_degrees is None:
            # v51:
            # Zero-residual tests showed the OpenHE/G1 reference naturally drives
            # the robot along negative X with yaw=0. The old yaw=180 auto-rotation
            # made the visual direction look nicer but conflicted with the dynamics.
            initial_yaw_degrees = 0.0

        self.initial_yaw_degrees = float(initial_yaw_degrees)
        self.initial_base_quat = self._yaw_to_quat_wxyz(
            np.deg2rad(self.initial_yaw_degrees)
        )

        # Keep the current v34/v33 correct-direction speed-cap behavior.
        self.max_correct_direction_speed = 0.40

        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.height_offset = float(height_offset)
        self.reference_speed = float(reference_speed)
        self.initial_stand_steps = int(initial_stand_steps)
        self.transition_steps = int(transition_steps)
        self.random_start = bool(random_start)
        self.include_contact_phase_observation = bool(include_contact_phase_observation)

        # v51:
        # The original OpenHE contact labels did not match the actual MuJoCo
        # collision contacts for this G1 XML. By default, v51 does NOT use
        # reference contact masks for observation/reward targets. It keeps actual
        # contact/slip/clearance terms and reference joint tracking.
        self.use_reference_contact_mask = bool(use_reference_contact_mask)

        # v53:
        # The first frames of this OpenHE segment produced sticky double-support
        # and backward/falling motion. A root-height sweep showed that starting
        # near local frame 25 with higher root height is much more dynamically
        # usable. reference_start_frame shifts the local phase used by normal
        # standing-start rollouts without changing observation/action shape.
        self.reference_start_frame = int(reference_start_frame) % self.num_frames

        # v54:
        # MuJoCo collision contacts stayed true for the swing foot even when the
        # visual foot site was higher. For gait learning, use functional foot
        # contact based on relative foot-site clearance: the lower foot is stance,
        # the higher foot is swing, and both are stance only when heights are close.
        self.use_functional_foot_contact = bool(use_functional_foot_contact)
        self.functional_contact_threshold = float(functional_contact_threshold)

        # v39 Reference State Initialization (RSI)
        # Training-only locomotion trick inspired by DeepMimic-style imitation RL.
        # When enabled, reset() starts from a random walking-reference frame instead
        # of always starting from the standing pose. This exposes PPO to the full
        # gait cycle and prevents the policy from only practicing early gait frames.
        self.reference_state_initialization = bool(reference_state_initialization)
        self.rsi_start_frame = int(rsi_start_frame)

        if rsi_end_frame is None:
            self.rsi_end_frame = self.num_frames - 1
        else:
            self.rsi_end_frame = int(rsi_end_frame)

        self.rsi_start_frame = int(
            np.clip(self.rsi_start_frame, 0, max(self.num_frames - 1, 0))
        )
        self.rsi_end_frame = int(
            np.clip(self.rsi_end_frame, self.rsi_start_frame, max(self.num_frames - 1, 0))
        )

        self.rsi_active_this_episode = False
        self.rsi_frame_this_episode = 0

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

        # v50: residual action ramp.
        # The old environment zeroed PPO residual actions during initial standing,
        # then allowed the full residual immediately at the walk transition. With a
        # saturated policy this creates a sudden right/left leg jerk. v50 gradually
        # enables residual actions using the same smooth transition alpha that blends
        # the reference pose.
        self.last_residual_alpha = 0.0
        self.last_reward_terms = {}

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

        self.data.qpos[:] = self.stand_qpos
        self.data.qvel[:] = 0.0
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = float(self.stand_qpos[2])
        self.data.qpos[3:7] = self.initial_base_quat
        mujoco.mj_forward(self.model, self.data)

        dummy_obs = self._build_observation()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
            dtype=np.float32,
        )

    def _yaw_to_quat_wxyz(self, yaw_radians):
        """
        Convert yaw angle to MuJoCo free-joint quaternion order [w, x, y, z].

        A yaw of 180 degrees makes the Unitree G1 body face the negative-X
        direction while still preserving an upright torso. This is used for the
        v37/v39 PPO-forward-facing walking setup.
        """
        half_yaw = 0.5 * float(yaw_radians)

        return np.array(
            [
                np.cos(half_yaw),
                0.0,
                0.0,
                np.sin(half_yaw),
            ],
            dtype=np.float64,
        )

    def _smoothstep(self, x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _build_per_joint_residual_scale(self):
        scales = np.zeros(self.num_actions, dtype=np.float64)

        for i, joint_name in enumerate(self.controlled_joint_names):
            name = str(joint_name)

            if "hip_pitch" in name:
                scales[i] = 0.55
            elif "knee" in name:
                scales[i] = 0.80
            elif "ankle_pitch" in name:
                scales[i] = 0.45
            elif "hip_roll" in name:
                scales[i] = 0.25
            elif "ankle_roll" in name:
                scales[i] = 0.20
            elif "hip_yaw" in name:
                scales[i] = 0.12
            elif "waist" in name:
                scales[i] = 0.08
            else:
                scales[i] = 0.12

        return scales

    def _map_reference_frame(self, frame_float):
        return (float(frame_float) + float(self.reference_start_frame)) % self.num_frames

    def _interpolate_reference(self, frame_float):
        frame_float = self._map_reference_frame(frame_float)

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

    def _get_reference_contact_for_step(self):
        if not self.has_reference_contact_mask:
            return None, None

        if not self.use_reference_contact_mask:
            return None, None

        frame_idx = int(round(self._map_reference_frame(self.motion_frame))) % self.num_frames

        left_expected = bool(self.reference_contact_mask[frame_idx, 0] > 0.5)
        right_expected = bool(self.reference_contact_mask[frame_idx, 1] > 0.5)

        return left_expected, right_expected

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

        collision_left_contact, collision_right_contact = self._get_foot_contacts()

        # Functional clearance: relative to the lower of the two foot sites.
        # This is more useful for learning visual swing/stance than raw collision
        # contact because the Unitree G1 ankle collision capsules can touch the
        # floor while the foot site is visibly lifted.
        pair_ground_height = float(min(left_pos[2], right_pos[2]))
        left_foot_clearance = float(max(left_pos[2] - pair_ground_height, 0.0))
        right_foot_clearance = float(max(right_pos[2] - pair_ground_height, 0.0))

        if self.use_functional_foot_contact:
            close_left = left_foot_clearance <= self.functional_contact_threshold
            close_right = right_foot_clearance <= self.functional_contact_threshold

            if close_left and close_right:
                left_contact = True
                right_contact = True
            elif left_foot_clearance < right_foot_clearance:
                left_contact = True
                right_contact = False
            else:
                left_contact = False
                right_contact = True
        else:
            left_contact = collision_left_contact
            right_contact = collision_right_contact

            left_foot_clearance = float(max(left_pos[2] - self.ground_foot_height, 0.0))
            right_foot_clearance = float(max(right_pos[2] - self.ground_foot_height, 0.0))

        left_foot_slip = 0.0
        right_foot_slip = 0.0

        if left_contact:
            left_foot_slip = float(np.linalg.norm(left_vel[:2]))

        if right_contact:
            right_foot_slip = float(np.linalg.norm(right_vel[:2]))

        foot_info = {
            "left_contact": bool(left_contact),
            "right_contact": bool(right_contact),
            "collision_left_contact": bool(collision_left_contact),
            "collision_right_contact": bool(collision_right_contact),
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

    def _get_transition_alpha_and_desired_velocity(self):
        if self.episode_step < self.initial_stand_steps:
            return 0.0, 0.0, 0.0

        transition_step = self.episode_step - self.initial_stand_steps
        transition_alpha = self._smoothstep(
            transition_step / max(self.transition_steps, 1)
        )
        desired_velocity = self.target_forward_velocity * transition_alpha
        direction_sign = 1.0 if self.target_forward_velocity >= 0.0 else -1.0

        return transition_alpha, desired_velocity, direction_sign

    def _build_observation(self):
        ref_joint_pos = self._get_reference_joint_positions_for_step()

        base_height = np.array([self.data.qpos[2]], dtype=np.float32)
        base_quat = self.data.qpos[3:7].astype(np.float32)
        base_velocity = self.data.qvel[0:6].astype(np.float32)

        joint_pos = self._get_joint_positions()
        joint_vel = self._get_joint_velocities()

        phase_frame = self._map_reference_frame(self.motion_frame)
        phase = 2.0 * np.pi * (phase_frame / max(self.num_frames - 1, 1))

        phase_features = np.array(
            [
                np.sin(phase),
                np.cos(phase),
                self.target_forward_velocity,
            ],
            dtype=np.float32,
        )

        obs_parts = [
            base_height,
            base_quat,
            0.1 * base_velocity,
            joint_pos,
            0.1 * joint_vel,
            ref_joint_pos,
            phase_features,
        ]

        if self.include_contact_phase_observation:
            transition_alpha, desired_velocity, _ = (
                self._get_transition_alpha_and_desired_velocity()
            )

            left_expected, right_expected = self._get_reference_contact_for_step()

            if left_expected is None:
                left_expected_value = 0.0
                right_expected_value = 0.0
            else:
                left_expected_value = 1.0 if left_expected else 0.0
                right_expected_value = 1.0 if right_expected else 0.0

            left_actual_value = (
                1.0 if self.last_foot_info.get("left_contact", False) else 0.0
            )
            right_actual_value = (
                1.0 if self.last_foot_info.get("right_contact", False) else 0.0
            )

            contact_phase_features = np.array(
                [
                    desired_velocity,
                    transition_alpha,
                    left_expected_value,
                    right_expected_value,
                    left_actual_value,
                    right_actual_value,
                ],
                dtype=np.float32,
            )

            obs_parts.append(contact_phase_features)

        obs = np.concatenate(obs_parts).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _clip_ctrl(self, actuator_id, value):
        ctrl_min, ctrl_max = self.model.actuator_ctrlrange[actuator_id]
        return float(np.clip(value, ctrl_min, ctrl_max))

    def _apply_position_control(self, action):
        ref_joint_pos = self._get_reference_joint_positions_for_step()

        transition_alpha, _, _ = self._get_transition_alpha_and_desired_velocity()

        if self.episode_step < self.initial_stand_steps:
            residual_alpha = 0.0
        else:
            # v55:
            # Use sqrt ramp so residual authority becomes available early enough
            # to lift the swing foot before the body accelerates into a fall.
            residual_alpha = float(np.sqrt(max(transition_alpha, 0.0)))

        self.last_residual_alpha = float(residual_alpha)

        # v55:
        # Per-joint residual authority. With --action_scale 1.0, hip/knee/ankle
        # residuals are large enough to reproduce the manual foot-lift test.
        residual = (
            self.action_scale
            * residual_alpha
            * self.per_joint_residual_scale
            * action
        )

        target_joint_pos = ref_joint_pos + residual
        applied_targets = np.zeros(self.num_actions, dtype=np.float32)

        for i, actuator_id in enumerate(self.actuator_ids):
            requested_target = self._clip_ctrl(actuator_id, target_joint_pos[i])

            if self.action_target_smoothing > 0.0 and self.episode_step > 0:
                target = (
                    self.action_target_smoothing * float(self.last_targets[i])
                    + (1.0 - self.action_target_smoothing) * requested_target
                )
                target = self._clip_ctrl(actuator_id, target)
            else:
                target = requested_target

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

        # v34: use MuJoCo subtree COM instead of only pelvis/root qpos.
        # The pelvis subtree contains the robot body under the pelvis.
        robot_com = self.data.subtree_com[self.pelvis_body_id].copy()

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

        left_expected, right_expected = self._get_reference_contact_for_step()

        transition_alpha, desired_velocity, direction_sign = (
            self._get_transition_alpha_and_desired_velocity()
        )

        target_speed = abs(desired_velocity)

        allowed_speed = max(target_speed + 0.03, 0.07)

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

        wrong_direction_amount = max(-directional_velocity, 0.0)

        wrong_direction_penalty = 0.0
        if transition_alpha > 0.25:
            wrong_direction_penalty = (
                12.0 * wrong_direction_amount
                + 8.0 * (wrong_direction_amount ** 2)
            )

        overspeed = max(directional_velocity - allowed_speed, 0.0)
        hard_overspeed = max(
            directional_velocity - self.max_correct_direction_speed,
            0.0,
        )

        overspeed_penalty = 0.0
        if transition_alpha > 0.20:
            overspeed_penalty = (
                16.0 * overspeed
                + 18.0 * (overspeed ** 2)
                + 30.0 * hard_overspeed
                + 30.0 * (hard_overspeed ** 2)
            )

        absolute_forward_speed = abs(forward_velocity)
        high_speed_penalty = 5.0 * max(absolute_forward_speed - 0.35, 0.0)

        # v42 lateral stability:
        # v40/v41 improved survival/contact exposure but several checkpoints failed
        # by sliding sideways with |y_velocity| near 0.8 m/s. Increase lateral
        # position and velocity penalties, and add an extra penalty once lateral
        # speed exceeds a safe walking band.
        lateral_position_penalty = 5.0 * (base_y ** 2) + 1.25 * abs(base_y)

        lateral_velocity_penalty = (
            4.0 * (lateral_velocity ** 2)
            + 1.20 * abs(lateral_velocity)
        )

        lateral_speed_excess = max(abs(lateral_velocity) - 0.30, 0.0)
        lateral_drift_penalty = 0.0
        if transition_alpha > 0.20:
            lateral_drift_penalty = (
                3.0 * lateral_speed_excess
                + 8.0 * (lateral_speed_excess ** 2)
            )

        action_penalty = 0.015 * np.mean(action ** 2)
        smoothness_penalty = 0.035 * np.mean((action - self.previous_action) ** 2)
        joint_velocity_penalty = 0.002 * np.mean(joint_vel ** 2)

        # v50: discourage saturated raw actions, especially before and during the
        # stand-to-walk transition. In earlier runs, the residual was not applied
        # during standing, so the policy could output saturated actions that later
        # caused a sudden leg jerk.
        abs_action = np.abs(action)
        saturation_fraction = float(np.mean(abs_action > 0.92))
        saturation_excess = np.maximum(abs_action - 0.86, 0.0)

        action_saturation_penalty = 0.0
        early_action_readiness_penalty = 0.0

        if transition_alpha > 0.10:
            action_saturation_penalty = (
                0.08 * float(np.mean(saturation_excess ** 2))
                + 0.04 * saturation_fraction
            )

        if transition_alpha < 0.65:
            early_action_readiness_penalty = (
                0.035 * float(np.mean(action ** 2))
                + 0.05 * saturation_fraction
            )

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
        double_support_sliding_penalty = 0.0

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

            if transition_alpha > 0.35 and contact_count == 2:
                slide_amount = max(directional_velocity - 0.18, 0.0)
                double_support_sliding_penalty = (
                    6.0 * slide_amount
                    + 5.0 * (left_foot_slip + right_foot_slip)
                )

        contact_phase_reward = 0.0
        contact_mismatch_penalty = 0.0
        phase_slip_penalty = 0.0
        phase_clearance_reward = 0.0
        phase_clearance_excess_penalty = 0.0

        use_generic_foot_terms = True

        if self.has_reference_contact_mask and self.use_reference_contact_mask and transition_alpha > 0.15:
            use_generic_foot_terms = False

            if left_contact == left_expected:
                contact_phase_reward += 1.00
            else:
                contact_mismatch_penalty += 1.25

            if right_contact == right_expected:
                contact_phase_reward += 1.00
            else:
                contact_mismatch_penalty += 1.25

            if left_expected:
                phase_slip_penalty += 1.8 * left_foot_slip
            if right_expected:
                phase_slip_penalty += 1.8 * right_foot_slip

            # v55:
            # Make swing-foot clearance a primary curriculum objective. Previous
            # policies received only ~0.15 reward for lift, so they preferred
            # low-action sliding. Manual tests show 5-8 cm clearance is feasible.
            clearance_target = 0.065
            stance_clearance_limit = 0.030

            if not left_expected:
                phase_clearance_reward += 3.00 * min(
                    left_foot_clearance / clearance_target,
                    1.0,
                )
            else:
                phase_clearance_excess_penalty += 1.00 * max(
                    left_foot_clearance - stance_clearance_limit,
                    0.0,
                )

            if not right_expected:
                phase_clearance_reward += 3.00 * min(
                    right_foot_clearance / clearance_target,
                    1.0,
                )
            else:
                phase_clearance_excess_penalty += 1.00 * max(
                    right_foot_clearance - stance_clearance_limit,
                    0.0,
                )

            phase_clearance_excess_penalty += 0.6 * (
                max(left_foot_clearance - 0.20, 0.0)
                + max(right_foot_clearance - 0.20, 0.0)
            )

        foot_slip_penalty = 0.0
        foot_clearance_reward = 0.0
        excessive_clearance_penalty = 0.0

        if use_generic_foot_terms:
            if transition_alpha > 0.15:
                foot_slip_penalty = 2.5 * (left_foot_slip + right_foot_slip)

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

                excessive_clearance_penalty = 1.0 * (
                    max(left_foot_clearance - 0.16, 0.0)
                    + max(right_foot_clearance - 0.16, 0.0)
                )
        else:
            foot_slip_penalty = phase_slip_penalty
            foot_clearance_reward = phase_clearance_reward
            excessive_clearance_penalty = phase_clearance_excess_penalty

        # v34: weaker COM-over-support-foot balance term.
        # This is active only during true single support.
        # v33 improved survival but hurt contact timing, so v34 reduces its influence.
        # It still targets the observed v29/v31 failure mode:
        # the swing foot stays lifted while the body COM drifts forward/laterally,
        # causing speed-cap termination instead of stable one-leg balance.
        single_leg_balance_reward = 0.0
        single_leg_balance_penalty = 0.0

        if transition_alpha > 0.35 and contact_count == 1:
            if left_contact and not right_contact:
                support_pos = self._get_site_position(self.left_foot_site_id)
            elif right_contact and not left_contact:
                support_pos = self._get_site_position(self.right_foot_site_id)
            else:
                support_pos = None

            if support_pos is not None:
                com_dx = float(robot_com[0] - support_pos[0])
                com_dy = float(robot_com[1] - support_pos[1])
                com_offset = float(np.hypot(com_dx, com_dy))

                single_leg_balance_reward = 0.60 * np.exp(
                    -(com_offset ** 2) / 0.02
                )
                single_leg_balance_penalty = 1.00 * max(com_offset - 0.18, 0.0)

        reward = (
            1.30 * tracking_reward
            + 1.80 * velocity_reward
            + 2.20 * upright_reward
            + 1.50 * height_reward
            + 0.20 * progress_reward
            + single_support_reward
            + double_support_reward
            + 1.00 * foot_clearance_reward
            + contact_phase_reward
            + single_leg_balance_reward
            + 0.15
            - wrong_direction_penalty
            - overspeed_penalty
            - high_speed_penalty
            - lateral_position_penalty
            - lateral_velocity_penalty
            - lateral_drift_penalty
            - action_penalty
            - smoothness_penalty
            - joint_velocity_penalty
            - action_saturation_penalty
            - early_action_readiness_penalty
            - low_height_penalty
            - tilt_penalty
            - collapse_penalty
            - no_contact_penalty
            - double_flight_penalty
            - foot_slip_penalty
            - excessive_clearance_penalty
            - contact_mismatch_penalty
            - double_support_sliding_penalty
            - single_leg_balance_penalty
        )

        self.last_reward_terms = {
            "reward_version": "v55_large_residual_swing_lift",
            "residual_alpha": float(self.last_residual_alpha),
            "action_saturation_fraction": float(saturation_fraction),
            "action_saturation_penalty": float(action_saturation_penalty),
            "early_action_readiness_penalty": float(early_action_readiness_penalty),
            "tracking_reward": float(tracking_reward),
            "velocity_reward": float(velocity_reward),
            "upright_reward": float(upright_reward),
            "height_reward": float(height_reward),
            "progress_reward": float(progress_reward),
            "contact_phase_reward": float(contact_phase_reward),
            "single_leg_balance_reward": float(single_leg_balance_reward),
            "single_leg_balance_penalty": float(single_leg_balance_penalty),
            "lateral_velocity_penalty": float(lateral_velocity_penalty),
            "lateral_drift_penalty": float(lateral_drift_penalty),
            "wrong_direction_penalty": float(wrong_direction_penalty),
            "overspeed_penalty": float(overspeed_penalty),
            "foot_slip_penalty": float(foot_slip_penalty),
            "contact_mismatch_penalty": float(contact_mismatch_penalty),
        }

        return float(reward)

    def _is_fallen(self):
        if self.episode_step < 20:
            return False

        base_height = float(self.data.qpos[2])
        base_x = float(self.data.qpos[0])
        base_y = float(self.data.qpos[1])
        forward_velocity = float(self.data.qvel[0])
        lateral_velocity = float(self.data.qvel[1])
        up_z = self._get_up_z()

        transition_alpha, _, direction_sign = (
            self._get_transition_alpha_and_desired_velocity()
        )
        directional_velocity = direction_sign * forward_velocity

        if base_height < 0.35:
            return True

        if base_height > 1.25:
            return True

        if up_z < 0.45:
            return True

        if self.episode_step > self.initial_stand_steps + 80:
            if abs(base_y) > 0.55:
                return True

            if directional_velocity < -0.20:
                return True

            if self.target_forward_velocity < 0.0 and base_x > 0.18:
                return True

            if self.target_forward_velocity > 0.0 and base_x < -0.18:
                return True

            if transition_alpha > 0.35 and directional_velocity > 0.45:
                return True

            if (
                transition_alpha > 0.55
                and directional_velocity > self.max_correct_direction_speed
            ):
                return True

            if abs(forward_velocity) > 0.80:
                return True

            if abs(lateral_velocity) > 0.65:
                return True

            if base_height < 0.50:
                return True

            if up_z < 0.60:
                return True

        return False

    def _select_rsi_frame(self):
        if self.rsi_end_frame <= self.rsi_start_frame:
            return int(self.rsi_start_frame)

        return int(
            self.np_random.integers(
                self.rsi_start_frame,
                self.rsi_end_frame + 1,
            )
        )

    def _apply_reference_state_initialization(self, frame_index):
        """
        Start the robot from a walking-reference state.

        This is used only for training when reference_state_initialization=True.
        Evaluation/showcase should keep RSI disabled so the robot starts normally
        from standing.

        The observation/action shape is unchanged, so v37 65-dimensional PPO
        checkpoints can be resumed.
        """

        local_frame_index = int(frame_index) % self.num_frames
        actual_frame_index = int(round(self._map_reference_frame(local_frame_index))) % self.num_frames

        ref_joint_pos = self.reference_joint_positions[actual_frame_index].astype(np.float64)
        ref_joint_vel = self.reference_joint_velocities[actual_frame_index].astype(np.float64)
        ref_root_pos = self.reference_root_positions[actual_frame_index].astype(np.float64)

        self.motion_frame = float(local_frame_index)

        self.data.qpos[:] = self.stand_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # Keep world origin near zero at reset. We do not teleport far along the
        # original dataset path because the PPO task uses target world velocity,
        # not global dataset displacement.
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0

        # Use reference root height when available, with a safety floor.
        if self.has_reference_root_positions:
            self.data.qpos[2] = float(max(ref_root_pos[2] + self.height_offset, 0.72))
        else:
            self.data.qpos[2] = float(self.stand_qpos[2])

        # Keep v37 forward-facing yaw.
        self.data.qpos[3:7] = self.initial_base_quat

        for i, qpos_address in enumerate(self.joint_qpos_addresses):
            self.data.qpos[qpos_address] = float(ref_joint_pos[i])

        for i, qvel_address in enumerate(self.joint_qvel_addresses):
            self.data.qvel[qvel_address] = float(ref_joint_vel[i])

        # Give the free root a small velocity consistent with the target command.
        # This avoids a harsh mismatch when starting from the middle of a gait.
        self.data.qvel[0] = float(self.target_forward_velocity)
        self.data.qvel[1] = 0.0
        self.data.qvel[2] = 0.0

        for i, actuator_id in enumerate(self.actuator_ids):
            target = self._clip_ctrl(actuator_id, ref_joint_pos[i])
            self.data.ctrl[actuator_id] = target
            self.last_targets[i] = target

        for item in self.upper_body_actuators:
            actuator_id = item["actuator_id"]
            target_qpos = item["target_qpos"]
            target = self._clip_ctrl(actuator_id, target_qpos)
            self.data.ctrl[actuator_id] = target

        mujoco.mj_forward(self.model, self.data)

    def _apply_standing_initialization(self):
        stand_joint_pos = self._get_stand_joint_positions()

        self.data.qpos[:] = self.stand_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = float(self.stand_qpos[2])
        self.data.qpos[3:7] = self.initial_base_quat

        for i, qpos_address in enumerate(self.joint_qpos_addresses):
            self.data.qpos[qpos_address] = stand_joint_pos[i]

        for i, actuator_id in enumerate(self.actuator_ids):
            target = self._clip_ctrl(actuator_id, stand_joint_pos[i])
            self.data.ctrl[actuator_id] = target
            self.last_targets[i] = target

        for item in self.upper_body_actuators:
            actuator_id = item["actuator_id"]
            target_qpos = item["target_qpos"]
            target = self._clip_ctrl(actuator_id, target_qpos)
            self.data.ctrl[actuator_id] = target

        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.episode_step = 0
        self.previous_action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_targets = np.zeros(self.num_actions, dtype=np.float32)
        self.last_residual_alpha = 0.0
        self.last_reward_terms = {}

        self.push_remaining_steps = 0
        self.current_push_force = np.zeros(3, dtype=np.float64)
        self.last_push_info = {"push_active": False, "push_force_magnitude": 0.0}
        self.data.xfrc_applied[:, :] = 0.0

        if self.enable_push:
            self._schedule_next_push()
        else:
            self.next_push_step = None

        self.rsi_active_this_episode = False
        self.rsi_frame_this_episode = 0

        if self.reference_state_initialization:
            self.rsi_active_this_episode = True
            self.rsi_frame_this_episode = self._select_rsi_frame()

            # Start directly in the walking phase.
            # This makes transition_alpha = 1.0 and prevents _apply_position_control()
            # from zeroing actions during RSI training episodes.
            self.episode_step = self.initial_stand_steps + self.transition_steps

            self._apply_reference_state_initialization(self.rsi_frame_this_episode)

        else:
            if self.random_start:
                self.motion_frame = float(self.np_random.integers(0, self.num_frames))
            else:
                self.motion_frame = 0.0

            self._apply_standing_initialization()

        self._update_previous_foot_positions()

        left_pos = self._get_site_position(self.left_foot_site_id)
        right_pos = self._get_site_position(self.right_foot_site_id)

        self.ground_foot_height = float(min(left_pos[2], right_pos[2]))

        initial_left_contact, initial_right_contact = self._get_foot_contacts()

        self.last_foot_info = {
            "left_contact": bool(initial_left_contact),
            "right_contact": bool(initial_right_contact),
            "left_foot_slip": 0.0,
            "right_foot_slip": 0.0,
            "left_foot_clearance": float(
                max(left_pos[2] - self.ground_foot_height, 0.0)
            ),
            "right_foot_clearance": float(
                max(right_pos[2] - self.ground_foot_height, 0.0)
            ),
        }

        observation = self._build_observation()

        left_expected, right_expected = self._get_reference_contact_for_step()

        info = {
            "reference_mode": self.reference_mode,
            "dataset_path": self.dataset_path,
            "motion_frame": self.motion_frame,
            "reference_start_frame": self.reference_start_frame,
            "reference_actual_frame": float(self._map_reference_frame(self.motion_frame)),
            "base_height": float(self.data.qpos[2]),
            "up_z": self._get_up_z(),
            "upper_body_joints_held": len(self.upper_body_actuators),
            "push_active": False,
            "push_force_magnitude": 0.0,
            "has_reference_root_positions": self.has_reference_root_positions,
            "has_reference_contact_mask": self.has_reference_contact_mask,
            "use_reference_contact_mask": self.use_reference_contact_mask,
            "use_functional_foot_contact": self.use_functional_foot_contact,
            "functional_contact_threshold": self.functional_contact_threshold,
            "residual_scale_max": float(np.max(self.per_joint_residual_scale)),
            "residual_scale_mean": float(np.mean(self.per_joint_residual_scale)),
            "include_contact_phase_observation": self.include_contact_phase_observation,
            "initial_yaw_degrees": self.initial_yaw_degrees,
            "action_target_smoothing": self.action_target_smoothing,
            "reward_version": "v55_large_residual_swing_lift",
            "residual_alpha": float(self.last_residual_alpha),
            "reference_state_initialization": self.reference_state_initialization,
            "rsi_active_this_episode": self.rsi_active_this_episode,
            "rsi_frame_this_episode": self.rsi_frame_this_episode,
            "left_contact": bool(self.last_foot_info["left_contact"]),
            "right_contact": bool(self.last_foot_info["right_contact"]),
            "collision_left_contact": bool(self.last_foot_info.get("collision_left_contact", False)),
            "collision_right_contact": bool(self.last_foot_info.get("collision_right_contact", False)),
            "left_foot_slip": float(self.last_foot_info["left_foot_slip"]),
            "right_foot_slip": float(self.last_foot_info["right_foot_slip"]),
            "left_foot_clearance": float(self.last_foot_info["left_foot_clearance"]),
            "right_foot_clearance": float(self.last_foot_info["right_foot_clearance"]),
            "left_expected_contact": left_expected,
            "right_expected_contact": right_expected,
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

        left_expected, right_expected = self._get_reference_contact_for_step()

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
            "has_reference_contact_mask": self.has_reference_contact_mask,
            "use_reference_contact_mask": self.use_reference_contact_mask,
            "use_functional_foot_contact": self.use_functional_foot_contact,
            "functional_contact_threshold": self.functional_contact_threshold,
            "residual_scale_max": float(np.max(self.per_joint_residual_scale)),
            "residual_scale_mean": float(np.mean(self.per_joint_residual_scale)),
            "include_contact_phase_observation": self.include_contact_phase_observation,
            "initial_yaw_degrees": self.initial_yaw_degrees,
            "action_target_smoothing": self.action_target_smoothing,
            "reward_version": "v55_large_residual_swing_lift",
            "residual_alpha": float(self.last_residual_alpha),
            "reference_state_initialization": self.reference_state_initialization,
            "rsi_active_this_episode": self.rsi_active_this_episode,
            "rsi_frame_this_episode": self.rsi_frame_this_episode,
            "left_contact": bool(self.last_foot_info["left_contact"]),
            "right_contact": bool(self.last_foot_info["right_contact"]),
            "collision_left_contact": bool(self.last_foot_info.get("collision_left_contact", False)),
            "collision_right_contact": bool(self.last_foot_info.get("collision_right_contact", False)),
            "left_foot_slip": float(self.last_foot_info["left_foot_slip"]),
            "right_foot_slip": float(self.last_foot_info["right_foot_slip"]),
            "left_foot_clearance": float(self.last_foot_info["left_foot_clearance"]),
            "right_foot_clearance": float(self.last_foot_info["right_foot_clearance"]),
            "left_expected_contact": left_expected,
            "right_expected_contact": right_expected,
        }

        info.update(self.last_reward_terms)

        self.previous_action = action.copy()
        self._update_previous_foot_positions()
        self.episode_step += 1

        return observation, reward, terminated, truncated, info

    def close(self):
        pass