from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


from envs.g1_closed_loop_tracking_env import (
    G1ClosedLoopTrackingEnv,
)


class G1ClosedLoopTrackingEnvV2(
    G1ClosedLoopTrackingEnv
):

    def __init__(
        self,

        dataset_path=
            "datasets/processed/"
            "g1_amass_walking_tracking_50hz_v2.npz",

        model_path=
            "third_party/mujoco_menagerie/"
            "unitree_g1/scene.xml",

        control_hz=50.0,

        fixed_start_frame=None,

        random_reference_start=True,

        reset_position_noise=0.0,

        reset_velocity_noise=0.0,

        target_smoothing=0.20,

        max_episode_steps=400,
    ):


        super().__init__(
            dataset_path=
                dataset_path,

            model_path=
                model_path,

            control_hz=
                control_hz,

            # V2 does NOT use a fixed command velocity.
            target_velocity=
                0.0,

            fixed_start_frame=
                fixed_start_frame,

            random_reference_start=
                random_reference_start,

            reset_position_noise=
                reset_position_noise,

            reset_velocity_noise=
                reset_velocity_noise,

            target_smoothing=
                target_smoothing,

            max_episode_steps=
                max_episode_steps,
        )


        root = Path(
            __file__
        ).resolve().parents[1]


        d = np.load(
            root
            / dataset_path,
            allow_pickle=True,
        )


        self.ref_root_quat = (
            np.asarray(
                d[
                    "root_quat_wxyz"
                ],
                dtype=np.float32,
            )
        )


        self.ref_root_ang_vel = (
            np.asarray(
                d[
                    "root_ang_vel_local"
                ],
                dtype=np.float32,
            )
        )


        if (
            self.ref_root_quat.shape
            != (
                self.num_frames,
                4,
            )
        ):

            raise RuntimeError(
                "Bad root quaternion shape."
            )


        if (
            self.ref_root_ang_vel.shape
            != (
                self.num_frames,
                3,
            )
        ):

            raise RuntimeError(
                "Bad root angular velocity shape."
            )


        # ---------------------------------------------------------
        # Observation V2 = 140
        #
        # Joint reference/current/errors:
        #   ref q            15
        #   ref qdot         15
        #   actual q         15
        #   actual qdot      15
        #   q error          15
        #   qdot error       15
        #                     --
        #                     90
        #
        # Root:
        #   actual quat       4
        #   reference quat    4
        #   actual qvel       6
        #   reference qvel    6
        #   root qvel error   6
        #                     --
        #                     26
        #
        # contacts            4
        # phase               2
        # root position error 3
        # previous action    15
        #
        # total             140
        # ---------------------------------------------------------

        self.observation_space = (
            spaces.Box(
                low=-np.inf,
                high=np.inf,

                shape=(140,),

                dtype=np.float32,
            )
        )


        self.episode_ref_origin = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )


        self.episode_world_origin = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )


    # =============================================================
    # ROOT REFERENCE HELPERS
    # =============================================================

    def _root_reference(
        self,
    ):

        idx = self._reference_index()

        return (
            self.ref_root_pos[
                idx
            ],

            self.ref_root_quat[
                idx
            ],

            self.ref_root_vel[
                idx
            ],

            self.ref_root_ang_vel[
                idx
            ],

            idx,
        )


    def _desired_root_position(
        self,
        idx,
    ):

        relative_motion = (
            self.ref_root_pos[
                idx
            ].astype(
                np.float64
            )
            - self.episode_ref_origin
        )


        return (
            self.episode_world_origin
            + relative_motion
        )


    @staticmethod
    def _quat_tracking_angle(
        qa,
        qb,
    ):

        qa = np.asarray(
            qa,
            dtype=np.float64,
        )

        qb = np.asarray(
            qb,
            dtype=np.float64,
        )


        qa = qa / max(
            np.linalg.norm(
                qa
            ),
            1e-12,
        )


        qb = qb / max(
            np.linalg.norm(
                qb
            ),
            1e-12,
        )


        dot = abs(
            float(
                np.dot(
                    qa,
                    qb,
                )
            )
        )


        dot = float(
            np.clip(
                dot,
                -1.0,
                1.0,
            )
        )


        return float(
            2.0
            * math.acos(
                dot
            )
        )


    # =============================================================
    # OBSERVATION
    # =============================================================

    def _build_obs(
        self,
    ):

        (
            ref_q,
            ref_qd,
            ref_root_linear,
            expected_contact,
            idx,
        ) = self._reference()


        q = self._joint_positions()

        qd = self._joint_velocities()


        q_error = (
            ref_q - q
        )


        qd_error = (
            ref_qd - qd
        )


        actual_quat = (
            self.data.qpos[
                3:7
            ].astype(
                np.float32
            )
        )


        ref_quat = (
            self.ref_root_quat[
                idx
            ].astype(
                np.float32
            )
        )


        actual_root_vel = (
            self.data.qvel[
                0:6
            ].astype(
                np.float32
            )
        )


        ref_root_vel = (
            np.concatenate(
                [
                    self.ref_root_vel[
                        idx
                    ],

                    self.ref_root_ang_vel[
                        idx
                    ],
                ]
            ).astype(
                np.float32
            )
        )


        root_vel_error = (
            ref_root_vel
            - actual_root_vel
        )


        desired_root = (
            self._desired_root_position(
                idx
            )
        )


        current_root = (
            self.data.qpos[
                0:3
            ].astype(
                np.float64
            )
        )


        root_position_error = (
            desired_root
            - current_root
        ).astype(
            np.float32
        )


        left_contact = float(
            self._foot_contact(
                self.left_sole
            )
        )


        right_contact = float(
            self._foot_contact(
                self.right_sole
            )
        )


        obs = np.concatenate(
            [
                ref_q.astype(
                    np.float32
                ),

                (
                    0.10
                    * ref_qd
                ).astype(
                    np.float32
                ),

                q.astype(
                    np.float32
                ),

                (
                    0.10
                    * qd
                ).astype(
                    np.float32
                ),

                q_error.astype(
                    np.float32
                ),

                (
                    0.10
                    * qd_error
                ).astype(
                    np.float32
                ),

                actual_quat,

                ref_quat,

                (
                    0.10
                    * actual_root_vel
                ),

                (
                    0.10
                    * ref_root_vel
                ),

                (
                    0.10
                    * root_vel_error
                ),

                np.array(
                    [
                        left_contact,
                        right_contact,
                    ],
                    dtype=np.float32,
                ),

                expected_contact.astype(
                    np.float32
                ),

                self._phase(
                    idx
                ),

                root_position_error,

                self.previous_action.astype(
                    np.float32
                ),
            ]
        )


        if obs.shape != (
            140,
        ):

            raise RuntimeError(
                f"Bad V2 obs shape: "
                f"{obs.shape}"
            )


        return np.nan_to_num(
            obs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


    # =============================================================
    # REWARD
    #
    # True motion tracking:
    #
    # NOT a fixed forward-speed task.
    # =============================================================

    def _reward(
        self,
        action,
    ):

        (
            ref_q,
            ref_qd,
            ref_root_linear,
            expected_contact,
            idx,
        ) = self._reference()


        q = self._joint_positions()

        qd = self._joint_velocities()


        q_error = (
            q - ref_q
        )

        qd_error = (
            qd - ref_qd
        )


        # ---------------------------------------------------------
        # JOINT TRACKING
        # ---------------------------------------------------------

        pose_reward = float(
            np.exp(
                -np.mean(
                    (
                        q_error
                        / 0.25
                    ) ** 2
                )
            )
        )


        joint_velocity_reward = float(
            np.exp(
                -np.mean(
                    (
                        qd_error
                        / 2.5
                    ) ** 2
                )
            )
        )


        # ---------------------------------------------------------
        # ROOT ORIENTATION
        # ---------------------------------------------------------

        orientation_error = (
            self._quat_tracking_angle(
                self.data.qpos[
                    3:7
                ],

                self.ref_root_quat[
                    idx
                ],
            )
        )


        orientation_reward = float(
            np.exp(
                -(
                    orientation_error
                    / 0.35
                ) ** 2
            )
        )


        # ---------------------------------------------------------
        # ROOT LINEAR VELOCITY
        # ---------------------------------------------------------

        actual_linear = (
            self.data.qvel[
                0:3
            ]
        )


        linear_error = (
            actual_linear
            - self.ref_root_vel[
                idx
            ]
        )


        linear_velocity_reward = float(
            np.exp(
                -np.mean(
                    (
                        linear_error
                        / 0.40
                    ) ** 2
                )
            )
        )


        # ---------------------------------------------------------
        # ROOT ANGULAR VELOCITY
        # ---------------------------------------------------------

        actual_angular = (
            self.data.qvel[
                3:6
            ]
        )


        angular_error = (
            actual_angular
            - self.ref_root_ang_vel[
                idx
            ]
        )


        angular_velocity_reward = float(
            np.exp(
                -np.mean(
                    (
                        angular_error
                        / 2.0
                    ) ** 2
                )
            )
        )


        # ---------------------------------------------------------
        # ROOT POSITION RELATIVE TO EPISODE START
        # ---------------------------------------------------------

        desired_root = (
            self._desired_root_position(
                idx
            )
        )


        root_error = (
            self.data.qpos[
                0:3
            ]
            - desired_root
        )


        root_position_reward = float(
            np.exp(
                -(
                    (
                        root_error[0]
                        / 0.30
                    ) ** 2

                    +

                    (
                        root_error[1]
                        / 0.12
                    ) ** 2

                    +

                    (
                        root_error[2]
                        / 0.10
                    ) ** 2
                )
            )
        )


        # ---------------------------------------------------------
        # BASIC PHYSICAL UPRIGHTNESS
        # ---------------------------------------------------------

        up = self._up_z()


        upright_reward = float(
            np.exp(
                -(
                    (
                        1.0
                        - up
                    )
                    / 0.20
                ) ** 2
            )
        )


        # ---------------------------------------------------------
        # CONTACT
        #
        # Current dataset has no valid labels.
        # This becomes active automatically later.
        # ---------------------------------------------------------

        actual_contact = np.array(
            [
                float(
                    self._foot_contact(
                        self.left_sole
                    )
                ),

                float(
                    self._foot_contact(
                        self.right_sole
                    )
                ),
            ],
            dtype=np.float32,
        )


        if self.has_reference_contact:

            contact_match = float(
                np.mean(
                    (
                        actual_contact
                        > 0.5
                    )
                    ==
                    (
                        expected_contact
                        > 0.5
                    )
                )
            )

        else:

            contact_match = 0.0


        # ---------------------------------------------------------
        # CONTROL REGULARIZATION
        # ---------------------------------------------------------

        action_penalty = (
            0.005
            * float(
                np.mean(
                    action ** 2
                )
            )
        )


        action_rate_penalty = (
            0.020
            * float(
                np.mean(
                    (
                        action
                        - self.previous_action
                    ) ** 2
                )
            )
        )


        # ---------------------------------------------------------
        # TOTAL
        # ---------------------------------------------------------

        reward = (
            1.50
            * pose_reward

            + 0.45
            * joint_velocity_reward

            + 1.10
            * orientation_reward

            + 0.85
            * linear_velocity_reward

            + 0.30
            * angular_velocity_reward

            + 0.90
            * root_position_reward

            + 0.50
            * upright_reward

            + (
                0.50
                * contact_match
                if self.has_reference_contact
                else 0.0
            )

            - action_penalty

            - action_rate_penalty
        )


        return (
            float(
                reward
            ),

            {
                "pose_reward":
                    pose_reward,

                "joint_velocity_reward":
                    joint_velocity_reward,

                "orientation_reward":
                    orientation_reward,

                "root_position_reward":
                    root_position_reward,

                "linear_velocity_reward":
                    linear_velocity_reward,

                "angular_velocity_reward":
                    angular_velocity_reward,

                "upright_reward":
                    upright_reward,

                "contact_match":
                    contact_match,

                "q_error_rms":
                    float(
                        np.sqrt(
                            np.mean(
                                q_error ** 2
                            )
                        )
                    ),

                "qd_error_rms":
                    float(
                        np.sqrt(
                            np.mean(
                                qd_error ** 2
                            )
                        )
                    ),

                "orientation_error_deg":
                    float(
                        math.degrees(
                            orientation_error
                        )
                    ),

                "root_position_error":
                    float(
                        np.linalg.norm(
                            root_error
                        )
                    ),

                "root_velocity_error":
                    float(
                        np.linalg.norm(
                            linear_error
                        )
                    ),
            },
        )


    # =============================================================
    # FULL REFERENCE-STATE INITIALIZATION
    # =============================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        # Temporary valid origins because parent's reset()
        # invokes our overridden _build_obs().
        self.episode_ref_origin = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )


        self.episode_world_origin = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )


        obs, info = super().reset(
            seed=seed,
            options=options,
        )


        frame = int(
            info[
                "reset_reference_frame"
            ]
        )


        # ---------------------------------------------------------
        # Full root pose
        # ---------------------------------------------------------

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0


        reference_z = float(
            self.ref_root_pos[
                frame,
                2,
            ]
        )


        self.data.qpos[2] = float(
            np.clip(
                reference_z,

                float(
                    self.stand_qpos[2]
                )
                - 0.12,

                float(
                    self.stand_qpos[2]
                )
                + 0.12,
            )
        )


        self.data.qpos[
            3:7
        ] = (
            self.ref_root_quat[
                frame
            ]
        )


        # ---------------------------------------------------------
        # Full root velocity
        #
        # qvel[0:3] global linear velocity
        # qvel[3:6] local-body angular velocity
        # ---------------------------------------------------------

        self.data.qvel[
            0:3
        ] = (
            self.ref_root_vel[
                frame
            ]
        )


        self.data.qvel[
            3:6
        ] = (
            self.ref_root_ang_vel[
                frame
            ]
        )


        # Parent already initialized joint q/qdot
        # from the V2 reference.


        self.previous_targets = (
            self.ref_q[
                frame
            ].astype(
                np.float64
            ).copy()
        )


        for i, item in enumerate(
            self.joint_info
        ):

            self.data.ctrl[
                item["aid"]
            ] = self._clip_ctrl(
                item["aid"],
                self.previous_targets[
                    i
                ],
            )


        for aid, target in (
            self.upper_body
        ):

            self.data.ctrl[
                aid
            ] = self._clip_ctrl(
                aid,
                target,
            )


        mujoco.mj_forward(
            self.model,
            self.data,
        )


        # ---------------------------------------------------------
        # Episode-local tracking coordinates
        # ---------------------------------------------------------

        self.episode_ref_origin = (
            self.ref_root_pos[
                frame
            ].astype(
                np.float64
            ).copy()
        )


        self.episode_world_origin = (
            self.data.qpos[
                0:3
            ].astype(
                np.float64
            ).copy()
        )


        info[
            "root_reference_initialized"
        ] = True


        info[
            "reference_vx"
        ] = float(
            self.ref_root_vel[
                frame,
                0,
            ]
        )


        info[
            "reference_yaw_deg"
        ] = float(
            math.degrees(
                self._yaw(
                    self.ref_root_quat[
                        frame
                    ]
                )
            )
        )


        return (
            self._build_obs(),
            info,
        )
