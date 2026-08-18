from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np


from envs.g1_closed_loop_tracking_env_v2 import (
    G1ClosedLoopTrackingEnvV2,
)


class G1ClosedLoopTrackingEnvV3(
    G1ClosedLoopTrackingEnvV2
):

    def __init__(
        self,

        dataset_path=
            "datasets/processed/"
            "g1_amass_walking_tracking_50hz_v4_grounded.npz",

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
            dataset_path=dataset_path,
            model_path=model_path,

            control_hz=control_hz,

            fixed_start_frame=fixed_start_frame,

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
            root / dataset_path,
            allow_pickle=True,
        )


        required = [
            "left_foot_pos",
            "right_foot_pos",

            "left_foot_vel",
            "right_foot_vel",

            "support_mask",
            "contact_mask",
        ]


        for key in required:

            if key not in d:

                raise RuntimeError(
                    f"Tracking V3 dataset "
                    f"missing key: {key}"
                )


        self.ref_left_foot_pos = (
            np.asarray(
                d["left_foot_pos"],
                dtype=np.float32,
            )
        )


        self.ref_right_foot_pos = (
            np.asarray(
                d["right_foot_pos"],
                dtype=np.float32,
            )
        )


        self.ref_left_foot_vel = (
            np.asarray(
                d["left_foot_vel"],
                dtype=np.float32,
            )
        )


        self.ref_right_foot_vel = (
            np.asarray(
                d["right_foot_vel"],
                dtype=np.float32,
            )
        )


        self.ref_support = (
            np.asarray(
                d["support_mask"],
                dtype=np.float32,
            )
        )


        # =========================================================
        # OBSERVATION
        #
        # V2 base observation                        140
        #
        # reference foot positions relative root      6
        # actual foot positions relative root         6
        # foot global position errors                 6
        # reference foot velocities                   6
        # stable-support phase                        2
        # -----------------------------------------------
        # total                                     166
        # =========================================================

        self.observation_space = (
            spaces.Box(
                low=-np.inf,
                high=np.inf,

                shape=(166,),

                dtype=np.float32,
            )
        )


        self.previous_left_foot_pos = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )


        self.previous_right_foot_pos = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )


    # =============================================================
    # FOOT REFERENCE
    # =============================================================

    def _desired_foot_positions(
        self,
        idx,
    ):

        translation = (
            self.episode_world_origin
            - self.episode_ref_origin
        )


        left = (
            self.ref_left_foot_pos[
                idx
            ].astype(
                np.float64
            )
            + translation
        )


        right = (
            self.ref_right_foot_pos[
                idx
            ].astype(
                np.float64
            )
            + translation
        )


        return (
            left,
            right,
        )


    # =============================================================
    # OBSERVATION
    # =============================================================

    def _build_obs(
        self,
    ):

        base = super()._build_obs()


        idx = (
            self._reference_index()
        )


        root_actual = (
            self.data.qpos[
                0:3
            ].astype(
                np.float64
            )
        )


        actual_left = (
            self.data.site_xpos[
                self.left_site
            ].astype(
                np.float64
            )
        )


        actual_right = (
            self.data.site_xpos[
                self.right_site
            ].astype(
                np.float64
            )
        )


        desired_left, desired_right = (
            self._desired_foot_positions(
                idx
            )
        )


        # Reference foot vectors relative to root.
        ref_left_relative = (
            self.ref_left_foot_pos[
                idx
            ]
            - self.ref_root_pos[
                idx
            ]
        )


        ref_right_relative = (
            self.ref_right_foot_pos[
                idx
            ]
            - self.ref_root_pos[
                idx
            ]
        )


        # Actual foot vectors relative to current root.
        actual_left_relative = (
            actual_left
            - root_actual
        )


        actual_right_relative = (
            actual_right
            - root_actual
        )


        left_error = (
            desired_left
            - actual_left
        )


        right_error = (
            desired_right
            - actual_right
        )


        ref_foot_velocity = (
            np.concatenate(
                [
                    self.ref_left_foot_vel[
                        idx
                    ],

                    self.ref_right_foot_vel[
                        idx
                    ],
                ]
            )
            * 0.20
        )


        extra = np.concatenate(
            [
                ref_left_relative,

                ref_right_relative,

                actual_left_relative,

                actual_right_relative,

                left_error,

                right_error,

                ref_foot_velocity,

                self.ref_support[
                    idx
                ],
            ]
        ).astype(
            np.float32
        )


        obs = np.concatenate(
            [
                base,
                extra,
            ]
        ).astype(
            np.float32
        )


        if obs.shape != (
            166,
        ):

            raise RuntimeError(
                f"Unexpected V3 observation "
                f"shape: {obs.shape}"
            )


        return np.nan_to_num(
            obs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


    # =============================================================
    # REWARD
    # =============================================================

    def _reward(
        self,
        action,
    ):

        (
            base_reward,
            terms,
        ) = super()._reward(
            action
        )


        idx = (
            self._reference_index()
        )


        actual_left = (
            self.data.site_xpos[
                self.left_site
            ].astype(
                np.float64
            )
        )


        actual_right = (
            self.data.site_xpos[
                self.right_site
            ].astype(
                np.float64
            )
        )


        desired_left, desired_right = (
            self._desired_foot_positions(
                idx
            )
        )


        left_error = (
            actual_left
            - desired_left
        )


        right_error = (
            actual_right
            - desired_right
        )


        # ---------------------------------------------------------
        # FOOT POSITION
        # ---------------------------------------------------------

        xy_error = np.array(
            [
                left_error[0],
                left_error[1],

                right_error[0],
                right_error[1],
            ],
            dtype=np.float64,
        )


        z_error = np.array(
            [
                left_error[2],
                right_error[2],
            ],
            dtype=np.float64,
        )


        foot_xy_reward = float(
            np.exp(
                -np.mean(
                    (
                        xy_error
                        / 0.10
                    ) ** 2
                )
            )
        )


        foot_z_reward = float(
            np.exp(
                -np.mean(
                    (
                        z_error
                        / 0.05
                    ) ** 2
                )
            )
        )


        # ---------------------------------------------------------
        # FOOT VELOCITY
        #
        # Computed from actual end-effector displacement.
        # ---------------------------------------------------------

        actual_left_velocity = (
            actual_left
            - self.previous_left_foot_pos
        ) / self.control_dt


        actual_right_velocity = (
            actual_right
            - self.previous_right_foot_pos
        ) / self.control_dt


        reference_left_velocity = (
            self.ref_left_foot_vel[
                idx
            ].astype(
                np.float64
            )
        )


        reference_right_velocity = (
            self.ref_right_foot_vel[
                idx
            ].astype(
                np.float64
            )
        )


        velocity_error = np.concatenate(
            [
                actual_left_velocity
                - reference_left_velocity,

                actual_right_velocity
                - reference_right_velocity,
            ]
        )


        foot_velocity_reward = float(
            np.exp(
                -np.mean(
                    (
                        velocity_error
                        / 0.80
                    ) ** 2
                )
            )
        )


        # ---------------------------------------------------------
        # SUPPORT FOOT SLIP
        # ---------------------------------------------------------

        support = (
            self.ref_support[
                idx
            ] > 0.5
        )


        support_slip_values = []


        if support[0]:

            support_slip_values.append(
                float(
                    np.linalg.norm(
                        actual_left_velocity[
                            0:2
                        ]
                    )
                )
            )


        if support[1]:

            support_slip_values.append(
                float(
                    np.linalg.norm(
                        actual_right_velocity[
                            0:2
                        ]
                    )
                )
            )


        if support_slip_values:

            support_slip = float(
                np.mean(
                    [
                        max(
                            v - 0.30,
                            0.0,
                        ) ** 2

                        for v
                        in support_slip_values
                    ]
                )
            )

        else:

            support_slip = 0.0


        # ---------------------------------------------------------
        # TOTAL V3 ADDITION
        # ---------------------------------------------------------

        reward = (
            base_reward

            + 0.70
            * foot_xy_reward

            + 0.45
            * foot_z_reward

            + 0.25
            * foot_velocity_reward

            - 0.15
            * support_slip
        )


        foot_position_error = float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [
                            left_error,
                            right_error,
                        ]
                    ) ** 2
                )
            )
        )


        foot_velocity_error = float(
            np.sqrt(
                np.mean(
                    velocity_error ** 2
                )
            )
        )


        terms.update(
            {
                "foot_xy_reward":
                    foot_xy_reward,

                "foot_z_reward":
                    foot_z_reward,

                "foot_velocity_reward":
                    foot_velocity_reward,

                "foot_position_error":
                    foot_position_error,

                "foot_velocity_error":
                    foot_velocity_error,

                "support_slip_penalty":
                    support_slip,
            }
        )


        return (
            float(
                reward
            ),

            terms,
        )


    # =============================================================
    # RESET
    # =============================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        obs, info = super().reset(
            seed=seed,
            options=options,
        )


        self.previous_left_foot_pos = (
            self.data.site_xpos[
                self.left_site
            ].astype(
                np.float64
            ).copy()
        )


        self.previous_right_foot_pos = (
            self.data.site_xpos[
                self.right_site
            ].astype(
                np.float64
            ).copy()
        )


        return (
            self._build_obs(),
            info,
        )


    # =============================================================
    # STEP
    # =============================================================

    def step(
        self,
        action,
    ):

        result = super().step(
            action
        )


        self.previous_left_foot_pos = (
            self.data.site_xpos[
                self.left_site
            ].astype(
                np.float64
            ).copy()
        )


        self.previous_right_foot_pos = (
            self.data.site_xpos[
                self.right_site
            ].astype(
                np.float64
            ).copy()
        )


        return result
