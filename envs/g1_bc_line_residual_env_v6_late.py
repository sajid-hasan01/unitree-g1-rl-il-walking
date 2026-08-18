import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.g1_bc_line_residual_env_v5_delta import (
    G1BCLineResidualEnvV5Delta,
)


class G1BCLineResidualEnvV6Late(
    G1BCLineResidualEnvV5Delta
):

    """
    Failure-focused residual PPO.

    Full deployment:

        steps 1..94:
            frozen V2 exactly

        step 95 onward:
            frozen V2
            +
            small V6 delta correction

    Training mode:

        reset() internally runs frozen V2 / zero delta
        through step 94.

        PPO therefore learns almost entirely from the late
        instability region instead of wasting updates on the
        already-good early gait.
    """

    def __init__(
        self,
        *args,
        correction_start_step=95,
        correction_ramp_frames=5,
        warm_start_training=False,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.correction_start_step = int(
            correction_start_step
        )

        self.correction_ramp_frames = max(
            1,
            int(
                correction_ramp_frames
            ),
        )

        self.warm_start_training = bool(
            warm_start_training
        )

        self._v6_last_gate = 0.0


    def _gate_for_step(
        self,
        upcoming_step,
    ):

        if upcoming_step < self.correction_start_step:
            return 0.0

        elapsed = (
            upcoming_step
            - self.correction_start_step
            + 1
        )

        return float(
            np.clip(
                elapsed
                / self.correction_ramp_frames,
                0.0,
                1.0,
            )
        )


    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        self._v6_last_gate = 0.0


        # =============================================================
        # TRAINING-ONLY WARM START
        #
        # Internally replay frozen V2 with EXACT zero V6 delta.
        #
        # No reward from these steps is returned to PPO.
        # PPO begins from the physical state immediately before
        # V6 is allowed to act.
        # =============================================================

        if self.warm_start_training:

            target_step = (
                self.correction_start_step
                - 1
            )

            zero_delta = np.zeros(
                self.action_space.shape,
                dtype=np.float32,
            )

            while self.episode_step < target_step:

                (
                    obs,
                    _reward,
                    terminated,
                    truncated,
                    info,
                ) = super().step(
                    zero_delta
                )

                if terminated or truncated:

                    raise RuntimeError(
                        "Frozen V2 terminated during "
                        "V6 warm-start before correction "
                        "activation."
                    )


            # Reset delta-history so the first trainable
            # correction is not penalized against any hidden
            # warm-start history.
            self._previous_delta_action = np.zeros(
                self.action_space.shape,
                dtype=np.float32,
            )

            self._v5_obs = np.asarray(
                obs,
                dtype=np.float32,
            ).copy()


        info = dict(info)

        info.update(
            {
                "v6_correction_start_step":
                    int(
                        self.correction_start_step
                    ),

                "v6_warm_start_training":
                    bool(
                        self.warm_start_training
                    ),

                "v6_gate":
                    0.0,

                "v6_physical_step":
                    int(
                        self.episode_step
                    ),
            }
        )

        return obs, info


    def step(
        self,
        delta_action,
    ):

        upcoming_step = (
            self.episode_step + 1
        )

        gate = self._gate_for_step(
            upcoming_step
        )

        raw_delta = np.asarray(
            delta_action,
            dtype=np.float32,
        ).reshape(
            self.action_space.shape
        )

        raw_delta = np.clip(
            raw_delta,
            -1.0,
            1.0,
        )


        # =============================================================
        # CRITICAL:
        #
        # Before step 95 this becomes EXACT ZERO.
        #
        # Therefore V6 literally cannot alter the successful
        # early V2 trajectory.
        # =============================================================

        effective_delta = (
            gate
            * raw_delta
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = super().step(
            effective_delta
        )


        info = dict(info)

        info.update(
            {
                "v6_gate":
                    float(gate),

                "v6_raw_delta_max":
                    float(
                        np.max(
                            np.abs(
                                raw_delta
                            )
                        )
                    ),

                "v6_effective_delta_max":
                    float(
                        np.max(
                            np.abs(
                                effective_delta
                            )
                        )
                    ),

                "v6_physical_step":
                    int(
                        self.episode_step
                    ),
            }
        )


        self._v6_last_gate = gate


        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )
