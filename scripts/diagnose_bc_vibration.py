import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DEFAULT_DATASET = (
    "datasets\\processed\\"
    "g1_openhe_walk3_subject4_1320_1620_legs_only_smooth_15dof_mjcontact.npz"
)

DEFAULT_CHECKPOINT = "models\\g1_bc_openhe_kinematic.pt"


class BCPolicy(nn.Module):

    def __init__(
        self,
        input_dim=3,
        hidden_dim=128,
        output_dim=15,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


def to_numpy(
    value,
):
    if isinstance(
        value,
        torch.Tensor,
    ):
        return (
            value
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

    return np.asarray(
        value,
        dtype=np.float32,
    )


def rms(
    value,
):
    value = np.asarray(
        value,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                value
                *
                value
            )
        )
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose temporal vibration "
            "in the OpenHE G1 BC policy."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
    )

    args = parser.parse_args()

    dataset_path = Path(
        args.dataset
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{dataset_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    data = np.load(
        dataset_path,
        allow_pickle=True,
    )

    observations = np.asarray(
        data[
            "il_observations"
        ],
        dtype=np.float32,
    )

    expert = np.asarray(
        data[
            "il_actions"
        ],
        dtype=np.float32,
    )

    joint_names = [
        str(
            x
        )
        for x in
        data[
            "controlled_joint_names"
        ].tolist()
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = BCPolicy(
        input_dim=int(
            checkpoint[
                "input_dim"
            ]
        ),

        hidden_dim=int(
            checkpoint[
                "hidden_dim"
            ]
        ),

        output_dim=int(
            checkpoint[
                "output_dim"
            ]
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    x_mean = to_numpy(
        checkpoint[
            "x_mean"
        ]
    )

    x_std = to_numpy(
        checkpoint[
            "x_std"
        ]
    )

    y_mean = to_numpy(
        checkpoint[
            "y_mean"
        ]
    )

    y_std = to_numpy(
        checkpoint[
            "y_std"
        ]
    )

    x = (
        (
            observations
            -
            x_mean
        )
        /
        x_std
    ).astype(
        np.float32
    )

    with torch.no_grad():

        pred_normalized = (
            model(
                torch
                .from_numpy(
                    x
                )
                .to(
                    device
                )
            )
            .cpu()
            .numpy()
        )

    predicted = (
        pred_normalized
        *
        y_std
        +
        y_mean
    ).astype(
        np.float32
    )

    error = (
        predicted
        -
        expert
    )

    expert_d1 = np.diff(
        expert,
        axis=0,
    )

    predicted_d1 = np.diff(
        predicted,
        axis=0,
    )

    expert_d2 = np.diff(
        expert,
        n=2,
        axis=0,
    )

    predicted_d2 = np.diff(
        predicted,
        n=2,
        axis=0,
    )

    print()

    print(
        "=" * 118
    )

    print(
        "BC VIBRATION DIAGNOSTIC"
    )

    print(
        "=" * 118
    )

    print(
        "Dataset:",
        dataset_path,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Device:",
        device,
    )

    print()

    print(
        "Overall integer-frame "
        "RMSE [rad]:",
        f"{rms(error):.8f}",
    )

    print(
        "Overall integer-frame "
        "max abs error [rad]:",
        f"{float(np.max(np.abs(error))):.8f}",
    )

    print()

    print(
        "idx | joint_name                 | "
        "target_std | pred_RMSE | max_abs  | "
        "d1_err_RMS | expert_d2 | pred_d2  | d2_ratio"
    )

    print(
        "-" * 118
    )

    rows = []

    for (
        index,
        joint_name,
    ) in enumerate(
        joint_names
    ):

        target_std = float(
            np.std(
                expert[
                    :,
                    index
                ],
                dtype=np.float64,
            )
        )

        prediction_rmse = rms(
            error[
                :,
                index
            ]
        )

        max_abs_error = float(
            np.max(
                np.abs(
                    error[
                        :,
                        index
                    ]
                )
            )
        )

        first_difference_error = rms(
            predicted_d1[
                :,
                index
            ]
            -
            expert_d1[
                :,
                index
            ]
        )

        expert_second_difference = rms(
            expert_d2[
                :,
                index
            ]
        )

        predicted_second_difference = rms(
            predicted_d2[
                :,
                index
            ]
        )

        if (
            expert_second_difference
            <
            1e-10
        ):

            if (
                predicted_second_difference
                >
                1e-10
            ):
                second_difference_ratio = (
                    float(
                        "inf"
                    )
                )

            else:
                second_difference_ratio = (
                    1.0
                )

        else:

            second_difference_ratio = (
                predicted_second_difference
                /
                expert_second_difference
            )

        row = {
            "index":
                index,

            "name":
                joint_name,

            "target_std":
                target_std,

            "prediction_rmse":
                prediction_rmse,

            "max_abs_error":
                max_abs_error,

            "first_difference_error":
                first_difference_error,

            "expert_second_difference":
                expert_second_difference,

            "predicted_second_difference":
                predicted_second_difference,

            "second_difference_ratio":
                second_difference_ratio,
        }

        rows.append(
            row
        )

        if np.isinf(
            second_difference_ratio
        ):
            ratio_text = (
                "INF"
            )

        else:
            ratio_text = (
                f"{second_difference_ratio:8.3f}"
            )

        print(
            f"{index:3d} | "
            f"{joint_name:<26s} | "
            f"{target_std:10.6f} | "
            f"{prediction_rmse:9.6f} | "
            f"{max_abs_error:8.6f} | "
            f"{first_difference_error:10.6f} | "
            f"{expert_second_difference:9.6f} | "
            f"{predicted_second_difference:9.6f} | "
            f"{ratio_text}"
        )

    print()

    print(
        "=" * 118
    )

    print(
        "CONSTANT / NEAR-CONSTANT TARGET JOINTS"
    )

    print(
        "=" * 118
    )

    constant_rows = [
        row
        for row in rows
        if (
            row[
                "target_std"
            ]
            <
            1e-6
        )
    ]

    if not constant_rows:

        print(
            "None."
        )

    else:

        for row in (
            constant_rows
        ):

            print(
                f"action[{row['index']:02d}] "
                f"{row['name']}: "
                f"target_std="
                f"{row['target_std']:.10f}, "
                f"prediction_RMSE="
                f"{row['prediction_rmse']:.8f}, "
                f"prediction_max_abs="
                f"{row['max_abs_error']:.8f}, "
                f"predicted_second_difference_RMS="
                f"{row['predicted_second_difference']:.8f}"
            )

    print()

    print(
        "=" * 118
    )

    print(
        "TOP 5 JOINTS BY PREDICTION RMSE"
    )

    print(
        "=" * 118
    )

    sorted_by_rmse = sorted(
        rows,
        key=lambda row:
            row[
                "prediction_rmse"
            ],
        reverse=True,
    )

    for (
        rank,
        row,
    ) in enumerate(
        sorted_by_rmse[
            :5
        ],
        start=1,
    ):

        print(
            f"{rank}. "
            f"action[{row['index']:02d}] "
            f"{row['name']} -> "
            f"RMSE="
            f"{row['prediction_rmse']:.8f} rad, "
            f"max_abs="
            f"{row['max_abs_error']:.8f} rad"
        )

    print()

    print(
        "=" * 118
    )

    print(
        "TOP 5 JOINTS BY TEMPORAL DIFFERENCE ERROR"
    )

    print(
        "=" * 118
    )

    sorted_by_temporal_error = sorted(
        rows,
        key=lambda row:
            row[
                "first_difference_error"
            ],
        reverse=True,
    )

    for (
        rank,
        row,
    ) in enumerate(
        sorted_by_temporal_error[
            :5
        ],
        start=1,
    ):

        print(
            f"{rank}. "
            f"action[{row['index']:02d}] "
            f"{row['name']} -> "
            f"d1_error_RMS="
            f"{row['first_difference_error']:.8f} rad/frame"
        )

    print()

    print(
        "=" * 118
    )


if __name__ == "__main__":
    main()