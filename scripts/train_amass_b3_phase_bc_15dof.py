from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseBCPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.LayerNorm(hidden_dim))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_phase_features(num_frames: int, num_harmonics: int = 4) -> np.ndarray:
    """
    Non-autoregressive phase/progress features.

    For a finite AMASS walking clip, progress is important because the B3 motion is
    not a perfect cyclic loop. Harmonic phase features help represent smooth gait.
    """
    progress = np.linspace(0.0, 1.0, num_frames, dtype=np.float32).reshape(-1, 1)

    features = [progress]

    for k in range(1, num_harmonics + 1):
        angle = 2.0 * math.pi * k * progress
        features.append(np.sin(angle).astype(np.float32))
        features.append(np.cos(angle).astype(np.float32))

    return np.concatenate(features, axis=1).astype(np.float32)


def stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


@torch.no_grad()
def evaluate(
    model: PhaseBCPolicy,
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()

    pred_norm = model(x_norm)
    pred = pred_norm * y_std + y_mean
    target = y_norm * y_std + y_mean

    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    max_abs = torch.max(torch.abs(pred - target)).item()

    return {
        "mse_rad2": mse,
        "mae_rad": mae,
        "max_abs_rad": max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="experiments/amass_b3_15dof_il/processed/g1_amass_b3_walk1_il_15dof.npz",
    )
    parser.add_argument(
        "--out_model",
        type=str,
        default="experiments/amass_b3_15dof_il_phase_bc/models/g1_amass_b3_phase_bc_15dof_best.pt",
    )
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_harmonics", type=int, default=4)
    parser.add_argument("--smooth_weight", type=float, default=0.10)
    parser.add_argument("--velocity_weight", type=float, default=0.02)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    out_model_path = Path(args.out_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    joint_pos_15 = np.asarray(data["joint_pos_15"], dtype=np.float32)
    joint_vel_15 = np.asarray(data["joint_vel_15"], dtype=np.float32)
    joint_names = [str(x) for x in data["controlled_joint_names"]]

    fps = 30.0
    if "fps" in data.files:
        fps_arr = np.asarray(data["fps"]).reshape(-1)
        if len(fps_arr) > 0:
            fps = float(fps_arr[0])

    x = build_phase_features(
        num_frames=joint_pos_15.shape[0],
        num_harmonics=args.num_harmonics,
    )

    y = joint_pos_15.astype(np.float32)

    # Validation is every 5th frame so validation covers the whole motion,
    # instead of only the final segment.
    all_idx = np.arange(len(x))
    val_mask = (all_idx % 5) == 0
    train_mask = ~val_mask

    x_train = x[train_mask]
    y_train = y[train_mask]
    x_val = x[val_mask]
    y_val = y[val_mask]

    x_mean_np, x_std_np = stats(x_train)
    y_mean_np, y_std_np = stats(y_train)

    x_train_norm = (x_train - x_mean_np) / x_std_np
    y_train_norm = (y_train - y_mean_np) / y_std_np
    x_val_norm = (x_val - x_mean_np) / x_std_np
    y_val_norm = (y_val - y_mean_np) / y_std_np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_t = torch.tensor(x_train_norm, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32, device=device)
    y_train_raw_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    x_val_t = torch.tensor(x_val_norm, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_norm, dtype=torch.float32, device=device)

    x_all_norm = (x - x_mean_np) / x_std_np
    x_all_t = torch.tensor(x_all_norm, dtype=torch.float32, device=device)
    y_all_raw_t = torch.tensor(y, dtype=torch.float32, device=device)

    x_mean = torch.tensor(x_mean_np, dtype=torch.float32, device=device)
    x_std = torch.tensor(x_std_np, dtype=torch.float32, device=device)
    y_mean = torch.tensor(y_mean_np, dtype=torch.float32, device=device)
    y_std = torch.tensor(y_std_np, dtype=torch.float32, device=device)

    input_dim = x.shape[1]
    action_dim = y.shape[1]
    hidden_dims = [256, 256, 128]

    model = PhaseBCPolicy(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    print("=" * 100)
    print("AMASS B3 15-DOF PHASE-CONDITIONED BC TRAINING")
    print("=" * 100)
    print("Dataset:", dataset_path)
    print("Output model:", out_model_path)
    print("Device:", device)
    print("Frames:", len(x))
    print("Train frames:", len(x_train))
    print("Val frames:", len(x_val))
    print("Input dim:", input_dim)
    print("Action dim:", action_dim)
    print("FPS:", fps)
    print("Hidden dims:", hidden_dims)
    print("=" * 100)

    for epoch in range(1, args.epochs + 1):
        model.train()

        pred_train_norm = model(x_train_t)
        loss_norm = F.mse_loss(pred_train_norm, y_train_t)

        pred_all_norm = model(x_all_t)
        pred_all_raw = pred_all_norm * y_std + y_mean

        loss_pos = F.mse_loss(pred_all_raw, y_all_raw_t)

        if pred_all_raw.shape[0] > 1:
            pred_diff = pred_all_raw[1:] - pred_all_raw[:-1]
            true_diff = y_all_raw_t[1:] - y_all_raw_t[:-1]
            loss_smooth = F.mse_loss(pred_diff, true_diff)

            pred_vel = pred_diff * fps
            true_vel = true_diff * fps
            loss_vel = F.mse_loss(pred_vel, true_vel)
        else:
            loss_smooth = torch.tensor(0.0, device=device)
            loss_vel = torch.tensor(0.0, device=device)

        loss = (
            loss_norm
            + loss_pos
            + args.smooth_weight * loss_smooth
            + args.velocity_weight * loss_vel
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        val_metrics = evaluate(
            model=model,
            x_norm=x_val_t,
            y_norm=y_val_t,
            y_mean=y_mean,
            y_std=y_std,
        )

        val_mse = val_metrics["mse_rad2"]

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "action_dim": action_dim,
                "hidden_dims": hidden_dims,
                "x_mean": x_mean_np,
                "x_std": x_std_np,
                "y_mean": y_mean_np,
                "y_std": y_std_np,
                "num_harmonics": args.num_harmonics,
                "controlled_joint_names": joint_names,
                "dataset_path": str(dataset_path),
                "fps": fps,
                "best_epoch": best_epoch,
                "best_val_mse_rad2": best_val_mse,
                "best_val_mae_rad": val_metrics["mae_rad"],
                "config": vars(args),
            }

            torch.save(checkpoint, out_model_path)
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 100 == 0 or val_mse <= best_val_mse:
            print(
                f"epoch={epoch:05d} "
                f"loss={loss.item():.8f} "
                f"norm={loss_norm.item():.8f} "
                f"pos={loss_pos.item():.8f} "
                f"smooth={loss_smooth.item():.8f} "
                f"vel={loss_vel.item():.8f} "
                f"val_mse={val_metrics['mse_rad2']:.8f} "
                f"val_mae={val_metrics['mae_rad']:.8f} "
                f"best_epoch={best_epoch}"
            )

        if epochs_without_improvement >= args.patience:
            print()
            print("Early stopping triggered.")
            break

    print()
    print("=" * 100)
    print("Training complete")
    print("=" * 100)
    print("Best epoch:", best_epoch)
    print("Best val MSE:", best_val_mse)
    print("Saved best model:", out_model_path)

    metrics_path = out_model_path.with_suffix(".json")
    metrics = {
        "dataset": str(dataset_path),
        "out_model": str(out_model_path),
        "best_epoch": best_epoch,
        "best_val_mse_rad2": best_val_mse,
        "frames": len(x),
        "input_dim": input_dim,
        "action_dim": action_dim,
        "fps": fps,
        "num_harmonics": args.num_harmonics,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", metrics_path)


if __name__ == "__main__":
    main()
