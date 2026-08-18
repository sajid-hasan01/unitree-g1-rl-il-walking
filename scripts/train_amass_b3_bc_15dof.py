from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCPolicy(nn.Module):
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def denorm(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x_norm * std + mean


@torch.no_grad()
def evaluate(
    model: BCPolicy,
    obs: torch.Tensor,
    target_norm: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()

    pred_norm = model(obs)
    pred = denorm(pred_norm, action_mean, action_std)
    target = denorm(target_norm, action_mean, action_std)

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
    parser.add_argument("--dataset", type=str, default="datasets/processed/g1_amass_b3_walk1_il_15dof.npz")
    parser.add_argument("--out_model", type=str, default="models/g1_amass_b3_bc_walking_best.pt")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=800)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--velocity_weight", type=float, default=0.001)
    parser.add_argument("--position_weight", type=float, default=10.0)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    out_model_path = Path(args.out_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    obs = np.asarray(data["il_observations"], dtype=np.float32)
    actions = np.asarray(data["il_actions"], dtype=np.float32)
    q_current = np.asarray(data["q_current"], dtype=np.float32)

    fps = 30.0
    if "fps" in data.files:
        fps_arr = np.asarray(data["fps"]).reshape(-1)
        if len(fps_arr) > 0:
            fps = float(fps_arr[0])
    dt = 1.0 / fps

    joint_names = [str(x) for x in data["controlled_joint_names"]]

    n = obs.shape[0]
    split = int(n * args.train_ratio)
    split = max(2, min(split, n - 2))

    train_obs_np = obs[:split]
    train_actions_np = actions[:split]
    train_q_current_np = q_current[:split]

    val_obs_np = obs[split:]
    val_actions_np = actions[split:]
    val_q_current_np = q_current[split:]

    obs_mean_np, obs_std_np = tensor_stats(train_obs_np)
    action_mean_np, action_std_np = tensor_stats(train_actions_np)

    train_obs_norm_np = (train_obs_np - obs_mean_np) / obs_std_np
    train_actions_norm_np = (train_actions_np - action_mean_np) / action_std_np

    val_obs_norm_np = (val_obs_np - obs_mean_np) / obs_std_np
    val_actions_norm_np = (val_actions_np - action_mean_np) / action_std_np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_obs = torch.tensor(train_obs_norm_np, dtype=torch.float32, device=device)
    train_actions_norm = torch.tensor(train_actions_norm_np, dtype=torch.float32, device=device)
    train_actions_raw = torch.tensor(train_actions_np, dtype=torch.float32, device=device)
    train_q_current = torch.tensor(train_q_current_np, dtype=torch.float32, device=device)

    val_obs = torch.tensor(val_obs_norm_np, dtype=torch.float32, device=device)
    val_actions_norm = torch.tensor(val_actions_norm_np, dtype=torch.float32, device=device)

    obs_mean = torch.tensor(obs_mean_np, dtype=torch.float32, device=device)
    obs_std = torch.tensor(obs_std_np, dtype=torch.float32, device=device)
    action_mean = torch.tensor(action_mean_np, dtype=torch.float32, device=device)
    action_std = torch.tensor(action_std_np, dtype=torch.float32, device=device)

    input_dim = obs.shape[1]
    action_dim = actions.shape[1]
    hidden_dims = [256, 256, 128]

    model = BCPolicy(input_dim=input_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    print("=" * 100)
    print("AMASS B3 15-DOF Behavior Cloning Training")
    print("=" * 100)
    print("Dataset:", dataset_path)
    print("Output model:", out_model_path)
    print("Device:", device)
    print("Samples:", n)
    print("Train samples:", split)
    print("Val samples:", n - split)
    print("Input dim:", input_dim)
    print("Action dim:", action_dim)
    print("FPS:", fps)
    print("Hidden dims:", hidden_dims)
    print("=" * 100)

    for epoch in range(1, args.epochs + 1):
        model.train()

        pred_norm = model(train_obs)
        pred_raw = denorm(pred_norm, action_mean, action_std)

        loss_norm = F.mse_loss(pred_norm, train_actions_norm)
        loss_pos = F.mse_loss(pred_raw, train_actions_raw)

        if pred_raw.shape[0] > 1:
            pred_diff = pred_raw[1:] - pred_raw[:-1]
            true_diff = train_actions_raw[1:] - train_actions_raw[:-1]
            loss_smooth = F.mse_loss(pred_diff, true_diff)
        else:
            loss_smooth = torch.tensor(0.0, device=device)

        pred_vel = (pred_raw - train_q_current) / dt
        true_vel = (train_actions_raw - train_q_current) / dt
        loss_vel = F.mse_loss(pred_vel, true_vel)

        loss = (
            loss_norm
            + args.position_weight * loss_pos
            + args.smooth_weight * loss_smooth
            + args.velocity_weight * loss_vel
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        val_metrics = evaluate(
            model=model,
            obs=val_obs,
            target_norm=val_actions_norm,
            action_mean=action_mean,
            action_std=action_std,
        )

        val_mse = val_metrics["mse_rad2"]

        improved = val_mse < best_val_mse

        if improved:
            best_val_mse = val_mse
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "action_dim": action_dim,
                "hidden_dims": hidden_dims,
                "obs_mean": obs_mean_np,
                "obs_std": obs_std_np,
                "action_mean": action_mean_np,
                "action_std": action_std_np,
                "controlled_joint_names": joint_names,
                "dataset_path": str(dataset_path),
                "source_raw_npz": str(data["source_raw_npz"]) if "source_raw_npz" in data.files else "",
                "fps": fps,
                "best_epoch": best_epoch,
                "best_val_mse_rad2": best_val_mse,
                "best_val_mae_rad": val_metrics["mae_rad"],
                "config": vars(args),
            }

            torch.save(checkpoint, out_model_path)
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 100 == 0 or improved:
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
        "samples": n,
        "train_samples": split,
        "val_samples": n - split,
        "input_dim": input_dim,
        "action_dim": action_dim,
        "fps": fps,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", metrics_path)


if __name__ == "__main__":
    main()

