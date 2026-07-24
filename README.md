# RL and IL Based Walking, Balance, and Push-Recovery Control of Unitree G1 in MuJoCo

This project develops a Unitree G1 humanoid robot walking controller in MuJoCo using imitation learning and reinforcement learning.

## Goal

The final goal is to create a single walking system that can:

- walk forward,
- maintain balance,
- recover from external pushes,
- use imitation learning from AMASS-retargeted G1 walking data,
- use reinforcement learning for dynamic stability and push recovery.

## Method

1. Prepare AMASS-retargeted G1 walking data.
2. Train a Behavior Cloning walking policy using imitation learning.
3. Fine-tune the walking policy in MuJoCo using PPO.
4. Add push disturbances and train push recovery.
5. Demonstrate final walking, balance, and push recovery.

## Project Structure

```text
configs/       configuration files
envs/          MuJoCo/Gymnasium environments
scripts/       training and demo scripts
models/        trained model files, ignored by Git
datasets/      datasets, ignored by Git
logs/          training logs, ignored by Git
results/       output videos/figures, ignored by Git
third_party/   external robot model or dependencies