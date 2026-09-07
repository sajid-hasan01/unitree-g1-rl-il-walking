# RL and IL-Based Bipedal Robot Walking of Unitree G1 Humanoid Robot in MuJoCo

## CSE499 Senior Design I & II

**North South University**  
Department of Electrical and Computer Engineering

### Team Members
- **Sajid Hasan**
- **Md. Nasim Ahmed**

### Faculty
**Dr. Mohammad Abdul Qayum**

---

## Project Overview

This project investigates **bipedal humanoid walking for the Unitree G1 robot in MuJoCo** using a combination of **Imitation Learning (IL)** and **Reinforcement Learning (RL)**.

The main idea is to first learn a natural walking pattern from human-derived motion using **Behavior Cloning**, and then use **Proximal Policy Optimization (PPO)** to learn small corrections for uneven terrain.

Instead of asking Reinforcement Learning to learn walking completely from scratch, our approach keeps the learned human-style walking motion as the base policy and lets PPO modify it when terrain adaptation is required.

The final kinematic pipeline is:

```text
Human Walking Motion
        ↓
OpenHE G1-Retargeted Motion
        ↓
Motion Preprocessing
        ↓
Behavior Cloning
        ↓
12-DOF Kinematic Walking Policy
        ↓
Residual PPO
        ↓
Uneven-Terrain Joint Adaptation
```

---

## Main Objectives

The main objectives of the project are:

- Generate natural humanoid walking for the Unitree G1 from human motion.
- Use OpenHE G1-retargeted walking data as expert demonstrations.
- Train Behavior Cloning policies to imitate the reference walking motion.
- Compare 15-DOF and 12-DOF lower-body walking representations.
- Use PPO as a residual controller rather than replacing the learned gait.
- Reduce undesirable foot-terrain penetration on uneven terrain.
- Study the trade-off between numerical terrain performance and natural gait preservation.

---

## Unitree G1 Robot

The project uses the **Unitree G1 humanoid robot model in MuJoCo**.

Important MuJoCo model properties include:

```text
nq    = 36
nv    = 35
nu    = 29
nbody = 31
```

The final learning system does not control all 36 generalized coordinates.

For the final kinematic policy, we control **12 lower-body leg joints**:

```text
Left leg  = 6 joints
Right leg = 6 joints
Total     = 12 joints
```

The three waist joints are fixed at their neutral positions:

```text
waist_yaw_joint   = 0 rad
waist_roll_joint  = 0 rad
waist_pitch_joint = 0 rad
```

Therefore:

```text
15-DOF = 12 leg joints + 3 waist joints
12-DOF = 12 leg joints with the waist fixed at neutral
```

---

## OpenHE Motion Dataset

The final imitation-learning pipeline uses a **G1-retargeted human walking sequence from OpenHE**.

Human motion was converted into corresponding Unitree G1 joint trajectories through motion retargeting.

Conceptually:

```text
Human Motion Capture
        ↓
Motion Retargeting
        ↓
Unitree G1 Joint Motion
        ↓
Behavior Cloning Dataset
```

For the final walking sequence:

```text
Selected frames : 1320–1620
Total frames    : 300
Frame rate      : 30 FPS
```

The processed Behavior Cloning input uses:

```text
1. Phase progress
2. Root velocity X
3. Root velocity Y
```

The target output consists of the robot joint angles from the expert OpenHE walking trajectory.

---

## Imitation Learning — Behavior Cloning

Behavior Cloning is used as the main Imitation Learning method.

The network learns to predict the expert robot joint configuration from the input observation.

The training objective is based on Mean Squared Error:

```text
L_BC = (1/N) Σ || πθ(x_t) - y_t ||²
```

where:

```text
x_t       = input observation
πθ(x_t)   = predicted robot joint angles
y_t       = expert OpenHE joint angles
```

The final 12-DOF Behavior Cloning network uses the architecture:

```text
3 → 128 → 128 → 128 → 12
```

with **SiLU activation functions**.

The final trained model is:

```text
g1_bc_openhe_kinematic_12dof.pt
```

Its measured performance is approximately:

```text
RMSE          : 0.00499 rad
MAE           : 0.00318 rad
Maximum Error : 0.06012 rad
```

This shows that the trained network closely reproduces the expert lower-body walking trajectory.

---

## Reinforcement Learning — Residual PPO

After obtaining the Behavior Cloning gait, we add Reinforcement Learning for uneven-terrain adaptation.

We use **standard PPO from Stable-Baselines3**.

The internal PPO optimization algorithm itself was **not modified**.

Instead, PPO is used inside a **Residual Reinforcement Learning architecture**.

Behavior Cloning produces the base walking pose:

```text
q_BC
```

PPO produces a correction:

```text
Δq_PPO
```

The final joint target is:

```text
q_final = q_BC + s × Δq_PPO
```

where:

```text
q_BC       = Behavior Cloning joint target
Δq_PPO     = PPO residual correction
s          = correction scale
q_final    = final joint target
```

Therefore, PPO does not generate the whole walking motion.

It learns **bounded corrections on top of the frozen BC gait**.

---

## PPO Observation and Action Space

The final kinematic PPO environment uses:

```text
Observation dimension : 62
Action dimension      : 12
Action range          : [-1, +1]
```

The 12 PPO actions correspond to the 12 controlled leg joints.

The 62-dimensional observation contains information such as:

- Behavior Cloning observation
- current BC joint pose
- upcoming BC motion
- previous residual correction
- foot-terrain geometry
- deficit relative to the flat BC reference
- local terrain preview
- walking progress

Residual actions are also smoothed before being applied to reduce abrupt joint changes.

---

## Uneven-Terrain Kinematic Environment

The final Residual PPO terrain experiments are **kinematic**.

MuJoCo is used for:

- loading the Unitree G1 model
- forward kinematics
- robot visualization
- foot geometry
- terrain geometry
- distance measurements

The final kinematic environment uses:

```python
mujoco.mj_forward()
```

and does not use:

```python
mujoco.mj_step()
```

for physics simulation.

The root trajectory is prescribed from the reference walking motion.

Therefore, the final Kinematic PPO experiment demonstrates:

> **learned terrain-aware leg-joint adaptation**

It should not be interpreted as full dynamic balance control or real-world physical locomotion.

---

## Main PPO Result

The frozen Behavior Cloning policy was first evaluated on the same uneven terrain.

### BC Baseline

```text
Penetrating frames       : 531
Deficit frames           : 579
Phase-aware penetration  : 499
Squared deficit          : 5.0846
```

The main Residual PPO model used for the final visual comparison is:

```text
g1_kinematic_ppo_v2_smoke_seed425.zip
```

This model was trained for approximately:

```text
8192 timesteps
```

### BC + Residual PPO

```text
Penetrating frames       : 238
Deficit frames           : 312
Phase-aware penetration  : 197
Squared deficit          : 0.9927
```

### Improvement

```text
Penetrating frames reduction : 55.2%
Squared deficit reduction     : 80.5%
```

This shows that Residual PPO learned useful joint corrections that substantially improved foot geometry on the uneven terrain.

---

## Longer PPO Training

We also trained balanced and longer PPO experiments, including checkpoints up to approximately:

```text
65,536 timesteps
```

Some longer-trained policies achieved much stronger penetration metrics.

For example, the 49K checkpoint reduced penetrating frames to approximately:

```text
52 frames
```

with a maximum penetration of approximately:

```text
41.7 mm
```

However, visual evaluation showed that the robot developed a more **bent or crouched leg posture**.

This produced an important research finding:

> Better terrain-contact metrics do not automatically mean better or more natural humanoid walking.

For this reason, the original Reward-V2 8K PPO model is currently considered the **best-looking overall PPO demonstration**, while the 49K checkpoint is considered a **geometry-focused result**.

---

## Key Research Finding

The main finding of the project is that a human-derived Behavior Cloning gait can provide a strong walking reference, while Residual PPO can learn useful terrain-aware corrections.

However, optimizing terrain-contact geometry alone can cause the RL policy to discover unnatural postures.

Future humanoid RL systems should therefore optimize both:

```text
Terrain adaptation
        +
Gait preservation
```

rather than considering penetration reduction alone.

---

## Repository Branches

The project is divided into dedicated Git branches because different stages use different experimental setups.

### `openhe-il-kinematic-bc`

Final OpenHE-based Kinematic Imitation Learning implementation.

Contains:

- OpenHE motion preprocessing
- 15-DOF Behavior Cloning
- final 12-DOF Behavior Cloning
- trained BC model
- evaluation scripts
- MuJoCo kinematic visualization

Branch:

```text
https://github.com/sajid-hasan01/unitree-g1-rl-il-walking/tree/openhe-il-kinematic-bc
```

---

### `openhe-rl-kinematic-ppo`

Final/current Kinematic Reinforcement Learning implementation.

Contains:

- uneven-terrain environments
- Residual PPO
- Reward V1 / V2 / V3 experiments
- balanced PPO training
- evaluation scripts
- diagnostic scripts
- trained PPO models
- 8K PPO models
- checkpointed 65K experiments
- 49K / 53K / 57K / 65K models
- experimental logs

Branch:

```text
https://github.com/sajid-hasan01/unitree-g1-rl-il-walking/tree/openhe-rl-kinematic-ppo
```

---

### `openhe-il-dynamic-physics`

Earlier OpenHE-based dynamic MuJoCo experiments involving:

- physics simulation
- contacts
- dynamic locomotion
- gait timing
- balance-related experiments
- foot-clearance experiments

Branch:

```text
https://github.com/sajid-hasan01/unitree-g1-rl-il-walking/tree/openhe-il-dynamic-physics
```

---

### `amass-il-baseline`

Earlier AMASS-based Imitation Learning experiments before the project transitioned to OpenHE.

```text
https://github.com/sajid-hasan01/unitree-g1-rl-il-walking/tree/amass-il-baseline
```

---

### `experiment/different-dataset`

Historical experiments using alternative motion datasets.

```text
https://github.com/sajid-hasan01/unitree-g1-rl-il-walking/tree/experiment/different-dataset
```

---

## Technologies

The project uses:

- Python
- MuJoCo
- Unitree G1
- PyTorch
- Gymnasium
- Stable-Baselines3
- Proximal Policy Optimization
- Behavior Cloning
- Residual Reinforcement Learning
- OpenHE G1-retargeted motion
- NumPy
- SciPy
- Matplotlib
- TensorBoard
- Git / GitHub

Main development environment:

```text
Python              3.11.9
MuJoCo              3.10.0
Gymnasium           1.3.0
Stable-Baselines3   2.9.0
PyTorch             2.13.0
NumPy               2.4.6
```

---

## Limitations

The current final Kinematic PPO work has several important limitations:

- Full dynamic balance is not demonstrated in the final kinematic environment.
- The global root trajectory is prescribed from the reference motion.
- PPO does not independently generate forward locomotion.
- Waist yaw, roll, and pitch remain fixed in the 12-DOF configuration.
- Current terrain experiments use a specific uneven-terrain arrangement.
- Longer PPO training can improve geometry while reducing gait naturalness.
- The current system is a simulation research prototype and is not presented as hardware-ready control for a physical Unitree G1.

---

## Future Work

Future work includes:

- extending Residual PPO to full MuJoCo dynamics,
- dynamic balance control,
- push-recovery training,
- random and unseen terrain generation,
- explicit gait-preservation rewards,
- full-body and waist adaptation,
- improved foot placement,
- stronger terrain generalization,
- and eventual simulation-to-real research.

---

## Final Conclusion

This project demonstrates an **OpenHE-based 12-DOF kinematic walking policy for Unitree G1 using Behavior Cloning**, followed by **Residual PPO for uneven-terrain adaptation**.

The main result shows that PPO can significantly reduce foot-terrain geometry errors while keeping Behavior Cloning as the underlying walking reference.

At the same time, longer training revealed that minimizing terrain penetration alone can produce less natural walking, highlighting the need for future reward functions that balance both **terrain adaptation and gait preservation**.

---

## Final Report

The complete CSE499 project report will be included in the `main` branch for detailed methodology, literature review, experiments, results, and analysis.

---

## Repository

```text
https://github.com/sajid-hasan01/unitree-g1-rl-il-walking
```

The `main` branch serves as the central project overview, while implementation code and trained models are maintained in the dedicated research branches.