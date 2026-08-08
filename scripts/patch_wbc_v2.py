from pathlib import Path

p = Path("envs/g1_wbc_taskspace_right_lift_env.py")
s = p.read_text(encoding="utf-8")

repls = {
    "support_lock_weight: float = 0.75,": "support_lock_weight: float = 0.70,",
    "support_xy_weight: float = 0.28,": "support_xy_weight: float = 0.22,",
    "support_z_weight: float = 1.35,": "support_z_weight: float = 1.20,",
    "support_ik_gain: float = 0.60,": "support_ik_gain: float = 0.50,",
    "support_ik_max_delta: float = 0.105,": "support_ik_max_delta: float = 0.090,",
    "torso_pitch_gain: float = 0.16,": "torso_pitch_gain: float = 0.20,",
    "torso_roll_gain: float = 0.10,": "torso_roll_gain: float = 0.08,",
    "angvel_pitch_gain: float = 0.075,": "angvel_pitch_gain: float = 0.115,",
    "angvel_roll_gain: float = 0.055,": "angvel_roll_gain: float = 0.045,",

    'kwargs.setdefault("swing_end", 0.60)': 'kwargs.setdefault("swing_end", 0.58)',
    'kwargs.setdefault("land_end", 0.78)': 'kwargs.setdefault("land_end", 0.68)',
    'kwargs.setdefault("target_clearance", 0.024)': 'kwargs.setdefault("target_clearance", 0.020)',
    'kwargs.setdefault("target_lateral_shift", 0.016)': 'kwargs.setdefault("target_lateral_shift", 0.012)',
    'kwargs.setdefault("ik_gain", 0.92)': 'kwargs.setdefault("ik_gain", 0.82)',
    'kwargs.setdefault("ik_damping", 0.055)': 'kwargs.setdefault("ik_damping", 0.060)',
    'kwargs.setdefault("ik_max_delta", 0.14)': 'kwargs.setdefault("ik_max_delta", 0.110)',
    'kwargs.setdefault("xy_hold_weight", 0.04)': 'kwargs.setdefault("xy_hold_weight", 0.140)',
    'kwargs.setdefault("z_lift_weight", 1.05)': 'kwargs.setdefault("z_lift_weight", 0.90)',

    'target[14] += -0.55 * pitch_task': 'target[14] += -0.68 * pitch_task',
    'target[0] += +0.45 * pitch_task': 'target[0] += +0.58 * pitch_task',
    'target[4] += -0.35 * pitch_task': 'target[4] += -0.46 * pitch_task',

    'rinfo["reward_version"] = "wbc_taskspace_right_lift_v1_torso_support"':
    'rinfo["reward_version"] = "wbc_taskspace_right_lift_v2_sagittal_capture_xy_lock"',
}

for old, new in repls.items():
    s = s.replace(old, new)

old_block = '''    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        # Parent creates the right-foot task-space lift and basic root feedback.
        target = super()._target_joint_position(action, info)

        sw = float(info["swing_env"])
        sh = float(info["shift_env"])
'''

new_block = '''    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        # V2: positive sagittal bias helped reduce angular velocity and backward velocity.
        action_for_parent = np.clip(np.asarray(action, dtype=np.float32).copy(), -1.0, 1.0)
        phase_for_bias = float(info["phase"])
        if phase_for_bias >= 0.38:
            action_for_parent[2] = np.clip(action_for_parent[2] + 0.85, -1.0, 1.0)

        # Parent creates the right-foot task-space lift and basic root feedback.
        target = super()._target_joint_position(action_for_parent, info)

        sw = float(info["swing_env"])
        sh = float(info["shift_env"])
'''

if old_block not in s:
    raise RuntimeError("Could not find _target_joint_position start block.")
s = s.replace(old_block, new_block)

old_block = '''        pitch_task = np.clip(
            -self.torso_pitch_gain * up_y - self.angvel_pitch_gain * av[1] - 0.06 * xv - 0.025 * x,
            -0.085,
            0.085,
        )
'''

new_block = '''        pitch_task = np.clip(
            -self.torso_pitch_gain * up_y - self.angvel_pitch_gain * av[1] - 0.10 * xv - 0.035 * x,
            -0.105,
            0.105,
        )
'''

if old_block not in s:
    raise RuntimeError("Could not find pitch_task block.")
s = s.replace(old_block, new_block)

old_block = '''        target[3] += +0.035 * support_env + 0.45 * height_task    # left knee
        target[0] += +0.18 * height_task                          # left hip pitch
        target[4] += -0.16 * height_task                          # left ankle pitch

        # WBC task 2: lock left support foot in task space.
'''

new_block = '''        target[3] += +0.025 * support_env + 0.38 * height_task    # left knee
        target[0] += +0.16 * height_task                          # left hip pitch
        target[4] += -0.14 * height_task                          # left ankle pitch

        # V2 sagittal capture: resist backward base velocity before x reaches hard limit.
        phase = float(info["phase"])
        if phase > 0.44:
            capture = float(np.clip(max(0.0, -xv - 0.18) * 0.75 + max(0.0, -x - 0.030) * 0.55, 0.0, 0.16))
            target[0] += +0.42 * capture      # left hip pitch
            target[4] += -0.34 * capture      # left ankle pitch
            target[14] += -0.24 * capture     # waist pitch
            target[9] += -0.08 * capture      # swing knee landing prep
            target[10] += -0.06 * capture     # swing ankle down

        # WBC task 2: lock left support foot in task space.
'''

if old_block not in s:
    raise RuntimeError("Could not find height/support block.")
s = s.replace(old_block, new_block)

old_block = '''        # WBC task 3: early landing help. When swing phase is decreasing or root
        # is tilting, reduce right knee lift and bring foot down sooner.
        if sw < 0.55 or up_z < 0.92 or xv < -0.45:
            landing_strength = float(np.clip((0.92 - up_z) * 3.0 + max(0.0, -xv - 0.45) * 1.2 + (0.55 - sw), 0.0, 1.0))
            target[9] += -0.18 * landing_strength      # right knee extends
            target[10] += -0.12 * landing_strength     # right ankle down
            target[6] += -0.035 * landing_strength     # right hip pitch back
'''

new_block = '''        # WBC task 3: proactive landing/capture.
        phase = float(info["phase"])
        if phase > 0.50 and (sw < 0.80 or up_z < 0.965 or xv < -0.22 or x < -0.05):
            landing_strength = float(np.clip(
                (0.965 - up_z) * 3.2
                + max(0.0, -xv - 0.22) * 1.7
                + max(0.0, -x - 0.05) * 1.2
                + max(0.0, 0.80 - sw) * 0.65,
                0.0,
                1.0,
            ))
            target[9] += -0.30 * landing_strength      # right knee extends
            target[10] += -0.20 * landing_strength     # right ankle down
            target[6] += -0.055 * landing_strength     # right hip pitch back
'''

if old_block not in s:
    raise RuntimeError("Could not find landing block.")
s = s.replace(old_block, new_block)

p.write_text(s, encoding="utf-8")
print("WBC v2 patch applied:", p)
