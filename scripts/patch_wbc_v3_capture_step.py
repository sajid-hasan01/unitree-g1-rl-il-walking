from pathlib import Path

p = Path("envs/g1_wbc_taskspace_right_lift_env.py")
s = p.read_text(encoding="utf-8")

backup = Path("envs/g1_wbc_taskspace_right_lift_env_v2_rejected_backup.py")
backup.write_text(s, encoding="utf-8")

# Safer v3 defaults: lower lift, shorter swing, earlier landing.
repls = {
    'kwargs.setdefault("swing_end", 0.58)': 'kwargs.setdefault("swing_end", 0.52)',
    'kwargs.setdefault("land_end", 0.68)': 'kwargs.setdefault("land_end", 0.60)',
    'kwargs.setdefault("target_clearance", 0.020)': 'kwargs.setdefault("target_clearance", 0.014)',
    'kwargs.setdefault("ik_gain", 0.82)': 'kwargs.setdefault("ik_gain", 0.70)',
    'kwargs.setdefault("ik_max_delta", 0.110)': 'kwargs.setdefault("ik_max_delta", 0.085)',
    'kwargs.setdefault("xy_hold_weight", 0.140)': 'kwargs.setdefault("xy_hold_weight", 0.180)',
    'kwargs.setdefault("z_lift_weight", 0.90)': 'kwargs.setdefault("z_lift_weight", 0.72)',
    'rinfo["reward_version"] = "wbc_taskspace_right_lift_v2_sagittal_capture_xy_lock"':
    'rinfo["reward_version"] = "wbc_taskspace_right_lift_v3_capture_step"'
}

for old, new in repls.items():
    s = s.replace(old, new)

# Insert capture-step right-foot target override.
marker = "    def _target_joint_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:\n"

capture_method = '''    def _target_foot_position(self, action: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        """
        WBC v3 capture-step foot placement.

        Negative x is the observed backward-fall direction in our logs.
        When root x/xv becomes unsafe during swing, the right foot is placed
        slightly backward/under the falling body and forced down for touchdown.
        """
        target = super()._target_foot_position(action, info)

        phase = float(info["phase"])
        sw = float(info["swing_env"])
        x = float(info["x_position"])
        xv = float(info["x_velocity"])
        up_z = float(info["up_z"])

        if phase > 0.46:
            capture = float(np.clip(
                max(0.0, -xv - 0.22) * 1.25
                + max(0.0, -x - 0.055) * 2.20
                + max(0.0, 0.94 - up_z) * 2.50,
                0.0,
                1.0,
            ))

            # Capture placement: move right-foot target in the falling direction.
            # Observed fall direction is negative x.
            target[0] += -0.070 * capture

            # Force touchdown by reducing vertical target once capture activates.
            target[2] -= 0.025 * capture
            target[2] = max(float(self.right_foot_p0[2]), float(target[2]))

        return target

'''

if "WBC v3 capture-step foot placement" not in s:
    if marker not in s:
        raise RuntimeError("Could not find _target_joint_position marker.")
    s = s.replace(marker, capture_method + marker, 1)

# Replace the v2 baked sagittal bias with the best tested residual direction:
# reduce lift, cancel sagittal bias, add lateral correction.
old_block = '''        # V2: Test 2 showed positive sagittal residual reduced angular velocity
        # and backward velocity compared with zero/negative residual. Bake a
        # bounded positive sagittal bias into the parent controller during
        # swing/landing, while keeping external residual action available.
        action_for_parent = np.clip(np.asarray(action, dtype=np.float32).copy(), -1.0, 1.0)
        phase_for_bias = float(info["phase"])
        if phase_for_bias >= 0.38:
            action_for_parent[2] = np.clip(action_for_parent[2] + 0.85, -1.0, 1.0)

        # Parent creates the right-foot task-space lift and basic root feedback.
        target = super()._target_joint_position(action_for_parent, info)
'''

new_block = '''        # WBC v3: bake in the best tested residual direction:
        # fixed_action=-1,0,-1,1 gave the best upright/lateral result.
        action_for_parent = np.clip(np.asarray(action, dtype=np.float32).copy(), -1.0, 1.0)
        phase_for_bias = float(info["phase"])
        if phase_for_bias >= 0.38:
            action_for_parent[0] = np.clip(action_for_parent[0] - 0.85, -1.0, 1.0)  # reduce lift
            action_for_parent[2] = np.clip(action_for_parent[2] - 0.85, -1.0, 1.0)  # cancel bad sagittal bias
            action_for_parent[3] = np.clip(action_for_parent[3] + 0.75, -1.0, 1.0)  # lateral correction

        # Parent creates the right-foot task-space lift and basic root feedback.
        target = super()._target_joint_position(action_for_parent, info)
'''

if old_block not in s:
    raise RuntimeError("Could not find v2 action bias block.")
s = s.replace(old_block, new_block)

# Replace proactive landing block with stronger capture touchdown.
old_block = '''        # WBC task 3: proactive landing/capture.
        # The previous WBC allowed the robot to keep the right foot high while
        # x velocity grew strongly negative. V2 starts landing earlier when
        # xv becomes negative or up_z begins dropping.
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

new_block = '''        # WBC task 3: capture touchdown.
        # Once backward root momentum starts, do not keep holding the foot up.
        # Extend right knee/ankle and place the foot down quickly.
        phase = float(info["phase"])
        if phase > 0.48:
            capture = float(np.clip(
                max(0.0, -xv - 0.22) * 1.35
                + max(0.0, -x - 0.055) * 2.40
                + max(0.0, 0.94 - up_z) * 2.60,
                0.0,
                1.0,
            ))

            if capture > 0.02:
                target[9] += -0.42 * capture       # right knee extends for touchdown
                target[10] += -0.30 * capture      # right ankle down
                target[6] += -0.075 * capture      # right hip moves foot toward capture placement

                # Support-leg push while the capture step comes down.
                target[0] += +0.26 * capture       # left hip pitch
                target[4] += -0.22 * capture       # left ankle pitch
                target[14] += -0.16 * capture      # waist pitch
'''

if old_block not in s:
    raise RuntimeError("Could not find v2 landing block.")
s = s.replace(old_block, new_block)

p.write_text(s, encoding="utf-8")
print("WBC v3 capture-step patch applied.")
print("Backup saved:", backup)
